//! "Dumb pipeline" that mirrors the self-play GPU/CPU handoff in
//! `snek-py::generate_selfplay` WITHOUT any MCTS, so we can isolate where the
//! live run loses throughput vs the pure `bench_batches` ceiling.
//!
//! Structure copied from the real loop:
//!   - one GPU worker thread owns the `Net` and forwards batches off an mpsc
//!     channel (recv obs Vec -> forward -> send back results);
//!   - the main thread keeps `shards` requests in flight (double-buffer), and on
//!     each result allocates a *fresh* obs Vec, optionally fills it, optionally
//!     burns CPU, then re-sends — exactly like `select_write` does.
//!
//! Knobs (env):
//!   PIPE_CPU_US      serial busy-spin microseconds per received result
//!                    (stand-in for expand_backup + select on the main thread)
//!   PIPE_LOAD_THREADS background busy-spin threads pinned hot for the whole run
//!                    (stand-in for the 16 rayon MCTS threads saturating cores)
//!   PIPE_FILL        1 = write the fresh obs buffer each call (default 1)
//!   SNEK_BENCH_C/H/W net input dims (default 14/11/11)
//!
//! Args: model [batch] [shards] [sims] [moves]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use snek_infer::Net;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn spin_us(us: u64) {
    if us == 0 {
        return;
    }
    let start = Instant::now();
    let dur = Duration::from_micros(us);
    // Volatile-ish busy work so the optimizer can't elide it.
    let mut x = 0u64;
    while start.elapsed() < dur {
        for _ in 0..256 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        std::hint::black_box(x);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model = args.get(1).map(String::as_str).unwrap_or("model.onnx");
    let batch = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512usize);
    let shards = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4usize);
    let sims = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1000usize);
    let moves = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20usize);

    let c = env_usize("SNEK_BENCH_C", 14);
    let h = env_usize("SNEK_BENCH_H", 11);
    let w = env_usize("SNEK_BENCH_W", 11);
    let obs = c * h * w;
    let cpu_us = env_usize("PIPE_CPU_US", 0) as u64;
    let load_threads = env_usize("PIPE_LOAD_THREADS", 0);
    let fill = env_usize("PIPE_FILL", 1) != 0;

    eprintln!(
        "model={model} c={c} h={h} w={w} batch={batch} shards={shards} sims={sims} moves={moves} \
         cpu_us={cpu_us} load_threads={load_threads} fill={fill}"
    );

    // Background load: hot busy-spin threads simulating MCTS saturating the box.
    let stop = Arc::new(AtomicBool::new(false));
    let mut loaders = Vec::new();
    for _ in 0..load_threads {
        let stop = Arc::clone(&stop);
        loaders.push(std::thread::spawn(move || {
            let mut x = 0u64;
            while !stop.load(Ordering::Relaxed) {
                for _ in 0..4096 {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                }
                std::hint::black_box(x);
            }
        }));
    }

    type Req = Option<(usize, Vec<f32>, usize)>;
    type Res = (usize, Vec<f32>, Vec<f32>);
    let (tx_req, rx_req) = std::sync::mpsc::channel::<Req>();
    let (tx_res, rx_res) = std::sync::mpsc::channel::<Res>();
    let onnx = model.to_string();
    let gpu = std::thread::spawn(move || -> Result<(f64, f64), String> {
        let mut net = Net::load(&onnx).map_err(|e| e.to_string())?;
        let (mut t_fwd, mut t_idle) = (Duration::ZERO, Duration::ZERO);
        loop {
            let wt = Instant::now();
            let msg = rx_req.recv();
            t_idle += wt.elapsed();
            let (id, obs_buf, m) = match msg {
                Ok(Some(x)) => x,
                _ => break,
            };
            let f = Instant::now();
            let (pol, val) = net.forward(&obs_buf, m, c, h, w).map_err(|e| e.to_string())?;
            t_fwd += f.elapsed();
            if tx_res.send((id, pol, val)).is_err() {
                break;
            }
        }
        Ok((t_fwd.as_secs_f64(), t_idle.as_secs_f64()))
    });

    // Template buffer to copy from when filling (mimics encode writing real data).
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut template = vec![0.0f32; batch * obs];
    for x in &mut template {
        *x = rng.gen_range(-1.0..1.0);
    }
    let make_obs = |m: usize| -> Vec<f32> {
        let mut buf = vec![0.0f32; m * obs];
        if fill {
            buf.copy_from_slice(&template[..m * obs]);
        }
        buf
    };

    // Warm up so we don't measure cudnn algo selection.
    for sid in 0..shards {
        tx_req.send(Some((sid, make_obs(batch), batch)))?;
    }
    let mut warm = shards;
    while warm > 0 {
        let _ = rx_res.recv()?;
        warm -= 1;
    }

    let start = Instant::now();
    let mut total_rows = 0usize;
    for _mv in 0..moves {
        for sid in 0..shards {
            tx_req.send(Some((sid, make_obs(batch), batch)))?;
        }
        let mut active = shards;
        let mut done = vec![0usize; shards];
        while active > 0 {
            let (sid, _pol, _val) = rx_res.recv()?;
            spin_us(cpu_us); // serial "MCTS" on the main thread
            done[sid] += 1;
            if done[sid] < sims {
                tx_req.send(Some((sid, make_obs(batch), batch)))?;
            } else {
                active -= 1;
            }
            total_rows += batch;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    let _ = tx_req.send(None);
    drop(tx_req);
    let (t_fwd, t_idle) = match gpu.join() {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(e.into()),
        Err(_) => return Err("gpu thread panicked".into()),
    };
    stop.store(true, Ordering::Relaxed);
    for l in loaders {
        let _ = l.join();
    }

    let rows_per_sec = total_rows as f64 / elapsed.max(1e-9);
    let gpu_busy = 100.0 * t_fwd / (t_fwd + t_idle).max(1e-9);
    println!(
        "rows_per_sec={rows_per_sec:.0} gpu_busy={gpu_busy:.1}% \
         gpu_fwd_s={t_fwd:.2} gpu_idle_s={t_idle:.2} elapsed_s={elapsed:.2} rows={total_rows}"
    );
    Ok(())
}
