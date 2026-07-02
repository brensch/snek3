//! Sustained-throughput probe for the live net (14x11x11, trunk 96x8, gpool 3).
//!
//! Holds ONE fixed batch of games in host memory and feeds it to the GPU over
//! and over, reporting throughput every second so we can see whether batch-512
//! throughput holds long-term (no thermal / allocator / fragmentation drift).
//!
//! Deliberately uses none of the self-play pipeline: no MCTS, no channels, no
//! game stepping. Just: fixed data in memory -> net -> repeat.
//!
//! Args: `sustain [batch] [seconds] [mode]`
//!   mode = resident (default) : upload the batch to the GPU once, then loop
//!                               forward+sync. The theoretical sustained ceiling.
//!          feed               : re-upload the same host buffer every iteration
//!                               (H2D) then forward+sync. "Feed from Rust" cost.
//!          full               : feed + forward + copy policy/value back to host
//!                               (D2H). The full real per-call inference path.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use snek_tch::AZNet;
use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};

fn main() {
    let dev = Device::Cuda(0);
    // Match the training stack exactly.
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    tch::Cuda::cudnn_set_benchmark(true);

    let batch: i64 = arg(1).unwrap_or(512);
    let seconds: f64 = arg(2).unwrap_or(60.0);
    let mode = std::env::args().nth(3).unwrap_or_else(|| "resident".into());
    let (c, h, w) = (14i64, 11i64, 11i64);

    let vs = nn::VarStore::new(dev);
    let net = AZNet::new(&vs.root(), c, 96, 8);

    // The "games": one fixed batch, generated once, then kept in host memory and
    // reused forever. Values are irrelevant to timing (conv cost is shape-bound),
    // but we make them plausible obs (mostly 0/1 planes) rather than pure noise.
    let n = (batch * c * h * w) as usize;
    let mut rng = StdRng::seed_from_u64(0x5EED_C0DE);
    let host: Vec<f32> = (0..n)
        .map(|_| if rng.gen::<f32>() < 0.15 { 1.0 } else { 0.0 })
        .collect();

    // Resident copy uploaded exactly once.
    let resident = Tensor::from_slice(&host).reshape([batch, c, h, w]).to_device(dev);

    // Reusable host output buffers for `full` mode (allocated once).
    let mut pol_host = vec![0.0f32; (batch * 4) as usize];
    let mut val_host = vec![0.0f32; batch as usize];

    let run_once = |pol_host: &mut Vec<f32>, val_host: &mut Vec<f32>| {
        tch::no_grad(|| match mode.as_str() {
            "resident" => {
                let (_p, _v) = net.forward(&resident);
                tch::Cuda::synchronize(0);
            }
            "feed" => {
                let x = Tensor::from_slice(&host).reshape([batch, c, h, w]).to_device(dev);
                let (_p, _v) = net.forward(&x);
                tch::Cuda::synchronize(0);
            }
            "full" => {
                let x = Tensor::from_slice(&host).reshape([batch, c, h, w]).to_device(dev);
                let (p, v) = net.forward(&x);
                let p = p.softmax(-1, Kind::Float).to_device(Device::Cpu);
                let v = v.to_device(Device::Cpu);
                let pl = pol_host.len();
                let vl = val_host.len();
                p.copy_data(pol_host, pl);
                v.copy_data(val_host, vl);
            }
            other => panic!("unknown mode {other}"),
        })
    };

    // Warmup: lets cuDNN autotune this fixed shape once and the caching allocator
    // settle, so the measured window is pure steady state.
    for _ in 0..64 {
        run_once(&mut pol_host, &mut val_host);
    }

    println!(
        "RTX / net 14x11x11 trunk96x8 | mode={mode} batch={batch} | per-second sustained throughput:"
    );
    println!("  t(s)   calls   rows/s   mean_ms");
    let start = Instant::now();
    let mut calls = 0u64;
    let mut last_mark = 0.0f64;
    let mut last_calls = 0u64;
    let mut min_rps = f64::INFINITY;
    let mut max_rps = 0.0f64;
    let mut sum_rps = 0.0f64;
    let mut marks = 0u64;
    loop {
        run_once(&mut pol_host, &mut val_host);
        calls += 1;
        let el = start.elapsed().as_secs_f64();
        if el - last_mark >= 1.0 {
            let dt = el - last_mark;
            let dcalls = calls - last_calls;
            let rps = dcalls as f64 * batch as f64 / dt;
            let mean_ms = 1000.0 * dt / dcalls.max(1) as f64;
            println!("  {el:5.1} {dcalls:7} {rps:9.0} {mean_ms:7.3}");
            min_rps = min_rps.min(rps);
            max_rps = max_rps.max(rps);
            sum_rps += rps;
            marks += 1;
            last_mark = el;
            last_calls = calls;
        }
        if el >= seconds {
            break;
        }
    }
    let el = start.elapsed().as_secs_f64();
    let overall = calls as f64 * batch as f64 / el;
    println!(
        "\nsummary mode={mode} batch={batch}: overall {overall:.0} rows/s over {el:.1}s ({calls} calls)"
    );
    if marks > 0 {
        println!(
            "  per-second rows/s: min {min_rps:.0}  mean {:.0}  max {max_rps:.0}  (spread {:.1}%)",
            sum_rps / marks as f64,
            100.0 * (max_rps - min_rps) / max_rps.max(1.0)
        );
    }
}

fn arg<T: std::str::FromStr>(i: usize) -> Option<T> {
    std::env::args().nth(i).and_then(|s| s.parse().ok())
}
