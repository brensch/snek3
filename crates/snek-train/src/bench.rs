//! GPU batch-size throughput sweep, surfaced through the dashboard. Mirrors the
//! `bench_tch` example: for each batch size it times H2D input + forward + D2H
//! outputs on the live net shape, reporting rows/sec (i.e. inferences/sec) so we
//! can pick the batch size that saturates the current GPU best. Runs only when
//! no training run is active — the GPU would otherwise be contended and the
//! numbers meaningless.

use crate::config::RunConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::time::Instant;
use tch::{nn, Device, Tensor};
use tokio::sync::mpsc::UnboundedSender;

/// Batch sizes (forward-tensor rows) to sweep.
const BATCHES: &[i64] = &[64, 128, 256, 512, 1024, 2048, 4096];
/// Seconds of timed forwards per batch size.
const SECONDS: f64 = 1.5;
/// Untimed warmup forwards before each measurement, so cuDNN has autotuned the
/// kernels for that shape before the clock starts.
const WARMUP: usize = 8;

/// A single benchmark progress event, streamed to the dashboard as JSON over SSE
/// (same contract as the trainer log stream). `kind` is the discriminant.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BenchEvent {
    /// Sweep is starting; carries the plan so the UI can lay out the table.
    Start {
        batches: Vec<i64>,
        seconds: f64,
        device: String,
        trunk_channels: i64,
        trunk_blocks: i64,
    },
    /// Warmup + measurement of `batch` (the `index`-th of `total`) has begun.
    Measuring {
        batch: i64,
        index: usize,
        total: usize,
    },
    /// A batch size finished measuring. `rows_per_sec` is the headline inf/s.
    Result {
        batch: i64,
        rows_per_sec: f64,
        calls_per_sec: f64,
        mean_ms: f64,
    },
    /// Sweep complete; `best_batch` had the highest `rows_per_sec`.
    Done { best_batch: i64 },
    /// The sweep could not run (e.g. no CUDA) or aborted.
    Error { detail: String },
}

/// Run the sweep synchronously — call from a dedicated thread. Emits events as it
/// goes; the sender being dropped (this returning) is the stream's end signal.
pub fn run_sweep(cfg: &RunConfig, tx: &UnboundedSender<BenchEvent>) {
    if !tch::Cuda::is_available() {
        let _ = tx.send(BenchEvent::Error {
            detail: "CUDA is not available to libtorch on this host".into(),
        });
        return;
    }
    // cuDNN autotune like the training stack, so measured throughput matches the
    // kernels self-play actually runs.
    tch::Cuda::cudnn_set_benchmark(true);
    let dev = Device::Cuda(0);
    let vs = nn::VarStore::new(dev);
    let net = snek_tch::AZNet::new(
        &vs.root(),
        snek_core::NUM_CHANNELS as i64,
        cfg.trunk_channels,
        cfg.trunk_blocks,
    );

    let batches: Vec<i64> = BATCHES.to_vec();
    let _ = tx.send(BenchEvent::Start {
        batches: batches.clone(),
        seconds: SECONDS,
        device: format!("cuda:0 ({})", tch::Cuda::device_count()),
        trunk_channels: cfg.trunk_channels,
        trunk_blocks: cfg.trunk_blocks,
    });

    let (c, h, w) = (
        snek_core::NUM_CHANNELS as i64,
        cfg.board as i64,
        cfg.board as i64,
    );
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let total = batches.len();
    let mut best: Option<(i64, f64)> = None;

    for (index, &batch) in batches.iter().enumerate() {
        let _ = tx.send(BenchEvent::Measuring {
            batch,
            index,
            total,
        });

        let n = (batch * c * h * w) as usize;
        let host: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        // CPU tensor reused across calls; each call copies it H2D like real serving.
        let cpu = Tensor::from_slice(&host).reshape([batch, c, h, w]);

        let run_once = || {
            tch::no_grad(|| {
                let x = cpu.to_device(dev); // H2D
                let (p, v) = net.forward(&x);
                let _ = p.to_device(Device::Cpu); // D2H + implicit sync
                let _ = v.to_device(Device::Cpu);
            })
        };

        for _ in 0..WARMUP {
            run_once();
        }
        let start = Instant::now();
        let mut calls = 0usize;
        while start.elapsed().as_secs_f64() < SECONDS {
            run_once();
            calls += 1;
        }
        let el = start.elapsed().as_secs_f64().max(1e-9);
        let rows_per_sec = (calls as i64 * batch) as f64 / el;
        let calls_per_sec = calls as f64 / el;
        let mean_ms = 1000.0 / calls_per_sec.max(1e-9);

        if best.map(|(_, r)| rows_per_sec > r).unwrap_or(true) {
            best = Some((batch, rows_per_sec));
        }
        let _ = tx.send(BenchEvent::Result {
            batch,
            rows_per_sec,
            calls_per_sec,
            mean_ms,
        });
    }

    let _ = tx.send(BenchEvent::Done {
        best_batch: best.map(|(b, _)| b).unwrap_or(0),
    });
}
