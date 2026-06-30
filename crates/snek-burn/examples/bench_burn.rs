//! Inference throughput of the burn/CUDA AZNet on the live net shape
//! (14x11x11, trunk 96x8), to compare against the ONNX/ort ceiling (~57k rows/s
//! at batch 512 on the RTX 5080). Mirrors `snek-infer/examples/bench_batches`:
//! per call we copy a host buffer up (`from_data` = H2D), forward, and read the
//! outputs back (`into_data` = D2H + sync), so the GPU work is actually awaited.

use burn::backend::Cuda;
use burn::prelude::*;
use burn::tensor::TensorData;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use snek_burn::AZNet;
use std::time::Instant;

type B = Cuda;

fn main() {
    let device = Default::default();
    let (c, h, w) = (14usize, 11usize, 11usize);
    let net = AZNet::<B>::new(c, 96, 8, 3, &device);

    let batches: Vec<usize> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![128, 256, 512, 1024, 2048]);
    let seconds: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let warmup = 16usize;

    let mut rng = StdRng::seed_from_u64(0x5EED);
    println!("batch,rows_per_sec,calls_per_sec,mean_ms,calls");
    for &batch in &batches {
        let host: Vec<f32> = (0..batch * c * h * w).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let data = TensorData::new(host, [batch, c, h, w]);

        let run_once = || {
            let input = Tensor::<B, 4>::from_data(data.clone(), &device); // H2D
            let (p, v) = net.forward(input);
            let _ = p.into_data(); // D2H + sync (await GPU)
            let _ = v.into_data();
        };

        for _ in 0..warmup {
            run_once();
        }
        let start = Instant::now();
        let mut calls = 0usize;
        while start.elapsed().as_secs_f64() < seconds {
            run_once();
            calls += 1;
        }
        let el = start.elapsed().as_secs_f64().max(1e-9);
        let rps = (calls * batch) as f64 / el;
        let cps = calls as f64 / el;
        println!("{batch},{rps:.0},{cps:.2},{:.3},{calls}", 1000.0 / cps.max(1e-9));
    }
}
