//! tch-rs/libtorch inference throughput on the live net shape (14x11x11, trunk
//! 96x8). Set mode to:
//! - `copy` (default): H2D input + forward + D2H outputs.
//! - `resident`: input stays on GPU; synchronize after forward.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use snek_tch::AZNet;
use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};

fn main() {
    let dev = Device::Cuda(0);
    // cuDNN benchmark/TF32 like the training stack (allow_tf32 is on there).
    tch::Cuda::cudnn_set_benchmark(true);
    let vs = nn::VarStore::new(dev);
    let net = AZNet::new(&vs.root(), 14, 96, 8, 3);

    let (c, h, w) = (14i64, 11i64, 11i64);
    let batches: Vec<i64> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<i64>| !v.is_empty())
        .unwrap_or_else(|| vec![128, 256, 512, 1024, 2048]);
    let seconds: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let mode = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "copy".to_string());
    let warmup = 16usize;

    let mut rng = StdRng::seed_from_u64(0x5EED);
    println!("mode,batch,rows_per_sec,calls_per_sec,mean_ms,calls");
    for &batch in &batches {
        let n = (batch * c * h * w) as usize;
        let host: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let cpu = Tensor::from_slice(&host).reshape([batch, c, h, w]); // CPU tensor, reused
        let gpu = Tensor::randn([batch, c, h, w], (Kind::Float, dev));

        let run_once = || {
            tch::no_grad(|| {
                if mode == "resident" {
                    let (_p, _v) = net.forward(&gpu);
                    tch::Cuda::synchronize(0);
                } else {
                    let x = cpu.to_device(dev); // H2D
                    let (p, v) = net.forward(&x);
                    let _ = p.to_device(Device::Cpu); // D2H + sync
                    let _ = v.to_device(Device::Cpu);
                }
            })
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
        let rps = (calls as i64 * batch) as f64 / el;
        let cps = calls as f64 / el;
        println!(
            "{mode},{batch},{rps:.0},{cps:.2},{:.3},{calls}",
            1000.0 / cps.max(1e-9)
        );
    }
}
