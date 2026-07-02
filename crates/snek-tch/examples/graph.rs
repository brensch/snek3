//! Benchmark: CUDA-graph replay vs eager for the live net at a fixed batch.
//! Same math, same H2D/D2H work — the only difference is per-launch overhead.
//!
//! Args: `graph [batch] [seconds]`  (defaults 512 20)

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use snek_tch::cudagraph::GraphedForward;
use snek_tch::AZNet;
use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};

fn main() {
    let batch: i64 = arg(1).unwrap_or(512);
    let seconds: f64 = arg(2).unwrap_or(20.0);
    let (c, h, w) = (14i64, 11i64, 11i64);

    let dev = Device::Cuda(0);
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    tch::Cuda::cudnn_set_benchmark(true);
    let vs = nn::VarStore::new(dev);
    let net = AZNet::new(&vs.root(), c, 96, 8);
    snek_tch::init_orthogonal(&vs, 2f64.sqrt());

    let n = (batch * c * h * w) as usize;
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let host: Vec<f32> = (0..n)
        .map(|_| if rng.gen::<f32>() < 0.15 { 1.0 } else { 0.0 })
        .collect();
    let mut pol = vec![0.0f32; (batch * 4) as usize];
    let mut val = vec![0.0f32; batch as usize];

    // ---- EAGER (identical to the engine's Gpu::forward path) ----
    let eager = |pol: &mut [f32], val: &mut [f32]| {
        tch::no_grad(|| {
            let x = Tensor::from_slice(&host)
                .reshape([batch, c, h, w])
                .to_device(dev);
            let (logits, value) = net.forward(&x);
            let p = logits.softmax(-1, Kind::Float).to_device(Device::Cpu);
            let v = value.to_device(Device::Cpu);
            p.copy_data(pol, (batch * 4) as usize);
            v.copy_data(val, batch as usize);
        });
    };
    for _ in 0..32 {
        eager(&mut pol, &mut val);
    }
    let (er, ec) = measure(seconds, || eager(&mut pol, &mut val));
    let eager_rps = ec as f64 * batch as f64 / er;
    println!(
        "eager: {eager_rps:9.0} rows/s   ({:.3} ms/call, {ec} calls)",
        1000.0 * er / ec as f64
    );

    // ---- CUDA GRAPH ----
    let mut g = GraphedForward::capture(&net, dev, batch, c, h, w);
    for _ in 0..32 {
        g.run(&host, &mut pol, &mut val);
    }
    let (gr, gc) = measure(seconds, || g.run(&host, &mut pol, &mut val));
    let graph_rps = gc as f64 * batch as f64 / gr;
    println!(
        "graph: {graph_rps:9.0} rows/s   ({:.3} ms/call, {gc} calls)",
        1000.0 * gr / gc as f64
    );

    println!(
        "\nspeedup: {:.2}x   ({:+.0} rows/s)",
        graph_rps / eager_rps.max(1.0),
        graph_rps - eager_rps
    );

    // Correctness sanity: graph outputs are a valid softmax + finite values.
    let psum: f32 = pol[..4].iter().sum();
    println!(
        "sanity: row0 policy sums to {psum:.4}, value[0]={:.4} (finite: {})",
        val[0],
        val.iter().all(|v| v.is_finite())
    );
}

fn measure(seconds: f64, mut f: impl FnMut()) -> (f64, u64) {
    let start = Instant::now();
    let mut calls = 0u64;
    while start.elapsed().as_secs_f64() < seconds {
        f();
        calls += 1;
    }
    (start.elapsed().as_secs_f64().max(1e-9), calls)
}

fn arg<T: std::str::FromStr>(i: usize) -> Option<T> {
    std::env::args().nth(i).and_then(|s| s.parse().ok())
}
