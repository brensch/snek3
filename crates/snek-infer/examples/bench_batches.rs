use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use snek_infer::Net;
use std::env;
use std::time::{Duration, Instant};

fn parse_batches(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|x| x.trim().parse::<usize>().ok())
        .filter(|&x| x > 0)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("runs/20260628-143633/model.onnx");
    let batches = args
        .get(2)
        .map(|s| parse_batches(s))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            vec![
                64, 96, 128, 192, 256, 384, 448, 480, 504, 512, 640, 768, 896, 1024, 1280, 1536,
                2048,
            ]
        });
    let seconds: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let warmup: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);

    let c = env::var("SNEK_BENCH_C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18usize);
    let h = env::var("SNEK_BENCH_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11usize);
    let w = env::var("SNEK_BENCH_W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11usize);
    let obs = c * h * w;
    let max_batch = *batches.iter().max().unwrap();

    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut input = vec![0.0f32; max_batch * obs];
    for x in &mut input {
        *x = rng.gen_range(-1.0..1.0);
    }

    eprintln!("model={model} c={c} h={h} w={w} seconds_per_batch={seconds} warmup={warmup}");
    println!("batch,rows_per_sec,calls_per_sec,mean_ms,calls,rows");

    let mut net = Net::load(model)?;
    for &batch in &batches {
        let slice = &input[..batch * obs];
        for _ in 0..warmup {
            let _ = net.forward(slice, batch, c, h, w)?;
        }

        let target = Duration::from_secs_f64(seconds);
        let start = Instant::now();
        let mut calls = 0usize;
        let mut rows = 0usize;
        while start.elapsed() < target {
            let _ = net.forward(slice, batch, c, h, w)?;
            calls += 1;
            rows += batch;
        }
        let elapsed = start.elapsed().as_secs_f64().max(1e-9);
        let rows_per_sec = rows as f64 / elapsed;
        let calls_per_sec = calls as f64 / elapsed;
        let mean_ms = 1000.0 / calls_per_sec.max(1e-9);
        println!("{batch},{rows_per_sec:.0},{calls_per_sec:.2},{mean_ms:.3},{calls},{rows}");
    }
    Ok(())
}
