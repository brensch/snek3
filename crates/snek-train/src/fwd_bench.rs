//! Chunk-size throughput sweep for the staged LE leaf forward (`--fwd-bench`).
//!
//! The self-play turn is one small root forward plus one large deepen forward,
//! both through [`forward_values_staged`]: encode into a pinned stage, one H2D
//! of the padded batch, fixed-shape chunk forwards enqueued with no host sync,
//! one D2H. `le_fwd_chunk` picks the chunk shape — this sweep measures real
//! rows/sec per candidate chunk on the run's actual net shape so the knob is
//! chosen from data, not folklore. Run it with training STOPPED: a contended
//! GPU makes the numbers meaningless.

use crate::config::RunConfig;
use crate::selfplay::le_selfplay::{forward_values_staged, PinnedStage};
use crate::state::RunPaths;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use snek_core::{obs_side, NUM_CHANNELS_TEMP};
use std::time::Instant;
use tch::{nn, Device};

/// Candidate `le_fwd_chunk` values (rows; all multiples of 4 seats).
const CHUNKS: &[usize] = &[128, 192, 256, 320, 384, 448, 512];
/// Untimed calls per (chunk, rows) shape so cuDNN autotune settles first.
const WARMUP: usize = 3;
/// Timed seconds per (chunk, rows) point.
const SECONDS: f64 = 4.0;

/// Time `forward_values_staged` at `rows` useful rows for one chunk size.
/// Returns (useful rows/sec, mean call ms). Padding waste is charged
/// naturally: padded rows cost GPU time but don't count as useful.
fn measure(
    net: &snek_tch::AZNet,
    device: Device,
    stage: &mut PinnedStage,
    rows: usize,
    side: i64,
    chunk: usize,
) -> (f64, f64) {
    let l15 = NUM_CHANNELS_TEMP * (side * side) as usize;
    let padded = rows.div_ceil(chunk) * chunk;
    // Fill the stage once — throughput doesn't depend on the values.
    let mut rng = StdRng::seed_from_u64(0x5EED);
    for v in stage.slice(padded * l15).iter_mut() {
        *v = rng.gen_range(-1.0f32..1.0);
    }
    for _ in 0..WARMUP {
        let _ = forward_values_staged(net, device, stage.tensor(), rows, padded, side, chunk);
    }
    let start = Instant::now();
    let mut calls = 0usize;
    while start.elapsed().as_secs_f64() < SECONDS {
        let _ = forward_values_staged(net, device, stage.tensor(), rows, padded, side, chunk);
        calls += 1;
    }
    let el = start.elapsed().as_secs_f64();
    ((calls * rows) as f64 / el, el * 1000.0 / calls as f64)
}

/// Load a checkpoint and print the value head's output on a spread of real
/// boards (all seats, serve-τ). Diagnostic for a dead/saturated value head:
/// healthy nets spread values with the position; a tanh-saturated head pins
/// |v| ≈ 1 everywhere and its gradient is ~0, so it never trains.
pub fn value_probe(paths: &RunPaths, ckpt: &std::path::Path) -> anyhow::Result<()> {
    let cfg = RunConfig::load(&paths.config)?;
    let device = Device::Cuda(0);
    let side = obs_side(cfg.board as usize) as i64;
    let n = cfg.num_snakes;
    let mut vs = nn::VarStore::new(device);
    let net = snek_tch::AZNet::new(
        &vs.root(),
        NUM_CHANNELS_TEMP as i64,
        cfg.trunk_channels,
        cfg.trunk_blocks,
    );
    vs.load(ckpt)?;
    let l15 = NUM_CHANNELS_TEMP * (side * side) as usize;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xB0A2D);
    let mut boards = vec![snek_core::standard_start(cfg.board, cfg.board, n, &mut rng)];
    for sc in snek_core::scenario::SCENARIOS.iter().take(3) {
        boards.push((sc.generate)(cfg.board, cfg.board, n, &mut rng));
    }
    let rows = boards.len() * n;
    let mut stage = PinnedStage::new(device);
    let buf = stage.slice(rows * l15);
    for (bi, b) in boards.iter().enumerate() {
        for s in 0..n {
            snek_core::encode_into_temp(
                b,
                s,
                cfg.response_tau,
                &mut buf[(bi * n + s) * l15..(bi * n + s + 1) * l15],
            );
        }
    }
    let vals = forward_values_staged(&net, device, stage.tensor(), rows, rows, side, rows);
    println!("value probe: {} (τ={})", ckpt.display(), cfg.response_tau);
    for (bi, chunk) in vals.chunks(n).enumerate() {
        let line: Vec<String> = chunk.iter().map(|v| format!("{v:+.3}")).collect();
        println!("  board {bi}: [{}]", line.join(", "));
    }
    let sat = vals.iter().filter(|v| v.abs() > 0.99).count();
    println!("  |v|>0.99: {}/{} — saturated head if ~all", sat, vals.len());
    Ok(())
}

/// Sweep chunk sizes on the run's net shape and print a decision table.
/// `big_rows`/`small_rows` approximate the two real per-turn forwards (deepen
/// and root); the combined column is the per-turn effective rate that actually
/// decides the knob.
pub fn run(paths: &RunPaths, big_rows: usize, small_rows: usize) -> anyhow::Result<()> {
    let cfg = RunConfig::load(&paths.config)?;
    let device = Device::Cuda(0);
    let side = obs_side(cfg.board as usize) as i64;
    let vs = nn::VarStore::new(device);
    let net = snek_tch::AZNet::new(
        &vs.root(),
        NUM_CHANNELS_TEMP as i64,
        cfg.trunk_channels,
        cfg.trunk_blocks,
    );
    println!(
        "fwd-bench: {}x{} net, obs {}x{}x{}, big={} rows small={} rows, current le_fwd_chunk={}",
        cfg.trunk_channels, cfg.trunk_blocks, NUM_CHANNELS_TEMP, side, side,
        big_rows, small_rows, cfg.le_fwd_chunk,
    );
    println!(
        "{:>6} | {:>10} {:>8} | {:>10} {:>8} | {:>10}",
        "chunk", "big row/s", "ms", "small row/s", "ms", "turn row/s"
    );
    let mut stage = PinnedStage::new(device);
    let mut best: Option<(usize, f64)> = None;
    for &chunk in CHUNKS {
        let (big_rps, big_ms) = measure(&net, device, &mut stage, big_rows, side, chunk);
        let (small_rps, small_ms) = measure(&net, device, &mut stage, small_rows, side, chunk);
        // One real turn = one small + one big call.
        let turn_s = big_rows as f64 / big_rps + small_rows as f64 / small_rps;
        let turn_rps = (big_rows + small_rows) as f64 / turn_s;
        println!(
            "{:>6} | {:>10.0} {:>8.1} | {:>10.0} {:>8.1} | {:>10.0}",
            chunk, big_rps, big_ms, small_rps, small_ms, turn_rps
        );
        if best.map(|(_, r)| turn_rps > r).unwrap_or(true) {
            best = Some((chunk, turn_rps));
        }
    }
    if let Some((chunk, rps)) = best {
        println!("best: le_fwd_chunk={chunk} ({rps:.0} effective rows/s per turn)");
    }
    Ok(())
}
