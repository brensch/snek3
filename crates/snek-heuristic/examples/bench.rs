//! Measure heuristic MCTS throughput: sims/sec across representative boards,
//! and the wall time of a full move at a few static sim counts. Used to pick
//! the default `HeuristicConfig::max_sims` (~200ms of search).
//!
//!   cargo run --release -p snek-heuristic --example bench

use std::time::Instant;

use snek_core::{standard_start, Board, Move};
use snek_heuristic::{heuristic_move_until, HeuristicConfig};

fn far() -> Instant {
    Instant::now() + std::time::Duration::from_secs(3600)
}

/// Advance a start position a few turns with the heuristic itself, so the
/// bench also covers mid-game shapes (longer snakes, fewer free cells).
fn advance(board: &mut Board, turns: u32, cfg: &HeuristicConfig) {
    let mut rng = rand_pcg();
    for _ in 0..turns {
        if board.is_terminal() {
            return;
        }
        let n = board.snakes.len();
        let moves: Vec<Move> = (0..n)
            .map(|i| {
                Move::from_index(heuristic_move_until(cfg, board, i, far()).move_index)
            })
            .collect();
        board.step_and_spawn(&moves, &mut rng);
    }
}

fn rand_pcg() -> impl rand::Rng {
    use rand::SeedableRng;
    rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42)
}

fn main() {
    let quick = HeuristicConfig {
        max_sims: 256,
        ..Default::default()
    };
    let mut boards: Vec<(String, Board)> = Vec::new();
    let mut rng = rand_pcg();
    for seats in [2usize, 4] {
        let start = standard_start(11, 11, seats, &mut rng);
        boards.push((format!("{seats}p start"), start.clone()));
        for turns in [30u32, 80] {
            let mut b = start.clone();
            advance(&mut b, turns, &quick);
            if !b.is_terminal() && b.snakes[0].alive() {
                boards.push((format!("{seats}p turn {}", b.turn), b));
            }
        }
    }

    for (name, board) in &boards {
        // Warm up, then time a fixed budget.
        let cfg = HeuristicConfig {
            max_sims: 20_000,
            ..Default::default()
        };
        heuristic_move_until(&cfg, board, 0, far());
        let t = Instant::now();
        let d = heuristic_move_until(&cfg, board, 0, far());
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{name:<14} {sims:>6} sims in {ms:>7.1}ms  ({rate:>8.0} sims/s, ~{in200:>6.0} sims/200ms)",
            sims = d.sims,
            ms = secs * 1000.0,
            rate = d.sims as f64 / secs,
            in200 = d.sims as f64 / secs * 0.2,
        );
    }
}
