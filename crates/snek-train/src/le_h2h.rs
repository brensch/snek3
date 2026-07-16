//! Head-to-head Logit-Equilibrium matches between two DIFFERENT nets.
//!
//! The LE search normally assumes one net prices every seat: one forward
//! prices all leaves, one solve produces everyone's equilibrium. Here the
//! joint-move TREE is shared (positions are net-independent and built once
//! per turn), but each net produces its own leaf values and its own
//! equilibrium solve over that same tree — every seat then plays the policy
//! from its OWNER's solve, i.e. each player best-responds under its own
//! beliefs. Selective deepening shapes the tree once using the two nets'
//! blended values: tree shape is second-order, and pricing the identical
//! expanded tree keeps the comparison symmetric.
//!
//! Matches are 2v2 with the seat split alternating parity every game, so
//! neither net gets a seat-index or start-position edge; with the caller's
//! fixed seed the games are paired, making this a low-variance comparison —
//! the gate's promotion signal. Head-to-head games are pure GPU (no heuristic
//! search), so a match is FASTER than a baseline eval of the same size.

use crate::config::RunConfig;
use crate::le_eval::{argmax4, survival_placements, RecordedEvalGame};
use crate::sample::{frame_from_board, FrameJson};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use snek_core::{encode_into_temp, obs_side, standard_start, Board, Move, MAX_SNAKES, NUM_CHANNELS_TEMP};
use snek_search::EqForest;
use std::sync::atomic::{AtomicU32, Ordering};
use tch::Device;

pub struct H2hOutcome {
    /// Games where a net-A seat was the sole survivor.
    pub a_wins: usize,
    /// Games where a net-B seat was the sole survivor.
    pub b_wins: usize,
    /// Games with no sole survivor (nobody scores).
    pub draws: usize,
    pub recorded: Vec<RecordedEvalGame>,
}

/// Which side owns seat `s` in game `g`: the even/odd seat split swaps parity
/// every game, so across an even number of games each net plays every seat
/// class equally often.
#[inline]
fn b_owns(g: usize, s: usize) -> bool {
    (s % 2 == 0) == (g % 2 == 0)
}

/// Play `games` head-to-head LE games between `net_a` and `net_b`, both at
/// `serve_tau` argmax (full serve strength). `gen_a`/`gen_b` label the
/// recorded games' seats so the viewer names both players.
#[allow(clippy::too_many_arguments)]
pub fn play_h2h(
    net_a: &snek_tch::AZNet,
    net_b: &snek_tch::AZNet,
    gen_a: u32,
    gen_b: u32,
    device: Device,
    cfg: &RunConfig,
    games: usize,
    serve_tau: f32,
    seed: u64,
    progress: Option<&AtomicU32>,
    record_games: usize,
) -> H2hOutcome {
    let n = cfg.num_snakes;
    let board = cfg.board;
    let side = obs_side(board as usize);
    let l15 = NUM_CHANNELS_TEMP * side * side;
    let iters = cfg.le_iters.max(1);
    let chunk_rows = if cfg.le_fwd_chunk > 0 {
        (cfg.le_fwd_chunk / n).max(1) * n
    } else {
        // The chunked path is also the fixed-shape path required while cuDNN
        // benchmark is ON; fall back to one board-row chunk if unset.
        n
    };

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut boards: Vec<Board> = (0..games)
        .map(|_| standard_start(board, board, n, &mut rng))
        .collect();
    let mut done = vec![false; games];
    let mut a_wins = 0usize;
    let mut b_wins = 0usize;
    let mut draws = 0usize;

    let record_games = record_games.min(games);
    let mut rec_frames: Vec<Vec<FrameJson>> = vec![Vec::new(); record_games];
    let mut death_turn: Vec<[Option<u32>; MAX_SNAKES]> = vec![[None; MAX_SNAKES]; games];

    let mut stage = crate::selfplay::le_selfplay::PinnedStage::new(device);

    while done.iter().any(|&d| !d) {
        let live: Vec<usize> = (0..games).filter(|&g| !done[g]).collect();
        let live_boards: Vec<Board> = live.iter().map(|&g| boards[g].clone()).collect();
        let build_depth = if cfg.le_top_k > 0 { 1 } else { cfg.le_depth.max(1) };
        let mut forest = EqForest::build(&live_boards, build_depth, cfg.draw_value);
        let tau_pg = vec![[serve_tau; MAX_SNAKES]; live_boards.len()];

        // Encode a leaf slice ONCE into the pinned stage, then forward the
        // same staged tensor through BOTH nets (the stage is read-only to the
        // forwards). One encode, two prices.
        let stage = &mut stage;
        let mut enc_fwd2 = |eb: &[Board]| -> (Vec<f32>, Vec<f32>) {
            let rows = eb.len() * n;
            if rows == 0 {
                return (Vec::new(), Vec::new());
            }
            let padded = rows.div_ceil(chunk_rows) * chunk_rows;
            let buf = stage.slice(padded * l15);
            buf[rows * l15..].fill(0.0);
            buf[..rows * l15].par_chunks_mut(n * l15).enumerate().for_each(|(e, chunk)| {
                for s in 0..n {
                    encode_into_temp(&eb[e], s, serve_tau, &mut chunk[s * l15..(s + 1) * l15]);
                }
            });
            let va = crate::selfplay::le_selfplay::forward_values_staged(
                net_a, device, stage.tensor(), rows, padded, side as i64, chunk_rows,
            );
            let vb = crate::selfplay::le_selfplay::forward_values_staged(
                net_b, device, stage.tensor(), rows, padded, side as i64, chunk_rows,
            );
            (va, vb)
        };

        let (mut va, mut vb) = enc_fwd2(forest.eval_boards());
        if cfg.le_top_k > 0 {
            // Shape the deepened tree from the blended beliefs, then price the
            // identical expansion with each net separately.
            let blend: Vec<f32> = va.iter().zip(&vb).map(|(a, b)| 0.5 * (a + b)).collect();
            let start = forest.deepen_topk(&blend, &tau_pg, iters, cfg.le_top_k, cfg.draw_value);
            let (ta, tb) = enc_fwd2(&forest.eval_boards()[start..]);
            va.extend(ta);
            vb.extend(tb);
        }
        let roots_a = forest.backup(&va, &tau_pg, iters);
        let roots_b = forest.backup(&vb, &tau_pg, iters);

        for (li, &g) in live.iter().enumerate() {
            let mut actions = [Move::Up; MAX_SNAKES];
            let mut policy = vec![0.0f32; n * 4];
            let mut values = vec![0.0f32; n];
            let mut play_pols = vec![[0.0f32; 4]; n];
            for s in 0..n {
                if !boards[g].snakes[s].alive() {
                    continue;
                }
                let owner = if b_owns(g, s) { &roots_b[li] } else { &roots_a[li] };
                actions[s] = argmax4(&owner.policy[s]);
                policy[s * 4..s * 4 + 4].copy_from_slice(&owner.policy[s]);
                values[s] = owner.value[s];
                play_pols[s][actions[s].index()] = 1.0;
            }
            if g < record_games {
                rec_frames[g].push(frame_from_board(
                    &boards[g], n, &policy, &values, &play_pols, &actions[..n],
                ));
            }

            let alive_before: [bool; MAX_SNAKES] =
                std::array::from_fn(|s| s < n && boards[g].snakes[s].alive());
            boards[g].step_and_spawn(&actions[..n], &mut rng);
            for s in 0..n {
                if alive_before[s] && !boards[g].snakes[s].alive() && death_turn[g][s].is_none() {
                    death_turn[g][s] = Some(boards[g].turn);
                }
            }

            if boards[g].is_terminal() {
                done[g] = true;
                match boards[g].winner() {
                    Some(w) if b_owns(g, w) => b_wins += 1,
                    Some(_) => a_wins += 1,
                    None => draws += 1,
                }
                if let Some(p) = progress {
                    p.fetch_add(1, Ordering::Relaxed);
                }
                if g < record_games {
                    let zero_pol = vec![0.0f32; n * 4];
                    let zero_val = vec![0.0f32; n];
                    let zero_play = vec![[0.0f32; 4]; n];
                    let noop = [Move::Up; MAX_SNAKES];
                    rec_frames[g].push(frame_from_board(
                        &boards[g], n, &zero_pol, &zero_val, &zero_play, &noop[..n],
                    ));
                }
            }
        }
    }

    let recorded: Vec<RecordedEvalGame> = (0..record_games)
        .map(|g| {
            let placements = survival_placements(&boards[g], &death_turn[g], n, |s| {
                if b_owns(g, s) { gen_b } else { gen_a }
            });
            RecordedEvalGame {
                frames: std::mem::take(&mut rec_frames[g]),
                placements,
                turns: boards[g].turn,
                winner: boards[g].winner().map(|s| s as i32),
            }
        })
        .collect();

    H2hOutcome { a_wins, b_wins, draws, recorded }
}
