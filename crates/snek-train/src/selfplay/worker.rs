//! One self-play worker: it owns a contiguous chunk of the in-flight games plus
//! preallocated, reused POD search trees, and drives them turn by turn. Every
//! forward is the identical padded batch shape (cuDNN autotunes exactly once).
//!
//! Each game records its whole history into its frame buffer from turn 0; a
//! finished game is pushed to the shared `recorded` buffer and its board reset to
//! a fresh start, so the chunk carries seamlessly into the next generation.

use super::gpu::Gpu;
use super::materialize::game_sample_count;
use super::rules::{mask_obvious_immediate_deaths, sample_move, terminal_value};
use super::tree::Tree;
use crate::config::RunConfig;
use crate::metrics::Counters;
use crate::sample::{frame_from_board, FrameJson, GameJson};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use snek_core::{encode_into, obs_side, standard_start, Board, Move, NUM_CHANNELS};
use snek_tch::cudagraph::GraphedForward;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Safety cap on frames buffered per recorded game, bounding memory if a game
/// runs pathologically long (e.g. `max_turns == 0` with near-perfect play).
const MAX_REC_FRAMES: usize = 4096;

/// Handles every worker shares: the GPU (behind its inference-queue mutex), the
/// finished-game accumulator, the sample gate, and the run-wide counters/flags.
#[derive(Clone)]
pub(super) struct Shared<'a> {
    pub(super) gpu: Arc<Gpu>,
    pub(super) gpu_lock: Arc<Mutex<()>>,
    /// Running training-sample count across all workers; gates the generation.
    pub(super) samples_total: Arc<AtomicUsize>,
    /// Every finished game this generation (a random subset is later displayed).
    pub(super) recorded: Arc<Mutex<Vec<GameJson>>>,
    /// Board-steps and games completed *in this call* (for per-generation rates).
    pub(super) call_turns: Arc<AtomicU64>,
    pub(super) call_games: Arc<AtomicU64>,
    pub(super) counters: Arc<Counters>,
    pub(super) stop: &'a AtomicBool,
    /// Sample target the generation stops at.
    pub(super) target: usize,
    pub(super) use_graph: bool,
}

/// Play this worker's chunk of games until the sample target is reached or a
/// pause is requested. Mutates the chunk slices in place so carry-out is free.
pub(super) fn play_chunk(
    shared: &Shared,
    cfg: &RunConfig,
    seed: u64,
    chunk: usize,
    bslice: &mut [Board],
    tslice: &mut [usize],
    rslice: &mut [Vec<FrameJson>],
) {
    let n = cfg.num_snakes;
    let side = obs_side(cfg.board as usize);
    let obs_len = NUM_CHANNELS * side * side;
    let board = cfg.board;
    let sims = cfg.sims.max(1);
    let target = shared.target;

    let games = bslice.len();
    let rows = games * n; // constant padded batch shape for this worker
    let mut rng =
        Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x9E37_79B9u64.wrapping_mul(chunk as u64 + 1));
    let cap = sims + 4;

    let mut trees: Vec<Tree> = (0..games)
        .map(|_| Tree::new(n, board, board, cfg.c_puct, cfg.draw_value, cap))
        .collect();

    let mut obs = vec![0.0f32; rows * obs_len];
    let mut pol = vec![0.0f32; rows * 4];
    let mut val = vec![0.0f32; rows];
    let mut root_pol = vec![0.0f32; n * 4];
    let mut root_val = vec![0.0f32; n];
    let mut play_pols = vec![[0.0f32; 4]; n];
    let mut actions = vec![Move::Up; n];

    // Capture this worker's fixed-shape forward as a CUDA graph. Done while
    // holding the GPU lock so no other worker issues CUDA work during the
    // (global-mode) stream capture. The capture pins a side stream current on this
    // thread, which replays then reuse.
    let mut graph = if shared.use_graph {
        let _g = shared.gpu_lock.lock().unwrap();
        let net_ref = unsafe { &*shared.gpu.net };
        Some(GraphedForward::capture(
            net_ref,
            shared.gpu.device,
            rows as i64,
            shared.gpu.c,
            shared.gpu.h,
            shared.gpu.w,
        ))
    } else {
        None
    };

    'gen: while !shared.stop.load(Ordering::Relaxed)
        && shared.samples_total.load(Ordering::Relaxed) < target
    {
        for g in 0..games {
            trees[g].reset(&bslice[g]);
        }

        for _ in 0..sims {
            if shared.stop.load(Ordering::Relaxed) {
                break 'gen;
            }
            for tree in &mut trees {
                tree.select();
            }
            for (g, tree) in trees.iter().enumerate() {
                let base = g * n * obs_len;
                match tree.pending_board() {
                    Some(b) => {
                        for a in 0..n {
                            let off = base + a * obs_len;
                            encode_into(b, a, &mut obs[off..off + obs_len]);
                        }
                    }
                    None => {
                        for x in obs[base..base + n * obs_len].iter_mut() {
                            *x = 0.0;
                        }
                    }
                }
            }
            let dt = {
                let _g = shared.gpu_lock.lock().unwrap();
                let t0 = Instant::now();
                match graph.as_mut() {
                    Some(g) => g.run(&obs, &mut pol, &mut val),
                    None => shared.gpu.forward(&obs, rows, &mut pol, &mut val),
                }
                t0.elapsed().as_micros() as u64
            };
            shared
                .counters
                .gpu_forward_us
                .fetch_add(dt, Ordering::Relaxed);
            shared.counters.gpu_requests.fetch_add(1, Ordering::Relaxed);
            shared
                .counters
                .gpu_rows
                .fetch_add(rows as u64, Ordering::Relaxed);
            shared
                .counters
                .inferences
                .fetch_add(rows as u64, Ordering::Relaxed);
            for g in 0..games {
                let pb = &pol[g * n * 4..(g + 1) * n * 4];
                let vb = &val[g * n..(g + 1) * n];
                trees[g].expand_backup(pb, vb);
            }
        }

        for g in 0..games {
            trees[g].root_targets(&mut root_pol, &mut root_val);

            // Choose moves (masked visit policy + exploration).
            for s in 0..n {
                let play =
                    mask_obvious_immediate_deaths(&bslice[g], s, &root_pol[s * 4..s * 4 + 4]);
                play_pols[s].copy_from_slice(&play);
                actions[s] = sample_move(&play, cfg.exploration_prob, &mut rng);
            }

            // Record this turn's frame (pre-step board + search readout). Every
            // game records from its turn 0, so the buffer always holds a whole
            // game — the frames are the training data.
            rslice[g].push(frame_from_board(
                &bslice[g], n, &root_pol, &root_val, &play_pols, &actions,
            ));

            bslice[g].step_and_spawn(&actions, &mut rng);
            tslice[g] += 1;
            shared.call_turns.fetch_add(1, Ordering::Relaxed);

            // A game ends on a terminal board, at the turn cap, or if it somehow
            // hits the frame safety cap (bounds memory for pathological configs
            // like `max_turns == 0`).
            let overrun = (cfg.max_turns > 0 && tslice[g] >= cfg.max_turns)
                || rslice[g].len() >= MAX_REC_FRAMES;
            if !bslice[g].is_terminal() && !overrun {
                continue;
            }

            let winner = bslice[g].winner();
            // Skip short draws entirely (no sample, no replay).
            if winner.is_none()
                && !overrun
                && cfg.skip_short_draw_turns > 0
                && rslice[g].len() <= cfg.skip_short_draw_turns
            {
                rslice[g].clear();
                bslice[g] = standard_start(board, board, n, &mut rng);
                tslice[g] = 0;
                continue;
            }

            // Append the terminal frame (zero policy, terminal values for display)
            // and move the whole game into the finished buffer.
            let term_vals: Vec<f32> = (0..n)
                .map(|s| terminal_value(winner, s, bslice[g].snakes[s].alive(), cfg.draw_value))
                .collect();
            let mut frames = std::mem::take(&mut rslice[g]);
            frames.push(frame_from_board(
                &bslice[g],
                n,
                &vec![0.0f32; n * 4],
                &term_vals,
                &vec![[0.0f32; 4]; n],
                &actions,
            ));
            let num_turns = frames.len() as u32;
            let game = GameJson {
                frames,
                winner: winner.map(|x| x as i32),
                num_turns,
            };
            let added = game_sample_count(&game, n);
            shared.recorded.lock().unwrap().push(game);

            let total = shared.samples_total.fetch_add(added, Ordering::Relaxed) + added;
            shared
                .counters
                .samples_collected
                .store(total.min(target) as u32, Ordering::Relaxed);
            shared.call_games.fetch_add(1, Ordering::Relaxed);
            shared
                .counters
                .completed_games
                .fetch_add(1, Ordering::Relaxed);
            shared
                .counters
                .completed_turns
                .fetch_add(tslice[g] as u64, Ordering::Relaxed);

            bslice[g] = standard_start(board, board, n, &mut rng);
            tslice[g] = 0;
        }
    }
}
