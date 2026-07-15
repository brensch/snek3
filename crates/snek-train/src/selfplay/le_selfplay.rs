//! Logit-Equilibrium self-play — the "correct game mode" analogue of
//! [`super::generate`].
//!
//! Single-pass by design (see `docs/le-overhaul.md`, Phase 3 fork): the LE leaf
//! batch is *variable-shape* every turn (games branch and terminate
//! independently), so there is no fixed-shape CUDA graph / double buffer. Each
//! turn is: one [`EqForest::build`] over all live games → encode every leaf once
//! per seat with the τ-conditioned value net → one batched value forward → one
//! per-node LE backup → record the root mixed policy + per-player value into the
//! frame → mix uniform exploration and sample the played move → step.
//!
//! τ is per-**game** (assigned at game start, constant for the game's life,
//! carried in `SelfPlayState::temp`), because games persist across generations
//! here.

use super::materialize::{
    game_matches_shape, game_sample_count, in_flight_matches_shape, materialize_le_game,
};
use super::rules::mix_root_dirichlet;
use super::{assign_heur_seats, GenOutcome, SelfPlayNet, SelfPlayState};
use crate::config::RunConfig;
use crate::metrics::Metrics;
use crate::replay::Samples;
use crate::sample::{frame_from_board, GameJson};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use snek_core::{
    encode_into_temp, obs_side, standard_start, Move, MAX_SNAKES, NUM_CHANNELS, NUM_CHANNELS_TEMP,
};
use snek_search::{obvious_immediate_death, EqForest};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tch::{no_grad, Device, Tensor};

/// Sample a per-episode inverse temperature τ ~ U[tau_min, tau_max].
fn sample_tau(cfg: &RunConfig, rng: &mut impl Rng) -> f32 {
    let lo = cfg.tau_min.min(cfg.tau_max);
    let hi = cfg.tau_max.max(cfg.tau_min);
    if hi <= lo {
        lo
    } else {
        rng.gen_range(lo..=hi)
    }
}

/// Sample a move index from a 4-way distribution (Up if degenerate/dead).
fn sample_move(dist: &[f32; 4], rng: &mut impl Rng) -> Move {
    let s: f32 = dist.iter().sum();
    if s <= 0.0 {
        return Move::Up;
    }
    let mut r = rng.gen::<f32>() * s;
    for (i, &p) in dist.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return Move::from_index(i);
        }
    }
    Move::from_index(3)
}

/// Value-head-only forward of a τ-conditioned (`NUM_CHANNELS_TEMP`) obs batch.
fn forward_values(net: &snek_tch::AZNet, device: Device, obs: &[f32], rows: usize, h: i64, w: i64) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    no_grad(|| {
        let x = Tensor::from_slice(obs)
            .reshape([rows as i64, NUM_CHANNELS_TEMP as i64, h, w])
            .to_device(device);
        let (_logits, value) = net.forward(&x);
        value.to_device(Device::Cpu).copy_data(&mut out, rows);
    });
    out
}

/// Persistent pinned-memory staging buffer for the chunked leaf forward.
/// Encoding writes straight into page-locked host memory, so each turn skips
/// BOTH the ~100s-of-MB `vec![0.0; ..]` alloc+zero (`encode_into` fully
/// overwrites its slice anyway) AND the extra host-side copy
/// `Tensor::from_slice` would do — and the single H2D then runs at pinned
/// transfer speed (~2-3× pageable). Grown on demand, reused across turns.
pub(crate) struct PinnedStage {
    buf: Option<Tensor>,
    cap: usize,
    device: Device,
}

impl PinnedStage {
    pub(crate) fn new(device: Device) -> Self {
        Self { buf: None, cap: 0, device }
    }

    /// Mutable float view of the first `len` elements, (re)allocating pinned
    /// storage if needed. Writing from a rayon loop is fine: plain f32 memory,
    /// uniquely borrowed through `&mut self`, no autograd attached.
    pub(crate) fn slice(&mut self, len: usize) -> &mut [f32] {
        if self.cap < len {
            // Grow in 1M-float (4MB) steps so a slowly rising leaf count
            // doesn't re-pin every turn.
            let cap = len.next_multiple_of(1 << 20);
            let t = Tensor::empty([cap as i64], (tch::Kind::Float, Device::Cpu));
            self.buf = Some(if matches!(self.device, Device::Cuda(_)) {
                t.pin_memory(self.device)
            } else {
                t
            });
            self.cap = cap;
        }
        let t = self.buf.as_ref().expect("stage allocated above");
        unsafe { std::slice::from_raw_parts_mut(t.data_ptr() as *mut f32, len) }
    }

    pub(crate) fn tensor(&self) -> &Tensor {
        self.buf.as_ref().expect("slice() must be called before tensor()")
    }
}

/// Chunked no-sync forward reading from a `PinnedStage` tensor. `padded` must
/// be a whole number of `chunk_rows` (the caller zero-fills the tail rows), so
/// every slice forward keeps the identical benchmark-friendly shape and the
/// input needs no GPU-side padding cat.
pub(crate) fn forward_values_staged(
    net: &snek_tch::AZNet,
    device: Device,
    stage: &Tensor,
    rows: usize,
    padded: usize,
    side: i64,
    chunk_rows: usize,
) -> Vec<f32> {
    let c = NUM_CHANNELS_TEMP as i64;
    let mut out = vec![0.0f32; rows];
    if rows == 0 {
        return out;
    }
    no_grad(|| {
        // One pinned H2D of the padded batch, then every slice forward is
        // enqueued with no host sync until the single D2H at the end.
        let gpu = stage
            .narrow(0, 0, padded as i64 * c * side * side)
            .view([padded as i64, c, side, side])
            .to_device(device);
        let nchunks = padded / chunk_rows;
        let mut vparts = Vec::with_capacity(nchunks);
        for ci in 0..nchunks {
            let slice = gpu.narrow(0, (ci * chunk_rows) as i64, chunk_rows as i64);
            let (_logits, value) = net.forward(&slice);
            vparts.push(value);
        }
        Tensor::cat(&vparts, 0)
            .narrow(0, 0, rows as i64)
            .to_device(Device::Cpu)
            .copy_data(&mut out, rows);
    });
    out
}

pub fn generate_le(
    net: &SelfPlayNet<'_>,
    cfg: &RunConfig,
    seed: u64,
    metrics: &Metrics,
    stop: &AtomicBool,
    state: &mut SelfPlayState,
) -> anyhow::Result<GenOutcome> {
    let counters = metrics.counters();
    let n = cfg.num_snakes;
    let board = cfg.board;
    let h = obs_side(board as usize);
    let w = obs_side(board as usize);
    let l15 = NUM_CHANNELS_TEMP * h * w;
    let target = cfg.samples_per_gen;
    let count = cfg.concurrent_games();
    let depth = cfg.le_depth.max(1);
    let iters = cfg.le_iters.max(1);
    let side = h as i64;
    // Chunked forward: target ROWS per chunk -> boards per chunk (board = n rows).
    let chunk_boards = if cfg.le_fwd_chunk > 0 {
        (cfg.le_fwd_chunk / n).max(1)
    } else {
        0
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x1E4A_B5A7_0C0D_1234u64);
    let gpu_net = unsafe { &*(net.net as *const snek_tch::AZNet) };
    let device = net.device;

    // ---- reconcile carried in-flight games (boards/turns/rec/temp quadruple) ----
    if state.boards.len() != state.turns.len() || state.boards.len() != state.rec.len() {
        state.boards.clear();
        state.turns.clear();
        state.rec.clear();
        state.temp.clear();
    }
    // Realign τ to the board count (fresh run: empty; resume: reuse; new slots get τ).
    while state.temp.len() < state.boards.len() {
        let t = sample_tau(cfg, &mut rng);
        state.temp.push(t);
    }
    state.temp.truncate(state.boards.len());

    let mut i = 0;
    while i < state.boards.len() {
        let resumable = !state.boards[i].is_terminal()
            && in_flight_matches_shape(&state.boards[i], &state.rec[i], board, n);
        if resumable {
            i += 1;
        } else {
            state.boards.swap_remove(i);
            state.turns.swap_remove(i);
            state.rec.swap_remove(i);
            state.temp.swap_remove(i);
        }
    }
    while state.boards.len() > count {
        let v = rng.gen_range(0..state.boards.len());
        state.boards.swap_remove(v);
        state.turns.swap_remove(v);
        state.rec.swap_remove(v);
        state.temp.swap_remove(v);
    }
    while state.boards.len() < count {
        let mut b = standard_start(board, board, n, &mut rng);
        b.heur_mask = assign_heur_seats(cfg, &mut rng);
        state.boards.push(b);
        state.turns.push(0);
        state.rec.push(Vec::new());
        let t = sample_tau(cfg, &mut rng);
        state.temp.push(t);
    }

    counters.inflight_games.store(count as u32, Ordering::Relaxed);
    counters.inflight_turn_sum.store(
        state.turns.iter().map(|&t| t as u64).sum(),
        Ordering::Relaxed,
    );
    state.finished.retain(|g| game_matches_shape(g, board, n));
    let mut samples_total: usize = state.finished.iter().map(|g| game_sample_count(g, n)).sum();
    counters.samples_target.store(target as u32, Ordering::Relaxed);
    counters
        .samples_collected
        .store(samples_total.min(target) as u32, Ordering::Relaxed);

    // ---- main loop: one equilibrium ply across all live games per iteration ----
    // Per-phase timers so we can see where the LE turn actually spends its wall
    // clock (build joint tree / encode leaves / GPU forward / SFP backup /
    // record+step). Logged once per gen; the whole point is to stop guessing.
    // Pinned staging reused across every turn's forwards (chunked path only).
    let mut stage = PinnedStage::new(device);
    let (mut t_build, mut t_encode, mut t_fwd, mut t_backup, mut t_step) =
        (0u128, 0u128, 0u128, 0u128, 0u128);
    let mut nturns: u64 = 0;
    while samples_total < target && !stop.load(Ordering::Relaxed) {
        let tb = Instant::now();
        // Selective deepening builds depth-1 then expands the top-k successors;
        // fixed mode builds straight to `depth`.
        let build_depth = if cfg.le_top_k > 0 { 1 } else { depth };
        let mut forest = EqForest::build(&state.boards, build_depth, cfg.draw_value);
        t_build += tb.elapsed().as_micros();
        let tau_pg: Vec<[f32; MAX_SNAKES]> =
            state.temp.iter().map(|&t| [t; MAX_SNAKES]).collect();
        let num_eval = forest.eval_boards().len();
        let rows = num_eval * n;

        let _ = rows;
        // Leaf-value forward, shared by both phases. When le_fwd_chunk>0 this is
        // the chunked+pipelined path (fixed-shape sub-batches at the GPU sweet
        // spot — benchmark stays on); otherwise one variable-shape forward. τ is
        // snapshotted so the closure owns it (no borrow of the mutated `state`).
        let temp_snapshot = state.temp.clone();
        let stage = &mut stage;
        let mut fwd = |eb: &[snek_core::Board], eg: &[u32]| -> Vec<f32> {
            let r = eb.len() * n;
            if r == 0 {
                return Vec::new();
            }
            let v = if chunk_boards > 0 {
                // Chunked path: encode straight into the persistent pinned
                // stage (no per-turn alloc/zero, no from_slice copy), pad on
                // the host to a whole number of fixed-shape chunks.
                let chunk = chunk_boards * n;
                let padded = r.div_ceil(chunk) * chunk;
                let buf = stage.slice(padded * l15);
                buf[r * l15..].fill(0.0); // pad rows only; encode overwrites the rest
                buf[..r * l15].par_chunks_mut(n * l15).enumerate().for_each(|(e, chunk)| {
                    let tau = temp_snapshot[eg[e] as usize];
                    for s in 0..n {
                        encode_into_temp(&eb[e], s, tau, &mut chunk[s * l15..(s + 1) * l15]);
                    }
                });
                forward_values_staged(gpu_net, device, stage.tensor(), r, padded, side, chunk)
            } else {
                // Legacy path (le_fwd_chunk=0): one variable-shape forward.
                let mut obs = vec![0.0f32; r * l15];
                obs.par_chunks_mut(n * l15).enumerate().for_each(|(e, chunk)| {
                    let tau = temp_snapshot[eg[e] as usize];
                    for s in 0..n {
                        encode_into_temp(&eb[e], s, tau, &mut chunk[s * l15..(s + 1) * l15]);
                    }
                });
                forward_values(gpu_net, device, &obs, r, side, side)
            };
            counters.gpu_requests.fetch_add(1, Ordering::Relaxed);
            counters.gpu_rows.fetch_add(r as u64, Ordering::Relaxed);
            counters.inferences.fetch_add(r as u64, Ordering::Relaxed);
            v
        };
        let tfwd = Instant::now();
        let mut vals = fwd(forest.eval_boards(), forest.eval_game());
        let dt = tfwd.elapsed().as_micros();
        t_fwd += dt;
        counters.gpu_forward_us.fetch_add(dt as u64, Ordering::Relaxed);

        // Selective deepening (docs/le-selective-depth.md): expand the top-k
        // joint successors one more ply and value-net the new leaves, then
        // re-solve. Two batched forwards total; skipped when le_top_k == 0.
        if cfg.le_top_k > 0 {
            let start = forest.deepen_topk(&vals, &tau_pg, iters, cfg.le_top_k, cfg.draw_value);
            let tfwd2 = Instant::now();
            let vals2 = fwd(&forest.eval_boards()[start..], &forest.eval_game()[start..]);
            let dt = tfwd2.elapsed().as_micros();
            t_fwd += dt;
            counters.gpu_forward_us.fetch_add(dt as u64, Ordering::Relaxed);
            vals.extend(vals2);
        }

        let tk = Instant::now();
        let roots = forest.backup(&vals, &tau_pg, iters);
        t_backup += tk.elapsed().as_micros();
        nturns += 1;
        let tstep = Instant::now();

        // Record the equilibrium frame, sample the played move, and step.
        for g in 0..state.boards.len() {
            let eq = &roots[g];
            let mut policy = vec![0.0f32; n * 4];
            let mut values = vec![0.0f32; n];
            let mut play_pols = vec![[0.0f32; 4]; n];
            let mut actions = vec![Move::Up; n];
            for s in 0..n {
                policy[s * 4..s * 4 + 4].copy_from_slice(&eq.policy[s]);
                values[s] = eq.value[s];
                // Played distribution = the clean LE policy, SAMPLED (never
                // argmax — a mixed equilibrium must stay mixed to be
                // non-exploitable). For the opening `sample_turns` moves we add
                // AlphaZero-style root Dirichlet noise for off-policy opening
                // diversity; after that it's the raw equilibrium. The TRAINING
                // target (`policy`, set above) is always the clean LE policy —
                // exploration noise never touches the label.
                let mut play = eq.policy[s];
                if state.turns[g] < cfg.sample_turns {
                    mix_root_dirichlet(
                        &mut play,
                        4,
                        cfg.dirichlet_frac,
                        cfg.dirichlet_alpha,
                        &mut rng,
                    );
                    // Keep the suicide mask intact under opening noise: Dirichlet
                    // must not resurrect a certain-death move the search pruned.
                    // If EVERY move is fatal (trapped), the masked policy sums to
                    // ~0 — fall back to the noised policy so the doomed snake
                    // still has a valid distribution (never all-zero => no NaN).
                    let mut masked = play;
                    for m in 0..4 {
                        if obvious_immediate_death(&state.boards[g], s, Move::from_index(m)) {
                            masked[m] = 0.0;
                        }
                    }
                    let sum: f32 = masked.iter().sum();
                    if sum > 1e-6 {
                        for p in masked.iter_mut() {
                            *p /= sum;
                        }
                        play = masked;
                    }
                }
                play_pols[s] = play;
                actions[s] = sample_move(&play, &mut rng);
            }
            let frame = frame_from_board(&state.boards[g], n, &policy, &values, &play_pols, &actions);
            state.rec[g].push(frame);
            // Real play must replenish food (SpawnFoodStandard); the plain `step`
            // (no food) is only for the search's deterministic lookahead. Using
            // it here made every game a foodless starvation race.
            state.boards[g].step_and_spawn(&actions, &mut rng);
            state.turns[g] += 1;
        }

        // Finalize any game that ended (or hit the turn cap) and reset its slot.
        for g in 0..state.boards.len() {
            // Games always run to a natural terminal — no turn cap. Health decay
            // (a snake must eat within ~100 turns or starve), growth on eating,
            // and the finite board guarantee termination.
            let over = state.boards[g].is_terminal();
            if !over {
                continue;
            }
            // A terminal frame (its alive mask is read by materialize as the
            // final-alive state); it is itself excluded from training.
            let zeros_p = vec![0.0f32; n * 4];
            let zeros_v = vec![0.0f32; n];
            let zeros_pp = vec![[0.0f32; 4]; n];
            let dummy = vec![Move::Up; n];
            let tframe = frame_from_board(&state.boards[g], n, &zeros_p, &zeros_v, &zeros_pp, &dummy);
            state.rec[g].push(tframe);

            let winner = state.boards[g].winner();
            let frames = std::mem::take(&mut state.rec[g]);
            let num_turns = frames.len() as u32;
            let game = GameJson {
                frames,
                winner: winner.map(|x| x as i32),
                num_turns,
                heur_mask: state.boards[g].heur_mask,
                temp: state.temp[g],
            };
            samples_total += game_sample_count(&game, n);
            state.finished.push(game);

            let mut b = standard_start(board, board, n, &mut rng);
            b.heur_mask = assign_heur_seats(cfg, &mut rng);
            state.boards[g] = b;
            state.turns[g] = 0;
            state.temp[g] = sample_tau(cfg, &mut rng);
        }

        t_step += tstep.elapsed().as_micros();
        counters.inflight_turn_sum.store(
            state.turns.iter().map(|&t| t as u64).sum(),
            Ordering::Relaxed,
        );
        counters
            .samples_collected
            .store(samples_total.min(target) as u32, Ordering::Relaxed);
    }

    // Per-gen perf breakdown (ms totals + per-turn) — the ground truth for where
    // the LE self-play wall clock goes, so throughput work is measured not guessed.
    if nturns > 0 {
        let ms = |x: u128| x as f64 / 1000.0;
        let per = |x: u128| x as f64 / 1000.0 / nturns as f64;
        metrics.log(format!(
            "LE perf: {nturns} turns | build {:.0}ms({:.1}) encode {:.0}ms({:.1}) forward {:.0}ms({:.1}) backup {:.0}ms({:.1}) step {:.0}ms({:.1}) [ms total(per-turn)]",
            ms(t_build), per(t_build), ms(t_encode), per(t_encode),
            ms(t_fwd), per(t_fwd), ms(t_backup), per(t_backup), ms(t_step), per(t_step),
        ));
    }

    // ---- completion / interrupt (mirrors `generate`) ----
    let completed = samples_total >= target && !stop.load(Ordering::Relaxed);
    if !completed {
        return Ok(GenOutcome::Interrupted);
    }

    let recorded = std::mem::take(&mut state.finished);
    // Samples store the 14-ch board obs; the τ plane is appended at train time
    // from `Samples.temp`, so obs_shape stays [NUM_CHANNELS, h, w].
    let mut samples = Samples {
        obs: Vec::new(),
        pol: Vec::new(),
        z: Vec::new(),
        turn: Vec::new(),
        temp: Vec::new(),
        obs_shape: [NUM_CHANNELS, h, w],
        // Total turns played across this gen's finished games — drives ms/turn
        // and turns/sec on the dashboard. Was hardcoded 0, which made the LE
        // throughput readout meaningless (ms/turn divided by 1).
        turns: recorded.iter().map(|g| g.num_turns as usize).sum(),
        games: recorded.len(),
    };
    for g in &recorded {
        materialize_le_game(g, n, cfg, &mut samples);
    }
    counters
        .samples_collected
        .store(samples.len().min(target) as u32, Ordering::Relaxed);

    // Dashboard subset.
    let mut sel_rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x5EED_D15D_1A17_9999u64);
    let mut rec2 = recorded;
    let k = cfg.sample_games.min(rec2.len());
    let display_games = if k == rec2.len() {
        std::mem::take(&mut rec2)
    } else {
        rand::seq::index::sample(&mut sel_rng, rec2.len(), k)
            .into_iter()
            .map(|i| rec2[i].clone())
            .collect()
    };

    Ok(GenOutcome::Complete {
        samples,
        display_games,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RunConfig;
    use crate::metrics::Metrics;
    use crate::selfplay::SelfPlayState;
    use crate::train::{build_optimizer, train_one};
    use tch::nn;

    fn tiny_le_cfg() -> RunConfig {
        let mut c = RunConfig::default();
        c.le_mode = true;
        c.le_depth = 1;
        c.le_iters = 40;
        c.num_snakes = 4;
        c.board = 11;
        c.gpu_batch_games = 1; // concurrent_games = 1 * GPU_PIPELINE_BUFFERS
        c.samples_per_gen = 80;
        c.sample_games = 1;
        c.tau_min = 0.5;
        c.tau_max = 10.0;
        c.le_exploration = 0.15;
        c.le_outcome_weight = 0.5;
        c
    }

    #[test]
    fn le_selfplay_and_train_smoke() {
        let device = Device::Cpu;
        let cfg = tiny_le_cfg();
        let vs = nn::VarStore::new(device);
        let net = snek_tch::AZNet::new(&vs.root(), NUM_CHANNELS_TEMP as i64, 8, 1);
        snek_tch::init_orthogonal(&vs, 2f64.sqrt());
        let metrics = Metrics::new();
        let stop = AtomicBool::new(false);
        let mut state = SelfPlayState::default();
        let sp_net = SelfPlayNet { net: &net, device };

        let outcome = generate_le(&sp_net, &cfg, 12345, &metrics, &stop, &mut state).unwrap();
        let samples = match outcome {
            GenOutcome::Complete { samples, .. } => samples,
            GenOutcome::Interrupted => panic!("unexpected interrupt"),
        };

        assert!(samples.len() > 0, "produced training samples");
        assert_eq!(samples.temp.len(), samples.len(), "every sample carries τ");
        assert!(samples
            .temp
            .iter()
            .all(|&t| t >= cfg.tau_min - 1e-3 && t <= cfg.tau_max + 1e-3));

        // The whole point: policies are *mixed* distributions summing to ~1,
        // and at least some are genuinely mixed (not collapsed one-hots).
        let mut any_mixed = false;
        for row in samples.pol.chunks(4) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-2, "policy row sums to 1 (got {s})");
            if row.iter().cloned().fold(0.0f32, f32::max) < 0.98 {
                any_mixed = true;
            }
        }
        assert!(any_mixed, "the equilibrium policy is genuinely mixed");
        assert!(
            samples.z.iter().all(|&v| v.abs() <= 1.0 + 1e-3),
            "value targets in [-1,1]"
        );

        // τ actually reaches the net: same board, different τ → different value.
        let hw = obs_side(cfg.board as usize) * obs_side(cfg.board as usize);
        let obs14 = vec![0.3f32; NUM_CHANNELS * hw];
        let a = snek_core::append_temp_planes(&obs14, &[0.5], hw);
        let b = snek_core::append_temp_planes(&obs14, &[18.0], hw);
        let side = obs_side(cfg.board as usize) as i64;
        let va = forward_values(&net, device, &a, 1, side, side)[0];
        let vb = forward_values(&net, device, &b, 1, side, side)[0];
        assert!((va - vb).abs() > 1e-7, "τ plane conditions the net (va={va}, vb={vb})");

        // A train step runs and yields finite losses on the LE (τ-conditioned) batch.
        let mut opt = build_optimizer(&vs, &cfg).unwrap();
        let m = train_one(&net, device, &mut opt, &samples, cfg.value_weight).unwrap();
        assert!(
            m.policy_loss.is_finite() && m.value_loss.is_finite(),
            "finite train losses ({}, {})",
            m.policy_loss,
            m.value_loss
        );
    }
}
