//! Self-play generation: a from-scratch, railed decoupled-PUCT engine.
//!
//! The only shared resource is the GPU (see [`gpu`]); the number of workers is
//! derived from the GPU batch size ([`RunConfig::concurrent_games`] — a double
//! buffer), not configured separately, so the only GPU dial is the batch size.
//! Each worker ([`worker`]) owns a fixed set of games plus preallocated, reused
//! POD search trees ([`tree`]).
//!
//! Every in-flight game records its *whole* history (one [`FrameJson`] per turn,
//! from turn 0) into the carried [`SelfPlayState`]. That frame history is the
//! single source of truth: a training sample's observation is a pure function of
//! a frame (bodies/health/food/hazards) and the frame also carries the search
//! policy, root value and alive mask — so on completion a finished game is
//! materialised straight from its frames ([`materialize`]) into `(obs, pol, z,
//! turn)` samples, and the same frames are what the dashboard replays. Nothing is
//! stored twice, so the whole session (in-flight games + finished games) persists
//! compactly and a pause/resume continues a game — and a generation — exactly
//! where it stopped.

mod gpu;
mod materialize;
mod rules;
mod tree;
mod worker;

use crate::config::RunConfig;
use crate::metrics::Metrics;
use crate::replay::Samples;
use crate::sample::{FrameJson, GameJson};
use gpu::Gpu;
use materialize::{game_matches_shape, game_sample_count, materialize_game};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use snek_core::{obs_side, standard_start, Board, NUM_CHANNELS};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tch::Device;

const MAXC: usize = 4; // max candidate moves per snake
const EPS: f32 = 1e-8;

pub struct SelfPlayNet<'a> {
    pub net: &'a snek_tch::AZNet,
    pub device: Device,
}

/// In-progress self-play state, owned by the trainer and carried across
/// generations *and* restarts. Persisted whole on pause (see [`crate::session`])
/// so a resume continues every in-flight game — and the current generation's
/// accumulated samples — exactly where they stopped.
#[derive(Default)]
pub struct SelfPlayState {
    /// In-flight game boards (`concurrent_games` of them).
    pub boards: Vec<Board>,
    /// Absolute engine turn count per in-flight game (parallel to `boards`).
    pub turns: Vec<usize>,
    /// Full frame history of each in-flight game, from its turn 0. The single
    /// source of truth for both training samples and browsable replays.
    pub rec: Vec<Vec<FrameJson>>,
    /// Every game that has finished in the current (not-yet-committed)
    /// generation. A random subset is written for the dashboard; all of them are
    /// materialised into this generation's training samples when the sample
    /// target is reached. Carried across a pause so the accumulated samples are
    /// never lost.
    pub finished: Vec<GameJson>,
}

impl SelfPlayState {
    /// Training samples the finished-game buffer would contribute to the current
    /// generation (i.e. progress toward `samples_per_gen`). For logging.
    pub fn pending_sample_count(&self, n: usize) -> usize {
        self.finished.iter().map(|g| game_sample_count(g, n)).sum()
    }
}

/// Outcome of one [`generate`] call.
pub enum GenOutcome {
    /// The sample target was reached. Carries this generation's training samples
    /// (whole games only) and the random subset of finished games chosen for the
    /// dashboard. The in-flight games remain in [`SelfPlayState`] for the next
    /// generation; the finished-game buffer has been drained.
    Complete {
        samples: Samples,
        display_games: Vec<GameJson>,
    },
    /// A pause was requested before the target was reached. [`SelfPlayState`] has
    /// been fully updated in place (in-flight games advanced, finished games
    /// retained), so persisting and reloading it resumes exactly where we
    /// stopped — same generation, same accumulated samples.
    Interrupted,
}

/// Play one self-play generation, resuming the carried `state`.
///
/// Every in-flight game records its whole history into `state.rec`; a finished
/// game is moved into `state.finished`. The generation ends when the finished
/// games hold at least `samples_per_gen` training samples ([`GenOutcome::Complete`])
/// or a pause is requested ([`GenOutcome::Interrupted`]). On completion the
/// finished games are materialised into whole-game training samples and a random
/// `sample_games` of them are returned for the dashboard; on interrupt everything
/// is left in `state` for a seamless resume.
pub fn generate(
    net: &SelfPlayNet<'_>,
    cfg: &RunConfig,
    seed: u64,
    metrics: &Metrics,
    stop: &AtomicBool,
    state: &mut SelfPlayState,
) -> anyhow::Result<GenOutcome> {
    let counters = metrics.counters();

    let n = cfg.num_snakes;
    let c = NUM_CHANNELS;
    let h = obs_side(cfg.board as usize);
    let w = obs_side(cfg.board as usize);
    let obs_len = c * h * w;
    let board = cfg.board;
    let target = cfg.samples_per_gen;

    let chunk_games = cfg.gpu_batch_games.max(1);
    // Concurrency is derived from the GPU batch size (a double buffer): one worker
    // per buffer, each holding one GPU batch of games.
    let count = cfg.concurrent_games();

    // (Re)initialise the in-flight buffer to exactly `count` games. Any structural
    // mismatch (batch size changed, or a corrupt/old snapshot with no frame
    // history) rebuilds it from fresh starts — that keeps the whole-game invariant
    // (a game's samples/replay always span its entire life) unconditionally true.
    let mut init_rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0xCA22_1E5D_F00Du64);
    if state.boards.len() != count || state.turns.len() != count || state.rec.len() != count {
        state.boards = (0..count)
            .map(|_| standard_start(board, board, n, &mut init_rng))
            .collect();
        state.turns = vec![0usize; count];
        state.rec = (0..count).map(|_| Vec::new()).collect();
    } else {
        // A carried board that is somehow already terminal gets a fresh start so
        // we never resume a dead game (its frames are dropped with it).
        for ((b, t), r) in state
            .boards
            .iter_mut()
            .zip(state.turns.iter_mut())
            .zip(state.rec.iter_mut())
        {
            if b.is_terminal() {
                *b = standard_start(board, board, n, &mut init_rng);
                *t = 0;
                r.clear();
            }
        }
    }

    // Drop carried finished games whose shape no longer matches the current config
    // (board size or snake count changed): their frames would encode to a
    // different obs and corrupt the shard.
    state.finished.retain(|g| game_matches_shape(g, board, n));

    // Shared accumulator: the finished games (seeded with any carried across a
    // resume) plus the running training-sample count that gates the generation.
    let carried_samples: usize = state.finished.iter().map(|g| game_sample_count(g, n)).sum();
    let recorded = Arc::new(Mutex::new(std::mem::take(&mut state.finished)));
    let samples_total = Arc::new(AtomicUsize::new(carried_samples));
    let call_turns = Arc::new(AtomicU64::new(0));
    let call_games = Arc::new(AtomicU64::new(0));

    counters
        .samples_target
        .store(target as u32, Ordering::Relaxed);
    counters
        .samples_collected
        .store(carried_samples.min(target) as u32, Ordering::Relaxed);

    let shared = worker::Shared {
        gpu: Arc::new(Gpu {
            net: net.net as *const snek_tch::AZNet,
            device: net.device,
            c: c as i64,
            h: h as i64,
            w: w as i64,
        }),
        gpu_lock: Arc::new(Mutex::new(())),
        samples_total: Arc::clone(&samples_total),
        recorded: Arc::clone(&recorded),
        call_turns: Arc::clone(&call_turns),
        call_games: Arc::clone(&call_games),
        counters: Arc::clone(&counters),
        stop,
        target,
        // Capture the fixed-shape forward as a CUDA graph (one launch per replay).
        // A big win on faster GPUs where per-launch overhead dominates. Set
        // SNEK_CUDA_GRAPH=0 to disable.
        use_graph: std::env::var("SNEK_CUDA_GRAPH").as_deref() != Ok("0"),
    };

    // Disjoint mutable views: each worker owns one contiguous chunk of games and
    // mutates the shared in-flight buffer in place (so carry-out is free).
    let board_chunks: Vec<&mut [Board]> = state.boards.chunks_mut(chunk_games).collect();
    let turn_chunks: Vec<&mut [usize]> = state.turns.chunks_mut(chunk_games).collect();
    let rec_chunks: Vec<&mut [Vec<FrameJson>]> = state.rec.chunks_mut(chunk_games).collect();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk, ((bslice, tslice), rslice)) in board_chunks
            .into_iter()
            .zip(turn_chunks)
            .zip(rec_chunks)
            .enumerate()
        {
            let shared = shared.clone();
            let cfg = cfg.clone();
            handles.push(scope.spawn(move || {
                worker::play_chunk(&shared, &cfg, seed, chunk, bslice, tslice, rslice)
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });

    // Release the workers' shared handles so `recorded` is uniquely owned again.
    drop(shared);
    let mut recorded = Arc::try_unwrap(recorded)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_default();

    // A pause always wins: even if the target was reached in the same sweep, we
    // snapshot rather than train, so the accumulated samples ride the session file
    // to the next launch instead of committing a shard mid-pause.
    let completed =
        samples_total.load(Ordering::Relaxed) >= target && !stop.load(Ordering::Relaxed);
    if !completed {
        state.finished = recorded; // hand the finished games back for persistence
        return Ok(GenOutcome::Interrupted);
    }

    // Materialise every finished game into whole-game training samples.
    let mut samples = Samples {
        obs: Vec::new(),
        pol: Vec::new(),
        z: Vec::new(),
        turn: Vec::new(),
        obs_shape: [c, h, w],
        turns: call_turns.load(Ordering::Relaxed) as usize,
        games: call_games.load(Ordering::Relaxed) as usize,
    };
    for g in &recorded {
        materialize_game(g, n, obs_len, cfg.draw_value, &mut samples);
    }
    counters
        .samples_collected
        .store(samples.len().min(target) as u32, Ordering::Relaxed);

    // Pick a uniform-random subset of the finished games for the dashboard.
    let mut sel_rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x5EED_D15D_1A17u64);
    let k = cfg.sample_games.min(recorded.len());
    let display_games = if k == recorded.len() {
        std::mem::take(&mut recorded)
    } else {
        rand::seq::index::sample(&mut sel_rng, recorded.len(), k)
            .into_iter()
            .map(|i| recorded[i].clone())
            .collect()
    };

    // finished buffer drained; in-flight games remain in `state` for the next gen.
    Ok(GenOutcome::Complete {
        samples,
        display_games,
    })
}
