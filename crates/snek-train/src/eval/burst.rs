//! GPU burst arena: league games at self-play speed, on self-play's schedule.
//!
//! Exactly the self-play shape, played across generations: a fixed set of
//! game slots stays saturated — a finished game is recorded and a fresh
//! matchup starts in its slot immediately — and a burst simply advances the
//! buffer until `burst_games` NEW games have completed. In-flight games are
//! carried in [`ArenaState`] between bursts (and across the trainer's
//! generations), so there is no cold start and no drain tail: the next burst
//! picks up mid-game exactly where this one froze.
//!
//! Batching is by net, made shape-stable by *stations*: K fixed weight slots,
//! each with a fixed block of batch rows. Matchups are drawn from the active
//! stations, so per-station row counts never change and each worker's whole
//! multi-net step (every station's forward+softmax over its block,
//! concatenated) is captured as ONE CUDA graph — a sim step is a single
//! kernel launch. New league entrants swap into a station between bursts.
//!
//! Faithfulness: each seat runs its own decoupled-PUCT search over the same
//! [`Tree`] self-play trains with, evaluating *all* snakes through that
//! seat's own net (a player only knows its own net) — no Dirichlet noise,
//! argmax play: the CPU league's semantics. Built-in heuristic players keep
//! their own static CPU budget, run overlapped with the GPU sim loop, and
//! are capped to a sliver of slots so their 20k-sim CPU moves never stall
//! the lockstep. HTTP players are latency-bound and stay on the CPU league.

use super::{
    fit_ratings, is_external, prune_game_files, ExternalRegistry, LeagueCtx, MatchRecord,
    Placement, KEEP_GAME_FILES,
};
use crate::config::RunConfig;
use crate::metrics::{Counters, Metrics};
use crate::sample::{frame_from_board, FrameJson};
use crate::selfplay::rules::{argmax_move, mask_obvious_immediate_deaths};
use crate::selfplay::tree::Tree;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use snek_core::{encode_into, obs_side, standard_start, Board, Move, NUM_CHANNELS};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;
use tch::{nn, Device};

/// Active net stations (weight slots with fixed batch rows). Small enough
/// that per-station batches stay in the GPU-efficient regime, large enough
/// for informative cross-comparisons; station cycling between bursts covers
/// the rest of the pool over time.
const STATIONS: usize = 8;

/// Recorded game files written per burst, so the eval viewer always has
/// fresh burst games to replay without writing hundreds of MB per cycle.
const RECORDED_GAMES_PER_BURST: usize = 8;

/// Burst games end here if nobody has won (mirrors the CPU league).
const LEAGUE_MAX_TURNS: u32 = 500;

/// Chance a heuristic-eligible slot's next matchup includes one built-in
/// heuristic player (matches the CPU league's forced-external rate).
const HEUR_PLAY_PROB: f64 = 1.0 / 3.0;

/// What one burst actually did, for the caller's log line.
pub struct BurstReport {
    pub games: usize,
    pub turns: u64,
    pub inferences: u64,
    pub seconds: f64,
}

/// One in-flight game, frozen between bursts. Trees and batch-row
/// assignments are rebuilt on thaw — only what defines the game persists.
pub struct CarriedGame {
    /// The matchup's players; seat s plays players[(s + rotation) % len].
    players: Vec<u32>,
    rotation: usize,
    board: Board,
    turn: u32,
    death_turn: Vec<Option<u32>>,
    /// Frames recorded so far (empty unless `record`).
    frames: Vec<FrameJson>,
    record: bool,
    /// Wall seconds this game accumulated in earlier bursts.
    prior_seconds: f64,
}

/// The arena's persistent state, owned by the trainer across generations
/// (and by the standalone runner for one long burst). Fresh default = every
/// slot starts a new game on the first burst.
#[derive(Default)]
pub struct ArenaState {
    /// In-flight games by slot; None starts fresh.
    pub slots: Vec<Option<CarriedGame>>,
    /// Active station gens. Reconciled each burst (entrant joins, one
    /// station cycles); games referencing a dropped station are abandoned.
    pub stations: Vec<u32>,
}

/// Run one burst: advance the carried arena until `cfg.burst_games` new
/// games complete (or `stop`), record them into `eval_dir`'s summary and
/// refit ratings once at the end. In-flight games are left in `state` for
/// the next burst.
pub fn run_burst(
    ctx: &LeagueCtx,
    cfg: &RunConfig,
    device: Device,
    metrics: &Metrics,
    stop: &AtomicBool,
    eval_dir: &Path,
    state: &mut ArenaState,
) -> anyhow::Result<BurstReport> {
    std::fs::create_dir_all(eval_dir)?;
    let (registry, _errors) = {
        let _guard = ctx.registry.lock().unwrap();
        super::load_external_registry(cfg, eval_dir)
    };
    // Built-in heuristics only; HTTP players stay on the CPU league (a 500ms
    // network round-trip per move would stall the whole lockstep).
    let heuristics: Vec<(u32, snek_heuristic::Baseline)> = registry
        .active
        .iter()
        .filter_map(|p| snek_heuristic::Baseline::parse(&p.spec).map(|b| (p.gen, b)))
        .collect();
    let net_pool = super::pool_members(&ctx.paths, cfg.league_entrant_gens.max(1));
    if net_pool.len() < 2 {
        anyhow::bail!("burst arena: pool has {} nets; need 2+", net_pool.len());
    }
    let n = cfg.num_snakes;
    let sims = if cfg.burst_sims > 0 {
        cfg.burst_sims
    } else {
        cfg.eval_sims
    }
    .max(1);
    let slots_total = cfg.gpu_batch_games.max(2);
    let target_games = cfg.burst_games.max(1);

    // ---- Snapshot history once: matchmaking runs lock-free in the workers ---
    let (ratings, career) = {
        let records = ctx.records.lock().unwrap();
        let mut career: HashMap<u32, u32> = HashMap::new();
        for r in records.iter() {
            for g in super::participants(r) {
                *career.entry(g).or_insert(0) += 1;
            }
        }
        let mut fit_pool = net_pool.clone();
        fit_pool.extend(heuristics.iter().map(|&(g, _)| g));
        (fit_ratings(&fit_pool, &records), career)
    };

    // ---- Reconcile stations: entrant joins, one under-covered net cycles ----
    let k_stations = STATIONS.min(net_pool.len());
    state.stations.retain(|g| net_pool.contains(g));
    state.stations.truncate(k_stations);
    let career_of = |g: u32| career.get(&g).copied().unwrap_or(0);
    let mut want: Vec<u32> = Vec::new();
    // The newest checkpoint (this burst's entrant) always plays.
    if let Some(&newest) = net_pool.last() {
        want.push(newest);
    }
    // The current top nets by Elo always defend: a burst's job is to test the
    // newest entrant against the best-so-far and to keep the leaders' ratings
    // sharp, so the champions must be loaded even once they're well-played
    // (career-based cycling alone would freeze them out). Two leaders plus the
    // entrant still leaves most stations for exploration below.
    let mut by_elo: Vec<u32> = net_pool.clone();
    by_elo.sort_by(|&a, &b| {
        ratings
            .get(&b)
            .unwrap_or(&0.0)
            .partial_cmp(ratings.get(&a).unwrap_or(&0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &leader in by_elo.iter().take(2) {
        if !want.contains(&leader) {
            want.push(leader);
        }
    }
    // The pool's least-played net cycles in, covering the pool across bursts.
    if let Some(&starved) = net_pool
        .iter()
        .filter(|g| !want.contains(g))
        .min_by_key(|&&g| career_of(g))
    {
        want.push(starved);
    }
    for g in want {
        if state.stations.contains(&g) {
            continue;
        }
        if state.stations.len() < k_stations {
            state.stations.push(g);
        } else if let Some(victim) = state
            .stations
            .iter()
            .copied()
            .max_by_key(|&s| career_of(s))
            .filter(|&v| v != g)
        {
            state.stations.retain(|&s| s != victim);
            state.stations.push(g);
        }
    }
    // Top up from the least-played remainder (first burst, or a small pool).
    while state.stations.len() < k_stations {
        let Some(&next) = net_pool
            .iter()
            .filter(|g| !state.stations.contains(g))
            .min_by_key(|&&g| career_of(g))
        else {
            break;
        };
        state.stations.push(next);
    }
    state.stations.sort_unstable();
    let stations = state.stations.clone();
    let station_of: HashMap<u32, usize> =
        stations.iter().enumerate().map(|(i, &g)| (g, i)).collect();

    // ---- Thaw carried games; abandon any the new station set can't seat ----
    state.slots.resize_with(slots_total, || None);
    state.slots.truncate(slots_total);
    let mut abandoned = 0usize;
    let mut carried = 0usize;
    for slot in state.slots.iter_mut() {
        let keep = slot.as_ref().is_some_and(|g| {
            g.board.width == cfg.board
                && g.board.snakes.len() == n
                && g.players.iter().all(|p| {
                    station_of.contains_key(p) || heuristics.iter().any(|&(hg, _)| hg == *p)
                })
        });
        if slot.is_some() && !keep {
            abandoned += 1;
            *slot = None;
        } else if slot.is_some() {
            carried += 1;
        }
    }
    metrics.log(format!(
        "arena burst: stations [{}], {carried} games carried, {abandoned} abandoned, target {target_games} @{sims} sims",
        stations
            .iter()
            .map(|g| format!("g{g}"))
            .collect::<Vec<_>>()
            .join(" "),
    ));

    // ---- Load station weights -----------------------------------------------
    let mut nets: Vec<LoadedNet> = Vec::with_capacity(stations.len());
    for &gen in &stations {
        let mut vs = nn::VarStore::new(device);
        let net = snek_tch::AZNet::new(
            &vs.root(),
            NUM_CHANNELS as i64,
            cfg.trunk_channels,
            cfg.trunk_blocks,
        );
        vs.load(ctx.paths.checkpoint_net(gen))?;
        vs.freeze();
        nets.push(LoadedNet { vs, net });
    }

    let counters = metrics.counters();
    counters
        .arena_target
        .store(target_games as u32, Ordering::Relaxed);
    counters.arena_done.store(0, Ordering::Relaxed);
    let started = Instant::now();
    let inf_before = counters.inferences.load(Ordering::Relaxed);

    // ---- Drive the slots: workers double-buffering the GPU ------------------
    let workers = std::env::var("SNEK_BURST_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&w| w >= 1)
        .unwrap_or(2)
        .min(slots_total);
    let shared = WorkerShared {
        gpu_lock: Mutex::new(()),
        completed: AtomicUsize::new(0),
        recorded: AtomicUsize::new(0),
        finished: Mutex::new(Vec::new()),
        planned: Mutex::new(HashMap::new()),
        steps: AtomicU64::new(0),
        completed_turns: AtomicU64::new(0),
        target: target_games,
        counters: &counters,
        stop,
    };
    let match_ctx = MatchCtx {
        stations: &stations,
        heuristics: &heuristics,
        ratings: &ratings,
        career: &career,
        num_snakes: n,
    };
    let chunk = slots_total.div_ceil(workers);
    let seed = ctx.next_seq.load(Ordering::Relaxed) ^ (started.elapsed().as_nanos() as u64);

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (w, slot_chunk) in state.slots.chunks_mut(chunk).enumerate() {
            let shared = &shared;
            let match_ctx = &match_ctx;
            let nets = &nets;
            handles.push(scope.spawn(move || {
                worker_loop(
                    slot_chunk,
                    nets,
                    match_ctx,
                    cfg,
                    sims,
                    LEAGUE_MAX_TURNS,
                    device,
                    shared,
                    seed ^ (w as u64) << 48,
                )
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });

    // ---- Commit --------------------------------------------------------------
    let mut finished = shared.finished.into_inner().unwrap();
    // Stable order: games are pushed concurrently by workers.
    finished.sort_by_key(|f| f.finished_order);
    let mut turns_total = 0u64;
    let mut new_records = Vec::with_capacity(finished.len());
    let mut recorded_games: Vec<(MatchRecord, Vec<u32>, Vec<FrameJson>)> = Vec::new();
    for f in finished {
        let seq = ctx.next_seq.fetch_add(1, Ordering::Relaxed);
        let record = MatchRecord {
            seq,
            placements: f.placements,
            turns: f.turns,
            sims: sims as u32,
            wall_seconds: f.wall_seconds,
            finished_unix_ms: chrono::Utc::now().timestamp_millis(),
        };
        turns_total += record.turns as u64;
        if let Some(frames) = f.frames {
            if recorded_games.len() < RECORDED_GAMES_PER_BURST {
                recorded_games.push((record.clone(), f.players, frames));
            }
        }
        new_records.push(record);
    }
    {
        let mut records = ctx.records.lock().unwrap();
        for record in &new_records {
            if let Err(err) = super::append_summary(&eval_dir.join("summary.jsonl"), record) {
                metrics.log(format!("burst arena: failed to append summary: {err}"));
            }
            records.push(record.clone());
        }
        records.sort_by_key(|r| r.seq);
        let mut fit_pool = net_pool.clone();
        fit_pool.extend(heuristics.iter().map(|&(g, _)| g));
        let fitted = fit_ratings(&fit_pool, &records);
        if let Err(err) =
            super::write_ratings(&eval_dir.join("ratings.json"), &records, &fitted, &registry)
        {
            metrics.log(format!("burst arena: failed to write ratings: {err}"));
        }
        for (record, players, frames) in &recorded_games {
            if let Err(err) = write_burst_game_file(eval_dir, record, players, &registry, frames) {
                metrics.log(format!("burst arena: failed to record game: {err}"));
            }
        }
        prune_game_files(eval_dir, KEEP_GAME_FILES);
    }
    counters.arena_target.store(0, Ordering::Relaxed);
    counters.arena_done.store(0, Ordering::Relaxed);

    Ok(BurstReport {
        games: new_records.len(),
        turns: turns_total,
        inferences: counters.inferences.load(Ordering::Relaxed) - inf_before,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// A finished burst game, before it gets a seq number.
struct FinishedGame {
    finished_order: usize,
    players: Vec<u32>,
    placements: Vec<Placement>,
    turns: u32,
    wall_seconds: f64,
    frames: Option<Vec<FrameJson>>,
}

/// Full-strength, GPU-batched round-robin among exactly `num_snakes`
/// checkpoints, driven by the *same* lockstep batched engine (`worker_loop`)
/// the training-cycle burst uses — every game seats all the given gens (with
/// rotation), so it's a controlled head-to-head, not the Elo league's
/// matchmaking. Prints each gen's win% + average rank + pairwise
/// later-vs-earlier record. Independent of the league rating system.
pub fn run_round_robin(
    checkpoint_dir: &Path,
    device: Device,
    metrics: &Metrics,
    cfg: &RunConfig,
    gens: &[u32],
    target_games: usize,
    sims: usize,
) -> anyhow::Result<()> {
    let n = cfg.num_snakes;
    anyhow::ensure!(
        gens.len() == n,
        "round-robin needs exactly num_snakes ({n}) gens, got {}",
        gens.len()
    );
    let stations: Vec<u32> = gens.to_vec();
    let mut nets: Vec<LoadedNet> = Vec::with_capacity(stations.len());
    for &gen in &stations {
        let path = checkpoint_dir.join(format!("net_{gen:04}.safetensors"));
        anyhow::ensure!(path.exists(), "missing checkpoint {}", path.display());
        let mut vs = nn::VarStore::new(device);
        let net = snek_tch::AZNet::new(
            &vs.root(),
            NUM_CHANNELS as i64,
            cfg.trunk_channels,
            cfg.trunk_blocks,
        );
        vs.load(&path)?;
        vs.freeze();
        nets.push(LoadedNet { vs, net });
    }
    let slots_total = cfg.gpu_batch_games.max(2);
    let counters = metrics.counters();
    counters.arena_target.store(target_games as u32, Ordering::Relaxed);
    counters.arena_done.store(0, Ordering::Relaxed);
    let stop = AtomicBool::new(false);
    let heuristics: Vec<(u32, snek_heuristic::Baseline)> = Vec::new();
    let ratings: HashMap<u32, f64> = HashMap::new();
    let career: HashMap<u32, u32> = HashMap::new();
    let shared = WorkerShared {
        gpu_lock: Mutex::new(()),
        completed: AtomicUsize::new(0),
        recorded: AtomicUsize::new(0),
        finished: Mutex::new(Vec::new()),
        planned: Mutex::new(HashMap::new()),
        steps: AtomicU64::new(0),
        completed_turns: AtomicU64::new(0),
        target: target_games,
        counters: &counters,
        stop: &stop,
    };
    let mc = MatchCtx {
        stations: &stations,
        heuristics: &heuristics,
        ratings: &ratings,
        career: &career,
        num_snakes: n,
    };
    let mut slots: Vec<Option<CarriedGame>> = (0..slots_total).map(|_| None).collect();
    let workers = std::env::var("SNEK_BURST_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&w| w >= 1)
        .unwrap_or(2)
        .min(slots_total);
    let chunk = slots_total.div_ceil(workers);
    let seed = 0xD00D_5EED_1234_9ABCu64;
    println!(
        "round-robin (GPU-batched): gens {:?} · target {} games · {} sims/move (league uses 64) · {} slots × {} workers",
        gens, target_games, sims, slots_total, workers
    );
    let started = Instant::now();
    let progress_done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        // Live progress: games done, rate, and ETA every 5s (worker_loop only
        // reports at the very end, so this is our only in-flight visibility).
        {
            let counters = &counters;
            let progress_done = &progress_done;
            let shared_ref = &shared;
            scope.spawn(move || {
                let (mut last, mut last_steps, mut lastt) = (0u32, 0u64, Instant::now());
                let (mut last_reqs, mut last_fus) = (0u64, 0u64);
                while !progress_done.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let d = counters.arena_done.load(Ordering::Relaxed);
                    let steps = shared_ref.steps.load(Ordering::Relaxed);
                    let done_turns = shared_ref.completed_turns.load(Ordering::Relaxed);
                    let avg_turn = steps.saturating_sub(done_turns) as f64 / slots_total.max(1) as f64;
                    let dt = lastt.elapsed().as_secs_f64().max(1e-9);
                    lastt = Instant::now();
                    let per_game_tps =
                        (steps.saturating_sub(last_steps)) as f64 / dt / slots_total.max(1) as f64;
                    // Forward-level benchmark: is a forward a big 512-row batch
                    // (intrinsic) or tiny (poor batching)? and how long does one
                    // take (graph fast-path ~tens of µs vs eager ~tens of ms)?
                    let reqs = counters.gpu_requests.load(Ordering::Relaxed);
                    let rows = counters.gpu_rows.load(Ordering::Relaxed);
                    let fus = counters.gpu_forward_us.load(Ordering::Relaxed);
                    let dreq = reqs.saturating_sub(last_reqs).max(1);
                    let rows_per_fwd = rows.wrapping_sub(0) as f64 / reqs.max(1) as f64;
                    let us_per_fwd = (fus.saturating_sub(last_fus)) as f64 / dreq as f64;
                    let dsteps = steps.saturating_sub(last_steps);
                    let lockstep_turns = dsteps as f64 / slots_total.max(1) as f64;
                    let fwd_per_turn = dreq as f64 / lockstep_turns.max(1e-9);
                    eprintln!(
                        "  {d}/{target_games} done · avg turn {avg_turn:.0} · {per_game_tps:.2} t/s/game · {:.0}s | BENCH: {rows_per_fwd:.0} rows/fwd · {us_per_fwd:.0}us/fwd · {fwd_per_turn:.0} fwd/lockstep-turn",
                        started.elapsed().as_secs_f64()
                    );
                    (last, last_steps, last_reqs, last_fus) = (d, steps, reqs, fus);
                }
            });
        }
        let mut handles = Vec::new();
        for (w, slot_chunk) in slots.chunks_mut(chunk).enumerate() {
            let shared = &shared;
            let mc = &mc;
            let nets = &nets;
            handles.push(scope.spawn(move || {
                worker_loop(
                    slot_chunk,
                    nets,
                    mc,
                    cfg,
                    sims,
                    0, // unlimited: run each game to a natural single-survivor terminal
                    device,
                    shared,
                    seed ^ ((w as u64) << 48),
                )
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        progress_done.store(true, Ordering::Relaxed);
    });
    // ---- Tally: per-gen win%/avg-rank + pairwise later-vs-earlier ----
    let finished = shared.finished.into_inner().unwrap();
    let idx_of: HashMap<u32, usize> = gens.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let (mut games, mut wins, mut rank_sum) = (vec![0u64; n], vec![0u64; n], vec![0u64; n]);
    let mut beats = vec![vec![0u64; n]; n];
    for f in &finished {
        let mut rank_of = vec![0u32; n];
        for p in &f.placements {
            if let Some(&i) = idx_of.get(&p.gen) {
                rank_of[i] = p.rank;
                games[i] += 1;
                rank_sum[i] += p.rank as u64;
                if p.rank == 1 {
                    wins[i] += 1;
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                if i != j && rank_of[i] > 0 && rank_of[i] < rank_of[j] {
                    beats[i][j] += 1;
                }
            }
        }
    }
    println!(
        "\ncompleted {} games in {:.0}s:\n  {:>6}   win%   avg_rank",
        finished.len(),
        started.elapsed().as_secs_f64(),
        "gen"
    );
    for (i, &g) in gens.iter().enumerate() {
        let gp = games[i].max(1) as f64;
        println!(
            "  {:>6}  {:>5.1}   {:.3}",
            g,
            100.0 * wins[i] as f64 / gp,
            rank_sum[i] as f64 / gp
        );
    }
    println!("\n  later-vs-earlier (later gen outranks earlier, % of decisive games):");
    for i in 0..n {
        for j in (i + 1)..n {
            let (later, earlier) = (beats[j][i], beats[i][j]);
            let dec = later + earlier;
            let pct = if dec > 0 {
                format!("{:.1}%", 100.0 * later as f64 / dec as f64)
            } else {
                "n/a".to_string()
            };
            println!(
                "    gen {:>5} vs gen {:>5}: {:>6}  ({} decisive)",
                gens[j], gens[i], pct, dec
            );
        }
    }
    Ok(())
}

/// Everything the workers share during one burst.
struct WorkerShared<'a> {
    gpu_lock: Mutex<()>,
    /// Games completed this burst; hitting `target` freezes the arena.
    completed: AtomicUsize,
    /// Games recorded with frames this burst (capped).
    recorded: AtomicUsize,
    finished: Mutex<Vec<FinishedGame>>,
    /// Games scheduled per player this burst, on top of `career` — spreads
    /// matchups instead of piling onto the newest entrant.
    planned: Mutex<HashMap<u32, u32>>,
    target: usize,
    counters: &'a Counters,
    stop: &'a AtomicBool,
    /// Total board-steps across all live slots (for a live turn-rate readout).
    steps: AtomicU64,
    /// Turns belonging to already-finished games, so in-flight average turn =
    /// (steps - completed_turns) / live_slots.
    completed_turns: AtomicU64,
}

/// Immutable matchmaking inputs, snapshotted at burst start.
struct MatchCtx<'a> {
    stations: &'a [u32],
    heuristics: &'a [(u32, snek_heuristic::Baseline)],
    ratings: &'a HashMap<u32, f64>,
    career: &'a HashMap<u32, u32>,
    num_snakes: usize,
}

/// How a seat gets its moves. Net seats own a fixed row group in their
/// station's batch block for the lifetime of the game.
enum SeatKind {
    Net { station: usize, group: usize },
    Heuristic(snek_heuristic::Baseline),
}

/// One live slot inside a worker: the persistent game core plus the
/// burst-local machinery (trees, row assignments, timers).
struct LiveSlot {
    game: CarriedGame,
    seats: Vec<SeatKind>,
    trees: Vec<Option<Tree>>,
    started: Instant,
}

/// A loaded station ready for batched forwards. The VarStore owns the
/// weights the AZNet references.
struct LoadedNet {
    #[allow(dead_code)]
    vs: nn::VarStore,
    net: snek_tch::AZNet,
}

// Safe: the weights are frozen and never mutated after loading, and every
// forward is serialized by the burst's GPU mutex (the same discipline as
// selfplay::gpu::Gpu).
unsafe impl Send for LoadedNet {}
unsafe impl Sync for LoadedNet {}

/// One worker: keep its slots saturated until the burst target is reached.
/// Each turn: dispatch heuristic seats onto rayon (they overlap the sim
/// loop), run `sims` lockstep tree simulations — the whole multi-station
/// step is one CUDA-graph replay — then step every board; finished games are
/// recorded and their slots restart with a fresh matchup immediately.
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    slots: &mut [Option<CarriedGame>],
    nets: &[LoadedNet],
    mc: &MatchCtx<'_>,
    cfg: &RunConfig,
    sims: usize,
    max_turns: u32,
    device: Device,
    shared: &WorkerShared<'_>,
    seed: u64,
) {
    let n = mc.num_snakes;
    let side = obs_side(cfg.board as usize);
    let obs_len = NUM_CHANNELS * side * side;
    let k = nets.len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x0DDB_1A5E_5BAD_5EEDu64);

    // Fixed station blocks: cap row groups per station, sized so any matchup
    // mix fits with slack. Row layout never changes, so the whole step
    // captures as one CUDA graph.
    let cap = (slots.len() * n).div_ceil(k.max(1)) + 2;
    let total_rows = k * cap * n;
    let mut free: Vec<Vec<usize>> = (0..k).map(|_| (0..cap).rev().collect()).collect();
    let mut obs_all = vec![0.0f32; total_rows * obs_len];
    let mut pol_all = vec![0.0f32; total_rows * 4];
    let mut val_all = vec![0.0f32; total_rows];
    // Heuristic seats stall the lockstep if the rayon pool can't absorb them
    // within one sim loop; only the first few slots may seat them.
    let heur_slots = (slots.len() / 16).max(1);

    // ---- Thaw carried games / start fresh ones into live slots -------------
    // Carried games claim their row groups first (they have history worth
    // keeping); fresh matchups then fit around whatever is left.
    let mut live: Vec<Option<LiveSlot>> = (0..slots.len()).map(|_| None).collect();
    for (idx, carried) in slots.iter_mut().enumerate() {
        let Some(game) = carried.take() else {
            continue;
        };
        let mut slot = LiveSlot {
            game,
            seats: Vec::new(),
            trees: Vec::new(),
            started: Instant::now(),
        };
        if rig_seats(&mut slot, mc, &mut free, cfg, sims, &mut rng) {
            live[idx] = Some(slot);
        }
        // else: dropped (capacity reshuffle, rare); the slot restarts fresh.
    }
    for (idx, slot) in live.iter_mut().enumerate() {
        if slot.is_some() {
            continue;
        }
        let mut game = new_game(mc, &free, idx < heur_slots, &shared.planned, &mut rng, cfg, idx);
        game.record =
            shared.recorded.fetch_add(1, Ordering::Relaxed) < RECORDED_GAMES_PER_BURST;
        if !game.record {
            shared.recorded.fetch_sub(1, Ordering::Relaxed);
        }
        {
            let mut planned = shared.planned.lock().unwrap();
            for &p in &game.players {
                *planned.entry(p).or_insert(0) += 1;
            }
        }
        let mut fresh = LiveSlot {
            game,
            seats: Vec::new(),
            trees: Vec::new(),
            started: Instant::now(),
        };
        let ok = rig_seats(&mut fresh, mc, &mut free, cfg, sims, &mut rng);
        debug_assert!(ok, "new_game only picks matchups the free lists can seat");
        *slot = Some(fresh);
    }
    let mut live: Vec<LiveSlot> = live.into_iter().map(|s| s.expect("every slot filled")).collect();

    // ---- Capture the whole multi-station step as ONE CUDA graph -------------
    let step = |input: &tch::Tensor| -> (tch::Tensor, tch::Tensor) {
        let mut probs_parts = Vec::with_capacity(k);
        let mut val_parts = Vec::with_capacity(k);
        for (s, loaded) in nets.iter().enumerate() {
            let slice = input.narrow(0, (s * cap * n) as i64, (cap * n) as i64);
            let (logits, value) = loaded.net.forward(&slice);
            probs_parts.push(logits.softmax(-1, tch::Kind::Float));
            val_parts.push(value);
        }
        (
            tch::Tensor::cat(&probs_parts, 0),
            tch::Tensor::cat(&val_parts, 0),
        )
    };
    let mut graph = if std::env::var("SNEK_CUDA_GRAPH").as_deref() != Ok("0") {
        let _g = shared.gpu_lock.lock().unwrap();
        Some(snek_tch::cudagraph::GraphedForward::capture_with(
            device,
            total_rows as i64,
            NUM_CHANNELS as i64,
            side as i64,
            side as i64,
            step,
        ))
    } else {
        None
    };

    let mut root_pol = vec![0.0f32; n * 4];
    let mut root_val = vec![0.0f32; n];
    let mut frame_pol = vec![0.0f32; n * 4];
    let mut frame_val = vec![0.0f32; n];
    let mut play_pols = vec![[0.0f32; 4]; n];
    let mut actions = vec![Move::Up; n];
    let (heur_tx, heur_rx) = mpsc::channel::<(usize, usize, usize)>();

    while shared.completed.load(Ordering::Relaxed) < shared.target
        && !shared.stop.load(Ordering::Relaxed)
    {
        // ---- Set up this turn: tree resets + heuristic dispatch -------------
        let mut heur_pending = 0usize;
        for (si, slot) in live.iter_mut().enumerate() {
            for seat in 0..n {
                if !slot.game.board.snakes[seat].alive() {
                    continue;
                }
                match slot.seats[seat] {
                    SeatKind::Net { .. } => {
                        if let Some(tree) = slot.trees[seat].as_mut() {
                            tree.reset(&slot.game.board);
                        }
                    }
                    SeatKind::Heuristic(kind) => {
                        let board = slot.game.board.clone();
                        let tx = heur_tx.clone();
                        let draw_value = cfg.draw_value;
                        heur_pending += 1;
                        rayon::spawn(move || {
                            let mv = std::panic::catch_unwind(|| {
                                let hcfg = snek_heuristic::HeuristicConfig {
                                    draw_value,
                                    ..Default::default()
                                };
                                let deadline =
                                    Instant::now() + std::time::Duration::from_secs(3600);
                                snek_heuristic::baseline_move_until(
                                    kind, &hcfg, &board, seat, deadline,
                                )
                                .move_index
                            })
                            .unwrap_or_else(|_| {
                                snek_heuristic::candidates(&board, seat)[0].index()
                            });
                            let _ = tx.send((si, seat, mv));
                        });
                    }
                }
            }
        }

        // ---- The sim loop ----------------------------------------------------
        for _ in 0..sims {
            if shared.stop.load(Ordering::Relaxed) {
                break;
            }
            let mut live_rows = 0u64;
            for slot in live.iter_mut() {
                for seat in 0..n {
                    let SeatKind::Net { station, group } = slot.seats[seat] else {
                        continue;
                    };
                    if !slot.game.board.snakes[seat].alive() {
                        continue; // its rows keep stale data; outputs unused
                    }
                    let base = ((station * cap + group) * n) * obs_len;
                    let tree = slot.trees[seat].as_mut().expect("net seat has tree");
                    tree.select();
                    match tree.pending_board() {
                        Some(b) => {
                            live_rows += n as u64;
                            for a in 0..n {
                                let off = base + a * obs_len;
                                encode_into(b, a, &mut obs_all[off..off + obs_len]);
                            }
                        }
                        None => {
                            for x in obs_all[base..base + n * obs_len].iter_mut() {
                                *x = 0.0;
                            }
                        }
                    }
                }
            }
            let dt = {
                let _g = shared.gpu_lock.lock().unwrap();
                let t0 = Instant::now();
                match graph.as_mut() {
                    Some(g) => g.run(&obs_all, &mut pol_all, &mut val_all),
                    None => forward_all(
                        nets, cap, device, &obs_all, total_rows, side, &mut pol_all,
                        &mut val_all,
                    ),
                }
                t0.elapsed().as_micros() as u64
            };
            let c = shared.counters;
            c.gpu_forward_us.fetch_add(dt, Ordering::Relaxed);
            c.gpu_requests.fetch_add(1, Ordering::Relaxed);
            c.gpu_rows.fetch_add(total_rows as u64, Ordering::Relaxed);
            c.inferences.fetch_add(live_rows, Ordering::Relaxed);
            for slot in live.iter_mut() {
                for seat in 0..n {
                    let SeatKind::Net { station, group } = slot.seats[seat] else {
                        continue;
                    };
                    if !slot.game.board.snakes[seat].alive() {
                        continue;
                    }
                    let row = (station * cap + group) * n;
                    let tree = slot.trees[seat].as_mut().expect("net seat has tree");
                    tree.expand_backup(&pol_all[row * 4..(row + n) * 4], &val_all[row..row + n]);
                }
            }
        }

        // ---- Collect heuristic answers (they ran during the sim loop) --------
        let mut heur_moves: HashMap<(usize, usize), usize> = HashMap::new();
        for _ in 0..heur_pending {
            let Ok((si, seat, mv)) = heur_rx.recv() else {
                break;
            };
            heur_moves.insert((si, seat), mv);
        }
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }

        // ---- Step every slot; finished games record and restart --------------
        for (si, slot) in live.iter_mut().enumerate() {
            for x in frame_pol.iter_mut() {
                *x = 0.0;
            }
            for x in frame_val.iter_mut() {
                *x = 0.0;
            }
            for seat in 0..n {
                actions[seat] = Move::Up;
                play_pols[seat] = [0.0; 4];
                if !slot.game.board.snakes[seat].alive() {
                    continue;
                }
                match slot.seats[seat] {
                    SeatKind::Net { .. } => {
                        let tree = slot.trees[seat].as_ref().expect("net seat has tree");
                        tree.root_targets(&mut root_pol, &mut root_val);
                        // Each seat plays from its OWN search; its view of the
                        // position comes from its own net.
                        let play = mask_obvious_immediate_deaths(
                            &slot.game.board,
                            seat,
                            &root_pol[seat * 4..seat * 4 + 4],
                        );
                        play_pols[seat].copy_from_slice(&play);
                        actions[seat] = argmax_move(&play);
                        frame_pol[seat * 4..seat * 4 + 4]
                            .copy_from_slice(&root_pol[seat * 4..seat * 4 + 4]);
                        frame_val[seat] = root_val[seat];
                    }
                    SeatKind::Heuristic(_) => {
                        let mv = heur_moves
                            .get(&(si, seat))
                            .copied()
                            .unwrap_or(Move::Up.index());
                        actions[seat] = Move::from_index(mv);
                        play_pols[seat][mv] = 1.0;
                        frame_pol[seat * 4 + mv] = 1.0;
                    }
                }
            }
            if slot.game.record {
                slot.game.frames.push(frame_from_board(
                    &slot.game.board,
                    n,
                    &frame_pol,
                    &frame_val,
                    &play_pols,
                    &actions,
                ));
            }
            slot.game.board.step_and_spawn(&actions, &mut rng);
            slot.game.turn += 1;
            shared.steps.fetch_add(1, Ordering::Relaxed);
            for seat in 0..n {
                if !slot.game.board.snakes[seat].alive() && slot.game.death_turn[seat].is_none() {
                    slot.game.death_turn[seat] = Some(slot.game.turn);
                }
            }

            // The game ends when at most one PLAYER has a living snake (the
            // elimination order is complete), or at the turn cap.
            let seat_player = |g: &CarriedGame, seat: usize| -> u32 {
                g.players[(seat + g.rotation) % g.players.len()]
            };
            let mut alive_players: Vec<u32> = slot
                .game
                .board
                .snakes
                .iter()
                .enumerate()
                .filter(|(_, s)| s.alive())
                .map(|(i, _)| seat_player(&slot.game, i))
                .collect();
            alive_players.sort_unstable();
            alive_players.dedup();
            if alive_players.len() > 1 && (max_turns == 0 || slot.game.turn < max_turns) {
                continue;
            }

            // Competition ranking by elimination order: outlasting beats.
            let death = &slot.game.death_turn;
            let score = |i: usize| death[i].map(|t| t as u64).unwrap_or(u64::MAX);
            let placements: Vec<Placement> = (0..n)
                .map(|seat| Placement {
                    gen: seat_player(&slot.game, seat),
                    seat: seat as u32,
                    rank: 1 + (0..n).filter(|&j| score(j) > score(seat)).count() as u32,
                    death_turn: death[seat],
                })
                .collect();
            let frames = if slot.game.record {
                slot.game.frames.push(frame_from_board(
                    &slot.game.board,
                    n,
                    &vec![0.0f32; n * 4],
                    &vec![0.0f32; n],
                    &vec![[0.0f32; 4]; n],
                    &vec![Move::Up; n],
                ));
                Some(std::mem::take(&mut slot.game.frames))
            } else {
                None
            };
            let order = shared.completed.fetch_add(1, Ordering::Relaxed);
            shared.counters.arena_done.fetch_add(1, Ordering::Relaxed);
            shared
                .completed_turns
                .fetch_add(slot.game.turn as u64, Ordering::Relaxed);
            shared.finished.lock().unwrap().push(FinishedGame {
                finished_order: order,
                players: slot.game.players.clone(),
                placements,
                turns: slot.game.turn,
                wall_seconds: slot.game.prior_seconds + slot.started.elapsed().as_secs_f64(),
                frames,
            });

            // Restart the slot immediately — the buffer never drains.
            for seat in 0..n {
                if let SeatKind::Net { station, group } = slot.seats[seat] {
                    free[station].push(group);
                }
            }
            slot.game = new_game(
                mc,
                &free,
                si < heur_slots,
                &shared.planned,
                &mut rng,
                cfg,
                si,
            );
            slot.game.record =
                shared.recorded.fetch_add(1, Ordering::Relaxed) < RECORDED_GAMES_PER_BURST;
            if !slot.game.record {
                shared.recorded.fetch_sub(1, Ordering::Relaxed);
            }
            {
                let mut planned = shared.planned.lock().unwrap();
                for &p in &slot.game.players {
                    *planned.entry(p).or_insert(0) += 1;
                }
            }
            slot.started = Instant::now();
            let ok = rig_seats(slot, mc, &mut free, cfg, sims, &mut rng);
            debug_assert!(ok, "fresh matchup must fit the groups its game freed");
        }
    }

    // ---- Freeze: hand the in-flight games back for the next burst -----------
    for (carried, slot) in slots.iter_mut().zip(live) {
        let mut game = slot.game;
        game.prior_seconds += slot.started.elapsed().as_secs_f64();
        *carried = Some(game);
    }
}

/// Pick a fresh cross-generation matchup from the stations with free batch
/// capacity: first player by rating uncertainty (career + already-planned
/// burst games), the rest by Bradley–Terry information against those already
/// picked (evenly matched pairs teach the fit most), scaled by uncertainty.
/// Heuristic-eligible slots roll the CPU league's 1-in-3 chance of seating a
/// built-in heuristic. The returned matchup is always seatable: seat counts
/// per player are checked against the stations' free lists, degrading to a
/// most-free / heuristic-filler pick under extreme capacity skew.
fn new_game(
    mc: &MatchCtx<'_>,
    free: &[Vec<usize>],
    allow_heur: bool,
    planned: &Mutex<HashMap<u32, u32>>,
    rng: &mut Xoshiro256PlusPlus,
    cfg: &RunConfig,
    slot_idx: usize,
) -> CarriedGame {
    let n = mc.num_snakes;
    let planned_snapshot: HashMap<u32, u32> = planned.lock().unwrap().clone();
    let elo = |g: u32| mc.ratings.get(&g).copied().unwrap_or(0.0);
    let games_of = |g: u32| {
        mc.career.get(&g).copied().unwrap_or(0) + planned_snapshot.get(&g).copied().unwrap_or(0)
    };
    let uncertainty = |g: u32| 1.0 / (1.0 + games_of(g) as f64);
    let info = |a: u32, b: u32| {
        let p = 1.0 / (1.0 + 10f64.powf((elo(b) - elo(a)) / 400.0));
        p * (1.0 - p)
    };
    let sample = |rng: &mut Xoshiro256PlusPlus, cands: &[u32], weights: &[f64]| -> u32 {
        let total: f64 = weights.iter().sum();
        let mut pick = rng.gen_range(0.0..total.max(f64::MIN_POSITIVE));
        for (i, w) in weights.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                return cands[i];
            }
        }
        *cands.last().expect("candidates non-empty")
    };
    let station_free = |g: u32| {
        mc.stations
            .iter()
            .position(|&s| s == g)
            .map(|k| free[k].len())
            .unwrap_or(usize::MAX) // heuristics need no capacity
    };
    // Seat count of players[i] under rotation r with m players and n seats.
    let seats_of = |i: usize, r: usize, m: usize| (0..n).filter(|s| (s + r) % m == i).count();

    let mut selected: Vec<u32> = Vec::with_capacity(n);
    if allow_heur && !mc.heuristics.is_empty() && rng.gen_bool(HEUR_PLAY_PROB) {
        selected.push(mc.heuristics[rng.gen_range(0..mc.heuristics.len())].0);
    }
    loop {
        let cands: Vec<u32> = mc
            .stations
            .iter()
            .enumerate()
            .filter(|&(k, _)| !free[k].is_empty())
            .map(|(_, &g)| g)
            .filter(|g| !selected.contains(g))
            .collect();
        if selected.len() >= n || cands.is_empty() {
            break;
        }
        let weights: Vec<f64> = cands
            .iter()
            .map(|&c| {
                if selected.is_empty() {
                    uncertainty(c)
                } else {
                    selected
                        .iter()
                        .map(|&s| info(c, s) * (uncertainty(c) + uncertainty(s)))
                        .sum::<f64>()
                        .max(f64::MIN_POSITIVE)
                }
            })
            .collect();
        selected.push(sample(rng, &cands, &weights));
    }
    // Degenerate capacity (all free groups piled in stations already picked,
    // or none): heuristics fill for free; they always keep m >= 2.
    while selected.len() < 2 {
        match mc.heuristics.iter().find(|&&(g, _)| !selected.contains(&g)) {
            Some(&(g, _)) => selected.push(g),
            None => break,
        }
    }
    let m = selected.len().max(1);
    let rotation = slot_idx.wrapping_add(rng.gen_range(0..m));
    // Feasibility: under this rotation a player may hold several seats (m < n
    // shares seats by rotation); every net player's station must have that
    // many free groups. Trim infeasible players (their seats fold onto the
    // rest) until it fits — total slack (2 groups/station) makes this rare.
    let mut players = selected;
    loop {
        let m = players.len();
        let feasible = players
            .iter()
            .enumerate()
            .all(|(i, &p)| station_free(p) >= seats_of(i, rotation % m, m));
        if feasible || m <= 1 {
            break;
        }
        // Drop the tightest net player; heuristics are never the constraint.
        let victim = players
            .iter()
            .enumerate()
            .filter(|&(_, &p)| station_free(p) != usize::MAX)
            .map(|(i, &p)| (i, station_free(p)))
            .min_by_key(|&(_, f)| f)
            .map(|(i, _)| i);
        match victim {
            Some(i) => {
                players.remove(i);
            }
            None => break,
        }
    }

    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(rng.gen());
    CarriedGame {
        rotation,
        players,
        board: standard_start(cfg.board, cfg.board, n, &mut rng2),
        turn: 0,
        death_turn: vec![None; n],
        frames: Vec::new(),
        record: false,
        prior_seconds: 0.0,
    }
}

/// Build the seat kinds and trees for a slot's current game, claiming one
/// row group per net seat from its station's free list. Returns false —
/// with everything it claimed released — if capacity doesn't allow it.
fn rig_seats(
    slot: &mut LiveSlot,
    mc: &MatchCtx<'_>,
    free: &mut [Vec<usize>],
    cfg: &RunConfig,
    sims: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> bool {
    let n = mc.num_snakes;
    let station_of: HashMap<u32, usize> = mc
        .stations
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, i))
        .collect();
    slot.seats.clear();
    slot.trees.clear();
    for seat in 0..n {
        let player = slot.game.players[(seat + slot.game.rotation) % slot.game.players.len()];
        match station_of.get(&player) {
            Some(&station) => {
                let Some(group) = free[station].pop() else {
                    // Roll back the groups this rig already claimed.
                    for kind in &slot.seats {
                        if let SeatKind::Net { station, group } = *kind {
                            free[station].push(group);
                        }
                    }
                    slot.seats.clear();
                    slot.trees.clear();
                    return false;
                };
                slot.seats.push(SeatKind::Net { station, group });
                slot.trees.push(Some(Tree::new(
                    n,
                    cfg.board,
                    cfg.board,
                    cfg.c_puct,
                    cfg.draw_value,
                    sims + 4,
                    0.0, // league play: no root exploration noise
                    cfg.dirichlet_alpha,
                    rng.gen(),
                    None, // arena heuristics are whole-agent seats, not in-tree models
                )));
            }
            None => {
                let baseline = mc
                    .heuristics
                    .iter()
                    .find(|&&(g, _)| g == player)
                    .map(|&(_, b)| b)
                    .unwrap_or(snek_heuristic::Baseline::Floodfill);
                slot.seats.push(SeatKind::Heuristic(baseline));
                slot.trees.push(None);
            }
        }
    }
    true
}

/// Non-graph fallback (SNEK_CUDA_GRAPH=0): one upload, per-station forwards
/// over fixed row blocks, one concatenated download.
#[allow(clippy::too_many_arguments)]
fn forward_all(
    nets: &[LoadedNet],
    cap: usize,
    device: Device,
    obs: &[f32],
    total_rows: usize,
    side: usize,
    pol: &mut [f32],
    val: &mut [f32],
) {
    let _ = cap;
    let n_rows_per_station = total_rows / nets.len().max(1);
    tch::no_grad(|| {
        let x = tch::Tensor::from_slice(obs)
            .reshape([
                total_rows as i64,
                NUM_CHANNELS as i64,
                side as i64,
                side as i64,
            ])
            .to_device(device);
        let mut probs_parts = Vec::with_capacity(nets.len());
        let mut val_parts = Vec::with_capacity(nets.len());
        for (k, loaded) in nets.iter().enumerate() {
            let slice = x.narrow(
                0,
                (k * n_rows_per_station) as i64,
                n_rows_per_station as i64,
            );
            let (logits, value) = loaded.net.forward(&slice);
            probs_parts.push(logits.softmax(-1, tch::Kind::Float));
            val_parts.push(value);
        }
        let probs = tch::Tensor::cat(&probs_parts, 0).to_device(Device::Cpu);
        let value = tch::Tensor::cat(&val_parts, 0).to_device(Device::Cpu);
        probs.copy_data(pol, total_rows * 4);
        value.copy_data(val, total_rows);
    });
}

/// Write one burst game's recording in the exact schema of the CPU league's
/// `game_SSSSSS.json`, so the eval viewer replays it unchanged.
fn write_burst_game_file(
    eval_dir: &Path,
    record: &MatchRecord,
    players: &[u32],
    registry: &ExternalRegistry,
    frames: &[FrameJson],
) -> anyhow::Result<()> {
    let winners: Vec<u32> = record
        .placements
        .iter()
        .filter(|p| p.rank == 1)
        .map(|p| p.seat)
        .collect();
    let file = serde_json::json!({
        "gen": players.iter().copied().filter(|&g| !is_external(g)).max().unwrap_or(0),
        "config": {
            "seq": record.seq,
            "players": players.iter().map(|&g| serde_json::json!({
                "gen": g,
                "name": registry.display(g),
            })).collect::<Vec<_>>(),
            "placements": record.placements,
            "sims": record.sims,
        },
        "games": [{
            "frames": frames,
            "winner": match winners.as_slice() {
                [only] => Some(*only as i32),
                _ => None,
            },
            "num_turns": record.turns,
        }],
    });
    let path = eval_dir.join(format!("game_{:06}.json", record.seq));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&file)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// A detached [`LeagueCtx`] for the standalone runner (`snek-train --burst`):
/// records seed from the run's canonical eval dir plus `out_dir` (so repeat
/// standalone bursts accumulate), but nothing is written to the canonical dir.
pub fn standalone_ctx(
    runs_dir: &Path,
    run_id: &str,
    out_dir: &Path,
    metrics: Metrics,
) -> anyhow::Result<(Arc<LeagueCtx>, RunConfig)> {
    let paths = crate::state::RunPaths::new(runs_dir, run_id);
    let cfg = RunConfig::load(&paths.config)?;
    let mut records = super::read_summaries(&paths.root);
    let canonical = paths.root.join("eval");
    if out_dir != canonical {
        records.extend(super::read_summary_file(&out_dir.join("summary.jsonl")));
    }
    records.sort_by_key(|m| m.seq);
    let next_seq = records.iter().map(|m| m.seq + 1).max().unwrap_or(0);
    let trainer = crate::trainer::TrainerHandle::new(runs_dir.to_path_buf(), metrics, cfg.clone());
    let ctx = Arc::new(LeagueCtx {
        paths,
        trainer,
        stop: Arc::new(AtomicBool::new(false)),
        records: Mutex::new(records),
        next_seq: AtomicU64::new(next_seq),
        registry: Mutex::new(()),
        logged_errors: Mutex::new(Default::default()),
        announced: AtomicBool::new(false),
    });
    Ok((ctx, cfg))
}

/// The run root a standalone burst writes into (`<runs>/<run>/<subdir>`).
pub fn standalone_out_dir(runs_dir: &Path, run_id: &str, subdir: &str) -> PathBuf {
    runs_dir.join(run_id).join(subdir)
}
