//! Continuous evaluation league. While a run is active, one arena game is
//! always in flight on pinned CPU cores: every checkpoint at a multiple of
//! `league_entrant_gens` generations joins the pool, the scheduler picks
//! `num_snakes` distinct nets per game (by expected information gain), plays
//! one game via the snek-server `arena` binary — every net controlling its own
//! snake — and refits ratings over *all* recorded results after every game.
//!
//! Ratings are the Plackett–Luce model: the multiplayer generalization of
//! Bradley–Terry (identical to it for two players), whose likelihood of a full
//! elimination order is a product of successive "who outlasts the rest"
//! choices. It is fitted by maximum likelihood with the standard MM algorithm
//! (Hunter 2004), ties averaged over their orderings, converted to the Elo
//! scale (400·log10) and anchored at the pool's earliest generation (Elo 0).
//!
//! Files in `runs/<id>/eval/`:
//!
//!   summary.jsonl        one line per game: the full placement ranking
//!   ratings.json         latest Plackett–Luce fit for every rated net
//!   match_SSSSSS.json    the game's frames + search readout, in the exact
//!                        schema of games/gen_NNNN.json
//!   arena_SSSSSS.log     that game's arena stderr
//!
//! Old match recordings are pruned (newest `KEEP_MATCH_FILES` kept); the
//! summary log and ratings are kept forever. The league stops (killing any
//! in-flight game) when the run pauses, and resumes with it.

use crate::config::RunConfig;
use crate::state::RunPaths;
use crate::trainer::TrainerHandle;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Recorded match game files kept on disk (newest first); logs likewise.
const KEEP_MATCH_FILES: usize = 60;

/// One league thread per process; a second run start waits for the old one.
static LEAGUE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// One line of `runs/<id>/eval/summary.jsonl`: one game's full placement
/// ranking. Legacy pre-multiplayer lines (pairwise wins/losses/draws) are
/// still read and converted into rankings for the fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchRecord {
    #[serde(default)]
    pub seq: u64,
    /// One entry per seat; rank 1 is best, ties share a rank.
    #[serde(default)]
    pub placements: Vec<Placement>,
    #[serde(default)]
    pub turns: u32,
    #[serde(default)]
    pub sims: u32,
    #[serde(default)]
    pub wall_seconds: f64,
    #[serde(default)]
    pub finished_unix_ms: i64,
    // Legacy v1 pairwise fields (read-only).
    #[serde(default, skip_serializing)]
    gen: Option<u32>,
    #[serde(default, skip_serializing)]
    opponent_gen: Option<u32>,
    #[serde(default, skip_serializing)]
    wins: u32,
    #[serde(default, skip_serializing)]
    losses: u32,
    #[serde(default, skip_serializing)]
    draws: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Placement {
    pub gen: u32,
    pub seat: u32,
    pub rank: u32,
    #[serde(default)]
    pub death_turn: Option<u32>,
}

/// `runs/<id>/eval/ratings.json`: the latest Plackett–Luce fit.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LeagueRatings {
    pub updated_unix_ms: i64,
    /// Every net with at least one recorded game, ascending by generation.
    pub ratings: Vec<LeagueRating>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeagueRating {
    pub gen: u32,
    /// Plackett–Luce strength on the Elo scale, anchored so the earliest rated
    /// generation is 0.
    pub elo: f64,
    pub games: u32,
    /// Rank-1 finishes (shared firsts count).
    pub wins: u32,
    pub avg_rank: f64,
}

/// An event line the arena prints to stdout: "turn" (per-turn progress, with
/// --events) or "game" (a finished game with its placements).
#[derive(Deserialize)]
struct GameEvent {
    event: String,
    #[serde(default)]
    index: u32,
    #[serde(default)]
    turn: u32,
    #[serde(default)]
    turns: u32,
    #[serde(default)]
    placements: Vec<ArenaPlacement>,
    /// The turn's board frame (turn events only): same JSON shape as one frame
    /// of games/gen_NNNN.json, passed through verbatim to the live view.
    #[serde(default)]
    frame: Option<serde_json::Value>,
}

/// A placement as the arena reports it: `net` indexes the --nets list.
#[derive(Deserialize)]
struct ArenaPlacement {
    seat: u32,
    net: u32,
    rank: u32,
    #[serde(default)]
    death_turn: Option<u32>,
}

/// Real-time state of the in-flight league game, published to the frontend
/// via the `/api/stream/eval` SSE endpoint. `active: false` between games.
#[derive(Debug, Clone, Serialize)]
pub struct LiveEval {
    pub active: bool,
    pub seq: u64,
    /// The players of this game, in --nets order (seat s plays player s % N).
    pub players: Vec<LivePlayer>,
    /// In-flight games (index within the match, current turn), ascending.
    pub games: Vec<LiveGame>,
    pub updated_unix_ms: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LivePlayer {
    pub gen: u32,
    /// Fitted league Elo entering the game.
    pub elo: f64,
    /// Career league games.
    pub games: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveGame {
    pub index: u32,
    pub turn: u32,
    /// Latest board frame (same shape as one frame of games/gen_NNNN.json).
    pub frame: Option<serde_json::Value>,
}

static LIVE: Mutex<LiveEval> = Mutex::new(LiveEval {
    active: false,
    seq: 0,
    players: Vec::new(),
    games: Vec::new(),
    updated_unix_ms: 0,
});

/// Snapshot of the in-flight game state (for the SSE endpoint).
pub fn live() -> LiveEval {
    LIVE.lock().unwrap().clone()
}

fn set_live(update: impl FnOnce(&mut LiveEval)) {
    let mut live = LIVE.lock().unwrap();
    update(&mut live);
    live.updated_unix_ms = chrono::Utc::now().timestamp_millis();
}

/// Start the league thread for the active run. Returns immediately; the thread
/// exits when `stop` is set (killing any in-flight game).
pub fn start_league(paths: RunPaths, trainer: TrainerHandle, stop: Arc<AtomicBool>) {
    std::thread::Builder::new()
        .name("eval-league".into())
        .spawn(move || {
            // Wait for a previous run's league (same process) to wind down.
            while LEAGUE_ACTIVE.swap(true, Ordering::SeqCst) {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            league_loop(&paths, &trainer, &stop);
            LEAGUE_ACTIVE.store(false, Ordering::SeqCst);
        })
        .expect("spawn eval league thread");
}

fn league_loop(paths: &RunPaths, trainer: &TrainerHandle, stop: &AtomicBool) {
    let metrics = trainer.metrics();
    let eval_dir = paths.root.join("eval");
    let bin = arena_bin();
    if !bin.exists() {
        metrics.log(format!(
            "eval league: arena binary not found at {} (build with `make arena-build` or set SNEK_ARENA_BIN); league disabled for this run",
            bin.display()
        ));
        return;
    }
    let mut records = read_summaries(&paths.root);
    let mut seq = records.iter().map(|m| m.seq + 1).max().unwrap_or(0);
    let mut announced = false;

    while !stop.load(Ordering::Relaxed) {
        let cfg = trainer.config();
        if cfg.league_entrant_gens == 0 {
            sleep_unless_stopped(stop, 30);
            continue;
        }
        let pool = pool_members(paths, cfg.league_entrant_gens);
        if pool.len() < 2 {
            sleep_unless_stopped(stop, 15);
            continue;
        }
        if !announced {
            metrics.log(format!(
                "eval league: running — pool of {} nets (entrant every {} gens), {} distinct nets per game, {} sims",
                pool.len(),
                cfg.league_entrant_gens,
                cfg.num_snakes.min(pool.len()),
                cfg.eval_sims,
            ));
            announced = true;
        }
        if std::fs::create_dir_all(&eval_dir).is_err() {
            sleep_unless_stopped(stop, 30);
            continue;
        }

        let ratings = fit_ratings(&pool, &records);
        let players = pick_players(&pool, &records, &ratings, cfg.num_snakes, seq);
        let career = |gen: u32| {
            records
                .iter()
                .flat_map(participants)
                .filter(|&g| g == gen)
                .count() as u32
        };
        let live_players: Vec<LivePlayer> = players
            .iter()
            .map(|&gen| LivePlayer {
                gen,
                elo: ratings.get(&gen).copied().unwrap_or(0.0),
                games: career(gen),
            })
            .collect();
        match run_match(
            &bin,
            paths,
            &eval_dir,
            &cfg,
            &players,
            live_players,
            seq,
            stop,
        ) {
            MatchResult::Done(record) => {
                let mut order = record.placements.clone();
                order.sort_by_key(|p| p.rank);
                metrics.log(format!(
                    "eval league #{seq}: {ranking} ({turns} turns) — {rated} nets rated",
                    ranking = order
                        .iter()
                        .map(|p| format!("gen_{:04}", p.gen))
                        .collect::<Vec<_>>()
                        .join(" > "),
                    turns = record.turns,
                    rated = pool.len(),
                ));
                if let Err(err) = append_summary(&eval_dir.join("summary.jsonl"), &record) {
                    metrics.log(format!("eval league: failed to append summary: {err}"));
                }
                records.push(record);
                let fitted = fit_ratings(&pool, &records);
                if let Err(err) = write_ratings(&eval_dir.join("ratings.json"), &records, &fitted) {
                    metrics.log(format!("eval league: failed to write ratings: {err}"));
                }
                prune_match_files(&eval_dir, KEEP_MATCH_FILES);
                seq += 1;
            }
            MatchResult::Stopped => break,
            MatchResult::Failed(err) => {
                metrics.log(format!("eval league #{seq}: game failed: {err}"));
                seq += 1;
                sleep_unless_stopped(stop, 30);
            }
        }
    }
    metrics.log("eval league: stopped".to_string());
}

enum MatchResult {
    Done(MatchRecord),
    Stopped,
    Failed(String),
}

/// Play one game (blocking): spawn the arena with one seat per selected net,
/// stream its stdout events into the live view, and build the placement record
/// from the game event. `stop` kills the arena promptly on a pause.
#[allow(clippy::too_many_arguments)]
fn run_match(
    bin: &Path,
    paths: &RunPaths,
    eval_dir: &Path,
    cfg: &RunConfig,
    players: &[u32],
    live_players: Vec<LivePlayer>,
    seq: u64,
    stop: &AtomicBool,
) -> MatchResult {
    let start = std::time::Instant::now();
    let record_path = eval_dir.join(format!("match_{seq:06}.json"));
    let log_path = eval_dir.join(format!("arena_{seq:06}.log"));
    let nets: Vec<String> = players
        .iter()
        .map(|&g| paths.checkpoint_net(g).display().to_string())
        .collect();
    let names: Vec<String> = players.iter().map(|&g| format!("gen_{g:04}")).collect();
    let cores = league_cores(players.len(), cfg.eval_cores);

    let mut command = std::process::Command::new(bin);
    command
        .args(["--nets", &nets.join(",")])
        .args(["--names", &names.join(",")])
        .args(["--games", "1"])
        .args(["--seats", &cfg.num_snakes.to_string()])
        .args(["--sims", &cfg.eval_sims.to_string()])
        .args(["--board", &cfg.board.to_string()])
        .args(["--cores", &cores])
        .args(["--cores-per-net", &cfg.eval_cores.to_string()])
        .args(["--seed", &seq.wrapping_mul(1_000_003).to_string()])
        .args([
            "--record-gen",
            &players.iter().copied().max().unwrap_or(0).to_string(),
        ])
        .arg("--record")
        .arg(&record_path)
        .arg("--events")
        .stdout(std::process::Stdio::piped());
    match std::fs::File::create(&log_path) {
        Ok(log) => {
            command.stderr(log);
        }
        Err(_) => {
            command.stderr(std::process::Stdio::null());
        }
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => return MatchResult::Failed(format!("spawn: {err}")),
    };
    // Publish the in-flight game to the live SSE view, and guarantee the
    // "active" flag drops on every exit path.
    set_live(|l| {
        *l = LiveEval {
            active: true,
            seq,
            players: live_players,
            games: Vec::new(),
            updated_unix_ms: 0,
        };
    });
    struct LiveClear;
    impl Drop for LiveClear {
        fn drop(&mut self) {
            set_live(|l| {
                l.active = false;
                l.games.clear();
            });
        }
    }
    let _live_clear = LiveClear;

    // Events arrive on the child's stdout; a reader thread forwards them so
    // the wait loop below can consume without blocking.
    let (event_tx, event_rx) = std::sync::mpsc::channel::<GameEvent>();
    let reader = child.stdout.take().map(|out| {
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(out).lines().map_while(Result::ok) {
                if let Ok(ev) = serde_json::from_str::<GameEvent>(&line) {
                    if event_tx.send(ev).is_err() {
                        break;
                    }
                }
            }
        })
    });
    let mut result: Option<MatchRecord> = None;
    let apply = |ev: GameEvent, result: &mut Option<MatchRecord>| match ev.event.as_str() {
        "turn" => set_live(|l| {
            match l.games.iter_mut().find(|g| g.index == ev.index) {
                Some(game) => {
                    game.turn = ev.turn;
                    game.frame = ev.frame;
                }
                None => l.games.push(LiveGame {
                    index: ev.index,
                    turn: ev.turn,
                    frame: ev.frame,
                }),
            }
            l.games.sort_by_key(|g| g.index);
        }),
        "game" => {
            let placements = ev
                .placements
                .iter()
                .map(|p| Placement {
                    gen: players.get(p.net as usize).copied().unwrap_or(0),
                    seat: p.seat,
                    rank: p.rank,
                    death_turn: p.death_turn,
                })
                .collect();
            *result = Some(MatchRecord {
                seq,
                placements,
                turns: ev.turns,
                sims: cfg.eval_sims as u32,
                wall_seconds: 0.0,
                finished_unix_ms: 0,
                gen: None,
                opponent_gen: None,
                wins: 0,
                losses: 0,
                draws: 0,
            });
            set_live(|l| l.games.retain(|g| g.index != ev.index));
        }
        _ => {}
    };
    let status = loop {
        if stop.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            if let Some(handle) = reader {
                let _ = handle.join();
            }
            return MatchResult::Stopped;
        }
        while let Ok(ev) = event_rx.try_recv() {
            apply(ev, &mut result);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(500)),
            Err(err) => return MatchResult::Failed(format!("wait: {err}")),
        }
    };
    if let Some(handle) = reader {
        let _ = handle.join();
    }
    while let Ok(ev) = event_rx.try_recv() {
        apply(ev, &mut result);
    }
    if !status.success() {
        return MatchResult::Failed(format!(
            "arena exited with {status} (see {})",
            log_path.display()
        ));
    }
    match result {
        Some(mut record) => {
            record.wall_seconds = start.elapsed().as_secs_f64();
            record.finished_unix_ms = chrono::Utc::now().timestamp_millis();
            MatchResult::Done(record)
        }
        None => MatchResult::Failed("arena exited without reporting a game".into()),
    }
}

/// Checkpoints eligible for the pool: every archived generation at a multiple
/// of `entrant_gens` (gen 0 included — it anchors the Elo scale). Ascending.
fn pool_members(paths: &RunPaths, entrant_gens: usize) -> Vec<u32> {
    let mut gens: Vec<u32> = match std::fs::read_dir(&paths.checkpoints) {
        Ok(rd) => rd
            .flatten()
            .filter_map(|e| {
                e.file_name()
                    .to_str()?
                    .strip_prefix("net_")?
                    .strip_suffix(".safetensors")?
                    .parse()
                    .ok()
            })
            .filter(|g: &u32| (*g as usize).is_multiple_of(entrant_gens))
            .collect(),
        Err(_) => Vec::new(),
    };
    gens.sort_unstable();
    gens.dedup();
    gens
}

/// The distinct gens a record involves.
fn participants(record: &MatchRecord) -> Vec<u32> {
    let mut gens: Vec<u32> = if record.placements.is_empty() {
        record.gen.into_iter().chain(record.opponent_gen).collect()
    } else {
        record.placements.iter().map(|p| p.gen).collect()
    };
    gens.sort_unstable();
    gens.dedup();
    gens
}

/// Select the players for the next game by expected information gain. Under
/// Plackett–Luce, the information a game carries about a pair is p(1−p) — the
/// Bradley–Terry Fisher information, maximal for evenly matched players — so
/// strong nets naturally meet; scaling by rating uncertainty (∝ 1/(1+games))
/// pulls newly admitted and under-played nets in. The first player is sampled
/// by uncertainty, each further seat by its summed pairwise weight against
/// those already selected. Sampling keeps occasional cross-table games
/// flowing, so the match graph stays connected and upsets can be detected.
fn pick_players(
    pool: &[u32],
    records: &[MatchRecord],
    ratings: &HashMap<u32, f64>,
    seats: usize,
    seq: u64,
) -> Vec<u32> {
    let mut games: HashMap<u32, u32> = pool.iter().map(|&g| (g, 0)).collect();
    for record in records {
        for g in participants(record) {
            if let Some(count) = games.get_mut(&g) {
                *count += 1;
            }
        }
    }
    let elo = |g: u32| ratings.get(&g).copied().unwrap_or(0.0);
    let uncertainty = |g: u32| 1.0 / (1.0 + games[&g] as f64);
    let info = |a: u32, b: u32| {
        let p = 1.0 / (1.0 + 10f64.powf((elo(b) - elo(a)) / 400.0));
        p * (1.0 - p)
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
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

    let k = seats.min(pool.len()).max(2);
    let mut selected = Vec::with_capacity(k);
    let weights: Vec<f64> = pool.iter().map(|&g| uncertainty(g)).collect();
    selected.push(sample(&mut rng, pool, &weights));
    while selected.len() < k {
        let cands: Vec<u32> = pool
            .iter()
            .copied()
            .filter(|g| !selected.contains(g))
            .collect();
        let weights: Vec<f64> = cands
            .iter()
            .map(|&c| {
                selected
                    .iter()
                    .map(|&s| info(c, s) * (uncertainty(c) + uncertainty(s)))
                    .sum::<f64>()
            })
            .collect();
        selected.push(sample(&mut rng, &cands, &weights));
    }
    // Rotate the seat order by seq so no player systematically holds seat 0.
    let rot = (seq % k as u64) as usize;
    selected.rotate_left(rot);
    selected
}

/// A weighted strict ordering of gens, best first.
type Ranking = (f64, Vec<u32>);

/// Expand a record into weighted strict rankings. Placement ties are averaged
/// over the orderings of each tied group (weight 1/k! each); legacy pairwise
/// lines expand to win/loss orderings with draws as half each way.
fn rankings_of(record: &MatchRecord) -> Vec<Ranking> {
    if !record.placements.is_empty() {
        // Group gens by rank (ascending); each group is a tied set.
        let mut rank_groups: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
        for p in &record.placements {
            rank_groups.entry(p.rank).or_default().push(p.gen);
        }
        let groups: Vec<Vec<u32>> = rank_groups.into_values().collect();
        let mut out: Vec<Ranking> = vec![(1.0, Vec::new())];
        for group in groups {
            let perms = permutations_capped(&group);
            let w = 1.0 / perms.len() as f64;
            out = out
                .into_iter()
                .flat_map(|(rw, prefix)| {
                    perms.iter().map(move |perm| {
                        let mut order = prefix.clone();
                        order.extend_from_slice(perm);
                        (rw * w, order)
                    })
                })
                .collect();
        }
        out
    } else if let (Some(a), Some(b)) = (record.gen, record.opponent_gen) {
        let mut out = Vec::new();
        if record.wins > 0 {
            out.push((record.wins as f64, vec![a, b]));
        }
        if record.losses > 0 {
            out.push((record.losses as f64, vec![b, a]));
        }
        if record.draws > 0 {
            out.push((record.draws as f64 * 0.5, vec![a, b]));
            out.push((record.draws as f64 * 0.5, vec![b, a]));
        }
        out
    } else {
        Vec::new()
    }
}

/// All orderings of a tied group, capped: groups larger than 4 fall back to
/// cyclic rotations (still symmetric across members, bounded cost).
fn permutations_capped(group: &[u32]) -> Vec<Vec<u32>> {
    if group.len() <= 1 {
        return vec![group.to_vec()];
    }
    if group.len() > 4 {
        return (0..group.len())
            .map(|r| {
                let mut v = group.to_vec();
                v.rotate_left(r);
                v
            })
            .collect();
    }
    let mut out = Vec::new();
    let mut items = group.to_vec();
    permute(&mut items, 0, &mut out);
    out
}

fn permute(items: &mut Vec<u32>, k: usize, out: &mut Vec<Vec<u32>>) {
    if k == items.len() {
        out.push(items.clone());
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, out);
        items.swap(k, i);
    }
}

/// Plackett–Luce maximum-likelihood ratings over every recorded game, via the
/// standard MM iteration (Hunter 2004); exactly Bradley–Terry for two-player
/// rankings. A half virtual tie between adjacent pool members keeps the graph
/// connected and every rating finite. Anchored so the earliest rated
/// generation sits at Elo 0.
fn fit_ratings(pool: &[u32], records: &[MatchRecord]) -> HashMap<u32, f64> {
    let mut rankings: Vec<Ranking> = records.iter().flat_map(rankings_of).collect();
    for w in pool.windows(2) {
        rankings.push((0.25, vec![w[0], w[1]]));
        rankings.push((0.25, vec![w[1], w[0]]));
    }
    let mut gens: Vec<u32> = pool.to_vec();
    gens.extend(rankings.iter().flat_map(|(_, r)| r.iter().copied()));
    gens.sort_unstable();
    gens.dedup();
    if gens.len() < 2 {
        return gens.into_iter().map(|g| (g, 0.0)).collect();
    }
    let index: HashMap<u32, usize> = gens.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let n = gens.len();
    let mut p = vec![1.0f64; n];
    for _ in 0..200 {
        let mut wins = vec![0.0f64; n];
        let mut denom = vec![0.0f64; n];
        for (w, order) in &rankings {
            let mut remaining: f64 = order.iter().map(|g| p[index[g]]).sum();
            for k in 0..order.len().saturating_sub(1) {
                let winner = index[&order[k]];
                wins[winner] += w;
                for g in &order[k..] {
                    denom[index[g]] += w / remaining;
                }
                remaining -= p[winner];
            }
        }
        for i in 0..n {
            if wins[i] > 0.0 && denom[i] > 0.0 {
                p[i] = wins[i] / denom[i];
            }
        }
        // Renormalize each round so the iteration stays well-conditioned.
        let mean = p.iter().sum::<f64>() / n as f64;
        for v in &mut p {
            *v /= mean;
        }
    }
    let anchor = p[0];
    gens.iter()
        .enumerate()
        .map(|(i, &g)| (g, 400.0 * (p[i] / anchor).log10()))
        .collect()
}

fn write_ratings(
    path: &Path,
    records: &[MatchRecord],
    fitted: &HashMap<u32, f64>,
) -> anyhow::Result<()> {
    // games, rank-1 finishes, rank sum per gen.
    let mut tally: HashMap<u32, (u32, u32, u64)> = HashMap::new();
    for record in records {
        if record.placements.is_empty() {
            if let (Some(a), Some(b)) = (record.gen, record.opponent_gen) {
                let games = record.wins + record.losses + record.draws;
                let ta = tally.entry(a).or_default();
                ta.0 += games;
                ta.1 += record.wins;
                ta.2 += (record.wins + record.losses * 2) as u64 + (record.draws as u64 * 3) / 2;
                let tb = tally.entry(b).or_default();
                tb.0 += games;
                tb.1 += record.losses;
                tb.2 += (record.losses + record.wins * 2) as u64 + (record.draws as u64 * 3) / 2;
            }
            continue;
        }
        for p in &record.placements {
            let t = tally.entry(p.gen).or_default();
            t.0 += 1;
            t.1 += (p.rank == 1) as u32;
            t.2 += p.rank as u64;
        }
    }
    let mut ratings: Vec<LeagueRating> = fitted
        .iter()
        .map(|(&gen, &elo)| {
            let (games, wins, rank_sum) = tally.get(&gen).copied().unwrap_or_default();
            LeagueRating {
                gen,
                elo,
                games,
                wins,
                avg_rank: if games > 0 {
                    rank_sum as f64 / games as f64
                } else {
                    0.0
                },
            }
        })
        .collect();
    ratings.sort_by_key(|r| r.gen);
    let doc = LeagueRatings {
        updated_unix_ms: chrono::Utc::now().timestamp_millis(),
        ratings,
    };
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&doc)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Keep only the newest `keep` recorded matches on disk (games + logs). The
/// summary log and ratings are never pruned.
fn prune_match_files(eval_dir: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(eval_dir) else {
        return;
    };
    let mut matches: Vec<(u64, PathBuf)> = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let seq = ["match_", "arena_", "out_"]
            .iter()
            .find_map(|p| name.strip_prefix(p))
            .and_then(|rest| rest.split(['_', '.']).next())
            .and_then(|s| s.parse::<u64>().ok());
        if let Some(seq) = seq {
            matches.push((seq, path));
        }
    }
    let mut seqs: Vec<u64> = matches.iter().map(|(s, _)| *s).collect();
    seqs.sort_unstable();
    seqs.dedup();
    if seqs.len() <= keep {
        return;
    }
    let cutoff = seqs[seqs.len() - keep];
    for (seq, path) in matches {
        if seq < cutoff {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn arena_bin() -> PathBuf {
    std::env::var("SNEK_ARENA_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("crates/snek-server/target/release/arena"))
}

/// Pin the league to the machine's highest-numbered cores (self-play's rayon
/// pool is unpinned, so the scheduler migrates it away): `players × per_net`
/// cores counted down from the top, as a flat spec the arena splits per net.
fn league_cores(players: usize, per_net: usize) -> String {
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let want = (players * per_net.max(1))
        .min(total.saturating_sub(1))
        .max(players);
    let lo = total.saturating_sub(want);
    (lo..total)
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn sleep_unless_stopped(stop: &AtomicBool, secs: u64) {
    for _ in 0..secs {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn append_summary(path: &Path, record: &MatchRecord) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string(record)?)?;
    Ok(())
}

/// Read a run's match history (oldest first). Missing file → empty.
pub fn read_summaries(run_root: &Path) -> Vec<MatchRecord> {
    let path = run_root.join("eval").join("summary.jsonl");
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Read a run's latest league ratings. Missing file → empty.
pub fn read_ratings(run_root: &Path) -> LeagueRatings {
    let path = run_root.join("eval").join("ratings.json");
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(seq: u64, order: &[u32]) -> MatchRecord {
        MatchRecord {
            seq,
            placements: order
                .iter()
                .enumerate()
                .map(|(i, &gen)| Placement {
                    gen,
                    seat: i as u32,
                    rank: i as u32 + 1,
                    death_turn: None,
                })
                .collect(),
            turns: 100,
            sims: 0,
            wall_seconds: 0.0,
            finished_unix_ms: 0,
            gen: None,
            opponent_gen: None,
            wins: 0,
            losses: 0,
            draws: 0,
        }
    }

    fn pairwise(
        seq: u64,
        gen: u32,
        opponent_gen: u32,
        wins: u32,
        losses: u32,
        draws: u32,
    ) -> MatchRecord {
        MatchRecord {
            seq,
            placements: Vec::new(),
            turns: 0,
            sims: 0,
            wall_seconds: 0.0,
            finished_unix_ms: 0,
            gen: Some(gen),
            opponent_gen: Some(opponent_gen),
            wins,
            losses,
            draws,
        }
    }

    #[test]
    fn fit_matches_bradley_terry_for_two_players() {
        let pool = vec![0, 5];
        let records = vec![pairwise(0, 5, 0, 8, 2, 0)];
        let elo = fit_ratings(&pool, &records);
        assert_eq!(elo[&0], 0.0);
        assert!(elo[&5] > 100.0 && elo[&5] < 400.0, "{elo:?}");
    }

    #[test]
    fn fit_orders_multiplayer_rankings_transitively() {
        let pool = vec![0, 5, 10, 15];
        // Consistent finishing order across several 4-player games.
        let records: Vec<MatchRecord> = (0..6).map(|s| game(s, &[15, 10, 5, 0])).collect();
        let elo = fit_ratings(&pool, &records);
        assert_eq!(elo[&0], 0.0);
        assert!(elo[&5] > elo[&0], "{elo:?}");
        assert!(elo[&10] > elo[&5], "{elo:?}");
        assert!(elo[&15] > elo[&10], "{elo:?}");
    }

    #[test]
    fn fit_stays_finite_with_ties_and_perfect_records() {
        let pool = vec![0, 5, 10];
        let mut tied = game(0, &[10, 5, 0]);
        // Make 5 and 0 tie for second.
        tied.placements[2].rank = 2;
        let records = vec![tied, game(1, &[10, 5, 0])];
        let elo = fit_ratings(&pool, &records);
        for g in &pool {
            assert!(elo[g].is_finite() && elo[g].abs() < 2000.0, "{elo:?}");
        }
    }

    #[test]
    fn picks_distinct_players_and_prioritizes_the_unrated() {
        let pool = vec![0, 5, 10, 15, 20];
        // gen 20 has never played; the rest share several games.
        let records: Vec<MatchRecord> = (0..5).map(|s| game(s, &[15, 10, 5, 0])).collect();
        let ratings = fit_ratings(&pool, &records);
        let mut with_new = 0;
        for seq in 0..20 {
            let players = pick_players(&pool, &records, &ratings, 4, seq);
            assert_eq!(players.len(), 4);
            let mut dedup = players.clone();
            dedup.sort_unstable();
            dedup.dedup();
            assert_eq!(dedup.len(), 4, "players must be distinct: {players:?}");
            if players.contains(&20) {
                with_new += 1;
            }
        }
        assert!(
            with_new > 12,
            "unrated entrant only drawn {with_new}/20 times"
        );
    }

    #[test]
    fn small_pools_use_every_member() {
        let pool = vec![0, 5];
        let records = Vec::new();
        let ratings = fit_ratings(&pool, &records);
        let players = pick_players(&pool, &records, &ratings, 4, 0);
        let mut sorted = players.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 5]);
    }
}
