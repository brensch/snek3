//! Continuous evaluation league. While a run is active, one arena match is
//! always in flight on pinned CPU cores: every checkpoint at a multiple of
//! `league_entrant_gens` generations joins the pool, the scheduler repeatedly
//! picks a pairing (least-played net first, opponents weighted toward similar
//! rating), plays one mirrored game pair via the snek-server `arena` binary,
//! and refits a Bradley–Terry Elo over *all* recorded results — after every
//! single game, streamed from the arena's stdout — anchored at the pool's
//! earliest generation (Elo 0). Ratings of every net keep tightening as long
//! as the run lives, and the Elo-by-generation curve is the run's progress.
//!
//! Files in `runs/<id>/eval/`:
//!
//!   summary.jsonl               one line per match (append-only; the full
//!                               match history the ratings are fitted from)
//!   ratings.json                latest Bradley–Terry fit for every rated net
//!   match_SSSSSS_GGGG_vs_OOOO.json  the match's games, frames + search
//!                               readout, in the schema of games/gen_NNNN.json
//!   arena_SSSSSS.log            that match's arena stderr
//!
//! Old match recordings are pruned (newest `KEEP_MATCH_FILES` kept); the
//! summary log and ratings are kept forever. The league stops (killing any
//! in-flight match) when the run pauses, and resumes with it.

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

/// Recorded match game files kept on disk (newest first); logs/outs likewise.
const KEEP_MATCH_FILES: usize = 60;

/// Games per league match: one mirrored seat-swapped pair. Matches are kept as
/// small as possible so games effectively run back-to-back and results reach
/// the ratings (and the dashboard) within minutes — statistical power comes
/// from the accumulated pool of matches, not from any single match.
const GAMES_PER_MATCH: usize = 2;

/// One league thread per process; a second run start waits for the old one.
static LEAGUE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// One line of `runs/<id>/eval/summary.jsonl`: the aggregate outcome of one
/// match. `wins` counts for the `gen` player (side A of that match).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSummary {
    /// Monotonic match number within the run (0 for legacy pre-league lines).
    #[serde(default)]
    pub seq: u64,
    pub gen: u32,
    pub opponent_gen: u32,
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    /// Mean score for `gen` in this match (win 1, draw 0.5) and its 95% CI.
    pub score: f64,
    pub score_ci95: f64,
    /// Pairwise Elo estimate from this match alone (noisy for small matches —
    /// the fitted league rating in ratings.json is the real signal).
    pub elo: f64,
    pub elo_lo: f64,
    pub elo_hi: f64,
    pub sims: u32,
    pub snakes: u32,
    pub wall_seconds: f64,
    pub finished_unix_ms: i64,
}

/// `runs/<id>/eval/ratings.json`: the latest Bradley–Terry fit.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LeagueRatings {
    pub updated_unix_ms: i64,
    /// Every net with at least one recorded game, ascending by generation.
    pub ratings: Vec<LeagueRating>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeagueRating {
    pub gen: u32,
    /// Bradley–Terry Elo, anchored so the earliest rated generation is 0.
    pub elo: f64,
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

/// An event line the arena prints to stdout: "turn" (per-turn progress, with
/// --events) or "game" (a finished game).
#[derive(Deserialize)]
struct GameEvent {
    event: String,
    #[serde(default)]
    index: u32,
    #[serde(default)]
    turn: u32,
    /// "A", "B", or null for a draw (game events only).
    #[serde(default)]
    winner: Option<String>,
}

/// Real-time state of the in-flight league match, published to the frontend
/// via the `/api/stream/eval` SSE endpoint. `active: false` between matches.
#[derive(Debug, Clone, Serialize)]
pub struct LiveEval {
    pub active: bool,
    pub seq: u64,
    pub gen_a: u32,
    pub gen_b: u32,
    pub games_total: u32,
    /// Cumulative tally for this match; wins count for gen_a.
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    /// In-flight games (index within the match, current turn), ascending.
    pub games: Vec<LiveGame>,
    pub updated_unix_ms: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveGame {
    pub index: u32,
    pub turn: u32,
}

static LIVE: Mutex<LiveEval> = Mutex::new(LiveEval {
    active: false,
    seq: 0,
    gen_a: 0,
    gen_b: 0,
    games_total: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    games: Vec::new(),
    updated_unix_ms: 0,
});

/// Snapshot of the in-flight match state (for the SSE endpoint).
pub fn live() -> LiveEval {
    LIVE.lock().unwrap().clone()
}

fn set_live(update: impl FnOnce(&mut LiveEval)) {
    let mut live = LIVE.lock().unwrap();
    update(&mut live);
    live.updated_unix_ms = chrono::Utc::now().timestamp_millis();
}

/// The slice of the arena's `--out` JSON we consume.
#[derive(Deserialize)]
struct ArenaOut {
    summary: ArenaSummary,
}

#[derive(Deserialize)]
struct ArenaSummary {
    a_wins: u32,
    b_wins: u32,
    draws: u32,
    score_a: f64,
    score_ci95: f64,
    elo_a_minus_b: f64,
    elo_ci95: [f64; 2],
    wall_seconds: f64,
}

/// Start the league thread for the active run. Returns immediately; the thread
/// exits when `stop` is set (killing any in-flight match) or if another league
/// instance never yields.
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
    let mut matches = read_summaries(&paths.root);
    let mut seq = matches.iter().map(|m| m.seq + 1).max().unwrap_or(0);
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
                "eval league: running — pool of {} nets (entrant every {} gens), back-to-back game pairs, {} sims, {} snakes",
                pool.len(),
                cfg.league_entrant_gens,
                cfg.eval_sims,
                cfg.num_snakes,
            ));
            announced = true;
        }
        if std::fs::create_dir_all(&eval_dir).is_err() {
            sleep_unless_stopped(stop, 30);
            continue;
        }

        let ratings = fit_ratings(&pool, &matches);
        let (gen_a, gen_b) = pick_pairing(&pool, &matches, &ratings, seq);
        // Refit and publish ratings after every finished game, not just at
        // match boundaries: the in-flight match joins the fit as a partial
        // result, so the Elo chart moves game by game.
        let mut on_game = |wins: u32, losses: u32, draws: u32| {
            let mut all = matches.clone();
            all.push(interim_summary(seq, gen_a, gen_b, wins, losses, draws));
            let fitted = fit_ratings(&pool, &all);
            let _ = write_ratings(&eval_dir.join("ratings.json"), &all, &fitted);
        };
        match run_match(
            &bin,
            paths,
            &eval_dir,
            &cfg,
            gen_a,
            gen_b,
            seq,
            stop,
            &mut on_game,
        ) {
            MatchResult::Done(summary) => {
                metrics.log(format!(
                    "eval league #{seq}: gen_{a:04} {w}-{l} gen_{b:04} ({d} draws) in {mins:.1}m — {rated} nets rated",
                    a = summary.gen,
                    b = summary.opponent_gen,
                    w = summary.wins,
                    l = summary.losses,
                    d = summary.draws,
                    mins = summary.wall_seconds / 60.0,
                    rated = pool.len(),
                ));
                if let Err(err) = append_summary(&eval_dir.join("summary.jsonl"), &summary) {
                    metrics.log(format!("eval league: failed to append summary: {err}"));
                }
                matches.push(summary);
                let fitted = fit_ratings(&pool, &matches);
                if let Err(err) = write_ratings(&eval_dir.join("ratings.json"), &matches, &fitted) {
                    metrics.log(format!("eval league: failed to write ratings: {err}"));
                }
                prune_match_files(&eval_dir, KEEP_MATCH_FILES);
                seq += 1;
            }
            MatchResult::Stopped => break,
            MatchResult::Failed(err) => {
                metrics.log(format!(
                    "eval league #{seq}: match gen_{gen_a:04} vs gen_{gen_b:04} failed: {err}"
                ));
                seq += 1;
                sleep_unless_stopped(stop, 30);
            }
        }
    }
    metrics.log("eval league: stopped".to_string());
}

enum MatchResult {
    Done(EvalSummary),
    Stopped,
    Failed(String),
}

/// A partial (in-flight) match as it enters the interim ratings fit.
fn interim_summary(
    seq: u64,
    gen: u32,
    opponent_gen: u32,
    wins: u32,
    losses: u32,
    draws: u32,
) -> EvalSummary {
    EvalSummary {
        seq,
        gen,
        opponent_gen,
        games: wins + losses + draws,
        wins,
        losses,
        draws,
        score: 0.0,
        score_ci95: 0.0,
        elo: 0.0,
        elo_lo: 0.0,
        elo_hi: 0.0,
        sims: 0,
        snakes: 0,
        wall_seconds: 0.0,
        finished_unix_ms: chrono::Utc::now().timestamp_millis(),
    }
}

/// Play one match, streaming the arena's per-game stdout events into
/// `on_game(wins, losses, draws)` (cumulative for side A) as they land, and
/// polling `stop` so a pause kills the arena promptly.
#[allow(clippy::too_many_arguments)]
fn run_match(
    bin: &Path,
    paths: &RunPaths,
    eval_dir: &Path,
    cfg: &RunConfig,
    gen_a: u32,
    gen_b: u32,
    seq: u64,
    stop: &AtomicBool,
    on_game: &mut dyn FnMut(u32, u32, u32),
) -> MatchResult {
    let record_path = eval_dir.join(format!("match_{seq:06}_{gen_a:04}_vs_{gen_b:04}.json"));
    let out_path = eval_dir.join(format!("out_{seq:06}.json"));
    let log_path = eval_dir.join(format!("arena_{seq:06}.log"));
    let (cores_a, cores_b) = eval_cores(cfg.eval_cores);

    let mut command = std::process::Command::new(bin);
    command
        .arg("--a")
        .arg(paths.checkpoint_net(gen_a))
        .arg("--b")
        .arg(paths.checkpoint_net(gen_b))
        .args(["--name-a", &format!("gen_{gen_a:04}")])
        .args(["--name-b", &format!("gen_{gen_b:04}")])
        .args(["--games", &GAMES_PER_MATCH.to_string()])
        .args(["--sims", &cfg.eval_sims.to_string()])
        .args(["--snakes", &cfg.num_snakes.to_string()])
        .args(["--board", &cfg.board.to_string()])
        .args(["--cores-a", &cores_a])
        .args(["--cores-b", &cores_b])
        .args(["--seed", &seq.wrapping_mul(1_000_003).to_string()])
        .args(["--record-gen", &gen_a.to_string()])
        .arg("--record")
        .arg(&record_path)
        .arg("--out")
        .arg(&out_path)
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
    // Publish the in-flight match to the live SSE view, and guarantee the
    // "active" flag drops on every exit path.
    set_live(|l| {
        *l = LiveEval {
            active: true,
            seq,
            gen_a,
            gen_b,
            games_total: GAMES_PER_MATCH as u32,
            wins: 0,
            losses: 0,
            draws: 0,
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
    let (mut wins, mut losses, mut draws) = (0u32, 0u32, 0u32);
    // "turn" events update the live view; "game" events also advance the tally
    // and trigger the per-game ratings refit.
    fn apply_event(
        ev: GameEvent,
        wins: &mut u32,
        losses: &mut u32,
        draws: &mut u32,
        on_game: &mut dyn FnMut(u32, u32, u32),
    ) {
        match ev.event.as_str() {
            "turn" => set_live(|l| {
                match l.games.iter_mut().find(|g| g.index == ev.index) {
                    Some(game) => game.turn = ev.turn,
                    None => l.games.push(LiveGame {
                        index: ev.index,
                        turn: ev.turn,
                    }),
                }
                l.games.sort_by_key(|g| g.index);
            }),
            "game" => {
                match ev.winner.as_deref() {
                    Some("A") => *wins += 1,
                    Some("B") => *losses += 1,
                    _ => *draws += 1,
                }
                let (w, l, d) = (*wins, *losses, *draws);
                set_live(|live| {
                    live.wins = w;
                    live.losses = l;
                    live.draws = d;
                    live.games.retain(|g| g.index != ev.index);
                });
                on_game(w, l, d);
            }
            _ => {}
        }
    }
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
            apply_event(ev, &mut wins, &mut losses, &mut draws, on_game);
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
    // Drain events that landed between the last poll and child exit.
    while let Ok(ev) = event_rx.try_recv() {
        apply_event(ev, &mut wins, &mut losses, &mut draws, on_game);
    }
    if !status.success() {
        return MatchResult::Failed(format!(
            "arena exited with {status} (see {})",
            log_path.display()
        ));
    }
    let out: ArenaOut = match std::fs::read(&out_path)
        .map_err(anyhow::Error::from)
        .and_then(|b| serde_json::from_slice(&b).map_err(anyhow::Error::from))
    {
        Ok(out) => out,
        Err(err) => return MatchResult::Failed(format!("parse {}: {err}", out_path.display())),
    };
    MatchResult::Done(EvalSummary {
        seq,
        gen: gen_a,
        opponent_gen: gen_b,
        games: GAMES_PER_MATCH as u32,
        wins: out.summary.a_wins,
        losses: out.summary.b_wins,
        draws: out.summary.draws,
        score: out.summary.score_a,
        score_ci95: out.summary.score_ci95,
        elo: out.summary.elo_a_minus_b,
        elo_lo: out.summary.elo_ci95[0],
        elo_hi: out.summary.elo_ci95[1],
        sims: cfg.eval_sims as u32,
        snakes: cfg.num_snakes as u32,
        wall_seconds: out.summary.wall_seconds,
        finished_unix_ms: chrono::Utc::now().timestamp_millis(),
    })
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

/// Pick the next pairing: the least-played pool member (newest on ties, so a
/// freshly admitted net gets rated quickly), against an opponent sampled with
/// weight favouring similar rating — informative games, but with enough spread
/// that the match graph stays connected across the whole pool.
fn pick_pairing(
    pool: &[u32],
    matches: &[EvalSummary],
    ratings: &HashMap<u32, f64>,
    seq: u64,
) -> (u32, u32) {
    let mut games: HashMap<u32, u32> = pool.iter().map(|&g| (g, 0)).collect();
    for m in matches {
        for g in [m.gen, m.opponent_gen] {
            if let Some(count) = games.get_mut(&g) {
                *count += m.games;
            }
        }
    }
    let &a = pool
        .iter()
        .min_by_key(|&&g| (games[&g], u32::MAX - g))
        .expect("pool has members");
    let elo_a = ratings.get(&a).copied().unwrap_or(0.0);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let others: Vec<u32> = pool.iter().copied().filter(|&g| g != a).collect();
    let weights: Vec<f64> = others
        .iter()
        .map(|&g| {
            let d = (ratings.get(&g).copied().unwrap_or(0.0) - elo_a) / 300.0;
            (-d * d).exp() + 0.05
        })
        .collect();
    let total: f64 = weights.iter().sum();
    let mut pick = rng.gen_range(0.0..total);
    for (i, w) in weights.iter().enumerate() {
        pick -= w;
        if pick <= 0.0 {
            return (a, others[i]);
        }
    }
    (a, *others.last().expect("pool has an opponent"))
}

/// Bradley–Terry maximum-likelihood ratings over every recorded match, via the
/// standard MM iteration. Draws count half a win each side. A half virtual
/// draw between adjacent pool members keeps the graph connected and every
/// rating finite (a net with only wins would otherwise diverge). Anchored so
/// the earliest rated generation sits at Elo 0.
fn fit_ratings(pool: &[u32], matches: &[EvalSummary]) -> HashMap<u32, f64> {
    let mut gens: Vec<u32> = pool.to_vec();
    for m in matches {
        gens.push(m.gen);
        gens.push(m.opponent_gen);
    }
    gens.sort_unstable();
    gens.dedup();
    if gens.len() < 2 {
        return gens.into_iter().map(|g| (g, 0.0)).collect();
    }
    let index: HashMap<u32, usize> = gens.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let n = gens.len();
    // wins[i][j] = win-equivalents of i over j.
    let mut wins = vec![vec![0.0f64; n]; n];
    for m in matches {
        let (i, j) = (index[&m.gen], index[&m.opponent_gen]);
        wins[i][j] += m.wins as f64 + m.draws as f64 * 0.5;
        wins[j][i] += m.losses as f64 + m.draws as f64 * 0.5;
    }
    for w in gens.windows(2) {
        let (i, j) = (index[&w[0]], index[&w[1]]);
        wins[i][j] += 0.25;
        wins[j][i] += 0.25;
    }

    let mut p = vec![1.0f64; n];
    for _ in 0..200 {
        let mut next = p.clone();
        for i in 0..n {
            let total_wins: f64 = (0..n).map(|j| wins[i][j]).sum();
            let denom: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let n_ij = wins[i][j] + wins[j][i];
                    if n_ij > 0.0 {
                        n_ij / (p[i] + p[j])
                    } else {
                        0.0
                    }
                })
                .sum();
            if denom > 0.0 && total_wins > 0.0 {
                next[i] = total_wins / denom;
            }
        }
        // Renormalize each round so the iteration stays well-conditioned.
        let mean = next.iter().sum::<f64>() / n as f64;
        for v in &mut next {
            *v /= mean;
        }
        p = next;
    }
    let anchor = p[0];
    gens.iter()
        .enumerate()
        .map(|(i, &g)| (g, 400.0 * (p[i] / anchor).log10()))
        .collect()
}

fn write_ratings(
    path: &Path,
    matches: &[EvalSummary],
    fitted: &HashMap<u32, f64>,
) -> anyhow::Result<()> {
    let mut tally: HashMap<u32, (u32, u32, u32, u32)> = HashMap::new(); // games, w, l, d
    for m in matches {
        let a = tally.entry(m.gen).or_default();
        a.0 += m.games;
        a.1 += m.wins;
        a.2 += m.losses;
        a.3 += m.draws;
        let b = tally.entry(m.opponent_gen).or_default();
        b.0 += m.games;
        b.1 += m.losses;
        b.2 += m.wins;
        b.3 += m.draws;
    }
    let mut ratings: Vec<LeagueRating> = fitted
        .iter()
        .map(|(&gen, &elo)| {
            let (games, wins, losses, draws) = tally.get(&gen).copied().unwrap_or_default();
            LeagueRating {
                gen,
                elo,
                games,
                wins,
                losses,
                draws,
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

/// Keep only the newest `keep` recorded matches on disk (games + logs + outs).
/// The summary log and ratings are never pruned.
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

/// Pin the eval to the machine's highest-numbered cores (self-play's rayon pool
/// is unpinned, so the scheduler migrates it away from the busy eval cores):
/// side B takes the top `per_side`, side A the `per_side` below that.
fn eval_cores(per_side: usize) -> (String, String) {
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let per_side = per_side.max(1).min(total / 2).max(1);
    let spec = |lo: usize, hi: usize| {
        (lo..hi)
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let b_lo = total.saturating_sub(per_side);
    let a_lo = total.saturating_sub(2 * per_side);
    (spec(a_lo, b_lo), spec(b_lo, total))
}

fn sleep_unless_stopped(stop: &AtomicBool, secs: u64) {
    for _ in 0..secs {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn append_summary(path: &Path, summary: &EvalSummary) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string(summary)?)?;
    Ok(())
}

/// Read a run's match history (oldest first). Missing file → empty.
pub fn read_summaries(run_root: &Path) -> Vec<EvalSummary> {
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

    fn m(seq: u64, gen: u32, opponent_gen: u32, wins: u32, losses: u32, draws: u32) -> EvalSummary {
        EvalSummary {
            seq,
            gen,
            opponent_gen,
            games: wins + losses + draws,
            wins,
            losses,
            draws,
            score: 0.0,
            score_ci95: 0.0,
            elo: 0.0,
            elo_lo: 0.0,
            elo_hi: 0.0,
            sims: 0,
            snakes: 0,
            wall_seconds: 0.0,
            finished_unix_ms: 0,
        }
    }

    #[test]
    fn fit_orders_transitively_and_anchors_at_zero() {
        let pool = vec![0, 5, 10];
        let matches = vec![
            m(0, 5, 0, 8, 2, 0),
            m(1, 10, 5, 8, 2, 0),
            m(2, 10, 0, 9, 1, 0),
        ];
        let elo = fit_ratings(&pool, &matches);
        assert_eq!(elo[&0], 0.0);
        assert!(
            elo[&5] > 50.0,
            "gen 5 should clearly beat the anchor: {elo:?}"
        );
        assert!(
            elo[&10] > elo[&5],
            "gen 10 should rank above gen 5: {elo:?}"
        );
    }

    #[test]
    fn fit_stays_finite_on_perfect_records() {
        // An undefeated net must not diverge (the virtual adjacency draws
        // regularize it).
        let pool = vec![0, 5];
        let matches = vec![m(0, 5, 0, 10, 0, 0)];
        let elo = fit_ratings(&pool, &matches);
        assert!(
            elo[&5].is_finite() && elo[&5] > 0.0 && elo[&5] < 2000.0,
            "{elo:?}"
        );
    }

    #[test]
    fn fit_rates_unplayed_pool_members_via_virtual_draws() {
        // A freshly admitted net with no games yet still gets a finite rating
        // (pulled toward its neighbours) instead of poisoning the fit.
        let pool = vec![0, 5, 10];
        let matches = vec![m(0, 5, 0, 6, 4, 0)];
        let elo = fit_ratings(&pool, &matches);
        assert!(elo[&10].is_finite(), "{elo:?}");
    }

    #[test]
    fn pairing_picks_least_played_and_a_distinct_opponent() {
        let pool = vec![0, 5, 10, 15];
        // gen 15 has no games yet; everyone else has played.
        let matches = vec![
            m(0, 5, 0, 2, 2, 0),
            m(1, 10, 5, 2, 2, 0),
            m(2, 10, 0, 2, 2, 0),
        ];
        let ratings = fit_ratings(&pool, &matches);
        for seq in 0..20 {
            let (a, b) = pick_pairing(&pool, &matches, &ratings, seq);
            assert_eq!(a, 15, "least-played (newest) member goes first");
            assert_ne!(a, b);
            assert!(pool.contains(&b));
        }
    }
}
