//! Concurrent net-vs-net evaluation. Every `eval_turns` generations the trainer
//! plays the current checkpoint against several older ones by spawning the
//! snek-server `arena` binary. Opponents are exponentially spaced — 1×, 2×,
//! 4×… `eval_turns` generations back (`eval_opponents` of them, clamped at gen
//! 0) — so each eval point yields one short-horizon "still improving?" result
//! and progressively longer-horizon "progress over time" results. The arena is
//! CPU-only and pins each side to its own cores, so matches run alongside the
//! generate/train loop without touching the GPU.
//!
//! Results land in `runs/<id>/eval/`:
//!
//!   summary.jsonl              one line per matchup (this module appends it)
//!   eval_GGGG_vs_OOOO.json     every game of that matchup, frames + search
//!                              readout, in the exact schema of games/gen_NNNN.json
//!   arena_GGGG_vs_OOOO.log     the arena's stderr, for debugging
//!
//! Matchups within an eval point run sequentially (constant CPU footprint), and
//! only one eval point runs at a time: if a new one comes due while the
//! previous is still playing, it is skipped with a log line — the cadence
//! self-paces to however long an eval actually takes.

use crate::config::RunConfig;
use crate::metrics::Metrics;
use crate::state::RunPaths;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

static EVAL_RUNNING: AtomicBool = AtomicBool::new(false);

/// One line of `runs/<id>/eval/summary.jsonl`: the aggregate outcome of one
/// matchup. `wins` counts for the *current* net (side A).
#[derive(Debug, Serialize, Deserialize)]
pub struct EvalSummary {
    pub gen: u32,
    pub opponent_gen: u32,
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    /// Mean score for the current net (win 1, draw 0.5) and its 95% CI.
    pub score: f64,
    pub score_ci95: f64,
    /// Elo of the current net minus the opponent, with the CI endpoints.
    pub elo: f64,
    pub elo_lo: f64,
    pub elo_hi: f64,
    pub sims: u32,
    pub snakes: u32,
    pub wall_seconds: f64,
    pub finished_unix_ms: i64,
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

/// The per-matchup knobs a spawned eval needs, snapshotted at spawn time so the
/// background thread is independent of later config edits.
struct EvalJob {
    bin: PathBuf,
    eval_dir: PathBuf,
    current: PathBuf,
    gen: u32,
    games: usize,
    sims: usize,
    snakes: usize,
    board: i8,
    cores_a: String,
    cores_b: String,
}

/// Kick off an eval point for `gen` if one is due. Non-blocking: the matchups
/// run in child processes driven by a background thread that appends one
/// summary line each.
pub fn maybe_spawn(paths: &RunPaths, cfg: &RunConfig, gen: u32, metrics: Metrics) {
    if cfg.eval_turns == 0 || gen == 0 || !(gen as usize).is_multiple_of(cfg.eval_turns) {
        return;
    }
    let current = paths.checkpoint_net(gen);
    if !current.exists() {
        metrics.log(format!(
            "eval gen {gen}: skipped — checkpoint missing ({})",
            current.display()
        ));
        return;
    }
    // Exponentially spaced opponents: 1×, 2×, 4×… eval_turns back, stopping at
    // gen 0 (which acts as a fixed anchor early in a run).
    let mut opponents: Vec<u32> = Vec::new();
    for k in 0..cfg.eval_opponents.max(1) {
        let horizon = (cfg.eval_turns as u32) << k;
        let opp = gen.saturating_sub(horizon);
        if opp == gen {
            break;
        }
        if !opponents.contains(&opp) && paths.checkpoint_net(opp).exists() {
            opponents.push(opp);
        }
        if opp == 0 {
            break;
        }
    }
    if opponents.is_empty() {
        metrics.log(format!("eval gen {gen}: skipped — no opponent checkpoints"));
        return;
    }
    let bin = arena_bin();
    if !bin.exists() {
        metrics.log(format!(
            "eval gen {gen}: skipped — arena binary not found at {} (build with `make arena-build` or set SNEK_ARENA_BIN)",
            bin.display()
        ));
        return;
    }
    if EVAL_RUNNING.swap(true, Ordering::SeqCst) {
        metrics.log(format!(
            "eval gen {gen}: skipped — previous eval still running (raise eval_turns or lower eval_games/eval_sims/eval_opponents)"
        ));
        return;
    }

    let eval_dir = paths.root.join("eval");
    if let Err(err) = std::fs::create_dir_all(&eval_dir) {
        metrics.log(format!("eval gen {gen}: failed to create eval dir: {err}"));
        EVAL_RUNNING.store(false, Ordering::SeqCst);
        return;
    }
    let (cores_a, cores_b) = eval_cores(cfg.eval_cores);
    metrics.log(format!(
        "eval gen {gen}: playing gen_{gen:04} vs {opps} — {games} games each, {sims} sims, {snakes} snakes, cores [{cores_a}] vs [{cores_b}]",
        opps = opponents
            .iter()
            .map(|o| format!("gen_{o:04}"))
            .collect::<Vec<_>>()
            .join(", "),
        games = cfg.eval_games,
        sims = cfg.eval_sims,
        snakes = cfg.num_snakes,
    ));

    let job = EvalJob {
        bin,
        eval_dir,
        current,
        gen,
        games: cfg.eval_games,
        sims: cfg.eval_sims,
        snakes: cfg.num_snakes,
        board: cfg.board,
        cores_a,
        cores_b,
    };
    let checkpoints = paths.clone();
    std::thread::Builder::new()
        .name(format!("eval-{gen}"))
        .spawn(move || {
            for opp in opponents {
                run_matchup(&job, checkpoints.checkpoint_net(opp), opp, &metrics);
            }
            EVAL_RUNNING.store(false, Ordering::SeqCst);
        })
        .expect("spawn eval thread");
}

/// Play one matchup (blocking): spawn the arena, wait, append the summary line.
fn run_matchup(job: &EvalJob, opponent: PathBuf, opponent_gen: u32, metrics: &Metrics) {
    let gen = job.gen;
    let out_path = job.eval_dir.join(format!("out_{gen:04}_{opponent_gen:04}.json"));
    let record_path = job
        .eval_dir
        .join(format!("eval_{gen:04}_vs_{opponent_gen:04}.json"));
    let log_path = job
        .eval_dir
        .join(format!("arena_{gen:04}_vs_{opponent_gen:04}.log"));
    let summary_path = job.eval_dir.join("summary.jsonl");

    let mut command = std::process::Command::new(&job.bin);
    command
        .arg("--a")
        .arg(&job.current)
        .arg("--b")
        .arg(&opponent)
        .args(["--name-a", &format!("gen_{gen:04}")])
        .args(["--name-b", &format!("gen_{opponent_gen:04}")])
        .args(["--games", &job.games.to_string()])
        .args(["--sims", &job.sims.to_string()])
        .args(["--snakes", &job.snakes.to_string()])
        .args(["--board", &job.board.to_string()])
        .args(["--cores-a", &job.cores_a])
        .args(["--cores-b", &job.cores_b])
        .args(["--seed", &(gen as u64 * 10_000 + opponent_gen as u64).to_string()])
        .args(["--record-gen", &gen.to_string()])
        .arg("--record")
        .arg(&record_path)
        .arg("--out")
        .arg(&out_path)
        .stdout(std::process::Stdio::null());
    match std::fs::File::create(&log_path) {
        Ok(log) => {
            command.stderr(log);
        }
        Err(_) => {
            command.stderr(std::process::Stdio::null());
        }
    }

    let result = (|| -> anyhow::Result<EvalSummary> {
        let status = command.spawn()?.wait()?;
        if !status.success() {
            anyhow::bail!("arena exited with {status}");
        }
        let out: ArenaOut = serde_json::from_slice(&std::fs::read(&out_path)?)?;
        Ok(EvalSummary {
            gen,
            opponent_gen,
            games: job.games as u32,
            wins: out.summary.a_wins,
            losses: out.summary.b_wins,
            draws: out.summary.draws,
            score: out.summary.score_a,
            score_ci95: out.summary.score_ci95,
            elo: out.summary.elo_a_minus_b,
            elo_lo: out.summary.elo_ci95[0],
            elo_hi: out.summary.elo_ci95[1],
            sims: job.sims as u32,
            snakes: job.snakes as u32,
            wall_seconds: out.summary.wall_seconds,
            finished_unix_ms: chrono::Utc::now().timestamp_millis(),
        })
    })();
    match result {
        Ok(summary) => {
            if let Err(err) = append_summary(&summary_path, &summary) {
                metrics.log(format!("eval gen {gen}: failed to append summary: {err}"));
            }
            metrics.log(format!(
                "eval gen {gen}: gen_{gen:04} {w}-{l} gen_{op:04} ({d} draws), score {score:.3} ± {ci:.3}, elo {elo:+.0} [{lo:+.0}..{hi:+.0}] in {mins:.1}m",
                w = summary.wins,
                l = summary.losses,
                op = summary.opponent_gen,
                d = summary.draws,
                score = summary.score,
                ci = summary.score_ci95,
                elo = summary.elo,
                lo = summary.elo_lo,
                hi = summary.elo_hi,
                mins = summary.wall_seconds / 60.0,
            ));
        }
        Err(err) => {
            metrics.log(format!(
                "eval gen {gen} vs gen_{opponent_gen:04}: failed: {err} (see {})",
                log_path.display()
            ));
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

fn append_summary(path: &Path, summary: &EvalSummary) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string(summary)?)?;
    Ok(())
}

/// Read a run's eval history (oldest first). Missing file → empty.
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
