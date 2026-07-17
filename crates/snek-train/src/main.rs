mod api;
mod bench;
mod config;
mod eval;
mod fwd_bench;
mod gate;
mod le_eval;
mod le_h2h;
mod metrics;
mod proto;
mod replay;
mod sample;
mod selfplay;
mod session;
mod sl_pretrain;
mod state;
mod train;
mod trainer;

use clap::Parser;
use config::RunConfig;
use metrics::Metrics;
use std::net::SocketAddr;
use std::path::PathBuf;
use trainer::TrainerHandle;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8050")]
    bind: SocketAddr,
    #[arg(long, default_value = "runs")]
    runs_dir: PathBuf,
    #[arg(long)]
    run_id: Option<String>,
    #[arg(long)]
    fresh: bool,
    #[arg(long, default_value_t = false)]
    start: bool,
    /// Directory of built frontend assets to serve at /. Skipped if missing.
    #[arg(long, default_value = "frontend/dist")]
    static_dir: PathBuf,
    /// Run one GPU burst of league games against this run id, then exit (no
    /// server). Results land in `<runs_dir>/<run>/<burst_out>/`.
    #[arg(long)]
    burst: Option<String>,
    /// Games to play in the standalone burst (default: the run config's
    /// `burst_games`).
    #[arg(long)]
    burst_games: Option<usize>,
    /// Sims per move in the standalone burst (default: the run config's
    /// `burst_sims`, falling back to `eval_sims`).
    #[arg(long)]
    burst_sims: Option<usize>,
    /// Output subdir under the run for standalone burst records. Kept apart
    /// from the canonical `eval/` so a mirror of a live run never diverges
    /// from the trainer's own files.
    #[arg(long, default_value = "eval-burst")]
    burst_out: String,
    /// Run this many back-to-back bursts, carrying the in-flight arena
    /// between them (exercises the trainer's freeze/thaw path).
    #[arg(long, default_value_t = 1)]
    burst_repeat: usize,
    /// Supervised-pretrain a FRESH net on a directory of GameFile `.json.zst`
    /// archives (e.g. scraped ladder games), report what it learned + its play
    /// strength vs the baselines, then exit. No self-play, no server.
    #[arg(long)]
    sl_pretrain: Option<PathBuf>,
    /// Where the pretrained net is written (best-validation checkpoint).
    #[arg(long, default_value = "checkpoints/sl_pretrain.safetensors")]
    sl_out: PathBuf,
    /// Max training epochs (early stopping usually ends it sooner).
    #[arg(long, default_value_t = 40)]
    sl_epochs: usize,
    /// Fraction of games held out for validation / early stopping.
    #[arg(long, default_value_t = 0.1)]
    sl_val_frac: f64,
    /// Play-strength games per baseline after training.
    #[arg(long, default_value_t = 24)]
    sl_eval_games: usize,
    /// Per-move time budget (ms) in the play-strength eval.
    #[arg(long, default_value_t = 40)]
    sl_eval_ms: u64,
    /// Play-strength eval only: measure this saved net vs the baselines and exit
    /// (no training). Uses `--sl-eval-games` / `--sl-eval-ms`.
    #[arg(long)]
    sl_eval_only: Option<PathBuf>,
    /// Full-strength round-robin among net checkpoints (comma-separated gens,
    /// e.g. "600,1200,1800,2300"): seat them head-to-head and report each gen's
    /// win rate + avg rank, testing for a skill trend the low-sim league hides.
    /// Uses `--run-id` (default snek3-21) for the checkpoint dir, then exits.
    #[arg(long)]
    rr: Option<String>,
    /// LE checkpoint sweep (comma-separated gens): each checkpoint plays the
    /// run's gate opponent (voronoi@gate_sims) from the SAME start seeds and
    /// its win rate is printed — used to pick the strongest historical net as
    /// the gating incumbent. Uses `--run-id` for checkpoints + config, then
    /// exits.
    #[arg(long)]
    le_sweep: Option<String>,
    /// Games per checkpoint in the LE sweep.
    #[arg(long, default_value_t = 64)]
    le_sweep_games: usize,
    /// Chunk-size throughput sweep for the staged LE leaf forward on the run's
    /// net shape (uses --run-id). Run with training stopped, then exits.
    #[arg(long, default_value_t = false)]
    fwd_bench: bool,
    /// Useful rows for the large (deepen) forward in the fwd bench.
    #[arg(long, default_value_t = 30720)]
    fwd_bench_big: usize,
    /// Useful rows for the small (root) forward in the fwd bench.
    #[arg(long, default_value_t = 3840)]
    fwd_bench_small: usize,
    /// Print the value head's output on real boards for a checkpoint (path to
    /// .safetensors; uses --run-id for the net shape). Diagnoses a saturated
    /// (untrainable) value head, then exits.
    #[arg(long)]
    value_probe: Option<PathBuf>,
    /// Games in the round-robin.
    #[arg(long, default_value_t = 160)]
    rr_games: usize,
    /// Sims/move in the round-robin (full strength; league uses 64).
    #[arg(long, default_value_t = 800)]
    rr_sims: usize,
    /// Concurrent games in the round-robin (keeps the GPU busy).
    #[arg(long, default_value_t = 24)]
    rr_conc: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    tch::Cuda::cudnn_set_benchmark(std::env::var("SNEK_CUDNN_BENCH").as_deref() != Ok("0"));
    let args = Args::parse();
    let metrics = Metrics::new();
    if let Some(run_id) = args.burst.as_deref() {
        return standalone_burst(&args, run_id, metrics);
    }
    if let Some(net) = args.sl_eval_only.clone() {
        return sl_pretrain::eval_only(&net, args.sl_eval_games, args.sl_eval_ms);
    }
    if let Some(spec) = args.le_sweep.clone() {
        let gens: Vec<u32> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let run_id = args.run_id.clone().ok_or_else(|| anyhow::anyhow!("--le-sweep needs --run-id"))?;
        anyhow::ensure!(tch::Cuda::is_available(), "le-sweep needs CUDA");
        return gate::sweep(
            &state::RunPaths::new(&args.runs_dir, &run_id),
            &gens,
            args.le_sweep_games,
        );
    }
    if let Some(ckpt) = args.value_probe.clone() {
        let run_id = args
            .run_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--value-probe needs --run-id"))?;
        anyhow::ensure!(tch::Cuda::is_available(), "value-probe needs CUDA");
        return fwd_bench::value_probe(&state::RunPaths::new(&args.runs_dir, &run_id), &ckpt);
    }
    if args.fwd_bench {
        let run_id = args
            .run_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--fwd-bench needs --run-id"))?;
        anyhow::ensure!(tch::Cuda::is_available(), "fwd-bench needs CUDA");
        return fwd_bench::run(
            &state::RunPaths::new(&args.runs_dir, &run_id),
            args.fwd_bench_big,
            args.fwd_bench_small,
        );
    }
    if let Some(spec) = args.rr.clone() {
        let gens: Vec<u32> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let run_id = args.run_id.clone().unwrap_or_else(|| "snek3-21".to_string());
        anyhow::ensure!(tch::Cuda::is_available(), "rr needs CUDA");
        let device = tch::Device::Cuda(0);
        let cfg = RunConfig::default();
        let ckpt_dir = args.runs_dir.join(&run_id).join("checkpoints");
        let _ = args.rr_conc;
        return eval::burst::run_round_robin(
            &ckpt_dir, device, &metrics, &cfg, &gens, args.rr_games, args.rr_sims,
        );
    }
    if let Some(dir) = args.sl_pretrain.clone() {
        return sl_pretrain::run(&sl_pretrain::SlArgs {
            dir,
            out: args.sl_out.clone(),
            epochs: args.sl_epochs,
            val_frac: args.sl_val_frac,
            eval_games: args.sl_eval_games,
            eval_ms: args.sl_eval_ms,
        });
    }
    let trainer = TrainerHandle::new(args.runs_dir, metrics.clone(), RunConfig::default());
    tokio::spawn(metrics.run_samplers());
    if args.start {
        trainer.start(trainer::StartRequest {
            run_id: args.run_id,
            fresh: Some(args.fresh),
            config: None,
        })?;
    }
    let static_dir = args
        .static_dir
        .join("index.html")
        .is_file()
        .then_some(args.static_dir.as_path());
    match static_dir {
        Some(dir) => tracing::info!(dir = %dir.display(), "serving frontend"),
        None => {
            tracing::warn!(dir = %args.static_dir.display(), "no built frontend found; serving API only")
        }
    }
    let app = api::router(trainer, static_dir);
    tracing::info!(bind = %args.bind, "snek-train API listening");
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// `--burst <run>`: play one GPU burst of league games against a run
/// directory and exit. Reads the run's config (with optional CLI overrides),
/// seeds match history from the canonical eval dir plus the output dir, and
/// writes summary/ratings/game files to the output dir only.
fn standalone_burst(args: &Args, run_id: &str, metrics: Metrics) -> anyhow::Result<()> {
    anyhow::ensure!(
        tch::Cuda::is_available(),
        "burst needs CUDA; check LD_PRELOAD/LD_LIBRARY_PATH"
    );
    let device = tch::Device::Cuda(0);
    let out_dir = eval::burst::standalone_out_dir(&args.runs_dir, run_id, &args.burst_out);
    let (ctx, mut cfg) = eval::burst::standalone_ctx(&args.runs_dir, run_id, &out_dir, metrics.clone())?;
    if let Some(games) = args.burst_games {
        cfg.burst_games = games;
    }
    if let Some(sims) = args.burst_sims {
        cfg.burst_sims = sims;
    }
    let sims = if cfg.burst_sims > 0 { cfg.burst_sims } else { cfg.eval_sims };
    tracing::info!(
        run_id,
        games = cfg.burst_games,
        sims,
        out = %out_dir.display(),
        "standalone burst starting"
    );
    // SIGINT/SIGTERM request a *graceful* stop: finish the in-flight lockstep
    // turn, freeze the arena, and flush every finished game to summary.jsonl —
    // an hour-long burst interrupted at game 47/50 keeps its 47 games. (The
    // burst loop below also stops repeating once the flag is set.)
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let stop = std::sync::Arc::clone(&stop);
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!("signal received: freezing arena and flushing finished games…");
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
        });
    }
    // Progress ticker: a line every few seconds with rates over the interval,
    // so slow spots are visible immediately.
    let done_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ticker = {
        let counters = metrics.counters();
        let done = std::sync::Arc::clone(&done_flag);
        let started = std::time::Instant::now();
        std::thread::spawn(move || {
            use std::sync::atomic::Ordering::Relaxed;
            let (mut last_done, mut last_inf, mut last_rows, mut last_reqs, mut last_fwd_us) =
                (0u32, 0u64, 0u64, 0u64, 0u64);
            let mut last_t = std::time::Instant::now();
            while !done.load(Relaxed) {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let dt = last_t.elapsed().as_secs_f64().max(1e-9);
                last_t = std::time::Instant::now();
                let games = counters.arena_done.load(Relaxed);
                let target = counters.arena_target.load(Relaxed);
                let inf = counters.inferences.load(Relaxed);
                let rows = counters.gpu_rows.load(Relaxed);
                let reqs = counters.gpu_requests.load(Relaxed);
                let fwd_us = counters.gpu_forward_us.load(Relaxed);
                println!(
                    "[{:>5.0}s] {games}/{target} games ({:.1} games/min) · {:.0}k inf/s · {} rows/fwd · gpu fwd {:.0}%",
                    started.elapsed().as_secs_f64(),
                    60.0 * (games.saturating_sub(last_done)) as f64 / dt,
                    (inf - last_inf) as f64 / dt / 1000.0,
                    (rows - last_rows) / (reqs - last_reqs).max(1),
                    100.0 * ((fwd_us - last_fwd_us) as f64 / 1e6) / dt,
                );
                (last_done, last_inf, last_rows, last_reqs, last_fwd_us) =
                    (games, inf, rows, reqs, fwd_us);
            }
        })
    };
    let mut arena = eval::burst::ArenaState::default();
    let mut result = Ok(());
    for i in 0..args.burst_repeat.max(1) {
        if stop.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        match eval::burst::run_burst(&ctx, &cfg, device, &metrics, &stop, &out_dir, &mut arena) {
            Ok(report) => println!(
                "burst {n}/{total} done: {games} games @{sims} sims, {turns} turns in {secs:.0}s ({gpm:.1} games/min, {inf:.0}k inf/s) -> {out}",
                n = i + 1,
                total = args.burst_repeat.max(1),
                games = report.games,
                turns = report.turns,
                secs = report.seconds,
                gpm = 60.0 * report.games as f64 / report.seconds.max(1e-9),
                inf = report.inferences as f64 / report.seconds.max(1e-9) / 1000.0,
                out = out_dir.display(),
            ),
            Err(err) => {
                result = Err(err);
                break;
            }
        }
    }
    done_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = ticker.join();
    result
}
