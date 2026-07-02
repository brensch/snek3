use crate::config::RunConfig;
use crate::metrics::Metrics;
use crate::proto::Phase;
use crate::replay::ReplayBuffer;
use crate::selfplay::{generate, GenOutcome, SelfPlayNet, SelfPlayState};
use crate::state::{load_trainer_state, save_trainer_state, RunPaths};
use crate::train::{build_optimizer, train_steps};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tch::{nn, Device};

/// How many generations of recorded sample games to retain on disk per run.
/// `0` keeps every generation forever.
const SAMPLE_GAMES_KEEP: usize = 0;

#[derive(Clone)]
pub struct TrainerHandle {
    runs_dir: PathBuf,
    metrics: Metrics,
    config: Arc<Mutex<RunConfig>>,
    active_run: Arc<Mutex<Option<String>>>,
    running: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    cuda_active: Arc<AtomicBool>,
    /// Set while a GPU batch-size benchmark is sweeping. Bench and a training run
    /// both want exclusive use of the GPU, so each refuses to start while the
    /// other holds this / `running`.
    bench_active: Arc<AtomicBool>,
}

#[derive(Debug, Deserialize)]
pub struct StartRequest {
    pub run_id: Option<String>,
    pub fresh: Option<bool>,
    /// Knob overrides for the new run. Applied to the in-memory config before the
    /// run loop spawns, so a fresh run picks them up via `self.config()`.
    pub config: Option<RunConfig>,
}

impl TrainerHandle {
    pub fn new(runs_dir: PathBuf, metrics: Metrics, config: RunConfig) -> Self {
        Self {
            runs_dir,
            metrics,
            config: Arc::new(Mutex::new(config)),
            active_run: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
            cuda_active: Arc::new(AtomicBool::new(false)),
            bench_active: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn metrics(&self) -> Metrics {
        self.metrics.clone()
    }

    pub fn config(&self) -> RunConfig {
        self.config.lock().unwrap().clone()
    }

    pub fn set_config(&self, cfg: RunConfig) {
        *self.config.lock().unwrap() = cfg;
    }

    pub fn start(&self, req: StartRequest) -> anyhow::Result<String> {
        if self.bench_active.load(Ordering::SeqCst) {
            anyhow::bail!("a GPU benchmark is running — wait for it to finish");
        }
        if self.running.swap(true, Ordering::SeqCst) {
            // A start while the previous loop is still draining its stop would
            // be silently swallowed (the loop exits and nothing restarts) —
            // refuse it instead so the caller can retry once fully stopped.
            if self.stop.load(Ordering::SeqCst) {
                anyhow::bail!("previous run is still stopping — retry in a moment");
            }
            return self
                .active_run
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| anyhow::anyhow!("trainer is running without active run"));
        }
        self.stop.store(false, Ordering::SeqCst);
        if let Some(config) = req.config {
            self.set_config(config);
        }
        let run_id = req.run_id.unwrap_or_else(timestamp_run_id);
        *self.active_run.lock().unwrap() = Some(run_id.clone());
        let handle = self.clone();
        let run_id_for_thread = run_id.clone();
        std::thread::spawn(move || {
            if let Err(err) = handle.run_loop(&run_id_for_thread, req.fresh.unwrap_or(false)) {
                tracing::error!(?err, "trainer failed");
            }
            handle.metrics.set_phase(Phase::Stopped);
            handle.running.store(false, Ordering::SeqCst);
        });
        Ok(run_id)
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.metrics.set_phase(Phase::Stopping);
        self.log("stop requested — interrupting self-play, saving snapshot");
    }

    pub fn runs_dir(&self) -> &Path {
        &self.runs_dir
    }

    pub fn active_run_id(&self) -> Option<String> {
        self.active_run.lock().unwrap().clone()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Claim the GPU for a benchmark sweep. Fails if a training run is active or a
    /// benchmark is already in flight; the caller must pair success with
    /// [`end_bench`]. Returns an error whose message explains the refusal.
    pub fn try_begin_bench(&self) -> anyhow::Result<()> {
        if self.running.load(Ordering::SeqCst) {
            anyhow::bail!("a training run is active — stop it before benchmarking");
        }
        if self.bench_active.swap(true, Ordering::SeqCst) {
            anyhow::bail!("a GPU benchmark is already running");
        }
        Ok(())
    }

    /// Release the GPU claimed by [`try_begin_bench`].
    pub fn end_bench(&self) {
        self.bench_active.store(false, Ordering::SeqCst);
    }

    /// Emit a trainer event to the terminal and the frontend log stream.
    pub fn log(&self, message: impl Into<String>) {
        self.metrics.log(message);
    }

    pub fn history(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let Some(run_id) = self.active_run.lock().unwrap().clone() else {
            return Ok(Vec::new());
        };
        let path = RunPaths::new(&self.runs_dir, &run_id).metrics;
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(path)?;
        let mut rows = Vec::new();
        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            if let Ok(row) = serde_json::from_str(line) {
                rows.push(row);
            }
        }
        Ok(rows)
    }

    pub fn run_state(&self) -> crate::proto::RunState {
        let phase =
            crate::metrics::phase_from_u32(self.metrics.counters().phase.load(Ordering::Relaxed));
        crate::proto::RunState {
            phase: phase as i32,
            generation: self.metrics.counters().generation.load(Ordering::Relaxed),
            run_id: self.active_run.lock().unwrap().clone().unwrap_or_default(),
            running: self.running.load(Ordering::Relaxed),
        }
    }

    pub fn device_label(&self) -> &'static str {
        if self.cuda_active.load(Ordering::Relaxed) {
            "cuda"
        } else {
            "cpu"
        }
    }

    fn run_loop(&self, run_id: &str, fresh: bool) -> anyhow::Result<()> {
        // Signal life immediately: restoring a large replay buffer below can
        // take tens of seconds, and until the first generation starts the
        // phase would otherwise sit at Stopped and make a resume look ignored.
        self.metrics.set_phase(Phase::Playing);
        let paths = RunPaths::new(&self.runs_dir, run_id);
        paths.ensure()?;
        let mut cfg = if !fresh && paths.config.exists() {
            RunConfig::load(&paths.config)?
        } else {
            self.config()
        };
        configure_search_threads(&mut cfg);
        cfg.save_atomic(&paths.config)?;
        self.set_config(cfg.clone());

        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            anyhow::bail!("CUDA is not available to libtorch; check LD_PRELOAD/LD_LIBRARY_PATH or set up a CUDA-enabled libtorch")
        };
        self.cuda_active
            .store(matches!(device, Device::Cuda(_)), Ordering::Relaxed);
        tracing::info!(?device, "trainer selected device");
        let mut vs = nn::VarStore::new(device);
        let net = snek_tch::AZNet::new(
            &vs.root(),
            snek_core::NUM_CHANNELS as i64,
            cfg.trunk_channels,
            cfg.trunk_blocks,
        );
        if !fresh && paths.net.exists() {
            vs.load(&paths.net)?;
        } else {
            snek_tch::init_orthogonal(&vs, 2f64.sqrt());
        }
        let mut opt = build_optimizer(&vs)?;
        let mut state = if fresh {
            Default::default()
        } else {
            load_trainer_state(&paths.trainer_state)?
        };
        if !fresh {
            self.log(format!("resuming run '{run_id}': restoring replay buffer…"));
        }
        let mut replay = if fresh {
            ReplayBuffer::new(cfg.buffer_size)
        } else {
            ReplayBuffer::restore(&paths.replay, cfg.buffer_size, state.generation)?
        };
        // Reload the whole self-play session so a resumed run continues its
        // in-flight games *and* the current generation's accumulated (finished)
        // games from where they stopped. Empty here means `generate` starts fresh.
        let mut sp: SelfPlayState = if fresh {
            SelfPlayState::default()
        } else {
            crate::session::load(&paths.session)?.unwrap_or_default()
        };
        self.log(format!(
            "{verb} run '{run_id}' at gen {gen}: buffer {buf} samples (avg turn {avg:.1}), {games} in-flight games, {fin} finished games buffered",
            verb = if fresh { "starting" } else { "resuming" },
            gen = state.generation,
            buf = replay.len(),
            avg = replay.avg_turn(),
            games = sp.boards.len(),
            fin = sp.finished.len(),
        ));
        // Continuous CPU evaluation league: plays checkpoint-vs-checkpoint arena
        // matches on pinned cores for as long as the run is active, maintaining
        // Bradley–Terry Elo ratings in runs/<id>/eval/. Stops when we stop.
        crate::eval::start_league(paths.clone(), self.clone(), self.stop.clone());

        while !self.stop.load(Ordering::Relaxed) {
            let cfg = self.config();
            cfg.save_atomic(&paths.config)?;
            self.metrics
                .counters()
                .generation
                .store(state.generation, Ordering::Relaxed);
            self.metrics.set_phase(Phase::Playing);
            let counters = self.metrics.counters();
            let gen_start = Instant::now();
            let inf_before = counters.inferences.load(Ordering::Relaxed);
            let fwd_us_before = counters.gpu_forward_us.load(Ordering::Relaxed);
            let outcome = generate(
                &SelfPlayNet { net: &net, device },
                &cfg,
                state.seed + state.generation as u64,
                &self.metrics,
                &self.stop,
                &mut sp,
            )?;
            // A pause interrupts the generation: snapshot the whole session (in-
            // flight games + this generation's accumulated finished games) and bail
            // out without training. Resume reloads it and continues the *same*
            // generation from the same sample count — nothing is lost or retrained.
            let (samples, display_games) = match outcome {
                GenOutcome::Interrupted => break,
                GenOutcome::Complete {
                    samples,
                    display_games,
                } => (samples, display_games),
            };
            let play_seconds = gen_start.elapsed().as_secs_f64();
            // Per-generation figures (not cumulative): `samples` reports this
            // generation's completed games / turns / samples directly.
            let gen_completed_games = samples.games as u32;
            let gen_turns = samples.turns as u32;
            let gen_samples = samples.len() as u32;
            let gen_inferences = counters.inferences.load(Ordering::Relaxed) - inf_before;
            let gpu_forward_seconds = counters
                .gpu_forward_us
                .load(Ordering::Relaxed)
                .saturating_sub(fwd_us_before) as f64
                / 1_000_000.0;

            if !display_games.is_empty() {
                if let Err(err) = crate::sample::write_generation(
                    &paths.games,
                    state.generation,
                    display_games,
                    serde_json::to_value(&cfg).unwrap_or_default(),
                    SAMPLE_GAMES_KEEP,
                ) {
                    tracing::warn!(%err, "failed to write sample games");
                }
            }
            replay.save_shard(&paths.replay, state.generation, &samples)?;
            state.samples_seen += gen_samples as u64;
            replay.add(samples);
            let buffer_len = replay.len() as u64;
            let avg_game_turn = replay.avg_turn();
            // Persist the session so a resume continues the in-flight games (the
            // finished buffer is now drained into this generation's shard).
            if let Err(err) = crate::session::save(&paths.session, &sp) {
                tracing::warn!(%err, "failed to save self-play session");
            }

            self.metrics.set_phase(Phase::Training);
            // The LR schedule is code-owned (see `train::lr_for`); re-applied
            // every generation so the decay advances with samples_seen.
            let lr = crate::train::lr_for(state.samples_seen);
            opt.set_lr(lr);
            let train_start = Instant::now();
            let losses = train_steps(
                &net,
                &vs,
                &mut opt,
                &replay,
                &cfg,
                state.seed ^ state.generation as u64,
                &counters,
            )?;
            let train_seconds = train_start.elapsed().as_secs_f64();
            counters.set_losses(losses.policy_loss, losses.value_loss, losses.target_entropy);

            self.metrics.set_phase(Phase::Checkpoint);
            vs.save(&paths.net)?;
            // Also archive this generation's weights, kept forever.
            vs.save(paths.checkpoint_net(state.generation))?;
            append_metric(
                &paths.metrics,
                &GenRecord {
                    generation: state.generation,
                    policy_loss: losses.policy_loss,
                    value_loss: losses.value_loss,
                    target_entropy: losses.target_entropy,
                    lr,
                    win_rate: 0.0,
                    completed_games: gen_completed_games,
                    samples: gen_samples,
                    turns: gen_turns,
                    buffer: buffer_len,
                    samples_seen: state.samples_seen,
                    gen_seconds: gen_start.elapsed().as_secs_f64(),
                    play_seconds,
                    train_seconds,
                    inferences: gen_inferences,
                    inferences_per_sec: safe_div(gen_inferences as f64, play_seconds),
                    games_per_sec: safe_div(gen_completed_games as f64, play_seconds),
                    turns_per_sec: safe_div(gen_turns as f64, play_seconds),
                    gpu_busy_pct: (100.0 * safe_div(gpu_forward_seconds, play_seconds))
                        .clamp(0.0, 100.0),
                    avg_game_turn,
                },
            )?;
            self.log(format!(
                "gen {gen} done: {games} games, {samples} samples, buffer {buf} (avg turn {avg:.1}), play {play:.1}s train {train:.1}s, ploss {ploss:.3} vloss {vloss:.3} lr {lr:.1e}",
                gen = state.generation,
                games = gen_completed_games,
                samples = gen_samples,
                buf = buffer_len,
                avg = avg_game_turn,
                play = play_seconds,
                train = train_seconds,
                ploss = losses.policy_loss,
                vloss = losses.value_loss,
            ));
            state.generation += 1;
            save_trainer_state(&paths.trainer_state, &state)?;
        }
        vs.save(&paths.net)?;
        save_trainer_state(&paths.trainer_state, &state)?;
        if let Err(err) = crate::session::save(&paths.session, &sp) {
            tracing::warn!(%err, "failed to save self-play session");
        }
        self.log(format!(
            "paused run '{run_id}' at gen {gen}: snapshot saved ({games} in-flight games, {fin} finished games, {done}/{target} samples this gen)",
            gen = state.generation,
            games = sp.boards.len(),
            fin = sp.finished.len(),
            done = sp.pending_sample_count(cfg.num_snakes),
            target = cfg.samples_per_gen,
        ));
        Ok(())
    }
}

/// One line of `runs/<id>/metrics.jsonl`: a complete per-generation summary. All
/// counts are for that generation only (not cumulative), except `samples_seen`.
#[derive(Serialize)]
struct GenRecord {
    generation: u32,
    policy_loss: f64,
    value_loss: f64,
    target_entropy: f64,
    /// Learning rate this generation actually trained at (after decay).
    lr: f64,
    win_rate: f64,
    completed_games: u32,
    samples: u32,
    turns: u32,
    buffer: u64,
    samples_seen: u64,
    gen_seconds: f64,
    play_seconds: f64,
    train_seconds: f64,
    inferences: u64,
    inferences_per_sec: f64,
    games_per_sec: f64,
    turns_per_sec: f64,
    gpu_busy_pct: f64,
    avg_game_turn: f64,
}

fn safe_div(a: f64, b: f64) -> f64 {
    if b > 0.0 {
        a / b
    } else {
        0.0
    }
}

fn append_metric(path: &Path, record: &GenRecord) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string(record)?)?;
    Ok(())
}

fn timestamp_run_id() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

fn configure_search_threads(cfg: &mut RunConfig) {
    if cfg.search_threads == 0 {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        cfg.search_threads = cpus.saturating_sub(2).max(1);
    }
    match rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.search_threads)
        .build_global()
    {
        Ok(()) => tracing::info!(
            search_threads = cfg.search_threads,
            "configured Rayon search pool"
        ),
        Err(err) => tracing::debug!(?err, "Rayon search pool was already configured"),
    }
}
