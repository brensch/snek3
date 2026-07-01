use crate::config::RunConfig;
use crate::metrics::Metrics;
use crate::proto::{GenerationSummary, Phase};
use crate::replay::ReplayBuffer;
use crate::selfplay::{generate, SelfPlayNet};
use crate::state::{load_trainer_state, save_trainer_state, RunPaths};
use crate::train::{build_optimizer, train_steps};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tch::{nn, Device};

/// How many generations of recorded sample games to retain on disk per run.
const SAMPLE_GAMES_KEEP: usize = 60;

#[derive(Clone)]
pub struct TrainerHandle {
    runs_dir: PathBuf,
    metrics: Metrics,
    config: Arc<Mutex<RunConfig>>,
    active_run: Arc<Mutex<Option<String>>>,
    running: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    cuda_active: Arc<AtomicBool>,
}

#[derive(Debug, Deserialize)]
pub struct StartRequest {
    pub run_id: Option<String>,
    pub fresh: Option<bool>,
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
        if self.running.swap(true, Ordering::SeqCst) {
            return self
                .active_run
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| anyhow::anyhow!("trainer is running without active run"));
        }
        self.stop.store(false, Ordering::SeqCst);
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
            cfg.gpool_every,
        );
        if !fresh && paths.net.exists() {
            vs.load(&paths.net)?;
        } else {
            snek_tch::init_orthogonal(&vs, 2f64.sqrt());
        }
        let mut opt = build_optimizer(&vs, cfg.lr)?;
        let mut state = if fresh {
            Default::default()
        } else {
            load_trainer_state(&paths.trainer_state)?
        };
        let mut replay = if fresh {
            ReplayBuffer::new(cfg.buffer_size)
        } else {
            ReplayBuffer::restore(&paths.replay, cfg.buffer_size)?
        };

        while !self.stop.load(Ordering::Relaxed) {
            let cfg = self.config();
            cfg.save_atomic(&paths.config)?;
            self.metrics
                .counters()
                .generation
                .store(state.generation, Ordering::Relaxed);
            self.metrics.set_phase(Phase::Playing);
            let (samples, recorded_games) = generate(
                &SelfPlayNet { net: &net, device },
                &cfg,
                state.seed + state.generation as u64,
                &self.metrics,
                &self.stop,
            )?;
            // Stop requested mid-generation: bail out now rather than spending a
            // full train + checkpoint cycle on this interrupted generation. The
            // net, trainer state, and replay shards from completed generations are
            // already on disk / saved below, so resume loses only this partial gen.
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            if !recorded_games.is_empty() {
                if let Err(err) = crate::sample::write_generation(
                    &paths.games,
                    state.generation,
                    recorded_games,
                    SAMPLE_GAMES_KEEP,
                ) {
                    tracing::warn!(%err, "failed to write sample games");
                }
            }
            replay.save_shard(&paths.replay, state.generation, &samples)?;
            state.samples_seen += samples.len() as u64;
            replay.add(samples);

            self.metrics.set_phase(Phase::Training);
            let losses = train_steps(
                &net,
                &vs,
                &mut opt,
                &replay,
                &cfg,
                state.seed ^ state.generation as u64,
            )?;
            self.metrics.counters().set_losses(
                losses.policy_loss,
                losses.value_loss,
                losses.target_entropy,
            );

            self.metrics.set_phase(Phase::Checkpoint);
            vs.save(&paths.net)?;
            append_metric(
                &paths.metrics,
                &GenerationSummary {
                    generation: state.generation,
                    policy_loss: losses.policy_loss,
                    value_loss: losses.value_loss,
                    win_rate: 0.0,
                    completed_games: self
                        .metrics
                        .counters()
                        .completed_games
                        .load(Ordering::Relaxed) as u32,
                    seconds: 0.0,
                },
            )?;
            state.generation += 1;
            save_trainer_state(&paths.trainer_state, &state)?;
        }
        vs.save(&paths.net)?;
        save_trainer_state(&paths.trainer_state, &state)?;
        Ok(())
    }
}

fn append_metric(path: &Path, summary: &GenerationSummary) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(
        f,
        "{}",
        serde_json::json!({
            "generation": summary.generation,
            "policy_loss": summary.policy_loss,
            "value_loss": summary.value_loss,
            "win_rate": summary.win_rate,
            "completed_games": summary.completed_games,
            "seconds": summary.seconds,
        })
    )?;
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
