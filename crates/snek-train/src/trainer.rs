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
        // cuDNN benchmark autotunes per input shape — a big win for the AZ
        // worker's fixed batch shapes, but CATASTROPHIC for LE self-play: its
        // batch (live-game leaves × seats) changes every single turn, so
        // benchmark re-ran a ~1.7s algorithm search on virtually every forward
        // (~1746ms/turn measured). Forcing it off in LE mode drops the forward to
        // ~112ms/turn (15× faster gens). Non-LE runs keep it on (main.rs default).
        if cfg.le_mode {
            // Chunked forward (le_fwd_chunk>0) makes every forward the SAME
            // fixed shape, so benchmark autotunes once and stays valid — turn it
            // back ON. Otherwise the batch shape varies every turn, so keep it
            // off (benchmark would re-run its ~1.7s algo search per forward).
            let bench = cfg.le_fwd_chunk > 0;
            tch::Cuda::cudnn_set_benchmark(bench);
            self.log(format!(
                "LE mode: cuDNN benchmark {} ({})",
                if bench { "ON" } else { "OFF" },
                if bench {
                    "fixed chunked forward shape"
                } else {
                    "variable per-turn batch shape"
                }
            ));
        }
        // Logit-Equilibrium mode conditions on τ via one extra input plane
        // (NUM_CHANNELS_TEMP = 15); the AZ path stays at NUM_CHANNELS (14).
        let in_ch = if cfg.le_mode {
            snek_core::NUM_CHANNELS_TEMP
        } else {
            snek_core::NUM_CHANNELS
        } as i64;
        let net = snek_tch::AZNet::new(&vs.root(), in_ch, cfg.trunk_channels, cfg.trunk_blocks);
        if !fresh && paths.net.exists() {
            vs.load(&paths.net)?;
        } else {
            snek_tch::init_orthogonal(&vs, 2f64.sqrt());
        }
        let mut opt = build_optimizer(&vs, &cfg)?;
        let mut state = if fresh {
            Default::default()
        } else {
            load_trainer_state(&paths.trainer_state)?
        };
        // Gating (LE mode): the INCUMBENT is a second, frozen net that
        // generates all self-play data and is what serving publishes; the live
        // net (`vs`/`net`) is the candidate, a pure learner between gates.
        // Promotion is the only way weights flow candidate -> incumbent. See
        // `gate` module docs for why (regressed nets poisoning their own data).
        let gating = cfg.le_mode && cfg.gate_gens > 0;
        let mut vs_inc = nn::VarStore::new(device);
        let net_inc =
            snek_tch::AZNet::new(&vs_inc.root(), in_ch, cfg.trunk_channels, cfg.trunk_blocks);
        let mut gate_meta = if gating && !fresh && paths.incumbent.exists() {
            vs_inc.load(&paths.incumbent)?;
            crate::gate::GateMeta::load(&paths.gate).unwrap_or_else(|| {
                crate::gate::GateMeta::founding(state.generation, state.seed ^ 0x6A7E)
            })
        } else {
            // Founding incumbent = the live weights (fresh init or resume of a
            // pre-gating run). Persist immediately so serving/resume see it.
            vs_inc.copy(&vs)?;
            let meta = crate::gate::GateMeta::founding(state.generation, state.seed ^ 0x6A7E);
            if gating {
                vs_inc.save(&paths.incumbent)?;
                meta.save(&paths.gate)?;
            }
            meta
        };
        if gating {
            self.log(format!(
                "gating ON: incumbent gen {} ({} promotions), gate every {} gens ({} games/side vs voronoi-{}), probe every {} gens (voronoi-{})",
                gate_meta.incumbent_gen, gate_meta.promotions, cfg.gate_gens, cfg.gate_games,
                cfg.gate_sims, cfg.probe_gens, cfg.probe_sims,
            ));
        }
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
        // Bradley–Terry Elo ratings in runs/<id>/eval/. Stops when we stop. The
        // returned context is shared with the GPU burst arena below so burst
        // games land in the same records.
        // The AZ checkpoint-vs-checkpoint league/burst serve 14-ch nets through the
        // MCTS server, so they are disabled in LE mode (15-ch nets + LE search);
        // LE gets its own held-out equilibrium eval (see instrumentation).
        let league = if cfg.le_mode {
            None
        } else {
            Some(crate::eval::start_league(
                paths.clone(),
                self.clone(),
                self.stop.clone(),
            ))
        };
        // The GPU burst arena's carried state: in-flight cross-generation
        // games persist across bursts (and generations), so each burst
        // resumes a saturated buffer instead of cold-starting.
        let mut arena = crate::eval::burst::ArenaState::default();

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
            // Under gating the frozen incumbent generates the data; the live
            // net only ever sees it through the training loss.
            let sp_net = SelfPlayNet { net: if gating { &net_inc } else { &net }, device };
            let seed = state.seed + state.generation as u64;
            let outcome = if cfg.le_mode {
                crate::selfplay::generate_le(&sp_net, &cfg, seed, &self.metrics, &self.stop, &mut sp)?
            } else {
                generate(&sp_net, &cfg, seed, &self.metrics, &self.stop, &mut sp)?
            };
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
            // The LR schedule's shape is code-owned (see `train::lr_for`);
            // re-applied every generation so the decay advances with
            // samples_seen.
            let lr = crate::train::lr_for(&cfg, state.samples_seen);
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
            // Serving publishes the net we'd actually deploy: the incumbent
            // when gating (promotion-guarded), else the live net.
            let serving_src = if gating { &paths.incumbent } else { &paths.net };
            if let Err(err) = publish_serving(serving_src, run_id, state.generation, &cfg) {
                self.log(format!("serving publish failed: {err:#}"));
            }
            // GATE (LE mode): every `gate_gens`, the candidate and the
            // incumbent play the same start seeds vs voronoi-`gate_sims`; the
            // candidate is promoted to data generator + serving only if it
            // wins strictly more. Also a floodfill sanity line, and every
            // `probe_gens` the incumbent faces voronoi-`probe_sims` — the
            // "super-heuristic" goal line. All rates land in the metrics row.
            let mut le_ff_winrate = None;
            let mut le_vor_winrate = None;
            let mut le_vor_incumbent = None;
            let mut le_vor_probe = None;
            let mut gate_promoted = None;
            let mut le_h2h_share = None;
            if gating
                && (state.generation as usize).is_multiple_of(cfg.gate_gens)
                && !self.stop.load(Ordering::Relaxed)
            {
                self.metrics.set_phase(crate::proto::Phase::Arena);
                let counters = self.metrics.counters();
                let ff_games = 16usize;
                let probe_due = cfg.probe_gens > 0
                    && (state.generation as usize).is_multiple_of(cfg.probe_gens);
                let probe_games = if probe_due { 32 } else { 0 };
                counters.arena_target.store(
                    (2 * cfg.gate_games.max(8) + ff_games + probe_games) as u32,
                    Ordering::Relaxed,
                );
                counters.arena_done.store(0, Ordering::Relaxed);
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0);

                // Floodfill sanity (greedy, cheap): the "is it not broken" line.
                let ff = snek_heuristic::Baseline::parse("floodfill").unwrap();
                let (vff, rec_ff) = crate::le_eval::eval_le_vs_baseline(
                    &net, device, &cfg, ff, 0, ff_games, cfg.response_tau,
                    state.seed ^ 0xE3E3, Some(&counters.arena_done),
                    state.generation, crate::eval::HEURISTIC_GEN, 2,
                );
                le_ff_winrate = Some(vff);
                if let Err(e) = crate::le_eval::write_eval_games(&paths.root, 0, now_ms, &rec_ff) {
                    self.log(format!("gate: failed to record floodfill games: {e}"));
                }

                let report = if cfg.gate_h2h {
                    crate::gate::run_gate_h2h(
                        &net, &net_inc, device, &cfg, &mut gate_meta, state.generation,
                        Some(&counters.arena_done), 4,
                    )
                } else {
                    crate::gate::run_gate(
                        &net, &net_inc, device, &cfg, &mut gate_meta, state.generation,
                        Some(&counters.arena_done), 4,
                    )
                };
                le_vor_winrate = Some(report.candidate_vor);
                le_vor_incumbent = Some(report.incumbent_vor);
                gate_promoted = Some(report.promoted);
                le_h2h_share = report.h2h_share;
                if report.promoted {
                    vs_inc.copy(&vs)?;
                    vs_inc.save(&paths.incumbent)?;
                    if let Err(err) =
                        publish_serving(&paths.incumbent, run_id, state.generation, &cfg)
                    {
                        self.log(format!("serving publish failed: {err:#}"));
                    }
                }
                if let Err(e) = gate_meta.save(&paths.gate) {
                    self.log(format!("gate: failed to save gate.json: {e}"));
                }
                if let Err(e) = crate::le_eval::write_eval_games(
                    &paths.root,
                    cfg.gate_sims as u32,
                    now_ms,
                    &report.recorded,
                ) {
                    self.log(format!("gate: failed to record gate games: {e}"));
                }
                self.log(format!(
                    "GATE gen {gen}: candidate {cw} vs incumbent(gen {ig}) {iw} ({mode}, {n} games) → {verdict} · cand-vor {cv:.0}% · ff {ff:.0}%",
                    gen = state.generation,
                    cw = report.candidate_wins,
                    iw = report.incumbent_wins,
                    n = report.games,
                    ig = if report.promoted { state.generation } else { gate_meta.incumbent_gen },
                    mode = if cfg.gate_h2h { "head-to-head" } else { "vs-baseline" },
                    verdict = if report.promoted { "PROMOTED" } else { "kept incumbent" },
                    cv = report.candidate_vor * 100.0,
                    ff = vff * 100.0,
                ));

                if probe_due {
                    let (probe, rec_probe) = crate::gate::run_probe(
                        &net_inc, device, &cfg, &gate_meta, probe_games,
                        state.seed ^ 0xF00D ^ state.generation as u64,
                        Some(&counters.arena_done),
                    );
                    le_vor_probe = Some(probe);
                    if let Err(e) = crate::le_eval::write_eval_games(
                        &paths.root,
                        cfg.probe_sims as u32,
                        now_ms,
                        &rec_probe,
                    ) {
                        self.log(format!("gate: failed to record probe games: {e}"));
                    }
                    self.log(format!(
                        "PROBE gen {gen}: incumbent(gen {ig}) vs voronoi-{sims}: {p:.0}%",
                        gen = state.generation,
                        ig = gate_meta.incumbent_gen,
                        sims = cfg.probe_sims,
                        p = probe * 100.0,
                    ));
                }
                counters.arena_target.store(0, Ordering::Relaxed);
                counters.arena_done.store(0, Ordering::Relaxed);
            }
            append_metric(
                &paths.metrics,
                &GenRecord {
                    generation: state.generation,
                    policy_loss: losses.policy_loss,
                    value_loss: losses.value_loss,
                    target_entropy: losses.target_entropy,
                    net_entropy: losses.net_entropy,
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
                    // True device utilization: mean of the NVML samples taken
                    // since the last gen record (the metrics sampler feeds
                    // gpu_util_sum/n every 250ms).
                    gpu_busy_pct: {
                        let sum = counters.gpu_util_sum.swap(0, Ordering::Relaxed);
                        let nsm = counters.gpu_util_n.swap(0, Ordering::Relaxed);
                        safe_div(sum as f64, nsm as f64)
                    },
                    avg_game_turn,
                    le_ff_winrate,
                    le_vor_winrate,
                    le_vor_incumbent,
                    le_vor_probe,
                    gate_promoted,
                    le_h2h_share,
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
            // GPU burst arena: the generation whose checkpoint just landed is a
            // new league entrant every `league_entrant_gens` — spend one cycle
            // rating it (and back-filling the pool) at self-play throughput
            // before the next generation starts.
            if !cfg.le_mode
                && cfg.burst_games > 0
                && cfg.league_entrant_gens > 0
                && (state.generation as usize).is_multiple_of(cfg.league_entrant_gens)
                && !self.stop.load(Ordering::Relaxed)
            {
                self.metrics.set_phase(crate::proto::Phase::Arena);
                let eval_dir = paths.root.join("eval");
                // A burst must never take training down: on panic, drop the
                // carried arena (it may be half-thawed) and move on — the next
                // entrant gets a fresh buffer.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::eval::burst::run_burst(
                        league.as_ref().expect("league present when not le_mode"),
                        &cfg,
                        device,
                        &self.metrics,
                        &self.stop,
                        &eval_dir,
                        &mut arena,
                    )
                }))
                .unwrap_or_else(|_| {
                    arena = Default::default();
                    Err(anyhow::anyhow!("burst panicked; arena state discarded"))
                });
                match result {
                    Ok(report) => self.log(format!(
                        "arena burst gen {gen}: {games} games @{sims} sims in {secs:.0}s ({gpm:.0} games/min, {inf:.0}k inf/s)",
                        gen = state.generation,
                        games = report.games,
                        sims = if cfg.burst_sims > 0 { cfg.burst_sims } else { cfg.eval_sims },
                        secs = report.seconds,
                        gpm = 60.0 * safe_div(report.games as f64, report.seconds),
                        inf = safe_div(report.inferences as f64, report.seconds) / 1000.0,
                    )),
                    Err(err) => self.log(format!(
                        "arena burst gen {gen} failed: {err}",
                        gen = state.generation
                    )),
                }
            }

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
    /// Entropy of the net's own policy output (nats), avg over the last train
    /// batch. Compare to `target_entropy`.
    net_entropy: f64,
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
    /// Held-out LE strength (win rate 0..1) vs heuristics the net never trains
    /// against — the "is it actually getting stronger" signal. Only present on
    /// gate gens (every `gate_gens`); omitted otherwise so the dashboard plots
    /// a point per measurement instead of a flat zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    le_ff_winrate: Option<f64>,
    /// Candidate vs voronoi-`gate_sims` (the gate match).
    #[serde(skip_serializing_if = "Option::is_none")]
    le_vor_winrate: Option<f64>,
    /// Incumbent's same-seed rate in the same gate match.
    #[serde(skip_serializing_if = "Option::is_none")]
    le_vor_incumbent: Option<f64>,
    /// Incumbent vs voronoi-`probe_sims` — the super-heuristic goal line.
    #[serde(skip_serializing_if = "Option::is_none")]
    le_vor_probe: Option<f64>,
    /// Whether this gate promoted the candidate to incumbent.
    #[serde(skip_serializing_if = "Option::is_none")]
    gate_promoted: Option<bool>,
    /// Candidate's share of decisive head-to-head gate games (0.5 = parity).
    #[serde(skip_serializing_if = "Option::is_none")]
    le_h2h_share: Option<f64>,
}

/// Copy the current net to `checkpoints/serving.safetensors` (+ provenance
/// json) and `git add/commit/push` the pair, keeping the repo's tracked
/// serving checkpoint — and therefore the snek3-api container — current with
/// the newest weights. Runs from the process cwd, which is the repo root
/// under `make train`. Set SNEK_NO_PUBLISH=1 to disable (e.g. on pods with
/// no git credentials); a missing checkpoints/ dir also skips quietly.
fn publish_serving(
    net: &Path,
    run_id: &str,
    generation: u32,
    cfg: &RunConfig,
) -> anyhow::Result<()> {
    if std::env::var_os("SNEK_NO_PUBLISH").is_some() {
        return Ok(());
    }
    let dir = Path::new("checkpoints");
    if !dir.is_dir() {
        return Ok(()); // not running from a repo checkout — nothing to publish
    }
    std::fs::copy(net, dir.join("serving.safetensors"))?;
    let exported_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let provenance = serde_json::json!({
        "source_run": run_id,
        "generation": generation,
        "exported_unix": exported_unix,
        "trunk_channels": cfg.trunk_channels,
        "trunk_blocks": cfg.trunk_blocks,
        "note": "auto-committed by snek-train at checkpoint save; must ship with matching crates/snek-tch",
    });
    std::fs::write(dir.join("serving.json"), serde_json::to_vec_pretty(&provenance)?)?;
    let files = ["checkpoints/serving.safetensors", "checkpoints/serving.json"];
    let msg = format!("serve: {run_id} gen {generation}");
    for args in [
        vec!["add", "--", files[0], files[1]],
        vec!["commit", "-q", "-m", &msg, "--", files[0], files[1]],
        vec!["push", "-q"],
    ] {
        let out = std::process::Command::new("git").args(&args).output()?;
        if !out.status.success() {
            anyhow::bail!(
                "git {}: {}",
                args[0],
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
    }
    Ok(())
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
