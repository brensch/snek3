use crate::proto::{Phase, StatsFrame};
use serde::Serialize;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

/// A human-readable trainer event, streamed to the terminal and the frontend.
#[derive(Clone, Debug, Serialize)]
pub struct LogEntry {
    pub t_unix_ms: u64,
    pub message: String,
}

#[derive(Default)]
pub struct Counters {
    pub generation: AtomicU32,
    pub phase: AtomicU32,
    pub samples_collected: AtomicU32,
    pub samples_target: AtomicU32,
    pub train_step: AtomicU32,
    pub train_steps_total: AtomicU32,
    pub completed_games: AtomicU64,
    /// Total turns across all completed games (paired with `completed_games`
    /// to derive a realtime mean game length).
    pub completed_turns: AtomicU64,
    pub inferences: AtomicU64,
    pub gpu_forward_us: AtomicU64,
    pub gpu_requests: AtomicU64,
    pub gpu_rows: AtomicU64,
    pub policy_loss_bits: AtomicU64,
    pub value_loss_bits: AtomicU64,
    pub target_entropy_bits: AtomicU64,
}

impl Counters {
    pub fn set_phase(&self, phase: Phase) {
        self.phase.store(phase as u32, Ordering::Relaxed);
    }

    pub fn set_losses(&self, policy: f64, value: f64, entropy: f64) {
        self.policy_loss_bits
            .store(policy.to_bits(), Ordering::Relaxed);
        self.value_loss_bits
            .store(value.to_bits(), Ordering::Relaxed);
        self.target_entropy_bits
            .store(entropy.to_bits(), Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct Metrics {
    counters: Arc<Counters>,
    stats_tx: broadcast::Sender<StatsFrame>,
    log_tx: broadcast::Sender<LogEntry>,
}

impl Metrics {
    pub fn new() -> Self {
        let (stats_tx, _) = broadcast::channel(256);
        let (log_tx, _) = broadcast::channel(256);
        Self {
            counters: Arc::new(Counters::default()),
            stats_tx,
            log_tx,
        }
    }

    pub fn counters(&self) -> Arc<Counters> {
        Arc::clone(&self.counters)
    }

    pub fn stats_rx(&self) -> broadcast::Receiver<StatsFrame> {
        self.stats_tx.subscribe()
    }

    pub fn log_rx(&self) -> broadcast::Receiver<LogEntry> {
        self.log_tx.subscribe()
    }

    /// Emit a trainer event: logged to the terminal and streamed to any frontend.
    pub fn log(&self, message: impl Into<String>) {
        let message = message.into();
        tracing::info!(target: "snek", "{message}");
        let _ = self.log_tx.send(LogEntry {
            t_unix_ms: unix_ms(),
            message,
        });
    }

    pub fn set_phase(&self, phase: Phase) {
        self.counters.set_phase(phase);
    }

    pub async fn run_samplers(self) {
        let mut last_at = Instant::now();
        let mut last_inf = 0u64;
        let mut last_games = 0u64;
        let mut last_turns = 0u64;
        let mut last_fwd_us = 0u64;
        let mut last_rows = 0u64;
        let mut last_reqs = 0u64;
        let mut inf_rate_ema = 0.0;
        let mut games_rate_ema = 0.0;
        let mut gpu_busy_ema = 0.0;
        let mut avg_turn_ema = 0.0;
        let mut initialized = false;
        let mut stats_tick = tokio::time::interval(Duration::from_millis(250));
        loop {
            stats_tick.tick().await;
            let now = Instant::now();
            let dt = now.duration_since(last_at).as_secs_f64().max(1e-9);
            let inf = self.counters.inferences.load(Ordering::Relaxed);
            let games = self.counters.completed_games.load(Ordering::Relaxed);
            let turns = self.counters.completed_turns.load(Ordering::Relaxed);
            let fwd_us = self.counters.gpu_forward_us.load(Ordering::Relaxed);
            let rows = self.counters.gpu_rows.load(Ordering::Relaxed);
            let reqs = self.counters.gpu_requests.load(Ordering::Relaxed);
            let raw_inf_rate = (inf - last_inf) as f64 / dt;
            let raw_games_rate = (games - last_games) as f64 / dt;
            let fwd_delta_us = fwd_us.saturating_sub(last_fwd_us);
            let raw_gpu_busy = (100.0 * fwd_delta_us as f64 / (dt * 1_000_000.0)).clamp(0.0, 100.0);
            // Mean length of games completing this window; only updated when
            // games actually finished, so it holds steady through training.
            let games_delta = games.saturating_sub(last_games);
            let raw_avg_turn = if games_delta > 0 {
                turns.saturating_sub(last_turns) as f64 / games_delta as f64
            } else {
                avg_turn_ema
            };
            let alpha = 0.35;
            if initialized {
                inf_rate_ema = alpha * raw_inf_rate + (1.0 - alpha) * inf_rate_ema;
                games_rate_ema = alpha * raw_games_rate + (1.0 - alpha) * games_rate_ema;
                gpu_busy_ema = alpha * raw_gpu_busy + (1.0 - alpha) * gpu_busy_ema;
                avg_turn_ema = alpha * raw_avg_turn + (1.0 - alpha) * avg_turn_ema;
            } else {
                inf_rate_ema = raw_inf_rate;
                games_rate_ema = raw_games_rate;
                gpu_busy_ema = raw_gpu_busy;
                avg_turn_ema = raw_avg_turn;
                initialized = true;
            }
            let req_delta = reqs.saturating_sub(last_reqs).max(1);
            let row_delta = rows.saturating_sub(last_rows);
            let gpu_rows_per_sec = if fwd_delta_us > 0 {
                row_delta as f64 / (fwd_delta_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            let frame = StatsFrame {
                t_unix_ms: unix_ms(),
                generation: self.counters.generation.load(Ordering::Relaxed),
                phase: phase_from_u32(self.counters.phase.load(Ordering::Relaxed)) as i32,
                inferences_per_sec: inf_rate_ema,
                games_per_sec: games_rate_ema,
                completed_games_total: games,
                samples_collected: self.counters.samples_collected.load(Ordering::Relaxed),
                samples_target: self.counters.samples_target.load(Ordering::Relaxed),
                train_step: self.counters.train_step.load(Ordering::Relaxed),
                train_steps_total: self.counters.train_steps_total.load(Ordering::Relaxed),
                gpu_busy_pct: gpu_busy_ema,
                batch_avg_rows: (row_delta / req_delta) as u32,
                policy_loss: f64::from_bits(self.counters.policy_loss_bits.load(Ordering::Relaxed)),
                value_loss: f64::from_bits(self.counters.value_loss_bits.load(Ordering::Relaxed)),
                target_entropy: f64::from_bits(
                    self.counters.target_entropy_bits.load(Ordering::Relaxed),
                ),
                gpu_rows_per_sec,
                avg_game_turn: avg_turn_ema,
            };
            last_at = now;
            last_inf = inf;
            last_games = games;
            last_turns = turns;
            last_fwd_us = fwd_us;
            last_rows = rows;
            last_reqs = reqs;
            let _ = self.stats_tx.send(frame);
        }
    }
}

pub fn phase_from_u32(v: u32) -> Phase {
    match v {
        1 => Phase::Playing,
        2 => Phase::Training,
        3 => Phase::Checkpoint,
        4 => Phase::Stopping,
        5 => Phase::Stopped,
        _ => Phase::Idle,
    }
}

pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
