use crate::proto::{GameSnapshot, GamesSnapshot, Phase, Point, Snake, StatsFrame};
use snek_core::Board;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

#[derive(Default)]
pub struct Counters {
    pub generation: AtomicU32,
    pub phase: AtomicU32,
    pub samples_collected: AtomicU32,
    pub samples_target: AtomicU32,
    pub completed_games: AtomicU64,
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
    games: Arc<Mutex<Vec<(usize, Board)>>>,
    stats_tx: broadcast::Sender<StatsFrame>,
    games_tx: broadcast::Sender<GamesSnapshot>,
}

impl Metrics {
    pub fn new() -> Self {
        let (stats_tx, _) = broadcast::channel(256);
        let (games_tx, _) = broadcast::channel(64);
        Self {
            counters: Arc::new(Counters::default()),
            games: Arc::new(Mutex::new(Vec::new())),
            stats_tx,
            games_tx,
        }
    }

    pub fn counters(&self) -> Arc<Counters> {
        Arc::clone(&self.counters)
    }

    pub fn stats_rx(&self) -> broadcast::Receiver<StatsFrame> {
        self.stats_tx.subscribe()
    }

    pub fn games_rx(&self) -> broadcast::Receiver<GamesSnapshot> {
        self.games_tx.subscribe()
    }

    pub fn set_phase(&self, phase: Phase) {
        self.counters.set_phase(phase);
    }

    pub fn replace_games(&self, boards: &[(usize, Board)]) {
        if self.games_tx.receiver_count() == 0 {
            return;
        }
        if let Ok(mut dst) = self.games.lock() {
            dst.clear();
            dst.extend_from_slice(boards);
        }
    }

    pub async fn run_samplers(self) {
        let mut last_at = Instant::now();
        let mut last_inf = 0u64;
        let mut last_games = 0u64;
        let mut last_fwd_us = 0u64;
        let mut last_rows = 0u64;
        let mut last_reqs = 0u64;
        let mut inf_rate_ema = 0.0;
        let mut games_rate_ema = 0.0;
        let mut gpu_busy_ema = 0.0;
        let mut initialized = false;
        let mut stats_tick = tokio::time::interval(Duration::from_millis(250));
        let mut games_tick = tokio::time::interval(Duration::from_millis(500));
        loop {
            tokio::select! {
                _ = stats_tick.tick() => {
                    let now = Instant::now();
                    let dt = now.duration_since(last_at).as_secs_f64().max(1e-9);
                    let inf = self.counters.inferences.load(Ordering::Relaxed);
                    let games = self.counters.completed_games.load(Ordering::Relaxed);
                    let fwd_us = self.counters.gpu_forward_us.load(Ordering::Relaxed);
                    let rows = self.counters.gpu_rows.load(Ordering::Relaxed);
                    let reqs = self.counters.gpu_requests.load(Ordering::Relaxed);
                    let raw_inf_rate = (inf - last_inf) as f64 / dt;
                    let raw_games_rate = (games - last_games) as f64 / dt;
                    let fwd_delta_us = fwd_us.saturating_sub(last_fwd_us);
                    let raw_gpu_busy = (100.0 * fwd_delta_us as f64 / (dt * 1_000_000.0)).clamp(0.0, 100.0);
                    let alpha = 0.35;
                    if initialized {
                        inf_rate_ema = alpha * raw_inf_rate + (1.0 - alpha) * inf_rate_ema;
                        games_rate_ema = alpha * raw_games_rate + (1.0 - alpha) * games_rate_ema;
                        gpu_busy_ema = alpha * raw_gpu_busy + (1.0 - alpha) * gpu_busy_ema;
                    } else {
                        inf_rate_ema = raw_inf_rate;
                        games_rate_ema = raw_games_rate;
                        gpu_busy_ema = raw_gpu_busy;
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
                        gpu_busy_pct: gpu_busy_ema,
                        batch_avg_rows: (row_delta / req_delta) as u32,
                        policy_loss: f64::from_bits(self.counters.policy_loss_bits.load(Ordering::Relaxed)),
                        value_loss: f64::from_bits(self.counters.value_loss_bits.load(Ordering::Relaxed)),
                        target_entropy: f64::from_bits(self.counters.target_entropy_bits.load(Ordering::Relaxed)),
                        gpu_rows_per_sec,
                    };
                    last_at = now;
                    last_inf = inf;
                    last_games = games;
                    last_fwd_us = fwd_us;
                    last_rows = rows;
                    last_reqs = reqs;
                    let _ = self.stats_tx.send(frame);
                }
                _ = games_tick.tick() => {
                    if self.games_tx.receiver_count() == 0 {
                        continue;
                    }
                    let snapshot = self.snapshot_games();
                    let _ = self.games_tx.send(snapshot);
                }
            }
        }
    }

    fn snapshot_games(&self) -> GamesSnapshot {
        let games = self.games.lock().map(|g| g.clone()).unwrap_or_default();
        GamesSnapshot {
            t_unix_ms: unix_ms(),
            games: games
                .into_iter()
                .map(|(turn, b)| GameSnapshot {
                    id: 0,
                    turn: turn as u32,
                    board_w: b.width as u32,
                    board_h: b.height as u32,
                    snakes: b
                        .snakes
                        .iter()
                        .map(|s| Snake {
                            alive: s.alive(),
                            health: s.health.max(0) as u32,
                            body: s
                                .body
                                .iter()
                                .map(|p| Point {
                                    x: p.x.max(0) as u32,
                                    y: p.y.max(0) as u32,
                                })
                                .collect(),
                        })
                        .collect(),
                    food: b
                        .food
                        .iter()
                        .map(|p| Point {
                            x: p.x.max(0) as u32,
                            y: p.y.max(0) as u32,
                        })
                        .collect(),
                })
                .collect(),
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
