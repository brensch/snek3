//! In-process league games: N players (a net checkpoint, a built-in heuristic
//! agent, or an external Battlesnake-protocol HTTP server), the exact
//! `snek-core` rules engine, and the same `serve_move_until` search live
//! serving uses. [`play_game`] is a plain blocking function — the trainer's
//! league and the `arena` CLI both call it directly; there is no worker-pool
//! or IPC machinery. Within a turn every living snake's search runs
//! concurrently on scoped threads (seats sharing a player run sequentially,
//! since a search needs `&mut` its net).

use crate::{serve_move_until_diagnostics, Config, Net};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::Serialize;
use serde_json::json;
use snek_core::{standard_start, Board, Move};
use std::time::{Duration, Instant};

/// Per-move search budget: a fixed simulation count (deterministic,
/// hardware-independent — the league default) or a wall-clock deadline.
#[derive(Clone, Copy)]
pub enum Budget {
    Sims(usize),
    TimeMs(u64),
}

/// Stand-in "no deadline" horizon for fixed-sims mode (matches `serve_move`).
const SIMS_DEADLINE: Duration = Duration::from_secs(3600);

/// What a player spec string names: a checkpoint weights path, a built-in
/// heuristic agent token, or an `http(s)://` Battlesnake server.
pub enum PlayerSpec {
    Net(String),
    Baseline(snek_heuristic::Baseline),
    Http(String),
}

impl PlayerSpec {
    pub fn parse(spec: &str) -> Self {
        if let Some(kind) = snek_heuristic::Baseline::parse(spec) {
            PlayerSpec::Baseline(kind)
        } else if spec.starts_with("http://") || spec.starts_with("https://") {
            PlayerSpec::Http(spec.trim_end_matches('/').to_string())
        } else {
            PlayerSpec::Net(spec.to_string())
        }
    }

    /// Display name when the caller didn't provide one: file stem, agent
    /// name, or the server's host part.
    pub fn default_name(&self, index: usize) -> String {
        match self {
            PlayerSpec::Baseline(kind) => kind.display_name().to_string(),
            PlayerSpec::Http(url) => url
                .split("://")
                .nth(1)
                .unwrap_or(url)
                .trim_end_matches('/')
                .to_string(),
            PlayerSpec::Net(path) => std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("net-{index}")),
        }
    }
}

/// A loaded, ready-to-play seat holder. One `Player` can hold several seats
/// of the same game (small pools), but never plays two games at once.
pub enum Player {
    Net(Box<Net>),
    Baseline(snek_heuristic::Baseline),
    Http(HttpPlayer),
}

pub struct HttpPlayer {
    label: String,
    endpoint: String,
    agent: ureq::Agent,
    timeout_ms: u64,
    errors: u64,
}

impl Player {
    /// Load a player onto `device` (nets only; agents and HTTP players ignore
    /// it). `trunk_channels`/`trunk_blocks` must match how the checkpoint was
    /// trained.
    pub fn load(
        spec: &PlayerSpec,
        device: tch::Device,
        trunk_channels: i64,
        trunk_blocks: i64,
        budget: Budget,
        label: &str,
    ) -> anyhow::Result<Player> {
        Ok(match spec {
            PlayerSpec::Net(path) => Player::Net(Box::new(Net::load_on(
                path,
                device,
                trunk_channels,
                trunk_blocks,
            )?)),
            PlayerSpec::Baseline(kind) => Player::Baseline(*kind),
            PlayerSpec::Http(url) => {
                // Battlesnake's standard reply budget is 500ms; in time mode
                // the shared per-move budget binds instead.
                let timeout_ms = match budget {
                    Budget::Sims(_) => 500,
                    Budget::TimeMs(ms) => ms.max(1),
                };
                Player::Http(HttpPlayer {
                    label: label.to_string(),
                    endpoint: format!("{url}/move"),
                    agent: ureq::AgentBuilder::new()
                        .timeout(Duration::from_millis(timeout_ms))
                        .build(),
                    timeout_ms,
                    errors: 0,
                })
            }
        })
    }

    /// One move for seat `me` on `board`: a full search (nets), the agent's
    /// own search (baselines), or a `/move` round-trip (HTTP).
    fn decide(&mut self, cfg: &Config, budget: Budget, board: &Board, me: usize) -> Decision {
        let deadline = match budget {
            Budget::Sims(_) => Instant::now() + SIMS_DEADLINE,
            Budget::TimeMs(ms) => Instant::now() + Duration::from_millis(ms),
        };
        match self {
            Player::Net(net) => {
                let d = serve_move_until_diagnostics(net, cfg, board, me, deadline);
                let n = board.snakes.len();
                let mut policy = [0f32; 4];
                let mut value = 0f32;
                if d.diagnostics.root_policy.len() == n * 4 {
                    policy.copy_from_slice(&d.diagnostics.root_policy[me * 4..me * 4 + 4]);
                    value = d.diagnostics.root_values.get(me).copied().unwrap_or(0.0);
                } else {
                    // Forced move / fallback paths skip the search: record the
                    // decision as a one-hot policy with no value estimate.
                    policy[d.move_index] = 1.0;
                }
                Decision {
                    move_index: d.move_index,
                    policy,
                    value,
                }
            }
            Player::Baseline(kind) => {
                // Heuristic sims are ~100× cheaper than a net forward, so in
                // fixed-sims mode the agent keeps its own static budget
                // (~200ms/move, still deterministic) instead of the nets'
                // sims; in time mode the shared deadline binds like everyone.
                let hcfg = snek_heuristic::HeuristicConfig {
                    max_sims: match budget {
                        Budget::Sims(_) => snek_heuristic::HeuristicConfig::default().max_sims,
                        Budget::TimeMs(_) => usize::MAX,
                    },
                    draw_value: cfg.draw_value,
                    ..Default::default()
                };
                let d = snek_heuristic::baseline_move_until(*kind, &hcfg, board, me, deadline);
                Decision {
                    move_index: d.move_index,
                    policy: d.policy,
                    value: d.value,
                }
            }
            Player::Http(http) => http.decide(board, me),
        }
    }
}

impl HttpPlayer {
    /// POST the standard Battlesnake state, play whatever comes back. A failed
    /// or illegal answer falls back to a sane legal move so one flaky server
    /// can't wedge the game.
    fn decide(&mut self, board: &Board, me: usize) -> Decision {
        let state = battlesnake_state(board, me, self.timeout_ms);
        let answer = self
            .agent
            .post(&self.endpoint)
            .send_json(state)
            .map_err(|e| e.to_string())
            .and_then(|resp| {
                resp.into_json::<serde_json::Value>()
                    .map_err(|e| e.to_string())
            })
            .and_then(|v| {
                match v
                    .get("move")
                    .and_then(|m| m.as_str())
                    .map(str::to_ascii_lowercase)
                    .as_deref()
                {
                    Some("up") => Ok(Move::Up),
                    Some("down") => Ok(Move::Down),
                    Some("left") => Ok(Move::Left),
                    Some("right") => Ok(Move::Right),
                    other => Err(format!("bad move {other:?}")),
                }
            });
        let mv = answer.unwrap_or_else(|err| {
            self.errors += 1;
            // First few failures loudly, then sampled — a dead server would
            // otherwise write one line per turn.
            if self.errors <= 3 || self.errors.is_multiple_of(100) {
                eprintln!(
                    "arena: {}: {} failed ({} so far): {err}",
                    self.label, self.endpoint, self.errors
                );
            }
            snek_heuristic::candidates(board, me)[0]
        });
        let mut policy = [0f32; 4];
        policy[mv.index()] = 1.0;
        Decision {
            move_index: mv.index(),
            policy,
            value: 0.0,
        }
    }
}

/// One seat's answer for one position: the move plus the search readout.
pub struct Decision {
    pub move_index: usize,
    pub policy: [f32; 4],
    pub value: f32,
}

/// The fixed parameters of a league game.
pub struct GameSettings {
    pub seats: usize,
    pub board_size: i8,
    pub max_turns: u32,
    pub cfg: Config,
    pub budget: Budget,
}

/// One recorded board turn. Field-for-field the schema of one frame of the
/// trainer's `games/gen_NNNN.json` (`sample.rs` in `snek-train` is the
/// canonical definition — keep the two in sync), so league recordings render
/// through the exact same viewer as self-play samples.
#[derive(Serialize, Clone)]
pub struct Frame {
    pub turn: u32,
    pub width: i32,
    pub height: i32,
    pub food: Vec<[i32; 2]>,
    pub hazards: Vec<[i32; 2]>,
    pub snakes: Vec<FrameSnake>,
}

#[derive(Serialize, Clone)]
pub struct FrameSnake {
    pub alive: bool,
    pub body: Vec<[i32; 2]>,
    pub health: i32,
    pub chosen_move: u32,
    pub policy: Vec<f64>,
    pub play_policy: Vec<f64>,
    pub value: f64,
}

/// One seat's result: which player held it and where it finished. Rank 1 is
/// best; ties share a rank (simultaneous deaths, or survivors at the cutoff).
#[derive(Serialize, Clone)]
pub struct Placement {
    pub seat: u32,
    pub player: u32,
    pub rank: u32,
    pub death_turn: Option<u32>,
}

pub struct Outcome {
    pub turns: u32,
    pub wall_ms: u64,
    pub placements: Vec<Placement>,
    pub frames: Vec<Frame>,
}

/// Seat `s` plays `players[(s + rotation) % players.len()]`. A rotation per
/// game keeps fixed rosters (the CLI) free of seat bias; callers that shuffle
/// players per game pass 0.
pub fn player_for_seat(seat: usize, rotation: usize, n_players: usize) -> usize {
    (seat + rotation) % n_players
}

/// Play one game to completion (blocking). One worker thread per player lives
/// for the whole game; every turn, all living seats' searches run on their
/// players' workers concurrently (seats sharing a player run in sequence),
/// the pre-step frame goes to `on_turn`, then the board steps. Returns `None`
/// if `stop` turned true mid-game.
///
/// Workers are persistent (not per-turn) on purpose: libtorch's intra-op
/// thread count is per-thread state, and lazily-initialized helper pools
/// attach to whichever thread first runs a forward. Ephemeral per-turn
/// threads made every turn spawn a fresh idle-spinning OpenMP team — ~50
/// runaway threads per game slot, measured. Long-lived workers that pin their
/// own intra-op count to 1 keep the thread count at exactly one per player.
///
/// The game ends for rating purposes as soon as at most one *player* has a
/// living snake — the elimination order is complete at that point. Survivors
/// (including everyone at a max-turns cutoff) tie at the top.
pub fn play_game(
    players: &mut [Player],
    rotation: usize,
    settings: &GameSettings,
    seed: u64,
    stop: &(dyn Fn() -> bool + Sync),
    on_turn: &mut dyn FnMut(&Frame),
) -> Option<Outcome> {
    let start = Instant::now();
    let n_players = players.len();
    let seats = settings.seats;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut board = standard_start(settings.board_size, settings.board_size, seats, &mut rng);
    let mut frames = Vec::new();
    let mut death_turn: Vec<Option<u32>> = vec![None; seats];
    let living_players = |board: &Board| {
        let mut alive: Vec<usize> = board
            .snakes
            .iter()
            .enumerate()
            .filter(|(_, s)| s.alive())
            .map(|(i, _)| player_for_seat(i, rotation, n_players))
            .collect();
        alive.sort_unstable();
        alive.dedup();
        alive
    };

    std::thread::scope(|scope| {
        // One worker per player for the game's lifetime. Jobs are (board,
        // seat); answers funnel through one shared channel. Dropping the job
        // senders (scope exit, any path) ends the workers.
        let (done_tx, done_rx) = std::sync::mpsc::channel::<(usize, Decision)>();
        let mut job_txs: Vec<std::sync::mpsc::Sender<(Board, usize)>> =
            Vec::with_capacity(n_players);
        for (pi, player) in players.iter_mut().enumerate() {
            let (job_tx, job_rx) = std::sync::mpsc::channel::<(Board, usize)>();
            let done = done_tx.clone();
            let cfg = &settings.cfg;
            let budget = settings.budget;
            std::thread::Builder::new()
                .name(format!("seat-worker-{pi}"))
                .spawn_scoped(scope, move || {
                    // Intra-op parallelism is a per-thread setting; without
                    // this, the first forward on this thread spawns an
                    // idle-spinning OpenMP team sized to the machine.
                    tch::set_num_threads(1);
                    while let Ok((board, seat)) = job_rx.recv() {
                        let decision = player.decide(cfg, budget, &board, seat);
                        if done.send((seat, decision)).is_err() {
                            return;
                        }
                    }
                })
                .expect("spawn seat worker");
            job_txs.push(job_tx);
        }
        drop(done_tx);

        while board.turn < settings.max_turns && living_players(&board).len() > 1 {
            if stop() {
                return None;
            }
            let mut decisions: Vec<Option<Decision>> = (0..seats).map(|_| None).collect();
            let mut pending = 0usize;
            for (i, snake) in board.snakes.iter().enumerate() {
                if snake.alive() {
                    let worker = &job_txs[player_for_seat(i, rotation, n_players)];
                    worker
                        .send((board.clone(), i))
                        .expect("seat worker exited");
                    pending += 1;
                }
            }
            for _ in 0..pending {
                let (seat, decision) = done_rx.recv().expect("seat worker exited");
                decisions[seat] = Some(decision);
            }
            let mut moves = vec![Move::Up; seats];
            for (i, decision) in decisions.iter().enumerate() {
                if let Some(d) = decision {
                    moves[i] = Move::from_index(d.move_index);
                }
            }
            // The frame is the pre-step position with each snake's decision,
            // so a replay renders exactly what each player saw when it moved.
            let frame = frame_from(&board, &moves, &decisions);
            on_turn(&frame);
            frames.push(frame);
            board.step_and_spawn(&moves, &mut rng);
            for (i, snake) in board.snakes.iter().enumerate() {
                if !snake.alive() && death_turn[i].is_none() {
                    death_turn[i] = Some(board.turn);
                }
            }
        }
        // Competition ranking by elimination order: outlasting a snake beats it.
        let score = |i: usize| death_turn[i].unwrap_or(u32::MAX);
        let placements: Vec<Placement> = (0..seats)
            .map(|i| Placement {
                seat: i as u32,
                player: player_for_seat(i, rotation, n_players) as u32,
                rank: 1 + (0..seats).filter(|&j| score(j) > score(i)).count() as u32,
                death_turn: death_turn[i],
            })
            .collect();
        Some(Outcome {
            turns: board.turn,
            wall_ms: start.elapsed().as_millis() as u64,
            placements,
            frames,
        })
    })
}

fn frame_from(board: &Board, moves: &[Move], decisions: &[Option<Decision>]) -> Frame {
    let coord = |p: &snek_core::Point| [p.x as i32, p.y as i32];
    let snakes = (0..board.snakes.len())
        .map(|i| {
            let snake = &board.snakes[i];
            let (policy, play_policy, value) = match &decisions[i] {
                Some(d) => {
                    // League play is argmax (no sampling), so the play policy
                    // is the one-hot of the chosen move.
                    let mut one_hot = vec![0.0f64; 4];
                    one_hot[d.move_index] = 1.0;
                    (
                        d.policy.iter().map(|&v| v as f64).collect(),
                        one_hot,
                        d.value as f64,
                    )
                }
                None => (vec![0.0; 4], vec![0.0; 4], 0.0),
            };
            FrameSnake {
                alive: snake.alive(),
                body: snake.body.iter().map(|p| coord(&p)).collect(),
                health: snake.health as i32,
                chosen_move: moves[i].index() as u32,
                policy,
                play_policy,
                value,
            }
        })
        .collect();
    Frame {
        turn: board.turn,
        width: board.width as i32,
        height: board.height as i32,
        food: board.food.iter().map(coord).collect(),
        hazards: board.hazards.iter().map(coord).collect(),
        snakes,
    }
}

/// The standard Battlesnake /move request body for `me`'s view of `board`.
/// Dead snakes are omitted, exactly like the real engine. `snek-core` shares
/// Battlesnake's coordinate system (x right, y up, origin bottom-left), so
/// positions map verbatim. Only /move is called — /start and /end are skipped
/// because every request carries the full state.
fn battlesnake_state(board: &Board, me: usize, timeout_ms: u64) -> serde_json::Value {
    let point = |p: &snek_core::Point| json!({ "x": p.x as i32, "y": p.y as i32 });
    let snake = |i: usize| {
        let s = &board.snakes[i];
        let body: Vec<_> = s.body.iter().map(|p| point(&p)).collect();
        json!({
            "id": format!("seat-{i}"),
            "name": format!("seat-{i}"),
            "health": s.health as i32,
            "body": body,
            "head": body.first().cloned().unwrap_or_else(|| json!({"x": 0, "y": 0})),
            "length": s.len(),
            "latency": "0",
            "shout": "",
        })
    };
    let alive: Vec<_> = (0..board.snakes.len())
        .filter(|&i| board.snakes[i].alive())
        .map(snake)
        .collect();
    json!({
        "game": {
            "id": "snek3-arena",
            "ruleset": { "name": "standard", "version": "v1" },
            "map": "standard",
            "source": "custom",
            "timeout": timeout_ms,
        },
        "turn": board.turn,
        "board": {
            "width": board.width as i32,
            "height": board.height as i32,
            "food": board.food.iter().map(&point).collect::<Vec<_>>(),
            "hazards": board.hazards.iter().map(point).collect::<Vec<_>>(),
            "snakes": alive,
        },
        "you": snake(me),
    })
}
