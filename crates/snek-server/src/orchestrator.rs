//! In-process Battlesnake arena.
//!
//! This does not call the search layer directly. Each decision is made by
//! rebuilding an official Battlesnake `/move` request and passing it through the
//! same server path that HTTP serving uses (`compute_move`), including timeout
//! parsing, request parsing, search diagnostics, and recorder emission.

use std::path::PathBuf;
use std::sync::Arc;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde_json::json;
use snek_core::{standard_start, Board, Move, Point};

use super::{build_app, compute_move, handle_end, App, AppSettings};

pub struct ArenaConfig {
    games: usize,
    models: Vec<String>,
    names: Vec<String>,
    board: i8,
    snakes: usize,
    timeout_ms: u64,
    seed: u64,
    max_turns: u32,
    log_dir: Option<PathBuf>,
}

impl ArenaConfig {
    pub fn from_env(
        default_model: &str,
        log_dir: Option<PathBuf>,
        settings: &AppSettings,
    ) -> Option<Self> {
        let games = env_or("SNEK_ARENA_GAMES", 0usize);
        if games == 0 {
            return None;
        }
        let snakes = env_or("SNEK_ARENA_SNAKES", 2usize)
            .max(1)
            .min(snek_core::MAX_SNAKES);
        let models = split_env("SNEK_ARENA_MODELS")
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![default_model.to_string()]);
        let names = split_env("SNEK_ARENA_NAMES").unwrap_or_else(|| {
            models
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    PathBuf::from(m)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("model-{i}"))
                })
                .collect()
        });
        Some(
            ArenaConfig {
                games,
                models,
                names,
                board: env_or("SNEK_ARENA_BOARD", 11i8),
                snakes,
                timeout_ms: env_or("SNEK_ARENA_TIMEOUT_MS", 100u64),
                seed: env_or("SNEK_ARENA_SEED", 1u64),
                max_turns: env_or("SNEK_ARENA_MAX_TURNS", 500u32),
                log_dir,
            }
            .with_arena_deadline(settings),
        )
    }

    fn with_arena_deadline(self, settings: &AppSettings) -> Self {
        let margin = env_or(
            "SNEK_ARENA_DEADLINE_MARGIN_MS",
            settings
                .deadline_margin_ms
                .min(self.timeout_ms.saturating_sub(1)),
        );
        if margin != settings.deadline_margin_ms {
            eprintln!(
                "snek-server arena: using SNEK_ARENA_DEADLINE_MARGIN_MS={margin}; \
                 set SNEK_DEADLINE_MARGIN_MS={margin} for identical HTTP serving behavior"
            );
        }
        self
    }
}

pub fn run(cfg: ArenaConfig, mut settings: AppSettings) {
    settings.default_timeout_ms = cfg.timeout_ms;
    settings.deadline_margin_ms = env_or(
        "SNEK_ARENA_DEADLINE_MARGIN_MS",
        settings
            .deadline_margin_ms
            .min(cfg.timeout_ms.saturating_sub(1)),
    );
    if cfg.log_dir.is_none() {
        eprintln!("snek-server arena: SNEK_MOVE_LOG_DIR is empty; games will not be recorded");
    }
    eprintln!(
        "snek-server arena: games={} snakes={} board={} timeout_ms={} max_turns={} models={}",
        cfg.games,
        cfg.snakes,
        cfg.board,
        cfg.timeout_ms,
        cfg.max_turns,
        cfg.models.join(",")
    );

    let players: Vec<Player> = (0..cfg.snakes)
        .map(|i| {
            let model = cfg.models[i % cfg.models.len()].clone();
            let name = cfg
                .names
                .get(i % cfg.names.len().max(1))
                .cloned()
                .unwrap_or_else(|| format!("snake-{i}"));
            let app = build_app(model, cfg.log_dir.clone(), &settings);
            Player { app, name }
        })
        .collect();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(cfg.seed);
    for game_idx in 0..cfg.games {
        let mut board = standard_start(cfg.board, cfg.board, cfg.snakes, &mut rng);
        let base_id = format!("arena-{}-{game_idx:04}", cfg.seed);
        let mut turns = 0u32;
        while !board.is_terminal() && turns < cfg.max_turns {
            let mut moves = vec![Move::Up; board.snakes.len()];
            for i in 0..board.snakes.len() {
                if !board.snakes[i].alive() {
                    continue;
                }
                let game_id = participant_game_id(&base_id, i);
                let body = request_body(&board, &game_id, i, &players, cfg.timeout_ms);
                let mv = compute_move(&players[i].app, &body).unwrap_or(0);
                moves[i] = Move::from_index(mv);
            }
            board.step_and_spawn(&moves, &mut rng);
            turns = board.turn;
        }

        for i in 0..players.len() {
            let game_id = participant_game_id(&base_id, i);
            let body = request_body(&board, &game_id, i, &players, cfg.timeout_ms);
            handle_end(&players[i].app, &body);
        }
        let result = match board.winner() {
            Some(i) => format!("winner={}", players[i].name),
            None => "draw".to_string(),
        };
        eprintln!(
            "snek-server arena: game {}/{} {} turns={} logs={}",
            game_idx + 1,
            cfg.games,
            result,
            board.turn,
            cfg.log_dir
                .as_ref()
                .map(|d| d.display().to_string())
                .unwrap_or_else(|| "disabled".into())
        );
    }
}

struct Player {
    app: Arc<App>,
    name: String,
}

fn participant_game_id(base_id: &str, snake_idx: usize) -> String {
    format!("{base_id}-s{snake_idx}")
}

fn request_body(
    board: &Board,
    game_id: &str,
    you_idx: usize,
    players: &[Player],
    timeout_ms: u64,
) -> String {
    let snakes: Vec<_> = board
        .snakes
        .iter()
        .enumerate()
        .filter(|(_, s)| s.alive())
        .map(|(i, s)| {
            let body = points_json(s.body.iter().collect::<Vec<_>>().as_slice());
            let head = s.head();
            json!({
                "id": snake_id(i),
                "name": players.get(i).map(|p| p.name.as_str()).unwrap_or("snake"),
                "health": s.health,
                "body": body,
                "head": point_json(head),
                "length": s.len(),
            })
        })
        .collect();
    json!({
        "game": {
            "id": game_id,
            "ruleset": {
                "name": "standard",
                "version": "v1.2.3",
                "settings": {
                    "foodSpawnChance": board.food_spawn_chance,
                    "minimumFood": board.min_food,
                    "hazardDamagePerTurn": board.hazard_damage,
                },
            },
            "timeout": timeout_ms,
        },
        "turn": board.turn,
        "board": {
            "height": board.height,
            "width": board.width,
            "food": points_json(&board.food),
            "hazards": points_json(&board.hazards),
            "snakes": snakes,
        },
        "you": { "id": snake_id(you_idx) },
    })
    .to_string()
}

fn point_json(p: Point) -> serde_json::Value {
    json!({ "x": p.x, "y": p.y })
}

fn points_json(points: &[Point]) -> Vec<serde_json::Value> {
    points.iter().copied().map(point_json).collect()
}

fn snake_id(i: usize) -> String {
    format!("arena-snake-{i}")
}

fn split_env(key: &str) -> Option<Vec<String>> {
    std::env::var(key).ok().map(|v| {
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
