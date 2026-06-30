//! Battlesnake `/move` API server — pure-Rust AlphaZero serving for a small CPU
//! box. Every `/move` runs decoupled-PUCT MCTS over the ONNX policy+value net via
//! [`snek_server::serve_move`] — the same search as self-play, so what we serve
//! matches what we trained. Stateless per move; `/end` finalizes the game log.
//!
//! Game recording (see [`recorder`]) is fully off the hot path: `/move` only
//! pushes a small in-memory record over a channel — no serialization, no
//! compression — and a background thread compresses + writes the whole game at
//! the end, so per-move compute is never spent on logging.
//!
//! Config (env vars):
//!   SNEK_MODEL        ONNX model path (default ./model.onnx)
//!   SNEK_PORT         listen port (default 8000)
//!   SNEK_THREADS      worker threads (default 2)
//!   SNEK_MAX_SIMS     safety cap on MCTS simulations per move (default 100000)
//!   SNEK_TIMEOUT_MS   fallback request timeout when JSON lacks game.timeout (default 500)
//!   SNEK_DEADLINE_MARGIN_MS  response margin reserved from timeout (default 150)
//!   SNEK_C_PUCT       PUCT exploration constant (default 1.5)
//!   SNEK_DRAW_VALUE   terminal value of a draw at leaves (default -0.25)
//!   SNEK_EVAL_CHUNK   max obs rows per ONNX forward (default 4096)
//!   SNEK_MOVE_LOG_DIR per-game compressed game log dir; empty disables (default logs/api_moves)
//!   SNEK_GAME_IDLE_SECS  finalize a silent game as incomplete after this many seconds (default 300)
//!   SNEK_LOG_ZSTD_LEVEL  zstd level for game logs, compressed at game end (default 19)

mod orchestrator;
mod recorder;
mod viewer;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use recorder::{ActionDebug, MoveRecord, Recorder, SearchInfo, SnakeState};
use rust_embed::RustEmbed;
use snek_core::json::parse_move_request;
use snek_infer::Net;
use snek_server::{env_or, serve_move_until_diagnostics, Config, SearchDecision, MOVES};
use tiny_http::{Header, Method, Response, Server};

/// The built Vite viewer, embedded into the binary so the API ships
/// self-contained. `SNEK_VIEWER_DIR` overrides this with an on-disk dir for
/// iterating on the frontend without rebuilding the server.
#[derive(RustEmbed)]
#[folder = "viewer/dist"]
struct ViewerAssets;

struct App {
    net: Mutex<Net>,
    cfg: Config,
    default_timeout_ms: u64,
    deadline_margin_ms: u64,
    /// Log dir, kept for the rare parse-error sidecar; `None` disables logging.
    log_dir: Option<PathBuf>,
    parse_err_lock: Mutex<()>,
    recorder: Option<Recorder>,
    /// Hex SHA-256 of the loaded model file (for replay fidelity checks).
    model_sha: String,
}

#[derive(Clone)]
struct AppSettings {
    cfg: Config,
    default_timeout_ms: u64,
    deadline_margin_ms: u64,
    idle_timeout: Duration,
    zstd_level: i32,
}

/// Hex SHA-256 of the model file, or `"unknown"` if it can't be read.
fn model_hash(path: &str) -> String {
    use sha2::{Digest, Sha256};
    match std::fs::read(path) {
        Ok(bytes) => {
            let mut h = Sha256::new();
            h.update(&bytes);
            format!("{:x}", h.finalize())
        }
        Err(_) => "unknown".to_string(),
    }
}

fn short_sha(sha: &str) -> &str {
    &sha[..sha.len().min(12)]
}

fn main() {
    let model = std::env::var("SNEK_MODEL").unwrap_or_else(|_| "model.onnx".into());
    let port: u16 = env_or("SNEK_PORT", 8000);
    let threads: usize = env_or("SNEK_THREADS", 2usize).max(1);
    let log_dir = move_log_dir();
    let settings = AppSettings {
        cfg: Config {
            max_sims: max_sims_from_env(),
            c_puct: env_or("SNEK_C_PUCT", 1.5f32),
            draw_value: env_or("SNEK_DRAW_VALUE", -0.25f32),
            eval_chunk: env_or("SNEK_EVAL_CHUNK", 4096usize),
            leaves_per_sim: env_or("SNEK_LEAVES_PER_SIM", 8usize).max(1),
            virtual_loss: env_or("SNEK_VIRTUAL_LOSS", 1.0f32),
        },
        default_timeout_ms: env_or("SNEK_TIMEOUT_MS", 500u64),
        deadline_margin_ms: env_or("SNEK_DEADLINE_MARGIN_MS", 150u64),
        idle_timeout: Duration::from_secs(env_or("SNEK_GAME_IDLE_SECS", 300u64)),
        zstd_level: env_or("SNEK_LOG_ZSTD_LEVEL", 19i32),
    };

    let app = build_app(model.clone(), log_dir.clone(), &settings);
    eprintln!(
        "snek-server: model={model} sha={} port={port} threads={threads} max_sims={} timeout_ms={} deadline_margin_ms={} c_puct={} draw_value={}",
        short_sha(&app.model_sha),
        settings.cfg.max_sims,
        settings.default_timeout_ms,
        settings.deadline_margin_ms,
        settings.cfg.c_puct,
        settings.cfg.draw_value
    );
    if app.recorder.is_some() {
        eprintln!(
            "snek-server: recording games to {} (idle_timeout={}s, zstd={})",
            log_dir
                .as_ref()
                .map(|d| d.display().to_string())
                .unwrap_or_default(),
            settings.idle_timeout.as_secs(),
            settings.zstd_level
        );
    }

    if let Some(arena) = orchestrator::ArenaConfig::from_env(&model, log_dir.clone(), &settings) {
        let arena_settings = settings.clone();
        std::thread::spawn(move || orchestrator::run(arena, arena_settings));
    }

    let server = Arc::new(Server::http(("0.0.0.0", port)).expect("bind"));
    let mut handles = Vec::new();
    for _ in 0..threads {
        let server = server.clone();
        let app = app.clone();
        handles.push(std::thread::spawn(move || worker(&server, &app)));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn build_app(model: String, log_dir: Option<PathBuf>, settings: &AppSettings) -> Arc<App> {
    let net =
        Net::load(&model).unwrap_or_else(|e| panic!("failed to load ONNX model '{model}': {e}"));
    // Content hash of the loaded weights, recorded with each game so the viewer
    // can tell whether a replay used the same model the game was played with
    // (the weights file, e.g. `latest.onnx`, can be overwritten by training).
    let model_sha = model_hash(&model);
    let recorder_cfg = serde_json::json!({
        "max_sims": settings.cfg.max_sims,
        "c_puct": settings.cfg.c_puct,
        "draw_value": settings.cfg.draw_value,
        "eval_chunk": settings.cfg.eval_chunk,
        "default_timeout_ms": settings.default_timeout_ms,
        "deadline_margin_ms": settings.deadline_margin_ms,
        "model_sha": model_sha,
    });
    let recorder = Recorder::spawn(
        log_dir.clone(),
        model,
        recorder_cfg,
        settings.idle_timeout,
        settings.zstd_level,
    );
    Arc::new(App {
        net: Mutex::new(net),
        cfg: settings.cfg.clone(),
        default_timeout_ms: settings.default_timeout_ms,
        deadline_margin_ms: settings.deadline_margin_ms,
        log_dir,
        parse_err_lock: Mutex::new(()),
        recorder,
        model_sha,
    })
}

fn worker(server: &Server, app: &App) {
    loop {
        let mut req = match server.recv() {
            Ok(r) => r,
            Err(_) => return,
        };
        let method = req.method().clone();
        let url = req.url().to_string();
        let (path, query) = match url.split_once('?') {
            Some((p, q)) => (p, q),
            None => (url.as_str(), ""),
        };
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);

        // Viewer API and the embedded SPA are GET-only and handled apart from the
        // Battlesnake routes so they can't collide.
        if method == Method::Get && path.starts_with("/viewer") {
            let r = viewer::handle(
                app.log_dir.as_ref(),
                &app.net,
                &app.cfg,
                &app.model_sha,
                path,
                query,
            );
            let _ = req.respond(text_response(
                r.status,
                "application/json",
                r.body.into_bytes(),
            ));
            continue;
        }
        if method == Method::Get && (path == "/app" || path.starts_with("/app/")) {
            let (status, ctype, bytes) = serve_asset(path);
            let _ = req.respond(text_response(status, ctype, bytes));
            continue;
        }

        let resp = match (&method, path) {
            (Method::Get, "/") => info_json(),
            (Method::Post, "/move") => handle_move(app, &body),
            (Method::Post, "/end") => {
                handle_end(app, &body);
                "{}".to_string()
            }
            (Method::Post, "/start") => "{}".to_string(),
            _ => "{}".to_string(),
        };
        let _ = req.respond(json_response(resp));
    }
}

fn json_response(body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    text_response(200, "application/json", body.into_bytes())
}

fn text_response(
    status: u16,
    content_type: &str,
    bytes: Vec<u8>,
) -> Response<std::io::Cursor<Vec<u8>>> {
    let header = Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes()).unwrap();
    Response::from_data(bytes)
        .with_status_code(status)
        .with_header(header)
}

/// Serve an embedded (or `SNEK_VIEWER_DIR`-overridden) viewer asset. Unknown
/// paths fall back to `index.html` so client-side routing works (SPA).
fn serve_asset(path: &str) -> (u16, &'static str, Vec<u8>) {
    // Strip the `/app` mount prefix; map the bare mount to index.html.
    let rel = path.trim_start_matches("/app").trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };

    let load = |name: &str| -> Option<Vec<u8>> {
        if let Ok(dir) = std::env::var("SNEK_VIEWER_DIR") {
            std::fs::read(PathBuf::from(dir).join(name)).ok()
        } else {
            ViewerAssets::get(name).map(|f| f.data.into_owned())
        }
    };

    let (name, bytes) = match load(rel) {
        Some(b) => (rel.to_string(), b),
        // SPA fallback: anything not a real file serves the shell.
        None => match load("index.html") {
            Some(b) => ("index.html".to_string(), b),
            None => return (404, "text/plain", b"viewer not built".to_vec()),
        },
    };
    (200, content_type_for(&name), bytes)
}

fn content_type_for(name: &str) -> &'static str {
    match name.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        _ => "application/octet-stream",
    }
}

fn info_json() -> String {
    r##"{"apiversion":"1","author":"brensch","color":"#3366ff","head":"default","tail":"default","version":"0.3.0"}"##.to_string()
}

fn handle_move(app: &App, body: &str) -> String {
    let mv = compute_move(app, body).unwrap_or_else(|| {
        log_parse_error(app, body);
        0
    });
    format!("{{\"move\":\"{}\"}}", MOVES[mv])
}

fn compute_move(app: &App, body: &str) -> Option<usize> {
    let started = Instant::now();
    // Parse the request once; reuse for the timeout and the in-memory record so
    // the hot path never re-parses or serializes/compresses anything.
    let request: serde_json::Value = serde_json::from_str(body).ok()?;
    let timeout_ms = request_timeout_ms(&request).unwrap_or(app.default_timeout_ms);
    let search_budget_ms = timeout_ms.saturating_sub(app.deadline_margin_ms).max(1);
    let deadline = started + Duration::from_millis(search_budget_ms);
    let (board, me) = parse_move_request(body).ok()?;
    let lock_started = Instant::now();
    // Tolerate a poisoned lock (e.g. a panic on the off-hot-path viewer replay):
    // `Net::forward` is stateless across calls, so live serving stays correct.
    let mut net = app.net.lock().unwrap_or_else(|p| p.into_inner());
    let lock_wait_ms = lock_started.elapsed().as_secs_f64() * 1000.0;
    let search_started = Instant::now();
    let decision = serve_move_until_diagnostics(&mut net, &app.cfg, &board, me, deadline);
    drop(net);
    let search_ms = search_started.elapsed().as_secs_f64() * 1000.0;
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    // Push a compact record into memory (a cheap channel send). All
    // serialization + zstd happens later, at game end, on the recorder thread.
    if let Some(rec) = &app.recorder {
        if let Some((game_id, meta, record)) =
            build_record(&request, me, &decision, [lock_wait_ms, search_ms, total_ms])
        {
            rec.record_move(game_id, meta, record);
        }
    }
    Some(decision.move_index)
}

/// Finalize a game on `/end` (the game-complete message): trigger the write off
/// the hot path. Does no I/O itself.
fn handle_end(app: &App, body: &str) {
    let Some(rec) = &app.recorder else {
        return;
    };
    if let Ok(request) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some((game_id, meta, record)) = build_terminal_record(&request) {
            rec.record_move(game_id.clone(), meta, record);
            rec.finish(game_id);
        } else if let Some(id) = request
            .get("game")
            .and_then(|g| g.get("id"))
            .and_then(|i| i.as_str())
        {
            rec.finish(id.to_string());
        }
    }
}

/// Build the per-move record from an already-parsed request. Cheap: a handful of
/// field reads, no serialization. `meta` (board size + ruleset) is attached only
/// on the first turn we see, since it never changes within a game.
fn build_record(
    request: &serde_json::Value,
    me: usize,
    decision: &SearchDecision,
    timing: [f64; 3],
) -> Option<(String, Option<recorder::GameMeta>, MoveRecord)> {
    let game_id = request
        .get("game")
        .and_then(|g| g.get("id"))
        .and_then(|i| i.as_str())?
        .to_string();
    let turn = request.get("turn").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let board = request.get("board")?;
    let width = board.get("width").and_then(|v| v.as_i64()).unwrap_or(0);
    let height = board.get("height").and_then(|v| v.as_i64()).unwrap_or(0);

    let food = points_xy(board.get("food"));
    let hazards = points_xy(board.get("hazards"));
    let snakes = board
        .get("snakes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|s| SnakeState {
                    id: s
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    name: s
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    health: s.get("health").and_then(|v| v.as_i64()).unwrap_or(0) as i16,
                    body: points_xy(s.get("body")),
                })
                .collect()
        })
        .unwrap_or_default();

    let diag = &decision.diagnostics;
    let root_actions = diag
        .root_actions
        .iter()
        .map(|row| {
            row.iter()
                .map(|a| ActionDebug(a.move_index as u8, a.prior, a.visits, a.q))
                .collect()
        })
        .collect();
    let search = SearchInfo {
        sims_completed: diag.sims_completed,
        terminal_only_sims: diag.terminal_only_sims,
        eval_rows: diag.eval_rows,
        forward_calls: diag.forward_calls,
        max_depth: diag.max_depth,
        stopped_reason: diag.stopped_reason,
        fallback_reason: diag.fallback_reason,
        root_policy: diag.root_policy.clone(),
        root_values: diag.root_values.clone(),
        root_actions,
    };

    let meta = if turn == 0 {
        Some(recorder::meta_from_request(request, width, height))
    } else {
        None
    };
    let record = MoveRecord {
        turn,
        you: Some(me),
        chosen_move: Some(decision.move_index as u8),
        food,
        hazards,
        snakes,
        search: Some(search),
        timing,
    };
    Some((game_id, meta, record))
}

/// Build the final board frame from Battlesnake's `/end` payload. `/end` does
/// not ask us to choose a move, but it carries the final `turn`/`board`/`you`
/// state, which is the terminal frame the viewer needs to show the result.
fn build_terminal_record(
    request: &serde_json::Value,
) -> Option<(String, Option<recorder::GameMeta>, MoveRecord)> {
    let game_id = request
        .get("game")
        .and_then(|g| g.get("id"))
        .and_then(|i| i.as_str())?
        .to_string();
    let turn = request.get("turn").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let board = request.get("board")?;
    let width = board.get("width").and_then(|v| v.as_i64()).unwrap_or(0);
    let height = board.get("height").and_then(|v| v.as_i64()).unwrap_or(0);
    let you_id = request
        .get("you")
        .and_then(|y| y.get("id"))
        .and_then(|v| v.as_str());

    let food = points_xy(board.get("food"));
    let hazards = points_xy(board.get("hazards"));
    let snakes: Vec<SnakeState> = board
        .get("snakes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|s| SnakeState {
                    id: s
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    name: s
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    health: s.get("health").and_then(|v| v.as_i64()).unwrap_or(0) as i16,
                    body: points_xy(s.get("body")),
                })
                .collect()
        })
        .unwrap_or_default();
    let me = you_id.and_then(|id| snakes.iter().position(|s| s.id == id));
    let meta = if turn == 0 {
        Some(recorder::meta_from_request(request, width, height))
    } else {
        None
    };
    let record = MoveRecord {
        turn,
        you: me,
        chosen_move: None,
        food,
        hazards,
        snakes,
        search: None,
        timing: [0.0, 0.0, 0.0],
    };
    Some((game_id, meta, record))
}

fn points_xy(v: Option<&serde_json::Value>) -> Vec<[i8; 2]> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| Some([p.get("x")?.as_i64()? as i8, p.get("y")?.as_i64()? as i8]))
                .collect()
        })
        .unwrap_or_default()
}

fn request_timeout_ms(request: &serde_json::Value) -> Option<u64> {
    let timeout = request.get("game")?.get("timeout")?;
    timeout
        .as_u64()
        .or_else(|| timeout.as_str().and_then(|s| s.parse().ok()))
}

fn max_sims_from_env() -> usize {
    std::env::var("SNEK_MAX_SIMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| std::env::var("SNEK_SIMS").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(100_000usize)
}

fn move_log_dir() -> Option<PathBuf> {
    let dir = std::env::var("SNEK_MOVE_LOG_DIR").unwrap_or_else(|_| "logs/api_moves".into());
    if dir.trim().is_empty() {
        return None;
    }
    let dir = PathBuf::from(dir);
    match std::fs::create_dir_all(&dir) {
        Ok(()) => Some(dir),
        Err(e) => {
            eprintln!(
                "snek-server: could not create move log dir {}: {e}",
                dir.display()
            );
            None
        }
    }
}

/// Unparseable requests are rare and worth keeping verbatim; append them to a
/// small plain-text sidecar rather than the compressed per-game logs.
fn log_parse_error(app: &App, body: &str) {
    let Some(dir) = &app.log_dir else {
        return;
    };
    let request_json = serde_json::from_str::<serde_json::Value>(body)
        .unwrap_or_else(|_| serde_json::json!({ "unparsed_body": body }));
    let entry = serde_json::json!({
        "kind": "snek-api-parse-error",
        "request": request_json,
    });
    let path = dir.join("parse_errors.jsonl");
    let _guard = app.parse_err_lock.lock().ok();
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut file) => {
            let _ = writeln!(file, "{}", entry);
        }
        Err(e) => {
            eprintln!(
                "snek-server: could not append parse-error log {}: {e}",
                path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn end_payload_builds_terminal_record_without_search() {
        let request = json!({
            "game": {
                "id": "game-terminal",
                "ruleset": {"name": "standard"}
            },
            "turn": 12,
            "board": {
                "width": 11,
                "height": 11,
                "food": [{"x": 4, "y": 5}],
                "hazards": [],
                "snakes": [{
                    "id": "me",
                    "name": "local",
                    "health": 91,
                    "body": [{"x": 1, "y": 2}, {"x": 1, "y": 1}]
                }]
            },
            "you": {"id": "me"}
        });

        let (game_id, meta, record) = build_terminal_record(&request).unwrap();

        assert_eq!(game_id, "game-terminal");
        assert!(meta.is_none());
        assert_eq!(record.turn, 12);
        assert_eq!(record.you, Some(0));
        assert_eq!(record.chosen_move, None);
        assert!(record.search.is_none());
        assert_eq!(record.food, vec![[4, 5]]);
        assert_eq!(record.snakes[0].body, vec![[1, 2], [1, 1]]);
    }

    #[test]
    fn end_payload_allows_dead_you_missing_from_board_snakes() {
        let request = json!({
            "game": {"id": "game-loss"},
            "turn": 7,
            "board": {
                "width": 11,
                "height": 11,
                "food": [],
                "snakes": []
            },
            "you": {"id": "me"}
        });

        let (_, _, record) = build_terminal_record(&request).unwrap();

        assert_eq!(record.you, None);
        assert!(record.snakes.is_empty());
    }
}
