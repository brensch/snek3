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

mod recorder;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use recorder::{ActionDebug, MoveRecord, Recorder, SearchInfo, SnakeState};
use snek_core::json::parse_move_request;
use snek_infer::Net;
use snek_server::{env_or, serve_move_until_diagnostics, Config, SearchDecision, MOVES};
use tiny_http::{Header, Method, Response, Server};

struct App {
    net: Mutex<Net>,
    cfg: Config,
    default_timeout_ms: u64,
    deadline_margin_ms: u64,
    /// Log dir, kept for the rare parse-error sidecar; `None` disables logging.
    log_dir: Option<PathBuf>,
    parse_err_lock: Mutex<()>,
    recorder: Option<Recorder>,
}

fn main() {
    let model = std::env::var("SNEK_MODEL").unwrap_or_else(|_| "model.onnx".into());
    let port: u16 = env_or("SNEK_PORT", 8000);
    let threads: usize = env_or("SNEK_THREADS", 2usize).max(1);
    let default_timeout_ms = env_or("SNEK_TIMEOUT_MS", 500u64);
    let deadline_margin_ms = env_or("SNEK_DEADLINE_MARGIN_MS", 150u64);
    let log_dir = move_log_dir();
    let idle_timeout = Duration::from_secs(env_or("SNEK_GAME_IDLE_SECS", 300u64));
    let zstd_level: i32 = env_or("SNEK_LOG_ZSTD_LEVEL", 19i32);
    let cfg = Config {
        max_sims: max_sims_from_env(),
        c_puct: env_or("SNEK_C_PUCT", 1.5f32),
        draw_value: env_or("SNEK_DRAW_VALUE", -0.25f32),
        eval_chunk: env_or("SNEK_EVAL_CHUNK", 4096usize),
    };

    let net =
        Net::load(&model).unwrap_or_else(|e| panic!("failed to load ONNX model '{model}': {e}"));
    eprintln!(
        "snek-server: model={model} port={port} threads={threads} max_sims={} timeout_ms={} deadline_margin_ms={} c_puct={} draw_value={}",
        cfg.max_sims,
        default_timeout_ms,
        deadline_margin_ms,
        cfg.c_puct,
        cfg.draw_value
    );

    let recorder_cfg = serde_json::json!({
        "max_sims": cfg.max_sims,
        "c_puct": cfg.c_puct,
        "draw_value": cfg.draw_value,
        "eval_chunk": cfg.eval_chunk,
        "default_timeout_ms": default_timeout_ms,
        "deadline_margin_ms": deadline_margin_ms,
    });
    let recorder = Recorder::spawn(log_dir.clone(), model, recorder_cfg, idle_timeout, zstd_level);
    if recorder.is_some() {
        eprintln!(
            "snek-server: recording games to {} (idle_timeout={}s, zstd={})",
            log_dir.as_ref().map(|d| d.display().to_string()).unwrap_or_default(),
            idle_timeout.as_secs(),
            zstd_level
        );
    }

    let app = Arc::new(App {
        net: Mutex::new(net),
        cfg,
        default_timeout_ms,
        deadline_margin_ms,
        log_dir,
        parse_err_lock: Mutex::new(()),
        recorder,
    });

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

fn worker(server: &Server, app: &App) {
    loop {
        let mut req = match server.recv() {
            Ok(r) => r,
            Err(_) => return,
        };
        let method = req.method().clone();
        let url = req.url().to_string();
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);

        let resp = match (&method, url.as_str()) {
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
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    Response::from_string(body).with_header(header)
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
    let mut net = app.net.lock().unwrap();
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
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(id) = v.get("game").and_then(|g| g.get("id")).and_then(|i| i.as_str()) {
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
                    id: s.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    name: s.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
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
        you: me,
        chosen_move: decision.move_index as u8,
        food,
        hazards,
        snakes,
        search,
        timing,
    };
    Some((game_id, meta, record))
}

fn points_xy(v: Option<&serde_json::Value>) -> Vec<[i8; 2]> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    Some([
                        p.get("x")?.as_i64()? as i8,
                        p.get("y")?.as_i64()? as i8,
                    ])
                })
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
