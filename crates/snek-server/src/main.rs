//! Battlesnake `/move` API server — pure-Rust AlphaZero serving for a small CPU
//! box. Every `/move` runs decoupled-PUCT MCTS over the ONNX policy+value net via
//! [`snek_server::serve_move`] — the same search as self-play, so what we serve
//! matches what we trained. Stateless per move, so /start and /end are no-ops.
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
//!   SNEK_MOVE_LOG     JSONL move log path; empty disables (default logs/api_moves.jsonl)

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use snek_core::json::parse_move_request;
use snek_infer::Net;
use snek_server::{env_or, serve_move_until_diagnostics, Config, SearchDecision, MOVES};
use tiny_http::{Header, Method, Response, Server};

struct App {
    net: Mutex<Net>,
    model: String,
    cfg: Config,
    default_timeout_ms: u64,
    deadline_margin_ms: u64,
    move_log: Option<Mutex<File>>,
}

fn main() {
    let model = std::env::var("SNEK_MODEL").unwrap_or_else(|_| "model.onnx".into());
    let port: u16 = env_or("SNEK_PORT", 8000);
    let threads: usize = env_or("SNEK_THREADS", 2usize).max(1);
    let default_timeout_ms = env_or("SNEK_TIMEOUT_MS", 500u64);
    let deadline_margin_ms = env_or("SNEK_DEADLINE_MARGIN_MS", 150u64);
    let move_log = open_move_log();
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

    let app = Arc::new(App {
        net: Mutex::new(net),
        model,
        cfg,
        default_timeout_ms,
        deadline_margin_ms,
        move_log,
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
            (Method::Post, "/start") | (Method::Post, "/end") => "{}".to_string(),
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
    let timeout_ms = request_timeout_ms(body).unwrap_or(app.default_timeout_ms);
    let search_budget_ms = timeout_ms.saturating_sub(app.deadline_margin_ms).max(1);
    let deadline = started + Duration::from_millis(search_budget_ms);
    let (board, me) = parse_move_request(body).ok()?;
    let lock_started = Instant::now();
    let mut net = app.net.lock().unwrap();
    let lock_wait_ms = lock_started.elapsed().as_secs_f64() * 1000.0;
    let search_started = Instant::now();
    let decision = serve_move_until_diagnostics(&mut net, &app.cfg, &board, me, deadline);
    let search_ms = search_started.elapsed().as_secs_f64() * 1000.0;
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    log_move(
        app,
        body,
        &decision,
        timeout_ms,
        search_budget_ms,
        lock_wait_ms,
        search_ms,
        total_ms,
    );
    Some(decision.move_index)
}

fn request_timeout_ms(body: &str) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let timeout = v.get("game")?.get("timeout")?;
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

fn open_move_log() -> Option<Mutex<File>> {
    let path = std::env::var("SNEK_MOVE_LOG").unwrap_or_else(|_| "logs/api_moves.jsonl".into());
    if path.trim().is_empty() {
        return None;
    }
    if let Some(parent) = Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(file) => Some(Mutex::new(file)),
        Err(e) => {
            eprintln!("snek-server: could not open move log {path}: {e}");
            None
        }
    }
}

fn log_parse_error(app: &App, body: &str) {
    let Some(log) = &app.move_log else {
        return;
    };
    let request_json = serde_json::from_str::<serde_json::Value>(body).unwrap_or_else(|_| {
        serde_json::json!({
            "unparsed_body": body,
        })
    });
    let entry = serde_json::json!({
        "kind": "snek-api-move",
        "ok": false,
        "error": "parse_move_request_failed",
        "request": request_json,
    });
    if let Ok(mut file) = log.lock() {
        let _ = writeln!(file, "{}", entry);
        let _ = file.flush();
    }
}

fn log_move(
    app: &App,
    body: &str,
    decision: &SearchDecision,
    timeout_ms: u64,
    search_budget_ms: u64,
    lock_wait_ms: f64,
    search_ms: f64,
    total_ms: f64,
) {
    let Some(log) = &app.move_log else {
        return;
    };
    let request_json = match serde_json::from_str::<serde_json::Value>(body) {
        Ok(v) => v,
        Err(_) => serde_json::json!({"unparsed_body": body}),
    };
    let game = request_json
        .get("game")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let board = request_json
        .get("board")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let you = request_json
        .get("you")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let diag = &decision.diagnostics;
    let root_actions = diag
        .root_actions
        .iter()
        .map(|row| {
            serde_json::Value::Array(
                row.iter()
                    .map(|a| {
                        serde_json::json!({
                            "move": MOVES[a.move_index],
                            "move_index": a.move_index,
                            "prior": a.prior,
                            "visits": a.visits,
                            "q": a.q,
                        })
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let entry = serde_json::json!({
        "kind": "snek-api-move",
        "ok": true,
        "game_id": game.get("id").and_then(|v| v.as_str()),
        "turn": request_json.get("turn").and_then(|v| v.as_u64()),
        "you_id": you.get("id").and_then(|v| v.as_str()),
        "you_name": you.get("name").and_then(|v| v.as_str()),
        "chosen_move": MOVES[decision.move_index],
        "chosen_move_index": decision.move_index,
        "model": app.model,
        "config": {
            "max_sims": app.cfg.max_sims,
            "c_puct": app.cfg.c_puct,
            "draw_value": app.cfg.draw_value,
            "eval_chunk": app.cfg.eval_chunk,
            "timeout_ms": timeout_ms,
            "deadline_margin_ms": app.deadline_margin_ms,
            "search_budget_ms": search_budget_ms,
        },
        "timing": {
            "lock_wait_ms": lock_wait_ms,
            "search_ms": search_ms,
            "total_ms": total_ms,
        },
        "search": {
            "sims_completed": diag.sims_completed,
            "eval_rows": diag.eval_rows,
            "forward_calls": diag.forward_calls,
            "stopped_reason": diag.stopped_reason,
            "fallback_reason": diag.fallback_reason,
            "root_policy": diag.root_policy,
            "root_values": diag.root_values,
            "root_actions": root_actions,
        },
        "board": board,
        "you": you,
        "request": request_json,
    });
    if let Ok(mut file) = log.lock() {
        let _ = writeln!(file, "{}", entry);
        let _ = file.flush();
    }
}
