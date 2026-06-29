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

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use snek_core::json::parse_move_request;
use snek_infer::Net;
use snek_server::{env_or, serve_move_until, Config, MOVES};
use tiny_http::{Header, Method, Response, Server};

struct App {
    net: Mutex<Net>,
    cfg: Config,
    default_timeout_ms: u64,
    deadline_margin_ms: u64,
}

fn main() {
    let model = std::env::var("SNEK_MODEL").unwrap_or_else(|_| "model.onnx".into());
    let port: u16 = env_or("SNEK_PORT", 8000);
    let threads: usize = env_or("SNEK_THREADS", 2usize).max(1);
    let default_timeout_ms = env_or("SNEK_TIMEOUT_MS", 500u64);
    let deadline_margin_ms = env_or("SNEK_DEADLINE_MARGIN_MS", 150u64);
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
        cfg,
        default_timeout_ms,
        deadline_margin_ms,
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
    let mv = compute_move(app, body).unwrap_or(0);
    format!("{{\"move\":\"{}\"}}", MOVES[mv])
}

fn compute_move(app: &App, body: &str) -> Option<usize> {
    let started = Instant::now();
    let timeout_ms = request_timeout_ms(body).unwrap_or(app.default_timeout_ms);
    let search_budget_ms = timeout_ms.saturating_sub(app.deadline_margin_ms).max(1);
    let deadline = started + Duration::from_millis(search_budget_ms);
    let (board, me) = parse_move_request(body).ok()?;
    let mut net = app.net.lock().unwrap();
    Some(serve_move_until(&mut net, &app.cfg, &board, me, deadline))
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
