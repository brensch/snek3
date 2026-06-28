//! Battlesnake `/move` API server — the most Albatross-faithful serving path,
//! in pure Rust for a small CPU box.
//!
//! On every `/move` it routes through [`snek_server::serve_move`]: an online MLE
//! of each opponent's *temperature* under the proxy net, then a fixed-depth
//! logit-equilibrium best-response search with our snake pinned rational and each
//! opponent pinned at its estimated tau (leaves evaluated by the proxy net). The
//! offline evaluator (`bin/eval.rs`) calls the same function, so what we measure
//! is exactly what we serve.
//!
//! Config (env vars):
//!   SNEK_MODEL        ONNX proxy model path (default ./model.onnx)
//!   SNEK_PORT         listen port (default 8000)
//!   SNEK_THREADS      worker threads (default 2)
//!   SNEK_DEPTH        search depth in plies (default 2)
//!   SNEK_ITERS        SFP iterations per node (default 120)
//!   SNEK_RESPONSE_TAU our (rational) temperature (default 12.0)
//!   SNEK_DRAW_VALUE   terminal value of a draw at search leaves (default -0.9)
//!   SNEK_EVAL_CHUNK   max obs rows per ONNX forward (default 4096)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use snek_core::json::parse_move_request;
use snek_infer::Net;
use snek_server::{env_or, serve_move, tau_grid, Config, GameState, MOVES, TAU_GRID_LEN};
use tiny_http::{Header, Method, Response, Server};

struct App {
    net: Mutex<Net>,
    games: Mutex<HashMap<String, GameState>>,
    grid: [f32; TAU_GRID_LEN],
    cfg: Config,
}

fn main() {
    let model = std::env::var("SNEK_MODEL").unwrap_or_else(|_| "model.onnx".into());
    let port: u16 = env_or("SNEK_PORT", 8000);
    let threads: usize = env_or("SNEK_THREADS", 2usize).max(1);
    let cfg = Config {
        depth: env_or("SNEK_DEPTH", 2u32),
        iters: env_or("SNEK_ITERS", 120usize),
        response_tau: env_or("SNEK_RESPONSE_TAU", 12.0f32),
        draw_value: env_or("SNEK_DRAW_VALUE", -0.9f32),
        eval_chunk: env_or("SNEK_EVAL_CHUNK", 4096usize),
    };

    let net =
        Net::load(&model).unwrap_or_else(|e| panic!("failed to load ONNX model '{model}': {e}"));
    eprintln!(
        "snek-server: model={model} port={port} threads={threads} depth={} iters={} \
         response_tau={} draw_value={}",
        cfg.depth, cfg.iters, cfg.response_tau, cfg.draw_value
    );

    let app = Arc::new(App {
        net: Mutex::new(net),
        games: Mutex::new(HashMap::new()),
        grid: tau_grid(),
        cfg,
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
            (Method::Post, "/start") => {
                start_game(app, &body);
                "{}".to_string()
            }
            (Method::Post, "/move") => handle_move(app, &body),
            (Method::Post, "/end") => {
                end_game(app, &body);
                "{}".to_string()
            }
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
    r##"{"apiversion":"1","author":"brensch","color":"#3366ff","head":"default","tail":"default","version":"0.2.0"}"##.to_string()
}

fn game_id(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("game")?.get("id")?.as_str().map(String::from))
        .unwrap_or_default()
}

fn start_game(app: &App, body: &str) {
    let id = game_id(body);
    app.games.lock().unwrap().insert(id, GameState::default());
}

fn end_game(app: &App, body: &str) {
    let id = game_id(body);
    app.games.lock().unwrap().remove(&id);
}

/// Snake ids in board order plus our own id, parsed alongside `parse_move_request`
/// (which only yields the index). Order matches `parse_move_request`'s, since both
/// iterate `board.snakes` in array order.
fn parse_ids(body: &str) -> (Vec<String>, String) {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return (Vec::new(), String::new()),
    };
    let you = v
        .get("you")
        .and_then(|y| y.get("id"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let ids = v
        .get("board")
        .and_then(|b| b.get("snakes"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|s| s.get("id").and_then(Value::as_str).unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    (ids, you)
}

fn handle_move(app: &App, body: &str) -> String {
    let mv = compute_move(app, body).unwrap_or(0);
    format!("{{\"move\":\"{}\"}}", MOVES[mv])
}

fn compute_move(app: &App, body: &str) -> Option<usize> {
    let (board, me) = parse_move_request(body).ok()?;
    let (ids, _you) = parse_ids(body);
    let gid = game_id(body);
    let mut games = app.games.lock().unwrap();
    let gs = games.entry(gid).or_default();
    let mut net = app.net.lock().unwrap();
    Some(serve_move(&mut net, &app.grid, &app.cfg, &board, &ids, me, gs))
}
