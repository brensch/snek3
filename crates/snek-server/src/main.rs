//! Battlesnake `/move` API server — pure-Rust AlphaZero serving for a small CPU
//! box. Every `/move` runs decoupled-PUCT MCTS over the ONNX policy+value net via
//! [`snek_server::serve_move`] — the same search as self-play, so what we serve
//! matches what we trained. Stateless per move, so /start and /end are no-ops.
//!
//! Config (env vars):
//!   SNEK_MODEL        ONNX model path (default ./model.onnx)
//!   SNEK_PORT         listen port (default 8000)
//!   SNEK_THREADS      worker threads (default 2)
//!   SNEK_SIMS         MCTS simulations per move (default 200)
//!   SNEK_C_PUCT       PUCT exploration constant (default 1.5)
//!   SNEK_DRAW_VALUE   terminal value of a draw at leaves (default -0.25)
//!   SNEK_EVAL_CHUNK   max obs rows per ONNX forward (default 4096)

use std::sync::{Arc, Mutex};

use snek_core::json::parse_move_request;
use snek_infer::Net;
use snek_server::{env_or, serve_move, Config, MOVES};
use tiny_http::{Header, Method, Response, Server};

struct App {
    net: Mutex<Net>,
    cfg: Config,
}

fn main() {
    let model = std::env::var("SNEK_MODEL").unwrap_or_else(|_| "model.onnx".into());
    let port: u16 = env_or("SNEK_PORT", 8000);
    let threads: usize = env_or("SNEK_THREADS", 2usize).max(1);
    let cfg = Config {
        sims: env_or("SNEK_SIMS", 200usize),
        c_puct: env_or("SNEK_C_PUCT", 1.5f32),
        draw_value: env_or("SNEK_DRAW_VALUE", -0.25f32),
        eval_chunk: env_or("SNEK_EVAL_CHUNK", 4096usize),
    };

    let net =
        Net::load(&model).unwrap_or_else(|e| panic!("failed to load ONNX model '{model}': {e}"));
    eprintln!(
        "snek-server: model={model} port={port} threads={threads} sims={} c_puct={} draw_value={}",
        cfg.sims, cfg.c_puct, cfg.draw_value
    );

    let app = Arc::new(App {
        net: Mutex::new(net),
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
    let (board, me) = parse_move_request(body).ok()?;
    let mut net = app.net.lock().unwrap();
    Some(serve_move(&mut net, &app.cfg, &board, me))
}
