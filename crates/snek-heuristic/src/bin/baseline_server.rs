//! Battlesnake HTTP wrapper around the heuristic baselines, so browser-board
//! games can seat cheap fixed-strength opponents without libtorch. Stateless
//! per move — one process serves any number of seats; point several board
//! `--url` entries at the same port.
//!
//! Config (env vars):
//!   SNEK_BASELINE_PORT     listen port (default 8100)
//!   SNEK_BASELINE_KIND     voronoi | floodfill (default voronoi)
//!   SNEK_BASELINE_MS       per-move search budget in ms (default 200)
//!   SNEK_BASELINE_SIMS     simulation cap per move (default 20000)
//!   SNEK_BASELINE_THREADS  worker threads (default 4, so concurrent seats
//!                          don't serialize their searches)

use std::sync::Arc;
use std::time::{Duration, Instant};

use snek_core::json::parse_move_request;
use snek_heuristic::{baseline_move_until, Baseline, HeuristicConfig};
use tiny_http::{Header, Method, Response, Server};

const MOVES: [&str; 4] = ["up", "down", "left", "right"];

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let port: u16 = env_or("SNEK_BASELINE_PORT", 8100);
    let kind = std::env::var("SNEK_BASELINE_KIND")
        .ok()
        .map(|s| Baseline::parse(&s).unwrap_or_else(|| panic!("unknown SNEK_BASELINE_KIND '{s}'")))
        .unwrap_or(Baseline::Voronoi);
    let budget = Duration::from_millis(env_or("SNEK_BASELINE_MS", 200));
    let threads: usize = env_or("SNEK_BASELINE_THREADS", 4usize).max(1);
    let cfg = HeuristicConfig {
        max_sims: env_or("SNEK_BASELINE_SIMS", HeuristicConfig::default().max_sims),
        ..HeuristicConfig::default()
    };
    let server = Arc::new(Server::http(("0.0.0.0", port)).expect("bind"));
    eprintln!(
        "snek-baseline: kind={} port={port} budget_ms={} max_sims={} threads={threads}",
        kind.token(),
        budget.as_millis(),
        cfg.max_sims
    );
    let mut handles = Vec::new();
    for _ in 0..threads {
        let server = server.clone();
        let cfg = cfg.clone();
        handles.push(std::thread::spawn(move || worker(&server, kind, &cfg, budget)));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn worker(server: &Server, kind: Baseline, cfg: &HeuristicConfig, budget: Duration) {
    loop {
        let mut req = match server.recv() {
            Ok(r) => r,
            Err(_) => return,
        };
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        let path = req.url().split('?').next().unwrap_or("/").to_string();
        let resp = match (req.method(), path.as_str()) {
            (Method::Get, "/") => info_json(kind),
            (Method::Post, "/move") => handle_move(kind, cfg, budget, &body),
            // /start and /end included: nothing to track between moves.
            _ => "{}".to_string(),
        };
        let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
        let _ = req.respond(Response::from_string(resp).with_header(header));
    }
}

fn info_json(kind: Baseline) -> String {
    format!(
        r##"{{"apiversion":"1","author":"brensch","color":"#95a5a6","head":"default","tail":"default","version":"baseline-{}"}}"##,
        kind.token()
    )
}

fn handle_move(kind: Baseline, cfg: &HeuristicConfig, budget: Duration, body: &str) -> String {
    let mv = parse_move_request(body)
        .ok()
        .map(|(board, me)| {
            baseline_move_until(kind, cfg, &board, me, Instant::now() + budget).move_index
        })
        .unwrap_or(0);
    format!("{{\"move\":\"{}\"}}", MOVES[mv])
}
