//! Read-only game viewer API, namespaced under `/viewer/*` so it never collides
//! with the Battlesnake routes (`/`, `/start`, `/move`, `/end`).
//!
//! Endpoints:
//!   GET /viewer/games                      list recorded game ids (newest first)
//!   GET /viewer/games/{id}                 the decompressed saved game JSON
//!   GET /viewer/games/{id}/tree?turn=N[&sims=M]
//!                                          replay turn N and return the full
//!                                          exploration tree. Defaults to the
//!                                          recorded sim count for a faithful
//!                                          reproduction of the in-game search.
//!
//! Everything here is off the hot path: it only runs when something hits a
//! `/viewer` URL, and the replay locks the same `Net` the `/move` handler uses.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use snek_core::json::parse_move_request;
use snek_search::{ActionStat, ChildEdge, NodeSnake, TreeNodeSnapshot, TreeSnapshot};

use snek_server::{serve_move_replay, Config, Net};

/// Result of handling a viewer request: a status code and a JSON (or text) body.
pub struct ViewerResponse {
    pub status: u16,
    pub body: String,
}

impl ViewerResponse {
    fn ok(body: String) -> Self {
        ViewerResponse { status: 200, body }
    }
    fn err(status: u16, msg: &str) -> Self {
        ViewerResponse {
            status,
            body: serde_json::json!({ "error": msg }).to_string(),
        }
    }
}

/// Mirror of the recorder's `safe_log_stem` so we resolve the same on-disk file.
fn safe_stem(s: &str) -> String {
    let mut out = String::with_capacity(s.len().max(1));
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown_game".into()
    } else {
        out
    }
}

#[derive(Serialize)]
struct GameListEntry {
    id: String,
    bytes: u64,
    /// File mtime as seconds since the unix epoch (for sorting / display).
    modified: u64,
}

fn list_games(dir: &Path) -> Vec<GameListEntry> {
    let mut entries: Vec<GameListEntry> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let name = path.file_name()?.to_str()?;
            let id = name.strip_suffix(".json.zst")?.to_string();
            let meta = e.metadata().ok()?;
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            Some(GameListEntry {
                id,
                bytes: meta.len(),
                modified,
            })
        })
        .collect();
    entries.sort_by(|a, b| b.modified.cmp(&a.modified));
    entries
}

/// Read and zstd-decompress a saved game into its JSON document.
fn read_game(dir: &Path, id: &str) -> Result<Value, String> {
    let path = dir.join(format!("{}.json.zst", safe_stem(id)));
    let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let json = zstd::stream::decode_all(bytes.as_slice()).map_err(|e| format!("decode: {e}"))?;
    serde_json::from_slice(&json).map_err(|e| format!("parse: {e}"))
}

fn model_hash(path: &str) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Some(format!("{:x}", h.finalize()))
}

/// Rebuild a Battlesnake `/move` request JSON for one recorded turn, so we can
/// reuse the canonical `parse_move_request` to get an exact `Board`.
fn request_for_turn(doc: &Value, mv: &Value) -> Result<String, String> {
    let board = doc.get("board").ok_or("game has no board dims")?;
    let width = board.get("width").and_then(Value::as_i64).unwrap_or(0);
    let height = board.get("height").and_then(Value::as_i64).unwrap_or(0);
    let turn = mv.get("turn").and_then(Value::as_u64).unwrap_or(0);
    let you = mv.get("you").and_then(Value::as_u64).unwrap_or(0) as usize;

    let to_points = |v: Option<&Value>| -> Vec<Value> {
        v.and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| {
                        let a = p.as_array()?;
                        Some(serde_json::json!({ "x": a.first()?, "y": a.get(1)? }))
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let snakes_in = mv
        .get("snakes")
        .and_then(Value::as_array)
        .ok_or("turn has no snakes")?;
    let snakes: Vec<Value> = snakes_in
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.get("id").cloned().unwrap_or(Value::Null),
                "health": s.get("health").cloned().unwrap_or(Value::Null),
                "body": to_points(s.get("body")),
            })
        })
        .collect();
    let you_id = snakes_in
        .get(you)
        .and_then(|s| s.get("id"))
        .cloned()
        .unwrap_or(Value::Null);

    let request = serde_json::json!({
        "turn": turn,
        "board": {
            "width": width,
            "height": height,
            "food": to_points(mv.get("food")),
            "hazards": to_points(mv.get("hazards")),
            "snakes": snakes,
        },
        "you": { "id": you_id },
    });
    Ok(request.to_string())
}

// --- serde mirrors of the snek-search snapshot types (those are serde-free) ---

#[derive(Serialize)]
struct ActionOut {
    #[serde(rename = "move")]
    move_index: usize,
    prior: f32,
    visits: f32,
    q: f32,
}
impl From<&ActionStat> for ActionOut {
    fn from(a: &ActionStat) -> Self {
        ActionOut {
            move_index: a.move_index,
            prior: a.prior,
            visits: a.visits,
            q: a.q,
        }
    }
}

#[derive(Serialize)]
struct ChildOut {
    child: usize,
    moves: Vec<usize>,
}
impl From<&ChildEdge> for ChildOut {
    fn from(c: &ChildEdge) -> Self {
        ChildOut {
            child: c.child,
            moves: c.moves.clone(),
        }
    }
}

#[derive(Serialize)]
struct SnakeOut {
    alive: bool,
    health: i16,
    body: Vec<[i8; 2]>,
}
impl From<&NodeSnake> for SnakeOut {
    fn from(s: &NodeSnake) -> Self {
        SnakeOut {
            alive: s.alive,
            health: s.health,
            body: s.body.clone(),
        }
    }
}

#[derive(Serialize)]
struct NodeOut {
    id: usize,
    depth: u32,
    terminal: bool,
    expanded: bool,
    total_visits: f32,
    term_value: Vec<f32>,
    actions: Vec<Vec<ActionOut>>,
    children: Vec<ChildOut>,
    snakes: Vec<SnakeOut>,
}
impl From<&TreeNodeSnapshot> for NodeOut {
    fn from(n: &TreeNodeSnapshot) -> Self {
        NodeOut {
            id: n.id,
            depth: n.depth,
            terminal: n.terminal,
            expanded: n.expanded,
            total_visits: n.total_visits,
            term_value: n.term_value.clone(),
            actions: n
                .actions
                .iter()
                .map(|row| row.iter().map(ActionOut::from).collect())
                .collect(),
            children: n.children.iter().map(ChildOut::from).collect(),
            snakes: n.snakes.iter().map(SnakeOut::from).collect(),
        }
    }
}

#[derive(Serialize)]
struct TreeOut {
    n_snakes: usize,
    node_count: usize,
    max_depth: u32,
    nodes: Vec<NodeOut>,
}
impl From<&TreeSnapshot> for TreeOut {
    fn from(t: &TreeSnapshot) -> Self {
        let max_depth = t.nodes.iter().map(|n| n.depth).max().unwrap_or(0);
        TreeOut {
            n_snakes: t.n_snakes,
            node_count: t.nodes.len(),
            max_depth,
            nodes: t.nodes.iter().map(NodeOut::from).collect(),
        }
    }
}

#[derive(Serialize)]
struct TreeResponse {
    turn: u64,
    you: usize,
    /// Move the live game recorded (for cross-checking the replay).
    recorded_move: Option<u64>,
    /// Move the replay chose.
    replay_move: usize,
    sims: usize,
    requested_sims: usize,
    terminal_only_sims: usize,
    forward_calls: usize,
    eval_rows: usize,
    root_policy: Vec<f32>,
    root_values: Vec<f32>,
    /// Model hash recorded with the game (may be absent in older logs).
    recorded_model_sha: Option<String>,
    /// Hash of the model the server replayed with.
    server_model_sha: String,
    /// Whether the replay used the same weights the game was recorded with. When
    /// false, the tree is illustrative — the network drifted since recording.
    model_match: bool,
    tree: Option<TreeOut>,
}

/// Find the recorded move with `turn == turn`.
fn find_turn<'a>(doc: &'a Value, turn: u64) -> Option<&'a Value> {
    doc.get("moves")
        .and_then(Value::as_array)?
        .iter()
        .find(|m| m.get("turn").and_then(Value::as_u64) == Some(turn))
}

/// Handle one `/viewer/...` request. `net`/`cfg` are needed only by the tree
/// replay; the caller holds the same net the `/move` handler uses.
pub fn handle(
    dir: Option<&PathBuf>,
    net: &std::sync::Mutex<Net>,
    cfg: &Config,
    server_model_sha: &str,
    path: &str,
    query: &str,
) -> ViewerResponse {
    let Some(dir) = dir else {
        return ViewerResponse::err(503, "game logging disabled (SNEK_MOVE_LOG_DIR empty)");
    };
    let rest = path.trim_start_matches("/viewer");
    let rest = rest.trim_start_matches('/');
    let segments: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();

    match segments.as_slice() {
        ["games"] => ViewerResponse::ok(serde_json::to_string(&list_games(dir)).unwrap()),
        ["games", id] => match read_game(dir, id) {
            Ok(doc) => ViewerResponse::ok(doc.to_string()),
            Err(e) => ViewerResponse::err(404, &e),
        },
        ["games", id, "tree"] => handle_tree(dir, net, cfg, server_model_sha, id, query),
        _ => ViewerResponse::err(404, "unknown viewer route"),
    }
}

fn handle_tree(
    dir: &Path,
    net: &std::sync::Mutex<Net>,
    cfg: &Config,
    server_model_sha: &str,
    id: &str,
    query: &str,
) -> ViewerResponse {
    let turn = query_param(query, "turn").and_then(|v| v.parse::<u64>().ok());
    let Some(turn) = turn else {
        return ViewerResponse::err(400, "missing or invalid ?turn=N");
    };
    let sims_override = query_param(query, "sims").and_then(|v| v.parse::<usize>().ok());

    let doc = match read_game(dir, id) {
        Ok(d) => d,
        Err(e) => return ViewerResponse::err(404, &e),
    };
    let Some(mv) = find_turn(&doc, turn) else {
        return ViewerResponse::err(404, "turn not found in game");
    };

    let recorded_model_sha = doc
        .get("config")
        .and_then(|c| c.get("model_sha"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let recorded_model_path = doc.get("model").and_then(Value::as_str);

    let recorded_sims = mv
        .get("search")
        .and_then(|s| s.get("sims_completed"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    // Default to the exact recorded sim count: a faithful reproduction of the
    // in-game tree. Cap an explicit override so a stray ?sims can't wedge a worker.
    let n_iters = sims_override
        .unwrap_or(recorded_sims)
        .min(cfg.max_sims)
        .max(1);

    let body = match request_for_turn(&doc, mv) {
        Ok(b) => b,
        Err(e) => return ViewerResponse::err(500, &e),
    };
    let (board, me) = match parse_move_request(&body) {
        Ok(v) => v,
        Err(e) => return ViewerResponse::err(500, &format!("rebuild board: {e}")),
    };

    let mut replay_model_sha = server_model_sha.to_string();
    let result = if recorded_model_sha.as_deref() == Some(server_model_sha) {
        // Recover from a poisoned lock: `Net::forward` is stateless across calls,
        // so a panic on another request must not wedge live `/move` serving.
        let mut net = net.lock().unwrap_or_else(|p| p.into_inner());
        let result = serve_move_replay(&mut net, cfg, &board, me, n_iters);
        drop(net);
        result
    } else if let (Some(path), Some(recorded_sha)) =
        (recorded_model_path, recorded_model_sha.as_deref())
    {
        match model_hash(path).filter(|sha| sha == recorded_sha) {
            Some(sha) => match Net::load(path) {
                Ok(mut replay_net) => {
                    replay_model_sha = sha;
                    serve_move_replay(&mut replay_net, cfg, &board, me, n_iters)
                }
                Err(_) => {
                    let mut net = net.lock().unwrap_or_else(|p| p.into_inner());
                    let result = serve_move_replay(&mut net, cfg, &board, me, n_iters);
                    drop(net);
                    result
                }
            },
            None => {
                let mut net = net.lock().unwrap_or_else(|p| p.into_inner());
                let result = serve_move_replay(&mut net, cfg, &board, me, n_iters);
                drop(net);
                result
            }
        }
    } else {
        let mut net = net.lock().unwrap_or_else(|p| p.into_inner());
        let result = serve_move_replay(&mut net, cfg, &board, me, n_iters);
        drop(net);
        result
    };

    let diag = &result.decision.diagnostics;
    let resp = TreeResponse {
        turn,
        you: me,
        recorded_move: mv.get("chosen_move").and_then(Value::as_u64),
        replay_move: result.decision.move_index,
        sims: diag.sims_completed,
        requested_sims: n_iters,
        terminal_only_sims: diag.terminal_only_sims,
        forward_calls: diag.forward_calls,
        eval_rows: diag.eval_rows,
        root_policy: diag.root_policy.clone(),
        root_values: diag.root_values.clone(),
        model_match: recorded_model_sha.as_deref() == Some(replay_model_sha.as_str()),
        recorded_model_sha,
        server_model_sha: replay_model_sha,
        tree: result.tree.as_ref().map(TreeOut::from),
    };
    ViewerResponse::ok(serde_json::to_string(&resp).unwrap())
}

/// Extract a single query parameter value (no percent-decoding needed for our
/// numeric params).
fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == key).then(|| v.to_string())
    })
}
