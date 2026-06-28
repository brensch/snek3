//! Shared Albatross-faithful serving logic, used by both the `/move` HTTP server
//! (`main.rs`) and the offline evaluator (`bin/eval.rs`).
//!
//! The heart is [`serve_move`]: the exact test-time procedure from
//! `train_albatross.py` — online MLE of each opponent's temperature under the
//! proxy net, then a heterogeneous-temperature logit-equilibrium best-response
//! search with our snake pinned rational and each opponent pinned at its MLE
//! tau. The HTTP server and the evaluator call this same function, so the
//! evaluator measures *exactly* what gets deployed.
//!
//! [`serve_move`] takes `net`/`grid`/`cfg` explicitly (rather than reaching into
//! a shared struct) so the evaluator can give each worker thread its own [`Net`]
//! and run games in parallel without serialising on a single inference mutex.

use std::collections::HashMap;

use snek_core::{encode_into, obs_side, Board, Move, Point, NUM_CHANNELS};
use snek_infer::Net;
use snek_search::Forest;

/// Log-spaced temperature grid for the opponent-tau MLE — geomspace(0.25, 20, 24),
/// matching `TAU_GRID` in `train_albatross.py`.
pub const TAU_GRID_LEN: usize = 24;
pub fn tau_grid() -> [f32; TAU_GRID_LEN] {
    let (lo, hi) = (0.25f64, 20.0f64);
    let mut g = [0.0f32; TAU_GRID_LEN];
    for (i, slot) in g.iter_mut().enumerate() {
        let t = i as f64 / (TAU_GRID_LEN as f64 - 1.0);
        *slot = (lo * (hi / lo).powf(t)) as f32;
    }
    g
}

pub struct Config {
    pub depth: u32,
    pub iters: usize,
    pub response_tau: f32,
    pub draw_value: f32,
    pub eval_chunk: usize,
}

/// Per-game memory used for online opponent modelling. Keyed by opponent snake id
/// so it survives other snakes dying (which shifts board indices).
#[derive(Default)]
pub struct GameState {
    last_board: Option<Board>,
    last_ids: Vec<String>,
    /// Cumulative log-likelihood of each opponent's observed moves, per grid tau.
    opp_ll: HashMap<String, [f64; TAU_GRID_LEN]>,
}

pub const MOVES: [&str; 4] = ["up", "down", "left", "right"];

pub fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Map a one-cell head displacement to a move index, or None if not a unit step.
pub fn move_from_delta(from: Point, to: Point) -> Option<usize> {
    match (to.x - from.x, to.y - from.y) {
        (0, 1) => Some(Move::Up.index()),
        (0, -1) => Some(Move::Down.index()),
        (-1, 0) => Some(Move::Left.index()),
        (1, 0) => Some(Move::Right.index()),
        _ => None,
    }
}

/// First non-suicidal move for snake `me` (off-board / reverse-onto-neck dropped),
/// falling back to Up. Used when the search can't run (already terminal) or the
/// equilibrium policy is empty.
pub fn safe_move(board: &Board, me: usize) -> usize {
    let s = &board.snakes[me];
    let head = s.head();
    let neck = if s.len() >= 2 {
        Some(s.body.get(1))
    } else {
        None
    };
    for m in Move::ALL {
        let nh = m.apply(head);
        if Some(nh) != neck && board.in_bounds(nh) {
            return m.index();
        }
    }
    Move::Up.index()
}

/// One Albatross-faithful move for snake `me` on `board`, given the snake ids in
/// board order and this game's accumulated opponent model `gs`. Mutates `gs` with
/// the latest opponent-move likelihoods and remembers the board for next turn.
///
/// This is the deployed serving procedure; both the HTTP `/move` handler and the
/// offline evaluator route through it so they cannot diverge.
pub fn serve_move(
    net: &mut Net,
    grid: &[f32; TAU_GRID_LEN],
    cfg: &Config,
    board: &Board,
    ids: &[String],
    me: usize,
    gs: &mut GameState,
) -> usize {
    let n = board.snakes.len();
    if ids.len() != n {
        // Couldn't align ids; still return a safe move.
        return safe_move(board, me);
    }

    let c = NUM_CHANNELS;
    let h = obs_side(board.width as usize);
    let w = obs_side(board.height as usize);
    let obs_size = c * h * w;
    let default_tau = (grid[0] * grid[TAU_GRID_LEN - 1]).sqrt();

    // ---- 1. Opponent modelling: score each opponent's latest move (MLE update) ----
    if let Some(prev) = gs.last_board.take() {
        for (j, id) in ids.iter().enumerate() {
            if j == me {
                continue;
            }
            // Find this opponent in the previous board by id.
            let Some(pj) = gs.last_ids.iter().position(|x| x == id) else {
                continue;
            };
            if pj >= prev.snakes.len() || !prev.snakes[pj].alive() || !board.snakes[j].alive() {
                continue;
            }
            let Some(midx) = move_from_delta(prev.snakes[pj].head(), board.snakes[j].head()) else {
                continue;
            };
            // Encode the state the opponent acted from, evaluate the proxy policy at
            // every grid temperature, and add log p(observed move) to that tau's LL.
            let mut obs = vec![0.0f32; TAU_GRID_LEN * obs_size];
            for t in 0..TAU_GRID_LEN {
                encode_into(&prev, pj, &mut obs[t * obs_size..(t + 1) * obs_size]);
            }
            let temps: Vec<f32> = grid.to_vec();
            let pol = match net.forward_temp(&obs, Some(&temps), TAU_GRID_LEN, c, h, w) {
                Ok((p, _)) => p,
                Err(_) => continue,
            };
            let ll = gs.opp_ll.entry(id.clone()).or_insert([0.0; TAU_GRID_LEN]);
            for t in 0..TAU_GRID_LEN {
                let p = pol[t * 4 + midx].max(1e-9);
                ll[t] += (p as f64).ln();
            }
        }
    }

    // ---- 2. Per-agent temperatures: us rational, each opponent at its MLE tau ----
    let mut tau_vec = vec![cfg.response_tau; n];
    let mut opp_taus: Vec<f32> = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        if i == me {
            continue;
        }
        let tau = gs
            .opp_ll
            .get(id)
            .map(|ll| {
                let best = ll
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(k, _)| k)
                    .unwrap_or(0);
                grid[best]
            })
            .unwrap_or(default_tau);
        tau_vec[i] = tau;
        opp_taus.push(tau);
    }
    // Leaf value net is conditioned on the opponents' rationality regime (mean
    // opponent tau), matching the response search in train_albatross.py.
    let leaf_temp = if opp_taus.is_empty() {
        cfg.response_tau
    } else {
        opp_taus.iter().sum::<f32>() / opp_taus.len() as f32
    };

    // ---- 3. Hetero-temperature equilibrium best-response search ----
    let mut forest = Forest::build(std::slice::from_ref(board), cfg.depth, cfg.draw_value);
    let ec = forest.eval_count();
    let mv = if ec == 0 {
        safe_move(board, me)
    } else {
        let mut leaf_obs = vec![0.0f32; ec * obs_size];
        forest.write_observations(&mut leaf_obs);
        let mut values = vec![0.0f32; ec];
        let mut s = 0usize;
        let mut ok = true;
        while s < ec {
            let e = (s + cfg.eval_chunk).min(ec);
            let temps = vec![leaf_temp; e - s];
            match net.forward_temp(&leaf_obs[s * obs_size..e * obs_size], Some(&temps), e - s, c, h, w)
            {
                Ok((_p, val)) => values[s..e].copy_from_slice(&val),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
            s = e;
        }
        if !ok {
            safe_move(board, me)
        } else {
            let (root_pol, _root_val) = forest.backup(&values, &tau_vec, cfg.iters);
            let slots = &root_pol[me * 4..me * 4 + 4];
            if slots.iter().sum::<f32>() <= 1e-8 {
                safe_move(board, me)
            } else {
                slots
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(k, _)| k)
                    .unwrap_or(0)
            }
        }
    };

    // ---- 4. Remember this turn for next turn's MLE ----
    gs.last_board = Some(board.clone());
    gs.last_ids = ids.to_vec();

    mv
}

/// Replay-frame JSON for one board state. Mirrors `board_snapshot_value` in
/// `snek-py` so eval replays render in the same dashboard board view.
pub fn board_snapshot_value(board: &Board) -> serde_json::Value {
    let food: Vec<[i8; 2]> = board.food.iter().map(|p| [p.x, p.y]).collect();
    let hazards: Vec<[i8; 2]> = board.hazards.iter().map(|p| [p.x, p.y]).collect();
    let snakes: Vec<serde_json::Value> = board
        .snakes
        .iter()
        .map(|s| {
            let body: Vec<[i8; 2]> = s.body.iter().map(|p| [p.x, p.y]).collect();
            serde_json::json!({"alive": s.alive(), "health": s.health, "body": body})
        })
        .collect();
    serde_json::json!({
        "turn": board.turn,
        "width": board.width,
        "height": board.height,
        "food": food,
        "hazards": hazards,
        "snakes": snakes,
    })
}
