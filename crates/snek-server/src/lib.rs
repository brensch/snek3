//! Pure-Rust AlphaZero serving: decoupled-PUCT MCTS over a single ONNX
//! policy+value net — the *same* search used in self-play (`crates/snek-search`
//! `MctsForest`), so what we serve matches what we trained.
//!
//! [`serve_move`] is stateless per move (board + our index only): no opponent
//! modelling, no temperature — the policy/value net plus the tree search handle
//! everything. The HTTP `/move` handler (`main.rs`) routes through it.

use snek_core::{obs_h, obs_w, Board, Move, NUM_CHANNELS};
use snek_infer::Net;
use snek_search::MctsForest;

pub struct Config {
    /// MCTS simulations per move (the inference-time search budget).
    pub sims: usize,
    /// PUCT exploration constant (match self-play for train/serve parity).
    pub c_puct: f32,
    /// Terminal value of a draw at search leaves (match training).
    pub draw_value: f32,
    /// Max leaf observations per ONNX forward pass.
    pub eval_chunk: usize,
}

pub const MOVES: [&str; 4] = ["up", "down", "left", "right"];

pub fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// First non-suicidal move for snake `me` (off-board / reverse-onto-neck dropped),
/// falling back to Up. Used when the search can't run (terminal) or returns empty.
pub fn safe_move(board: &Board, me: usize) -> usize {
    let s = &board.snakes[me];
    let head = s.head();
    let neck = if s.len() >= 2 { Some(s.body.get(1)) } else { None };
    for m in Move::ALL {
        let nh = m.apply(head);
        if Some(nh) != neck && board.in_bounds(nh) {
            return m.index();
        }
    }
    Move::Up.index()
}

/// One AlphaZero move for snake `me` on `board`: run `cfg.sims` decoupled-PUCT
/// simulations (leaves evaluated by `net`), then play the most-visited root
/// action. Identical search to self-play, so serving cannot diverge from training.
pub fn serve_move(net: &mut Net, cfg: &Config, board: &Board, me: usize) -> usize {
    if board.is_terminal() || !board.snakes[me].alive() {
        return safe_move(board, me);
    }
    let (c, h, w) = (NUM_CHANNELS, obs_h(board), obs_w(board));
    let n_snakes = board.snakes.len();
    let mut forest =
        MctsForest::new_with_draw_value(std::slice::from_ref(board), cfg.c_puct, cfg.draw_value);
    let obs_size = forest.obs_size();

    for _ in 0..cfg.sims {
        let pending = forest.select();
        if pending.is_empty() {
            break; // tree fully resolved (all terminal)
        }
        // Each pending leaf needs one egocentric encoding per snake (per-snake
        // policy/value), laid out [pending, agent]. Total rows = pending * n.
        let rows = pending.len() * n_snakes;
        let mut obs = vec![0.0f32; rows * obs_size];
        forest.write_pending_obs(&pending, &mut obs);

        let mut pol = vec![0.0f32; rows * 4];
        let mut val = vec![0.0f32; rows];
        let mut s = 0;
        while s < rows {
            let e = (s + cfg.eval_chunk).min(rows);
            match net.forward(&obs[s * obs_size..e * obs_size], e - s, c, h, w) {
                Ok((p, v)) => {
                    pol[s * 4..e * 4].copy_from_slice(&p);
                    val[s..e].copy_from_slice(&v);
                }
                Err(_) => return safe_move(board, me),
            }
            s = e;
        }
        forest.expand_backup(&pending, &pol, &val);
    }

    // root_targets: visit-count policy [count*N*4]; count == 1 here.
    let (policies, _values) = forest.root_targets();
    let slots = &policies[me * 4..me * 4 + 4];
    if slots.iter().sum::<f32>() <= 1e-8 {
        return safe_move(board, me);
    }
    slots
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, _)| k)
        .unwrap_or(0)
}
