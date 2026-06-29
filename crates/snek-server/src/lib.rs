//! Pure-Rust AlphaZero serving: decoupled-PUCT MCTS over a single ONNX
//! policy+value net — the *same* search used in self-play (`crates/snek-search`
//! `MctsForest`), so what we serve matches what we trained.
//!
//! [`serve_move_until`] is stateless per move (board + our index only): no opponent
//! modelling, no temperature — the policy/value net plus the tree search handle
//! everything. The HTTP `/move` handler (`main.rs`) routes through it.

use std::time::Instant;

use snek_core::{obs_h, obs_w, Board, Move, NUM_CHANNELS};
use snek_infer::Net;
use snek_search::MctsForest;

#[derive(Clone, Debug)]
pub struct RootActionDebug {
    pub move_index: usize,
    pub prior: f32,
    pub visits: f32,
    pub q: f32,
}

#[derive(Clone, Debug)]
pub struct SearchDiagnostics {
    pub sims_completed: usize,
    pub eval_rows: usize,
    pub forward_calls: usize,
    pub stopped_reason: &'static str,
    pub fallback_reason: Option<&'static str>,
    pub root_policy: Vec<f32>,
    pub root_values: Vec<f32>,
    pub root_actions: Vec<Vec<RootActionDebug>>,
}

#[derive(Clone, Debug)]
pub struct SearchDecision {
    pub move_index: usize,
    pub diagnostics: SearchDiagnostics,
}

pub struct Config {
    /// Safety cap on MCTS simulations. Live serving is deadline-bound first.
    pub max_sims: usize,
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

fn fallback_decision(board: &Board, me: usize, reason: &'static str) -> SearchDecision {
    SearchDecision {
        move_index: safe_move(board, me),
        diagnostics: SearchDiagnostics {
            sims_completed: 0,
            eval_rows: 0,
            forward_calls: 0,
            stopped_reason: reason,
            fallback_reason: Some(reason),
            root_policy: Vec::new(),
            root_values: Vec::new(),
            root_actions: Vec::new(),
        },
    }
}

/// One AlphaZero move for snake `me` on `board`: run decoupled-PUCT until the
/// request deadline or `cfg.max_sims`, then play the most-visited root action.
/// Identical search to self-play, so serving cannot diverge from training.
pub fn serve_move_until_diagnostics(
    net: &mut Net,
    cfg: &Config,
    board: &Board,
    me: usize,
    deadline: Instant,
) -> SearchDecision {
    if board.is_terminal() || !board.snakes[me].alive() {
        return fallback_decision(board, me, "terminal_or_dead");
    }
    let (c, h, w) = (NUM_CHANNELS, obs_h(board), obs_w(board));
    let n_snakes = board.snakes.len();
    let mut forest =
        MctsForest::new_with_draw_value(std::slice::from_ref(board), cfg.c_puct, cfg.draw_value);
    let obs_size = forest.obs_size();
    let mut sims_completed = 0usize;
    let mut eval_rows = 0usize;
    let mut forward_calls = 0usize;
    let mut stopped_reason = "max_sims";

    for _ in 0..cfg.max_sims {
        if Instant::now() >= deadline {
            stopped_reason = "deadline";
            break;
        }
        let pending = forest.select();
        if pending.is_empty() {
            stopped_reason = "tree_resolved";
            break; // tree fully resolved (all terminal)
        }
        // Each pending leaf needs one egocentric encoding per snake (per-snake
        // policy/value), laid out [pending, agent]. Total rows = pending * n.
        let rows = pending.len() * n_snakes;
        eval_rows += rows;
        let mut obs = vec![0.0f32; rows * obs_size];
        forest.write_pending_obs(&pending, &mut obs);

        let mut pol = vec![0.0f32; rows * 4];
        let mut val = vec![0.0f32; rows];
        let mut s = 0;
        while s < rows {
            let e = (s + cfg.eval_chunk).min(rows);
            forward_calls += 1;
            match net.forward(&obs[s * obs_size..e * obs_size], e - s, c, h, w) {
                Ok((p, v)) => {
                    pol[s * 4..e * 4].copy_from_slice(&p);
                    val[s..e].copy_from_slice(&v);
                }
                Err(_) => return fallback_decision(board, me, "net_forward_error"),
            }
            s = e;
        }
        forest.expand_backup(&pending, &pol, &val);
        sims_completed += 1;
    }

    // root_targets: visit-count policy [count*N*4]; count == 1 here.
    let (policies, values) = forest.root_targets();
    let root_actions = forest
        .root_debug_first()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(move_index, prior, visits, q)| RootActionDebug {
                    move_index,
                    prior,
                    visits,
                    q,
                })
                .collect()
        })
        .collect::<Vec<_>>();
    let slots = &policies[me * 4..me * 4 + 4];
    if slots.iter().sum::<f32>() <= 1e-8 {
        let mut decision = fallback_decision(board, me, "empty_root_policy");
        decision.diagnostics.sims_completed = sims_completed;
        decision.diagnostics.eval_rows = eval_rows;
        decision.diagnostics.forward_calls = forward_calls;
        decision.diagnostics.stopped_reason = stopped_reason;
        decision.diagnostics.root_policy = policies;
        decision.diagnostics.root_values = values;
        decision.diagnostics.root_actions = root_actions;
        return decision;
    }
    let move_index = slots
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, _)| k)
        .unwrap_or(0);
    SearchDecision {
        move_index,
        diagnostics: SearchDiagnostics {
            sims_completed,
            eval_rows,
            forward_calls,
            stopped_reason,
            fallback_reason: None,
            root_policy: policies,
            root_values: values,
            root_actions,
        },
    }
}

pub fn serve_move_until(
    net: &mut Net,
    cfg: &Config,
    board: &Board,
    me: usize,
    deadline: Instant,
) -> usize {
    serve_move_until_diagnostics(net, cfg, board, me, deadline).move_index
}

/// Fixed-budget wrapper for tests/tools that do not have a request deadline.
pub fn serve_move(net: &mut Net, cfg: &Config, board: &Board, me: usize) -> usize {
    serve_move_until(
        net,
        cfg,
        board,
        me,
        Instant::now() + std::time::Duration::from_secs(3600),
    )
}
