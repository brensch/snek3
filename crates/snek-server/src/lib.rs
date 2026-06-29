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
use snek_search::{MctsForest, TreeSnapshot};

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
    pub terminal_only_sims: usize,
    pub eval_rows: usize,
    pub forward_calls: usize,
    /// Max search-tree depth reached (root = 0).
    pub max_depth: u32,
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
            terminal_only_sims: 0,
            eval_rows: 0,
            forward_calls: 0,
            max_depth: 0,
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
    let mut terminal_only_sims = 0usize;
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
            terminal_only_sims += 1;
            sims_completed += 1;
            continue;
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
    let max_depth = forest.max_depth_first();
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
        decision.diagnostics.terminal_only_sims = terminal_only_sims;
        decision.diagnostics.eval_rows = eval_rows;
        decision.diagnostics.forward_calls = forward_calls;
        decision.diagnostics.max_depth = max_depth;
        decision.diagnostics.stopped_reason = stopped_reason;
        decision.diagnostics.root_policy = policies;
        decision.diagnostics.root_values = values;
        decision.diagnostics.root_actions = root_actions;
        return decision;
    }
    let move_index = choose_root_action(
        slots,
        root_actions.get(me).map(Vec::as_slice).unwrap_or(&[]),
    );
    SearchDecision {
        move_index,
        diagnostics: SearchDiagnostics {
            sims_completed,
            terminal_only_sims,
            eval_rows,
            forward_calls,
            max_depth,
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

/// A faithful replay of a recorded move plus the full exploration tree.
#[derive(Clone, Debug)]
pub struct ReplayResult {
    pub decision: SearchDecision,
    pub tree: Option<TreeSnapshot>,
}

/// Re-run the search for a recorded position bounded by an *exact* simulation
/// count rather than a wall-clock deadline. Because serving search is fully
/// deterministic (strict-argmax DUCT, no noise/temperature), running the same
/// `n_iters` the live move managed reproduces that move's tree node-for-node —
/// this is what the viewer's tree explorer renders. Returns the decision, the
/// same diagnostics the live move recorded, and the captured tree.
pub fn serve_move_replay(
    net: &mut Net,
    cfg: &Config,
    board: &Board,
    me: usize,
    n_iters: usize,
) -> ReplayResult {
    if board.is_terminal() || !board.snakes[me].alive() {
        return ReplayResult {
            decision: fallback_decision(board, me, "terminal_or_dead"),
            tree: None,
        };
    }
    let (c, h, w) = (NUM_CHANNELS, obs_h(board), obs_w(board));
    let n_snakes = board.snakes.len();
    let mut forest =
        MctsForest::new_with_draw_value(std::slice::from_ref(board), cfg.c_puct, cfg.draw_value);
    let obs_size = forest.obs_size();
    let mut sims_completed = 0usize;
    let mut terminal_only_sims = 0usize;
    let mut eval_rows = 0usize;
    let mut forward_calls = 0usize;

    for _ in 0..n_iters {
        let pending = forest.select();
        if pending.is_empty() {
            terminal_only_sims += 1;
            sims_completed += 1;
            continue;
        }
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
                Err(_) => {
                    return ReplayResult {
                        decision: fallback_decision(board, me, "net_forward_error"),
                        tree: None,
                    }
                }
            }
            s = e;
        }
        forest.expand_backup(&pending, &pol, &val);
        sims_completed += 1;
    }

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
    let max_depth = forest.max_depth_first();
    let tree = forest.tree_snapshot_first();
    let move_index = choose_root_action(
        &policies[me * 4..me * 4 + 4],
        root_actions.get(me).map(Vec::as_slice).unwrap_or(&[]),
    );
    ReplayResult {
        decision: SearchDecision {
            move_index,
            diagnostics: SearchDiagnostics {
                sims_completed,
                terminal_only_sims,
                eval_rows,
                forward_calls,
                max_depth,
                stopped_reason: "replay_n_iters",
                fallback_reason: None,
                root_policy: policies,
                root_values: values,
                root_actions,
            },
        },
        tree,
    }
}

fn choose_root_action(policy_slots: &[f32], actions: &[RootActionDebug]) -> usize {
    let has_positive_prior = actions.iter().any(|a| a.prior > 1e-8);
    let mut best: Option<&RootActionDebug> = None;
    for action in actions {
        if action.visits <= 0.0 {
            continue;
        }
        if has_positive_prior && action.prior <= 1e-8 {
            continue;
        }
        let replace = match best {
            None => true,
            Some(current) => action
                .visits
                .total_cmp(&current.visits)
                .then_with(|| action.q.total_cmp(&current.q))
                .then_with(|| action.prior.total_cmp(&current.prior))
                .is_gt(),
        };
        if replace {
            best = Some(action);
        }
    }
    if let Some(action) = best {
        return action.move_index;
    }

    let mut best_idx = 0usize;
    let mut best_prob = f32::NEG_INFINITY;
    for (idx, &prob) in policy_slots.iter().enumerate() {
        if prob > best_prob {
            best_idx = idx;
            best_prob = prob;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_root_action_ignores_masked_death_in_visit_tie() {
        let actions = vec![
            RootActionDebug {
                move_index: 0,
                prior: 0.62,
                visits: 1.0,
                q: -0.67,
            },
            RootActionDebug {
                move_index: 1,
                prior: 0.38,
                visits: 1.0,
                q: -0.60,
            },
            RootActionDebug {
                move_index: 3,
                prior: 0.0,
                visits: 1.0,
                q: -1.0,
            },
        ];
        assert_eq!(
            choose_root_action(&[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0], &actions),
            1
        );
    }

    #[test]
    fn choose_root_action_falls_back_to_policy_without_debug_actions() {
        assert_eq!(choose_root_action(&[0.1, 0.2, 0.7, 0.0], &[]), 2);
    }
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
