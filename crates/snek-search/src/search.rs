//! Shared board helpers for the MCTS search: per-snake legal-move candidates and
//! exact terminal values.
//!
//! These were extracted from the former fixed-depth Logit-Equilibrium `Forest`
//! search (now removed); live self-play and serving both use
//! [`crate::mcts::MctsForest`].

use snek_core::{Board, Move, MAX_SNAKES};

/// Placeholder move for eliminated snakes (ignored by `step`).
const DUMMY_MOVE: Move = Move::Up;

/// Candidate moves for one snake: drop strictly-dominated suicides (reversing
/// onto the neck, stepping off the board). A trapped snake keeps all moves (it
/// dies regardless). Eliminated snakes get a single dummy move.
pub(crate) fn candidates(board: &Board, i: usize) -> Vec<Move> {
    let s = &board.snakes[i];
    if !s.alive() {
        return vec![DUMMY_MOVE];
    }
    let head = s.head();
    let neck = if s.len() >= 2 {
        Some(s.body.get(1))
    } else {
        None
    };
    let mut v = Vec::with_capacity(4);
    for m in Move::ALL {
        let nh = m.apply(head);
        if Some(nh) == neck || !board.in_bounds(nh) {
            continue;
        }
        v.push(m);
    }
    if v.is_empty() {
        v.extend_from_slice(&Move::ALL);
    }
    v
}

/// Like [`candidates`] but also drops moves that are certain immediate death
/// (into a standing body or wall). Used ONLY by the LE search: a suicide move is
/// strictly dominated, so the equilibrium must never spend probability on it —
/// without this the LE solve assigns 30-46% to walking into a wall at any tau
/// (the collapsed value rates a small snake's alive ~ dead), and self-play
/// becomes mostly random crashes. Falls back to the plain candidate set when
/// EVERY move is fatal (trapped) so the snake still has a (doomed) move.
/// (The AZ path keeps the unfiltered `candidates` + masks the prior instead.)
pub(crate) fn le_candidates(board: &Board, i: usize) -> Vec<Move> {
    let base = candidates(board, i);
    let safe: Vec<Move> = base
        .iter()
        .copied()
        .filter(|&m| !crate::mcts::obvious_immediate_death(board, i, m))
        .collect();
    if safe.is_empty() {
        base
    } else {
        safe
    }
}

/// Zero-sum per-agent loss for a non-winner: the `n-1` losers share `-1/(n-1)`
/// so a game sums to 0. A flat -1 would be 3:1 loss-dominated in 4p and saturate
/// the tanh value head at -1 (see the training-side `loser_value`).
#[inline]
pub(crate) fn loser_value(n: usize) -> f32 {
    -1.0 / (n.max(2) - 1) as f32
}

/// Exact per-agent value at a terminal board: winner +1, losers share -1/(n-1),
/// draw configurable.
pub(crate) fn terminal_values_with_draw(board: &Board, draw_value: f32) -> [f32; MAX_SNAKES] {
    let mut v = [0.0f32; MAX_SNAKES];
    let n = board.snakes.len();
    match board.winner() {
        Some(w) => {
            for (i, value) in v.iter_mut().enumerate().take(n) {
                *value = if i == w { 1.0 } else { loser_value(n) };
            }
        }
        None => {
            for x in v.iter_mut().take(n) {
                *x = draw_value;
            }
        }
    }
    v
}
