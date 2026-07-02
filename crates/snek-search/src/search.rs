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

/// Exact per-agent value at a terminal board: winner +1, losers -1, draw configurable.
pub(crate) fn terminal_values_with_draw(board: &Board, draw_value: f32) -> [f32; MAX_SNAKES] {
    let mut v = [0.0f32; MAX_SNAKES];
    match board.winner() {
        Some(w) => {
            for (i, value) in v.iter_mut().enumerate().take(board.snakes.len()) {
                *value = if i == w { 1.0 } else { -1.0 };
            }
        }
        None => {
            for x in v.iter_mut().take(board.snakes.len()) {
                *x = draw_value;
            }
        }
    }
    v
}
