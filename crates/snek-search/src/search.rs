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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::obvious_immediate_death;
    use snek_core::{EliminatedCause, Move, Point};

    // The suicide mask is the last line of defence between the τ-softened
    // equilibrium and certain death — every gap here becomes real blunders in
    // self-play (58% of le-4 deaths; the opponent-head-cell gap in le-6).
    // Battery over every "certain vs merely dangerous" distinction.

    fn me(cells: &[(i8, i8)]) -> Board {
        let mut b = Board::new(7, 7);
        b.add_snake(&cells.iter().map(|&(x, y)| Point::new(x, y)).collect::<Vec<_>>());
        b
    }

    fn with_opponent(mut b: Board, cells: &[(i8, i8)]) -> Board {
        b.add_snake(&cells.iter().map(|&(x, y)| Point::new(x, y)).collect::<Vec<_>>());
        b
    }

    #[test]
    fn walls_are_certain_death() {
        // Head in the bottom-left corner: both wall moves masked, others not.
        let b = me(&[(0, 0), (1, 0), (2, 0)]);
        assert!(obvious_immediate_death(&b, 0, Move::Down));
        assert!(obvious_immediate_death(&b, 0, Move::Left));
        assert!(!obvious_immediate_death(&b, 0, Move::Up));
    }

    #[test]
    fn own_body_is_certain_death_but_own_tail_vacates() {
        // Coil: head (2,2), body up over (2,3),(3,3),(3,2); tail (3,2) vacates.
        let b = me(&[(2, 2), (2, 3), (3, 3), (3, 2)]);
        assert!(obvious_immediate_death(&b, 0, Move::Up), "own neck");
        assert!(
            !obvious_immediate_death(&b, 0, Move::Right),
            "own tail vacates this turn — legal"
        );
    }

    #[test]
    fn own_tail_after_eating_does_not_vacate() {
        // Same coil but the tail segment is DUPLICATED (just ate): the pop
        // removes one twin, the cell stays occupied — certain death.
        let b = me(&[(2, 2), (2, 3), (3, 3), (3, 2), (3, 2)]);
        assert!(obvious_immediate_death(&b, 0, Move::Right));
    }

    #[test]
    fn opponent_tail_after_eating_does_not_vacate() {
        let b = me(&[(2, 2), (2, 1), (1, 1)]);
        // Opponent tail at (3,2), duplicated (just ate) — moving there dies.
        let b = with_opponent(b, &[(5, 2), (4, 2), (3, 2), (3, 2)]);
        assert!(obvious_immediate_death(&b, 0, Move::Right));
        // Un-duplicated control: tail vacates, move is legal.
        let b2 = me(&[(2, 2), (2, 1), (1, 1)]);
        let b2 = with_opponent(b2, &[(5, 2), (4, 2), (3, 2)]);
        assert!(!obvious_immediate_death(&b2, 0, Move::Right));
    }

    #[test]
    fn dead_opponents_do_not_block() {
        let b = me(&[(2, 2), (2, 1), (1, 1)]);
        let mut b = with_opponent(b, &[(3, 2), (4, 2), (5, 2)]);
        b.snakes[1].eliminated = Some(EliminatedCause::Collision);
        assert!(!obvious_immediate_death(&b, 0, Move::Right));
    }

    #[test]
    fn le_candidates_excludes_certain_deaths() {
        // Head (2,2): up = own neck (excluded by candidates already), right =
        // opponent mid-body (masked), left/down open.
        let b = me(&[(2, 2), (2, 3), (2, 4)]);
        let b = with_opponent(b, &[(3, 3), (3, 2), (3, 1), (4, 1)]);
        let c = le_candidates(&b, 0);
        assert!(c.contains(&Move::Left) && c.contains(&Move::Down));
        assert!(!c.contains(&Move::Right), "opponent standing body masked");
        assert!(!c.contains(&Move::Up), "neck reversal excluded");
    }

    #[test]
    fn le_candidates_trapped_falls_back_to_nonempty() {
        // Corner cell with own body sealing both open sides: every move is
        // fatal, but the candidate set must never be empty (no NaN policies —
        // the doomed snake still plays a move).
        let b = me(&[(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)]);
        let c = le_candidates(&b, 0);
        assert!(!c.is_empty());
        for m in &c {
            assert!(obvious_immediate_death(&b, 0, *m), "trapped: all fatal");
        }
    }
}
