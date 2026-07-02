//! Board / move helpers the search and play loop share: legal candidate moves,
//! obvious-death masking, terminal valuation, and the exploration sampler.
//! (Formerly in snek-search; inlined so self-play owns its search.)

use super::{EPS, MAXC};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use snek_core::{Board, Move, MAX_SNAKES};

/// Legal candidate move indices (0..4) for snake `i`, plus their count. Drops
/// reversing-onto-neck and off-board moves; a trapped snake keeps all four.
#[inline]
pub(super) fn candidates(board: &Board, i: usize) -> ([u8; MAXC], usize) {
    let mut out = [0u8; MAXC];
    let s = &board.snakes[i];
    if !s.alive() {
        return (out, 1); // dummy move (Up); ignored by step
    }
    let head = s.head();
    let neck = if s.len() >= 2 {
        Some(s.body.get(1))
    } else {
        None
    };
    let mut k = 0usize;
    for m in Move::ALL {
        let nh = m.apply(head);
        if Some(nh) == neck || !board.in_bounds(nh) {
            continue;
        }
        out[k] = m.index() as u8;
        k += 1;
    }
    if k == 0 {
        for (idx, m) in Move::ALL.iter().enumerate() {
            out[idx] = m.index() as u8;
        }
        k = 4;
    }
    (out, k)
}

#[inline]
pub(super) fn terminal_values(board: &Board, draw: f32) -> [f32; MAX_SNAKES] {
    let mut v = [0.0f32; MAX_SNAKES];
    match board.winner() {
        Some(w) => {
            for (i, x) in v.iter_mut().enumerate().take(board.snakes.len()) {
                *x = if i == w { 1.0 } else { -1.0 };
            }
        }
        None => {
            for x in v.iter_mut().take(board.snakes.len()) {
                *x = draw;
            }
        }
    }
    v
}

/// Would `mv` walk snake `snake_idx` straight into a wall, its own body, or a
/// non-tail segment of another snake? Used to mask hopeless priors/plays.
pub(super) fn obvious_immediate_death(board: &Board, snake_idx: usize, mv: Move) -> bool {
    let Some(snake) = board.snakes.get(snake_idx) else {
        return false;
    };
    if !snake.alive() || snake.body.is_empty() {
        return false;
    }
    let next = mv.apply(snake.head());
    if !board.in_bounds(next) {
        return true;
    }
    let mut body = snake.body;
    body.advance(next);
    if body.collides_excluding_head(next) {
        return true;
    }
    for (i, other) in board.snakes.iter().enumerate() {
        if i == snake_idx || !other.alive() {
            continue;
        }
        for j in 1..other.len().saturating_sub(1) {
            if other.body.get(j) == next {
                return true;
            }
        }
    }
    false
}

/// Renormalise `probs` (4 moves) onto the non-obviously-fatal moves.
pub(super) fn mask_obvious_immediate_deaths(
    board: &Board,
    snake_idx: usize,
    probs: &[f32],
) -> [f32; 4] {
    let mut original = [0.0f32; 4];
    let mut total = 0.0f32;
    for (i, o) in original.iter_mut().enumerate() {
        *o = probs.get(i).copied().unwrap_or(0.0).max(0.0);
        total += *o;
    }
    if total <= EPS
        || board
            .snakes
            .get(snake_idx)
            .map(|s| !s.alive())
            .unwrap_or(true)
    {
        return original;
    }
    let mut out = [0.0f32; 4];
    let mut safe_mass = 0.0f32;
    let mut safe_count = 0usize;
    for i in 0..4 {
        if !obvious_immediate_death(board, snake_idx, Move::from_index(i)) {
            safe_count += 1;
            out[i] = original[i];
            safe_mass += original[i];
        }
    }
    if safe_count == 0 {
        return original;
    }
    if safe_mass > EPS {
        for x in out.iter_mut() {
            *x /= safe_mass;
        }
    } else {
        let u = 1.0 / safe_count as f32;
        for (i, o) in out.iter_mut().enumerate() {
            if !obvious_immediate_death(board, snake_idx, Move::from_index(i)) {
                *o = u;
            }
        }
    }
    out
}

/// Training value for one sample: winner +1, losers -1, draw configurable.
#[inline]
pub(super) fn terminal_value(
    winner: Option<usize>,
    snake: usize,
    alive_final: bool,
    draw_value: f32,
) -> f32 {
    match winner {
        Some(w) if w == snake => 1.0,
        Some(_) => -1.0,
        None if alive_final => draw_value,
        None => -1.0,
    }
}

/// Sample a move from the play policy, mixing in a uniform exploration floor over
/// the legal (non-zero) moves.
pub(super) fn sample_move<R: Rng>(policy: &[f32], exploration_prob: f32, rng: &mut R) -> Move {
    let mut p = [0.0f32; 4];
    let legal = policy.iter().filter(|&&v| v > 0.0).count().max(1);
    for i in 0..4 {
        p[i] = if policy[i] > 0.0 {
            (1.0 - exploration_prob) * policy[i] + exploration_prob / legal as f32
        } else {
            0.0
        };
    }
    let idx = WeightedIndex::new(p)
        .map(|d| d.sample(rng))
        .unwrap_or_else(|_| rng.gen_range(0..4));
    Move::from_index(idx)
}
