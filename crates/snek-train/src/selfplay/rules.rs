//! Board / move helpers the search and play loop share: legal candidate moves,
//! obvious-death masking, terminal valuation, and the exploration sampler.
//! (Formerly in snek-search; inlined so self-play owns its search.)

use super::{EPS, MAXC};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use rand_distr::Dirichlet;
use snek_core::{Board, Move, MAX_SNAKES};

/// Legal candidate move indices (0..4) for snake `i`, plus their count. Drops
/// reversing-onto-neck and off-board moves; a trapped snake keeps all four.
#[inline]
pub(crate) fn candidates(board: &Board, i: usize) -> ([u8; MAXC], usize) {
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

/// Zero-sum per-seat loss for a finished game: the sole survivor gets +1 and the
/// other `n-1` seats share the loss at `-1/(n-1)`, so the game sums to 0. This is
/// deliberately NOT a flat -1: in a 4-player free-for-all a +1/-1 outcome is 3:1
/// loss-dominated (mean -0.5), which drags the tanh value head onto the -1 rail
/// where its gradient vanishes and the value signal dies. Zero-mean keeps it
/// centred and using its full range so LE-search leaves actually differ.
#[inline]
pub(crate) fn loser_value(n: usize) -> f32 {
    -1.0 / (n.max(2) - 1) as f32
}

#[inline]
pub(crate) fn terminal_values(board: &Board, draw: f32) -> [f32; MAX_SNAKES] {
    let mut v = [0.0f32; MAX_SNAKES];
    let n = board.snakes.len();
    match board.winner() {
        Some(w) => {
            for (i, x) in v.iter_mut().enumerate().take(n) {
                *x = if i == w { 1.0 } else { loser_value(n) };
            }
        }
        None => {
            for x in v.iter_mut().take(n) {
                *x = draw;
            }
        }
    }
    v
}

/// Would `mv` walk snake `snake_idx` straight into a wall, its own body, or a
/// non-tail segment of another snake? Used to mask hopeless priors/plays.
pub(crate) fn obvious_immediate_death(board: &Board, snake_idx: usize, mv: Move) -> bool {
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
pub(crate) fn mask_obvious_immediate_deaths(
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
pub(crate) fn terminal_value(
    winner: Option<usize>,
    snake: usize,
    n: usize,
    draw_value: f32,
) -> f32 {
    match winner {
        Some(w) if w == snake => 1.0,
        Some(_) => loser_value(n),
        // No winner = every seat died together (a true draw): neutral for all.
        None => draw_value,
    }
}

/// Mix Dirichlet noise into a root prior in place, the AlphaZero way:
/// p ← (1-frac)·p + frac·Dir(alpha) over the prior's support (the masked-legal
/// candidates — moves the mask zeroed stay zero). Applied at the search root
/// only, BEFORE the simulations run, so the visit-count training target itself
/// explores moves the raw prior dislikes. `slots` is the candidate count; the
/// support is the subset with mass.
pub(crate) fn mix_root_dirichlet<R: Rng>(
    prior: &mut [f32; MAXC],
    slots: usize,
    frac: f32,
    alpha: f32,
    rng: &mut R,
) {
    if frac <= 0.0 {
        return;
    }
    let mut support = [0usize; MAXC];
    let mut k = 0;
    for (a, &p) in prior.iter().enumerate().take(slots) {
        if p > EPS {
            support[k] = a;
            k += 1;
        }
    }
    if k < 2 {
        return;
    }
    if let Ok(dir) = Dirichlet::new_with_size(alpha, k) {
        let noise = dir.sample(rng);
        for (j, &a) in support[..k].iter().enumerate() {
            prior[a] = (1.0 - frac) * prior[a] + frac * noise[j];
        }
    }
}

/// Sample a move from the play policy.
pub(crate) fn sample_move<R: Rng>(policy: &[f32], rng: &mut R) -> Move {
    let idx = WeightedIndex::new(policy)
        .map(|d| d.sample(rng))
        .unwrap_or_else(|_| rng.gen_range(0..4));
    Move::from_index(idx)
}

/// The play policy's argmax (post-opening move selection, temperature → 0).
pub(crate) fn argmax_move(policy: &[f32]) -> Move {
    let mut best = 0usize;
    for i in 1..policy.len().min(4) {
        if policy[i] > policy[best] {
            best = i;
        }
    }
    Move::from_index(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn root_dirichlet_respects_support_and_normalization() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut moved = false;
        for _ in 0..100 {
            let mut p = [0.7, 0.0, 0.2, 0.1]; // move 1 masked out
            mix_root_dirichlet(&mut p, 4, 0.25, 0.3, &mut rng);
            assert_eq!(p[1], 0.0, "masked move must stay zero");
            let sum: f32 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "stays a distribution, got {sum}");
            assert!(p.iter().all(|&v| v >= 0.0));
            moved |= (p[0] - 0.7).abs() > 1e-3;
        }
        assert!(moved, "noise must actually perturb the prior");
    }

    #[test]
    fn root_dirichlet_disabled_or_degenerate_is_identity() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut p = [0.7, 0.0, 0.2, 0.1];
        mix_root_dirichlet(&mut p, 4, 0.0, 0.3, &mut rng);
        assert_eq!(p, [0.7, 0.0, 0.2, 0.1]);
        // A single-move support has nothing to explore.
        let mut lone = [0.0, 1.0, 0.0, 0.0];
        mix_root_dirichlet(&mut lone, 4, 0.25, 0.3, &mut rng);
        assert_eq!(lone, [0.0, 1.0, 0.0, 0.0]);
    }
}
