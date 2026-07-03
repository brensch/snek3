//! The strongest heuristic evaluation we know how to build without a net,
//! for [`crate::Baseline::Voronoi`]. Three upgrades over the flood-fill eval:
//!
//! 1. **Time-expanded ownership race.** All heads flood outward together, one
//!    turn per BFS layer, and body cells unblock as tails vacate them
//!    (segment k of an L-long snake frees its cell after L−k turns; growth
//!    from food is ignored). A cell belongs to whoever can be there first —
//!    with same-turn contests won by the *longer* snake, since it wins the
//!    head-to-head; equal lengths kill the claim (nobody owns, nobody passes).
//!    This is dramatically more truthful than static occupancy: tail-chasing
//!    is recognized as safe, and "control" accounts for body decay.
//!
//! 2. **Hunger and traps.** A snake that cannot reach food it *owns* before
//!    starving is dying no matter how much area it holds, and a snake whose
//!    reachable region (tail decay included) is smaller than its body cannot
//!    escape its pocket. Both get explicit penalty terms.
//!
//! 3. **Guaranteed-loss pruning** ([`safe_candidates`]): candidate moves that
//!    step onto a square a longer-or-equal opponent head can also reach next
//!    turn are dropped from the search outright (unless every move loses, in
//!    which case the search keeps them all and picks the least bad).

use snek_core::{Board, Move, Point, MAX_SNAKES};

/// Blend weights. Positive terms sum to 0.85; the trap term only subtracts.
const W_AREA: f32 = 0.40;
const W_HEALTH: f32 = 0.10;
const W_LENGTH: f32 = 0.15;
const W_FOOD: f32 = 0.20;
const W_TRAP: f32 = 0.35;

/// Sentinels for the ownership grid (real owners are snake indices).
const UNCLAIMED: u8 = u8::MAX;
const CONTESTED: u8 = u8::MAX - 1;

pub(crate) fn evaluate(board: &Board, draw_value: f32) -> [f32; MAX_SNAKES] {
    if let Some(vals) = crate::terminal_values(board, draw_value) {
        return vals;
    }
    let mut vals = [-1.0f32; MAX_SNAKES];
    let w = board.width as usize;
    let h = board.height as usize;
    let cells = w * h;
    let idx = |p: &Point| p.y as usize * w + p.x as usize;
    let alive: Vec<usize> = (0..board.snakes.len())
        .filter(|&i| board.snakes[i].alive())
        .collect();

    // Turns until each body cell is vacated by its occupant (0 = free now).
    let mut blocked_until = vec![0u16; cells];
    for &i in &alive {
        let s = &board.snakes[i];
        let len = s.len();
        for (k, seg) in s.body.iter().enumerate() {
            let c = idx(&seg);
            blocked_until[c] = blocked_until[c].max((len - k) as u16);
        }
    }

    // --- Ownership race: all heads flood together, longer snakes win
    // same-turn contests. `owner`/`arrival` describe who can be at each cell
    // first and when.
    let mut owner = vec![UNCLAIMED; cells];
    let mut arrival = vec![0u16; cells];
    // Longer snakes claim first within a turn; equal lengths contest below.
    let mut race_order = alive.clone();
    race_order.sort_by_key(|&i| std::cmp::Reverse(board.snakes[i].len()));
    let mut frontier: Vec<Vec<usize>> = vec![Vec::new(); board.snakes.len()];
    for &i in &alive {
        frontier[i].push(idx(&board.snakes[i].head()));
    }
    let mut t: u16 = 0;
    while frontier.iter().any(|f| !f.is_empty()) && (t as usize) <= cells {
        t += 1;
        let mut next: Vec<Vec<usize>> = vec![Vec::new(); board.snakes.len()];
        for &i in &race_order {
            let my_len = board.snakes[i].len();
            for &c in &frontier[i] {
                // A contested claim from this snake's own frontier is void.
                if t > 1 && owner[c] != i as u8 {
                    continue;
                }
                let (x, y) = ((c % w) as i32, (c / w) as i32);
                for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                        continue;
                    }
                    let nc = ny as usize * w + nx as usize;
                    if blocked_until[nc] > t {
                        continue;
                    }
                    match owner[nc] {
                        UNCLAIMED => {
                            owner[nc] = i as u8;
                            arrival[nc] = t;
                            next[i].push(nc);
                        }
                        // A same-turn claim by an equal-length rival is a
                        // mutual head-to-head: the cell belongs to nobody and
                        // stops both floods. (Longer rivals claimed earlier
                        // in `race_order` and simply keep the cell.)
                        o if o != CONTESTED
                            && o != i as u8
                            && arrival[nc] == t
                            && board.snakes[o as usize].len() == my_len =>
                        {
                            owner[nc] = CONTESTED;
                        }
                        _ => {}
                    }
                }
            }
        }
        frontier = next;
    }

    let mut area = [0usize; MAX_SNAKES];
    for c in 0..cells {
        if owner[c] != UNCLAIMED && owner[c] != CONTESTED {
            area[owner[c] as usize] += 1;
        }
    }
    // Earliest arrival at a food cell this snake owns the race to.
    let mut food_dist = [u16::MAX; MAX_SNAKES];
    for f in &board.food {
        let c = idx(f);
        let o = owner[c];
        if o != UNCLAIMED && o != CONTESTED {
            let o = o as usize;
            food_dist[o] = food_dist[o].min(arrival[c]);
        }
    }

    let free = (0..cells).filter(|&c| blocked_until[c] == 0).count().max(1);
    let n_alive = alive.len() as f32;
    for &i in &alive {
        let s = &board.snakes[i];
        let len = s.len() as f32;
        // Equal split of the free cells scores 0; owning everything → +1.
        let area_score =
            ((area[i] as f32 / free as f32) * n_alive - 1.0).clamp(-1.0, 1.0);
        let health = (s.health as f32 / 100.0) * 2.0 - 1.0;
        let longest_other = alive
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| board.snakes[j].len())
            .max()
            .unwrap_or(0) as f32;
        let length = ((len - longest_other) / 4.0).clamp(-1.0, 1.0);
        // Hunger: how tight is the race between health and the nearest food
        // we can win? Only matters in proportion to how hungry we are.
        let urgency = ((100.0 - s.health as f32) / 60.0).clamp(0.0, 1.0);
        let margin = if food_dist[i] == u16::MAX {
            -1.0
        } else {
            ((s.health as f32 - food_dist[i] as f32) / 25.0).clamp(-1.0, 1.0)
        };
        let food = urgency * margin;
        // Trap: a reachable region smaller than the body is a death sentence.
        let reach = reachable_cells(board, i, &blocked_until, w, h);
        let trap = ((reach as f32 - len) / len).clamp(-1.0, 0.0);
        vals[i] = (W_AREA * area_score
            + W_HEALTH * health
            + W_LENGTH * length
            + W_FOOD * food
            + W_TRAP * trap)
            .clamp(-1.0, 1.0);
    }
    vals
}

/// How many cells snake `i` can reach at all (opponents parked, tails
/// decaying): BFS where a cell is enterable at step t+1 if its occupant will
/// have vacated by then. The head cell itself is not counted.
fn reachable_cells(board: &Board, i: usize, blocked_until: &[u16], w: usize, h: usize) -> usize {
    let head = board.snakes[i].head();
    let mut seen = vec![false; w * h];
    let mut queue = std::collections::VecDeque::new();
    let start = head.y as usize * w + head.x as usize;
    seen[start] = true;
    queue.push_back((start, 0u16));
    let mut count = 0usize;
    while let Some((c, t)) = queue.pop_front() {
        let (x, y) = ((c % w) as i32, (c / w) as i32);
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                continue;
            }
            let nc = ny as usize * w + nx as usize;
            if seen[nc] || blocked_until[nc] > t + 1 {
                continue;
            }
            seen[nc] = true;
            count += 1;
            queue.push_back((nc, t + 1));
        }
    }
    count
}

/// [`crate::candidates`] minus guaranteed head-to-head losses: moves onto a
/// square a longer-or-equal live opponent's head can also reach next turn.
/// If every candidate is such a square the full set is kept — the search
/// itself must pick the least bad of a forced position.
pub(crate) fn safe_candidates(board: &Board, i: usize) -> Vec<Move> {
    let base = crate::candidates(board, i);
    if base.len() <= 1 || !board.snakes[i].alive() {
        return base;
    }
    let my_len = board.snakes[i].len();
    let mut danger: Vec<Point> = Vec::new();
    for (j, s) in board.snakes.iter().enumerate() {
        if j == i || !s.alive() || s.len() < my_len {
            continue;
        }
        for m in crate::candidates(board, j) {
            danger.push(m.apply(s.head()));
        }
    }
    let head = board.snakes[i].head();
    let safe: Vec<Move> = base
        .iter()
        .copied()
        .filter(|m| !danger.contains(&m.apply(head)))
        .collect();
    if safe.is_empty() {
        base
    } else {
        safe
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{baseline_move_until, Baseline, HeuristicConfig};
    use std::time::{Duration, Instant};

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    fn test_cfg() -> HeuristicConfig {
        HeuristicConfig {
            max_sims: 512,
            ..Default::default()
        }
    }

    fn board_with(snakes: &[&[(i8, i8)]], food: &[(i8, i8)]) -> Board {
        let mut b = Board::new(11, 11);
        for s in snakes {
            let pts: Vec<Point> = s.iter().map(|&(x, y)| Point::new(x, y)).collect();
            b.add_snake(&pts);
        }
        for &(x, y) in food {
            b.food.push(Point::new(x, y));
        }
        b
    }

    #[test]
    fn tail_chasing_is_recognized_as_safe_space() {
        // A snake curled next to its own tail: static occupancy calls the
        // tail cell blocked, tail decay knows it frees up next turn, so the
        // reachable region must cover far more than the snake's length.
        let b = board_with(
            &[
                &[(2, 2), (2, 3), (3, 3), (3, 2)],
                &[(8, 8), (8, 9), (8, 10)],
            ],
            &[],
        );
        let blocked = {
            let w = 11usize;
            let mut blocked = vec![0u16; w * 11];
            for s in b.snakes.iter().filter(|s| s.alive()) {
                for (k, seg) in s.body.iter().enumerate() {
                    let c = seg.y as usize * w + seg.x as usize;
                    blocked[c] = blocked[c].max((s.len() - k) as u16);
                }
            }
            blocked
        };
        let reach = reachable_cells(&b, 0, &blocked, 11, 11);
        assert!(reach > 50, "tail decay should open the board: {reach}");
    }

    #[test]
    fn longer_snake_wins_contested_cells() {
        // Two heads equidistant from the middle column; the longer snake
        // must own the contested ground rather than splitting it.
        let b = board_with(
            &[
                &[(2, 5), (1, 5), (0, 5), (0, 4), (0, 3)],
                &[(8, 5), (9, 5), (10, 5)],
            ],
            &[],
        );
        let vals = evaluate(&b, -0.25);
        assert!(
            vals[0] > vals[1] + 0.15,
            "longer snake should dominate the race: {vals:?}"
        );
    }

    #[test]
    fn starving_without_reachable_food_is_penalized() {
        let mut with_food = board_with(
            &[&[(5, 5), (5, 4), (5, 3)], &[(0, 10), (1, 10), (2, 10)]],
            &[(5, 7)],
        );
        let mut no_food = board_with(
            &[&[(5, 5), (5, 4), (5, 3)], &[(0, 10), (1, 10), (2, 10)]],
            &[],
        );
        with_food.snakes[0].health = 12;
        no_food.snakes[0].health = 12;
        let fed = evaluate(&with_food, -0.25);
        let hungry = evaluate(&no_food, -0.25);
        assert!(
            fed[0] > hungry[0] + 0.1,
            "close owned food must beat starving: {fed:?} vs {hungry:?}"
        );
    }

    #[test]
    fn never_steps_into_a_longer_heads_square() {
        // Opponent (longer) head at (5,6): it can reach (5,5)'s neighbors
        // (4,6)... our head at (4,5) moving Right lands on (5,5), adjacent to
        // the longer head — a losable head-to-head. Up (4,6) is also its
        // square. Pruning must leave only retreating moves.
        let b = board_with(
            &[
                &[(4, 5), (3, 5), (2, 5)],
                &[(5, 6), (6, 6), (7, 6), (7, 5), (7, 4)],
            ],
            &[],
        );
        let safe = safe_candidates(&b, 0);
        for m in &safe {
            let target = m.apply(b.snakes[0].head());
            for opp in crate::candidates(&b, 1) {
                assert_ne!(
                    target,
                    opp.apply(b.snakes[1].head()),
                    "move {m:?} steps into the longer head's reach"
                );
            }
        }
        assert!(!safe.is_empty());
    }

    #[test]
    fn voronoi_agent_is_deterministic_and_legal() {
        let b = board_with(
            &[&[(3, 3), (3, 2), (3, 1)], &[(7, 7), (7, 8), (7, 9)]],
            &[(5, 5)],
        );
        let cfg = test_cfg();
        let a = baseline_move_until(Baseline::Voronoi, &cfg, &b, 0, far_deadline());
        let c = baseline_move_until(Baseline::Voronoi, &cfg, &b, 0, far_deadline());
        assert_eq!(a.move_index, c.move_index);
        assert_eq!(a.policy, c.policy);
        assert!(a.sims > 0);
    }
}
