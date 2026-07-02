//! A cheap, fixed-strength baseline agent: decoupled-UCT MCTS over the exact
//! `snek-core` rules, with leaf positions scored by a flood-fill heuristic
//! (Voronoi area control + health + relative length) instead of a neural net.
//!
//! It exists to anchor the evaluation league: nets drift as training runs, but
//! this player never learns, so its fitted Elo is a stable reference point
//! ("is the newest checkpoint actually better, or did the pool just shift?").
//!
//! The search mirrors the shape of the real serving search (per-snake
//! decoupled selection over joint moves, deterministic argmax readout, the
//! same food-free `Board::step` lookahead) but is orders of magnitude cheaper:
//! a leaf evaluation is a few flood fills over ≤`board²` cells, no tensors.

use std::collections::HashMap;
use std::time::Instant;

use snek_core::{Board, Move, MAX_SNAKES};

/// The `--nets` token that selects this agent in the arena (also accepted:
/// "floodfill").
pub const NET_TOKEN: &str = "heuristic";

/// Display name for scoreboards.
pub const DISPLAY_NAME: &str = "floodfill";

/// True if an arena model spec names the heuristic agent instead of a weights
/// file.
pub fn is_heuristic_spec(spec: &str) -> bool {
    spec.eq_ignore_ascii_case(NET_TOKEN) || spec.eq_ignore_ascii_case(DISPLAY_NAME)
}

#[derive(Clone, Debug)]
pub struct HeuristicConfig {
    /// Simulation cap per move.
    pub max_sims: usize,
    /// UCT exploration constant.
    pub c_uct: f32,
    /// Leaf value of a terminal draw (match the net search's convention).
    pub draw_value: f32,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            // Sized to ~200ms/move on one CPU core: the search sustains
            // ~100–220k sims/s in release (worst case 4-player mid-game
            // ~100k/s — see examples/bench.rs), so leaf evals are ~100×
            // cheaper than a net forward and the baseline can afford a far
            // deeper static budget than the nets' sim count.
            max_sims: 20_000,
            c_uct: 1.2,
            draw_value: -0.25,
        }
    }
}

/// The move plus the same readout shape the net search reports, so arena
/// recordings and the viewer render heuristic seats like any other player.
#[derive(Clone, Debug)]
pub struct HeuristicDecision {
    pub move_index: usize,
    /// Root visit-count policy over the 4 global moves.
    pub policy: [f32; 4],
    /// Mean backed-up root value for the deciding snake.
    pub value: f32,
    pub sims: usize,
}

/// One search-tree node. Selection statistics are decoupled per snake (each
/// snake runs its own UCT over its candidate moves); children are keyed by the
/// joint candidate choice.
struct Node {
    board: Board,
    /// Per snake: legal candidates (dead/trapped snakes get the usual dummy /
    /// full set, exactly like the net search's candidate rule).
    cands: Vec<Vec<Move>>,
    /// Per snake, per candidate: visit count and summed backed-up value.
    visits: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// Selections that passed through this node.
    total: f32,
    children: HashMap<u32, usize>,
    terminal: bool,
}

impl Node {
    fn new(board: Board) -> Self {
        let terminal = board.is_terminal();
        let cands: Vec<Vec<Move>> = (0..board.snakes.len())
            .map(|i| candidates(&board, i))
            .collect();
        let visits = cands.iter().map(|c| vec![0.0; c.len()]).collect();
        let values = cands.iter().map(|c| vec![0.0; c.len()]).collect();
        Self {
            board,
            cands,
            visits,
            values,
            total: 0.0,
            children: HashMap::new(),
            terminal,
        }
    }

    /// UCT pick for one snake: first unvisited candidate, else argmax of
    /// Q + c·sqrt(ln(total+1)/n). Fully deterministic (stable tie-break).
    fn select(&self, i: usize, c_uct: f32) -> usize {
        let n_cands = self.cands[i].len();
        if n_cands == 1 {
            return 0;
        }
        if let Some(c) = (0..n_cands).find(|&c| self.visits[i][c] == 0.0) {
            return c;
        }
        let ln_total = (self.total + 1.0).ln();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..n_cands {
            let n = self.visits[i][c];
            let q = self.values[i][c] / n;
            let score = q + c_uct * (ln_total / n).sqrt();
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        best
    }
}

/// Candidate moves for one snake: drop reversing onto the neck and stepping
/// off the board; a trapped snake keeps all moves; eliminated snakes get a
/// single dummy. (Same rule as `snek-search`.)
fn candidates(board: &Board, i: usize) -> Vec<Move> {
    let s = &board.snakes[i];
    if !s.alive() {
        return vec![Move::Up];
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

/// Per-snake leaf value in [-1, 1]. Terminal boards score exactly like the net
/// search (winner +1, losers −1, draw `draw_value`); live boards blend Voronoi
/// area control, health, and length relative to the longest opponent.
fn evaluate(board: &Board, draw_value: f32) -> [f32; MAX_SNAKES] {
    let mut vals = [-1.0f32; MAX_SNAKES];
    if board.is_terminal() {
        match board.winner() {
            Some(w) => vals[w] = 1.0,
            None => {
                for v in vals.iter_mut().take(board.snakes.len()) {
                    *v = draw_value;
                }
            }
        }
        return vals;
    }

    let w = board.width as usize;
    let h = board.height as usize;
    let cells = w * h;
    let mut occupied = vec![false; cells];
    for s in &board.snakes {
        if !s.alive() {
            continue;
        }
        for seg in s.body.iter() {
            if board.in_bounds(seg) {
                occupied[seg.y as usize * w + seg.x as usize] = true;
            }
        }
    }

    // Per-snake BFS distance from the head over free cells; a cell belongs to
    // the strictly closest head (ties are contested and belong to nobody).
    let n = board.snakes.len();
    let alive: Vec<usize> = (0..n).filter(|&i| board.snakes[i].alive()).collect();
    let mut best = vec![u16::MAX; cells];
    let mut owner = vec![usize::MAX; cells];
    let mut dist = vec![u16::MAX; cells];
    let mut queue = std::collections::VecDeque::new();
    for &i in &alive {
        dist.iter_mut().for_each(|d| *d = u16::MAX);
        queue.clear();
        let head = board.snakes[i].head();
        dist[head.y as usize * w + head.x as usize] = 0;
        queue.push_back((head.x as i32, head.y as i32));
        while let Some((x, y)) = queue.pop_front() {
            let d = dist[y as usize * w + x as usize];
            for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                    continue;
                }
                let ci = ny as usize * w + nx as usize;
                if occupied[ci] || dist[ci] != u16::MAX {
                    continue;
                }
                dist[ci] = d + 1;
                queue.push_back((nx, ny));
            }
        }
        for c in 0..cells {
            if occupied[c] || dist[c] == u16::MAX {
                continue;
            }
            match dist[c].cmp(&best[c]) {
                std::cmp::Ordering::Less => {
                    best[c] = dist[c];
                    owner[c] = i;
                }
                std::cmp::Ordering::Equal => owner[c] = usize::MAX, // contested
                std::cmp::Ordering::Greater => {}
            }
        }
    }
    let free = occupied.iter().filter(|&&o| !o).count().max(1);
    let mut owned = [0usize; MAX_SNAKES];
    for c in 0..cells {
        if owner[c] != usize::MAX {
            owned[owner[c]] += 1;
        }
    }

    let n_alive = alive.len() as f32;
    for &i in &alive {
        let s = &board.snakes[i];
        // Equal split scores 0; owning everything saturates at +1.
        let area = ((owned[i] as f32 / free as f32) * n_alive - 1.0).clamp(-1.0, 1.0);
        let health = (s.health as f32 / 100.0) * 2.0 - 1.0;
        let longest_other = alive
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| board.snakes[j].len())
            .max()
            .unwrap_or(0) as f32;
        let length = ((s.len() as f32 - longest_other) / 4.0).clamp(-1.0, 1.0);
        vals[i] = 0.55 * area + 0.25 * health + 0.20 * length;
    }
    vals
}

/// One move for snake `me`: run decoupled-UCT until `cfg.max_sims` or the
/// deadline, then play the most-visited root candidate. Deterministic for a
/// fixed sim budget (no randomness anywhere in the search).
pub fn heuristic_move_until(
    cfg: &HeuristicConfig,
    board: &Board,
    me: usize,
    deadline: Instant,
) -> HeuristicDecision {
    let mut policy = [0.0f32; 4];
    if board.is_terminal() || me >= board.snakes.len() || !board.snakes[me].alive() {
        policy[Move::Up.index()] = 1.0;
        return HeuristicDecision {
            move_index: Move::Up.index(),
            policy,
            value: 0.0,
            sims: 0,
        };
    }
    let root_cands = candidates(board, me);
    if root_cands.len() == 1 {
        let mv = root_cands[0].index();
        policy[mv] = 1.0;
        return HeuristicDecision {
            move_index: mv,
            policy,
            value: 0.0,
            sims: 0,
        };
    }

    let mut nodes = vec![Node::new(board.clone())];
    let mut sims = 0usize;
    let mut path: Vec<(usize, [usize; MAX_SNAKES])> = Vec::with_capacity(64);
    while sims < cfg.max_sims {
        if sims.is_multiple_of(32) && Instant::now() >= deadline {
            break;
        }
        path.clear();
        let mut cur = 0usize;
        let leaf_vals = loop {
            if nodes[cur].terminal {
                break evaluate(&nodes[cur].board, cfg.draw_value);
            }
            let n = nodes[cur].board.snakes.len();
            let mut cidx = [0usize; MAX_SNAKES];
            let mut key = 0u32;
            for (i, slot) in cidx.iter_mut().enumerate().take(n) {
                let c = nodes[cur].select(i, cfg.c_uct);
                *slot = c;
                key = key * 4 + c as u32;
            }
            path.push((cur, cidx));
            match nodes[cur].children.get(&key) {
                Some(&child) => cur = child,
                None => {
                    let mut child_board = nodes[cur].board.clone();
                    let moves: Vec<Move> =
                        (0..n).map(|i| nodes[cur].cands[i][cidx[i]]).collect();
                    child_board.step(&moves);
                    let vals = evaluate(&child_board, cfg.draw_value);
                    let id = nodes.len();
                    nodes.push(Node::new(child_board));
                    nodes[cur].children.insert(key, id);
                    break vals;
                }
            }
        };
        for &(nid, cidx) in &path {
            let node = &mut nodes[nid];
            node.total += 1.0;
            let n = node.board.snakes.len();
            for i in 0..n {
                node.visits[i][cidx[i]] += 1.0;
                node.values[i][cidx[i]] += leaf_vals[i];
            }
        }
        sims += 1;
    }

    // Readout: visit-count policy for `me`, argmax-visits move (Q breaks ties).
    let root = &nodes[0];
    let total_visits: f32 = root.visits[me].iter().sum();
    let mut best_c = 0usize;
    let mut value = 0.0f32;
    for (c, &mv) in root.cands[me].iter().enumerate() {
        let n = root.visits[me][c];
        if total_visits > 0.0 {
            policy[mv.index()] = n / total_visits;
        }
        value += root.values[me][c];
        let better = (root.visits[me][c], qval(root, me, c))
            > (root.visits[me][best_c], qval(root, me, best_c));
        if c > 0 && better {
            best_c = c;
        }
    }
    if total_visits > 0.0 {
        value /= total_visits;
    }
    HeuristicDecision {
        move_index: root.cands[me][best_c].index(),
        policy,
        value,
        sims,
    }
}

fn qval(node: &Node, i: usize, c: usize) -> f32 {
    let n = node.visits[i][c];
    if n > 0.0 {
        node.values[i][c] / n
    } else {
        f32::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snek_core::Point;
    use std::time::Duration;

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    /// Small budget so the suite stays fast in debug builds; the default is
    /// sized for release-mode arena play.
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
    fn never_walks_off_the_board() {
        // Head in the corner, neck to the right: only Up survives.
        let b = board_with(&[&[(0, 0), (1, 0), (2, 0)], &[(5, 5), (5, 6), (5, 7)]], &[]);
        let d = heuristic_move_until(&test_cfg(), &b, 0, far_deadline());
        assert_eq!(Move::from_index(d.move_index), Move::Up);
    }

    #[test]
    fn prefers_open_space_over_a_pocket() {
        // A wall of opponent body leaves a small pocket left and open space
        // right; area control must steer right.
        let b = board_with(
            &[
                &[(2, 5), (2, 4), (2, 3)],
                &[(1, 10), (1, 9), (1, 8), (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1)],
            ],
            &[],
        );
        let d = heuristic_move_until(&test_cfg(), &b, 0, far_deadline());
        assert_ne!(Move::from_index(d.move_index), Move::Left);
    }

    #[test]
    fn deterministic_for_a_fixed_budget() {
        let b = board_with(&[&[(3, 3), (3, 2), (3, 1)], &[(7, 7), (7, 8), (7, 9)]], &[(5, 5)]);
        let cfg = test_cfg();
        let a = heuristic_move_until(&cfg, &b, 0, far_deadline());
        let c = heuristic_move_until(&cfg, &b, 0, far_deadline());
        assert_eq!(a.move_index, c.move_index);
        assert_eq!(a.policy, c.policy);
        assert!(a.sims > 0);
        let sum: f32 = a.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy should normalize: {:?}", a.policy);
    }

    #[test]
    fn dead_or_terminal_positions_fall_back_safely() {
        let mut b = board_with(&[&[(3, 3), (3, 2), (3, 1)], &[(7, 7), (7, 8), (7, 9)]], &[]);
        b.snakes[1].health = 0; // treat as eliminated via starvation on next step
        b.step(&[Move::Up, Move::Up]);
        assert!(b.is_terminal());
        let d = heuristic_move_until(&test_cfg(), &b, 0, far_deadline());
        assert_eq!(d.sims, 0);
    }
}
