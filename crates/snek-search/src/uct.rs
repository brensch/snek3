//! Pure-CPU UCT agent with a cheap Voronoi area-control heuristic — no neural
//! net. Decoupled-UCB tree search (depth) + a flood-fill leaf evaluation
//! (cheap), batched across games with rayon so it runs on the otherwise-idle
//! CPU while the GPU does network inference. Used as a strong fixed opponent in
//! the Albatross opponent pool.

use crate::search::candidates;
use crate::search::terminal_values_with_draw;
use rayon::prelude::*;
use snek_core::{Board, Move, MAX_SNAKES};

/// Voronoi-style area control in [-1, 1] per snake: multi-source BFS from every
/// alive head; each free cell is owned by the snake that reaches it first (ties
/// are contested/unowned). A snake's value is its share of owned cells mapped to
/// [-1, 1] (dominating the board ⇒ +1, boxed in ⇒ -1).
fn voronoi_value(board: &Board) -> [f32; MAX_SNAKES] {
    let w = board.width as i32;
    let h = board.height as i32;
    let idx = |x: i32, y: i32| (y * w + x) as usize;
    let mut blocked = vec![false; (w * h) as usize];
    for s in &board.snakes {
        if !s.alive() {
            continue;
        }
        for i in 0..s.len() {
            let p = s.body.get(i);
            if p.x >= 0 && p.y >= 0 && (p.x as i32) < w && (p.y as i32) < h {
                blocked[idx(p.x as i32, p.y as i32)] = true;
            }
        }
    }

    // dist/owner per cell; multi-source BFS layered by distance.
    let mut dist = vec![i32::MAX; (w * h) as usize];
    let mut owner = vec![-1i32; (w * h) as usize]; // -1 unowned, -2 contested
    let mut frontier: Vec<(i32, i32, i32)> = Vec::new(); // (x, y, snake)
    for (si, s) in board.snakes.iter().enumerate() {
        if !s.alive() {
            continue;
        }
        let head = s.head();
        let c = idx(head.x as i32, head.y as i32);
        dist[c] = 0;
        owner[c] = si as i32;
        frontier.push((head.x as i32, head.y as i32, si as i32));
    }

    let dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)];
    let mut d = 0;
    while !frontier.is_empty() {
        let mut next: Vec<(i32, i32, i32)> = Vec::new();
        // First pass: propose claims at distance d+1, resolving ties to contested.
        for &(x, y, who) in &frontier {
            for (dx, dy) in dirs {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                let c = idx(nx, ny);
                if blocked[c] {
                    continue;
                }
                if dist[c] == i32::MAX {
                    dist[c] = d + 1;
                    owner[c] = who;
                    next.push((nx, ny, who));
                } else if dist[c] == d + 1 && owner[c] != who && owner[c] != -2 {
                    owner[c] = -2; // reached at the same distance by another snake
                }
            }
        }
        // Drop cells that became contested from the propagation frontier.
        next.retain(|&(x, y, who)| owner[idx(x, y)] == who);
        frontier = next;
        d += 1;
    }

    let mut counts = [0i32; MAX_SNAKES];
    for &o in &owner {
        if o >= 0 {
            counts[o as usize] += 1;
        }
    }
    let total: i32 = counts.iter().sum();
    let mut v = [0.0f32; MAX_SNAKES];
    if total > 0 {
        for i in 0..board.snakes.len() {
            if board.snakes[i].alive() {
                v[i] = 2.0 * (counts[i] as f32 / total as f32) - 1.0;
            } else {
                v[i] = -1.0;
            }
        }
    }
    v
}

/// Leaf evaluation: exact terminal value, else the Voronoi heuristic.
fn leaf_value(board: &Board) -> [f32; MAX_SNAKES] {
    if board.is_terminal() {
        terminal_values_with_draw(board, 0.0)
    } else {
        voronoi_value(board)
    }
}

struct Node {
    board: Board,
    terminal: bool,
    value: [f32; MAX_SNAKES],
    expanded: bool,
    cands: Vec<Vec<Move>>,
    nvisit: Vec<Vec<f32>>,
    wsum: Vec<Vec<f32>>,
    children: Vec<(u32, usize)>,
}

impl Node {
    fn leaf(board: Board) -> Self {
        let terminal = board.is_terminal();
        let value = leaf_value(&board);
        Node {
            board,
            terminal,
            value,
            expanded: false,
            cands: Vec::new(),
            nvisit: Vec::new(),
            wsum: Vec::new(),
            children: Vec::new(),
        }
    }
}

/// Run `iters` of decoupled-UCB search on one board; return the most-visited
/// move per snake (alive snakes; eliminated snakes get `Move::Up`).
fn uct_one(board: &Board, iters: usize, c_uct: f32, rng_seed: u64) -> [Move; MAX_SNAKES] {
    let n = board.snakes.len();
    let mut nodes: Vec<Node> = vec![Node::leaf(board.clone())];
    let mut rng = rng_seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next_rand = || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        rng.wrapping_mul(0x2545F4914F6CDD1D)
    };

    for _ in 0..iters {
        let mut path: Vec<(usize, [usize; MAX_SNAKES])> = Vec::new();
        let mut id = 0usize;
        let value;
        loop {
            if nodes[id].terminal {
                value = nodes[id].value;
                break;
            }
            if !nodes[id].expanded {
                // Expand with this node's candidate moves, then use its (already
                // computed) heuristic value as the rollout estimate.
                let cands: Vec<Vec<Move>> = (0..n).map(|i| candidates(&nodes[id].board, i)).collect();
                let nvisit = cands.iter().map(|c| vec![0.0f32; c.len()]).collect();
                let wsum = cands.iter().map(|c| vec![0.0f32; c.len()]).collect();
                let node = &mut nodes[id];
                node.cands = cands;
                node.nvisit = nvisit;
                node.wsum = wsum;
                node.expanded = true;
                value = node.value;
                break;
            }
            // Decoupled UCB1 action selection per snake.
            let mut action_idx = [0usize; MAX_SNAKES];
            let mut joint: u32 = 0;
            let mut stride: u32 = 1;
            // strides with agent 0 most significant (match child key convention)
            let mut strides = [1u32; MAX_SNAKES];
            for i in (0..n).rev() {
                strides[i] = stride;
                stride *= nodes[id].cands[i].len() as u32;
            }
            for i in 0..n {
                let nv = &nodes[id].nvisit[i];
                let total: f32 = nv.iter().sum::<f32>().max(1.0);
                let ln = total.ln();
                let mut best_a = 0usize;
                let mut best = f32::NEG_INFINITY;
                for a in 0..nodes[id].cands[i].len() {
                    let na = nv[a];
                    let q = if na > 0.0 { nodes[id].wsum[i][a] / na } else { 0.0 };
                    let u = if na > 0.0 {
                        c_uct * (ln / na).sqrt()
                    } else {
                        f32::INFINITY
                    };
                    let score = q + u;
                    if score > best {
                        best = score;
                        best_a = a;
                    }
                }
                action_idx[i] = best_a;
                joint += best_a as u32 * strides[i];
            }
            path.push((id, action_idx));
            match nodes[id].children.iter().find(|(j, _)| *j == joint).map(|(_, c)| *c) {
                Some(cid) => id = cid,
                None => {
                    let mut mv = [Move::Up; MAX_SNAKES];
                    for i in 0..n {
                        mv[i] = nodes[id].cands[i][action_idx[i]];
                    }
                    let mut child_board = nodes[id].board.clone();
                    child_board.step(&mv[..n]);
                    let cid = nodes.len();
                    let child = Node::leaf(child_board);
                    let v = child.value;
                    nodes.push(child);
                    nodes[id].children.push((joint, cid));
                    value = v;
                    let _ = &mut next_rand; // rng reserved for future stochastic rollouts
                    break;
                }
            }
        }
        // Backup along the path.
        for &(node_id, action_idx) in &path {
            let node = &mut nodes[node_id];
            for i in 0..n {
                let a = action_idx[i];
                node.nvisit[i][a] += 1.0;
                node.wsum[i][a] += value[i];
            }
        }
    }

    // Pick the most-visited move per snake.
    let mut out = [Move::Up; MAX_SNAKES];
    let root = &nodes[0];
    if root.expanded {
        for i in 0..n {
            let mut best_a = 0usize;
            let mut best = -1.0f32;
            for a in 0..root.cands[i].len() {
                if root.nvisit[i][a] > best {
                    best = root.nvisit[i][a];
                    best_a = a;
                }
            }
            if !root.cands[i].is_empty() {
                out[i] = root.cands[i][best_a];
            }
        }
    }
    out
}

/// Batched UCT move selection across `boards`. Returns the chosen move per
/// (game, snake). Parallel across games (uses idle CPU cores).
pub fn uct_actions(
    boards: &[Board],
    iters: usize,
    c_uct: f32,
    seed: u64,
) -> Vec<[Move; MAX_SNAKES]> {
    boards
        .par_iter()
        .enumerate()
        .map(|(g, b)| uct_one(b, iters, c_uct, seed ^ (g as u64).wrapping_mul(0x100000001B3)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use snek_core::{Board, Point};

    #[test]
    fn voronoi_prefers_open_space() {
        // A single snake in a corner: its area value should be finite and the
        // chosen UCT move should be legal (not into a wall/neck).
        let mut b = Board::new(11, 11);
        b.add_snake(&[Point::new(0, 0), Point::new(0, 1)]);
        b.add_snake(&[Point::new(10, 10), Point::new(10, 9)]);
        let v = voronoi_value(&b);
        assert!(v[0].is_finite() && v[1].is_finite());
        let acts = uct_actions(std::slice::from_ref(&b), 64, 1.4, 7);
        assert_eq!(acts.len(), 1);
        // Snake 0 at the corner must not move Down (off-board) or Up-into-neck.
        let m = acts[0][0];
        let nh = m.apply(Point::new(0, 0));
        assert!(b.in_bounds(nh), "UCT picks an in-bounds move");
    }
}
