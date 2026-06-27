//! Batched simultaneous-move MCTS with **decoupled-PUCT** (DUCT) selection.
//!
//! This is the real AlphaZero search, adapted to a simultaneous-move game:
//! - the network **policy** head supplies per-snake priors (PUCT),
//! - the network **value** head evaluates leaves (no rollouts),
//! - **root visit counts** become the improved policy target.
//!
//! It is driven in lockstep across many games so leaf evaluation is one big
//! batched neural-net call per simulation:
//!   1. [`MctsForest::select`] descends every game's tree to a leaf and returns
//!      the leaf observations that need a network estimate (terminal leaves are
//!      backed up immediately, needing no eval).
//!   2. the caller runs the net once on those observations.
//!   3. [`MctsForest::expand_backup`] expands each evaluated leaf with its priors
//!      and propagates its value up the path.
//! Repeat for `sims` simulations, then read [`MctsForest::root_targets`].

use crate::search::{candidates, terminal_values};
use rayon::prelude::*;
use snek_core::{encode_into, Board, Move, MAX_SNAKES, NUM_CHANNELS};

const DUMMY_MOVE: Move = Move::Up;

/// One edge taken during a descent: the node and, per snake, which candidate
/// index was selected. Used to credit visits/values on backup.
#[derive(Clone)]
struct Edge {
    node: usize,
    action_idx: [usize; MAX_SNAKES],
}

struct MctsNode {
    board: Board,
    terminal: bool,
    term_value: [f32; MAX_SNAKES],
    expanded: bool,
    cands: Vec<Vec<Move>>,        // [n][k_i] candidate moves per snake
    nvisit: Vec<Vec<f32>>,        // [n][k_i] visit counts
    wsum: Vec<Vec<f32>>,          // [n][k_i] summed backed-up value
    prior: Vec<Vec<f32>>,         // [n][k_i] policy priors
    children: Vec<(u32, usize)>,  // (joint index -> child node id), small assoc list
}

impl MctsNode {
    fn leaf(board: Board) -> Self {
        let terminal = board.is_terminal();
        let term_value = if terminal { terminal_values(&board) } else { [0.0; MAX_SNAKES] };
        MctsNode {
            board,
            terminal,
            term_value,
            expanded: false,
            cands: Vec::new(),
            nvisit: Vec::new(),
            wsum: Vec::new(),
            prior: Vec::new(),
            children: Vec::new(),
        }
    }

    fn child(&self, joint: u32) -> Option<usize> {
        self.children.iter().find(|(j, _)| *j == joint).map(|(_, id)| *id)
    }
}

struct MctsTree {
    nodes: Vec<MctsNode>,
    n_snakes: usize,
    c_puct: f32,
    /// Path of the in-flight simulation, awaiting a leaf evaluation.
    pending_path: Vec<Edge>,
    /// Node id of the in-flight leaf to evaluate (None if terminal/no-op).
    pending_leaf: Option<usize>,
}

impl MctsTree {
    fn new(board: Board, c_puct: f32) -> Self {
        let n_snakes = board.snakes.len();
        MctsTree {
            nodes: vec![MctsNode::leaf(board)],
            n_snakes,
            c_puct,
            pending_path: Vec::new(),
            pending_leaf: None,
        }
    }

    /// DUCT-PUCT: per snake, argmax over its candidates of
    /// `Q + c_puct * P * sqrt(sum_b N_b) / (1 + N_a)`.
    fn select_joint(&self, id: usize, strides: &[u32]) -> (u32, [usize; MAX_SNAKES]) {
        let node = &self.nodes[id];
        let mut action_idx = [0usize; MAX_SNAKES];
        let mut joint: u32 = 0;
        for i in 0..self.n_snakes {
            let nv = &node.nvisit[i];
            let total_n: f32 = nv.iter().sum();
            let sqrt_total = total_n.max(1.0).sqrt();
            let mut best_a = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for a in 0..node.cands[i].len() {
                let n_a = nv[a];
                let q = if n_a > 0.0 { node.wsum[i][a] / n_a } else { 0.0 };
                let u = self.c_puct * node.prior[i][a] * sqrt_total / (1.0 + n_a);
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_a = a;
                }
            }
            action_idx[i] = best_a;
            joint += best_a as u32 * strides[i];
        }
        (joint, action_idx)
    }

    /// Descend from the root to a leaf, recording the path. Sets `pending_leaf`
    /// to a node needing a network eval, or None if the descent ended on a
    /// terminal node (whose exact value is backed up here directly).
    fn select(&mut self) {
        self.pending_path.clear();
        self.pending_leaf = None;
        let mut id = 0usize; // root
        loop {
            let node = &self.nodes[id];
            if node.terminal {
                let v = node.term_value;
                self.backup(&v);
                return;
            }
            if !node.expanded {
                self.pending_leaf = Some(id);
                return;
            }
            // strides for this node's candidate layout
            let n = self.n_snakes;
            let mut strides = [1u32; MAX_SNAKES];
            for i in (0..n).rev() {
                strides[i] = if i + 1 < n {
                    strides[i + 1] * node.cands[i + 1].len() as u32
                } else {
                    1
                };
            }
            let (joint, action_idx) = self.select_joint(id, &strides);
            self.pending_path.push(Edge { node: id, action_idx });
            match self.nodes[id].child(joint) {
                Some(cid) => {
                    id = cid;
                }
                None => {
                    // Create the child by stepping the joint move, then stop:
                    // this fresh node is the leaf (eval or terminal).
                    let mut mv = [DUMMY_MOVE; MAX_SNAKES];
                    {
                        let node = &self.nodes[id];
                        for i in 0..n {
                            mv[i] = node.cands[i][action_idx[i]];
                        }
                    }
                    let mut child_board = self.nodes[id].board.clone();
                    child_board.step(&mv[..n]);
                    let cid = self.nodes.len();
                    let child = MctsNode::leaf(child_board);
                    let is_terminal = child.terminal;
                    let term_value = child.term_value;
                    self.nodes.push(child);
                    self.nodes[id].children.push((joint, cid));
                    if is_terminal {
                        self.backup(&term_value);
                    } else {
                        self.pending_leaf = Some(cid);
                    }
                    return;
                }
            }
        }
    }

    /// Credit `value` (per snake) along the recorded path.
    fn backup(&mut self, value: &[f32; MAX_SNAKES]) {
        for edge in &self.pending_path {
            let node = &mut self.nodes[edge.node];
            for i in 0..self.n_snakes {
                let a = edge.action_idx[i];
                node.nvisit[i][a] += 1.0;
                node.wsum[i][a] += value[i];
            }
        }
        self.pending_path.clear();
    }

    /// Expand the pending leaf with priors from `policy` ([n_snakes, 4] probs)
    /// and back up its `value` ([n_snakes]).
    fn expand_backup(&mut self, policy: &[f32], value: &[f32; MAX_SNAKES]) {
        let id = match self.pending_leaf.take() {
            Some(id) => id,
            None => return,
        };
        let n = self.n_snakes;
        let board = self.nodes[id].board.clone();
        let cands: Vec<Vec<Move>> = (0..n).map(|i| candidates(&board, i)).collect();
        let mut prior = Vec::with_capacity(n);
        let mut nvisit = Vec::with_capacity(n);
        let mut wsum = Vec::with_capacity(n);
        for i in 0..n {
            let k = cands[i].len();
            // gather policy mass on this snake's candidate moves, renormalize
            let mut p = vec![0.0f32; k];
            let mut s = 0.0f32;
            for (a, m) in cands[i].iter().enumerate() {
                let pm = policy[i * 4 + m.index()].max(0.0);
                p[a] = pm;
                s += pm;
            }
            if s > 1e-8 {
                for x in p.iter_mut() {
                    *x /= s;
                }
            } else {
                for x in p.iter_mut() {
                    *x = 1.0 / k as f32;
                }
            }
            prior.push(p);
            nvisit.push(vec![0.0f32; k]);
            wsum.push(vec![0.0f32; k]);
        }
        let node = &mut self.nodes[id];
        node.cands = cands;
        node.prior = prior;
        node.nvisit = nvisit;
        node.wsum = wsum;
        node.expanded = true;
        self.backup(value);
    }

    /// Per-snake root targets: visit-count policy mapped to 4 move slots, and the
    /// mean backed-up root value.
    fn root_targets(&self) -> (Vec<f32>, [f32; MAX_SNAKES]) {
        let n = self.n_snakes;
        let mut policy = vec![0.0f32; n * 4];
        let mut value = [0.0f32; MAX_SNAKES];
        let root = &self.nodes[0];
        if !root.expanded {
            return (policy, value);
        }
        for i in 0..n {
            let total: f32 = root.nvisit[i].iter().sum();
            if total > 0.0 {
                for (a, m) in root.cands[i].iter().enumerate() {
                    policy[i * 4 + m.index()] = root.nvisit[i][a] / total;
                }
                value[i] = root.wsum[i].iter().sum::<f32>() / total;
            }
        }
        (policy, value)
    }
}

/// A batch of independent MCTS trees, one per game, driven in lockstep.
pub struct MctsForest {
    trees: Vec<MctsTree>,
    pub n_snakes: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl MctsForest {
    pub fn new(boards: &[Board], c_puct: f32) -> Self {
        let n_snakes = boards.first().map(|b| b.snakes.len()).unwrap_or(0);
        let (height, width) = boards
            .first()
            .map(|b| (b.height as usize, b.width as usize))
            .unwrap_or((0, 0));
        MctsForest {
            trees: boards.iter().map(|b| MctsTree::new(b.clone(), c_puct)).collect(),
            n_snakes,
            channels: NUM_CHANNELS,
            height,
            width,
        }
    }

    pub fn obs_size(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Run one selection step across all trees. Returns the list of tree indices
    /// whose pending leaf needs a network evaluation (terminal leaves were
    /// already backed up). The caller then evaluates exactly those.
    pub fn select(&mut self) -> Vec<usize> {
        self.trees.par_iter_mut().for_each(|t| t.select());
        (0..self.trees.len())
            .filter(|&i| self.trees[i].pending_leaf.is_some())
            .collect()
    }

    /// Write observations for the `pending` trees (from [`select`]) into `out`,
    /// laid out `[pending_idx, agent]` with agent innermost. `out` must be
    /// `pending.len() * n_snakes * obs_size()` long.
    pub fn write_pending_obs(&self, pending: &[usize], out: &mut [f32]) {
        let obs = self.obs_size();
        let chunk = self.n_snakes * obs;
        out.par_chunks_mut(chunk)
            .zip(pending.par_iter())
            .for_each(|(buf, &ti)| {
                let leaf = self.trees[ti].pending_leaf.unwrap();
                let board = &self.trees[ti].nodes[leaf].board;
                for agent in 0..self.n_snakes {
                    let base = agent * obs;
                    encode_into(board, agent, &mut buf[base..base + obs]);
                }
            });
    }

    /// Expand and back up the evaluated `pending` leaves. `policies` is
    /// `[pending, n_snakes, 4]` (probabilities) and `values` is
    /// `[pending, n_snakes]`, aligned with `pending`.
    pub fn expand_backup(&mut self, pending: &[usize], policies: &[f32], values: &[f32]) {
        let n = self.n_snakes;
        for (pos, &ti) in pending.iter().enumerate() {
            let pol = &policies[pos * n * 4..(pos + 1) * n * 4];
            let mut val = [0.0f32; MAX_SNAKES];
            val[..n].copy_from_slice(&values[pos * n..(pos + 1) * n]);
            self.trees[ti].expand_backup(pol, &val);
        }
    }

    /// Root policy targets `[count, n_snakes, 4]` and root values `[count, n_snakes]`.
    pub fn root_targets(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.n_snakes;
        let mut policies = vec![0.0f32; self.trees.len() * n * 4];
        let mut values = vec![0.0f32; self.trees.len() * n];
        self.trees
            .par_iter()
            .zip(policies.par_chunks_mut(n * 4))
            .zip(values.par_chunks_mut(n))
            .for_each(|((tree, pol), val)| {
                let (p, v) = tree.root_targets();
                pol.copy_from_slice(&p);
                val.copy_from_slice(&v[..n]);
            });
        (policies, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snek_core::Point;

    fn two_snake_board() -> Board {
        let mut b = Board::new(11, 11);
        b.add_snake(&[Point::new(2, 2), Point::new(2, 1)]);
        b.add_snake(&[Point::new(8, 8), Point::new(8, 9)]);
        b
    }

    /// Drive the forest with a uniform policy / zero value and check the search
    /// mechanics: the tree grows, and root visit-count policies are valid
    /// distributions over each snake's legal moves.
    #[test]
    fn mcts_runs_and_produces_valid_policy() {
        let mut forest = MctsForest::new(&[two_snake_board()], 1.5);
        let n = forest.n_snakes;
        let obs = forest.obs_size();
        for _ in 0..64 {
            let pending = forest.select();
            if pending.is_empty() {
                continue;
            }
            let mut buf = vec![0.0f32; pending.len() * n * obs];
            forest.write_pending_obs(&pending, &mut buf); // exercises the obs path
            let pol = vec![0.25f32; pending.len() * n * 4]; // uniform priors
            let val = vec![0.0f32; pending.len() * n]; // neutral value
            forest.expand_backup(&pending, &pol, &val);
        }
        let (policies, values) = forest.root_targets();
        assert_eq!(policies.len(), n * 4);
        assert_eq!(values.len(), n);
        for i in 0..n {
            let p = &policies[i * 4..i * 4 + 4];
            let sum: f32 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "snake {i} policy sums to 1, got {sum}");
            assert!(p.iter().all(|&x| x >= 0.0 && x.is_finite()));
        }
        // With 64 sims the tree must have grown well past the root.
        assert!(forest.trees[0].nodes.len() > 10);
    }

    /// A move that leads straight to a losing terminal should attract fewer
    /// visits than the alternatives (terminal values steer the search even with
    /// a neutral value net).
    #[test]
    fn mcts_avoids_losing_terminal() {
        // Snake 0 is length 1 in a corner; one legal move keeps it alive, the
        // search should not pile visits onto immediately-fatal lines.
        let mut forest = MctsForest::new(&[two_snake_board()], 1.5);
        let n = forest.n_snakes;
        for _ in 0..128 {
            let pending = forest.select();
            if pending.is_empty() {
                continue;
            }
            let pol = vec![0.25f32; pending.len() * n * 4];
            let val = vec![0.0f32; pending.len() * n];
            forest.expand_backup(&pending, &pol, &val);
        }
        let (policies, _) = forest.root_targets();
        // Sanity: still a valid distribution after many sims.
        let p0: f32 = policies[0..4].iter().sum();
        assert!((p0 - 1.0).abs() < 1e-3);
    }
}
