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
//!
//! Repeat for `sims` simulations, then read [`MctsForest::root_targets`].

use crate::search::{candidates, terminal_values_with_draw};
use rayon::prelude::*;
use snek_core::{encode_into, obs_side, Board, Move, MAX_SNAKES, NUM_CHANNELS};

const DUMMY_MOVE: Move = Move::Up;

pub fn obvious_immediate_death(board: &Board, snake_idx: usize, mv: Move) -> bool {
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

pub fn mask_obvious_immediate_deaths(board: &Board, snake_idx: usize, probs: &[f32]) -> [f32; 4] {
    let mut original = [0.0f32; 4];
    let mut total = 0.0f32;
    for i in 0..4 {
        original[i] = probs.get(i).copied().unwrap_or(0.0).max(0.0);
        total += original[i];
    }
    if total <= 1e-8
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
        let death = obvious_immediate_death(board, snake_idx, Move::from_index(i));
        if !death {
            safe_count += 1;
            out[i] = original[i];
            safe_mass += original[i];
        }
    }
    if safe_count == 0 {
        return original;
    }
    if safe_mass > 1e-8 {
        for x in out.iter_mut() {
            *x /= safe_mass;
        }
    } else {
        let u = 1.0 / safe_count as f32;
        for i in 0..4 {
            if !obvious_immediate_death(board, snake_idx, Move::from_index(i)) {
                out[i] = u;
            }
        }
    }
    out
}

/// The single forced move index for snake `me`, if it has exactly one legal
/// candidate (reversing-onto-neck and off-board moves dropped — the same move
/// set the tree branches over). When a snake is forced, search cannot change its
/// answer, so a caller that only needs `me`'s move can skip the search entirely.
/// Returns `None` if the board is terminal, `me` is dead, or `me` has 0 or ≥2
/// candidates.
pub fn forced_move(board: &Board, me: usize) -> Option<usize> {
    if board.is_terminal() {
        return None;
    }
    let s = board.snakes.get(me)?;
    if !s.alive() {
        return None;
    }
    let cands = candidates(board, me);
    if cands.len() == 1 {
        Some(cands[0].index())
    } else {
        None
    }
}

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
    cands: Vec<Vec<Move>>,       // [n][k_i] candidate moves per snake
    nvisit: Vec<Vec<f32>>,       // [n][k_i] visit counts
    wsum: Vec<Vec<f32>>,         // [n][k_i] summed backed-up value
    prior: Vec<Vec<f32>>,        // [n][k_i] policy priors
    children: Vec<(u32, usize)>, // (joint index -> child node id), small assoc list
}

impl MctsNode {
    fn leaf(board: Board, draw_value: f32) -> Self {
        let terminal = board.is_terminal();
        let term_value = if terminal {
            terminal_values_with_draw(&board, draw_value)
        } else {
            [0.0; MAX_SNAKES]
        };
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
        self.children
            .iter()
            .find(|(j, _)| *j == joint)
            .map(|(_, id)| *id)
    }
}

/// One leaf collected during a batched (virtual-loss) selection round, with the
/// path taken to reach it so the real backup can undo the virtual loss.
struct PendingLeaf {
    leaf: usize,
    path: Vec<Edge>,
}

struct MctsTree {
    nodes: Vec<MctsNode>,
    n_snakes: usize,
    c_puct: f32,
    draw_value: f32,
    /// Path of the in-flight simulation, awaiting a leaf evaluation.
    pending_path: Vec<Edge>,
    /// Node id of the in-flight leaf to evaluate (None if terminal/no-op).
    pending_leaf: Option<usize>,
    /// Leaves collected this batched round (used only by the batched serving
    /// path; the single-leaf self-play path leaves this empty).
    batch: Vec<PendingLeaf>,
    /// Unexpanded leaves already collected this round, so a second descent that
    /// lands on the same leaf doesn't double-expand it.
    inflight: std::collections::HashSet<usize>,
}

impl MctsTree {
    fn new_with_draw_value(board: Board, c_puct: f32, draw_value: f32) -> Self {
        let n_snakes = board.snakes.len();
        MctsTree {
            nodes: vec![MctsNode::leaf(board, draw_value)],
            n_snakes,
            c_puct,
            draw_value,
            pending_path: Vec::new(),
            pending_leaf: None,
            batch: Vec::new(),
            inflight: std::collections::HashSet::new(),
        }
    }

    /// Mixed-radix strides for a node's candidate layout (agent 0 most
    /// significant), matching the joint-index convention used everywhere else.
    fn node_strides(&self, id: usize) -> [u32; MAX_SNAKES] {
        let node = &self.nodes[id];
        let mut strides = [1u32; MAX_SNAKES];
        for i in (0..self.n_snakes).rev() {
            strides[i] = if i + 1 < self.n_snakes {
                strides[i + 1] * node.cands[i + 1].len() as u32
            } else {
                1
            };
        }
        strides
    }

    /// Add (`+vloss` visits, `-vloss` value) along `path` so concurrent descents
    /// in the same round are steered toward *different* leaves (virtual loss).
    fn apply_virtual_loss(&mut self, path: &[Edge], vloss: f32) {
        for edge in path {
            let node = &mut self.nodes[edge.node];
            for i in 0..self.n_snakes {
                let a = edge.action_idx[i];
                node.nvisit[i][a] += vloss;
                node.wsum[i][a] -= vloss;
            }
        }
    }

    /// Undo the virtual loss on `path` (used when a descent collides with an
    /// already-collected leaf and is abandoned).
    fn remove_virtual_loss(&mut self, path: &[Edge], vloss: f32) {
        for edge in path {
            let node = &mut self.nodes[edge.node];
            for i in 0..self.n_snakes {
                let a = edge.action_idx[i];
                node.nvisit[i][a] -= vloss;
                node.wsum[i][a] += vloss;
            }
        }
    }

    /// Real backup along `path`: remove the virtual loss applied during selection
    /// and credit the true `value`, leaving stats exactly as a sequential visit
    /// would.
    fn backup_path(&mut self, path: &[Edge], value: &[f32; MAX_SNAKES], vloss: f32) {
        for edge in path {
            let node = &mut self.nodes[edge.node];
            for (i, &v) in value.iter().enumerate().take(self.n_snakes) {
                let a = edge.action_idx[i];
                node.nvisit[i][a] += 1.0 - vloss;
                node.wsum[i][a] += v + vloss;
            }
        }
    }

    /// Batched selection with virtual loss: descend up to `max_leaves` times,
    /// collecting distinct unexpanded leaves for one batched net eval. Terminal
    /// leaves reached en route are backed up immediately. Returns the collected
    /// leaf node ids (their boards/paths are held in `self.batch` for
    /// `write`/`expand_backup`). A descent that lands on an already-collected
    /// leaf ends the round (its virtual loss is rolled back).
    fn select_batch(&mut self, max_leaves: usize, vloss: f32) -> Vec<usize> {
        self.batch.clear();
        self.inflight.clear();
        let n = self.n_snakes;
        let mut leaves = Vec::new();
        'round: for _ in 0..max_leaves {
            let mut path: Vec<Edge> = Vec::new();
            let mut id = 0usize;
            loop {
                if self.nodes[id].terminal {
                    let v = self.nodes[id].term_value;
                    self.backup_path(&path, &v, vloss);
                    break;
                }
                if !self.nodes[id].expanded {
                    if self.inflight.contains(&id) {
                        // Collision (e.g. forced line virtual loss can't separate):
                        // abandon this descent and stop collecting more.
                        self.remove_virtual_loss(&path, vloss);
                        break 'round;
                    }
                    self.inflight.insert(id);
                    leaves.push(id);
                    self.batch.push(PendingLeaf { leaf: id, path });
                    break;
                }
                let strides = self.node_strides(id);
                let (joint, action_idx) = self.select_joint(id, &strides);
                path.push(Edge {
                    node: id,
                    action_idx,
                });
                self.apply_virtual_loss(&path[path.len() - 1..], vloss);
                match self.nodes[id].child(joint) {
                    Some(cid) => id = cid,
                    None => {
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
                        let child = MctsNode::leaf(child_board, self.draw_value);
                        let is_terminal = child.terminal;
                        let term_value = child.term_value;
                        self.nodes.push(child);
                        self.nodes[id].children.push((joint, cid));
                        if is_terminal {
                            self.backup_path(&path, &term_value, vloss);
                        } else {
                            self.inflight.insert(cid);
                            leaves.push(cid);
                            self.batch.push(PendingLeaf { leaf: cid, path });
                        }
                        break;
                    }
                }
            }
        }
        leaves
    }

    /// Expand every leaf collected by [`select_batch`] with its priors and back
    /// up its value (virtual loss removed). `policies` is `[batch, n_snakes, 4]`
    /// and `values` is `[batch, n_snakes]`, aligned with the returned leaf order.
    fn expand_backup_batch(&mut self, policies: &[f32], values: &[f32], vloss: f32) {
        let n = self.n_snakes;
        let batch = std::mem::take(&mut self.batch);
        for (pos, pending) in batch.into_iter().enumerate() {
            let id = pending.leaf;
            let pol = &policies[pos * n * 4..(pos + 1) * n * 4];
            self.expand_leaf(id, pol);
            let mut leaf_value = [0.0f32; MAX_SNAKES];
            for i in 0..n {
                leaf_value[i] = if self.nodes[id].board.snakes[i].alive() {
                    values[pos * n + i]
                } else {
                    -1.0
                };
            }
            self.backup_path(&pending.path, &leaf_value, vloss);
        }
        self.inflight.clear();
    }

    /// Set candidates/priors on an unexpanded leaf (death-masked, normalized) —
    /// the expansion half of `expand_backup`, factored out for the batched path.
    fn expand_leaf(&mut self, id: usize, policy: &[f32]) {
        let n = self.n_snakes;
        let board = self.nodes[id].board.clone();
        let cands: Vec<Vec<Move>> = (0..n).map(|i| candidates(&board, i)).collect();
        let mut prior = Vec::with_capacity(n);
        let mut nvisit = Vec::with_capacity(n);
        let mut wsum = Vec::with_capacity(n);
        for i in 0..n {
            let k = cands[i].len();
            let masked_policy = mask_obvious_immediate_deaths(&board, i, &policy[i * 4..i * 4 + 4]);
            let mut p = vec![0.0f32; k];
            let mut s = 0.0f32;
            for (a, m) in cands[i].iter().enumerate() {
                p[a] = masked_policy[m.index()];
                s += p[a];
            }
            if s > 1e-8 {
                for x in p.iter_mut() {
                    *x /= s;
                }
            } else if k > 0 {
                let safe_count = cands[i]
                    .iter()
                    .filter(|&&m| !obvious_immediate_death(&board, i, m))
                    .count();
                if safe_count > 0 {
                    let u = 1.0 / safe_count as f32;
                    for (a, x) in p.iter_mut().enumerate() {
                        if !obvious_immediate_death(&board, i, cands[i][a]) {
                            *x = u;
                        }
                    }
                } else {
                    let u = 1.0 / k as f32;
                    p.fill(u);
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
            let has_positive_prior = node.prior[i].iter().any(|&p| p > 1e-8);
            let mut best_a = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (a, &n_a) in nv.iter().enumerate().take(node.cands[i].len()) {
                if has_positive_prior && node.prior[i][a] <= 1e-8 {
                    continue;
                }
                let q = if n_a > 0.0 {
                    node.wsum[i][a] / n_a
                } else {
                    0.0
                };
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
            self.pending_path.push(Edge {
                node: id,
                action_idx,
            });
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
                    let child = MctsNode::leaf(child_board, self.draw_value);
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
            for (i, &agent_value) in value.iter().enumerate().take(self.n_snakes) {
                let a = edge.action_idx[i];
                node.nvisit[i][a] += 1.0;
                node.wsum[i][a] += agent_value;
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
        self.expand_leaf(id, policy);
        let mut leaf_value = *value;
        for (i, v) in leaf_value.iter_mut().enumerate().take(n) {
            if !self.nodes[id].board.snakes[i].alive() {
                *v = -1.0;
            }
        }
        self.backup(&leaf_value);
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

    fn root_debug(&self) -> Vec<Vec<(usize, f32, f32, f32)>> {
        let root = &self.nodes[0];
        let mut out = Vec::with_capacity(self.n_snakes);
        for i in 0..self.n_snakes {
            let mut row = Vec::new();
            if root.expanded {
                for (a, m) in root.cands[i].iter().enumerate() {
                    let visits = root.nvisit[i][a];
                    let q = if visits > 0.0 {
                        root.wsum[i][a] / visits
                    } else {
                        0.0
                    };
                    row.push((m.index(), root.prior[i][a], visits, q));
                }
            }
            out.push(row);
        }
        out
    }

    /// Max tree depth (root = 0). Cheap: one pass over the arena, which is built
    /// root-first so every child has a higher id than its parent.
    fn max_depth(&self) -> u32 {
        let mut depth = vec![0u32; self.nodes.len()];
        let mut max = 0u32;
        for (id, node) in self.nodes.iter().enumerate() {
            for &(_, cid) in &node.children {
                depth[cid] = depth[id] + 1;
                max = max.max(depth[cid]);
            }
        }
        max
    }

    /// Full snapshot of the search tree for inspection (the viewer's tree
    /// explorer). Walks every node, decoding per-snake action stats, child edges,
    /// and each node's board so a frontend can render and drill into any line.
    fn snapshot(&self) -> TreeSnapshot {
        // Depth via the parent map implied by children edges; the arena is built
        // root-first so a single forward pass assigns every node its depth.
        let mut depth = vec![0u32; self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            for &(_, cid) in &node.children {
                depth[cid] = depth[id] + 1;
            }
        }

        let nodes = self
            .nodes
            .iter()
            .enumerate()
            .map(|(id, node)| {
                let actions = (0..self.n_snakes)
                    .map(|i| {
                        node.cands
                            .get(i)
                            .map(|cands| {
                                cands
                                    .iter()
                                    .enumerate()
                                    .map(|(a, m)| {
                                        let visits = node.nvisit[i][a];
                                        let q = if visits > 0.0 {
                                            node.wsum[i][a] / visits
                                        } else {
                                            0.0
                                        };
                                        ActionStat {
                                            move_index: m.index(),
                                            prior: node.prior[i][a],
                                            visits,
                                            q,
                                        }
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect::<Vec<Vec<ActionStat>>>();

                // Decode each child's joint index back into per-snake action
                // slots (same mixed-radix layout `select` builds them with), then
                // map those slots to move indices for display. Only expanded nodes
                // carry candidate layouts; leaves have none and also no children.
                let children = if node.expanded {
                    let mut strides = [1u32; MAX_SNAKES];
                    for i in (0..self.n_snakes).rev() {
                        strides[i] = if i + 1 < self.n_snakes {
                            strides[i + 1] * node.cands[i + 1].len() as u32
                        } else {
                            1
                        };
                    }
                    node.children
                        .iter()
                        .map(|&(joint, cid)| {
                            let moves = (0..self.n_snakes)
                                .map(|i| {
                                    let slot = ((joint / strides[i]) as usize)
                                        % node.cands[i].len().max(1);
                                    node.cands[i].get(slot).map(|m| m.index()).unwrap_or(0)
                                })
                                .collect();
                            ChildEdge { child: cid, moves }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                let total_visits = node.nvisit.first().map(|v| v.iter().sum()).unwrap_or(0.0);
                let snakes = node
                    .board
                    .snakes
                    .iter()
                    .map(|s| NodeSnake {
                        alive: s.alive(),
                        health: s.health,
                        body: s.body.iter().map(|p| [p.x, p.y]).collect(),
                    })
                    .collect();

                TreeNodeSnapshot {
                    id,
                    depth: depth[id],
                    terminal: node.terminal,
                    expanded: node.expanded,
                    total_visits,
                    term_value: if node.terminal {
                        node.term_value[..self.n_snakes].to_vec()
                    } else {
                        Vec::new()
                    },
                    actions,
                    children,
                    snakes,
                }
            })
            .collect();

        TreeSnapshot {
            n_snakes: self.n_snakes,
            nodes,
        }
    }
}

/// Per-action search stats for one snake at one node: the prior, accumulated
/// visit count, and mean backed-up value (Q).
#[derive(Clone, Debug)]
pub struct ActionStat {
    pub move_index: usize,
    pub prior: f32,
    pub visits: f32,
    pub q: f32,
}

/// An edge from a node to one explored child, labelled with the per-snake move
/// that produced it.
#[derive(Clone, Debug)]
pub struct ChildEdge {
    pub child: usize,
    /// Per-snake move index taken to reach `child`.
    pub moves: Vec<usize>,
}

/// One snake's body on a snapshot node, head first as `[x, y]` pairs.
#[derive(Clone, Debug)]
pub struct NodeSnake {
    pub alive: bool,
    pub health: i16,
    pub body: Vec<[i8; 2]>,
}

/// One node of a [`TreeSnapshot`].
#[derive(Clone, Debug)]
pub struct TreeNodeSnapshot {
    pub id: usize,
    pub depth: u32,
    pub terminal: bool,
    pub expanded: bool,
    /// Sims that passed through this node.
    pub total_visits: f32,
    /// Terminal value per snake (empty unless `terminal`).
    pub term_value: Vec<f32>,
    /// Per-snake action stats: `actions[snake][candidate]`.
    pub actions: Vec<Vec<ActionStat>>,
    pub children: Vec<ChildEdge>,
    /// Board at this node, for rendering.
    pub snakes: Vec<NodeSnake>,
}

/// A full search-tree snapshot for inspection. Node 0 is the root.
#[derive(Clone, Debug)]
pub struct TreeSnapshot {
    pub n_snakes: usize,
    pub nodes: Vec<TreeNodeSnapshot>,
}

/// A batch of independent MCTS trees, one per game, driven in lockstep.
pub struct MctsForest {
    trees: Vec<MctsTree>,
    /// Trees excluded from further selection (their root policy is already exact
    /// — see [`MctsForest::freeze_forced_roots`]). Frozen trees drop out of the
    /// eval batch, so self-play stops spending sims on already-decided games.
    frozen: Vec<bool>,
    pub n_snakes: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl MctsForest {
    pub fn new(boards: &[Board], c_puct: f32) -> Self {
        Self::new_with_draw_value(boards, c_puct, 0.0)
    }

    pub fn new_with_draw_value(boards: &[Board], c_puct: f32, draw_value: f32) -> Self {
        let n_snakes = boards.first().map(|b| b.snakes.len()).unwrap_or(0);
        // Observation canvas dims are absolute board coords (obs_side = side).
        let (height, width) = boards
            .first()
            .map(|b| (obs_side(b.height as usize), obs_side(b.width as usize)))
            .unwrap_or((0, 0));
        MctsForest {
            trees: boards
                .iter()
                .map(|b| MctsTree::new_with_draw_value(b.clone(), c_puct, draw_value))
                .collect(),
            frozen: vec![false; boards.len()],
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
    /// already backed up). Frozen trees ([`freeze_forced_roots`]) are skipped.
    /// The caller then evaluates exactly the returned trees.
    pub fn select(&mut self) -> Vec<usize> {
        self.trees
            .par_iter_mut()
            .zip(self.frozen.par_iter())
            .for_each(|(t, &fz)| {
                if fz {
                    t.pending_leaf = None;
                } else {
                    t.select();
                }
            });
        (0..self.trees.len())
            .filter(|&i| !self.frozen[i] && self.trees[i].pending_leaf.is_some())
            .collect()
    }

    /// Freeze every still-active tree whose root decision is already exact *no
    /// matter how many more sims run*: all alive snakes have ≤1 legal candidate,
    /// so each snake's root visit policy is one-hot on its single move. Call this
    /// once the roots have been expanded and credited at least one visit (so the
    /// one-hot is materialized in the visit counts). Frozen trees are then skipped
    /// by [`select`], dropping them from the eval batch. Returns the number newly
    /// frozen. (Terminal-root trees already produce no leaves, so are left alone.)
    pub fn freeze_forced_roots(&mut self) -> usize {
        let mut newly = 0;
        for i in 0..self.trees.len() {
            if self.frozen[i] {
                continue;
            }
            let root = &self.trees[i].nodes[0];
            if !root.expanded {
                continue;
            }
            let board = &root.board;
            let all_forced = (0..self.n_snakes).all(|s| {
                s >= board.snakes.len()
                    || !board.snakes[s].alive()
                    || candidates(board, s).len() <= 1
            });
            if all_forced {
                self.frozen[i] = true;
                newly += 1;
            }
        }
        newly
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

    /// Batched virtual-loss selection on the **first tree only** (the serving
    /// case is a single board). Collects up to `max_leaves` distinct leaves for
    /// one batched net forward; terminal leaves are backed up immediately.
    /// Returns the number of leaves collected (0 once the tree can't produce a
    /// fresh leaf — e.g. fully terminal). Pair with [`write_batch_obs_first`] and
    /// [`expand_backup_batch_first`].
    pub fn select_batch_first(&mut self, max_leaves: usize, vloss: f32) -> usize {
        match self.trees.first_mut() {
            Some(tree) => tree.select_batch(max_leaves, vloss).len(),
            None => 0,
        }
    }

    /// Write observations for the first tree's collected batch into `out`, laid
    /// out `[leaf, agent]` (agent innermost), `batch_len * n_snakes * obs_size`
    /// floats.
    pub fn write_batch_obs_first(&self, out: &mut [f32]) {
        let obs = self.obs_size();
        let chunk = self.n_snakes * obs;
        let Some(tree) = self.trees.first() else {
            return;
        };
        out.par_chunks_mut(chunk)
            .zip(tree.batch.par_iter())
            .for_each(|(buf, pending)| {
                let board = &tree.nodes[pending.leaf].board;
                for agent in 0..self.n_snakes {
                    let base = agent * obs;
                    encode_into(board, agent, &mut buf[base..base + obs]);
                }
            });
    }

    /// Expand+back up the first tree's batch (removing virtual loss). `policies`
    /// is `[batch, n_snakes, 4]`, `values` is `[batch, n_snakes]`.
    pub fn expand_backup_batch_first(&mut self, policies: &[f32], values: &[f32], vloss: f32) {
        if let Some(tree) = self.trees.first_mut() {
            tree.expand_backup_batch(policies, values, vloss);
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

    /// Root action diagnostics for the first tree in this forest.
    /// Each snake row contains `(move_index, prior, visits, q)`.
    pub fn root_debug_first(&self) -> Vec<Vec<(usize, f32, f32, f32)>> {
        self.trees
            .first()
            .map(|tree| tree.root_debug())
            .unwrap_or_default()
    }

    /// Full tree snapshot for the first tree in this forest (the viewer uses this
    /// to render and explore the exploration tree of a replayed move).
    pub fn tree_snapshot_first(&self) -> Option<TreeSnapshot> {
        self.trees.first().map(|tree| tree.snapshot())
    }

    /// Max search depth reached in the first tree (root = 0). Cheap enough to
    /// record on every served move.
    pub fn max_depth_first(&self) -> u32 {
        self.trees.first().map(|tree| tree.max_depth()).unwrap_or(0)
    }

    /// `(best_visits, second_visits)` over snake `me`'s root candidate visit
    /// counts in the first tree, restricted to unmasked (positive-prior)
    /// candidates — the only ones `choose_root_action` can pick. Deadline-bound
    /// serving uses this to stop once the leader is mathematically uncatchable in
    /// the remaining budget. `None` until the root is expanded.
    pub fn root_visit_gap_first(&self, me: usize) -> Option<(f32, f32)> {
        let tree = self.trees.first()?;
        let root = &tree.nodes[0];
        if !root.expanded || me >= self.n_snakes {
            return None;
        }
        let has_prior = root.prior[me].iter().any(|&p| p > 1e-8);
        let (mut best, mut second) = (0.0f32, 0.0f32);
        for (a, &nv) in root.nvisit[me].iter().enumerate() {
            if has_prior && root.prior[me][a] <= 1e-8 {
                continue;
            }
            if nv > best {
                second = best;
                best = nv;
            } else if nv > second {
                second = nv;
            }
        }
        Some((best, second))
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

    fn dead_leaf_ffa_board() -> Board {
        let mut b = Board::new(11, 11);
        // Snake 0 moving Up hits its own body at (5,6), but the game continues
        // with the other snakes alive. That leaf must be -1 for snake 0 even if
        // the value net gives a bad estimate for the dead snake.
        b.add_snake(&[
            Point::new(5, 5),
            Point::new(5, 4),
            Point::new(4, 4),
            Point::new(4, 5),
            Point::new(5, 6),
            Point::new(4, 6),
        ]);
        b.add_snake(&[Point::new(9, 9), Point::new(9, 8)]);
        b.add_snake(&[Point::new(1, 9), Point::new(1, 8)]);
        b.add_snake(&[Point::new(9, 1), Point::new(9, 2)]);
        b
    }

    fn opponent_body_board(tail_at_target: bool) -> Board {
        let mut b = Board::new(7, 7);
        b.add_snake(&[Point::new(2, 2), Point::new(2, 1)]);
        if tail_at_target {
            b.add_snake(&[
                Point::new(5, 2),
                Point::new(5, 1),
                Point::new(4, 1),
                Point::new(3, 1),
                Point::new(3, 2),
            ]);
        } else {
            b.add_snake(&[
                Point::new(5, 2),
                Point::new(4, 2),
                Point::new(3, 2),
                Point::new(3, 1),
            ]);
        }
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
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "snake {i} policy sums to 1, got {sum}"
            );
            assert!(p.iter().all(|&x| x >= 0.0 && x.is_finite()));
        }
        // With 64 sims the tree must have grown well past the root.
        assert!(forest.trees[0].nodes.len() > 10);
    }

    /// The tree snapshot must walk every node without panicking — including the
    /// unexpanded leaves that carry no candidate layout (a past regression) — and
    /// report a sane depth and child topology.
    #[test]
    fn tree_snapshot_covers_leaves_and_depth() {
        let mut forest = MctsForest::new(&[two_snake_board()], 1.5);
        let n = forest.n_snakes;
        let obs = forest.obs_size();
        for _ in 0..48 {
            let pending = forest.select();
            if pending.is_empty() {
                continue;
            }
            let mut buf = vec![0.0f32; pending.len() * n * obs];
            forest.write_pending_obs(&pending, &mut buf);
            let pol = vec![0.25f32; pending.len() * n * 4];
            let val = vec![0.0f32; pending.len() * n];
            forest.expand_backup(&pending, &pol, &val);
        }

        let snap = forest.tree_snapshot_first().expect("snapshot");
        assert_eq!(snap.n_snakes, n);
        assert_eq!(snap.nodes.len(), forest.trees[0].nodes.len());
        assert_eq!(snap.nodes[0].id, 0);
        assert_eq!(snap.nodes[0].depth, 0);

        // Unexpanded nodes (freshly created leaves, terminals) carry no candidate
        // layout — the snapshot must not try to decode child joints for them
        // (a past panic indexed their empty candidate list). Every child move
        // must still decode into the legal 0..4 range.
        for nd in &snap.nodes {
            if !nd.expanded {
                assert!(nd.children.is_empty(), "unexpanded node has no children");
                assert!(nd.actions.iter().all(|row| row.is_empty()));
            }
            for c in &nd.children {
                assert_eq!(c.moves.len(), n);
                assert!(c.moves.iter().all(|&m| m < 4));
            }
        }

        let reported = forest.max_depth_first();
        let computed = snap.nodes.iter().map(|nd| nd.depth).max().unwrap_or(0);
        assert_eq!(reported, computed);
        assert!(reported >= 1, "tree should be deeper than the root");
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

    #[test]
    fn mcts_masks_obvious_self_death_prior_at_root() {
        let mut forest = MctsForest::new(&[dead_leaf_ffa_board()], 1.5);
        let n = forest.n_snakes;

        let pending = forest.select();
        assert_eq!(pending, vec![0]);
        let mut root_pol = vec![0.01f32; n * 4];
        root_pol[Move::Up.index()] = 0.97;
        let root_val = vec![0.0f32; n];
        forest.expand_backup(&pending, &root_pol, &root_val);

        let root = &forest.trees[0].nodes[0];
        let up_idx = root.cands[0]
            .iter()
            .position(|&m| m == Move::Up)
            .expect("Up remains a candidate");
        assert_eq!(root.prior[0][up_idx], 0.0);

        for _ in 0..16 {
            let pending = forest.select();
            if pending.is_empty() {
                continue;
            }
            let pol = vec![0.25f32; pending.len() * n * 4];
            let val = vec![0.0f32; pending.len() * n];
            forest.expand_backup(&pending, &pol, &val);
        }
        let root = &forest.trees[0].nodes[0];
        assert_eq!(root.nvisit[0][up_idx], 0.0);
    }

    #[test]
    fn mcts_never_selects_masked_death_when_safe_moves_exist() {
        let mut forest = MctsForest::new(&[dead_leaf_ffa_board()], 1.5);
        let n = forest.n_snakes;

        let pending = forest.select();
        assert_eq!(pending, vec![0]);
        let mut root_pol = vec![0.01f32; n * 4];
        root_pol[Move::Up.index()] = 0.97;
        let root_val = vec![0.0f32; n];
        forest.expand_backup(&pending, &root_pol, &root_val);

        let up_idx = forest.trees[0].nodes[0].cands[0]
            .iter()
            .position(|&m| m == Move::Up)
            .expect("Up remains a candidate");
        {
            let root = &mut forest.trees[0].nodes[0];
            for (a, prior) in root.prior[0].iter().enumerate() {
                if a != up_idx && *prior > 0.0 {
                    root.nvisit[0][a] = 1.0;
                    root.wsum[0][a] = -1.0;
                }
            }
        }

        for _ in 0..64 {
            let pending = forest.select();
            if pending.is_empty() {
                continue;
            }
            let pol = vec![0.25f32; pending.len() * n * 4];
            let val = vec![0.0f32; pending.len() * n];
            forest.expand_backup(&pending, &pol, &val);
        }

        let root = &forest.trees[0].nodes[0];
        assert_eq!(root.prior[0][up_idx], 0.0);
        assert_eq!(root.nvisit[0][up_idx], 0.0);
    }

    /// Batched virtual-loss selection must grow the tree, leave no virtual-loss
    /// residue (all visit counts non-negative and root visits integral), and
    /// produce a valid policy — i.e. behave like the single-leaf path at the root.
    #[test]
    fn mcts_batched_selection_is_consistent() {
        let mut forest = MctsForest::new(&[two_snake_board()], 1.5);
        let n = forest.n_snakes;
        let obs = forest.obs_size();
        for _ in 0..32 {
            let k = forest.select_batch_first(8, 1.0);
            if k == 0 {
                continue;
            }
            let mut buf = vec![0.0f32; k * n * obs];
            forest.write_batch_obs_first(&mut buf);
            let pol = vec![0.25f32; k * n * 4];
            let val = vec![0.0f32; k * n];
            forest.expand_backup_batch_first(&pol, &val, 1.0);
        }
        let (policies, values) = forest.root_targets();
        for i in 0..n {
            let p = &policies[i * 4..i * 4 + 4];
            let sum: f32 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "snake {i} policy sums to {sum}");
            assert!(p.iter().all(|&x| x >= 0.0 && x.is_finite()));
        }
        assert!(values.iter().all(|v| v.is_finite()));
        // No virtual-loss residue: every node's visit counts are non-negative and
        // each root action's visit count is (near-)integral after backups.
        for node in &forest.trees[0].nodes {
            for row in &node.nvisit {
                for &nv in row {
                    assert!(nv >= -1e-4, "negative visit residue {nv}");
                    assert!((nv - nv.round()).abs() < 1e-3, "non-integral visits {nv}");
                }
            }
        }
        assert!(
            forest.trees[0].nodes.len() > 10,
            "tree must grow past the root"
        );
    }

    /// Cost: a snake with a single legal move is detected as forced, so serving
    /// can skip the search entirely (zero sims, zero net forwards). A snake with
    /// real choices is not forced.
    #[test]
    fn forced_move_skips_search_only_when_truly_forced() {
        let mut b = Board::new(11, 11);
        // Snake 0 in the corner, neck above: Up=neck, Down/Left off-board, only
        // Right is legal -> forced.
        b.add_snake(&[Point::new(0, 0), Point::new(0, 1)]);
        // Snake 1 in open space: Down/Left/Right all legal -> not forced.
        b.add_snake(&[Point::new(5, 5), Point::new(5, 6)]);
        assert_eq!(forced_move(&b, 0), Some(Move::Right.index()));
        assert_eq!(forced_move(&b, 1), None);
    }

    /// Cost: each batched round is one net forward, so collecting up to `k` leaves
    /// per round means far fewer forwards than the single-leaf path (1 leaf = 1
    /// forward). Verify a round really batches ~k leaves and that overall we
    /// evaluate strictly more leaves than we spend forward passes.
    #[test]
    fn batched_selection_amortizes_forward_passes() {
        let k = 16;
        let mut forest = MctsForest::new(&[two_snake_board()], 1.5);
        let n = forest.n_snakes;
        let obs = forest.obs_size();
        let (mut rounds, mut leaves, mut max_round) = (0usize, 0usize, 0usize);
        for _ in 0..20 {
            let got = forest.select_batch_first(k, 1.0); // == one net forward
            if got == 0 {
                break;
            }
            rounds += 1;
            leaves += got;
            max_round = max_round.max(got);
            let mut buf = vec![0.0f32; got * n * obs];
            forest.write_batch_obs_first(&mut buf);
            let pol = vec![0.25f32; got * n * 4];
            let val = vec![0.0f32; got * n];
            forest.expand_backup_batch_first(&pol, &val, 1.0);
        }
        // Single-leaf serving would need one forward *per leaf*; batching needs one
        // forward *per round*, so leaves > rounds proves the amortization.
        assert!(
            leaves > rounds,
            "batching must evaluate >1 leaf per forward (leaves={leaves}, rounds={rounds})"
        );
        // And an open board should fill close to a full batch in a round.
        assert!(
            max_round >= k / 2,
            "a round should batch ~k leaves, got at most {max_round}"
        );
    }

    /// The early-stop signal (`root_visit_gap_first`) reports the leading and
    /// runner-up root visit counts for the served snake, ordered best >= second —
    /// the quantity the serving loop compares against the remaining budget.
    #[test]
    fn root_visit_gap_orders_leader_and_runner_up() {
        let mut forest = MctsForest::new(&[two_snake_board()], 1.5);
        let n = forest.n_snakes;
        for _ in 0..128 {
            let pending = forest.select();
            if pending.is_empty() {
                continue;
            }
            // Bias snake 0's prior hard toward one move so a clear leader emerges.
            let mut pol = vec![0.25f32; pending.len() * n * 4];
            for leaf in 0..pending.len() {
                let base = leaf * n * 4;
                pol[base..base + 4].copy_from_slice(&[0.9, 0.0333, 0.0333, 0.0334]);
            }
            let val = vec![0.0f32; pending.len() * n];
            forest.expand_backup(&pending, &pol, &val);
        }
        let (best, second) = forest.root_visit_gap_first(0).expect("root expanded");
        assert!(best >= second, "best {best} should be >= second {second}");
        assert!(best > 0.0, "leader must have visits");
    }

    /// Cost (self-play Tier 1): a game where every alive snake has one legal move
    /// is frozen after the root is credited — it drops out of selection (no more
    /// sims spent) while its recorded policy stays an exact one-hot. A normal game
    /// keeps searching.
    #[test]
    fn freeze_forced_roots_drops_decided_games_with_exact_policy() {
        // game 0: both snakes cornered with a single legal move -> fully forced.
        let mut forced = Board::new(11, 11);
        forced.add_snake(&[Point::new(0, 0), Point::new(0, 1)]); // only Right
        forced.add_snake(&[Point::new(10, 10), Point::new(10, 9)]); // only Left
                                                                    // game 1: open board -> not forced.
        let mut forest = MctsForest::new(&[forced, two_snake_board()], 1.5);
        let n = forest.n_snakes;

        let mut frozen_count = 0;
        for sim in 0..8 {
            let pending = forest.select();
            if !pending.is_empty() {
                let pol = vec![0.25f32; pending.len() * n * 4];
                let val = vec![0.0f32; pending.len() * n];
                forest.expand_backup(&pending, &pol, &val);
            }
            if sim == 1 {
                frozen_count = forest.freeze_forced_roots();
            }
        }
        assert_eq!(frozen_count, 1, "only the fully-forced game freezes");

        // Cost: the frozen game is no longer selected (dropped from the batch).
        for _ in 0..4 {
            assert!(
                !forest.select().contains(&0),
                "frozen game 0 must not be selected again"
            );
        }
        // Correctness: its target is the exact one-hot (Right for snake 0, Left
        // for snake 1), identical to what a full search would produce.
        let (policies, _) = forest.root_targets();
        assert!(
            (policies[Move::Right.index()] - 1.0).abs() < 1e-6,
            "snake0 -> Right"
        );
        assert!(
            (policies[4 + Move::Left.index()] - 1.0).abs() < 1e-6,
            "snake1 -> Left"
        );
    }

    #[test]
    fn obvious_immediate_death_distinguishes_opponent_tail() {
        assert!(obvious_immediate_death(
            &opponent_body_board(false),
            0,
            Move::Right
        ));
        assert!(!obvious_immediate_death(
            &opponent_body_board(true),
            0,
            Move::Right
        ));
    }
}
