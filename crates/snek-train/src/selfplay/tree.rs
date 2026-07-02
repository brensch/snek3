//! POD MCTS node + reusable per-game tree arena.
//!
//! Nodes are fixed-size POD (candidates <= 4, snakes <= MAX_SNAKES), so a game's
//! tree is a flat `Vec<Node>` reused turn after turn with no hot-path heap
//! traffic and no per-turn barrier.

use super::rules::{
    candidates, mask_obvious_immediate_deaths, obvious_immediate_death, terminal_values,
};
use super::{EPS, MAXC};
use snek_core::{Board, Move, MAX_SNAKES};

struct Node {
    board: Board,
    terminal: bool,
    term_value: [f32; MAX_SNAKES],
    expanded: bool,
    ncand: [usize; MAX_SNAKES],
    cand: [[u8; MAXC]; MAX_SNAKES],
    prior: [[f32; MAXC]; MAX_SNAKES],
    nvisit: [[f32; MAXC]; MAX_SNAKES],
    wsum: [[f32; MAXC]; MAX_SNAKES],
    children: Vec<(u32, u32)>, // (joint index -> child id); reused via clear()
}

impl Node {
    fn empty(w: i8, h: i8) -> Self {
        Node {
            board: Board::new(w, h),
            terminal: false,
            term_value: [0.0; MAX_SNAKES],
            expanded: false,
            ncand: [0; MAX_SNAKES],
            cand: [[0; MAXC]; MAX_SNAKES],
            prior: [[0.0; MAXC]; MAX_SNAKES],
            nvisit: [[0.0; MAXC]; MAX_SNAKES],
            wsum: [[0.0; MAXC]; MAX_SNAKES],
            children: Vec::new(),
        }
    }

    fn reset_leaf_flags(&mut self, draw: f32) {
        self.terminal = self.board.is_terminal();
        self.term_value = if self.terminal {
            terminal_values(&self.board, draw)
        } else {
            [0.0; MAX_SNAKES]
        };
        self.expanded = false;
        self.children.clear();
    }
}

#[derive(Clone, Copy)]
struct Edge {
    node: u32,
    action: [u8; MAX_SNAKES],
}

pub(super) struct Tree {
    nodes: Vec<Node>,
    len: usize,
    n: usize,
    w: i8,
    h: i8,
    c_puct: f32,
    draw: f32,
    pending: Option<usize>,
    path: Vec<Edge>,
}

impl Tree {
    pub(super) fn new(n: usize, w: i8, h: i8, c_puct: f32, draw: f32, cap: usize) -> Self {
        let mut nodes = Vec::with_capacity(cap);
        for _ in 0..cap {
            nodes.push(Node::empty(w, h));
        }
        Tree {
            nodes,
            len: 0,
            n,
            w,
            h,
            c_puct,
            draw,
            pending: None,
            path: Vec::with_capacity(64),
        }
    }

    pub(super) fn reset(&mut self, board: &Board) {
        self.nodes[0].board.clone_from(board);
        self.nodes[0].reset_leaf_flags(self.draw);
        self.len = 1;
        self.pending = None;
        self.path.clear();
    }

    /// The board of the leaf awaiting a network evaluation, if any. The play loop
    /// encodes it into the batch; a terminal descent leaves no pending leaf.
    pub(super) fn pending_board(&self) -> Option<&Board> {
        self.pending.map(|id| &self.nodes[id].board)
    }

    #[inline]
    fn ensure_slot(&mut self) -> usize {
        let id = self.len;
        if id == self.nodes.len() {
            self.nodes.push(Node::empty(self.w, self.h));
        }
        id
    }

    fn spawn_child(&mut self, parent: usize, mv: &[Move]) -> usize {
        let id = self.ensure_slot();
        let (left, right) = self.nodes.split_at_mut(id);
        right[0].board.clone_from(&left[parent].board);
        right[0].board.step(mv);
        right[0].reset_leaf_flags(self.draw);
        self.len += 1;
        id
    }

    #[inline]
    fn select_joint(&self, id: usize) -> (u32, [u8; MAX_SNAKES]) {
        let node = &self.nodes[id];
        let mut strides = [1u32; MAX_SNAKES];
        for i in (0..self.n).rev() {
            strides[i] = if i + 1 < self.n {
                strides[i + 1] * node.ncand[i + 1] as u32
            } else {
                1
            };
        }
        let mut action = [0u8; MAX_SNAKES];
        let mut joint = 0u32;
        for i in 0..self.n {
            let k = node.ncand[i];
            let total_n: f32 = node.nvisit[i][..k].iter().sum();
            let sqrt_total = total_n.max(1.0).sqrt();
            let has_prior = node.prior[i][..k].iter().any(|&p| p > EPS);
            let mut best_a = 0usize;
            let mut best = f32::NEG_INFINITY;
            for a in 0..k {
                if has_prior && node.prior[i][a] <= EPS {
                    continue;
                }
                let n_a = node.nvisit[i][a];
                let q = if n_a > 0.0 {
                    node.wsum[i][a] / n_a
                } else {
                    0.0
                };
                let u = self.c_puct * node.prior[i][a] * sqrt_total / (1.0 + n_a);
                let score = q + u;
                if score > best {
                    best = score;
                    best_a = a;
                }
            }
            action[i] = best_a as u8;
            joint += best_a as u32 * strides[i];
        }
        (joint, action)
    }

    /// Descend to a leaf; terminal leaves are backed up immediately (pending=None).
    pub(super) fn select(&mut self) {
        self.path.clear();
        self.pending = None;
        let mut id = 0usize;
        loop {
            if self.nodes[id].terminal {
                let v = self.nodes[id].term_value;
                self.backup(&v);
                return;
            }
            if !self.nodes[id].expanded {
                self.pending = Some(id);
                return;
            }
            let (joint, action) = self.select_joint(id);
            self.path.push(Edge {
                node: id as u32,
                action,
            });
            match self.child(id, joint) {
                Some(cid) => id = cid,
                None => {
                    let mut mv = [Move::Up; MAX_SNAKES];
                    {
                        let node = &self.nodes[id];
                        for i in 0..self.n {
                            mv[i] = Move::from_index(node.cand[i][action[i] as usize] as usize);
                        }
                    }
                    let cid = self.spawn_child(id, &mv[..self.n]);
                    self.nodes[id].children.push((joint, cid as u32));
                    if self.nodes[cid].terminal {
                        let v = self.nodes[cid].term_value;
                        self.backup(&v);
                    } else {
                        self.pending = Some(cid);
                    }
                    return;
                }
            }
        }
    }

    #[inline]
    fn child(&self, id: usize, joint: u32) -> Option<usize> {
        self.nodes[id]
            .children
            .iter()
            .find(|(j, _)| *j == joint)
            .map(|(_, c)| *c as usize)
    }

    fn backup(&mut self, value: &[f32; MAX_SNAKES]) {
        for edge in &self.path {
            let node = &mut self.nodes[edge.node as usize];
            for (i, &v) in value.iter().enumerate().take(self.n) {
                let a = edge.action[i] as usize;
                node.nvisit[i][a] += 1.0;
                node.wsum[i][a] += v;
            }
        }
        self.path.clear();
    }

    /// Expand the pending leaf with the network's `policy`/`value` and back the
    /// value up the path. A no-op if there is no pending leaf (terminal descent).
    pub(super) fn expand_backup(&mut self, policy: &[f32], value: &[f32]) {
        let Some(id) = self.pending.take() else {
            return;
        };
        let n = self.n;
        let board = self.nodes[id].board.clone();
        for i in 0..n {
            let (cand, k) = candidates(&board, i);
            let masked = mask_obvious_immediate_deaths(&board, i, &policy[i * 4..i * 4 + 4]);
            let mut p = [0.0f32; MAXC];
            let mut s = 0.0f32;
            for a in 0..k {
                p[a] = masked[cand[a] as usize];
                s += p[a];
            }
            if s > EPS {
                for x in p.iter_mut().take(k) {
                    *x /= s;
                }
            } else if k > 0 {
                let safe: usize = (0..k)
                    .filter(|&a| {
                        !obvious_immediate_death(&board, i, Move::from_index(cand[a] as usize))
                    })
                    .count();
                if safe > 0 {
                    let u = 1.0 / safe as f32;
                    for a in 0..k {
                        if !obvious_immediate_death(&board, i, Move::from_index(cand[a] as usize)) {
                            p[a] = u;
                        }
                    }
                } else {
                    let u = 1.0 / k as f32;
                    for x in p.iter_mut().take(k) {
                        *x = u;
                    }
                }
            }
            let node = &mut self.nodes[id];
            node.ncand[i] = k;
            node.cand[i] = cand;
            node.prior[i] = p;
            node.nvisit[i] = [0.0; MAXC];
            node.wsum[i] = [0.0; MAXC];
        }
        self.nodes[id].expanded = true;

        let mut val = [0.0f32; MAX_SNAKES];
        for (i, v) in val.iter_mut().enumerate().take(n) {
            *v = if self.nodes[id].board.snakes[i].alive() {
                value[i]
            } else {
                -1.0
            };
        }
        self.backup(&val);
    }

    /// Root visit-count policy (`[n,4]`) and mean root value (`[n]`).
    pub(super) fn root_targets(&self, pol: &mut [f32], val: &mut [f32]) {
        for v in pol.iter_mut() {
            *v = 0.0;
        }
        for v in val.iter_mut() {
            *v = 0.0;
        }
        let root = &self.nodes[0];
        if !root.expanded {
            return;
        }
        for i in 0..self.n {
            let k = root.ncand[i];
            let total: f32 = root.nvisit[i][..k].iter().sum();
            if total > 0.0 {
                for a in 0..k {
                    pol[i * 4 + root.cand[i][a] as usize] = root.nvisit[i][a] / total;
                }
                val[i] = root.wsum[i][..k].iter().sum::<f32>() / total;
            }
        }
    }
}
