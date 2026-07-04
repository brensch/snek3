//! POD MCTS node + reusable per-game tree arena.
//!
//! Nodes are fixed-size POD (candidates <= 4, snakes <= MAX_SNAKES), so a game's
//! tree is a flat `Vec<Node>` reused turn after turn with no hot-path heap
//! traffic and no per-turn barrier.

use super::rules::{
    candidates, mask_obvious_immediate_deaths, mix_root_dirichlet, obvious_immediate_death,
    terminal_values,
};
use super::{EPS, MAXC};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
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

pub(crate) struct Tree {
    nodes: Vec<Node>,
    len: usize,
    n: usize,
    w: i8,
    h: i8,
    c_puct: f32,
    draw: f32,
    /// AlphaZero root exploration: prior ← (1-frac)·prior + frac·Dir(alpha)
    /// per snake at the root, sampled fresh every search (i.e. every turn, on
    /// root expansion). The visit-count training target is read from the
    /// noised search, so exploration reaches the targets — noise applied only
    /// to the played move cannot (snek3-14 plateaued exactly that way).
    noise_frac: f32,
    noise_alpha: f32,
    rng: Xoshiro256PlusPlus,
    pending: Option<usize>,
    path: Vec<Edge>,
}

impl Tree {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        n: usize,
        w: i8,
        h: i8,
        c_puct: f32,
        draw: f32,
        cap: usize,
        noise_frac: f32,
        noise_alpha: f32,
        noise_seed: u64,
    ) -> Self {
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
            noise_frac,
            noise_alpha,
            rng: Xoshiro256PlusPlus::seed_from_u64(noise_seed),
            pending: None,
            path: Vec::with_capacity(64),
        }
    }

    pub(crate) fn reset(&mut self, board: &Board) {
        self.nodes[0].board.clone_from(board);
        self.nodes[0].reset_leaf_flags(self.draw);
        self.len = 1;
        self.pending = None;
        self.path.clear();
    }

    /// The board of the leaf awaiting a network evaluation, if any. The play loop
    /// encodes it into the batch; a terminal descent leaves no pending leaf.
    pub(crate) fn pending_board(&self) -> Option<&Board> {
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
    pub(crate) fn select(&mut self) {
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
    pub(crate) fn expand_backup(&mut self, policy: &[f32], value: &[f32]) {
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
            // Root only: AlphaZero Dirichlet exploration over the masked-legal
            // candidates, so the search (and therefore the visit-count target)
            // is forced to try moves the raw prior dislikes.
            if id == 0 && board.snakes[i].alive() {
                mix_root_dirichlet(&mut p, k, self.noise_frac, self.noise_alpha, &mut self.rng);
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

    /// The root prior for snake `i` mapped back onto move indices (tests).
    #[cfg(test)]
    fn root_prior(&self, i: usize) -> [f32; 4] {
        let root = &self.nodes[0];
        let mut out = [0.0f32; 4];
        for a in 0..root.ncand[i] {
            out[root.cand[i][a] as usize] = root.prior[i][a];
        }
        out
    }

    /// Root visit-count policy (`[n,4]`) and mean root value (`[n]`).
    pub(crate) fn root_targets(&self, pol: &mut [f32], val: &mut [f32]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use snek_core::standard_start;

    fn expanded_root(noise_frac: f32, seed: u64) -> Tree {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let board = standard_start(11, 11, 4, &mut rng);
        let mut tree = Tree::new(4, 11, 11, 1.5, -0.25, 64, noise_frac, 0.3, seed);
        tree.reset(&board);
        tree.select();
        assert!(tree.pending_board().is_some(), "fresh root must be pending");
        let policy = vec![0.25f32; 16];
        let value = vec![0.0f32; 4];
        tree.expand_backup(&policy, &value);
        tree
    }

    #[test]
    fn root_noise_perturbs_priors_and_keeps_them_normalized() {
        let clean = expanded_root(0.0, 9);
        let noised = expanded_root(0.5, 9);
        let mut any_moved = false;
        for i in 0..4 {
            let c = clean.root_prior(i);
            let n = noised.root_prior(i);
            let sum: f32 = n.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "snake {i} prior sums to {sum}");
            // Moves the mask zeroed must stay zero under noise.
            for m in 0..4 {
                if c[m] == 0.0 {
                    assert_eq!(n[m], 0.0, "snake {i} move {m} revived by noise");
                }
            }
            any_moved |= (0..4).any(|m| (c[m] - n[m]).abs() > 1e-3);
        }
        assert!(any_moved, "noise must change at least one root prior");
    }

    #[test]
    fn root_noise_resamples_every_search() {
        // Same tree, two consecutive turns (reset + expand): the Dirichlet
        // draw must differ, so exploration varies turn to turn.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let board = standard_start(11, 11, 4, &mut rng);
        let mut tree = Tree::new(4, 11, 11, 1.5, -0.25, 64, 0.5, 0.3, 11);
        let policy = vec![0.25f32; 16];
        let value = vec![0.0f32; 4];
        let mut draws = Vec::new();
        for _ in 0..2 {
            tree.reset(&board);
            tree.select();
            tree.expand_backup(&policy, &value);
            draws.push((0..4).map(|i| tree.root_prior(i)).collect::<Vec<_>>());
        }
        assert_ne!(draws[0], draws[1], "root noise must be fresh per search");
    }

    #[test]
    fn non_root_nodes_get_no_noise() {
        // Descend to a child and expand it: with a uniform net policy its
        // priors must be exactly uniform over the safe candidates (no noise).
        let mut tree = expanded_root(1.0, 5);
        tree.select();
        if tree.pending_board().is_some() {
            let policy = vec![0.25f32; 16];
            let value = vec![0.0f32; 4];
            let id = tree.pending.unwrap();
            tree.expand_backup(&policy, &value);
            let node = &tree.nodes[id];
            for i in 0..4 {
                let k = node.ncand[i];
                let support: Vec<f32> = node.prior[i][..k]
                    .iter()
                    .copied()
                    .filter(|&p| p > EPS)
                    .collect();
                for &p in &support {
                    assert!(
                        (p - support[0]).abs() < 1e-6,
                        "child prior must be the un-noised uniform: {:?}",
                        &node.prior[i][..k]
                    );
                }
            }
        }
    }
}
