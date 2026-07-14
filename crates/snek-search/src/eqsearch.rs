//! Fixed-depth, full-width joint-move search with a per-node Logit-Equilibrium
//! backup — the Albatross-faithful "correct game mode" search for a
//! simultaneous-move, multi-player game.
//!
//! Two-phase so leaf evaluation is a single batched neural-net call:
//!   1. [`EqForest::build`] expands every root's tree to a fixed depth over the
//!      *joint* move space, recording the boards at non-terminal leaves.
//!   2. The caller encodes those leaf boards (once per seat, plus its own
//!      temperature conditioning) and runs the value net, handing back a flat
//!      `values` slice laid out `values[eval_id * n + seat]`.
//!   3. [`EqForest::backup`] propagates values up each tree, solving a logit
//!      equilibrium at every internal node (per-agent temperature `tau`), and
//!      returns each root's equilibrium **mixed** policy (per seat, over the 4
//!      moves) and per-player equilibrium value.
//!
//! The search is libtorch-free: the caller owns the net. This is what lets the
//! identical search drive both self-play (trainer) and serving.

use crate::le;
use crate::search::{le_candidates, terminal_values_with_draw};
use rayon::prelude::*;
use snek_core::{Board, Move, MAX_SNAKES};

const DUMMY_MOVE: Move = Move::Up;

enum NodeKind {
    /// Game already over here; `value` is exact terminal payoff.
    Terminal,
    /// Non-terminal leaf needing a network value estimate.
    Eval { eval_id: usize },
    /// Interior node: per-agent candidate moves and child ids in joint-action
    /// order (row-major, agent 0 most significant).
    Internal {
        cands: Vec<Vec<Move>>,
        children: Vec<usize>,
    },
}

struct Node {
    kind: NodeKind,
    value: [f32; MAX_SNAKES],
}

struct Tree {
    nodes: Vec<Node>,
    /// Root id. Children are pushed before their parent, so ids are in
    /// post-order (a node's children have strictly smaller ids) and the root is
    /// the last node.
    root: usize,
}

/// A built forest awaiting leaf evaluation.
pub struct EqForest {
    trees: Vec<Tree>,
    eval_boards: Vec<Board>,
    /// Alive bitmask (bit i = seat i alive) for each eval board, so `backup` can
    /// force a dead seat's leaf value to -1 regardless of what the net returned.
    eval_alive: Vec<u32>,
    /// Game (root) index each eval board belongs to, so the caller can pick that
    /// game's per-episode τ when encoding the leaf (τ is per-game, constant for
    /// the game's life — games persist across training generations here).
    eval_game: Vec<u32>,
    n: usize,
}

fn alive_mask(board: &Board) -> u32 {
    let mut m = 0u32;
    for (i, s) in board.snakes.iter().enumerate() {
        if s.alive() {
            m |= 1 << i;
        }
    }
    m
}

/// Per-seat value of an eval leaf: the net value where alive, the zero-sum loser
/// value where already dead (kept consistent with terminal payoffs).
fn eval_leaf_value(alive: u32, eval_id: usize, values: &[f32], n: usize) -> [f32; MAX_SNAKES] {
    let dead = crate::search::loser_value(n);
    let mut v = [0.0f32; MAX_SNAKES];
    for (i, vi) in v.iter_mut().enumerate().take(n) {
        *vi = if alive & (1 << i) != 0 {
            values[eval_id * n + i]
        } else {
            dead
        };
    }
    v
}

/// Recursively compute a node's per-seat equilibrium value (post-order, memoized).
/// Recursion — not the old id-order loop — because selective deepening adds a
/// node's children *after* it, so ids are no longer children-before-parent.
/// Trees are depth ≤ 2 so the recursion is shallow.
#[allow(clippy::too_many_arguments)]
fn node_value(
    tree: &Tree,
    id: usize,
    eval_alive: &[u32],
    values: &[f32],
    n: usize,
    tau: &[f32],
    iters: usize,
    memo: &mut [Option<[f32; MAX_SNAKES]>],
) -> [f32; MAX_SNAKES] {
    if let Some(v) = memo[id] {
        return v;
    }
    let v = match &tree.nodes[id].kind {
        NodeKind::Terminal => tree.nodes[id].value,
        NodeKind::Eval { eval_id } => eval_leaf_value(eval_alive[*eval_id], *eval_id, values, n),
        NodeKind::Internal { cands, children } => {
            let cand_lens: Vec<usize> = cands.iter().map(|c| c.len()).collect();
            let payoffs: Vec<[f32; MAX_SNAKES]> = children
                .iter()
                .map(|&c| node_value(tree, c, eval_alive, values, n, tau, iters, memo))
                .collect();
            le::solve(&cand_lens, &payoffs, tau, iters).values
        }
    };
    memo[id] = Some(v);
    v
}

/// Column strides for decoding a row-major joint index (agent 0 most significant).
fn joint_strides(cand_lens: &[usize]) -> Vec<usize> {
    let n = cand_lens.len();
    let mut stride = vec![1usize; n];
    for i in (0..n).rev() {
        stride[i] = if i + 1 < n {
            stride[i + 1] * cand_lens[i + 1]
        } else {
            1
        };
    }
    stride
}

fn expand_node(
    board: Board,
    depth: u32,
    draw_value: f32,
    nodes: &mut Vec<Node>,
    eval_boards: &mut Vec<Board>,
    eval_alive: &mut Vec<u32>,
) -> usize {
    if board.is_terminal() {
        let id = nodes.len();
        nodes.push(Node {
            kind: NodeKind::Terminal,
            value: terminal_values_with_draw(&board, draw_value),
        });
        return id;
    }
    if depth == 0 {
        let eval_id = eval_boards.len();
        eval_alive.push(alive_mask(&board));
        eval_boards.push(board);
        let id = nodes.len();
        nodes.push(Node {
            kind: NodeKind::Eval { eval_id },
            value: [0.0; MAX_SNAKES],
        });
        return id;
    }

    let n = board.snakes.len();
    let cands: Vec<Vec<Move>> = (0..n).map(|i| le_candidates(&board, i)).collect();
    let total: usize = cands.iter().map(|c| c.len()).product();

    let mut strides = vec![1usize; n];
    for i in (0..n).rev() {
        strides[i] = if i + 1 < n {
            strides[i + 1] * cands[i + 1].len()
        } else {
            1
        };
    }

    let mut children = Vec::with_capacity(total);
    let mut mv = vec![DUMMY_MOVE; n];
    for joint in 0..total {
        for i in 0..n {
            let ai = (joint / strides[i]) % cands[i].len();
            mv[i] = cands[i][ai];
        }
        let mut child = board.clone();
        child.step(&mv);
        let cid = expand_node(child, depth - 1, draw_value, nodes, eval_boards, eval_alive);
        children.push(cid);
    }

    let id = nodes.len();
    nodes.push(Node {
        kind: NodeKind::Internal { cands, children },
        value: [0.0; MAX_SNAKES],
    });
    id
}

/// One root's equilibrium readout.
#[derive(Clone, Debug)]
pub struct RootEq {
    /// Per-seat mixed strategy over the 4 moves (`Move::ALL` order), with
    /// non-candidate/illegal moves at 0. Dead seats are all-zero.
    pub policy: [[f32; 4]; MAX_SNAKES],
    /// Per-seat equilibrium expected value at the root (the bootstrapped value
    /// target).
    pub value: [f32; MAX_SNAKES],
}

impl EqForest {
    /// Build the forest by expanding each root to `depth` plies over joint moves.
    /// `depth` must be >= 1. Terminal leaves use `draw_value` for draws.
    pub fn build(roots: &[Board], depth: u32, draw_value: f32) -> Self {
        debug_assert!(depth >= 1, "equilibrium search needs depth >= 1");
        let n = roots.first().map(|b| b.snakes.len()).unwrap_or(0);

        // Build trees in parallel, each with local eval buffers, then assemble
        // and offset the eval ids into the shared eval list.
        let built: Vec<(Tree, Vec<Board>, Vec<u32>)> = roots
            .par_iter()
            .map(|b| {
                let mut nodes = Vec::new();
                let mut eb = Vec::new();
                let mut ea = Vec::new();
                let root = expand_node(b.clone(), depth, draw_value, &mut nodes, &mut eb, &mut ea);
                (Tree { nodes, root }, eb, ea)
            })
            .collect();

        let mut trees = Vec::with_capacity(built.len());
        let mut eval_boards = Vec::new();
        let mut eval_alive = Vec::new();
        let mut eval_game = Vec::new();
        for (g, (mut tree, eb, ea)) in built.into_iter().enumerate() {
            let offset = eval_boards.len();
            for node in &mut tree.nodes {
                if let NodeKind::Eval { eval_id } = &mut node.kind {
                    *eval_id += offset;
                }
            }
            for _ in 0..eb.len() {
                eval_game.push(g as u32);
            }
            eval_boards.extend(eb);
            eval_alive.extend(ea);
            trees.push(tree);
        }

        EqForest {
            trees,
            eval_boards,
            eval_alive,
            eval_game,
            n,
        }
    }

    /// Boards at non-terminal leaves needing a value estimate. The caller must
    /// produce `values` of length `eval_boards().len() * n_snakes()`, laid out
    /// `values[eval_id * n + seat]`.
    pub fn eval_boards(&self) -> &[Board] {
        &self.eval_boards
    }

    pub fn n_snakes(&self) -> usize {
        self.n
    }

    /// Game (root) index each eval board belongs to (parallel to `eval_boards`),
    /// so the caller can pick that game's τ when encoding the leaf.
    pub fn eval_game(&self) -> &[u32] {
        &self.eval_game
    }

    /// Number of roots (parallel games) in the forest.
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// Propagate leaf values up, solving a logit equilibrium at each internal
    /// node with `iters` SFP iterations. `tau_per_game[g]` is game g's per-agent
    /// inverse temperature (first `n_snakes` entries used); heterogeneous entries
    /// give the SBRLE exploit path. Returns one [`RootEq`] per root.
    pub fn backup(
        &self,
        values: &[f32],
        tau_per_game: &[[f32; MAX_SNAKES]],
        iters: usize,
    ) -> Vec<RootEq> {
        let n = self.n;
        debug_assert_eq!(tau_per_game.len(), self.trees.len());

        self.trees
            .par_iter()
            .enumerate()
            .map(|(g, tree)| {
                let tau = &tau_per_game[g][..n];
                let mut memo = vec![None; tree.nodes.len()];
                match &tree.nodes[tree.root].kind {
                    NodeKind::Internal { cands, children } => {
                        let cand_lens: Vec<usize> = cands.iter().map(|c| c.len()).collect();
                        let payoffs: Vec<[f32; MAX_SNAKES]> = children
                            .iter()
                            .map(|&c| {
                                node_value(
                                    tree,
                                    c,
                                    &self.eval_alive,
                                    values,
                                    n,
                                    tau,
                                    iters,
                                    &mut memo,
                                )
                            })
                            .collect();
                        let sol = le::solve(&cand_lens, &payoffs, tau, iters);
                        let mut root_policy = [[0.0f32; 4]; MAX_SNAKES];
                        for i in 0..n {
                            for (ai, &mv) in cands[i].iter().enumerate() {
                                root_policy[i][mv.index()] += sol.policies[i][ai];
                            }
                        }
                        RootEq {
                            policy: root_policy,
                            value: sol.values,
                        }
                    }
                    // A root that is already terminal (or, defensively, an
                    // un-expanded leaf) has no equilibrium to solve.
                    NodeKind::Terminal => RootEq {
                        policy: [[0.0f32; 4]; MAX_SNAKES],
                        value: tree.nodes[tree.root].value,
                    },
                    NodeKind::Eval { eval_id } => RootEq {
                        policy: [[0.0f32; 4]; MAX_SNAKES],
                        value: eval_leaf_value(self.eval_alive[*eval_id], *eval_id, values, n),
                    },
                }
            })
            .collect()
    }

    /// Selective deepening (progressive depth-2). Given depth-1 leaf `values`,
    /// solve each root's LE, and expand ONE more ply only the `top_k` joint
    /// successors with the highest equilibrium reach probability. Returns the
    /// index into `eval_boards()` at which the new depth-2 leaves begin — the
    /// caller value-nets `eval_boards()[start..]`, appends those to `values`, and
    /// calls [`backup`] again with the combined slice. `top_k == 0` is a no-op
    /// (stays fixed depth-1). See docs/le-selective-depth.md.
    pub fn deepen_topk(
        &mut self,
        values: &[f32],
        tau_per_game: &[[f32; MAX_SNAKES]],
        iters: usize,
        top_k: usize,
        draw_value: f32,
    ) -> usize {
        let start = self.eval_boards.len();
        if top_k == 0 {
            return start;
        }
        let n = self.n;
        // Phase A (parallel per tree): solve the root LE, pick the top_k
        // deepenable joint successors by reach probability, expand each one ply
        // into tree-local buffers. New Eval nodes hold a LOCAL eval id, fixed up
        // to a global id in phase B.
        let EqForest {
            trees,
            eval_boards,
            eval_alive,
            eval_game,
            n: _,
        } = self;
        let ealive: &[u32] = eval_alive;
        let eboards: &[Board] = eval_boards;
        let per_tree: Vec<(Vec<Board>, Vec<u32>, Vec<usize>)> = trees
            .par_iter_mut()
            .enumerate()
            .map(|(g, tree)| {
                let tau = &tau_per_game[g][..n];
                let mut local_boards: Vec<Board> = Vec::new();
                let mut local_alive: Vec<u32> = Vec::new();
                let mut new_eval_nodes: Vec<usize> = Vec::new();
                let (cands, children) = match &tree.nodes[tree.root].kind {
                    NodeKind::Internal { cands, children } => (cands.clone(), children.clone()),
                    _ => return (local_boards, local_alive, new_eval_nodes),
                };
                let cand_lens: Vec<usize> = cands.iter().map(|c| c.len()).collect();
                let mut memo = vec![None; tree.nodes.len()];
                let payoffs: Vec<[f32; MAX_SNAKES]> = children
                    .iter()
                    .map(|&c| node_value(tree, c, ealive, values, n, tau, iters, &mut memo))
                    .collect();
                let sol = le::solve(&cand_lens, &payoffs, tau, iters);
                let stride = joint_strides(&cand_lens);
                // Reach probability of each joint successor, keeping only the
                // deepenable (Eval, i.e. non-terminal) ones.
                let mut reach: Vec<(f32, usize)> = children
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| matches!(tree.nodes[c].kind, NodeKind::Eval { .. }))
                    .map(|(j, _)| {
                        let mut p = 1.0f32;
                        for i in 0..n {
                            let ai = (j / stride[i]) % cand_lens[i];
                            p *= sol.policies[i][ai];
                        }
                        (p, j)
                    })
                    .collect();
                reach.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                reach.truncate(top_k);
                for (_, j) in reach {
                    let child_id = children[j];
                    let eid = match tree.nodes[child_id].kind {
                        NodeKind::Eval { eval_id } => eval_id,
                        _ => continue,
                    };
                    let board = eboards[eid].clone();
                    let sc: Vec<Vec<Move>> = (0..n).map(|i| le_candidates(&board, i)).collect();
                    let stotal: usize = sc.iter().map(|c| c.len()).product();
                    let sstride = joint_strides(&sc.iter().map(|c| c.len()).collect::<Vec<_>>());
                    let mut sub_children = Vec::with_capacity(stotal);
                    let mut mv = vec![DUMMY_MOVE; n];
                    for sj in 0..stotal {
                        for i in 0..n {
                            let ai = (sj / sstride[i]) % sc[i].len();
                            mv[i] = sc[i][ai];
                        }
                        let mut c = board.clone();
                        c.step(&mv);
                        let cid = tree.nodes.len();
                        if c.is_terminal() {
                            tree.nodes.push(Node {
                                kind: NodeKind::Terminal,
                                value: terminal_values_with_draw(&c, draw_value),
                            });
                        } else {
                            let local_eid = local_boards.len();
                            local_alive.push(alive_mask(&c));
                            local_boards.push(c);
                            tree.nodes.push(Node {
                                kind: NodeKind::Eval { eval_id: local_eid },
                                value: [0.0; MAX_SNAKES],
                            });
                            new_eval_nodes.push(cid);
                        }
                        sub_children.push(cid);
                    }
                    tree.nodes[child_id].kind = NodeKind::Internal {
                        cands: sc,
                        children: sub_children,
                    };
                }
                (local_boards, local_alive, new_eval_nodes)
            })
            .collect();
        // Phase B (sequential): splice tree-local eval buffers into the global
        // list, rewriting each new Eval node's local id to its global id.
        let mut offset = eval_boards.len();
        for (g, (lb, la, node_ids)) in per_tree.into_iter().enumerate() {
            for nid in node_ids {
                if let NodeKind::Eval { eval_id } = &mut trees[g].nodes[nid].kind {
                    *eval_id += offset;
                }
            }
            for _ in 0..lb.len() {
                eval_game.push(g as u32);
            }
            offset += lb.len();
            eval_boards.extend(lb);
            eval_alive.extend(la);
        }
        start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snek_core::{standard_start, Board};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand::SeedableRng;

    fn fresh(n: usize) -> Board {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        standard_start(11, 11, n, &mut rng)
    }

    #[test]
    fn selective_depth2_expands_and_backs_up() {
        let boards = vec![fresh(4)];
        let mut forest = EqForest::build(&boards, 1, -0.25);
        let n = forest.n_snakes();
        let m1 = forest.eval_boards().len();
        assert!(m1 > 0);
        let vals1 = vec![0.0f32; m1 * n];
        let tau = vec![[6.0f32; MAX_SNAKES]; forest.len()];
        // Deepen the top-3 joint successors one more ply.
        let start = forest.deepen_topk(&vals1, &tau, 60, 3, -0.25);
        assert_eq!(start, m1, "new leaves begin right after the depth-1 leaves");
        let m2 = forest.eval_boards().len();
        assert!(m2 > m1, "selective deepening added depth-2 leaves");
        let mut allvals = vals1.clone();
        allvals.extend(vec![0.0f32; (m2 - m1) * n]);
        let out = forest.backup(&allvals, &tau, 60);
        assert_eq!(out.len(), 1);
        for seat in 0..n {
            let sum: f32 = out[0].policy[seat].iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "seat {seat} policy still normalized");
        }
    }

    #[test]
    fn top_k_zero_matches_plain_depth1() {
        let boards = vec![fresh(4)];
        let mut forest = EqForest::build(&boards, 1, -0.25);
        let n = forest.n_snakes();
        let m1 = forest.eval_boards().len();
        let vals = vec![0.1f32; m1 * n];
        let tau = vec![[5.0f32; MAX_SNAKES]; forest.len()];
        let base = forest.backup(&vals, &tau, 80);
        let start = forest.deepen_topk(&vals, &tau, 80, 0, -0.25);
        assert_eq!(start, m1);
        assert_eq!(forest.eval_boards().len(), m1, "top_k=0 adds no leaves");
        let after = forest.backup(&vals, &tau, 80);
        for seat in 0..n {
            for m in 0..4 {
                assert!(
                    (base[0].policy[seat][m] - after[0].policy[seat][m]).abs() < 1e-4,
                    "top_k=0 is a no-op"
                );
            }
        }
    }

    #[test]
    fn deepening_shifts_policy_when_depth2_disagrees() {
        // Depth-1 sees flat leaves (uniform-ish policy). Then we deepen and feed
        // the depth-2 leaves a strong per-seat signal; the re-solve must move the
        // policy away from the flat depth-1 answer — i.e. depth-2 info propagates.
        let boards = vec![fresh(4)];
        let mut forest = EqForest::build(&boards, 1, -0.25);
        let n = forest.n_snakes();
        let m1 = forest.eval_boards().len();
        let vals1 = vec![0.0f32; m1 * n];
        let tau = vec![[8.0f32; MAX_SNAKES]; forest.len()];
        let flat = forest.backup(&vals1, &tau, 120);
        // Rebuild fresh for the deepened path (backup above didn't mutate).
        let start = forest.deepen_topk(&vals1, &tau, 120, 6, -0.25);
        let m2 = forest.eval_boards().len();
        assert!(m2 > m1);
        let mut allvals = vals1.clone();
        // Depth-2 leaves: seat 0 great (+1), everyone else bad (-1).
        for _ in m1..m2 {
            allvals.push(1.0);
            for _ in 1..n {
                allvals.push(-1.0);
            }
        }
        let deep = forest.backup(&allvals, &tau, 120);
        let moved: f32 = (0..4)
            .map(|m| (deep[0].policy[0][m] - flat[0].policy[0][m]).abs())
            .sum();
        assert!(moved > 1e-3, "depth-2 values changed seat 0's policy ({moved})");
    }

    #[test]
    fn depth1_produces_mixed_legal_policy_and_bounded_values() {
        let boards = vec![fresh(4)];
        let forest = EqForest::build(&boards, 1, -0.25);
        let n = forest.n_snakes();
        assert_eq!(n, 4);
        // Uniform "net says 0 everywhere" evaluation.
        let vals = vec![0.0f32; forest.eval_boards().len() * n];
        let tau = vec![[6.0f32; MAX_SNAKES]; forest.len()];
        let out = forest.backup(&vals, &tau, 120);
        assert_eq!(out.len(), 1);
        let eq = &out[0];
        for seat in 0..n {
            let sum: f32 = eq.policy[seat].iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "seat {seat} policy sums to 1");
            // With a flat leaf value, the equilibrium should be ~uniform over the
            // legal candidates (i.e. not a collapsed one-hot).
            let maxp = eq.policy[seat].iter().cloned().fold(0.0f32, f32::max);
            assert!(maxp < 0.99, "seat {seat} policy is mixed, not argmax");
            assert!(eq.value[seat].abs() <= 1.0 + 1e-3, "value in range");
        }
    }

    #[test]
    fn eval_layout_size_matches() {
        // All roots in a forest must share a snake count (single eval stride).
        let boards = vec![fresh(4), fresh(4)];
        let forest = EqForest::build(&boards, 1, 0.0);
        let need = forest.eval_boards().len() * forest.n_snakes();
        let vals = vec![0.1f32; need];
        let tau = vec![[4.0f32; MAX_SNAKES]; forest.len()];
        let out = forest.backup(&vals, &tau, 60);
        assert_eq!(out.len(), 2);
    }
}
