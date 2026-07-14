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
                // Post-order value fill; ids already ordered children-before-parent.
                let mut node_val = vec![[0.0f32; MAX_SNAKES]; tree.nodes.len()];
                let mut root_policy = [[0.0f32; 4]; MAX_SNAKES];

                for id in 0..tree.nodes.len() {
                    match &tree.nodes[id].kind {
                        NodeKind::Terminal => {
                            node_val[id] = tree.nodes[id].value;
                        }
                        NodeKind::Eval { eval_id } => {
                            let alive = self.eval_alive[*eval_id];
                            // A seat already dead at this leaf has lost — score it
                            // with the same zero-sum loser value as a terminal, so
                            // the value scale is consistent and un-saturated.
                            let dead = crate::search::loser_value(n);
                            let mut v = [0.0f32; MAX_SNAKES];
                            for i in 0..n {
                                v[i] = if alive & (1 << i) != 0 {
                                    values[eval_id * n + i]
                                } else {
                                    dead
                                };
                            }
                            node_val[id] = v;
                        }
                        NodeKind::Internal { cands, children } => {
                            let cand_lens: Vec<usize> = cands.iter().map(|c| c.len()).collect();
                            // Payoff matrix: each joint action's child value vector.
                            let payoffs: Vec<[f32; MAX_SNAKES]> =
                                children.iter().map(|&c| node_val[c]).collect();
                            let sol = le::solve(&cand_lens, &payoffs, tau, iters);
                            node_val[id] = sol.values;
                            if id == tree.root {
                                // Map each seat's equilibrium mix over its
                                // candidates back onto the 4 move slots.
                                for i in 0..n {
                                    for (ai, &mv) in cands[i].iter().enumerate() {
                                        root_policy[i][mv.index()] += sol.policies[i][ai];
                                    }
                                }
                            }
                        }
                    }
                }

                RootEq {
                    policy: root_policy,
                    value: node_val[tree.root],
                }
            })
            .collect()
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
