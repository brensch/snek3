//! Fixed-depth, full-width search over joint moves with a per-node Logit
//! Equilibrium backup. Built in two phases so leaf evaluation is a single
//! batched neural-net call:
//!
//! 1. [`Forest::build`] expands every root's tree to a fixed depth, recording
//!    the boards at non-terminal leaves that need a value estimate.
//! 2. The caller writes those leaf observations ([`Forest::write_observations`]),
//!    runs the network once, and hands back the values.
//! 3. [`Forest::backup`] propagates values up each tree, solving an equilibrium
//!    at every internal node, and returns the root equilibrium policies.

use crate::le;
use snek_core::{encode_into, Board, Move, MAX_SNAKES, NUM_CHANNELS};

/// Placeholder move for eliminated snakes (ignored by `step`).
const DUMMY_MOVE: Move = Move::Up;

enum NodeKind {
    /// Game already over here; `value` is exact.
    Terminal,
    /// Non-terminal leaf needing a network estimate.
    Eval { eval_id: usize },
    /// Interior node with per-agent candidate moves and child node ids in
    /// joint-action (row-major, agent 0 most significant) order.
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
    /// Root node id. Children are pushed before their parent, so the root is the
    /// last node expanded, not node 0.
    root: usize,
}

pub struct Forest {
    trees: Vec<Tree>,
    eval_boards: Vec<Board>,
    pub n_snakes: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

/// Candidate moves for one snake: drop strictly-dominated suicides (reversing
/// onto the neck, stepping off the board). A trapped snake keeps all moves (it
/// dies regardless). Eliminated snakes get a single dummy move.
fn candidates(board: &Board, i: usize) -> Vec<Move> {
    let s = &board.snakes[i];
    if !s.alive() {
        return vec![DUMMY_MOVE];
    }
    let head = s.head();
    let neck = if s.len() >= 2 { Some(s.body.get(1)) } else { None };
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

/// Exact per-agent value at a terminal board: winner +1, losers -1, draw 0.
fn terminal_values(board: &Board) -> [f32; MAX_SNAKES] {
    let mut v = [0.0f32; MAX_SNAKES];
    match board.winner() {
        Some(w) => {
            for i in 0..board.snakes.len() {
                v[i] = if i == w { 1.0 } else { -1.0 };
            }
        }
        None => { /* draw: all zero */ }
    }
    v
}

impl Forest {
    /// Build the search forest for `boards`, expanding each to `depth` plies.
    pub fn build(boards: &[Board], depth: u32) -> Forest {
        let n_snakes = boards.first().map(|b| b.snakes.len()).unwrap_or(0);
        let (height, width) = boards
            .first()
            .map(|b| (b.height as usize, b.width as usize))
            .unwrap_or((0, 0));

        let mut forest = Forest {
            trees: Vec::with_capacity(boards.len()),
            eval_boards: Vec::new(),
            n_snakes,
            channels: NUM_CHANNELS,
            height,
            width,
        };

        for board in boards {
            let mut nodes = Vec::new();
            let root = forest.expand(board.clone(), depth, &mut nodes);
            forest.trees.push(Tree { nodes, root });
        }
        forest
    }

    /// Recursively expand `board`, pushing nodes into `nodes` and returning the
    /// new node's id.
    fn expand(&mut self, board: Board, depth: u32, nodes: &mut Vec<Node>) -> usize {
        if board.is_terminal() {
            let id = nodes.len();
            nodes.push(Node {
                kind: NodeKind::Terminal,
                value: terminal_values(&board),
            });
            return id;
        }
        if depth == 0 {
            let eval_id = self.eval_boards.len();
            self.eval_boards.push(board);
            let id = nodes.len();
            nodes.push(Node {
                kind: NodeKind::Eval { eval_id },
                value: [0.0; MAX_SNAKES],
            });
            return id;
        }

        let n = board.snakes.len();
        let cands: Vec<Vec<Move>> = (0..n).map(|i| candidates(&board, i)).collect();
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
            let mut child_board = board.clone();
            child_board.step(&mv);
            let child_id = self.expand(child_board, depth - 1, nodes);
            children.push(child_id);
        }

        let id = nodes.len();
        nodes.push(Node {
            kind: NodeKind::Internal { cands, children },
            value: [0.0; MAX_SNAKES],
        });
        id
    }

    /// Number of `(leaf, agent)` value estimates the network must produce.
    pub fn eval_count(&self) -> usize {
        self.eval_boards.len() * self.n_snakes
    }

    /// Floats per single observation (`channels * height * width`).
    pub fn obs_size(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Write all leaf observations into `out`, which must be
    /// `eval_count() * obs_size()` long, laid out as `[eval_id, agent]`.
    pub fn write_observations(&self, out: &mut [f32]) {
        let obs = self.obs_size();
        debug_assert_eq!(out.len(), self.eval_count() * obs);
        for (eval_id, board) in self.eval_boards.iter().enumerate() {
            for agent in 0..self.n_snakes {
                let base = (eval_id * self.n_snakes + agent) * obs;
                encode_into(board, agent, &mut out[base..base + obs]);
            }
        }
    }

    /// Back up `values` (length `eval_count()`, indexed `[eval_id, agent]`) and
    /// return root equilibrium policies as a flat `[num_roots * n_snakes * 4]`
    /// array (probability per move; non-candidate moves are 0).
    pub fn backup(&mut self, values: &[f32], tau: f32, iters: usize) -> Vec<f32> {
        debug_assert_eq!(values.len(), self.eval_count());
        let n = self.n_snakes;
        let mut out = vec![0.0f32; self.trees.len() * n * 4];

        for t in 0..self.trees.len() {
            // Move the tree out so backup_node can borrow &self (eval_boards)
            // alongside &mut tree without aliasing the forest's trees vector.
            let mut tree =
                std::mem::replace(&mut self.trees[t], Tree { nodes: Vec::new(), root: 0 });
            let root = tree.root;
            let policies = self.backup_node(&mut tree, root, values, tau, iters);
            self.trees[t] = tree;

            // A single-snake (or already-terminal-root) game yields no policy;
            // leave it uniform-free (all zeros) and let the caller handle it.
            if let Some(policies) = policies {
                for (i, slots) in policies.iter().enumerate() {
                    let base = (t * n + i) * 4;
                    out[base..base + 4].copy_from_slice(slots);
                }
            }
        }
        out
    }

    /// Post-order backup returning the root node's per-agent policies (over its
    /// candidate moves) for the root only; interior values are stored on nodes.
    fn backup_node(
        &self,
        tree: &mut Tree,
        node_id: usize,
        values: &[f32],
        tau: f32,
        iters: usize,
    ) -> Option<Vec<Vec<f32>>> {
        // Resolve this node's value, recursing into children first.
        let (value, root_policies) = match &tree.nodes[node_id].kind {
            NodeKind::Terminal => (tree.nodes[node_id].value, None),
            NodeKind::Eval { eval_id } => {
                let eval_id = *eval_id;
                let board = &self.eval_boards[eval_id];
                let mut v = [0.0f32; MAX_SNAKES];
                for i in 0..self.n_snakes {
                    v[i] = if board.snakes[i].alive() {
                        values[eval_id * self.n_snakes + i]
                    } else {
                        -1.0
                    };
                }
                (v, None)
            }
            NodeKind::Internal { cands, children } => {
                let cands = cands.clone();
                let children = children.clone();
                for &c in &children {
                    self.backup_node(tree, c, values, tau, iters);
                }
                let cand_lens: Vec<usize> = cands.iter().map(|c| c.len()).collect();
                let payoffs: Vec<[f32; MAX_SNAKES]> =
                    children.iter().map(|&c| tree.nodes[c].value).collect();
                let sol = le::solve(&cand_lens, &payoffs, tau, iters);
                (sol.values, Some((cands, sol.policies)))
            }
        };
        tree.nodes[node_id].value = value;

        // Map this node's candidate-space policy onto the 4 move slots. Only the
        // root's return value is used by the caller; children's are discarded.
        root_policies.map(|(cands, policies)| {
            cands
                .iter()
                .zip(policies.iter())
                .map(|(cand_moves, pol)| {
                    let mut slots = vec![0.0f32; 4];
                    for (a, m) in cand_moves.iter().enumerate() {
                        slots[m.index()] = pol[a];
                    }
                    slots
                })
                .collect()
        })
    }
}
