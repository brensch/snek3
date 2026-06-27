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
use rayon::prelude::*;
use snek_core::{encode_into, obs_side, Board, Move, MAX_SNAKES, NUM_CHANNELS};

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
pub(crate) fn candidates(board: &Board, i: usize) -> Vec<Move> {
    let s = &board.snakes[i];
    if !s.alive() {
        return vec![DUMMY_MOVE];
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

/// Exact per-agent value at a terminal board: winner +1, losers -1, draw configurable.
pub(crate) fn terminal_values_with_draw(board: &Board, draw_value: f32) -> [f32; MAX_SNAKES] {
    let mut v = [0.0f32; MAX_SNAKES];
    match board.winner() {
        Some(w) => {
            for i in 0..board.snakes.len() {
                v[i] = if i == w { 1.0 } else { -1.0 };
            }
        }
        None => {
            for x in v.iter_mut().take(board.snakes.len()) {
                *x = draw_value;
            }
        }
    }
    v
}

/// Exact per-agent value at a terminal board: winner +1, losers -1, draw 0.
pub(crate) fn terminal_values(board: &Board) -> [f32; MAX_SNAKES] {
    terminal_values_with_draw(board, 0.0)
}

/// Recursively expand `board`, pushing nodes into `nodes` and returning the
/// new node's id. `eval_boards` is local to one root tree during parallel build;
/// eval ids are offset when the forest is assembled.
fn expand_node(
    board: Board,
    depth: u32,
    nodes: &mut Vec<Node>,
    eval_boards: &mut Vec<Board>,
) -> usize {
    if board.is_terminal() {
        let id = nodes.len();
        nodes.push(Node {
            kind: NodeKind::Terminal,
            value: terminal_values(&board),
        });
        return id;
    }
    if depth == 0 {
        let eval_id = eval_boards.len();
        eval_boards.push(board);
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
        let child_id = expand_node(child_board, depth - 1, nodes, eval_boards);
        children.push(child_id);
    }

    let id = nodes.len();
    nodes.push(Node {
        kind: NodeKind::Internal { cands, children },
        value: [0.0; MAX_SNAKES],
    });
    id
}

fn offset_eval_ids(tree: &mut Tree, offset: usize) {
    for node in &mut tree.nodes {
        if let NodeKind::Eval { eval_id } = &mut node.kind {
            *eval_id += offset;
        }
    }
}

/// Post-order backup returning the node's per-agent root policies (over its
/// candidate moves) only for the requested node; child policies are discarded.
fn backup_node(
    eval_boards: &[Board],
    n_snakes: usize,
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
            let board = &eval_boards[eval_id];
            let mut v = [0.0f32; MAX_SNAKES];
            for i in 0..n_snakes {
                v[i] = if board.snakes[i].alive() {
                    values[eval_id * n_snakes + i]
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
                backup_node(eval_boards, n_snakes, tree, c, values, tau, iters);
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

impl Forest {
    /// Build the search forest for `boards`, expanding each to `depth` plies.
    pub fn build(boards: &[Board], depth: u32) -> Forest {
        let n_snakes = boards.first().map(|b| b.snakes.len()).unwrap_or(0);
        // Observation canvas dims are egocentric (head-centred): 2*side-1.
        let (height, width) = boards
            .first()
            .map(|b| (obs_side(b.height as usize), obs_side(b.width as usize)))
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
            let mut eval_boards = Vec::new();
            let root = expand_node(board.clone(), depth, &mut nodes, &mut eval_boards);
            let mut tree = Tree { nodes, root };
            let offset = forest.eval_boards.len();
            offset_eval_ids(&mut tree, offset);
            forest.eval_boards.extend(eval_boards);
            forest.trees.push(tree);
        }
        forest
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
        let eval_chunk = self.n_snakes * obs;
        out.par_chunks_mut(eval_chunk)
            .zip(self.eval_boards.par_iter())
            .for_each(|(chunk, board)| {
                for agent in 0..self.n_snakes {
                    let base = agent * obs;
                    encode_into(board, agent, &mut chunk[base..base + obs]);
                }
            });
    }

    /// Back up `values` (length `eval_count()`, indexed `[eval_id, agent]`) and
    /// return `(policies, root_values)`:
    /// * `policies` — root equilibrium policies, flat `[num_roots * n_snakes * 4]`
    ///   (probability per move; non-candidate moves are 0).
    /// * `root_values` — per-agent equilibrium expected value at each root, flat
    ///   `[num_roots * n_snakes]`. This is the bootstrapped value the search
    ///   assigns to the current state, used as a TD target during training.
    pub fn backup(&mut self, values: &[f32], tau: f32, iters: usize) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(values.len(), self.eval_count());
        let n = self.n_snakes;
        let mut out = vec![0.0f32; self.trees.len() * n * 4];
        let mut root_vals = vec![0.0f32; self.trees.len() * n];

        self.trees
            .par_iter_mut()
            .zip(out.par_chunks_mut(n * 4))
            .zip(root_vals.par_chunks_mut(n))
            .for_each(|((tree, out_chunk), val_chunk)| {
                let root = tree.root;
                let policies = backup_node(&self.eval_boards, n, tree, root, values, tau, iters);
                // Root per-agent equilibrium value (set on the root node by backup_node).
                let rv = tree.nodes[root].value;
                val_chunk.copy_from_slice(&rv[..n]);

                // A single-snake (or already-terminal-root) game yields no policy;
                // leave it uniform-free (all zeros) and let the caller handle it.
                if let Some(policies) = policies {
                    for (i, slots) in policies.iter().enumerate() {
                        let base = i * 4;
                        out_chunk[base..base + 4].copy_from_slice(slots);
                    }
                }
            });
        (out, root_vals)
    }
}
