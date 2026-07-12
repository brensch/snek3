//! Cheap, fixed-strength baseline agents: decoupled-UCT MCTS over the exact
//! `snek-core` rules, with leaf positions scored by a heuristic instead of a
//! neural net.
//!
//! They exist to anchor the evaluation league: nets drift as training runs,
//! but these players never learn, so their fitted Elo is a stable reference
//! point ("is the newest checkpoint actually better, or did the pool just
//! shift?").
//!
//! Two agents share one search engine and differ only in evaluation and move
//! pruning:
//!
//! - [`Baseline::Floodfill`] — the original: static Voronoi area control +
//!   health + relative length.
//! - [`Baseline::Voronoi`] — the strongest heuristic we know how to build
//!   (see `voronoi.rs`): time-expanded Voronoi ownership with tail decay and
//!   head-to-head dominance, starvation awareness, trap detection, and
//!   guaranteed-loss move pruning.
//!
//! The search mirrors the shape of the real serving search (per-snake
//! decoupled selection over joint moves, deterministic argmax readout, the
//! same food-free `Board::step` lookahead) but is orders of magnitude cheaper:
//! a leaf evaluation is a few flood fills over ≤`board²` cells, no tensors.

use std::collections::HashMap;
use std::time::Instant;

use snek_core::{Board, Move, MAX_SNAKES};

mod voronoi;

/// A built-in heuristic agent an arena `--nets` entry can name instead of a
/// weights file. Parsing is the single registry of accepted spec tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Baseline {
    /// Flood-fill MCTS: static Voronoi area + health + length leaf eval.
    Floodfill,
    /// Time-expanded Voronoi MCTS with dominance, hunger and trap terms.
    Voronoi,
}

impl Baseline {
    pub const ALL: [Baseline; 2] = [Baseline::Floodfill, Baseline::Voronoi];

    /// The agent an arena model spec names, if any ("heuristic"/"floodfill"
    /// for the original, "voronoi" for the stronger agent).
    pub fn parse(spec: &str) -> Option<Baseline> {
        let s = spec.trim();
        if s.eq_ignore_ascii_case("heuristic") || s.eq_ignore_ascii_case("floodfill") {
            Some(Baseline::Floodfill)
        } else if s.eq_ignore_ascii_case("voronoi") {
            Some(Baseline::Voronoi)
        } else {
            None
        }
    }

    /// Canonical `--nets` token.
    pub fn token(self) -> &'static str {
        match self {
            Baseline::Floodfill => "heuristic",
            Baseline::Voronoi => "voronoi",
        }
    }

    /// Display name for scoreboards.
    pub fn display_name(self) -> &'static str {
        match self {
            Baseline::Floodfill => "floodfill",
            Baseline::Voronoi => "voronoi",
        }
    }

    /// How the shared engine is specialized for this agent.
    fn engine(self) -> Engine {
        match self {
            Baseline::Floodfill => Engine {
                eval: evaluate,
                prune_headon: false,
            },
            Baseline::Voronoi => Engine {
                eval: voronoi::evaluate,
                prune_headon: true,
            },
        }
    }
}

/// True if an arena model spec names a built-in heuristic agent instead of a
/// weights file.
pub fn is_heuristic_spec(spec: &str) -> bool {
    Baseline::parse(spec).is_some()
}

/// A leaf evaluation: per-snake values in [-1, 1].
type EvalFn = fn(&Board, f32) -> [f32; MAX_SNAKES];

/// The two knobs that differentiate the baseline agents on the shared search.
struct Engine {
    eval: EvalFn,
    /// Prune candidate moves that step into a square a longer (or equal)
    /// opponent head can also reach — a lost/mutual head-to-head.
    prune_headon: bool,
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
    fn new(board: Board, prune_headon: bool) -> Self {
        let terminal = board.is_terminal();
        let cands: Vec<Vec<Move>> = (0..board.snakes.len())
            .map(|i| {
                if prune_headon {
                    voronoi::safe_candidates(&board, i)
                } else {
                    candidates(&board, i)
                }
            })
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
/// single dummy. (Same rule as `snek-search`.) Public so the arena can pick a
/// sane fallback move when an external HTTP player fails to answer.
pub fn candidates(board: &Board, i: usize) -> Vec<Move> {
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

/// One move for snake `me` from a real MCTS search of `sims` simulations — the
/// same decoupled-UCT engine the arena/serving baseline uses, so it is a
/// genuinely strong space-control opponent, not the 1-ply [`greedy_move`].
///
/// Deterministic (the sim budget is the only stopping condition — no wall clock),
/// which keeps self-play reproducible. `sims == 0` falls back to [`greedy_move`].
/// Costs ~`sims / 100k` seconds of one CPU core; run it while the GPU is busy on
/// the net's forwards and a budget comparable to the net's own sim count is
/// nearly free.
pub fn strong_move(kind: Baseline, board: &Board, me: usize, sims: usize, draw_value: f32) -> Move {
    if sims == 0 {
        return greedy_move(kind, board, me, draw_value);
    }
    let cfg = HeuristicConfig {
        max_sims: sims,
        draw_value,
        ..HeuristicConfig::default()
    };
    // A far deadline so the sim count is the sole limit (deterministic play).
    let deadline = Instant::now() + std::time::Duration::from_secs(3600);
    let dec = baseline_move_until(kind, &cfg, board, me, deadline);
    Move::from_index(dec.move_index)
}

/// One move for snake `me` by a cheap, deterministic 1-ply greedy over the
/// heuristic's leaf evaluation: for each of `me`'s safe candidates, step the
/// board once (me takes the candidate; every other snake takes its own first
/// safe candidate) and keep the move that maximises `me`'s evaluated value.
///
/// No tree and no rollouts — a handful of flood-fills per call — so it is cheap
/// enough to evaluate at every node of the net's search as an in-tree opponent
/// model, while still playing coherent space-control. Deterministic: fixed
/// candidate order, ties break to the lowest move index. Uses the food-free
/// [`Board::step`] the search uses, so it matches the lookahead exactly.
pub fn greedy_move(kind: Baseline, board: &Board, me: usize, draw_value: f32) -> Move {
    let engine = kind.engine();
    if board.is_terminal() || me >= board.snakes.len() || !board.snakes[me].alive() {
        return Move::Up;
    }
    let cands_for = |i: usize| -> Vec<Move> {
        let c = if engine.prune_headon {
            voronoi::safe_candidates(board, i)
        } else {
            candidates(board, i)
        };
        if c.is_empty() {
            candidates(board, i)
        } else {
            c
        }
    };
    let my_cands = cands_for(me);
    if my_cands.len() == 1 {
        return my_cands[0];
    }
    let n = board.snakes.len();
    // Each opponent's assumed reply: its own first safe candidate. Cheap and
    // fixed, so the greedy is a genuine (if shallow) space-control choice
    // rather than an assumption that opponents stand still.
    let mut opp_move = [Move::Up; MAX_SNAKES];
    for (j, slot) in opp_move.iter_mut().enumerate().take(n) {
        if j != me && board.snakes[j].alive() {
            *slot = cands_for(j)[0];
        }
    }
    let mut best = my_cands[0];
    let mut best_val = f32::NEG_INFINITY;
    for &m in &my_cands {
        let mut nb = board.clone();
        let mut moves = opp_move;
        moves[me] = m;
        nb.step(&moves[..n]);
        let v = (engine.eval)(&nb, draw_value)[me];
        if v > best_val {
            best_val = v;
            best = m;
        }
    }
    best
}

/// Terminal scoring shared by every leaf eval: winner +1, losers −1, draw
/// `draw_value` (matches the net search's convention). None for live boards.
pub(crate) fn terminal_values(board: &Board, draw_value: f32) -> Option<[f32; MAX_SNAKES]> {
    if !board.is_terminal() {
        return None;
    }
    let mut vals = [-1.0f32; MAX_SNAKES];
    match board.winner() {
        Some(w) => vals[w] = 1.0,
        None => {
            for v in vals.iter_mut().take(board.snakes.len()) {
                *v = draw_value;
            }
        }
    }
    Some(vals)
}

/// Per-snake leaf value in [-1, 1] for the flood-fill agent: live boards blend
/// Voronoi area control, health, and length relative to the longest opponent.
fn evaluate(board: &Board, draw_value: f32) -> [f32; MAX_SNAKES] {
    if let Some(vals) = terminal_values(board, draw_value) {
        return vals;
    }
    let mut vals = [-1.0f32; MAX_SNAKES];

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

/// One move for snake `me` from the flood-fill agent (kept as the historic
/// entry point; equivalent to `baseline_move_until(Baseline::Floodfill, …)`).
pub fn heuristic_move_until(
    cfg: &HeuristicConfig,
    board: &Board,
    me: usize,
    deadline: Instant,
) -> HeuristicDecision {
    baseline_move_until(Baseline::Floodfill, cfg, board, me, deadline)
}

/// One move for snake `me`: run decoupled-UCT until `cfg.max_sims` or the
/// deadline, then play the most-visited root candidate. Deterministic for a
/// fixed sim budget (no randomness anywhere in the search).
pub fn baseline_move_until(
    kind: Baseline,
    cfg: &HeuristicConfig,
    board: &Board,
    me: usize,
    deadline: Instant,
) -> HeuristicDecision {
    let engine = kind.engine();
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
    let root_cands = if engine.prune_headon {
        voronoi::safe_candidates(board, me)
    } else {
        candidates(board, me)
    };
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

    let mut nodes = vec![Node::new(board.clone(), engine.prune_headon)];
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
                break (engine.eval)(&nodes[cur].board, cfg.draw_value);
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
                    let vals = (engine.eval)(&child_board, cfg.draw_value);
                    let id = nodes.len();
                    nodes.push(Node::new(child_board, engine.prune_headon));
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
