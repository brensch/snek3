//! Self-play generation: a from-scratch, railed decoupled-PUCT engine.
//!
//! The only shared resource is the GPU, so it is guarded by a `Mutex` — the lock
//! handoff *is* the inference queue: the instant one worker unlocks, another that
//! has finished its CPU phase is already blocked on it and takes over, so the GPU
//! never idles. Each of `num_chunks` worker threads owns a fixed set of
//! `gpu_batch_games` games plus preallocated, reused, POD search trees; every
//! forward is the identical padded batch shape (cuDNN autotunes exactly once).
//!
//! Nodes are fixed-size POD (candidates <= 4, snakes <= MAX_SNAKES), so a game's
//! tree is a flat `Vec<Node>` reused turn after turn with no hot-path heap
//! traffic and no per-turn barrier.
//!
//! Per turn each game records a training sample (root board obs, root visit
//! policy, root value, alive mask); on completion the trajectory is materialised
//! into `(obs, pol, z, turn)` (bootstrap value or terminal value), and the first
//! `sample_games` finished games are also kept as full `GameJson` replays.

use crate::config::RunConfig;
use crate::metrics::{Counters, Metrics};
use crate::replay::Samples;
use crate::sample::{frame_from_board, FrameJson, GameJson};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use snek_core::{encode_into, obs_side, standard_start, Board, Move, MAX_SNAKES, NUM_CHANNELS};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tch::{no_grad, Device, Kind, Tensor};

const MAXC: usize = 4; // max candidate moves per snake
const EPS: f32 = 1e-8;

/// Safety cap on frames buffered per recorded game, bounding memory if a game
/// runs pathologically long (e.g. `max_turns == 0` with near-perfect play).
const MAX_REC_FRAMES: usize = 4096;

pub struct SelfPlayNet<'a> {
    pub net: &'a snek_tch::AZNet,
    pub device: Device,
}

// ---------------------------------------------------------------------------
// Board helpers (formerly in snek-search; inlined so self-play owns its search).
// ---------------------------------------------------------------------------

/// Legal candidate move indices (0..4) for snake `i`, plus their count. Drops
/// reversing-onto-neck and off-board moves; a trapped snake keeps all four.
#[inline]
fn candidates(board: &Board, i: usize) -> ([u8; MAXC], usize) {
    let mut out = [0u8; MAXC];
    let s = &board.snakes[i];
    if !s.alive() {
        return (out, 1); // dummy move (Up); ignored by step
    }
    let head = s.head();
    let neck = if s.len() >= 2 { Some(s.body.get(1)) } else { None };
    let mut k = 0usize;
    for m in Move::ALL {
        let nh = m.apply(head);
        if Some(nh) == neck || !board.in_bounds(nh) {
            continue;
        }
        out[k] = m.index() as u8;
        k += 1;
    }
    if k == 0 {
        for (idx, m) in Move::ALL.iter().enumerate() {
            out[idx] = m.index() as u8;
        }
        k = 4;
    }
    (out, k)
}

#[inline]
fn terminal_values(board: &Board, draw: f32) -> [f32; MAX_SNAKES] {
    let mut v = [0.0f32; MAX_SNAKES];
    match board.winner() {
        Some(w) => {
            for (i, x) in v.iter_mut().enumerate().take(board.snakes.len()) {
                *x = if i == w { 1.0 } else { -1.0 };
            }
        }
        None => {
            for x in v.iter_mut().take(board.snakes.len()) {
                *x = draw;
            }
        }
    }
    v
}

/// Would `mv` walk snake `snake_idx` straight into a wall, its own body, or a
/// non-tail segment of another snake? Used to mask hopeless priors/plays.
fn obvious_immediate_death(board: &Board, snake_idx: usize, mv: Move) -> bool {
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

/// Renormalise `probs` (4 moves) onto the non-obviously-fatal moves.
fn mask_obvious_immediate_deaths(board: &Board, snake_idx: usize, probs: &[f32]) -> [f32; 4] {
    let mut original = [0.0f32; 4];
    let mut total = 0.0f32;
    for i in 0..4 {
        original[i] = probs.get(i).copied().unwrap_or(0.0).max(0.0);
        total += original[i];
    }
    if total <= EPS
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
        if !obvious_immediate_death(board, snake_idx, Move::from_index(i)) {
            safe_count += 1;
            out[i] = original[i];
            safe_mass += original[i];
        }
    }
    if safe_count == 0 {
        return original;
    }
    if safe_mass > EPS {
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

/// Training value for one sample: winner +1, losers -1, draw configurable.
#[inline]
fn terminal_value(winner: Option<usize>, snake: usize, alive_final: bool, draw_value: f32) -> f32 {
    match winner {
        Some(w) if w == snake => 1.0,
        Some(_) => -1.0,
        None if alive_final => draw_value,
        None => -1.0,
    }
}

// ---------------------------------------------------------------------------
// POD MCTS node + reusable per-game tree arena.
// ---------------------------------------------------------------------------

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

struct Tree {
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
    fn new(n: usize, w: i8, h: i8, c_puct: f32, draw: f32, cap: usize) -> Self {
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

    fn reset(&mut self, board: &Board) {
        self.nodes[0].board.clone_from(board);
        self.nodes[0].reset_leaf_flags(self.draw);
        self.len = 1;
        self.pending = None;
        self.path.clear();
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
                let q = if n_a > 0.0 { node.wsum[i][a] / n_a } else { 0.0 };
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
    fn select(&mut self) {
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
            for i in 0..self.n {
                let a = edge.action[i] as usize;
                node.nvisit[i][a] += 1.0;
                node.wsum[i][a] += value[i];
            }
        }
        self.path.clear();
    }

    fn expand_backup(&mut self, policy: &[f32], value: &[f32]) {
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
        for i in 0..n {
            val[i] = if self.nodes[id].board.snakes[i].alive() {
                value[i]
            } else {
                -1.0
            };
        }
        self.backup(&val);
    }

    /// Root visit-count policy (`[n,4]`) and mean root value (`[n]`).
    fn root_targets(&self, pol: &mut [f32], val: &mut [f32]) {
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

// ---------------------------------------------------------------------------
// GPU: shared immutable net guarded by a mutex (the mutex IS the inference queue).
// ---------------------------------------------------------------------------

struct Gpu {
    net: *const snek_tch::AZNet,
    device: Device,
    c: i64,
    h: i64,
    w: i64,
}
// Safe: every access is serialised by the GPU mutex, and the scoped workers all
// join before `generate` returns, so this borrow of the net cannot dangle.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    fn forward(&self, obs: &[f32], rows: usize, pol: &mut [f32], val: &mut [f32]) {
        let net = unsafe { &*self.net };
        no_grad(|| {
            let x = Tensor::from_slice(obs)
                .reshape([rows as i64, self.c, self.h, self.w])
                .to_device(self.device);
            let (logits, value) = net.forward(&x);
            let probs = logits.softmax(-1, Kind::Float).to_device(Device::Cpu);
            let value = value.to_device(Device::Cpu);
            probs.copy_data(pol, rows * 4);
            value.copy_data(val, rows);
        });
    }
}

/// Per-game training trajectory: one recorded entry per turn (root board obs for
/// every snake, the root visit policy, the root value, the alive mask, and the
/// absolute game turn). Materialised into samples on game completion.
#[derive(Default)]
struct Traj {
    obs: Vec<f32>,    // steps * n * obs_len
    pol: Vec<f32>,    // steps * n * 4
    val: Vec<f32>,    // steps * n
    alive: Vec<bool>, // steps * n
    turn: Vec<u32>,   // steps
    steps: usize,
}

impl Traj {
    fn clear(&mut self) {
        self.obs.clear();
        self.pol.clear();
        self.val.clear();
        self.alive.clear();
        self.turn.clear();
        self.steps = 0;
    }
}

/// One worker's accumulated training samples.
#[derive(Default)]
struct WorkerOut {
    obs: Vec<f32>,
    pol: Vec<f32>,
    z: Vec<f32>,
    turn: Vec<u32>,
    turns: usize, // total board-steps this worker advanced
    games: usize, // completed (materialised) games
}

/// Generate one self-play generation. `boards`/`turns` carry the in-progress
/// games across generations and restarts: they hold `cfg.count` games and are
/// mutated in place (a completed game is replaced by a fresh start; games still
/// running at the sample target are left mid-play for the next generation to
/// resume). Empty / mismatched-length vecs are (re)initialised to fresh games.
pub fn generate(
    net: &SelfPlayNet<'_>,
    cfg: &RunConfig,
    seed: u64,
    metrics: &Metrics,
    stop: &AtomicBool,
    boards: &mut Vec<Board>,
    turns: &mut Vec<usize>,
) -> anyhow::Result<(Samples, Vec<GameJson>)> {
    let counters = metrics.counters();
    counters
        .samples_target
        .store(cfg.samples_per_gen as u32, Ordering::Relaxed);
    counters.samples_collected.store(0, Ordering::Relaxed);

    let n = cfg.num_snakes;
    let c = NUM_CHANNELS;
    let h = obs_side(cfg.board as usize);
    let w = obs_side(cfg.board as usize);
    let obs_len = c * h * w;
    let board = cfg.board;
    let sims = cfg.sims.max(1);
    let target = cfg.samples_per_gen;

    let chunk_games = cfg.gpu_batch_games.max(1);

    // (Re)initialise or repair the carried game state to exactly `count` games.
    let mut init_rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0xCA22_1E5D_F00Du64);
    if boards.len() != cfg.count || turns.len() != cfg.count {
        *boards = (0..cfg.count)
            .map(|_| standard_start(board, board, n, &mut init_rng))
            .collect();
        *turns = vec![0usize; cfg.count];
    } else {
        for (b, t) in boards.iter_mut().zip(turns.iter_mut()) {
            if b.is_terminal() {
                *b = standard_start(board, board, n, &mut init_rng);
                *t = 0;
            }
        }
    }

    let gpu = Arc::new(Gpu {
        net: net.net as *const snek_tch::AZNet,
        device: net.device,
        c: c as i64,
        h: h as i64,
        w: w as i64,
    });
    let gpu_lock = Arc::new(Mutex::new(()));
    let samples_total = Arc::new(AtomicUsize::new(0));
    let recorded: Arc<Mutex<Vec<GameJson>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded_len = Arc::new(AtomicUsize::new(0));

    // Disjoint mutable views: each worker owns one contiguous chunk of games and
    // mutates the shared `boards`/`turns` in place (so carry-out is free).
    let board_chunks: Vec<&mut [Board]> = boards.chunks_mut(chunk_games).collect();
    let turn_chunks: Vec<&mut [usize]> = turns.chunks_mut(chunk_games).collect();

    let worker_outs: Vec<WorkerOut> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk, (bslice, tslice)) in board_chunks.into_iter().zip(turn_chunks).enumerate() {
            let gpu = Arc::clone(&gpu);
            let gpu_lock = Arc::clone(&gpu_lock);
            let samples_total = Arc::clone(&samples_total);
            let recorded = Arc::clone(&recorded);
            let recorded_len = Arc::clone(&recorded_len);
            let counters: Arc<Counters> = Arc::clone(&counters);
            let cfg = cfg.clone();
            handles.push(scope.spawn(move || -> WorkerOut {
                let games = bslice.len();
                let rows = games * n; // constant padded batch shape for this worker
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                    seed ^ 0x9E37_79B9u64.wrapping_mul(chunk as u64 + 1),
                );
                let cap = sims + 4;

                let mut trees: Vec<Tree> = (0..games)
                    .map(|_| Tree::new(n, board, board, cfg.c_puct, cfg.draw_value, cap))
                    .collect();
                let mut trajs: Vec<Traj> = (0..games).map(|_| Traj::default()).collect();
                let mut rec_frames: Vec<Vec<FrameJson>> = vec![Vec::new(); games];

                let mut obs = vec![0.0f32; rows * obs_len];
                let mut pol = vec![0.0f32; rows * 4];
                let mut val = vec![0.0f32; rows];
                let mut root_pol = vec![0.0f32; n * 4];
                let mut root_val = vec![0.0f32; n];
                let mut play_pols = vec![[0.0f32; 4]; n];
                let mut actions = vec![Move::Up; n];

                let mut out = WorkerOut::default();

                'gen: while !stop.load(Ordering::Relaxed)
                    && samples_total.load(Ordering::Relaxed) < target
                {
                    for g in 0..games {
                        trees[g].reset(&bslice[g]);
                    }

                    for _ in 0..sims {
                        if stop.load(Ordering::Relaxed) {
                            break 'gen;
                        }
                        for g in 0..games {
                            trees[g].select();
                        }
                        for (g, tree) in trees.iter().enumerate() {
                            let base = g * n * obs_len;
                            match tree.pending {
                                Some(leaf) => {
                                    let b = &tree.nodes[leaf].board;
                                    for a in 0..n {
                                        let off = base + a * obs_len;
                                        encode_into(b, a, &mut obs[off..off + obs_len]);
                                    }
                                }
                                None => {
                                    for x in obs[base..base + n * obs_len].iter_mut() {
                                        *x = 0.0;
                                    }
                                }
                            }
                        }
                        let dt = {
                            let _g = gpu_lock.lock().unwrap();
                            let t0 = Instant::now();
                            gpu.forward(&obs, rows, &mut pol, &mut val);
                            t0.elapsed().as_micros() as u64
                        };
                        counters.gpu_forward_us.fetch_add(dt, Ordering::Relaxed);
                        counters.gpu_requests.fetch_add(1, Ordering::Relaxed);
                        counters.gpu_rows.fetch_add(rows as u64, Ordering::Relaxed);
                        counters.inferences.fetch_add(rows as u64, Ordering::Relaxed);
                        for g in 0..games {
                            if trees[g].pending.is_some() {
                                let pb = &pol[g * n * 4..(g + 1) * n * 4];
                                let vb = &val[g * n..(g + 1) * n];
                                trees[g].expand_backup(pb, vb);
                            }
                        }
                    }

                    let want_record = recorded_len.load(Ordering::Relaxed) < cfg.sample_games;
                    for g in 0..games {
                        trees[g].root_targets(&mut root_pol, &mut root_val);

                        // Record this turn's training sample (pre-step board).
                        let t = &mut trajs[g];
                        let obase = t.obs.len();
                        t.obs.resize(obase + n * obs_len, 0.0);
                        for s in 0..n {
                            let off = obase + s * obs_len;
                            encode_into(&bslice[g], s, &mut t.obs[off..off + obs_len]);
                            t.alive.push(bslice[g].snakes[s].alive());
                        }
                        t.pol.extend_from_slice(&root_pol);
                        t.val.extend_from_slice(&root_val);
                        t.turn.push(bslice[g].turn);
                        t.steps += 1;

                        // Choose moves (masked visit policy + exploration).
                        for s in 0..n {
                            let play = mask_obvious_immediate_deaths(
                                &bslice[g],
                                s,
                                &root_pol[s * 4..s * 4 + 4],
                            );
                            play_pols[s].copy_from_slice(&play);
                            actions[s] = sample_move(&play, cfg.exploration_prob, &mut rng);
                        }

                        if want_record && rec_frames[g].len() < MAX_REC_FRAMES {
                            rec_frames[g].push(frame_from_board(
                                &bslice[g],
                                n,
                                &root_pol,
                                &root_val,
                                &play_pols,
                                &actions,
                            ));
                        }

                        bslice[g].step_and_spawn(&actions, &mut rng);
                        tslice[g] += 1;
                        out.turns += 1;

                        let overrun = cfg.max_turns > 0 && tslice[g] >= cfg.max_turns;
                        if !bslice[g].is_terminal() && !overrun {
                            continue;
                        }

                        let winner = bslice[g].winner();
                        // Skip short draws entirely (no sample, no record).
                        if winner.is_none()
                            && !overrun
                            && cfg.skip_short_draw_turns > 0
                            && trajs[g].steps <= cfg.skip_short_draw_turns
                        {
                            trajs[g].clear();
                            rec_frames[g].clear();
                            bslice[g] = standard_start(board, board, n, &mut rng);
                            tslice[g] = 0;
                            continue;
                        }

                        // Finalise the recorded replay (append the terminal frame).
                        if want_record && !rec_frames[g].is_empty() {
                            let term_vals: Vec<f32> = (0..n)
                                .map(|s| {
                                    terminal_value(
                                        winner,
                                        s,
                                        bslice[g].snakes[s].alive(),
                                        cfg.draw_value,
                                    )
                                })
                                .collect();
                            let mut frames = std::mem::take(&mut rec_frames[g]);
                            frames.push(frame_from_board(
                                &bslice[g],
                                n,
                                &vec![0.0f32; n * 4],
                                &term_vals,
                                &vec![[0.0f32; 4]; n],
                                &actions,
                            ));
                            let num_turns = frames.len() as u32;
                            let mut guard = recorded.lock().unwrap();
                            if guard.len() < cfg.sample_games {
                                guard.push(GameJson {
                                    frames,
                                    winner: winner.map(|x| x as i32),
                                    num_turns,
                                });
                                recorded_len.store(guard.len(), Ordering::Relaxed);
                            }
                        }

                        let before = out.z.len();
                        materialize(
                            &trajs[g],
                            n,
                            obs_len,
                            winner,
                            cfg.draw_value,
                            cfg.bootstrap_value,
                            &mut out,
                        );
                        let added = out.z.len() - before;
                        trajs[g].clear();
                        rec_frames[g].clear();
                        out.games += 1;
                        counters.completed_games.fetch_add(1, Ordering::Relaxed);
                        let total = samples_total.fetch_add(added, Ordering::Relaxed) + added;
                        counters
                            .samples_collected
                            .store(total.min(target) as u32, Ordering::Relaxed);

                        bslice[g] = standard_start(board, board, n, &mut rng);
                        tslice[g] = 0;
                    }
                }

                out
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge worker outputs, truncate to exactly `samples_per_gen`.
    let mut s_obs = Vec::new();
    let mut s_pol = Vec::new();
    let mut s_z = Vec::new();
    let mut s_turn = Vec::new();
    let mut total_turns = 0usize;
    let mut total_games = 0usize;
    for out in &worker_outs {
        total_turns += out.turns;
        total_games += out.games;
        if s_z.len() >= target {
            continue;
        }
        let take = (target - s_z.len()).min(out.z.len());
        s_obs.extend_from_slice(&out.obs[..take * obs_len]);
        s_pol.extend_from_slice(&out.pol[..take * 4]);
        s_z.extend_from_slice(&out.z[..take]);
        s_turn.extend_from_slice(&out.turn[..take]);
    }

    counters
        .samples_collected
        .store(s_z.len().min(target) as u32, Ordering::Relaxed);

    let recorded = Arc::try_unwrap(recorded)
        .map(|m| m.into_inner().unwrap())
        .unwrap_or_default();

    Ok((
        Samples {
            obs: s_obs,
            pol: s_pol,
            z: s_z,
            turn: s_turn,
            obs_shape: [c, h, w],
            turns: total_turns,
            games: total_games,
        },
        recorded,
    ))
}

/// Materialise one finished game's trajectory into training samples: one sample
/// per (step, snake) where the snake was alive at that step. Value is the
/// bootstrap root value, or the exact terminal value.
fn materialize(
    t: &Traj,
    n: usize,
    obs_len: usize,
    winner: Option<usize>,
    draw_value: f32,
    bootstrap: bool,
    out: &mut WorkerOut,
) {
    let final_alive_base = t.steps.saturating_sub(1) * n;
    for st in 0..t.steps {
        for s in 0..n {
            if !t.alive[st * n + s] {
                continue;
            }
            let oi = (st * n + s) * obs_len;
            out.obs.extend_from_slice(&t.obs[oi..oi + obs_len]);
            let pi = (st * n + s) * 4;
            out.pol.extend_from_slice(&t.pol[pi..pi + 4]);
            out.z.push(if bootstrap {
                t.val[st * n + s]
            } else {
                terminal_value(winner, s, t.alive[final_alive_base + s], draw_value)
            });
            out.turn.push(t.turn[st]);
        }
    }
}

fn sample_move<R: Rng>(policy: &[f32], exploration_prob: f32, rng: &mut R) -> Move {
    let mut p = [0.0f32; 4];
    let legal = policy.iter().filter(|&&v| v > 0.0).count().max(1);
    for i in 0..4 {
        p[i] = if policy[i] > 0.0 {
            (1.0 - exploration_prob) * policy[i] + exploration_prob / legal as f32
        } else {
            0.0
        };
    }
    let idx = WeightedIndex::new(p)
        .map(|d| d.sample(rng))
        .unwrap_or_else(|_| rng.gen_range(0..4));
    Move::from_index(idx)
}
