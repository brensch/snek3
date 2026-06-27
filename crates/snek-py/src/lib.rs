//! Python bindings (`snek`) over `snek-core`.
//!
//! Exposes a vectorised `GameBatch` for self-play plus helpers for the server.
//! Observations are returned as zero-copy numpy arrays.

use numpy::ndarray::Array;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::{prelude::*, ThreadPoolBuilder};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::Rng;
use snek_core::{encode_into, standard_start, Board, Move, NUM_CHANNELS};
use snek_infer::Net;
use snek_search::{Forest, MctsForest};

fn move_from_u8(v: u8) -> Move {
    match v {
        0 => Move::Up,
        1 => Move::Down,
        2 => Move::Left,
        _ => Move::Right,
    }
}

/// Egocentric observation for snake `me` written contiguously into `out`.
fn encode_one(board: &Board, me: usize, out: &mut [f32]) {
    encode_into(board, me, out);
}

/// A batch of independent games sharing dimensions and snake count, advanced in
/// lockstep. Finished games hold their terminal state until reset.
#[pyclass]
struct GameBatch {
    boards: Vec<Board>,
    width: i8,
    height: i8,
    num_snakes: usize,
    rng: Xoshiro256PlusPlus,
    /// Transient search forest between `prepare_search` and `backup_search`.
    forest: Option<Forest>,
    /// Transient MCTS forest, driven by mcts_select / mcts_expand_backup.
    mcts: Option<MctsForest>,
}

#[pymethods]
impl GameBatch {
    #[new]
    #[pyo3(signature = (width, height, num_snakes, count, seed=0))]
    fn new(width: i8, height: i8, num_snakes: usize, count: usize, seed: u64) -> PyResult<Self> {
        if count == 0 {
            return Err(PyValueError::new_err("count must be > 0"));
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let boards = (0..count)
            .map(|_| standard_start(width, height, num_snakes, &mut rng))
            .collect();
        Ok(GameBatch {
            boards,
            width,
            height,
            num_snakes,
            rng,
            forest: None,
            mcts: None,
        })
    }

    /// Build a single-game batch from a Battlesnake `/move` request body.
    /// Returns `(batch, me_index)` where `me_index` is the controlled snake.
    /// Used by the server: run a search, then read `policy[0, me_index]`.
    #[staticmethod]
    fn from_request(body: &str) -> PyResult<(GameBatch, usize)> {
        let (board, me) =
            snek_core::json::parse_move_request(body).map_err(PyValueError::new_err)?;
        let (width, height) = (board.width, board.height);
        let num_snakes = board.snakes.len();
        Ok((
            GameBatch {
                boards: vec![board],
                width,
                height,
                num_snakes,
                rng: Xoshiro256PlusPlus::seed_from_u64(0),
                forest: None,
                mcts: None,
            },
            me,
        ))
    }

    #[getter]
    fn count(&self) -> usize {
        self.boards.len()
    }

    #[getter]
    fn num_snakes(&self) -> usize {
        self.num_snakes
    }

    #[getter]
    fn width(&self) -> i8 {
        self.width
    }

    #[getter]
    fn height(&self) -> i8 {
        self.height
    }

    #[getter]
    fn channels(&self) -> usize {
        NUM_CHANNELS
    }

    /// Advance every non-terminal game. `actions` has shape `[count, num_snakes]`
    /// of dtype uint8 (0=Up,1=Down,2=Left,3=Right). Moves for eliminated snakes
    /// and finished games are ignored.
    fn step(&mut self, actions: PyReadonlyArray2<u8>) -> PyResult<()> {
        let a = actions.as_array();
        if a.shape() != [self.boards.len(), self.num_snakes] {
            return Err(PyValueError::new_err(format!(
                "actions shape {:?} != [{}, {}]",
                a.shape(),
                self.boards.len(),
                self.num_snakes
            )));
        }
        let n = self.num_snakes;
        let mut moves: Vec<Move> = vec![Move::Up; n];
        // Advance the real games with food spawning. `rng` and `boards` are
        // disjoint fields, so we can borrow both at once.
        let rng = &mut self.rng;
        for (g, board) in self.boards.iter_mut().enumerate() {
            if board.is_terminal() {
                continue;
            }
            for s in 0..n {
                moves[s] = move_from_u8(a[[g, s]]);
            }
            board.step_and_spawn(&moves, rng);
        }
        Ok(())
    }

    /// Egocentric observations for every snake in every game:
    /// shape `[count, num_snakes, channels, height, width]`, dtype float32.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray5<f32>> {
        let h = self.height as usize;
        let w = self.width as usize;
        let n = self.num_snakes;
        let c = NUM_CHANNELS;
        let per_obs = c * h * w;
        let mut flat = vec![0.0f32; self.boards.len() * n * per_obs];
        flat.par_chunks_mut(n * per_obs)
            .zip(self.boards.par_iter())
            .for_each(|(chunk, board)| {
                for s in 0..n {
                    let base = s * per_obs;
                    encode_one(board, s, &mut chunk[base..base + per_obs]);
                }
            });
        let arr = Array::from_shape_vec((self.boards.len(), n, c, h, w), flat)
            .expect("shape matches length");
        arr.into_pyarray_bound(py)
    }

    /// Flood-fill / area-control baseline action per snake, shape
    /// `[count, num_snakes]`, dtype uint8. A fixed (non-learning) opponent.
    fn baseline_actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let n = self.num_snakes;
        let mut flat = vec![0u8; self.boards.len() * n];
        for (g, board) in self.boards.iter().enumerate() {
            for s in 0..n {
                flat[g * n + s] = snek_core::baseline::baseline_action(board, s) as u8;
            }
        }
        Array::from_shape_vec((self.boards.len(), n), flat)
            .unwrap()
            .into_pyarray_bound(py)
    }

    /// Per-snake alive mask, shape `[count, num_snakes]`, dtype uint8.
    fn alive<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let n = self.num_snakes;
        let mut flat = vec![0u8; self.boards.len() * n];
        flat.par_chunks_mut(n)
            .zip(self.boards.par_iter())
            .for_each(|(chunk, board)| {
                for s in 0..n {
                    chunk[s] = board.snakes[s].alive() as u8;
                }
            });
        Array::from_shape_vec((self.boards.len(), n), flat)
            .unwrap()
            .into_pyarray_bound(py)
    }

    /// Non-reversal move mask, shape `[count, num_snakes, 4]`, dtype uint8
    /// (1 = allowed). Excludes moving straight back onto the neck; the search
    /// determines true legality by stepping.
    fn legal_moves<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        let n = self.num_snakes;
        let mut flat = vec![1u8; self.boards.len() * n * 4];
        for (g, board) in self.boards.iter().enumerate() {
            for s in 0..n {
                let snake = &board.snakes[s];
                if !snake.alive() || snake.len() < 2 {
                    continue;
                }
                let head = snake.head();
                let neck = snake.body.get(1);
                for (mi, mv) in Move::ALL.iter().enumerate() {
                    if mv.apply(head) == neck {
                        flat[(g * n + s) * 4 + mi] = 0;
                    }
                }
            }
        }
        Array::from_shape_vec((self.boards.len(), n, 4), flat)
            .unwrap()
            .into_pyarray_bound(py)
    }

    /// Per-game terminal flag, shape `[count]`, dtype uint8.
    fn done<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
        let flat: Vec<u8> = self.boards.iter().map(|b| b.is_terminal() as u8).collect();
        flat.into_pyarray_bound(py)
    }

    /// Per-game winner, shape `[count]`, dtype int8. -1 = ongoing or draw,
    /// otherwise the surviving snake index.
    fn winners<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i8>> {
        let flat: Vec<i8> = self
            .boards
            .iter()
            .map(|b| b.winner().map(|i| i as i8).unwrap_or(-1))
            .collect();
        flat.into_pyarray_bound(py)
    }

    /// Phase 1 of a search step: build a fixed-`depth` equilibrium tree for the
    /// current state of every game and return the leaf observations the network
    /// must evaluate, shape `[num_evals, channels, height, width]`, float32.
    /// `num_evals` is `(non-terminal leaves across all games) * num_snakes`.
    /// Pair each `backup_search` with exactly one `prepare_search`.
    fn prepare_search<'py>(&mut self, py: Python<'py>, depth: u32) -> Bound<'py, PyArray4<f32>> {
        let forest = Forest::build(&self.boards, depth);
        let m = forest.eval_count();
        let obs_size = forest.obs_size();
        let (c, h, w) = (forest.channels, forest.height, forest.width);
        let mut flat = vec![0.0f32; m * obs_size];
        forest.write_observations(&mut flat);
        self.forest = Some(forest);
        Array::from_shape_vec((m, c, h, w), flat)
            .expect("shape matches length")
            .into_pyarray_bound(py)
    }

    /// Phase 2 of a search step: back up the per-leaf `values` (length
    /// `num_evals`, the rows returned by `prepare_search`) through every tree,
    /// solving a Logit Equilibrium at temperature `tau` (`iters` SFP steps) at
    /// each node. Returns root equilibrium policies, shape
    /// `[count, num_snakes, 4]`, float32. Terminal-root games yield all zeros.
    #[pyo3(signature = (values, tau=6.0, iters=200))]
    fn backup_search<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<f32>,
        tau: f32,
        iters: usize,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let mut forest = self
            .forest
            .take()
            .ok_or_else(|| PyValueError::new_err("call prepare_search before backup_search"))?;
        let v = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        if v.len() != forest.eval_count() {
            return Err(PyValueError::new_err(format!(
                "values length {} != expected {}",
                v.len(),
                forest.eval_count()
            )));
        }
        let (policy, _values) = forest.backup(v, tau, iters);
        Array::from_shape_vec((self.boards.len(), self.num_snakes, 4), policy)
            .map(|a| a.into_pyarray_bound(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Like `backup_search`, but also returns the per-agent root equilibrium
    /// value (the bootstrapped value of the current state, used as a TD target).
    /// Returns `(policies [count, N, 4], root_values [count, N])`, both float32.
    #[pyo3(signature = (values, tau=6.0, iters=200))]
    fn backup_search_values<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<f32>,
        tau: f32,
        iters: usize,
    ) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<f32>>)> {
        let mut forest = self
            .forest
            .take()
            .ok_or_else(|| PyValueError::new_err("call prepare_search before backup_search"))?;
        let v = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        if v.len() != forest.eval_count() {
            return Err(PyValueError::new_err(format!(
                "values length {} != expected {}",
                v.len(),
                forest.eval_count()
            )));
        }
        let (policy, root_vals) = forest.backup(v, tau, iters);
        let pol = Array::from_shape_vec((self.boards.len(), self.num_snakes, 4), policy)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray_bound(py);
        let vals = Array::from_shape_vec((self.boards.len(), self.num_snakes), root_vals)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray_bound(py);
        Ok((pol, vals))
    }

    /// Begin a batched AlphaZero MCTS over the current boards. Drive it with
    /// repeated mcts_select / mcts_expand_backup, then read mcts_root_targets.
    #[pyo3(signature = (c_puct=1.5))]
    fn mcts_new(&mut self, c_puct: f32) {
        self.mcts = Some(MctsForest::new(&self.boards, c_puct));
    }

    /// One MCTS selection step across all games. Returns `(pending, obs)`:
    /// `pending` [k] int64 game indices whose leaf needs a network estimate
    /// (terminal leaves were already backed up), and `obs`
    /// [k, num_snakes, C, H, W] float32 leaf observations for those games.
    fn mcts_select<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray5<f32>>)> {
        let forest = self
            .mcts
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("call mcts_new before mcts_select"))?;
        let pending = forest.select();
        let n = forest.n_snakes;
        let (c, h, w) = (forest.channels, forest.height, forest.width);
        let obs_size = forest.obs_size();
        let mut flat = vec![0.0f32; pending.len() * n * obs_size];
        forest.write_pending_obs(&pending, &mut flat);
        let pend_arr = PyArray1::from_vec_bound(py, pending.iter().map(|&x| x as i64).collect());
        let obs_arr = Array::from_shape_vec((pending.len(), n, c, h, w), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray_bound(py);
        Ok((pend_arr, obs_arr))
    }

    /// Expand and back up the evaluated leaves. `pending` [k] int64 (from
    /// mcts_select), `policies` [k*num_snakes*4] probabilities, `values`
    /// [k*num_snakes], both flattened row-major.
    fn mcts_expand_backup(
        &mut self,
        pending: PyReadonlyArray1<i64>,
        policies: PyReadonlyArray1<f32>,
        values: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let forest = self
            .mcts
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("call mcts_new before mcts_expand_backup"))?;
        let pend: Vec<usize> = pending
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .iter()
            .map(|&x| x as usize)
            .collect();
        let pol = policies.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let val = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        forest.expand_backup(&pend, pol, val);
        Ok(())
    }

    /// Read MCTS root targets: `(policies [count, num_snakes, 4]` visit-count
    /// distributions, `values [count, num_snakes]` mean root values)`.
    fn mcts_root_targets<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<f32>>)> {
        let forest = self
            .mcts
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("call mcts_new before mcts_root_targets"))?;
        let (pol, val) = forest.root_targets();
        let pol_arr = Array::from_shape_vec((self.boards.len(), self.num_snakes, 4), pol)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray_bound(py);
        let val_arr = Array::from_shape_vec((self.boards.len(), self.num_snakes), val)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray_bound(py);
        Ok((pol_arr, val_arr))
    }

    /// JSON snapshot of game `i`'s board state, for recording replays:
    /// `{turn, width, height, food: [[x,y]], hazards: [[x,y]],
    ///   snakes: [{alive, health, body: [[x,y]]}]}` (body is head-first).
    fn snapshot(&self, i: usize) -> PyResult<String> {
        let board = self
            .boards
            .get(i)
            .ok_or_else(|| PyValueError::new_err("game index out of range"))?;
        Ok(board_snapshot_value(board).to_string())
    }

    /// Reset every finished game to a fresh standard start. Returns how many
    /// games were reset.
    fn reset_done(&mut self) -> usize {
        let mut reset = 0;
        for board in self.boards.iter_mut() {
            if board.is_terminal() {
                *board = standard_start(self.width, self.height, self.num_snakes, &mut self.rng);
                reset += 1;
            }
        }
        reset
    }
}

/// Parse a Battlesnake `/move` request and return the egocentric observation for
/// the controlled snake (`you`): a tuple `(obs, me_index, legal_mask)` where
/// `obs` has shape `[channels, height, width]` and `legal_mask` is a length-4
/// uint8 array.
#[pyfunction]
fn encode_move_request<'py>(
    py: Python<'py>,
    body: &str,
) -> PyResult<(Bound<'py, PyArray3<f32>>, usize, Bound<'py, numpy::PyArray1<u8>>)> {
    let (board, me) =
        snek_core::json::parse_move_request(body).map_err(PyValueError::new_err)?;
    let h = board.height as usize;
    let w = board.width as usize;
    let c = NUM_CHANNELS;
    let mut flat = vec![0.0f32; c * h * w];
    encode_one(&board, me, &mut flat);
    let obs = Array::from_shape_vec((c, h, w), flat).unwrap().into_pyarray_bound(py);

    let mut legal = vec![1u8; 4];
    let snake = &board.snakes[me];
    if snake.alive() && snake.len() >= 2 {
        let head = snake.head();
        let neck = snake.body.get(1);
        for (mi, mv) in Move::ALL.iter().enumerate() {
            if mv.apply(head) == neck {
                legal[mi] = 0;
            }
        }
    }
    // Also forbid moves that step off the board.
    for (mi, mv) in Move::ALL.iter().enumerate() {
        if !board.in_bounds(mv.apply(snake.head())) {
            legal[mi] = 0;
        }
    }
    Ok((obs, me, legal.into_pyarray_bound(py)))
}

/// Configure the global Rayon thread pool used by Rust search/encoding.
///
/// Returns true when this call initialized the pool, false if Rayon had already
/// been initialized and the existing pool is being used.
#[pyfunction]
fn set_search_threads(threads: usize) -> PyResult<bool> {
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be > 0"));
    }
    match ThreadPoolBuilder::new().num_threads(threads).build_global() {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Sample one move from a 4-slot policy, mixing `explore` of a uniform over the
/// snake's *legal* (nonzero) moves. Dead/terminal snakes (all-zero) -> Up (ignored).
fn sample_move(probs: &[f32], explore: f32, rng: &mut Xoshiro256PlusPlus) -> Move {
    let k = probs.iter().filter(|&&p| p > 0.0).count();
    if k == 0 {
        return Move::Up;
    }
    let u = 1.0 / k as f32;
    let mut p = [0.0f32; 4];
    let mut total = 0.0f32;
    for i in 0..4 {
        p[i] = if probs[i] > 0.0 { (1.0 - explore) * probs[i] + explore * u } else { 0.0 };
        total += p[i];
    }
    let mut r = rng.gen::<f32>() * total;
    for i in 0..4 {
        r -= p[i];
        if r <= 0.0 {
            return Move::from_index(i);
        }
    }
    Move::from_index(3)
}

/// Per-game pending step records (flattened, step-major) until the game ends.
#[derive(Default, Clone)]
struct Slot {
    obs: Vec<f32>,    // each step: n*obs_size
    pol: Vec<f32>,    // each step: n*4
    value: Vec<f32>,  // each step: n  (search root value, for bootstrapped target)
    alive: Vec<bool>, // each step: n
    frames: Vec<serde_json::Value>,
    steps: usize,
}

struct CompletedGameSummary {
    turns: u32,
    winner: Option<usize>,
    overrun: bool,
    short_draw: bool,
    samples: usize,
}

impl CompletedGameSummary {
    fn to_json_string(&self) -> String {
        serde_json::json!({
            "turns": self.turns,
            "winner": self.winner.map(|w| w as i8).unwrap_or(-1),
            "overrun": self.overrun,
            "short_draw": self.short_draw,
            "samples": self.samples,
        })
        .to_string()
    }
}

fn board_snapshot_value(board: &Board) -> serde_json::Value {
    let food: Vec<[i8; 2]> = board.food.iter().map(|p| [p.x, p.y]).collect();
    let hazards: Vec<[i8; 2]> = board.hazards.iter().map(|p| [p.x, p.y]).collect();
    let snakes: Vec<serde_json::Value> = board
        .snakes
        .iter()
        .map(|s| {
            let body: Vec<[i8; 2]> = s.body.iter().map(|p| [p.x, p.y]).collect();
            serde_json::json!({"alive": s.alive(), "health": s.health, "body": body})
        })
        .collect();
    serde_json::json!({
        "turn": board.turn,
        "width": board.width,
        "height": board.height,
        "food": food,
        "hazards": hazards,
        "snakes": snakes,
    })
}

/// Select one MCTS leaf per (game, agent) and write their observations.
/// Returns `(obs_flat, m)` where `m = pending_games * n`.
fn select_write(forest: &mut MctsForest, pend: &mut Vec<usize>, n: usize, obs_size: usize) -> (Vec<f32>, usize) {
    *pend = forest.select();
    let m = pend.len() * n;
    let mut buf = vec![0.0f32; m * obs_size];
    forest.write_pending_obs(pend, &mut buf);
    (buf, m)
}

/// AlphaZero self-play entirely in Rust with a **CPU/GPU double-buffer pipeline**:
/// a dedicated GPU thread runs ONNX/CUDA inference (via `ort`) continuously while
/// the main thread plays the MCTS of the *other* buffer, then they swap — so the
/// GPU is always inferring and the CPU is always playing. Fresh games each call.
/// Returns `(obs [N,C,H,W], policy [N,4], value [N], stats)` zero-copy. Policy
/// target = root visit counts; value target = undiscounted game outcome.
#[pyfunction]
#[pyo3(signature = (onnx_path, board=11, num_snakes=2, count=1024, sims=32, c_puct=1.5,
    samples_per_gen=30000, seed=0, exploration_prob=0.25, max_turns=0, eval_chunk=16384,
    draw_value=0.0, skip_short_draw_turns=0, record_games=0, bootstrap_value=false))]
#[allow(clippy::too_many_arguments)]
fn generate_selfplay<'py>(
    py: Python<'py>,
    onnx_path: &str,
    board: i8,
    num_snakes: usize,
    count: usize,
    sims: usize,
    c_puct: f32,
    samples_per_gen: usize,
    seed: u64,
    exploration_prob: f32,
    max_turns: i64,
    eval_chunk: usize,
    draw_value: f32,
    skip_short_draw_turns: usize,
    record_games: usize,
    bootstrap_value: bool,
) -> PyResult<(
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyDict>,
)> {
    let c = NUM_CHANNELS;
    let h = board as usize;
    let w = board as usize;
    let obs_size = c * h * w;
    let n = num_snakes;

    // --- GPU worker thread: owns the ort Net, infers batches off a channel ---
    type Req = Option<(u8, Vec<f32>, usize)>; // (buffer id, obs flat, m); None = stop
    type Res = (u8, Vec<f32>, Vec<f32>);      // (buffer id, policy probs, values)
    let (tx_req, rx_req) = std::sync::mpsc::channel::<Req>();
    let (tx_res, rx_res) = std::sync::mpsc::channel::<Res>();
    let onnx = onnx_path.to_string();
    let gpu = std::thread::spawn(move || -> Result<(f64, f64, usize), String> {
        use std::time::{Duration, Instant};
        let mut net = Net::load(&onnx).map_err(|e| e.to_string())?;
        let (mut t_fwd, mut t_idle) = (Duration::ZERO, Duration::ZERO);
        let mut evals = 0usize;
        loop {
            let wt = Instant::now();
            let msg = rx_req.recv();
            t_idle += wt.elapsed();
            let (id, obs, m) = match msg {
                Ok(Some(x)) => x,
                _ => break,
            };
            evals += m;
            let f = Instant::now();
            let mut pol = vec![0.0f32; m * 4];
            let mut val = vec![0.0f32; m];
            let mut s = 0;
            while s < m {
                let e = (s + eval_chunk).min(m);
                let (p, v) = net
                    .forward(&obs[s * obs_size..e * obs_size], e - s, c, h, w)
                    .map_err(|er| er.to_string())?;
                pol[s * 4..e * 4].copy_from_slice(&p);
                val[s..e].copy_from_slice(&v);
                s = e;
            }
            t_fwd += f.elapsed();
            if tx_res.send((id, pol, val)).is_err() {
                break;
            }
        }
        if std::env::var("SNEK_PIPELINE_TIMING").is_ok() {
            let tot = (t_fwd + t_idle).as_secs_f64().max(1e-9);
            eprintln!(
                "[gpu] forward={:.2}s idle={:.2}s busy={:.0}%",
                t_fwd.as_secs_f64(),
                t_idle.as_secs_f64(),
                100.0 * t_fwd.as_secs_f64() / tot
            );
        }
        Ok((t_fwd.as_secs_f64(), t_idle.as_secs_f64(), evals))
    });

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    // Two buffers of `count` games each. While the GPU infers one, the CPU plays the other.
    let mut boards: Vec<Vec<Board>> = (0..2)
        .map(|_| (0..count).map(|_| standard_start(board, board, n, &mut rng)).collect())
        .collect();
    let mut slots: Vec<Vec<Slot>> = (0..2).map(|_| vec![Slot::default(); count]).collect();
    let mut turns: Vec<Vec<u32>> = (0..2).map(|_| vec![0u32; count]).collect();

    let mut out_obs: Vec<f32> = Vec::new();
    let mut out_pol: Vec<f32> = Vec::new();
    let mut out_z: Vec<f32> = Vec::new();
    let mut collected = 0usize;
    let mut skipped_short_draw_games = 0usize;
    let mut skipped_short_draw_samples = 0usize;
    let mut recorded_games_json: Vec<String> = Vec::new();
    let mut recorded_game_candidates = 0usize;
    let mut completed_games_json: Vec<String> = Vec::new();
    let mut actions: Vec<Move> = vec![Move::Up; n];

    // Run the pipeline. On any error we drop the channels, which stops the GPU thread.
    use std::time::{Duration, Instant};
    let (mut t_recv, mut t_mcts, mut t_rp) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let mut run = || -> Result<(), String> {
        while collected < samples_per_gen {
            let mut forests = [
                MctsForest::new_with_draw_value(&boards[0], c_puct, draw_value),
                MctsForest::new_with_draw_value(&boards[1], c_puct, draw_value),
            ];
            let mut pend: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            let mut sims_done = [0usize, 0usize];

            // prime: submit buffer 0, prepare + queue buffer 1
            let (o0, m0) = select_write(&mut forests[0], &mut pend[0], n, obs_size);
            tx_req.send(Some((0, o0, m0))).map_err(|_| "gpu gone".to_string())?;
            let (o1, m1) = select_write(&mut forests[1], &mut pend[1], n, obs_size);
            let mut queued: Option<(u8, Vec<f32>, usize)> = Some((1, o1, m1));

            // Pipeline: always one batch on the GPU + one queued; CPU does the other's backup+select.
            while sims_done[0] < sims || sims_done[1] < sims {
                let w = Instant::now();
                let (id, pol, val) = rx_res.recv().map_err(|_| "gpu gone".to_string())?;
                t_recv += w.elapsed();
                if let Some(q) = queued.take() {
                    tx_req.send(Some(q)).map_err(|_| "gpu gone".to_string())?;
                }
                let m0 = Instant::now();
                let bi = id as usize;
                forests[bi].expand_backup(&pend[bi], &pol, &val);
                sims_done[bi] += 1;
                if sims_done[bi] < sims {
                    let (o, m) = select_write(&mut forests[bi], &mut pend[bi], n, obs_size);
                    queued = Some((bi as u8, o, m));
                }
                t_mcts += m0.elapsed();
            }

            // Both buffers: read root policy, record, play a move, finalize finished games.
            let rp0 = Instant::now();
            for b in 0..2 {
                let (root_pol, root_val) = forests[b].root_targets();
                let bds = &mut boards[b];
                let slt = &mut slots[b];
                let trn = &mut turns[b];
                for g in 0..count {
                    let bd = &bds[g];
                    let slot = &mut slt[g];
                    if record_games > 0 {
                        slot.frames.push(board_snapshot_value(bd));
                    }
                    for s in 0..n {
                        let base = slot.obs.len();
                        slot.obs.resize(base + obs_size, 0.0);
                        encode_into(bd, s, &mut slot.obs[base..base + obs_size]);
                        slot.alive.push(bd.snakes[s].alive());
                    }
                    slot.pol.extend_from_slice(&root_pol[g * n * 4..(g + 1) * n * 4]);
                    slot.value.extend_from_slice(&root_val[g * n..(g + 1) * n]);
                    slot.steps += 1;
                }
                for g in 0..count {
                    for s in 0..n {
                        let base = (g * n + s) * 4;
                        actions[s] = sample_move(&root_pol[base..base + 4], exploration_prob, &mut rng);
                    }
                    bds[g].step_and_spawn(&actions, &mut rng);
                    trn[g] += 1;
                }
                for g in 0..count {
                    let overrun = max_turns > 0 && trn[g] as i64 >= max_turns;
                    if !(bds[g].is_terminal() || overrun) {
                        continue;
                    }
                    if record_games > 0 {
                        slt[g].frames.push(board_snapshot_value(&bds[g]));
                    }
                    let winner = bds[g].winner();
                    let slot = std::mem::take(&mut slt[g]);
                    let live_samples = slot.alive.iter().filter(|&&a| a).count();
                    let short_terminal_draw =
                        winner.is_none() && !overrun && skip_short_draw_turns > 0 && slot.steps <= skip_short_draw_turns;
                    completed_games_json.push(
                        CompletedGameSummary {
                            turns: trn[g],
                            winner,
                            overrun,
                            short_draw: short_terminal_draw,
                            samples: live_samples,
                        }
                        .to_json_string(),
                    );
                    if short_terminal_draw {
                        skipped_short_draw_games += 1;
                        skipped_short_draw_samples += live_samples;
                        if record_games > 0 && !slot.frames.is_empty() {
                            let game = serde_json::json!({
                                "opponent": "net",
                                "winner": -1,
                                "num_turns": slot.frames.len(),
                                "frames": slot.frames,
                            });
                            recorded_game_candidates += 1;
                            if recorded_games_json.len() < record_games {
                                recorded_games_json.push(game.to_string());
                            } else {
                                let replace = rng.gen_range(0..recorded_game_candidates);
                                if replace < record_games {
                                    recorded_games_json[replace] = game.to_string();
                                }
                            }
                        }
                        bds[g] = standard_start(board, board, n, &mut rng);
                        trn[g] = 0;
                        continue;
                    }
                    for st in 0..slot.steps {
                        for s in 0..n {
                            if !slot.alive[st * n + s] {
                                continue;
                            }
                            let oi = (st * n + s) * obs_size;
                            out_obs.extend_from_slice(&slot.obs[oi..oi + obs_size]);
                            let pi = (st * n + s) * 4;
                            out_pol.extend_from_slice(&slot.pol[pi..pi + 4]);
                            // Value target: equilibrium-bootstrapped (the search's root
                            // value at this state, grounded by terminal nodes within the
                            // tree) when enabled, else the flat Monte-Carlo game outcome.
                            out_z.push(if bootstrap_value {
                                slot.value[st * n + s]
                            } else {
                                match winner {
                                    Some(wi) if wi == s => 1.0,
                                    Some(_) => -1.0,
                                    None => draw_value,
                                }
                            });
                            collected += 1;
                        }
                    }
                    if record_games > 0 && !slot.frames.is_empty() {
                        let game = serde_json::json!({
                            "opponent": "net",
                            "winner": winner.map(|wi| wi as i8).unwrap_or(-1),
                            "num_turns": slot.frames.len(),
                            "frames": slot.frames,
                        });
                        recorded_game_candidates += 1;
                        if recorded_games_json.len() < record_games {
                            recorded_games_json.push(game.to_string());
                        } else {
                            let replace = rng.gen_range(0..recorded_game_candidates);
                            if replace < record_games {
                                recorded_games_json[replace] = game.to_string();
                            }
                        }
                    }
                    bds[g] = standard_start(board, board, n, &mut rng);
                    trn[g] = 0;
                }
            }
            t_rp += rp0.elapsed();
        }
        Ok(())
    };
    let run_res = run();
    if std::env::var("SNEK_PIPELINE_TIMING").is_ok() {
        eprintln!(
            "[cpu] recv_wait={:.2}s mcts={:.2}s record/play={:.2}s",
            t_recv.as_secs_f64(),
            t_mcts.as_secs_f64(),
            t_rp.as_secs_f64()
        );
    }
    let _ = tx_req.send(None); // stop the GPU thread
    drop(tx_req);
    let join = gpu.join();
    run_res.map_err(PyValueError::new_err)?;
    let (gpu_forward_seconds, gpu_idle_seconds, inference_count) = match join {
        Ok(Ok(stats)) => stats,
        Ok(Err(e)) => return Err(PyValueError::new_err(format!("gpu thread: {e}"))),
        Err(_) => return Err(PyValueError::new_err("gpu thread panicked")),
    };

    let obs_arr = Array::from_shape_vec((collected, c, h, w), out_obs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray_bound(py);
    let pol_arr = Array::from_shape_vec((collected, 4), out_pol)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray_bound(py);
    let z_arr = PyArray1::from_vec_bound(py, out_z);
    let stats = PyDict::new_bound(py);
    let gpu_total = (gpu_forward_seconds + gpu_idle_seconds).max(1e-9);
    stats.set_item("inference_count", inference_count)?;
    stats.set_item("inference_seconds", gpu_forward_seconds)?;
    stats.set_item("inference_per_sec", inference_count as f64 / gpu_forward_seconds.max(1e-9))?;
    stats.set_item("gpu_forward_seconds", gpu_forward_seconds)?;
    stats.set_item("gpu_idle_seconds", gpu_idle_seconds)?;
    stats.set_item("gpu_busy_pct", 100.0 * gpu_forward_seconds / gpu_total)?;
    stats.set_item("cpu_recv_wait_seconds", t_recv.as_secs_f64())?;
    stats.set_item("cpu_mcts_seconds", t_mcts.as_secs_f64())?;
    stats.set_item("cpu_record_play_seconds", t_rp.as_secs_f64())?;
    stats.set_item("skipped_short_draw_games", skipped_short_draw_games)?;
    stats.set_item("skipped_short_draw_samples", skipped_short_draw_samples)?;
    stats.set_item("draw_value", draw_value)?;
    stats.set_item("bootstrap_value", bootstrap_value)?;
    stats.set_item("skip_short_draw_turns", skip_short_draw_turns)?;
    stats.set_item("recorded_game_candidates", recorded_game_candidates)?;
    stats.set_item("recorded_games_json", recorded_games_json)?;
    stats.set_item("completed_games_json", completed_games_json)?;
    Ok((obs_arr, pol_arr, z_arr, stats))
}

#[pymodule]
fn snek(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CHANNELS", NUM_CHANNELS)?;
    m.add_class::<GameBatch>()?;
    m.add_function(wrap_pyfunction!(encode_move_request, m)?)?;
    m.add_function(wrap_pyfunction!(set_search_threads, m)?)?;
    m.add_function(wrap_pyfunction!(generate_selfplay, m)?)?;
    Ok(())
}
