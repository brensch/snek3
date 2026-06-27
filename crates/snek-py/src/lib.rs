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
        let v = serde_json::json!({
            "turn": board.turn,
            "width": board.width,
            "height": board.height,
            "food": food,
            "hazards": hazards,
            "snakes": snakes,
        });
        Ok(v.to_string())
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
    alive: Vec<bool>, // each step: n
    steps: usize,
}

/// Persistent AlphaZero self-play driver, entirely in Rust: decoupled-PUCT MCTS
/// with batched ONNX/CUDA inference (via `ort`), no Python round-trips. Games
/// **persist across `generate` calls**, so a large `count` (big GPU batches) does
/// not waste inference on games left unfinished when a sample target is hit.
#[pyclass]
struct SelfPlay {
    boards: Vec<Board>,
    slots: Vec<Slot>,
    turns: Vec<u32>,
    rng: Xoshiro256PlusPlus,
    board: i8,
    num_snakes: usize,
}

#[pymethods]
impl SelfPlay {
    #[new]
    #[pyo3(signature = (board=11, num_snakes=2, count=1024, seed=0))]
    fn new(board: i8, num_snakes: usize, count: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let boards = (0..count)
            .map(|_| standard_start(board, board, num_snakes, &mut rng))
            .collect();
        SelfPlay {
            boards,
            slots: vec![Slot::default(); count],
            turns: vec![0u32; count],
            rng,
            board,
            num_snakes,
        }
    }

    /// Advance self-play until at least `samples_target` samples are collected
    /// from finished games, keeping unfinished games for the next call. Returns
    /// `(obs [N,C,H,W], policy [N,4], value [N])` zero-copy. Policy target = root
    /// visit counts; value target = undiscounted game outcome.
    #[pyo3(signature = (onnx_path, sims=32, c_puct=1.5, samples_target=30000,
        exploration_prob=0.25, max_turns=0, eval_chunk=16384))]
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &mut self,
        py: Python<'py>,
        onnx_path: &str,
        sims: usize,
        c_puct: f32,
        samples_target: usize,
        exploration_prob: f32,
        max_turns: i64,
        eval_chunk: usize,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>)> {
        let c = NUM_CHANNELS;
        let h = self.board as usize;
        let w = self.board as usize;
        let obs_size = c * h * w;
        let n = self.num_snakes;
        let count = self.boards.len();
        let board = self.board;

        let mut net = Net::load(onnx_path)
            .map_err(|e| PyValueError::new_err(format!("ort load failed: {e}")))?;

        let mut out_obs: Vec<f32> = Vec::new();
        let mut out_pol: Vec<f32> = Vec::new();
        let mut out_z: Vec<f32> = Vec::new();
        let mut collected = 0usize;
        let mut leaf_buf: Vec<f32> = Vec::new();
        let mut actions: Vec<Move> = vec![Move::Up; n];

        while collected < samples_target {
            let mut forest = MctsForest::new(&self.boards, c_puct);
            for _ in 0..sims {
                let pending = forest.select();
                if pending.is_empty() {
                    continue;
                }
                let m = pending.len() * n;
                leaf_buf.clear();
                leaf_buf.resize(m * obs_size, 0.0);
                forest.write_pending_obs(&pending, &mut leaf_buf);
                let mut pol_all = vec![0.0f32; m * 4];
                let mut val_all = vec![0.0f32; m];
                let mut start = 0;
                while start < m {
                    let end = (start + eval_chunk).min(m);
                    let (p, v) = net
                        .forward(&leaf_buf[start * obs_size..end * obs_size], end - start, c, h, w)
                        .map_err(|e| PyValueError::new_err(format!("ort forward: {e}")))?;
                    pol_all[start * 4..end * 4].copy_from_slice(&p);
                    val_all[start..end].copy_from_slice(&v);
                    start = end;
                }
                forest.expand_backup(&pending, &pol_all, &val_all);
            }
            let (root_pol, _root_val) = forest.root_targets();

            // record current positions
            for g in 0..count {
                let bd = &self.boards[g];
                let slot = &mut self.slots[g];
                for s in 0..n {
                    let base = slot.obs.len();
                    slot.obs.resize(base + obs_size, 0.0);
                    encode_into(bd, s, &mut slot.obs[base..base + obs_size]);
                    slot.alive.push(bd.snakes[s].alive());
                }
                slot.pol.extend_from_slice(&root_pol[g * n * 4..(g + 1) * n * 4]);
                slot.steps += 1;
            }

            // play one move per game
            for g in 0..count {
                for s in 0..n {
                    let base = (g * n + s) * 4;
                    actions[s] = sample_move(&root_pol[base..base + 4], exploration_prob, &mut self.rng);
                }
                self.boards[g].step_and_spawn(&actions, &mut self.rng);
                self.turns[g] += 1;
            }

            // finalize finished games (emit samples, then reset that game)
            for g in 0..count {
                let overrun = max_turns > 0 && self.turns[g] as i64 >= max_turns;
                if !(self.boards[g].is_terminal() || overrun) {
                    continue;
                }
                let winner = self.boards[g].winner();
                let slot = std::mem::take(&mut self.slots[g]);
                for st in 0..slot.steps {
                    for s in 0..n {
                        if !slot.alive[st * n + s] {
                            continue;
                        }
                        let oi = (st * n + s) * obs_size;
                        out_obs.extend_from_slice(&slot.obs[oi..oi + obs_size]);
                        let pi = (st * n + s) * 4;
                        out_pol.extend_from_slice(&slot.pol[pi..pi + 4]);
                        out_z.push(match winner {
                            Some(wi) if wi == s => 1.0,
                            Some(_) => -1.0,
                            None => 0.0,
                        });
                        collected += 1;
                    }
                }
                self.boards[g] = standard_start(board, board, n, &mut self.rng);
                self.turns[g] = 0;
            }
        }

        let obs_arr = Array::from_shape_vec((collected, c, h, w), out_obs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray_bound(py);
        let pol_arr = Array::from_shape_vec((collected, 4), out_pol)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray_bound(py);
        let z_arr = PyArray1::from_vec_bound(py, out_z);
        Ok((obs_arr, pol_arr, z_arr))
    }
}

#[pymodule]
fn snek(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CHANNELS", NUM_CHANNELS)?;
    m.add_class::<GameBatch>()?;
    m.add_function(wrap_pyfunction!(encode_move_request, m)?)?;
    m.add_function(wrap_pyfunction!(set_search_threads, m)?)?;
    m.add_class::<SelfPlay>()?;
    Ok(())
}
