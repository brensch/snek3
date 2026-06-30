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
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::{prelude::*, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use snek_core::{encode_into, obs_side, standard_start, Board, Move, NUM_CHANNELS};
use snek_infer::Net;
#[cfg(test)]
use snek_search::obvious_immediate_death;
use snek_search::{mask_obvious_immediate_deaths, MctsForest};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static CANCEL_SELFPLAY: AtomicBool = AtomicBool::new(false);
static SELFPLAY_PROGRESS_ACTIVE: AtomicBool = AtomicBool::new(false);
static SELFPLAY_PROGRESS_DONE: AtomicUsize = AtomicUsize::new(0);
static SELFPLAY_PROGRESS_TOTAL: AtomicUsize = AtomicUsize::new(0);
static SELFPLAY_PROGRESS_COMPLETED_GAMES: AtomicUsize = AtomicUsize::new(0);
static SELFPLAY_PROGRESS_INFERENCES: AtomicUsize = AtomicUsize::new(0);
static SELFPLAY_PROGRESS_BATCH_MAX_ROWS: AtomicUsize = AtomicUsize::new(0);
static SELFPLAY_PROGRESS_GPU_REQUESTS: AtomicUsize = AtomicUsize::new(0);
static SELFPLAY_PROGRESS_GPU_ROWS: AtomicUsize = AtomicUsize::new(0);
static SELFPLAY_PROGRESS_GPU_LAST_ROWS: AtomicUsize = AtomicUsize::new(0);
static SELFPLAY_PROGRESS_GPU_FORWARD_US: AtomicU64 = AtomicU64::new(0);
static SELFPLAY_PROGRESS_GPU_IDLE_US: AtomicU64 = AtomicU64::new(0);
static SELFPLAY_PROGRESS_CPU_RECV_US: AtomicU64 = AtomicU64::new(0);
static SELFPLAY_PROGRESS_CPU_MCTS_US: AtomicU64 = AtomicU64::new(0);
static SELFPLAY_PROGRESS_CPU_RECORD_PLAY_US: AtomicU64 = AtomicU64::new(0);
static NEXT_SELFPLAY_STATE_ID: AtomicU64 = AtomicU64::new(1);
static SELFPLAY_STATES: OnceLock<Mutex<HashMap<u64, SelfPlayState>>> = OnceLock::new();

fn duration_us(d: std::time::Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

fn selfplay_states() -> &'static Mutex<HashMap<u64, SelfPlayState>> {
    SELFPLAY_STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

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
        let h = obs_side(self.height as usize);
        let w = obs_side(self.width as usize);
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
        let pol = policies
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let val = values
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
        Ok(board_snapshot_value(board, None, None).to_string())
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
) -> PyResult<(
    Bound<'py, PyArray3<f32>>,
    usize,
    Bound<'py, numpy::PyArray1<u8>>,
)> {
    let (board, me) = snek_core::json::parse_move_request(body).map_err(PyValueError::new_err)?;
    let h = obs_side(board.height as usize);
    let w = obs_side(board.width as usize);
    let c = NUM_CHANNELS;
    let mut flat = vec![0.0f32; c * h * w];
    encode_one(&board, me, &mut flat);
    let obs = Array::from_shape_vec((c, h, w), flat)
        .unwrap()
        .into_pyarray_bound(py);

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

/// Request cancellation of long-running Rust self-play from the dashboard/API.
#[pyfunction]
fn request_cancel() {
    CANCEL_SELFPLAY.store(true, Ordering::SeqCst);
}

/// Clear any pending self-play cancellation before a new generation/run starts.
#[pyfunction]
fn clear_cancel() {
    CANCEL_SELFPLAY.store(false, Ordering::SeqCst);
}

#[pyfunction]
fn selfplay_progress(py: Python<'_>) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("active", SELFPLAY_PROGRESS_ACTIVE.load(Ordering::Relaxed))?;
    d.set_item("done", SELFPLAY_PROGRESS_DONE.load(Ordering::Relaxed))?;
    d.set_item("total", SELFPLAY_PROGRESS_TOTAL.load(Ordering::Relaxed))?;
    d.set_item(
        "completed_games",
        SELFPLAY_PROGRESS_COMPLETED_GAMES.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "inferences",
        SELFPLAY_PROGRESS_INFERENCES.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "batch_max_rows",
        SELFPLAY_PROGRESS_BATCH_MAX_ROWS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "gpu_requests",
        SELFPLAY_PROGRESS_GPU_REQUESTS.load(Ordering::Relaxed),
    )?;
    d.set_item("gpu_rows", SELFPLAY_PROGRESS_GPU_ROWS.load(Ordering::Relaxed))?;
    d.set_item(
        "gpu_last_rows",
        SELFPLAY_PROGRESS_GPU_LAST_ROWS.load(Ordering::Relaxed),
    )?;
    d.set_item(
        "gpu_forward_seconds",
        SELFPLAY_PROGRESS_GPU_FORWARD_US.load(Ordering::Relaxed) as f64 / 1_000_000.0,
    )?;
    d.set_item(
        "gpu_idle_seconds",
        SELFPLAY_PROGRESS_GPU_IDLE_US.load(Ordering::Relaxed) as f64 / 1_000_000.0,
    )?;
    d.set_item(
        "cpu_recv_wait_seconds",
        SELFPLAY_PROGRESS_CPU_RECV_US.load(Ordering::Relaxed) as f64 / 1_000_000.0,
    )?;
    d.set_item(
        "cpu_mcts_seconds",
        SELFPLAY_PROGRESS_CPU_MCTS_US.load(Ordering::Relaxed) as f64 / 1_000_000.0,
    )?;
    d.set_item(
        "cpu_record_play_seconds",
        SELFPLAY_PROGRESS_CPU_RECORD_PLAY_US.load(Ordering::Relaxed) as f64 / 1_000_000.0,
    )?;
    Ok(d.into())
}

fn selfplay_state_info_dict(py: Python<'_>, state: &SelfPlayState) -> PyResult<PyObject> {
    let mut nonempty_slots = 0usize;
    let mut pending_steps = 0usize;
    let mut pending_alive_samples = 0usize;
    let mut pending_frames = 0usize;
    let mut active_turn_sum = 0usize;
    let mut active_turn_max = 0u32;
    for ((board_buf, slot_buf), turn_buf) in state
        .boards
        .iter()
        .zip(state.slots.iter())
        .zip(state.turns.iter())
    {
        for ((_board, slot), &turn) in board_buf.iter().zip(slot_buf.iter()).zip(turn_buf.iter()) {
            if slot.steps == 0 {
                continue;
            }
            nonempty_slots += 1;
            pending_steps += slot.steps;
            pending_alive_samples += slot.alive.iter().filter(|&&alive| alive).count();
            pending_frames += slot.frames.len();
            active_turn_sum += turn as usize;
            active_turn_max = active_turn_max.max(turn);
        }
    }
    let d = PyDict::new_bound(py);
    d.set_item("board", state.board)?;
    d.set_item("num_snakes", state.num_snakes)?;
    d.set_item("count", state.count)?;
    d.set_item(
        "gpu_batch_games",
        state.boards.iter().map(Vec::len).max().unwrap_or(0),
    )?;
    d.set_item("shards", state.boards.len())?;
    d.set_item("buffers", state.boards.len())?;
    d.set_item("slots", state.boards.iter().map(Vec::len).sum::<usize>())?;
    d.set_item("nonempty_slots", nonempty_slots)?;
    d.set_item("pending_steps", pending_steps)?;
    d.set_item("pending_alive_samples", pending_alive_samples)?;
    d.set_item("pending_frames", pending_frames)?;
    d.set_item("completed_games", state.completed_samples.len())?;
    d.set_item("completed_samples", state.collected)?;
    d.set_item("active_turn_sum", active_turn_sum)?;
    d.set_item("active_turn_max", active_turn_max)?;
    d.set_item(
        "active_turn_mean",
        if nonempty_slots > 0 {
            active_turn_sum as f64 / nonempty_slots as f64
        } else {
            0.0
        },
    )?;
    Ok(d.into())
}

fn check_cancelled() -> Result<(), String> {
    if CANCEL_SELFPLAY.load(Ordering::SeqCst) {
        return Err("cancelled".to_string());
    }
    Python::with_gil(|py| {
        if py.check_signals().is_err() {
            CANCEL_SELFPLAY.store(true, Ordering::SeqCst);
            Err("interrupted".to_string())
        } else {
            Ok(())
        }
    })
}

/// Create an in-process continuous self-play state and return its handle.
#[pyfunction]
#[pyo3(signature = (board=11, num_snakes=4, count=512, seed=0, gpu_batch_games=128))]
fn create_selfplay_state(
    board: i8,
    num_snakes: usize,
    count: usize,
    seed: u64,
    gpu_batch_games: usize,
) -> PyResult<u64> {
    if count == 0 {
        return Err(PyValueError::new_err("count must be > 0"));
    }
    if gpu_batch_games == 0 {
        return Err(PyValueError::new_err("gpu_batch_games must be > 0"));
    }
    let id = NEXT_SELFPLAY_STATE_ID.fetch_add(1, Ordering::SeqCst);
    let state = new_selfplay_state(board, num_snakes, count, gpu_batch_games, seed);
    selfplay_states()
        .lock()
        .map_err(|_| PyValueError::new_err("self-play state lock poisoned"))?
        .insert(id, state);
    Ok(id)
}

/// Drop a continuous self-play state handle.
#[pyfunction]
fn drop_selfplay_state(state_id: u64) -> PyResult<bool> {
    let removed = selfplay_states()
        .lock()
        .map_err(|_| PyValueError::new_err("self-play state lock poisoned"))?
        .remove(&state_id)
        .is_some();
    Ok(removed)
}

#[pyfunction]
fn save_selfplay_state(state_id: u64, path: &str) -> PyResult<bool> {
    let states = selfplay_states()
        .lock()
        .map_err(|_| PyValueError::new_err("self-play state lock poisoned"))?;
    let Some(state) = states.get(&state_id) else {
        return Ok(false);
    };
    save_selfplay_state_to_path(path, state).map_err(PyValueError::new_err)?;
    Ok(true)
}

#[pyfunction]
fn load_selfplay_state(path: &str) -> PyResult<u64> {
    let state = load_selfplay_state_from_path(path).map_err(PyValueError::new_err)?;
    let id = NEXT_SELFPLAY_STATE_ID.fetch_add(1, Ordering::SeqCst);
    selfplay_states()
        .lock()
        .map_err(|_| PyValueError::new_err("self-play state lock poisoned"))?
        .insert(id, state);
    Ok(id)
}

#[pyfunction]
fn selfplay_state_info(py: Python<'_>, state_id: u64) -> PyResult<PyObject> {
    let states = selfplay_states()
        .lock()
        .map_err(|_| PyValueError::new_err("self-play state lock poisoned"))?;
    let Some(state) = states.get(&state_id) else {
        return Ok(py.None());
    };
    selfplay_state_info_dict(py, state)
}

/// Sample one move from a 4-slot policy, mixing `explore` of a uniform over the
/// snake's *legal* (nonzero) moves. Dead/terminal snakes (all-zero) -> Up (ignored).
fn sample_move_with_play_policy(
    probs: &[f32],
    explore: f32,
    rng: &mut Xoshiro256PlusPlus,
) -> (Move, [f32; 4]) {
    let k = probs.iter().filter(|&&p| p > 0.0).count();
    if k == 0 {
        return (Move::Up, [1.0, 0.0, 0.0, 0.0]);
    }
    let u = 1.0 / k as f32;
    let mut p = [0.0f32; 4];
    let mut total = 0.0f32;
    for i in 0..4 {
        p[i] = if probs[i] > 0.0 {
            (1.0 - explore) * probs[i] + explore * u
        } else {
            0.0
        };
        total += p[i];
    }
    let mut r = rng.gen::<f32>() * total;
    for i in 0..4 {
        r -= p[i];
        if r <= 0.0 {
            return (Move::from_index(i), p);
        }
    }
    (Move::from_index(3), p)
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

#[derive(Clone, Serialize, Deserialize)]
struct SerSlot {
    obs: Vec<f32>,
    pol: Vec<f32>,
    value: Vec<f32>,
    alive: Vec<bool>,
    frames: Vec<String>,
    steps: usize,
}

impl From<&Slot> for SerSlot {
    fn from(slot: &Slot) -> Self {
        Self {
            obs: slot.obs.clone(),
            pol: slot.pol.clone(),
            value: slot.value.clone(),
            alive: slot.alive.clone(),
            frames: slot.frames.iter().map(|v| v.to_string()).collect(),
            steps: slot.steps,
        }
    }
}

fn deserialize_slot(slot: SerSlot) -> Result<Slot, String> {
    Ok(Slot {
        obs: slot.obs,
        pol: slot.pol,
        value: slot.value,
        alive: slot.alive,
        frames: slot
            .frames
            .into_iter()
            .map(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, String>>()?,
        steps: slot.steps,
    })
}

#[derive(Clone, Serialize, Deserialize)]
struct SerSnake {
    body: Vec<[i8; 2]>,
    health: i16,
    eliminated: Option<u8>,
}

#[derive(Clone, Serialize, Deserialize)]
struct SerBoard {
    width: i8,
    height: i8,
    turn: u32,
    snakes: Vec<SerSnake>,
    food: Vec<[i8; 2]>,
    hazards: Vec<[i8; 2]>,
    hazard_damage: i16,
    min_food: i32,
    food_spawn_chance: i32,
}

#[derive(Clone, Serialize, Deserialize)]
struct SerSelfPlayState {
    version: u32,
    board: i8,
    num_snakes: usize,
    count: usize,
    boards: Vec<Vec<SerBoard>>,
    slots: Vec<Vec<SerSlot>>,
    turns: Vec<Vec<u32>>,
    rng: Xoshiro256PlusPlus,
    completed_samples: Vec<SerGameSamples>,
    collected: usize,
    skipped_short_draw_games: usize,
    skipped_short_draw_samples: usize,
    recorded_games_json: Vec<String>,
    completed_games_json: Vec<String>,
    recorded_game_candidates: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct SerSelfPlayStateV1 {
    version: u32,
    board: i8,
    num_snakes: usize,
    count: usize,
    boards: Vec<Vec<SerBoard>>,
    slots: Vec<Vec<SerSlot>>,
    turns: Vec<Vec<u32>>,
    rng: Xoshiro256PlusPlus,
}

const SELFPLAY_STATE_VERSION: u32 = 2;

#[derive(Clone, Serialize, Deserialize)]
struct SerGameSamples {
    turns: u32,
    obs: Vec<f32>,
    pol: Vec<f32>,
    z: Vec<f32>,
    samples: usize,
}

#[derive(Clone)]
struct SelfPlayState {
    board: i8,
    num_snakes: usize,
    count: usize,
    boards: Vec<Vec<Board>>,
    slots: Vec<Vec<Slot>>,
    turns: Vec<Vec<u32>>,
    rng: Xoshiro256PlusPlus,
    completed_samples: Vec<GameSamples>,
    collected: usize,
    skipped_short_draw_games: usize,
    skipped_short_draw_samples: usize,
    recorded_games_json: Vec<String>,
    completed_games_json: Vec<String>,
    recorded_game_candidates: usize,
}

impl From<&GameSamples> for SerGameSamples {
    fn from(samples: &GameSamples) -> Self {
        Self {
            turns: samples.turns,
            obs: samples.obs.clone(),
            pol: samples.pol.clone(),
            z: samples.z.clone(),
            samples: samples.samples,
        }
    }
}

impl From<SerGameSamples> for GameSamples {
    fn from(samples: SerGameSamples) -> Self {
        Self {
            turns: samples.turns,
            obs: samples.obs,
            pol: samples.pol,
            z: samples.z,
            samples: samples.samples,
        }
    }
}

fn cause_to_u8(c: snek_core::EliminatedCause) -> u8 {
    match c {
        snek_core::EliminatedCause::Collision => 0,
        snek_core::EliminatedCause::SelfCollision => 1,
        snek_core::EliminatedCause::OutOfBounds => 2,
        snek_core::EliminatedCause::OutOfHealth => 3,
        snek_core::EliminatedCause::HeadToHead => 4,
        snek_core::EliminatedCause::Hazard => 5,
    }
}

fn cause_from_u8(v: u8) -> Result<snek_core::EliminatedCause, String> {
    match v {
        0 => Ok(snek_core::EliminatedCause::Collision),
        1 => Ok(snek_core::EliminatedCause::SelfCollision),
        2 => Ok(snek_core::EliminatedCause::OutOfBounds),
        3 => Ok(snek_core::EliminatedCause::OutOfHealth),
        4 => Ok(snek_core::EliminatedCause::HeadToHead),
        5 => Ok(snek_core::EliminatedCause::Hazard),
        _ => Err(format!("unknown eliminated cause {v}")),
    }
}

fn serialize_board(board: &Board) -> SerBoard {
    SerBoard {
        width: board.width,
        height: board.height,
        turn: board.turn,
        snakes: board
            .snakes
            .iter()
            .map(|s| SerSnake {
                body: s.body.iter().map(|p| [p.x, p.y]).collect(),
                health: s.health,
                eliminated: s.eliminated.map(cause_to_u8),
            })
            .collect(),
        food: board.food.iter().map(|p| [p.x, p.y]).collect(),
        hazards: board.hazards.iter().map(|p| [p.x, p.y]).collect(),
        hazard_damage: board.hazard_damage,
        min_food: board.min_food,
        food_spawn_chance: board.food_spawn_chance,
    }
}

fn deserialize_board(board: SerBoard) -> Result<Board, String> {
    let mut out = Board::new(board.width, board.height);
    out.turn = board.turn;
    out.food = board
        .food
        .into_iter()
        .map(|p| snek_core::Point::new(p[0], p[1]))
        .collect();
    out.hazards = board
        .hazards
        .into_iter()
        .map(|p| snek_core::Point::new(p[0], p[1]))
        .collect();
    out.hazard_damage = board.hazard_damage;
    out.min_food = board.min_food;
    out.food_spawn_chance = board.food_spawn_chance;
    for snake in board.snakes {
        let mut body = snek_core::Body::new();
        let points: Vec<snek_core::Point> = snake
            .body
            .into_iter()
            .map(|p| snek_core::Point::new(p[0], p[1]))
            .collect();
        body.init_from_head_first(&points);
        out.snakes.push(snek_core::Snake {
            body,
            health: snake.health,
            eliminated: snake.eliminated.map(cause_from_u8).transpose()?,
        });
    }
    Ok(out)
}

impl From<&SelfPlayState> for SerSelfPlayState {
    fn from(state: &SelfPlayState) -> Self {
        Self {
            version: SELFPLAY_STATE_VERSION,
            board: state.board,
            num_snakes: state.num_snakes,
            count: state.count,
            boards: state
                .boards
                .iter()
                .map(|buf| buf.iter().map(serialize_board).collect())
                .collect(),
            slots: state
                .slots
                .iter()
                .map(|buf| buf.iter().map(SerSlot::from).collect())
                .collect(),
            turns: state.turns.clone(),
            rng: state.rng.clone(),
            completed_samples: state
                .completed_samples
                .iter()
                .map(SerGameSamples::from)
                .collect(),
            collected: state.collected,
            skipped_short_draw_games: state.skipped_short_draw_games,
            skipped_short_draw_samples: state.skipped_short_draw_samples,
            recorded_games_json: state.recorded_games_json.clone(),
            completed_games_json: state.completed_games_json.clone(),
            recorded_game_candidates: state.recorded_game_candidates,
        }
    }
}

fn deserialize_selfplay_state(state: SerSelfPlayState) -> Result<SelfPlayState, String> {
    if state.version != SELFPLAY_STATE_VERSION {
        return Err(format!(
            "unsupported self-play state version {}",
            state.version
        ));
    }
    let boards = state
        .boards
        .into_iter()
        .map(|buf| buf.into_iter().map(deserialize_board).collect())
        .collect::<Result<Vec<Vec<Board>>, String>>()?;
    let slots = state
        .slots
        .into_iter()
        .map(|buf| buf.into_iter().map(deserialize_slot).collect())
        .collect::<Result<Vec<Vec<Slot>>, String>>()?;
    Ok(SelfPlayState {
        board: state.board,
        num_snakes: state.num_snakes,
        count: state.count,
        boards,
        slots,
        turns: state.turns,
        rng: state.rng,
        completed_samples: state
            .completed_samples
            .into_iter()
            .map(GameSamples::from)
            .collect(),
        collected: state.collected,
        skipped_short_draw_games: state.skipped_short_draw_games,
        skipped_short_draw_samples: state.skipped_short_draw_samples,
        recorded_games_json: state.recorded_games_json,
        completed_games_json: state.completed_games_json,
        recorded_game_candidates: state.recorded_game_candidates,
    })
}

fn deserialize_selfplay_state_v1(state: SerSelfPlayStateV1) -> Result<SelfPlayState, String> {
    if state.version != 1 {
        return Err(format!(
            "unsupported self-play state version {}",
            state.version
        ));
    }
    let boards = state
        .boards
        .into_iter()
        .map(|buf| buf.into_iter().map(deserialize_board).collect())
        .collect::<Result<Vec<Vec<Board>>, String>>()?;
    let slots = state
        .slots
        .into_iter()
        .map(|buf| buf.into_iter().map(deserialize_slot).collect())
        .collect::<Result<Vec<Vec<Slot>>, String>>()?;
    Ok(SelfPlayState {
        board: state.board,
        num_snakes: state.num_snakes,
        count: state.count,
        boards,
        slots,
        turns: state.turns,
        rng: state.rng,
        completed_samples: Vec::new(),
        collected: 0,
        skipped_short_draw_games: 0,
        skipped_short_draw_samples: 0,
        recorded_games_json: Vec::new(),
        completed_games_json: Vec::new(),
        recorded_game_candidates: 0,
    })
}

fn save_selfplay_state_to_path(path: &str, state: &SelfPlayState) -> Result<(), String> {
    let path = std::path::Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let bytes = bincode::serialize(&SerSelfPlayState::from(state)).map_err(|e| e.to_string())?;
    let tmp = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}."))
            .unwrap_or_default()
    ));
    std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn load_selfplay_state_from_path(path: &str) -> Result<SelfPlayState, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    match bincode::deserialize::<SerSelfPlayState>(&bytes) {
        Ok(state) => deserialize_selfplay_state(state),
        Err(v2_err) => {
            let state: SerSelfPlayStateV1 = bincode::deserialize(&bytes).map_err(|v1_err| {
                format!("could not read self-play state v2 ({v2_err}) or v1 ({v1_err})")
            })?;
            deserialize_selfplay_state_v1(state)
        }
    }
}

fn shard_sizes(count: usize, gpu_batch_games: usize) -> Vec<usize> {
    let mut remaining = count;
    let mut out = Vec::new();
    while remaining > 0 {
        let n = remaining.min(gpu_batch_games);
        out.push(n);
        remaining -= n;
    }
    out
}

fn new_selfplay_state(
    board: i8,
    num_snakes: usize,
    count: usize,
    gpu_batch_games: usize,
    seed: u64,
) -> SelfPlayState {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let sizes = shard_sizes(count, gpu_batch_games);
    let boards: Vec<Vec<Board>> = sizes
        .iter()
        .map(|&shard_count| {
            (0..shard_count)
                .map(|_| standard_start(board, board, num_snakes, &mut rng))
                .collect()
        })
        .collect();
    let slots: Vec<Vec<Slot>> = sizes
        .iter()
        .map(|&shard_count| vec![Slot::default(); shard_count])
        .collect();
    let turns: Vec<Vec<u32>> = sizes
        .iter()
        .map(|&shard_count| vec![0u32; shard_count])
        .collect();
    SelfPlayState {
        board,
        num_snakes,
        count,
        boards,
        slots,
        turns,
        rng,
        completed_samples: Vec::new(),
        collected: 0,
        skipped_short_draw_games: 0,
        skipped_short_draw_samples: 0,
        recorded_games_json: Vec::new(),
        completed_games_json: Vec::new(),
        recorded_game_candidates: 0,
    }
}

fn compatible_selfplay_state(
    state: &SelfPlayState,
    board: i8,
    num_snakes: usize,
    count: usize,
    gpu_batch_games: usize,
) -> bool {
    state.board == board
        && state.num_snakes == num_snakes
        && state.count == count
        && state
            .boards
            .iter()
            .map(Vec::len)
            .eq(shard_sizes(count, gpu_batch_games))
}

#[derive(Clone)]
struct GameSamples {
    turns: u32,
    obs: Vec<f32>,
    pol: Vec<f32>,
    z: Vec<f32>,
    samples: usize,
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

fn terminal_sample_value(
    winner: Option<usize>,
    snake: usize,
    alive_before_terminal_step: bool,
    draw_value: f32,
) -> f32 {
    match winner {
        Some(wi) if wi == snake => 1.0,
        Some(_) => -1.0,
        None if alive_before_terminal_step => draw_value,
        None => -1.0,
    }
}

fn board_snapshot_value(
    board: &Board,
    policy: Option<&[f32]>,
    value: Option<&[f32]>,
) -> serde_json::Value {
    let food: Vec<[i8; 2]> = board.food.iter().map(|p| [p.x, p.y]).collect();
    let hazards: Vec<[i8; 2]> = board.hazards.iter().map(|p| [p.x, p.y]).collect();
    let snakes: Vec<serde_json::Value> = board
        .snakes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let body: Vec<[i8; 2]> = s.body.iter().map(|p| [p.x, p.y]).collect();
            let mut snake = serde_json::json!({
                "alive": s.alive(),
                "health": s.health,
                "body": body,
            });
            if let Some(pol) = policy {
                if pol.len() >= (i + 1) * 4 {
                    snake["policy"] = serde_json::json!([
                        pol[i * 4],
                        pol[i * 4 + 1],
                        pol[i * 4 + 2],
                        pol[i * 4 + 3],
                    ]);
                }
            }
            if let Some(val) = value {
                if val.len() > i {
                    snake["value"] = serde_json::json!(val[i]);
                }
            }
            snake
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

fn annotate_frame_action(
    frame: &mut serde_json::Value,
    snake: usize,
    play_policy: [f32; 4],
    chosen: Move,
) {
    if let Some(snake_v) = frame
        .get_mut("snakes")
        .and_then(|snakes| snakes.get_mut(snake))
    {
        snake_v["play_policy"] = serde_json::json!(play_policy);
        snake_v["chosen_move"] = serde_json::json!(chosen.index());
    }
}

fn balanced_sample_output(
    games: &mut [GameSamples],
    target: usize,
    obs_size: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize, Vec<usize>) {
    if games.is_empty() || target == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), 0, Vec::new());
    }

    let max_bucket = games
        .iter()
        .map(|g| (g.turns / 10) as usize)
        .max()
        .unwrap_or(0);
    let mut buckets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); max_bucket + 1];
    for (gi, game) in games.iter().enumerate() {
        let bi = (game.turns / 10) as usize;
        buckets[bi].reserve(game.samples);
        for si in 0..game.samples {
            buckets[bi].push((gi, si));
        }
    }
    for bucket in buckets.iter_mut() {
        bucket.shuffle(rng);
    }

    let active: Vec<usize> = buckets
        .iter()
        .enumerate()
        .filter_map(|(i, b)| (!b.is_empty()).then_some(i))
        .collect();
    let mut offsets = vec![0usize; buckets.len()];
    let mut selected = Vec::with_capacity(target);
    while selected.len() < target {
        let mut progressed = false;
        for &bi in &active {
            if selected.len() >= target {
                break;
            }
            if offsets[bi] < buckets[bi].len() {
                selected.push((bi, buckets[bi][offsets[bi]]));
                offsets[bi] += 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    selected.shuffle(rng);
    let mut out_obs = Vec::with_capacity(selected.len() * obs_size);
    let mut out_pol = Vec::with_capacity(selected.len() * 4);
    let mut out_z = Vec::with_capacity(selected.len());
    let mut selected_buckets = vec![0usize; buckets.len()];
    for (bi, (gi, si)) in selected {
        let game = &games[gi];
        let oi = si * obs_size;
        out_obs.extend_from_slice(&game.obs[oi..oi + obs_size]);
        let pi = si * 4;
        out_pol.extend_from_slice(&game.pol[pi..pi + 4]);
        out_z.push(game.z[si]);
        selected_buckets[bi] += 1;
    }
    let count = out_z.len();
    (out_obs, out_pol, out_z, count, selected_buckets)
}

/// Select one MCTS leaf per (game, agent) and write their observations.
/// Returns `(obs_flat, m)` where `m = pending_games * n`.
fn select_write(
    forest: &mut MctsForest,
    pend: &mut Vec<usize>,
    n: usize,
    obs_size: usize,
) -> (Vec<f32>, usize) {
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
    samples_per_gen=30000, seed=0, exploration_prob=0.25, max_turns=0, gpu_batch_games=128,
    draw_value=0.0, skip_short_draw_turns=0, record_games=0, bootstrap_value=false,
    state_id=None, state_path=None))]
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
    gpu_batch_games: usize,
    draw_value: f32,
    skip_short_draw_turns: usize,
    record_games: usize,
    bootstrap_value: bool,
    state_id: Option<u64>,
    state_path: Option<String>,
) -> PyResult<(
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyDict>,
)> {
    clear_cancel();
    if gpu_batch_games == 0 {
        return Err(PyValueError::new_err("gpu_batch_games must be > 0"));
    }
    SELFPLAY_PROGRESS_ACTIVE.store(true, Ordering::Relaxed);
    SELFPLAY_PROGRESS_TOTAL.store(samples_per_gen, Ordering::Relaxed);
    SELFPLAY_PROGRESS_INFERENCES.store(0, Ordering::Relaxed);
    SELFPLAY_PROGRESS_BATCH_MAX_ROWS
        .store(gpu_batch_games.min(count) * num_snakes, Ordering::Relaxed);
    SELFPLAY_PROGRESS_GPU_REQUESTS.store(0, Ordering::Relaxed);
    SELFPLAY_PROGRESS_GPU_ROWS.store(0, Ordering::Relaxed);
    SELFPLAY_PROGRESS_GPU_LAST_ROWS.store(0, Ordering::Relaxed);
    SELFPLAY_PROGRESS_GPU_FORWARD_US.store(0, Ordering::Relaxed);
    SELFPLAY_PROGRESS_GPU_IDLE_US.store(0, Ordering::Relaxed);
    SELFPLAY_PROGRESS_CPU_RECV_US.store(0, Ordering::Relaxed);
    SELFPLAY_PROGRESS_CPU_MCTS_US.store(0, Ordering::Relaxed);
    SELFPLAY_PROGRESS_CPU_RECORD_PLAY_US.store(0, Ordering::Relaxed);
    let c = NUM_CHANNELS;
    let h = obs_side(board as usize);
    let w = obs_side(board as usize);
    let obs_size = c * h * w;
    let n = num_snakes;

    // --- GPU worker thread: owns the ort Net, infers batches off a channel ---
    type Req = Option<(usize, Vec<f32>, usize)>; // (shard id, obs flat, m); None = stop
    type Res = (usize, Vec<f32>, Vec<f32>); // (shard id, policy probs, values)
    let (tx_req, rx_req) = std::sync::mpsc::channel::<Req>();
    let (tx_res, rx_res) = std::sync::mpsc::channel::<Res>();
    let onnx = onnx_path.to_string();
    let inference_progress = Arc::new(AtomicUsize::new(0));
    let inference_progress_gpu = Arc::clone(&inference_progress);
    let gpu = std::thread::spawn(move || -> Result<(f64, f64, usize), String> {
        use std::time::{Duration, Instant};
        let mut net = Net::load(&onnx).map_err(|e| e.to_string())?;
        let (mut t_fwd, mut t_idle) = (Duration::ZERO, Duration::ZERO);
        let mut evals = 0usize;
        loop {
            let wt = Instant::now();
            let msg = rx_req.recv();
            let idle = wt.elapsed();
            t_idle += idle;
            SELFPLAY_PROGRESS_GPU_IDLE_US.fetch_add(duration_us(idle), Ordering::Relaxed);
            let (id, obs, m) = match msg {
                Ok(Some(x)) => x,
                _ => break,
            };
            evals += m;
            SELFPLAY_PROGRESS_GPU_REQUESTS.fetch_add(1, Ordering::Relaxed);
            SELFPLAY_PROGRESS_GPU_ROWS.fetch_add(m, Ordering::Relaxed);
            SELFPLAY_PROGRESS_GPU_LAST_ROWS.store(m, Ordering::Relaxed);
            inference_progress_gpu.store(evals, Ordering::Relaxed);
            SELFPLAY_PROGRESS_INFERENCES.store(evals, Ordering::Relaxed);
            let f = Instant::now();
            let (pol, val) = net.forward(&obs, m, c, h, w).map_err(|er| er.to_string())?;
            let forward = f.elapsed();
            t_fwd += forward;
            SELFPLAY_PROGRESS_GPU_FORWARD_US.fetch_add(duration_us(forward), Ordering::Relaxed);
            if tx_res.send((id, pol, val)).is_err() {
                break;
            }
        }
        Ok((t_fwd.as_secs_f64(), t_idle.as_secs_f64(), evals))
    });

    let state = if let Some(id) = state_id {
        let mut states = selfplay_states()
            .lock()
            .map_err(|_| PyValueError::new_err("self-play state lock poisoned"))?;
        match states.remove(&id) {
            Some(st) if compatible_selfplay_state(&st, board, n, count, gpu_batch_games) => st,
            _ => new_selfplay_state(board, n, count, gpu_batch_games, seed),
        }
    } else {
        new_selfplay_state(board, n, count, gpu_batch_games, seed)
    };
    let mut boards = state.boards;
    let mut slots = state.slots;
    let mut turns = state.turns;
    let mut rng = state.rng;
    let state_path_for_run = state_path.clone();

    let mut completed_samples = state.completed_samples;
    let mut collected = state.collected;
    let mut skipped_short_draw_games = state.skipped_short_draw_games;
    let mut skipped_short_draw_samples = state.skipped_short_draw_samples;
    let mut recorded_games_json = state.recorded_games_json;
    let mut recorded_game_candidates = state.recorded_game_candidates;
    let mut completed_games_json = state.completed_games_json;
    let mut actions: Vec<Move> = vec![Move::Up; n];
    SELFPLAY_PROGRESS_DONE.store(collected.min(samples_per_gen), Ordering::Relaxed);
    SELFPLAY_PROGRESS_COMPLETED_GAMES.store(completed_samples.len(), Ordering::Relaxed);

    // Run the pipeline. On any error we drop the channels, which stops the GPU thread.
    use std::time::{Duration, Instant};
    // `run` is a `move` closure so it OWNS the !Sync Receiver (and a cloned
    // Sender); we keep the original `tx_req` to stop the GPU thread afterwards.
    // It returns the accumulated outputs since boards/rng/channels aren't needed
    // after the loop. This lets us run it under `py.allow_threads`.
    let tx_req_c = tx_req.clone();
    type SpOut = (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        usize,
        usize,
        usize,
        Vec<String>,
        Vec<String>,
        usize,
        Vec<usize>,
        SelfPlayState,
        f64,
        f64,
        f64,
    );
    let run = move || -> Result<SpOut, String> {
        macro_rules! check_cancelled_or_persist {
            () => {
                if let Err(e) = check_cancelled() {
                    let state = SelfPlayState {
                        board,
                        num_snakes: n,
                        count,
                        boards: boards.clone(),
                        slots: slots.clone(),
                        turns: turns.clone(),
                        rng: rng.clone(),
                        completed_samples: completed_samples.clone(),
                        collected,
                        skipped_short_draw_games,
                        skipped_short_draw_samples,
                        recorded_games_json: recorded_games_json.clone(),
                        completed_games_json: completed_games_json.clone(),
                        recorded_game_candidates,
                    };
                    if let Some(path) = state_path_for_run.as_deref() {
                        save_selfplay_state_to_path(path, &state)?;
                    }
                    if let Some(id) = state_id {
                        selfplay_states()
                            .lock()
                            .map_err(|_| "self-play state lock poisoned".to_string())?
                            .insert(id, state);
                    }
                    return Err(e);
                }
            };
        }
        let (mut t_recv, mut t_mcts, mut t_rp) =
            (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        while collected < samples_per_gen {
            check_cancelled_or_persist!();
            let shard_count = boards.len();
            let mut forests: Vec<MctsForest> = boards
                .iter()
                .map(|buf| MctsForest::new_with_draw_value(buf, c_puct, draw_value))
                .collect();
            let mut pend: Vec<Vec<usize>> = vec![Vec::new(); shard_count];
            let mut sims_done = vec![0usize; shard_count];
            let mut active_shards = 0usize;

            for sid in 0..shard_count {
                let (obs, m) = select_write(&mut forests[sid], &mut pend[sid], n, obs_size);
                tx_req_c
                    .send(Some((sid, obs, m)))
                    .map_err(|_| "gpu gone".to_string())?;
                active_shards += 1;
            }

            while active_shards > 0 {
                check_cancelled_or_persist!();
                let w = Instant::now();
                let (sid, pol, val) = rx_res.recv().map_err(|_| "gpu gone".to_string())?;
                let recv_wait = w.elapsed();
                t_recv += recv_wait;
                SELFPLAY_PROGRESS_CPU_RECV_US
                    .fetch_add(duration_us(recv_wait), Ordering::Relaxed);
                let m0 = Instant::now();
                forests[sid].expand_backup(&pend[sid], &pol, &val);
                sims_done[sid] += 1;
                // Tier-1 cost cut: once roots are expanded and have a credited
                // visit (the 2nd backup), freeze games whose every alive snake has
                // a single legal move — their root policy is already an exact
                // one-hot, so they drop out of the eval batch instead of burning
                // the remaining sims. Targets are unchanged (zero quality cost).
                if sims_done[sid] == 2 {
                    forests[sid].freeze_forced_roots();
                }
                if sims_done[sid] < sims {
                    let (obs, m) = select_write(&mut forests[sid], &mut pend[sid], n, obs_size);
                    tx_req_c
                        .send(Some((sid, obs, m)))
                        .map_err(|_| "gpu gone".to_string())?;
                } else {
                    active_shards -= 1;
                }
                let mcts_elapsed = m0.elapsed();
                t_mcts += mcts_elapsed;
                SELFPLAY_PROGRESS_CPU_MCTS_US
                    .fetch_add(duration_us(mcts_elapsed), Ordering::Relaxed);
            }

            // Both buffers: read root policy, record, play a move, finalize finished games.
            let rp0 = Instant::now();
            for sid in 0..shard_count {
                let (root_pol, root_val) = forests[sid].root_targets();
                let bds = &mut boards[sid];
                let slt = &mut slots[sid];
                let trn = &mut turns[sid];
                let shard_games = bds.len();
                for g in 0..shard_games {
                    if g % 64 == 0 {
                        check_cancelled_or_persist!();
                    }
                    let bd = &bds[g];
                    let slot = &mut slt[g];
                    let pol_slice = &root_pol[g * n * 4..(g + 1) * n * 4];
                    let val_slice = &root_val[g * n..(g + 1) * n];
                    if record_games > 0 {
                        slot.frames.push(board_snapshot_value(
                            bd,
                            Some(pol_slice),
                            Some(val_slice),
                        ));
                    }
                    for s in 0..n {
                        let base = slot.obs.len();
                        slot.obs.resize(base + obs_size, 0.0);
                        encode_into(bd, s, &mut slot.obs[base..base + obs_size]);
                        slot.alive.push(bd.snakes[s].alive());
                    }
                    slot.pol.extend_from_slice(pol_slice);
                    slot.value.extend_from_slice(val_slice);
                    slot.steps += 1;
                }
                for g in 0..shard_games {
                    if g % 64 == 0 {
                        check_cancelled_or_persist!();
                    }
                    for s in 0..n {
                        let base = (g * n + s) * 4;
                        let play_base =
                            mask_obvious_immediate_deaths(&bds[g], s, &root_pol[base..base + 4]);
                        let (chosen, play_policy) =
                            sample_move_with_play_policy(&play_base, exploration_prob, &mut rng);
                        actions[s] = chosen;
                        if record_games > 0 {
                            if let Some(frame) = slt[g].frames.last_mut() {
                                annotate_frame_action(frame, s, play_policy, chosen);
                            }
                        }
                    }
                    bds[g].step_and_spawn(&actions, &mut rng);
                    trn[g] += 1;
                }
                for g in 0..shard_games {
                    if g % 64 == 0 {
                        check_cancelled_or_persist!();
                    }
                    let overrun = max_turns > 0 && trn[g] as i64 >= max_turns;
                    if !(bds[g].is_terminal() || overrun) {
                        continue;
                    }
                    if record_games > 0 {
                        slt[g]
                            .frames
                            .push(board_snapshot_value(&bds[g], None, None));
                    }
                    let winner = bds[g].winner();
                    let slot = std::mem::take(&mut slt[g]);
                    let live_samples = slot.alive.iter().filter(|&&a| a).count();
                    let short_terminal_draw = winner.is_none()
                        && !overrun
                        && skip_short_draw_turns > 0
                        && slot.steps <= skip_short_draw_turns;
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
                    let final_alive_base = slot.steps.saturating_sub(1) * n;
                    let mut game_obs: Vec<f32> = Vec::with_capacity(live_samples * obs_size);
                    let mut game_pol: Vec<f32> = Vec::with_capacity(live_samples * 4);
                    let mut game_z: Vec<f32> = Vec::with_capacity(live_samples);
                    for st in 0..slot.steps {
                        for s in 0..n {
                            if !slot.alive[st * n + s] {
                                continue;
                            }
                            let oi = (st * n + s) * obs_size;
                            game_obs.extend_from_slice(&slot.obs[oi..oi + obs_size]);
                            let pi = (st * n + s) * 4;
                            game_pol.extend_from_slice(&slot.pol[pi..pi + 4]);
                            // Value target: equilibrium-bootstrapped (the search's root
                            // value at this state, grounded by terminal nodes within the
                            // tree) when enabled, else the flat Monte-Carlo game outcome.
                            game_z.push(if bootstrap_value {
                                slot.value[st * n + s]
                            } else {
                                terminal_sample_value(
                                    winner,
                                    s,
                                    slot.alive[final_alive_base + s],
                                    draw_value,
                                )
                            });
                        }
                    }
                    let game_sample_count = game_z.len();
                    collected += game_sample_count;
                    SELFPLAY_PROGRESS_DONE.store(collected.min(samples_per_gen), Ordering::Relaxed);
                    SELFPLAY_PROGRESS_COMPLETED_GAMES
                        .store(completed_samples.len() + 1, Ordering::Relaxed);
                    completed_samples.push(GameSamples {
                        turns: trn[g],
                        obs: game_obs,
                        pol: game_pol,
                        z: game_z,
                        samples: game_sample_count,
                    });
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
            let rp_elapsed = rp0.elapsed();
            t_rp += rp_elapsed;
            SELFPLAY_PROGRESS_CPU_RECORD_PLAY_US
                .fetch_add(duration_us(rp_elapsed), Ordering::Relaxed);
        }
        let (out_obs, out_pol, out_z, balanced_collected, selected_length_buckets) =
            balanced_sample_output(&mut completed_samples, samples_per_gen, obs_size, &mut rng);
        Ok((
            out_obs,
            out_pol,
            out_z,
            balanced_collected,
            skipped_short_draw_games,
            skipped_short_draw_samples,
            recorded_games_json,
            completed_games_json,
            recorded_game_candidates,
            selected_length_buckets,
            SelfPlayState {
                board,
                num_snakes: n,
                count,
                boards,
                slots,
                turns,
                rng,
                completed_samples: Vec::new(),
                collected: 0,
                skipped_short_draw_games: 0,
                skipped_short_draw_samples: 0,
                recorded_games_json: Vec::new(),
                completed_games_json: Vec::new(),
                recorded_game_candidates: 0,
            },
            t_recv.as_secs_f64(),
            t_mcts.as_secs_f64(),
            t_rp.as_secs_f64(),
        ))
    };
    // Release the GIL for the whole CPU self-play loop (pure Rust; the GPU net
    // runs in its own thread). Otherwise this multi-minute call would starve the
    // in-process dashboard's uvicorn thread, freezing the live UI.
    let run_res = py.allow_threads(run);
    let _ = tx_req.send(None); // stop the GPU thread
    drop(tx_req);
    let join = gpu.join();
    SELFPLAY_PROGRESS_ACTIVE.store(false, Ordering::Relaxed);
    let (
        out_obs,
        out_pol,
        out_z,
        collected,
        skipped_short_draw_games,
        skipped_short_draw_samples,
        recorded_games_json,
        completed_games_json,
        recorded_game_candidates,
        selected_length_buckets,
        next_state,
        cpu_recv_wait_seconds,
        cpu_mcts_seconds,
        cpu_record_play_seconds,
    ) = run_res.map_err(PyValueError::new_err)?;
    if let Some(path) = state_path.as_deref() {
        save_selfplay_state_to_path(path, &next_state).map_err(PyValueError::new_err)?;
    }
    if let Some(id) = state_id {
        selfplay_states()
            .lock()
            .map_err(|_| PyValueError::new_err("self-play state lock poisoned"))?
            .insert(id, next_state);
    }
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
    stats.set_item(
        "inference_per_sec",
        inference_count as f64 / gpu_forward_seconds.max(1e-9),
    )?;
    stats.set_item("gpu_forward_seconds", gpu_forward_seconds)?;
    stats.set_item("gpu_idle_seconds", gpu_idle_seconds)?;
    stats.set_item("gpu_busy_pct", 100.0 * gpu_forward_seconds / gpu_total)?;
    let gpu_request_count = SELFPLAY_PROGRESS_GPU_REQUESTS.load(Ordering::Relaxed);
    let gpu_rows = SELFPLAY_PROGRESS_GPU_ROWS.load(Ordering::Relaxed);
    let batch_max_rows = SELFPLAY_PROGRESS_BATCH_MAX_ROWS.load(Ordering::Relaxed);
    stats.set_item("gpu_request_count", gpu_request_count)?;
    stats.set_item("gpu_rows", gpu_rows)?;
    stats.set_item(
        "gpu_request_rows_avg",
        gpu_rows as f64 / gpu_request_count.max(1) as f64,
    )?;
    stats.set_item("gpu_request_rows_max", batch_max_rows)?;
    stats.set_item("cpu_recv_wait_seconds", cpu_recv_wait_seconds)?;
    stats.set_item("cpu_mcts_seconds", cpu_mcts_seconds)?;
    stats.set_item("cpu_record_play_seconds", cpu_record_play_seconds)?;
    stats.set_item("skipped_short_draw_games", skipped_short_draw_games)?;
    stats.set_item("skipped_short_draw_samples", skipped_short_draw_samples)?;
    stats.set_item("draw_value", draw_value)?;
    stats.set_item("bootstrap_value", bootstrap_value)?;
    stats.set_item("skip_short_draw_turns", skip_short_draw_turns)?;
    stats.set_item("recorded_game_candidates", recorded_game_candidates)?;
    stats.set_item("length_balanced_samples", true)?;
    stats.set_item("continuous_selfplay", state_id.is_some())?;
    stats.set_item("selected_length_buckets", selected_length_buckets)?;
    stats.set_item("recorded_games_json", recorded_games_json)?;
    stats.set_item("completed_games_json", completed_games_json)?;
    Ok((obs_arr, pol_arr, z_arr, stats))
}

#[cfg(test)]
mod tests {
    use super::{
        load_selfplay_state_from_path, mask_obvious_immediate_deaths, new_selfplay_state,
        obvious_immediate_death, save_selfplay_state_to_path, terminal_sample_value, GameSamples,
        SelfPlayState, SerSelfPlayStateV1,
    };
    use rand::RngCore;
    use snek_core::{Board, Move, Point};

    #[test]
    fn terminal_draw_only_rewards_final_survivors() {
        let draw = -0.25;
        assert_eq!(terminal_sample_value(None, 0, false, draw), -1.0);
        assert_eq!(terminal_sample_value(None, 1, true, draw), draw);
        assert_eq!(terminal_sample_value(None, 2, true, draw), draw);
        assert_eq!(terminal_sample_value(None, 3, false, draw), -1.0);
    }

    #[test]
    fn winner_gets_win_and_all_others_lose() {
        let draw = -0.25;
        assert_eq!(terminal_sample_value(Some(2), 2, true, draw), 1.0);
        assert_eq!(terminal_sample_value(Some(2), 1, true, draw), -1.0);
        assert_eq!(terminal_sample_value(Some(2), 3, false, draw), -1.0);
    }

    #[test]
    fn obvious_immediate_death_masks_wall_and_self_collision() {
        let mut board = Board::new(5, 5);
        board.add_snake(&[
            Point::new(1, 1),
            Point::new(1, 2),
            Point::new(2, 2),
            Point::new(2, 1),
        ]);
        assert!(obvious_immediate_death(&board, 0, Move::Up));
        assert!(!obvious_immediate_death(&board, 0, Move::Left));

        let masked = mask_obvious_immediate_deaths(&board, 0, &[0.25, 0.25, 0.25, 0.25]);
        assert_eq!(masked[Move::Up.index()], 0.0);
        assert!(masked[Move::Left.index()] > 0.0);
        assert!((masked.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn obvious_immediate_death_allows_own_tail_chase() {
        let mut board = Board::new(5, 5);
        board.add_snake(&[
            Point::new(1, 1),
            Point::new(1, 2),
            Point::new(0, 2),
            Point::new(0, 1),
        ]);
        assert!(!obvious_immediate_death(&board, 0, Move::Left));
    }

    #[test]
    fn obvious_immediate_death_masks_opponent_interior_but_not_tail() {
        let mut board = Board::new(7, 7);
        board.add_snake(&[Point::new(2, 2), Point::new(2, 1)]);
        board.add_snake(&[
            Point::new(5, 2),
            Point::new(4, 2),
            Point::new(3, 2),
            Point::new(3, 1),
        ]);
        assert!(obvious_immediate_death(&board, 0, Move::Right));

        let mut tail_board = Board::new(7, 7);
        tail_board.add_snake(&[Point::new(2, 2), Point::new(2, 1)]);
        tail_board.add_snake(&[
            Point::new(5, 2),
            Point::new(5, 1),
            Point::new(4, 1),
            Point::new(3, 1),
            Point::new(3, 2),
        ]);
        assert!(!obvious_immediate_death(&tail_board, 0, Move::Right));
    }

    fn assert_selfplay_state_eq(a: &SelfPlayState, b: &SelfPlayState) {
        assert_eq!(a.board, b.board);
        assert_eq!(a.num_snakes, b.num_snakes);
        assert_eq!(a.count, b.count);
        assert_eq!(a.turns, b.turns);
        assert_eq!(a.boards.len(), b.boards.len());
        for (ba, bb) in a.boards.iter().flatten().zip(b.boards.iter().flatten()) {
            assert_eq!(ba.width, bb.width);
            assert_eq!(ba.height, bb.height);
            assert_eq!(ba.turn, bb.turn);
            assert_eq!(ba.food, bb.food);
            assert_eq!(ba.hazards, bb.hazards);
            assert_eq!(ba.hazard_damage, bb.hazard_damage);
            assert_eq!(ba.min_food, bb.min_food);
            assert_eq!(ba.food_spawn_chance, bb.food_spawn_chance);
            assert_eq!(ba.snakes.len(), bb.snakes.len());
            for (sa, sb) in ba.snakes.iter().zip(bb.snakes.iter()) {
                assert_eq!(sa.health, sb.health);
                assert_eq!(sa.eliminated, sb.eliminated);
                assert_eq!(
                    sa.body.iter().collect::<Vec<_>>(),
                    sb.body.iter().collect::<Vec<_>>()
                );
            }
        }
        assert_eq!(a.slots.len(), b.slots.len());
        for (sa, sb) in a.slots.iter().flatten().zip(b.slots.iter().flatten()) {
            assert_eq!(sa.obs, sb.obs);
            assert_eq!(sa.pol, sb.pol);
            assert_eq!(sa.value, sb.value);
            assert_eq!(sa.alive, sb.alive);
            assert_eq!(sa.frames, sb.frames);
            assert_eq!(sa.steps, sb.steps);
        }
        assert_eq!(a.completed_samples.len(), b.completed_samples.len());
        for (ga, gb) in a.completed_samples.iter().zip(b.completed_samples.iter()) {
            assert_eq!(ga.turns, gb.turns);
            assert_eq!(ga.obs, gb.obs);
            assert_eq!(ga.pol, gb.pol);
            assert_eq!(ga.z, gb.z);
            assert_eq!(ga.samples, gb.samples);
        }
        assert_eq!(a.collected, b.collected);
        assert_eq!(a.skipped_short_draw_games, b.skipped_short_draw_games);
        assert_eq!(a.skipped_short_draw_samples, b.skipped_short_draw_samples);
        assert_eq!(a.recorded_games_json, b.recorded_games_json);
        assert_eq!(a.completed_games_json, b.completed_games_json);
        assert_eq!(a.recorded_game_candidates, b.recorded_game_candidates);
        let mut ra = a.rng.clone();
        let mut rb = b.rng.clone();
        for _ in 0..16 {
            assert_eq!(ra.next_u64(), rb.next_u64());
        }
    }

    #[test]
    fn selfplay_state_round_trips_full_inflight_slots() {
        let mut state = new_selfplay_state(11, 2, 3, 2, 12345);
        state.boards[0][1].turn = 17;
        state.boards[0][1].food.push(Point::new(4, 5));
        state.boards[0][1].hazards.push(Point::new(6, 7));
        state.boards[0][1].snakes[0].health = 42;
        state.boards[0][1].snakes[1].eliminated = Some(snek_core::EliminatedCause::Collision);
        state.turns[0][1] = 17;
        state.slots[0][1].obs = vec![1.25, -2.5, 3.75];
        state.slots[0][1].pol = vec![0.1, 0.2, 0.3, 0.4];
        state.slots[0][1].value = vec![-0.5, 0.75];
        state.slots[0][1].alive = vec![true, false, true];
        state.slots[0][1].frames = vec![serde_json::json!({"turn": 17, "note": "roundtrip"})];
        state.slots[0][1].steps = 1;
        state.completed_samples = vec![GameSamples {
            turns: 12,
            obs: vec![0.25, 0.5, 0.75],
            pol: vec![0.0, 1.0, 0.0, 0.0],
            z: vec![1.0],
            samples: 1,
        }];
        state.collected = 1;
        state.skipped_short_draw_games = 2;
        state.skipped_short_draw_samples = 3;
        state.recorded_games_json = vec![serde_json::json!({"winner": 0}).to_string()];
        state.completed_games_json = vec![serde_json::json!({"turns": 12}).to_string()];
        state.recorded_game_candidates = 4;
        let _ = state.rng.next_u64();

        let path = std::env::temp_dir().join(format!(
            "snek-selfplay-state-{}-{}.bin",
            std::process::id(),
            state.rng.next_u64()
        ));
        save_selfplay_state_to_path(path.to_str().unwrap(), &state).unwrap();
        let loaded = load_selfplay_state_from_path(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).ok();
        assert_selfplay_state_eq(&state, &loaded);
    }

    #[test]
    fn selfplay_state_loads_v1_snapshots_without_completed_samples() {
        let state = new_selfplay_state(11, 4, 2, 1, 123);
        let v1 = SerSelfPlayStateV1 {
            version: 1,
            board: state.board,
            num_snakes: state.num_snakes,
            count: state.count,
            boards: state
                .boards
                .iter()
                .map(|buf| buf.iter().map(super::serialize_board).collect())
                .collect(),
            slots: state
                .slots
                .iter()
                .map(|buf| buf.iter().map(super::SerSlot::from).collect())
                .collect(),
            turns: state.turns.clone(),
            rng: state.rng.clone(),
        };
        let mut tmp_rng = state.rng.clone();
        let path = std::env::temp_dir().join(format!(
            "snek-selfplay-state-v1-{}-{}.bin",
            std::process::id(),
            tmp_rng.next_u64()
        ));
        std::fs::write(path.to_str().unwrap(), bincode::serialize(&v1).unwrap()).unwrap();
        let loaded = load_selfplay_state_from_path(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).ok();
        assert_eq!(loaded.board, state.board);
        assert_eq!(loaded.num_snakes, state.num_snakes);
        assert_eq!(loaded.count, state.count);
        assert!(loaded.completed_samples.is_empty());
        assert_eq!(loaded.collected, 0);
    }
}

#[pymodule]
fn snek(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CHANNELS", NUM_CHANNELS)?;
    m.add_class::<GameBatch>()?;
    m.add_function(wrap_pyfunction!(encode_move_request, m)?)?;
    m.add_function(wrap_pyfunction!(set_search_threads, m)?)?;
    m.add_function(wrap_pyfunction!(request_cancel, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cancel, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay_progress, m)?)?;
    m.add_function(wrap_pyfunction!(create_selfplay_state, m)?)?;
    m.add_function(wrap_pyfunction!(drop_selfplay_state, m)?)?;
    m.add_function(wrap_pyfunction!(save_selfplay_state, m)?)?;
    m.add_function(wrap_pyfunction!(load_selfplay_state, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay_state_info, m)?)?;
    m.add_function(wrap_pyfunction!(generate_selfplay, m)?)?;
    Ok(())
}
