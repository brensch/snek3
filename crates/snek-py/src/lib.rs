//! Python bindings (`snek`) over `snek-core`.
//!
//! Exposes a vectorised `GameBatch` for self-play plus helpers for the server.
//! Observations are returned as zero-copy numpy arrays.

use numpy::ndarray::Array;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArray4, PyArray5, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use snek_core::{encode_into, standard_start, Board, Move, NUM_CHANNELS};
use snek_search::Forest;

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
        let mut moves: Vec<Move> = vec![Move::Up; self.num_snakes];
        for (g, board) in self.boards.iter_mut().enumerate() {
            if board.is_terminal() {
                continue;
            }
            for s in 0..self.num_snakes {
                moves[s] = move_from_u8(a[[g, s]]);
            }
            board.step(&moves);
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
        for (g, board) in self.boards.iter().enumerate() {
            for s in 0..n {
                let base = (g * n + s) * per_obs;
                encode_one(board, s, &mut flat[base..base + per_obs]);
            }
        }
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
        for (g, board) in self.boards.iter().enumerate() {
            for s in 0..n {
                flat[g * n + s] = board.snakes[s].alive() as u8;
            }
        }
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
        let policy = forest.backup(v, tau, iters);
        Array::from_shape_vec((self.boards.len(), self.num_snakes, 4), policy)
            .map(|a| a.into_pyarray_bound(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
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

#[pymodule]
fn snek(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CHANNELS", NUM_CHANNELS)?;
    m.add_class::<GameBatch>()?;
    m.add_function(wrap_pyfunction!(encode_move_request, m)?)?;
    Ok(())
}
