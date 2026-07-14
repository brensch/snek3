//! Board -> neural-net observation encoding.
//!
//! Planes are in **absolute board coordinates** (cell `(x, y)` -> `y*w + x`),
//! encoded from snake `me`'s perspective. The net locates `me` via the `my_head`
//! plane and aggregates globally (KataGo-style global pooling), so head-centering
//! is unnecessary and we avoid the 3.6x cost of a `2*side-1` egocentric canvas.
//!
//! Opponents are handled **permutation-invariantly** so the channel count is
//! fixed for any snake count: spatial occupancy is unioned over all opponents,
//! and per-opponent scalars (health, length-vs-me) are written at each
//! opponent's head cell. 1v1 / 3-player / 4-player FFA all share this layout.
//!
//! Channel layout — schema v1, previously mirrored by archived `azsnek/obs_schema.py`:
//!   0  my_head
//!   1  my_body              (segments excluding the head)
//!   2  my_tail_countdown    (body cells, i/len head-first: ~1 near the tail)
//!   3  my_health            (health/100, broadcast)
//!   4  my_length            (len/area, broadcast)
//!   5  opp_heads            (union)
//!   6  opp_body             (union, excluding heads)
//!   7  opp_tail_countdown   (union, i/len)
//!   8  opp_len_vs_me        ((opp_len - my_len)/area, at the opponent's head)
//!   9  opp_health           (opp health/100, at the opponent's head)
//!   10 opp_danger_heads     (opponent heads with len >= my len)
//!   11 food
//!   12 hazards
//!   13 board_mask           (1 over real board cells)
//!
//! The observation is deliberately *opponent-type agnostic*: it encodes the
//! board, not who is playing. A heuristic sparring partner in self-play (see
//! `Board::heur_mask`) is shaped into training only through the positions and
//! outcomes it produces — never through an input flag, which the net could not
//! see at deployment and would only let it compartmentalise an anti-heuristic
//! policy it can't use in the field.

use crate::Board;

pub const NUM_CHANNELS: usize = 14;

/// Observation canvas side for a board side `side`. Absolute coordinates, so the
/// canvas is exactly the board (kept as a function for call-site clarity and in
/// case a future mode pads the canvas).
#[inline]
pub const fn obs_side(side: usize) -> usize {
    side
}

/// Observation canvas height for `board`.
#[inline]
pub fn obs_h(board: &Board) -> usize {
    obs_side(board.height as usize)
}

/// Observation canvas width for `board`.
#[inline]
pub fn obs_w(board: &Board) -> usize {
    obs_side(board.width as usize)
}

/// Size in floats of one encoded observation for the given board.
#[inline]
pub fn obs_len(board: &Board) -> usize {
    NUM_CHANNELS * obs_h(board) * obs_w(board)
}

// ---- Logit-Equilibrium temperature (τ) conditioning ----
//
// The LE net is conditioned on a per-episode inverse-temperature τ by appending
// one extra constant input plane holding `τ / TEMP_SCALE`. It is a full input
// plane (not a scalar concatenated at the pooled value head) *on purpose*: the
// equilibrium **policy** target depends on τ, so τ must reach the trunk and both
// heads — a scalar tapped only into the value MLP would leave the policy blind to
// τ. `TEMP_SCALE` keeps the plane value O(0.005..0.12) for τ ∈ [0.5, 12].

/// Divisor that maps τ to its input-plane value (matches the validated Albatross
/// net: `temperature_scale = 100`).
pub const TEMP_SCALE: f32 = 100.0;

/// Channels of the τ-conditioned observation (board channels + 1 τ plane).
pub const NUM_CHANNELS_TEMP: usize = NUM_CHANNELS + 1;

/// Size in floats of one τ-conditioned observation.
#[inline]
pub fn obs_len_temp(board: &Board) -> usize {
    NUM_CHANNELS_TEMP * obs_h(board) * obs_w(board)
}

/// Encode the board from seat `me`'s perspective and append the τ plane, into
/// `out` (must be `obs_len_temp(board)` long, fully overwritten).
pub fn encode_into_temp(board: &Board, me: usize, tau: f32, out: &mut [f32]) {
    let hw = obs_h(board) * obs_w(board);
    encode_into(board, me, &mut out[..NUM_CHANNELS * hw]);
    let plane = tau / TEMP_SCALE;
    for x in out[NUM_CHANNELS * hw..NUM_CHANNELS_TEMP * hw].iter_mut() {
        *x = plane;
    }
}

/// Build a τ-conditioned batch from a batch of plain board observations: given
/// `obs14` = `rows` rows of `NUM_CHANNELS*hw` and a per-row `temp`, return `rows`
/// rows of `NUM_CHANNELS_TEMP*hw` with the τ plane appended to each. Used at
/// training time so the (already D4-augmented) board obs get τ concatenated
/// without re-encoding — and so D4 augmentation never touches the τ plane.
pub fn append_temp_planes(obs14: &[f32], temp: &[f32], hw: usize) -> Vec<f32> {
    let rows = temp.len();
    let in_row = NUM_CHANNELS * hw;
    let out_row = NUM_CHANNELS_TEMP * hw;
    debug_assert_eq!(obs14.len(), rows * in_row);
    let mut out = vec![0.0f32; rows * out_row];
    for r in 0..rows {
        out[r * out_row..r * out_row + in_row]
            .copy_from_slice(&obs14[r * in_row..(r + 1) * in_row]);
        let plane = temp[r] / TEMP_SCALE;
        for x in out[r * out_row + in_row..(r + 1) * out_row].iter_mut() {
            *x = plane;
        }
    }
    out
}

/// Encode the board from snake `me`'s perspective into `out` (absolute coords).
/// `out` must be `obs_len(board)` long and is fully overwritten.
pub fn encode_into(board: &Board, me: usize, out: &mut [f32]) {
    let w = board.width as usize;
    let h = board.height as usize;
    debug_assert_eq!(out.len(), NUM_CHANNELS * h * w);
    out.fill(0.0);

    let area = (w * h) as f32;
    let plane = |c: usize| c * h * w;
    let idx = |x: i8, y: i8| -> usize { y as usize * w + x as usize };

    let me_snake = &board.snakes[me];
    let my_len = me_snake.len();

    // 0/1/2: my head, body, tail countdown.
    if me_snake.alive() {
        let head = me_snake.head();
        if board.in_bounds(head) {
            out[plane(0) + idx(head.x, head.y)] = 1.0;
        }
        let l = my_len.max(1) as f32;
        for k in 1..my_len {
            let p = me_snake.body.get(k);
            if board.in_bounds(p) {
                let i = idx(p.x, p.y);
                out[plane(1) + i] = 1.0;
                out[plane(2) + i] = k as f32 / l; // ~1 near the tail (k = len-1)
            }
        }
    }

    // 3/4: my health and length, broadcast.
    let my_health = (me_snake.health.max(0) as f32) / 100.0;
    for c in &mut out[plane(3)..plane(4)] {
        *c = my_health;
    }
    let my_len_norm = my_len as f32 / area;
    for c in &mut out[plane(4)..plane(5)] {
        *c = my_len_norm;
    }

    // 5..=10: opponents (union occupancy + per-opponent scalars at heads).
    for (j, opp) in board.snakes.iter().enumerate() {
        if j == me || !opp.alive() {
            continue;
        }
        let opp_len = opp.len();
        let head = opp.head();
        if board.in_bounds(head) {
            let i = idx(head.x, head.y);
            out[plane(5) + i] = 1.0;
            out[plane(8) + i] = (opp_len as f32 - my_len as f32) / area;
            out[plane(9) + i] = (opp.health.max(0) as f32) / 100.0;
            if opp_len >= my_len {
                out[plane(10) + i] = 1.0;
            }
        }
        let l = opp_len.max(1) as f32;
        for k in 1..opp_len {
            let p = opp.body.get(k);
            if board.in_bounds(p) {
                let i = idx(p.x, p.y);
                out[plane(6) + i] = 1.0;
                out[plane(7) + i] = k as f32 / l;
            }
        }
    }

    // 11: food.
    for &f in &board.food {
        if board.in_bounds(f) {
            out[plane(11) + idx(f.x, f.y)] = 1.0;
        }
    }

    // 12: hazards.
    for &hz in &board.hazards {
        if board.in_bounds(hz) {
            out[plane(12) + idx(hz.x, hz.y)] = 1.0;
        }
    }

    // 13: board mask (all real cells; useful if a mode ever pads the canvas).
    for c in &mut out[plane(13)..plane(14)] {
        *c = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Point};

    #[test]
    fn temp_plane_appends_and_preserves_board() {
        let mut b = Board::new(11, 11);
        b.add_snake(&[Point::new(0, 0), Point::new(0, 1)]);
        b.add_snake(&[Point::new(10, 10), Point::new(10, 9)]);
        let hw = obs_h(&b) * obs_w(&b);
        let mut o14 = vec![0.0f32; obs_len(&b)];
        encode_into(&b, 0, &mut o14);
        let mut o15 = vec![0.0f32; obs_len_temp(&b)];
        encode_into_temp(&b, 0, 8.0, &mut o15);
        // Board channels are byte-identical to the plain encoding.
        assert_eq!(&o15[..NUM_CHANNELS * hw], &o14[..]);
        // The extra plane is a constant τ/TEMP_SCALE.
        let plane = &o15[NUM_CHANNELS * hw..NUM_CHANNELS_TEMP * hw];
        assert_eq!(plane.len(), hw);
        assert!(plane.iter().all(|&v| (v - 8.0 / TEMP_SCALE).abs() < 1e-6));
    }

    #[test]
    fn append_temp_planes_matches_encode_into_temp() {
        let mut b = Board::new(11, 11);
        b.add_snake(&[Point::new(3, 3), Point::new(3, 2)]);
        b.add_snake(&[Point::new(7, 7), Point::new(7, 6)]);
        let hw = obs_h(&b) * obs_w(&b);
        let (l14, l15) = (obs_len(&b), obs_len_temp(&b));
        let mut o14 = vec![0.0f32; 2 * l14];
        encode_into(&b, 0, &mut o14[..l14]);
        encode_into(&b, 1, &mut o14[l14..]);
        let batch = append_temp_planes(&o14, &[2.0, 9.0], hw);
        assert_eq!(batch.len(), 2 * l15);
        let mut exp0 = vec![0.0f32; l15];
        encode_into_temp(&b, 0, 2.0, &mut exp0);
        let mut exp1 = vec![0.0f32; l15];
        encode_into_temp(&b, 1, 9.0, &mut exp1);
        assert_eq!(&batch[..l15], &exp0[..]);
        assert_eq!(&batch[l15..], &exp1[..]);
    }

    #[test]
    fn absolute_coords_and_per_opponent_scalars() {
        let mut b = Board::new(11, 11);
        b.add_snake(&[Point::new(0, 0), Point::new(0, 1)]); // me, head at (0,0)
        b.add_snake(&[Point::new(10, 10), Point::new(10, 9)]); // opp, head at (10,10)
        assert_eq!(obs_side(11), 11);
        let (w, oh, ow) = (11usize, 11usize, 11usize);
        let mut out = vec![0.0f32; obs_len(&b)];
        encode_into(&b, 0, &mut out);
        let plane = |c: usize| c * oh * ow;

        // my_head (0): single cell at absolute (0,0).
        let my_head = &out[plane(0)..plane(1)];
        assert_eq!(my_head.iter().filter(|&&v| v != 0.0).count(), 1);
        assert_eq!(my_head[0 * w + 0], 1.0);

        // my_health (3) full at start; my_length (4) = 2/121.
        assert_eq!(out[plane(3)], 1.0);
        assert!((out[plane(4)] - 2.0 / 121.0).abs() < 1e-6);

        // opp head at (10,10) -> idx 120: head, danger (equal len), health, len_vs_me=0.
        let oi = 10 * w + 10;
        assert_eq!(out[plane(5) + oi], 1.0);
        assert_eq!(out[plane(10) + oi], 1.0); // opp len (2) >= my len (2)
        assert_eq!(out[plane(9) + oi], 1.0); // opp full health
        assert!(out[plane(8) + oi].abs() < 1e-6); // equal length -> 0

        // mask (13) marks every cell.
        assert_eq!(
            out[plane(13)..plane(14)]
                .iter()
                .filter(|&&v| v != 0.0)
                .count(),
            121
        );
    }
}
