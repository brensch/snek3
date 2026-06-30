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
//! Channel layout — KEEP IN SYNC with `python/azsnek/obs_schema.py` (v1):
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
