//! Egocentric observation encoding: turn a [`Board`] into stacked feature
//! planes from one snake's point of view, with that snake's head fixed at the
//! centre of the canvas. Generic over snake count, so the same layout serves
//! duel and FFA.
//!
//! The canvas is `(2*width-1) x (2*height-1)`: large enough that, with the head
//! at the centre, every board cell maps into the canvas regardless of where the
//! head is. Off-board canvas cells stay zero, and the board-mask channel marks
//! which canvas cells correspond to real board cells (essential now that most of
//! the canvas is off-board). Head-centring makes spatial patterns
//! translation-invariant relative to the head, so the net learns a pattern once
//! instead of relearning it at every board position.
//!
//! Channel layout (`NUM_CHANNELS` planes, each `obs_h * obs_w`):
//! 0. my head (always the canvas centre)
//! 1. my body (segments excluding the head)
//! 2. my health, broadcast to every cell (normalized to [0, 1])
//! 3. opponents' heads (union over all live opponents)
//! 4. opponents' bodies (union, excluding their heads)
//! 5. opponents' heads belonging to snakes at least as long as me
//!    (i.e. snakes that would win or tie a head-to-head — a danger map)
//! 6. food
//! 7. hazards
//! 8. board mask (ones over canvas cells that are real board cells)

use crate::Board;

pub const NUM_CHANNELS: usize = 9;

/// Egocentric canvas side for a board side `side` (head centred): `2*side - 1`.
#[inline]
pub const fn obs_side(side: usize) -> usize {
    2 * side - 1
}

/// Egocentric canvas height for `board`.
#[inline]
pub fn obs_h(board: &Board) -> usize {
    obs_side(board.height as usize)
}

/// Egocentric canvas width for `board`.
#[inline]
pub fn obs_w(board: &Board) -> usize {
    obs_side(board.width as usize)
}

/// Size in floats of one encoded observation for the given board.
#[inline]
pub fn obs_len(board: &Board) -> usize {
    NUM_CHANNELS * obs_h(board) * obs_w(board)
}

/// Encode the board from snake `me`'s perspective into `out`, head-centred.
/// `out` must be `obs_len(board)` long and is fully overwritten.
pub fn encode_into(board: &Board, me: usize, out: &mut [f32]) {
    let w = board.width as usize;
    let h = board.height as usize;
    let ow = obs_side(w);
    let oh = obs_side(h);
    debug_assert_eq!(out.len(), NUM_CHANNELS * oh * ow);
    out.fill(0.0);

    let me_snake = &board.snakes[me];
    let my_len = me_snake.len();

    // Canvas centre: my head sits here. Offsets are relative to the head.
    let head = me_snake.head();
    let (hx, hy) = (head.x as i32, head.y as i32);
    let (cx0, cy0) = ((w - 1) as i32, (h - 1) as i32); // centre index

    let plane = |c: usize| c * oh * ow;
    // Map a board point to a canvas linear index relative to the head, or None
    // if it falls outside the canvas (only possible for a stale/dead head).
    let cell = |x: i8, y: i8| -> Option<usize> {
        let cx = x as i32 - hx + cx0;
        let cy = y as i32 - hy + cy0;
        if cx < 0 || cy < 0 || cx >= ow as i32 || cy >= oh as i32 {
            None
        } else {
            Some(cy as usize * ow + cx as usize)
        }
    };

    // Channel 0/1: my head (centre) and body.
    if me_snake.alive() {
        if let Some(i) = cell(head.x, head.y) {
            out[plane(0) + i] = 1.0;
        }
        for k in 1..me_snake.len() {
            let p = me_snake.body.get(k);
            if board.in_bounds(p) {
                if let Some(i) = cell(p.x, p.y) {
                    out[plane(1) + i] = 1.0;
                }
            }
        }
    }

    // Channel 2: my health broadcast (global scalar; fill whole plane).
    let health_norm = (me_snake.health.max(0) as f32) / 100.0;
    for c in &mut out[plane(2)..plane(3)] {
        *c = health_norm;
    }

    // Channels 3/4/5: opponents.
    for (j, opp) in board.snakes.iter().enumerate() {
        if j == me || !opp.alive() {
            continue;
        }
        let oh_head = opp.head();
        if board.in_bounds(oh_head) {
            if let Some(i) = cell(oh_head.x, oh_head.y) {
                out[plane(3) + i] = 1.0;
                if opp.len() >= my_len {
                    out[plane(5) + i] = 1.0;
                }
            }
        }
        for k in 1..opp.len() {
            let p = opp.body.get(k);
            if board.in_bounds(p) {
                if let Some(i) = cell(p.x, p.y) {
                    out[plane(4) + i] = 1.0;
                }
            }
        }
    }

    // Channel 6: food.
    for &f in &board.food {
        if board.in_bounds(f) {
            if let Some(i) = cell(f.x, f.y) {
                out[plane(6) + i] = 1.0;
            }
        }
    }

    // Channel 7: hazards.
    for &hz in &board.hazards {
        if board.in_bounds(hz) {
            if let Some(i) = cell(hz.x, hz.y) {
                out[plane(7) + i] = 1.0;
            }
        }
    }

    // Channel 8: board mask — canvas cells that are real board cells.
    for by in 0..h as i8 {
        for bx in 0..w as i8 {
            if let Some(i) = cell(bx, by) {
                out[plane(8) + i] = 1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Point};

    #[test]
    fn head_is_centred_and_mask_counts_board_cells() {
        let mut b = Board::new(11, 11);
        b.add_snake(&[Point::new(0, 0), Point::new(0, 1)]); // me in a corner
        b.add_snake(&[Point::new(10, 10), Point::new(10, 9)]);
        let ow = obs_side(11);
        let oh = obs_side(11);
        assert_eq!(ow, 21);
        let mut out = vec![0.0f32; obs_len(&b)];
        encode_into(&b, 0, &mut out);

        // Head plane (0): exactly one cell set, at the canvas centre (10,10).
        let plane = |c: usize| c * oh * ow;
        let head_plane = &out[plane(0)..plane(1)];
        let set: Vec<usize> = head_plane
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(set, vec![10 * ow + 10], "head must be the single centre cell");

        // Mask plane (8): exactly width*height real cells marked.
        let mask = &out[plane(8)..plane(9)];
        let marked = mask.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(marked, 11 * 11, "mask marks exactly the real board cells");

        // The opponent in the opposite corner is 10 away in each axis, so it
        // lands at canvas (centre+10, centre+10) = (20,20), still in-canvas.
        let opp_head = &out[plane(3)..plane(4)];
        assert_eq!(
            opp_head.iter().filter(|&&v| v != 0.0).count(),
            1,
            "opponent head present and in-canvas"
        );
        assert_eq!(opp_head[20 * ow + 20], 1.0);
    }
}
