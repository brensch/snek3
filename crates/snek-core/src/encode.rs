//! Egocentric observation encoding: turn a [`Board`] into stacked feature
//! planes from one snake's point of view. Generic over snake count, so the same
//! layout serves duel and FFA.
//!
//! Channel layout (`NUM_CHANNELS` planes, each `height * width`):
//! 0. my head
//! 1. my body (segments excluding the head)
//! 2. my health, broadcast to every cell (normalized to [0, 1])
//! 3. opponents' heads (union over all live opponents)
//! 4. opponents' bodies (union, excluding their heads)
//! 5. opponents' heads belonging to snakes at least as long as me
//!    (i.e. snakes that would win or tie a head-to-head — a danger map)
//! 6. food
//! 7. hazards
//! 8. board mask (all ones over valid cells; helps convs sense the border)

use crate::Board;

pub const NUM_CHANNELS: usize = 9;

/// Size in floats of one encoded observation for the given board.
#[inline]
pub fn obs_len(board: &Board) -> usize {
    NUM_CHANNELS * board.height as usize * board.width as usize
}

/// Encode the board from snake `me`'s perspective into `out`, which must be
/// `obs_len(board)` long. `out` is fully overwritten (cleared then filled).
pub fn encode_into(board: &Board, me: usize, out: &mut [f32]) {
    let w = board.width as usize;
    let h = board.height as usize;
    debug_assert_eq!(out.len(), NUM_CHANNELS * h * w);
    out.fill(0.0);

    let plane = |c: usize| c * h * w;
    let at = |x: i8, y: i8| (y as usize) * w + (x as usize);

    let me_snake = &board.snakes[me];
    let my_len = me_snake.len();

    // Channel 0/1: my head and body.
    if me_snake.alive() {
        let head = me_snake.head();
        if board.in_bounds(head) {
            out[plane(0) + at(head.x, head.y)] = 1.0;
        }
        for i in 1..me_snake.len() {
            let p = me_snake.body.get(i);
            if board.in_bounds(p) {
                out[plane(1) + at(p.x, p.y)] = 1.0;
            }
        }
    }

    // Channel 2: my health broadcast.
    let health_norm = (me_snake.health.max(0) as f32) / 100.0;
    for cell in &mut out[plane(2)..plane(3)] {
        *cell = health_norm;
    }

    // Channels 3/4/5: opponents.
    for (j, opp) in board.snakes.iter().enumerate() {
        if j == me || !opp.alive() {
            continue;
        }
        let head = opp.head();
        if board.in_bounds(head) {
            out[plane(3) + at(head.x, head.y)] = 1.0;
            if opp.len() >= my_len {
                out[plane(5) + at(head.x, head.y)] = 1.0;
            }
        }
        for i in 1..opp.len() {
            let p = opp.body.get(i);
            if board.in_bounds(p) {
                out[plane(4) + at(p.x, p.y)] = 1.0;
            }
        }
    }

    // Channel 6: food.
    for &f in &board.food {
        if board.in_bounds(f) {
            out[plane(6) + at(f.x, f.y)] = 1.0;
        }
    }

    // Channel 7: hazards.
    for &hz in &board.hazards {
        if board.in_bounds(hz) {
            out[plane(7) + at(hz.x, hz.y)] = 1.0;
        }
    }

    // Channel 8: board mask (all valid cells).
    for cell in &mut out[plane(8)..plane(9)] {
        *cell = 1.0;
    }
}
