//! Turn finished games (frame histories) into training samples.
//!
//! A training sample's observation is a pure function of a recorded frame, so a
//! finished game is materialised straight from its frames — no observations are
//! ever stored, only re-derived here on completion.

use super::rules::terminal_value;
use crate::replay::Samples;
use crate::sample::{FrameJson, GameJson};
use snek_core::{encode_into, Board, EliminatedCause, Point};

/// Does a finished game match the current board size and snake count? Frames from
/// a differently-shaped run would encode to the wrong observation, so mismatches
/// are dropped rather than fed into a shard.
pub(super) fn game_matches_shape(g: &GameJson, board: i8, n: usize) -> bool {
    g.frames
        .first()
        .is_some_and(|f| f.width == board as i32 && f.height == board as i32 && f.snakes.len() == n)
}

/// Number of training samples a finished game contributes: one per (pre-terminal
/// frame, snake alive at that frame). Mirrors [`materialize_game`]'s emission so
/// the generation's sample gate stays exact without building any observations.
pub(super) fn game_sample_count(g: &GameJson, n: usize) -> usize {
    let steps = g.frames.len().saturating_sub(1); // exclude the terminal frame
    let mut count = 0;
    for f in g.frames.iter().take(steps) {
        for s in 0..n {
            if f.snakes.get(s).is_some_and(|x| x.alive) {
                count += 1;
            }
        }
    }
    count
}

/// Reconstruct a [`Board`] from a recorded frame, faithfully enough to re-encode
/// it. Only the fields [`encode_into`] reads are restored (bodies, health, alive,
/// food, hazards, dimensions); the elimination *cause* is irrelevant to encoding.
fn board_from_frame(f: &FrameJson) -> Board {
    let mut b = Board::new(f.width as i8, f.height as i8);
    b.turn = f.turn;
    for sn in &f.snakes {
        let body: Vec<Point> = sn
            .body
            .iter()
            .map(|c| Point::new(c[0] as i8, c[1] as i8))
            .collect();
        b.add_snake(&body);
        let idx = b.snakes.len() - 1;
        b.snakes[idx].health = sn.health as i16;
        b.snakes[idx].eliminated = if sn.alive {
            None
        } else {
            Some(EliminatedCause::Collision)
        };
    }
    b.food = f
        .food
        .iter()
        .map(|c| Point::new(c[0] as i8, c[1] as i8))
        .collect();
    b.hazards = f
        .hazards
        .iter()
        .map(|c| Point::new(c[0] as i8, c[1] as i8))
        .collect();
    b
}

/// Materialise one finished game's frames into training samples: one sample per
/// (pre-terminal frame, snake alive at that frame). Observations are re-encoded
/// from the frame; the value target is the exact terminal value derived from the
/// game's outcome and the snake's final alive state.
pub(super) fn materialize_game(
    g: &GameJson,
    n: usize,
    obs_len: usize,
    draw_value: f32,
    out: &mut Samples,
) {
    let steps = g.frames.len().saturating_sub(1); // exclude the terminal frame
    if steps == 0 {
        return;
    }
    let winner = g.winner.map(|w| w as usize);
    let last = &g.frames[steps - 1]; // last pre-terminal frame: final alive mask
    let mut obs = vec![0.0f32; obs_len];
    for f in g.frames.iter().take(steps) {
        let bd = board_from_frame(f);
        for s in 0..n {
            let Some(sn) = f.snakes.get(s) else { continue };
            if !sn.alive {
                continue;
            }
            encode_into(&bd, s, &mut obs);
            out.obs.extend_from_slice(&obs);
            for k in 0..4 {
                out.pol
                    .push(sn.policy.get(k).copied().unwrap_or(0.0) as f32);
            }
            let alive_final = last.snakes.get(s).is_some_and(|x| x.alive);
            out.z
                .push(terminal_value(winner, s, alive_final, draw_value));
            out.turn.push(f.turn);
        }
    }
}
