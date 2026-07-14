//! Turn finished games (frame histories) into training samples.
//!
//! A training sample's observation is a pure function of a recorded frame, so a
//! finished game is materialised straight from its frames — no observations are
//! ever stored, only re-derived here on completion.

use super::rules::terminal_value;
use crate::replay::Samples;
use crate::sample::{FrameJson, GameJson};
use snek_core::{encode_into, Board, EliminatedCause, Point, MAX_SNAKES};

/// Per-seat **rank/survival** outcome in [-1, +1], zero-mean. A seat's value is
/// set by how many opponents it outlasted: the sole survivor (+1) down to the
/// first to die (-1), graded linearly, ties shared. This replaces a flat
/// winner/loser payoff (+1 / -1/(n-1)) because that gave every non-winner the
/// SAME value — so a behind snake had no gradient to stay alive, the value head
/// collapsed alive~dead, and the net walked into walls. Rewarding "die later"
/// gives survival a gradient at every position (and matches the Battlesnake
/// ladder, which scores placement, not just first).
///
/// KEPT for a future frame-debiased attempt: applied naively (le-5) it saturated
/// the value at +1 because long games make survivor alive-frames dominate the
/// labels. A fix would center/reweight the targets so they stay zero-mean over
/// the actual sample distribution, not just per game.
#[allow(dead_code)]
fn rank_outcomes(frames: &[FrameJson], n: usize) -> [f32; MAX_SNAKES] {
    let mut last_alive = [-1i64; MAX_SNAKES];
    for (t, fr) in frames.iter().enumerate() {
        for s in 0..n {
            if fr.snakes.get(s).is_some_and(|x| x.alive) {
                last_alive[s] = t as i64;
            }
        }
    }
    let mut out = [0.0f32; MAX_SNAKES];
    if n <= 1 {
        return out;
    }
    for s in 0..n {
        // How many seats this one outlasted (ties count half) → [0, n-1].
        let mut score = 0.0f32;
        for j in 0..n {
            if j == s {
                continue;
            }
            if last_alive[j] < last_alive[s] {
                score += 1.0;
            } else if last_alive[j] == last_alive[s] {
                score += 0.5;
            }
        }
        out[s] = 2.0 * score / (n as f32 - 1.0) - 1.0;
    }
    out
}

/// Does a finished game match the current board size and snake count? Frames from
/// a differently-shaped run would encode to the wrong observation, so mismatches
/// are dropped rather than fed into a shard.
pub(crate) fn game_matches_shape(g: &GameJson, board: i8, n: usize) -> bool {
    g.frames
        .first()
        .is_some_and(|f| f.width == board as i32 && f.height == board as i32 && f.snakes.len() == n)
}

/// Can a carried in-flight game be resumed under the current config? Its board
/// must be the right size, and its snake count must match — taken from the game's
/// own start frame (which held all `n` snakes) when it has one, else from the live
/// board for a game paused before recording its first frame. A mismatch would
/// encode frames to the wrong observation, so such games are dropped and replaced.
pub(super) fn in_flight_matches_shape(b: &Board, rec: &[FrameJson], board: i8, n: usize) -> bool {
    if b.width != board || b.height != board {
        return false;
    }
    match rec.first() {
        Some(f) => f.width == board as i32 && f.height == board as i32 && f.snakes.len() == n,
        None => b.snakes.len() == n,
    }
}

/// Number of training samples a finished game contributes: one per (pre-terminal
/// frame, snake alive at that frame). Mirrors [`materialize_game`]'s emission so
/// the generation's sample gate stays exact without building any observations.
pub(super) fn game_sample_count(g: &GameJson, n: usize) -> usize {
    let steps = g.frames.len().saturating_sub(1); // exclude the terminal frame
    let mut count = 0;
    for f in g.frames.iter().take(steps) {
        for s in 0..n {
            // Heuristic sparring seats are never trained on (we learn to beat
            // them, not imitate them), so they contribute no samples.
            if (g.heur_mask >> s) & 1 == 1 {
                continue;
            }
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
pub(crate) fn materialize_game(
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
    let mut obs = vec![0.0f32; obs_len];
    for f in g.frames.iter().take(steps) {
        let bd = board_from_frame(f);
        for s in 0..n {
            // Never emit a training sample for a heuristic sparring seat (learn
            // to beat it, not imitate it). The opponent's influence reaches the
            // net purely through the board positions and game outcome it caused,
            // never through an opponent-type input.
            if (g.heur_mask >> s) & 1 == 1 {
                continue;
            }
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
            out.z.push(terminal_value(winner, s, n, draw_value));
            out.turn.push(f.turn);
        }
    }
}

/// Materialise a **Logit-Equilibrium** game. Same per-(frame, alive net seat)
/// emission as [`materialize_game`], but:
///  * policy target = the recorded LE root policy (`SnakeJson::policy`) — already
///    a mixed distribution, not a one-hot/visit target;
///  * value target = `(1-λ)·LE_root_value + λ·game_outcome` where the LE root
///    value is the recorded bootstrap (`SnakeJson::value`) and λ =
///    `cfg.le_outcome_weight` (see the value-target fork in `docs/le-overhaul.md`);
///  * each sample carries the game's episode τ in `Samples::temp` for the
///    τ-conditioned net.
pub(crate) fn materialize_le_game(
    g: &GameJson,
    n: usize,
    cfg: &crate::config::RunConfig,
    out: &mut Samples,
) {
    let steps = g.frames.len().saturating_sub(1); // exclude the terminal frame
    if steps == 0 {
        return;
    }
    let winner = g.winner.map(|w| w as usize);
    let lambda = cfg.le_outcome_weight.clamp(0.0, 1.0);
    let draw_value = cfg.draw_value;
    let bd0 = board_from_frame(&g.frames[0]);
    let obs_len = snek_core::NUM_CHANNELS * (bd0.height as usize) * (bd0.width as usize);
    let mut obs = vec![0.0f32; obs_len];
    for f in g.frames.iter().take(steps) {
        let bd = board_from_frame(f);
        for s in 0..n {
            // Heuristic sparring seats are never trained on (Stage B), same as AZ.
            if (g.heur_mask >> s) & 1 == 1 {
                continue;
            }
            let Some(sn) = f.snakes.get(s) else { continue };
            if !sn.alive {
                continue;
            }
            encode_into(&bd, s, &mut obs);
            out.obs.extend_from_slice(&obs);
            for k in 0..4 {
                out.pol.push(sn.policy.get(k).copied().unwrap_or(0.0) as f32);
            }
            // NOTE: rank/survival reward (rank_outcomes) was tried in le-5 but
            // saturated the value at the +1 rail (long games => winner's many
            // alive-frames dominate the labels => value shoved high, tanh stuck).
            // Reverted to flat zero-sum here; keep rank_outcomes for a future
            // frame-debiased attempt. The suicide mask (le_candidates) stays.
            let outcome = terminal_value(winner, s, n, draw_value);
            let bootstrap = sn.value as f32;
            out.z.push((1.0 - lambda) * bootstrap + lambda * outcome);
            out.temp.push(g.temp);
            out.turn.push(f.turn);
        }
    }
}
