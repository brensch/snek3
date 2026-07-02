//! Persist the whole in-progress self-play session so a paused run resumes
//! seamlessly: the in-flight boards, their full frame histories, and every game
//! finished so far in the current (not-yet-committed) generation.
//!
//! Boards are snapshotted through snek-core's public fields (the engine carries
//! no serde); frames and finished games already serialize — they share the
//! dashboard schema in [`crate::sample`]. Because a training sample's observation
//! is derived from its frame, persisting frames alone is enough to reconstruct
//! the generation's accumulated samples on resume, so nothing bulky (no raw
//! observations) is written here.
//!
//! The file keeps its historical name (`selfplay.json`) and the frame/finished
//! fields default when absent, so an older snapshot (boards + turns only) still
//! loads — its games simply restart on the first resume.

use crate::sample::{FrameJson, GameJson};
use crate::selfplay::SelfPlayState;
use serde::{Deserialize, Serialize};
use snek_core::{Board, EliminatedCause, Point};
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct SnakeSnap {
    body: Vec<[i8; 2]>,
    health: i16,
    alive: bool,
}

#[derive(Serialize, Deserialize)]
struct BoardSnap {
    width: i8,
    height: i8,
    turn: u32,
    snakes: Vec<SnakeSnap>,
    food: Vec<[i8; 2]>,
    hazards: Vec<[i8; 2]>,
}

#[derive(Serialize, Deserialize, Default)]
struct SessionSnap {
    turns: Vec<usize>,
    boards: Vec<BoardSnap>,
    /// In-flight per-game frame histories (parallel to `boards`).
    #[serde(default)]
    rec: Vec<Vec<FrameJson>>,
    /// Games finished in the current generation, awaiting the sample target.
    #[serde(default)]
    finished: Vec<GameJson>,
}

fn pts(ps: &[Point]) -> Vec<[i8; 2]> {
    ps.iter().map(|p| [p.x, p.y]).collect()
}

fn to_points(ps: &[[i8; 2]]) -> Vec<Point> {
    ps.iter().map(|&[x, y]| Point::new(x, y)).collect()
}

impl BoardSnap {
    fn from_board(b: &Board) -> Self {
        Self {
            width: b.width,
            height: b.height,
            turn: b.turn,
            snakes: b
                .snakes
                .iter()
                .map(|s| SnakeSnap {
                    body: s.body.iter().map(|p| [p.x, p.y]).collect(),
                    health: s.health,
                    alive: s.alive(),
                })
                .collect(),
            food: pts(&b.food),
            hazards: pts(&b.hazards),
        }
    }

    fn to_board(&self) -> Board {
        let mut board = Board::new(self.width, self.height);
        board.turn = self.turn;
        for snake in &self.snakes {
            board.add_snake(&to_points(&snake.body));
            let idx = board.snakes.len() - 1;
            board.snakes[idx].health = snake.health;
            // Exact elimination cause is irrelevant to continued play; only
            // alive/dead matters (alive() == eliminated.is_none()).
            board.snakes[idx].eliminated = if snake.alive {
                None
            } else {
                Some(EliminatedCause::Collision)
            };
        }
        board.food = to_points(&self.food);
        board.hazards = to_points(&self.hazards);
        board
    }
}

/// Write the whole self-play session to `path` atomically.
pub fn save(path: &Path, state: &SelfPlayState) -> anyhow::Result<()> {
    let snap = SessionSnap {
        turns: state.turns.clone(),
        boards: state.boards.iter().map(BoardSnap::from_board).collect(),
        rec: state.rec.clone(),
        finished: state.finished.clone(),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&snap)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Load a saved session, or `None` if there is no snapshot.
pub fn load(path: &Path) -> anyhow::Result<Option<SelfPlayState>> {
    if !path.exists() {
        return Ok(None);
    }
    let snap: SessionSnap = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(Some(SelfPlayState {
        boards: snap.boards.iter().map(BoardSnap::to_board).collect(),
        turns: snap.turns,
        rec: snap.rec,
        finished: snap.finished,
    }))
}
