//! Persist the in-progress self-play boards so a resumed run continues its games
//! instead of discarding them. Boards are snapshotted through the engine's public
//! fields (no serde needed in snek-core): bodies, health, alive, food, hazards and
//! the game turn are enough to reconstruct an equivalent `Board`.

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
struct CarrySnap {
    turns: Vec<usize>,
    boards: Vec<BoardSnap>,
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

/// Write the carried boards + turn counters to `path` atomically.
pub fn save(path: &Path, boards: &[Board], turns: &[usize]) -> anyhow::Result<()> {
    let snap = CarrySnap {
        turns: turns.to_vec(),
        boards: boards.iter().map(BoardSnap::from_board).collect(),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&snap)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Load carried boards + turns, or `None` if there is no snapshot.
pub fn load(path: &Path) -> anyhow::Result<Option<(Vec<Board>, Vec<usize>)>> {
    if !path.exists() {
        return Ok(None);
    }
    let snap: CarrySnap = serde_json::from_slice(&std::fs::read(path)?)?;
    let boards = snap.boards.iter().map(BoardSnap::to_board).collect();
    Ok(Some((boards, snap.turns)))
}
