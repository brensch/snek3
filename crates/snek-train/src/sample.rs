//! Recorded self-play sample games: the small, browsable replays the viewer
//! renders. Self-play discards per-move detail once it has extracted training
//! targets, so for a handful of games each generation we additionally capture
//! full frames (bodies, the search policy, the value head, the move played) and
//! write them to `runs/<id>/games/gen_NNNN.json`.
//!
//! The on-disk schema is deliberately the same one the archived Python trainer
//! wrote, so old runs and new runs are viewable through one code path. These
//! structs are the single source of truth for that schema — both the writer
//! (here) and the API reader deserialize into them.

use serde::{Deserialize, Serialize};
use snek_core::{Board, Move};
use std::path::Path;

/// A board coordinate as the `[x, y]` pair used on disk.
pub type Coord = [i32; 2];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameFileJson {
    pub gen: u32,
    pub games: Vec<GameJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameJson {
    pub frames: Vec<FrameJson>,
    /// Winning snake index, or `None` for a draw. Serialized as `null` on draws.
    #[serde(default)]
    pub winner: Option<i32>,
    #[serde(default)]
    pub num_turns: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameJson {
    pub turn: u32,
    pub width: i32,
    pub height: i32,
    pub food: Vec<Coord>,
    #[serde(default)]
    pub hazards: Vec<Coord>,
    pub snakes: Vec<SnakeJson>,
}

// `#[serde(default)]` on the search-readout fields so older recordings (which
// omitted e.g. `chosen_move` for already-eliminated snakes) still deserialize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnakeJson {
    pub alive: bool,
    #[serde(default)]
    pub body: Vec<Coord>,
    #[serde(default)]
    pub health: i32,
    #[serde(default)]
    pub chosen_move: u32,
    /// Search visit-count policy over the four moves.
    #[serde(default)]
    pub policy: Vec<f64>,
    /// Masked policy actually sampled from when choosing the move.
    #[serde(default)]
    pub play_policy: Vec<f64>,
    #[serde(default)]
    pub value: f64,
}

/// Capture the pre-step state of one board as a recordable frame. `policy` and
/// `values` are the search outputs for this board (length `n*4` and `n`);
/// `play_pols[s]` is the masked play distribution and `actions[s]` the move
/// chosen for snake `s`.
pub fn frame_from_board(
    board: &Board,
    n: usize,
    policy: &[f32],
    values: &[f32],
    play_pols: &[[f32; 4]],
    actions: &[Move],
) -> FrameJson {
    let snakes = (0..n)
        .map(|s| {
            let snake = &board.snakes[s];
            SnakeJson {
                alive: snake.alive(),
                body: snake.body.iter().map(|p| [p.x as i32, p.y as i32]).collect(),
                health: snake.health as i32,
                chosen_move: actions[s].index() as u32,
                policy: policy[s * 4..s * 4 + 4].iter().map(|&v| v as f64).collect(),
                play_policy: play_pols[s].iter().map(|&v| v as f64).collect(),
                value: values[s] as f64,
            }
        })
        .collect();
    FrameJson {
        turn: board.turn,
        width: board.width as i32,
        height: board.height as i32,
        food: board.food.iter().map(|p| [p.x as i32, p.y as i32]).collect(),
        hazards: board.hazards.iter().map(|p| [p.x as i32, p.y as i32]).collect(),
        snakes,
    }
}

/// Write one generation's recorded games and prune old generations so the
/// `games/` directory keeps at most `keep` files.
pub fn write_generation(
    games_dir: &Path,
    gen: u32,
    games: Vec<GameJson>,
    keep: usize,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(games_dir)?;
    let path = games_dir.join(format!("gen_{gen:04}.json"));
    let payload = GameFileJson { gen, games };
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&payload)?)?;
    std::fs::rename(&tmp, &path)?;
    prune(games_dir, keep);
    Ok(())
}

fn prune(games_dir: &Path, keep: usize) {
    let mut files: Vec<_> = match std::fs::read_dir(games_dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map(|e| e == "json").unwrap_or(false)
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("gen_"))
                        .unwrap_or(false)
            })
            .collect(),
        Err(_) => return,
    };
    if files.len() <= keep {
        return;
    }
    files.sort();
    let remove = files.len() - keep;
    for path in files.into_iter().take(remove) {
        let _ = std::fs::remove_file(path);
    }
}
