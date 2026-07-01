//! Read-only run browser: turns the on-disk `runs/<id>/` layout into the
//! protobuf messages defined in `viewer.proto`. Everything here reads files the
//! trainer already writes (config.json, metrics.jsonl, trainer_state.json,
//! games/gen_*.json) and tolerates both the current Rust layout and the older
//! Python layout (meta.json / status.json), so historical runs stay viewable.

use crate::proto;
use crate::sample::GameFileJson;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Resolve a run directory, refusing anything that escapes `runs_dir`.
pub fn resolve_run(runs_dir: &Path, run_id: &str) -> Option<PathBuf> {
    if run_id.is_empty()
        || run_id.contains('/')
        || run_id.contains('\\')
        || run_id.contains("..")
    {
        return None;
    }
    let path = runs_dir.join(run_id);
    if path.is_dir() {
        Some(path)
    } else {
        None
    }
}

pub fn run_list(runs_dir: &Path, active: Option<&str>, running: bool) -> proto::RunListReply {
    let mut runs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(runs_dir) {
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let run_id = entry.file_name().to_string_lossy().to_string();
            runs.push(summary(&entry.path(), &run_id, active, running));
        }
    }
    runs.sort_by(|a, b| b.updated_unix_ms.cmp(&a.updated_unix_ms));
    proto::RunListReply { runs }
}

pub fn run_detail(
    root: &Path,
    run_id: &str,
    active: Option<&str>,
    running: bool,
) -> proto::RunDetail {
    proto::RunDetail {
        summary: Some(summary(root, run_id, active, running)),
        config_json: read_text(&root.join("config.json"))
            .or_else(|| read_text(&root.join("meta.json")))
            .unwrap_or_default(),
        metrics: metrics(root),
        game_gens: game_gens(root),
    }
}

/// Load one generation's recorded games and convert to the wire format.
pub fn game_file(root: &Path, gen: u32) -> Option<proto::GameFile> {
    let path = root.join("games").join(format!("gen_{gen:04}.json"));
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) => {
            tracing::warn!(?path, %err, "read game file failed");
            return None;
        }
    };
    let parsed: GameFileJson = match serde_json::from_str(&text) {
        Ok(parsed) => parsed,
        Err(err) => {
            tracing::warn!(?path, %err, "parse game file failed");
            return None;
        }
    };
    Some(convert_game_file(parsed))
}

fn summary(root: &Path, run_id: &str, active: Option<&str>, running: bool) -> proto::RunSummary {
    let cfg = read_json(&root.join("config.json")).or_else(|| read_json(&root.join("meta.json")));
    let st = read_json(&root.join("trainer_state.json"))
        .or_else(|| read_json(&root.join("status.json")));
    let rows = metrics(root);
    let latest = rows.last();
    let is_live = active == Some(run_id);

    proto::RunSummary {
        run_id: run_id.to_string(),
        live: is_live,
        running: is_live && running,
        generation: u32_field(st.as_ref(), "generation")
            .or_else(|| latest.map(|m| m.generation))
            .unwrap_or(0),
        total_generations: u32_field(st.as_ref(), "total_generations").unwrap_or(0),
        board: u32_field(cfg.as_ref(), "board").unwrap_or(0),
        num_snakes: u32_field(cfg.as_ref(), "num_snakes").unwrap_or(0),
        policy_loss: latest.map(|m| m.policy_loss).unwrap_or(0.0),
        value_loss: latest.map(|m| m.value_loss).unwrap_or(0.0),
        win_rate: latest.map(|m| m.win_rate).unwrap_or(0.0),
        has_win_rate: latest.map(|m| m.has_win_rate).unwrap_or(false),
        game_gen_count: game_gens(root).len() as u32,
        updated_unix_ms: mtime_ms(root),
    }
}

#[derive(Deserialize)]
struct MetricJson {
    #[serde(alias = "gen")]
    generation: Option<u32>,
    policy_loss: Option<f64>,
    value_loss: Option<f64>,
    win_rate: Option<f64>,
    completed_games: Option<u32>,
    target_entropy: Option<f64>,
    samples: Option<u32>,
    turns: Option<u32>,
    buffer: Option<u64>,
    gen_seconds: Option<f64>,
    play_seconds: Option<f64>,
    train_seconds: Option<f64>,
    inferences_per_sec: Option<f64>,
    games_per_sec: Option<f64>,
    gpu_busy_pct: Option<f64>,
    avg_game_turn: Option<f64>,
}

fn metrics(root: &Path) -> Vec<proto::MetricRow> {
    let Some(text) = read_text(&root.join("metrics.jsonl")) else {
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .enumerate()
        .filter_map(|(i, line)| {
            let row: MetricJson = serde_json::from_str(line).ok()?;
            // The trainer writes win_rate: 0.0 until real evaluation is wired,
            // so treat a non-positive rate as "not evaluated" rather than 0%.
            let win_rate = row.win_rate.filter(|v| v.is_finite() && *v > 0.0);
            Some(proto::MetricRow {
                generation: row.generation.unwrap_or(i as u32),
                policy_loss: row.policy_loss.unwrap_or(0.0),
                value_loss: row.value_loss.unwrap_or(0.0),
                win_rate: win_rate.unwrap_or(0.0),
                has_win_rate: win_rate.is_some(),
                completed_games: row.completed_games.unwrap_or(0),
                target_entropy: row.target_entropy.unwrap_or(0.0),
                samples: row.samples.unwrap_or(0),
                turns: row.turns.unwrap_or(0),
                buffer: row.buffer.unwrap_or(0),
                gen_seconds: row.gen_seconds.unwrap_or(0.0),
                play_seconds: row.play_seconds.unwrap_or(0.0),
                train_seconds: row.train_seconds.unwrap_or(0.0),
                inferences_per_sec: row.inferences_per_sec.unwrap_or(0.0),
                games_per_sec: row.games_per_sec.unwrap_or(0.0),
                gpu_busy_pct: row.gpu_busy_pct.unwrap_or(0.0),
                avg_game_turn: row.avg_game_turn.unwrap_or(0.0),
            })
        })
        .collect()
}

/// Available recorded generations, newest first. Cheap: only reads filenames.
fn game_gens(root: &Path) -> Vec<proto::GameGen> {
    let mut gens: Vec<u32> = match std::fs::read_dir(root.join("games")) {
        Ok(rd) => rd
            .flatten()
            .filter_map(|e| gen_from_name(&e.file_name().to_string_lossy()))
            .collect(),
        Err(_) => Vec::new(),
    };
    gens.sort_unstable();
    gens.dedup();
    gens.into_iter()
        .rev()
        .map(|gen| proto::GameGen { gen, num_games: 0 })
        .collect()
}

fn gen_from_name(name: &str) -> Option<u32> {
    name.strip_prefix("gen_")?.strip_suffix(".json")?.parse().ok()
}

fn convert_game_file(file: GameFileJson) -> proto::GameFile {
    proto::GameFile {
        gen: file.gen,
        config_json: file.config.map(|v| v.to_string()).unwrap_or_default(),
        games: file
            .games
            .into_iter()
            .map(|g| proto::Game {
                num_turns: if g.num_turns > 0 {
                    g.num_turns
                } else {
                    g.frames.len() as u32
                },
                winner: g.winner.unwrap_or(-1),
                frames: g
                    .frames
                    .into_iter()
                    .map(|f| proto::Frame {
                        turn: f.turn,
                        width: f.width as u32,
                        height: f.height as u32,
                        food: points(&f.food),
                        hazards: points(&f.hazards),
                        snakes: f
                            .snakes
                            .into_iter()
                            .map(|s| proto::SnakeFrame {
                                alive: s.alive,
                                body: points(&s.body),
                                health: s.health,
                                chosen_move: s.chosen_move,
                                policy: s.policy,
                                play_policy: s.play_policy,
                                value: s.value,
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn points(coords: &[[i32; 2]]) -> Vec<proto::Point> {
    coords
        .iter()
        .map(|&[x, y]| proto::Point { x, y })
        .collect()
}

fn read_text(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&read_text(path)?).ok()
}

fn u32_field(value: Option<&serde_json::Value>, key: &str) -> Option<u32> {
    value?.get(key)?.as_u64().map(|v| v as u32)
}

fn mtime_ms(root: &Path) -> i64 {
    std::fs::metadata(root)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
