//! Off-hot-path game recorder.
//!
//! The `/move` handler does **no disk I/O**: it builds a compact [`MoveRecord`]
//! from data it already computed and hands it to the recorder over an `mpsc`
//! channel. A single background thread owns the in-flight games
//! (`HashMap<game_id, GameLog>`), appends records in memory, and only touches
//! disk when a game finishes.
//!
//! A game is written once, as a single zstd-compressed JSON file
//! `<game_id>.json.zst`. Static info (ids, ruleset, board size, snake roster)
//! is stored once; per-turn records keep only what changes plus the full search
//! diagnostics. Compared to the old per-move JSONL this is ~20-25x smaller and
//! still inspectable with `zstdcat file.json.zst | jq`.
//!
//! A game is finalized when:
//!   * the server gets `/end` for it (`Finish::Complete`), or
//!   * it goes silent for longer than the idle timeout (`Finish::Incomplete`),
//!     swept by the recorder thread every `SWEEP_INTERVAL`.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};
use serde_json::Value;

/// How often the recorder thread wakes to sweep for idle games.
const SWEEP_INTERVAL: Duration = Duration::from_secs(30);

/// Diagnostic floats (priors, Q, values, policy) are network outputs far noisier
/// than 1e-4; rounding them at write time roughly halves the compressed size
/// (long f32 decimals are the bulk of the entropy). This is the dominant lever —
/// it beats any compressor swap. Done on the recorder thread, never the hot path.
#[inline]
fn round4(x: f32) -> f64 {
    (x as f64 * 1e4).round() / 1e4
}

fn ser_round4_vec<S: Serializer>(v: &[f32], s: S) -> Result<S::Ok, S::Error> {
    let mut seq = s.serialize_seq(Some(v.len()))?;
    for &x in v {
        seq.serialize_element(&round4(x))?;
    }
    seq.end()
}

fn ser_round2_arr<S: Serializer>(v: &[f64; 3], s: S) -> Result<S::Ok, S::Error> {
    let mut seq = s.serialize_seq(Some(v.len()))?;
    for &x in v {
        seq.serialize_element(&((x * 1e2).round() / 1e2))?;
    }
    seq.end()
}

/// One snake's state on a single turn. `name`/`id` repeat across turns but zstd
/// dedups them trivially; we keep `id` so dead-snake removal can't desync the
/// roster mapping, and drop the per-move cosmetics the old format stored.
#[derive(Clone, Serialize)]
pub struct SnakeState {
    pub id: String,
    pub name: String,
    pub health: i16,
    /// Body, head first, as `[x, y]` pairs. A snake vanishing from one turn to
    /// the next is how death is encoded (the request drops eliminated snakes).
    pub body: Vec<[i8; 2]>,
}

/// Per-action search debug, serialized compact as `[move_index, prior, visits, q]`
/// with `prior`/`q` rounded to 4 dp and `visits` emitted as an integer count.
#[derive(Clone)]
pub struct ActionDebug(pub u8, pub f32, pub f32, pub f32);

impl Serialize for ActionDebug {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(4))?;
        seq.serialize_element(&self.0)?;
        seq.serialize_element(&round4(self.1))?;
        seq.serialize_element(&(self.2.round() as i64))?;
        seq.serialize_element(&round4(self.3))?;
        seq.end()
    }
}

/// The search diagnostics for one move (the valuable, non-redundant signal).
#[derive(Clone, Serialize)]
pub struct SearchInfo {
    pub sims_completed: usize,
    pub terminal_only_sims: usize,
    pub eval_rows: usize,
    pub forward_calls: usize,
    pub max_depth: u32,
    pub stopped_reason: &'static str,
    pub fallback_reason: Option<&'static str>,
    #[serde(serialize_with = "ser_round4_vec")]
    pub root_policy: Vec<f32>,
    #[serde(serialize_with = "ser_round4_vec")]
    pub root_values: Vec<f32>,
    /// Per-snake root actions.
    pub root_actions: Vec<Vec<ActionDebug>>,
}

/// One recorded move: board state this turn + our decision + search + timing.
#[derive(Clone, Serialize)]
pub struct MoveRecord {
    pub turn: u32,
    /// Index of our snake within `snakes` this turn.
    pub you: Option<usize>,
    pub chosen_move: Option<u8>,
    pub food: Vec<[i8; 2]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hazards: Vec<[i8; 2]>,
    pub snakes: Vec<SnakeState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchInfo>,
    /// `[lock_wait_ms, search_ms, total_ms]`, rounded to 2 dp.
    #[serde(serialize_with = "ser_round2_arr")]
    pub timing: [f64; 3],
}

/// Static per-game info, captured the first time we see the game.
#[derive(Clone, Serialize)]
pub struct GameMeta {
    width: i64,
    height: i64,
    ruleset: Value,
}

/// How a game ended, recorded in the output.
#[derive(Clone, Copy)]
pub enum Finish {
    /// `/end` received from the game server.
    Complete,
    /// No request for longer than the idle timeout.
    Incomplete,
}

impl Finish {
    fn label(self) -> &'static str {
        match self {
            Finish::Complete => "complete",
            Finish::Incomplete => "incomplete",
        }
    }
}

/// Message from a request handler to the recorder thread.
enum Msg {
    Move {
        game_id: String,
        meta: Option<GameMeta>,
        record: Box<MoveRecord>,
    },
    Finish {
        game_id: String,
        how: Finish,
    },
}

/// Accumulated state for one in-flight game.
struct GameLog {
    meta: Option<GameMeta>,
    moves: Vec<MoveRecord>,
    last_seen: Instant,
}

/// The serialized game document.
#[derive(Serialize)]
struct GameDoc<'a> {
    game_id: &'a str,
    model: &'a str,
    config: &'a Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    ruleset: Option<&'a Value>,
    board: BoardDims,
    /// Roster (id, name) from the first recorded turn.
    roster: Vec<RosterEntry>,
    finished: Finished,
    moves: &'a [MoveRecord],
}

#[derive(Serialize)]
struct BoardDims {
    width: i64,
    height: i64,
}

#[derive(Serialize)]
struct RosterEntry {
    id: String,
    name: String,
}

#[derive(Serialize)]
struct Finished {
    state: &'static str,
    move_count: usize,
    final_turn: Option<u32>,
}

/// Handle held by the HTTP app; cloneable across worker threads.
#[derive(Clone)]
pub struct Recorder {
    tx: Sender<Msg>,
}

impl Recorder {
    /// Spawn the recorder thread. `model`/`config` are global constants stored
    /// once per game. Returns `None` if `dir` is `None` (recording disabled).
    pub fn spawn(
        dir: Option<PathBuf>,
        model: String,
        config: Value,
        idle_timeout: Duration,
        zstd_level: i32,
    ) -> Option<Recorder> {
        let dir = dir?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            run(rx, dir, model, config, idle_timeout, zstd_level);
        });
        Some(Recorder { tx })
    }

    /// Append a move to a game. Non-blocking; never touches disk.
    pub fn record_move(&self, game_id: String, meta: Option<GameMeta>, record: MoveRecord) {
        let _ = self.tx.send(Msg::Move {
            game_id,
            meta,
            record: Box::new(record),
        });
    }

    /// Finalize a game now (called on `/end`).
    pub fn finish(&self, game_id: String) {
        let _ = self.tx.send(Msg::Finish {
            game_id,
            how: Finish::Complete,
        });
    }
}

/// Build the optional per-game meta from a parsed request. Only worth sending on
/// the first turn we see; the recorder keeps the first non-`None` it receives.
pub fn meta_from_request(request: &Value, width: i64, height: i64) -> GameMeta {
    let ruleset = request
        .get("game")
        .and_then(|g| g.get("ruleset"))
        .cloned()
        .unwrap_or(Value::Null);
    GameMeta {
        width,
        height,
        ruleset,
    }
}

fn run(
    rx: Receiver<Msg>,
    dir: PathBuf,
    model: String,
    config: Value,
    idle_timeout: Duration,
    zstd_level: i32,
) {
    let mut games: HashMap<String, GameLog> = HashMap::new();
    // Sweep at least as often as the idle timeout (and at most `SWEEP_INTERVAL`),
    // so a short timeout is honored promptly and a long one isn't busy-polled.
    let sweep = SWEEP_INTERVAL.min(idle_timeout).max(Duration::from_secs(1));
    loop {
        match rx.recv_timeout(sweep) {
            Ok(Msg::Move {
                game_id,
                meta,
                record,
            }) => {
                let log = games.entry(game_id).or_insert_with(|| GameLog {
                    meta: None,
                    moves: Vec::new(),
                    last_seen: Instant::now(),
                });
                if log.meta.is_none() {
                    log.meta = meta;
                }
                log.moves.push(*record);
                log.last_seen = Instant::now();
            }
            Ok(Msg::Finish { game_id, how }) => {
                if let Some(log) = games.remove(&game_id) {
                    finalize(&dir, &model, &config, game_id, log, how, zstd_level);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        // Sweep idle games regardless of why we woke (cheap; map is small).
        let stale: Vec<String> = games
            .iter()
            .filter(|(_, log)| log.last_seen.elapsed() >= idle_timeout)
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale {
            if let Some(log) = games.remove(&id) {
                finalize(
                    &dir,
                    &model,
                    &config,
                    id,
                    log,
                    Finish::Incomplete,
                    zstd_level,
                );
            }
        }
    }
    // Channel closed: flush whatever is left as incomplete (synchronously, so we
    // don't race process exit).
    for (id, log) in games.drain() {
        write_game(
            &dir,
            &model,
            &config,
            &id,
            &log,
            Finish::Incomplete,
            zstd_level,
        );
    }
}

/// Hand a finished game off to a detached thread for serialize + compress +
/// write. Keeps the recorder thread free to keep appending records for other
/// in-flight games while one game compresses (zstd-19 can take a while).
fn finalize(
    dir: &PathBuf,
    model: &str,
    config: &Value,
    game_id: String,
    log: GameLog,
    how: Finish,
    zstd_level: i32,
) {
    let (dir, model, config) = (dir.clone(), model.to_string(), config.clone());
    std::thread::spawn(move || {
        write_game(&dir, &model, &config, &game_id, &log, how, zstd_level);
    });
}

fn write_game(
    dir: &PathBuf,
    model: &str,
    config: &Value,
    game_id: &str,
    log: &GameLog,
    how: Finish,
    zstd_level: i32,
) {
    if log.moves.is_empty() {
        return;
    }
    let roster = log
        .moves
        .first()
        .map(|m| {
            m.snakes
                .iter()
                .map(|s| RosterEntry {
                    id: s.id.clone(),
                    name: s.name.clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    let (width, height, ruleset) = match &log.meta {
        Some(m) => (m.width, m.height, Some(&m.ruleset)),
        None => (0, 0, None),
    };
    let doc = GameDoc {
        game_id,
        model,
        config,
        ruleset,
        board: BoardDims { width, height },
        roster,
        finished: Finished {
            state: how.label(),
            move_count: log.moves.len(),
            final_turn: log.moves.last().map(|m| m.turn),
        },
        moves: &log.moves,
    };

    let json = match serde_json::to_vec(&doc) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("snek-server: recorder serialize failed for {game_id}: {e}");
            return;
        }
    };
    let compressed = match zstd::stream::encode_all(json.as_slice(), zstd_level) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("snek-server: recorder compress failed for {game_id}: {e}");
            return;
        }
    };

    let stem = safe_log_stem(game_id);
    let final_path = dir.join(format!("{stem}.json.zst"));
    let tmp_path = dir.join(format!("{stem}.json.zst.tmp"));
    if let Err(e) = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(&compressed)?;
        f.sync_all()?;
        fs::rename(&tmp_path, &final_path)
    })() {
        eprintln!(
            "snek-server: recorder write failed for {}: {e}",
            final_path.display()
        );
        let _ = fs::remove_file(&tmp_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_move(turn: u32) -> MoveRecord {
        MoveRecord {
            turn,
            you: Some(0),
            chosen_move: Some(0),
            food: vec![[6, 10], [0, 6]],
            hazards: vec![],
            snakes: vec![SnakeState {
                id: "me".into(),
                name: "local-8000".into(),
                health: 100 - turn as i16,
                body: vec![[5, 9], [5, 8]],
            }],
            search: Some(SearchInfo {
                sims_completed: 148,
                terminal_only_sims: 0,
                eval_rows: 296,
                forward_calls: 148,
                max_depth: 6,
                stopped_reason: "tree_resolved",
                fallback_reason: None,
                root_policy: vec![0.42, 0.26, 0.0, 0.30],
                root_values: vec![0.017, -0.05],
                root_actions: vec![vec![
                    ActionDebug(0, 0.11, 63.0, 0.10),
                    ActionDebug(1, 0.31, 39.0, -0.01),
                ]],
            }),
            timing: [0.0, 229.0, 229.1],
        }
    }

    #[test]
    fn write_game_roundtrips_through_zstd() {
        let dir = std::env::temp_dir().join("snek_recorder_test");
        let _ = fs::create_dir_all(&dir);
        let log = GameLog {
            meta: Some(GameMeta {
                width: 11,
                height: 11,
                ruleset: json!({"name": "standard"}),
            }),
            moves: vec![sample_move(0), sample_move(1)],
            last_seen: Instant::now(),
        };
        let config = json!({"max_sims": 100000});
        write_game(
            &dir,
            "net.safetensors",
            &config,
            "game-abc",
            &log,
            Finish::Complete,
            19,
        );

        let path = dir.join("game-abc.json.zst");
        let bytes = fs::read(&path).expect("output written");
        let json = zstd::stream::decode_all(bytes.as_slice()).expect("valid zstd");
        let doc: Value = serde_json::from_slice(&json).expect("valid json");

        assert_eq!(doc["game_id"], "game-abc");
        assert_eq!(doc["finished"]["state"], "complete");
        assert_eq!(doc["finished"]["move_count"], 2);
        assert_eq!(doc["finished"]["final_turn"], 1);
        assert_eq!(doc["board"]["width"], 11);
        assert_eq!(doc["roster"][0]["name"], "local-8000");
        assert_eq!(doc["moves"].as_array().unwrap().len(), 2);
        // Compact action arrays: [move_index, prior, visits, q]; visits is an int.
        let act = &doc["moves"][0]["search"]["root_actions"][0][0];
        assert_eq!(act[0], 0);
        assert_eq!(act[2], 63); // 63.0 visits serialized as integer
        assert!(act[2].is_i64() || act[2].is_u64());
        // hazards omitted when empty.
        assert!(doc["moves"][0].get("hazards").is_none());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn incomplete_state_is_recorded() {
        let dir = std::env::temp_dir().join("snek_recorder_test");
        let _ = fs::create_dir_all(&dir);
        let log = GameLog {
            meta: None,
            moves: vec![sample_move(0)],
            last_seen: Instant::now(),
        };
        write_game(
            &dir,
            "m",
            &json!({}),
            "game-inc",
            &log,
            Finish::Incomplete,
            3,
        );
        let bytes = fs::read(dir.join("game-inc.json.zst")).unwrap();
        let json = zstd::stream::decode_all(bytes.as_slice()).unwrap();
        let doc: Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(doc["finished"]["state"], "incomplete");
        // No meta seen → ruleset omitted.
        assert!(doc.get("ruleset").is_none());
        let _ = fs::remove_file(dir.join("game-inc.json.zst"));
    }
}

fn safe_log_stem(s: &str) -> String {
    let mut out = String::with_capacity(s.len().max(1));
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown_game".into()
    } else {
        out
    }
}
