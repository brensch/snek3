//! Continuous evaluation league. While a run is active, `league_games` game
//! slots run in-process on unpinned CPU threads: each slot picks the most
//! informative opponents for one game (every checkpoint at a multiple of
//! `league_entrant_gens` generations is in the pool, plus the external
//! baselines/API players), plays it via [`snek_server::arena::play_game`] —
//! one search thread per living snake — records it, refits ratings over all
//! history, and immediately starts the next game. The game is the unit
//! throughout: opponent selection, recording, rating updates.
//!
//! Ratings are the Plackett–Luce model: the multiplayer generalization of
//! Bradley–Terry (identical to it for two players), whose likelihood of a full
//! elimination order is a product of successive "who outlasts the rest"
//! choices. It is fitted by maximum likelihood with the standard MM algorithm
//! (Hunter 2004), ties averaged over their orderings, converted to the Elo
//! scale (400·log10) and anchored at the pool's earliest generation (Elo 0).
//!
//! Files in `runs/<id>/eval/`:
//!
//!   summary.jsonl        one line per game: the full placement ranking
//!   ratings.json         latest Plackett–Luce fit for every rated net
//!   game_SSSSSS.json     one game's frames + search readout, in the exact
//!                        schema of games/gen_NNNN.json
//!
//! Old game recordings are pruned (newest `KEEP_GAME_FILES` kept); the
//! summary log and ratings are kept forever. The league stops (abandoning any
//! in-flight game) when the run pauses, and resumes with it.

pub mod burst;

use crate::config::RunConfig;
use crate::state::RunPaths;
use crate::trainer::TrainerHandle;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use snek_server::arena::{play_game, Budget, GameSettings, Player, PlayerSpec};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Recorded game files kept on disk (newest first).
const KEEP_GAME_FILES: usize = 200;

/// League games end here if nobody has won yet; survivors tie at rank 1.
const LEAGUE_MAX_TURNS: u32 = 500;

/// External (non-checkpoint) league players live at ids counting down from
/// `u32::MAX`, far above any real checkpoint generation. Fixed built-in ids:
/// the flood-fill MCTS baseline and the stronger time-expanded-Voronoi MCTS
/// baseline (both `snek-heuristic`, seated by literal player specs). They
/// never learn, so their fitted Elo is a stable reference every net is
/// constantly measured against.
pub const HEURISTIC_GEN: u32 = u32::MAX;
pub const VORONOI_GEN: u32 = u32::MAX - 1;

/// Configured API players are assigned ids from here downward (persisted in
/// eval/players.json so a player's id survives config edits and restarts).
const API_GEN_START: u32 = u32::MAX - 16;

/// Every id at or above this is an external player, not a checkpoint.
pub const EXTERNAL_GEN_MIN: u32 = u32::MAX - 255;

pub fn is_external(gen: u32) -> bool {
    gen >= EXTERNAL_GEN_MIN
}

/// Chance a league game is forced to include one external player, drawn
/// uniformly from the roster (they also flow through the normal
/// information-gain sampling on top of this).
const EXTERNAL_PLAY_PROB: f64 = 1.0 / 3.0;

/// Persistent per-net sampling floor for the top of the leaderboard, on top
/// of the 1/(1+games) exploration weight. Pure exploration collapses
/// attention onto whichever net has simply played least, so an aged leader
/// (few games relative to fresh entrants, but decision-critical) stops being
/// sampled and its rating ossifies on a thin, stale set. A leader floor keeps
/// the current best nets — the ones the run's decisions hinge on, and the
/// ones a fresh entrant must beat to count as progress — continuously
/// re-tested against the changing field. Scaled by the net's squared
/// leaderboard percentile, so it concentrates on the very top: the #1 net
/// gets the full floor (≈ a net with `1/FLOOR` games' worth of pull), the
/// median net almost none.
const LEADER_ATTENTION_FLOOR: f64 = 0.1;

/// One external league player: a built-in heuristic agent or a configured
/// Battlesnake-protocol API server. `spec` is the player spec string that
/// seats it (a heuristic token or an http url).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalPlayer {
    pub gen: u32,
    pub name: String,
    pub spec: String,
}

/// The run's external players: the fixed built-ins plus the config's API
/// entries. `known` additionally keeps every player ever registered (from
/// eval/players.json), so historical records stay nameable after a config
/// removes a player.
pub struct ExternalRegistry {
    pub active: Vec<ExternalPlayer>,
    pub known: Vec<ExternalPlayer>,
}

impl ExternalRegistry {
    pub fn name(&self, gen: u32) -> Option<&str> {
        self.known
            .iter()
            .find(|p| p.gen == gen)
            .map(|p| p.name.as_str())
    }

    fn spec(&self, gen: u32) -> Option<&str> {
        self.known
            .iter()
            .find(|p| p.gen == gen)
            .map(|p| p.spec.as_str())
    }

    fn display(&self, gen: u32) -> String {
        self.name(gen)
            .map(str::to_string)
            .unwrap_or_else(|| player_name(gen))
    }
}

/// Resolve the external roster for `cfg`, assigning fresh ids to API players
/// seen for the first time, and persist the union to eval/players.json.
/// Malformed config entries are skipped (returned so the caller can log).
pub fn load_external_registry(cfg: &RunConfig, eval_dir: &Path) -> (ExternalRegistry, Vec<String>) {
    let builtins = vec![
        ExternalPlayer {
            gen: HEURISTIC_GEN,
            name: "floodfill".into(),
            spec: "heuristic".into(),
        },
        ExternalPlayer {
            gen: VORONOI_GEN,
            name: "voronoi".into(),
            spec: "voronoi".into(),
        },
    ];
    let path = eval_dir.join("players.json");
    let saved: Vec<ExternalPlayer> = std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let mut next_gen = saved
        .iter()
        .map(|p| p.gen)
        .filter(|&g| g <= API_GEN_START)
        .min()
        .map(|g| g - 1)
        .unwrap_or(API_GEN_START);
    let mut active = builtins.clone();
    let mut errors = Vec::new();
    for entry in &cfg.league_api_players {
        let Some((name, url)) = entry.split_once('=') else {
            errors.push(format!(
                "league_api_players entry {entry:?} is not \"name=url\"; skipped"
            ));
            continue;
        };
        let (name, url) = (name.trim(), url.trim().trim_end_matches('/'));
        if name.is_empty() || !(url.starts_with("http://") || url.starts_with("https://")) {
            errors.push(format!(
                "league_api_players entry {entry:?} needs a name and an http(s) url; skipped"
            ));
            continue;
        }
        if active.iter().any(|p| p.name == name) {
            errors.push(format!("duplicate league player name {name:?}; skipped"));
            continue;
        }
        // The name is the stable identity; the id sticks across restarts.
        let gen = saved
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.gen)
            .unwrap_or_else(|| {
                let g = next_gen;
                next_gen -= 1;
                g
            });
        active.push(ExternalPlayer {
            gen,
            name: name.to_string(),
            spec: url.to_string(),
        });
    }
    // known = everything ever seen; active entries override saved specs.
    let mut known = active.clone();
    for p in saved {
        if !known.iter().any(|k| k.gen == p.gen) {
            known.push(p);
        }
    }
    known.sort_by_key(|p| std::cmp::Reverse(p.gen));
    let doc = serde_json::to_vec_pretty(&known).unwrap_or_default();
    if std::fs::read(&path).ok().as_deref() != Some(doc.as_slice()) {
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, &doc).is_ok() {
            let _ = std::fs::rename(tmp, &path);
        }
    }
    (ExternalRegistry { active, known }, errors)
}

/// Display name for a league player id (fallback when no registry is at
/// hand; the registry knows configured API players' real names).
pub fn player_name(gen: u32) -> String {
    match gen {
        HEURISTIC_GEN => "floodfill".to_string(),
        VORONOI_GEN => "voronoi".to_string(),
        g if is_external(g) => format!("api_{}", API_GEN_START.saturating_sub(g)),
        g => format!("gen_{g:04}"),
    }
}

/// One league thread per process; a second run start waits for the old one.
static LEAGUE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// One line of `runs/<id>/eval/summary.jsonl`: one game's full placement
/// ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchRecord {
    #[serde(default)]
    pub seq: u64,
    /// One entry per seat; rank 1 is best, ties share a rank.
    #[serde(default)]
    pub placements: Vec<Placement>,
    #[serde(default)]
    pub turns: u32,
    #[serde(default)]
    pub sims: u32,
    #[serde(default)]
    pub wall_seconds: f64,
    #[serde(default)]
    pub finished_unix_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Placement {
    pub gen: u32,
    pub seat: u32,
    pub rank: u32,
    #[serde(default)]
    pub death_turn: Option<u32>,
}

/// `runs/<id>/eval/ratings.json`: the latest Plackett–Luce fit.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LeagueRatings {
    pub updated_unix_ms: i64,
    /// Every net with at least one recorded game, ascending by generation.
    pub ratings: Vec<LeagueRating>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeagueRating {
    pub gen: u32,
    /// Display name ("gen_0042", "floodfill", or a configured API player's
    /// name). Empty in files written before names existed.
    #[serde(default)]
    pub name: String,
    /// Plackett–Luce strength on the Elo scale, anchored so the earliest rated
    /// generation is 0.
    pub elo: f64,
    pub games: u32,
    /// Rank-1 finishes (shared firsts count == placements[0]).
    pub wins: u32,
    pub avg_rank: f64,
    /// Finishes by rank: placements[r-1] = number of r-th-place finishes.
    /// Length is the deepest rank this net ever reached. Empty in older files.
    #[serde(default)]
    pub placements: Vec<u32>,
}

/// Real-time state of the in-flight league games, published to the frontend
/// via the shared SSE event stream. One entry per game slot currently playing.
#[derive(Debug, Clone, Serialize)]
pub struct LiveEval {
    /// The league is enabled and slots are running (false while paused).
    pub active: bool,
    /// In-flight games, ascending by seq. Each carries its own players.
    pub games: Vec<LiveGame>,
    pub updated_unix_ms: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveGame {
    pub seq: u64,
    pub turn: u32,
    /// This game's players; seat s plays players[s % players.len()].
    pub players: Vec<LivePlayer>,
    /// Latest board frame (same shape as one frame of games/gen_NNNN.json).
    pub frame: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LivePlayer {
    pub gen: u32,
    /// Display name (external players carry their registry name).
    pub name: String,
    /// Fitted league Elo entering the game.
    pub elo: f64,
    /// Career league games.
    pub games: u32,
}

static LIVE: Mutex<LiveEval> = Mutex::new(LiveEval {
    active: false,
    games: Vec::new(),
    updated_unix_ms: 0,
});

/// Snapshot of the in-flight game state (for the SSE endpoint).
pub fn live() -> LiveEval {
    LIVE.lock().unwrap().clone()
}

/// Every live-state change is also broadcast, so the SSE endpoint pushes a
/// frame per turn instead of polling on a timer.
static LIVE_TX: std::sync::OnceLock<tokio::sync::broadcast::Sender<LiveEval>> =
    std::sync::OnceLock::new();

fn live_tx() -> &'static tokio::sync::broadcast::Sender<LiveEval> {
    LIVE_TX.get_or_init(|| tokio::sync::broadcast::channel(64).0)
}

/// Subscribe to live-state changes (one message per board turn while league
/// games are in flight).
pub fn live_rx() -> tokio::sync::broadcast::Receiver<LiveEval> {
    live_tx().subscribe()
}

fn set_live(update: impl FnOnce(&mut LiveEval)) {
    let mut live = LIVE.lock().unwrap();
    update(&mut live);
    live.updated_unix_ms = chrono::Utc::now().timestamp_millis();
    let _ = live_tx().send(live.clone());
}

/// Start the league for the active run. Returns immediately; the supervisor
/// thread and its game slots exit when `stop` is set (abandoning any
/// in-flight game). The returned context is shared with the GPU burst arena
/// (see [`burst`]) so burst games land in the same records/ratings as the
/// CPU slots'.
pub fn start_league(
    paths: RunPaths,
    trainer: TrainerHandle,
    stop: Arc<AtomicBool>,
) -> Arc<LeagueCtx> {
    let mut records = read_summaries(&paths.root);
    records.sort_by_key(|r| r.seq);
    let next_seq = records.iter().map(|m| m.seq + 1).max().unwrap_or(0);
    let ctx = Arc::new(LeagueCtx {
        paths,
        trainer,
        stop: stop.clone(),
        records: Mutex::new(records),
        next_seq: AtomicU64::new(next_seq),
        registry: Mutex::new(()),
        logged_errors: Mutex::new(Default::default()),
        announced: AtomicBool::new(false),
    });
    let loop_ctx = Arc::clone(&ctx);
    std::thread::Builder::new()
        .name("eval-league".into())
        .spawn(move || {
            // Wait for a previous run's league (same process) to wind down.
            while LEAGUE_ACTIVE.swap(true, Ordering::SeqCst) {
                if loop_ctx.stop.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            league_loop(loop_ctx);
            LEAGUE_ACTIVE.store(false, Ordering::SeqCst);
        })
        .expect("spawn eval league thread");
    ctx
}

/// State shared by the supervisor, every game slot, and the GPU burst arena.
pub struct LeagueCtx {
    paths: RunPaths,
    trainer: TrainerHandle,
    stop: Arc<AtomicBool>,
    /// All recorded games (in-memory mirror of summary.jsonl), plus every
    /// write that must be serialized with it (summary append, ratings file,
    /// pruning).
    records: Mutex<Vec<MatchRecord>>,
    next_seq: AtomicU64,
    /// Serializes players.json read/modify/write across slots.
    registry: Mutex<()>,
    /// Config problems repeat every game; log each distinct message once.
    logged_errors: Mutex<std::collections::HashSet<String>>,
    announced: AtomicBool,
}

/// Supervisor: keeps `league_games` slot threads alive while the run is
/// active, growing/shrinking with config changes, and tears everything down
/// on stop.
fn league_loop(ctx: Arc<LeagueCtx>) {
    let metrics = ctx.trainer.metrics();
    let stop = ctx.stop.clone();

    let mut slots: Vec<(Arc<AtomicBool>, std::thread::JoinHandle<()>)> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        let cfg = ctx.trainer.config();
        let want = if cfg.league_entrant_gens == 0 {
            0
        } else {
            cfg.league_games.max(1)
        };
        // Reap slots that exited (finished a shrink request, or panicked —
        // the slot logs its own panic; a fresh slot takes its place).
        slots.retain(|(_, handle)| !handle.is_finished());
        while slots.len() < want {
            let slot_stop = Arc::new(AtomicBool::new(false));
            let ctx = Arc::clone(&ctx);
            let flag = Arc::clone(&slot_stop);
            let handle = std::thread::Builder::new()
                .name(format!("league-slot-{}", slots.len()))
                .spawn(move || slot_loop(ctx, flag))
                .expect("spawn league slot");
            slots.push((slot_stop, handle));
        }
        for (flag, _) in slots.iter().skip(want) {
            flag.store(true, Ordering::Relaxed);
        }
        let active = want > 0;
        if live().active != active {
            set_live(|l| l.active = active);
        }
        sleep_unless_stopped(&stop, 2);
    }
    for (flag, _) in &slots {
        flag.store(true, Ordering::Relaxed);
    }
    for (_, handle) in slots {
        let _ = handle.join();
    }
    set_live(|l| {
        l.active = false;
        l.games.clear();
    });
    metrics.log("eval league: stopped".to_string());
}

/// One game slot: pick opponents, play one game, record + refit, repeat.
fn slot_loop(ctx: Arc<LeagueCtx>, slot_stop: Arc<AtomicBool>) {
    // League searches must never steal CPU from the GPU-feeding self-play
    // threads. Niceness is per-thread on Linux and inherited by the search
    // threads play_game spawns, so one call here demotes the whole slot:
    // league games consume only otherwise-idle cores.
    #[cfg(target_os = "linux")]
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, 15);
    }
    let metrics = ctx.trainer.metrics();
    let stopped = || ctx.stop.load(Ordering::Relaxed) || slot_stop.load(Ordering::Relaxed);
    let nap = |secs: u64| {
        for _ in 0..secs {
            if stopped() {
                return;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    };

    while !stopped() {
        let cfg = ctx.trainer.config();
        if cfg.league_entrant_gens == 0 {
            nap(15); // the supervisor is about to retire this slot
            continue;
        }
        let eval_dir = ctx.paths.root.join("eval");
        if std::fs::create_dir_all(&eval_dir).is_err() {
            nap(30);
            continue;
        }
        // External players (built-in heuristics + configured API snakes)
        // always hold pool slots, so games (and a meaningful Elo) start from
        // the very first checkpoint.
        let (registry, config_errors) = {
            let _guard = ctx.registry.lock().unwrap();
            load_external_registry(&cfg, &eval_dir)
        };
        for err in config_errors {
            if ctx.logged_errors.lock().unwrap().insert(err.clone()) {
                metrics.log(format!("eval league: {err}"));
            }
        }
        let mut pool = pool_members(&ctx.paths, cfg.league_entrant_gens);
        pool.extend(registry.active.iter().map(|p| p.gen));
        pool.sort_unstable();
        pool.dedup();
        if pool.len() < 2 {
            nap(15);
            continue;
        }
        if !ctx.announced.swap(true, Ordering::Relaxed) {
            metrics.log(format!(
                "eval league: running — pool of {} players incl. {} external ({}), entrant every {} gens, {} distinct players per game, {} sims, {} concurrent games",
                pool.len(),
                registry.active.len(),
                registry
                    .active
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                cfg.league_entrant_gens,
                cfg.num_snakes.min(pool.len()),
                cfg.eval_sims,
                cfg.league_games.max(1),
            ));
        }

        let seq = ctx.next_seq.fetch_add(1, Ordering::Relaxed);
        let (players, live_players) = {
            let records = ctx.records.lock().unwrap();
            let ratings = fit_ratings(&pool, &records);
            let players = pick_players(
                &pool,
                &records,
                &HashMap::new(),
                &ratings,
                cfg.num_snakes,
                seq,
            );
            let career = |gen: u32| {
                records
                    .iter()
                    .flat_map(participants)
                    .filter(|&g| g == gen)
                    .count() as u32
            };
            let live_players: Vec<LivePlayer> = players
                .iter()
                .map(|&gen| LivePlayer {
                    gen,
                    name: registry.display(gen),
                    elo: ratings.get(&gen).copied().unwrap_or(0.0),
                    games: career(gen),
                })
                .collect();
            (players, live_players)
        };

        let budget = Budget::Sims(cfg.eval_sims.max(1));
        let mut seats: Vec<Player> = Vec::with_capacity(players.len());
        let mut load_error = None;
        for &gen in &players {
            let spec_str = registry
                .spec(gen)
                .map(str::to_string)
                .unwrap_or_else(|| ctx.paths.checkpoint_net(gen).display().to_string());
            match Player::load(
                &PlayerSpec::parse(&spec_str),
                tch::Device::Cpu,
                cfg.trunk_channels,
                cfg.trunk_blocks,
                budget,
                &registry.display(gen),
            ) {
                Ok(player) => seats.push(player),
                Err(err) => {
                    load_error = Some(format!("{}: {err}", registry.display(gen)));
                    break;
                }
            }
        }
        if let Some(err) = load_error {
            // A checkpoint can vanish mid-pick (pruning); just try again.
            if ctx.logged_errors.lock().unwrap().insert(err.clone()) {
                metrics.log(format!("eval league: failed to load player: {err}"));
            }
            nap(10);
            continue;
        }

        // Search parameters match training (c_puct/draw_value from the run
        // config); the argmax serving search itself guarantees determinism.
        let settings = GameSettings {
            seats: cfg.num_snakes,
            board_size: cfg.board,
            max_turns: LEAGUE_MAX_TURNS,
            cfg: snek_server::Config {
                max_sims: cfg.eval_sims.max(1),
                c_puct: cfg.c_puct,
                draw_value: cfg.draw_value,
                eval_chunk: 4096,
                leaves_per_sim: 8,
                virtual_loss: 1.0,
            },
            budget,
            // One intra-op thread per seat worker — league searches must
            // never fan out onto training's cores.
            torch_threads: 1,
        };
        set_live(|l| {
            l.games.push(LiveGame {
                seq,
                turn: 0,
                players: live_players.clone(),
                frame: None,
            });
            l.games.sort_by_key(|g| g.seq);
        });
        let seed = seq.wrapping_mul(1_000_003);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            play_game(
                &mut seats,
                0, // pick_players already rotates seat order by seq
                &settings,
                seed,
                &stopped,
                &mut |frame| {
                    let value = serde_json::to_value(frame).ok();
                    set_live(|l| {
                        if let Some(game) = l.games.iter_mut().find(|g| g.seq == seq) {
                            game.turn = frame.turn;
                            game.frame = value;
                        }
                    });
                },
            )
        }));
        set_live(|l| l.games.retain(|g| g.seq != seq));

        let outcome = match result {
            Ok(Some(outcome)) => outcome,
            Ok(None) => continue, // stopped mid-game; loop condition exits
            Err(_) => {
                metrics.log(format!(
                    "eval league #{seq}: game panicked (players {players:?}); slot continuing"
                ));
                nap(30);
                continue;
            }
        };

        let record = MatchRecord {
            seq,
            placements: outcome
                .placements
                .iter()
                .map(|p| Placement {
                    gen: players.get(p.player as usize).copied().unwrap_or(0),
                    seat: p.seat,
                    rank: p.rank,
                    death_turn: p.death_turn,
                })
                .collect(),
            turns: outcome.turns,
            sims: cfg.eval_sims as u32,
            wall_seconds: outcome.wall_ms as f64 / 1000.0,
            finished_unix_ms: chrono::Utc::now().timestamp_millis(),
        };
        if let Err(err) = write_game_file(&eval_dir, seq, &players, &registry, &record, &outcome) {
            metrics.log(format!("eval league #{seq}: failed to record game: {err}"));
        }

        let mut order = record.placements.clone();
        order.sort_by_key(|p| p.rank);
        metrics.log(format!(
            "eval league #{seq}: {ranking} ({turns} turns, {secs:.0}s)",
            ranking = order
                .iter()
                .map(|p| registry.display(p.gen))
                .collect::<Vec<_>>()
                .join(" > "),
            turns = record.turns,
            secs = record.wall_seconds,
        ));

        {
            let mut records = ctx.records.lock().unwrap();
            if let Err(err) = append_summary(&eval_dir.join("summary.jsonl"), &record) {
                metrics.log(format!("eval league: failed to append summary: {err}"));
            }
            records.push(record);
            records.sort_by_key(|r| r.seq);
            let fitted = fit_ratings(&pool, &records);
            if let Err(err) =
                write_ratings(&eval_dir.join("ratings.json"), &records, &fitted, &registry)
            {
                metrics.log(format!("eval league: failed to write ratings: {err}"));
            }
            prune_game_files(&eval_dir, KEEP_GAME_FILES);
        }
    }
}

/// Write one game's recording as `game_SSSSSS.json`, in the exact schema of
/// the trainer's `games/gen_NNNN.json` so the frontend reuses one viewer. The
/// `config` slot carries the players and placements for the reader.
fn write_game_file(
    eval_dir: &Path,
    seq: u64,
    players: &[u32],
    registry: &ExternalRegistry,
    record: &MatchRecord,
    outcome: &snek_server::arena::Outcome,
) -> anyhow::Result<()> {
    let winners: Vec<u32> = record
        .placements
        .iter()
        .filter(|p| p.rank == 1)
        .map(|p| p.seat)
        .collect();
    let file = serde_json::json!({
        "gen": players.iter().copied().filter(|&g| !is_external(g)).max().unwrap_or(0),
        "config": {
            "seq": seq,
            "players": players.iter().map(|&g| serde_json::json!({
                "gen": g,
                "name": registry.display(g),
            })).collect::<Vec<_>>(),
            "placements": record.placements,
            "sims": record.sims,
        },
        "games": [{
            "frames": outcome.frames,
            "winner": match winners.as_slice() {
                [only] => Some(*only as i32),
                _ => None,
            },
            "num_turns": outcome.turns,
        }],
    });
    let path = eval_dir.join(format!("game_{seq:06}.json"));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&file)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Checkpoints eligible for the pool: every archived generation at a multiple
/// of `entrant_gens` (gen 0 included — it anchors the Elo scale). Ascending.
fn pool_members(paths: &RunPaths, entrant_gens: usize) -> Vec<u32> {
    let mut gens: Vec<u32> = match std::fs::read_dir(&paths.checkpoints) {
        Ok(rd) => rd
            .flatten()
            .filter_map(|e| {
                e.file_name()
                    .to_str()?
                    .strip_prefix("net_")?
                    .strip_suffix(".safetensors")?
                    .parse()
                    .ok()
            })
            .filter(|g: &u32| (*g as usize).is_multiple_of(entrant_gens))
            .collect(),
        Err(_) => Vec::new(),
    };
    gens.sort_unstable();
    gens.dedup();
    gens
}

/// The distinct gens a record involves.
fn participants(record: &MatchRecord) -> Vec<u32> {
    let mut gens: Vec<u32> = record.placements.iter().map(|p| p.gen).collect();
    gens.sort_unstable();
    gens.dedup();
    gens
}

/// Select the players for the next game by expected information gain. Under
/// Plackett–Luce, the information a game carries about a pair is p(1−p) — the
/// Bradley–Terry Fisher information, maximal for evenly matched players — so
/// strong nets naturally meet; scaling by rating uncertainty (∝ 1/(1+games))
/// pulls newly admitted and under-played nets in. The first player is sampled
/// by uncertainty, each further seat by its summed pairwise weight against
/// those already selected. Sampling keeps occasional cross-table games
/// flowing, so the match graph stays connected and upsets can be detected.
///
/// `extra_games` are per-player counts to add on top of the record history:
/// the burst arena picks its whole schedule up front, so it counts the games
/// it has already *planned* to keep one under-played net from soaking up
/// every matchup. The CPU league passes an empty map.
fn pick_players(
    pool: &[u32],
    records: &[MatchRecord],
    extra_games: &HashMap<u32, u32>,
    ratings: &HashMap<u32, f64>,
    seats: usize,
    seq: u64,
) -> Vec<u32> {
    let mut games: HashMap<u32, u32> = pool
        .iter()
        .map(|&g| (g, extra_games.get(&g).copied().unwrap_or(0)))
        .collect();
    for record in records {
        for g in participants(record) {
            if let Some(count) = games.get_mut(&g) {
                *count += 1;
            }
        }
    }
    let elo = |g: u32| ratings.get(&g).copied().unwrap_or(0.0);
    let uncertainty = |g: u32| 1.0 / (1.0 + games[&g] as f64);
    let info = |a: u32, b: u32| {
        let p = 1.0 / (1.0 + 10f64.powf((elo(b) - elo(a)) / 400.0));
        p * (1.0 - p)
    };
    // Leaderboard percentile among the *nets* (externals are fixed references,
    // not the thing we're trying to rate precisely), 0 at the bottom to 1 at
    // the top. Drives the leader floor below.
    let mut nets_by_elo: Vec<u32> = pool.iter().copied().filter(|&g| !is_external(g)).collect();
    nets_by_elo.sort_by(|&a, &b| elo(a).partial_cmp(&elo(b)).unwrap_or(std::cmp::Ordering::Equal));
    let leader_pct: HashMap<u32, f64> = nets_by_elo
        .iter()
        .enumerate()
        .map(|(i, &g)| {
            let p = if nets_by_elo.len() > 1 {
                i as f64 / (nets_by_elo.len() - 1) as f64
            } else {
                1.0
            };
            (g, p)
        })
        .collect();
    // Sampling attention for a player: exploration (cover the under-played)
    // plus a persistent floor on the leaderboard's top so aged leaders keep
    // being pressure-tested instead of dropping out once younger nets have
    // fewer games. Used both to anchor a match and to draw its opponents.
    let attention = |g: u32| {
        let pct = leader_pct.get(&g).copied().unwrap_or(0.0);
        uncertainty(g) + LEADER_ATTENTION_FLOOR * pct * pct
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let sample = |rng: &mut Xoshiro256PlusPlus, cands: &[u32], weights: &[f64]| -> u32 {
        let total: f64 = weights.iter().sum();
        let mut pick = rng.gen_range(0.0..total.max(f64::MIN_POSITIVE));
        for (i, w) in weights.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                return cands[i];
            }
        }
        *cands.last().expect("candidates non-empty")
    };

    let k = seats.min(pool.len()).max(2);
    let mut selected = Vec::with_capacity(k);
    // Regularly force one external player in (they also flow through the
    // normal sampling below), so nets are constantly compared against
    // players that never change.
    let externals: Vec<u32> = pool.iter().copied().filter(|&g| is_external(g)).collect();
    if !externals.is_empty() && rng.gen_bool(EXTERNAL_PLAY_PROB) {
        selected.push(externals[rng.gen_range(0..externals.len())]);
    }
    if selected.is_empty() {
        let weights: Vec<f64> = pool.iter().map(|&g| attention(g)).collect();
        selected.push(sample(&mut rng, pool, &weights));
    }
    while selected.len() < k {
        let cands: Vec<u32> = pool
            .iter()
            .copied()
            .filter(|g| !selected.contains(g))
            .collect();
        let weights: Vec<f64> = cands
            .iter()
            .map(|&c| {
                selected
                    .iter()
                    .map(|&s| info(c, s) * (attention(c) + attention(s)))
                    .sum::<f64>()
            })
            .collect();
        selected.push(sample(&mut rng, &cands, &weights));
    }
    // Rotate the seat order by seq so no player systematically holds seat 0.
    let rot = (seq % k as u64) as usize;
    selected.rotate_left(rot);
    selected
}

/// A weighted strict ordering of gens, best first.
type Ranking = (f64, Vec<u32>);

/// Expand a record into weighted strict rankings. Placement ties are averaged
/// over the orderings of each tied group (weight 1/k! each).
fn rankings_of(record: &MatchRecord) -> Vec<Ranking> {
    if record.placements.is_empty() {
        return Vec::new();
    }
    // Group gens by rank (ascending); each group is a tied set.
    let mut rank_groups: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for p in &record.placements {
        rank_groups.entry(p.rank).or_default().push(p.gen);
    }
    let groups: Vec<Vec<u32>> = rank_groups.into_values().collect();
    let mut out: Vec<Ranking> = vec![(1.0, Vec::new())];
    for group in groups {
        let perms = permutations_capped(&group);
        let w = 1.0 / perms.len() as f64;
        out = out
            .into_iter()
            .flat_map(|(rw, prefix)| {
                perms.iter().map(move |perm| {
                    let mut order = prefix.clone();
                    order.extend_from_slice(perm);
                    (rw * w, order)
                })
            })
            .collect();
    }
    out
}

/// All orderings of a tied group, capped: groups larger than 4 fall back to
/// cyclic rotations (still symmetric across members, bounded cost).
fn permutations_capped(group: &[u32]) -> Vec<Vec<u32>> {
    if group.len() <= 1 {
        return vec![group.to_vec()];
    }
    if group.len() > 4 {
        return (0..group.len())
            .map(|r| {
                let mut v = group.to_vec();
                v.rotate_left(r);
                v
            })
            .collect();
    }
    let mut out = Vec::new();
    let mut items = group.to_vec();
    permute(&mut items, 0, &mut out);
    out
}

fn permute(items: &mut Vec<u32>, k: usize, out: &mut Vec<Vec<u32>>) {
    if k == items.len() {
        out.push(items.clone());
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, out);
        items.swap(k, i);
    }
}

/// Plackett–Luce maximum-likelihood ratings over every recorded game, via the
/// standard MM iteration (Hunter 2004); exactly Bradley–Terry for two-player
/// rankings. A half virtual tie between adjacent pool members keeps the graph
/// connected and every rating finite. Anchored so the earliest rated
/// generation sits at Elo 0.
fn fit_ratings(pool: &[u32], records: &[MatchRecord]) -> HashMap<u32, f64> {
    let mut rankings: Vec<Ranking> = records.iter().flat_map(rankings_of).collect();
    for w in pool.windows(2) {
        rankings.push((0.25, vec![w[0], w[1]]));
        rankings.push((0.25, vec![w[1], w[0]]));
    }
    let mut gens: Vec<u32> = pool.to_vec();
    gens.extend(rankings.iter().flat_map(|(_, r)| r.iter().copied()));
    gens.sort_unstable();
    gens.dedup();
    if gens.len() < 2 {
        return gens.into_iter().map(|g| (g, 0.0)).collect();
    }
    let index: HashMap<u32, usize> = gens.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let n = gens.len();
    let mut p = vec![1.0f64; n];
    for _ in 0..200 {
        let mut wins = vec![0.0f64; n];
        let mut denom = vec![0.0f64; n];
        for (w, order) in &rankings {
            let mut remaining: f64 = order.iter().map(|g| p[index[g]]).sum();
            for k in 0..order.len().saturating_sub(1) {
                let winner = index[&order[k]];
                wins[winner] += w;
                for g in &order[k..] {
                    denom[index[g]] += w / remaining;
                }
                remaining -= p[winner];
            }
        }
        for i in 0..n {
            if wins[i] > 0.0 && denom[i] > 0.0 {
                p[i] = wins[i] / denom[i];
            }
        }
        // Renormalize each round so the iteration stays well-conditioned.
        let mean = p.iter().sum::<f64>() / n as f64;
        for v in &mut p {
            *v /= mean;
        }
    }
    let anchor = p[0];
    gens.iter()
        .enumerate()
        .map(|(i, &g)| (g, 400.0 * (p[i] / anchor).log10()))
        .collect()
}

fn write_ratings(
    path: &Path,
    records: &[MatchRecord],
    fitted: &HashMap<u32, f64>,
    registry: &ExternalRegistry,
) -> anyhow::Result<()> {
    // Per gen: games, rank sum, and a histogram of finishes by rank
    // (placements[r-1] = number of r-th-place finishes).
    #[derive(Default)]
    struct Tally {
        games: u32,
        rank_sum: u64,
        placements: Vec<u32>,
    }
    let mut tally: HashMap<u32, Tally> = HashMap::new();
    for record in records {
        for p in &record.placements {
            let t = tally.entry(p.gen).or_default();
            t.games += 1;
            t.rank_sum += p.rank as u64;
            let idx = p.rank.saturating_sub(1) as usize;
            if idx >= t.placements.len() {
                t.placements.resize(idx + 1, 0);
            }
            t.placements[idx] += 1;
        }
    }
    let mut ratings: Vec<LeagueRating> = fitted
        .iter()
        .map(|(&gen, &elo)| {
            let t = tally.get(&gen);
            let games = t.map_or(0, |t| t.games);
            let rank_sum = t.map_or(0, |t| t.rank_sum);
            let placements = t.map(|t| t.placements.clone()).unwrap_or_default();
            LeagueRating {
                gen,
                name: registry.display(gen),
                elo,
                games,
                wins: placements.first().copied().unwrap_or(0),
                avg_rank: if games > 0 {
                    rank_sum as f64 / games as f64
                } else {
                    0.0
                },
                placements,
            }
        })
        .collect();
    ratings.sort_by_key(|r| r.gen);
    let doc = LeagueRatings {
        updated_unix_ms: chrono::Utc::now().timestamp_millis(),
        ratings,
    };
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&doc)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Keep only the newest `keep` recorded games on disk. The summary log and
/// ratings are never pruned.
fn prune_game_files(eval_dir: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(eval_dir) else {
        return;
    };
    let mut files: Vec<(u64, std::path::PathBuf)> = rd
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let seq: u64 = name
                .strip_prefix("game_")?
                .strip_suffix(".json")?
                .parse()
                .ok()?;
            Some((seq, path))
        })
        .collect();
    if files.len() <= keep {
        return;
    }
    files.sort_by_key(|&(seq, _)| std::cmp::Reverse(seq));
    for (_, path) in files.into_iter().skip(keep) {
        let _ = std::fs::remove_file(path);
    }
}

fn sleep_unless_stopped(stop: &AtomicBool, secs: u64) {
    for _ in 0..secs {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn append_summary(path: &Path, record: &MatchRecord) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string(record)?)?;
    Ok(())
}

/// Read a run's game history, ascending by seq. Missing file → empty.
pub fn read_summaries(run_root: &Path) -> Vec<MatchRecord> {
    read_summary_file(&run_root.join("eval").join("summary.jsonl"))
}

/// Read one summary.jsonl. Missing file → empty.
pub fn read_summary_file(path: &Path) -> Vec<MatchRecord> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut records: Vec<MatchRecord> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .filter(|r: &MatchRecord| !r.placements.is_empty())
        .collect();
    records.sort_by_key(|r| r.seq);
    records
}

/// Read a run's latest league ratings. Missing file → empty.
pub fn read_ratings(run_root: &Path) -> LeagueRatings {
    let path = run_root.join("eval").join("ratings.json");
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(seq: u64, order: &[u32]) -> MatchRecord {
        MatchRecord {
            seq,
            placements: order
                .iter()
                .enumerate()
                .map(|(i, &gen)| Placement {
                    gen,
                    seat: i as u32,
                    rank: i as u32 + 1,
                    death_turn: None,
                })
                .collect(),
            turns: 100,
            sims: 0,
            wall_seconds: 0.0,
            finished_unix_ms: 0,
        }
    }

    /// `wins` two-player games where `gen` beats `opponent`, then `losses`
    /// the other way (one record per game, like the league writes).
    fn head_to_head(gen: u32, opponent: u32, wins: u32, losses: u32) -> Vec<MatchRecord> {
        let mut records: Vec<MatchRecord> = (0..wins)
            .map(|i| game(i as u64, &[gen, opponent]))
            .collect();
        records.extend((0..losses).map(|i| game((wins + i) as u64, &[opponent, gen])));
        records
    }

    /// Sample one full finishing order from the Plackett–Luce model with the
    /// given (gen, strength) pairs: repeatedly pick the next finisher with
    /// probability proportional to strength among those remaining.
    fn sample_order(strengths: &[(u32, f64)], rng: &mut Xoshiro256PlusPlus) -> Vec<u32> {
        let mut remaining = strengths.to_vec();
        let mut order = Vec::with_capacity(remaining.len());
        while !remaining.is_empty() {
            let total: f64 = remaining.iter().map(|&(_, s)| s).sum();
            let mut pick = rng.gen_range(0.0..total);
            let mut chosen = remaining.len() - 1;
            for (i, &(_, s)) in remaining.iter().enumerate() {
                pick -= s;
                if pick <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            order.push(remaining.remove(chosen).0);
        }
        order
    }

    fn shuffled(gens: &[u32], rng: &mut Xoshiro256PlusPlus) -> Vec<u32> {
        let mut v = gens.to_vec();
        for i in (1..v.len()).rev() {
            v.swap(i, rng.gen_range(0..=i));
        }
        v
    }

    #[test]
    fn two_player_fit_matches_bradley_terry_closed_form() {
        // With an 8–2 head-to-head record plus the fit's built-in half virtual
        // tie (0.25 wins each way), the Bradley–Terry MLE odds are exactly
        // (8 + 0.25) : (2 + 0.25), so the Elo gap has a closed form the MM
        // iteration must land on.
        let pool = vec![0, 5];
        let records = head_to_head(5, 0, 8, 2);
        let elo = fit_ratings(&pool, &records);
        let expected = 400.0 * (8.25f64 / 2.25).log10();
        assert!(
            (elo[&5] - expected).abs() < 0.5,
            "fitted {} vs closed form {expected}",
            elo[&5]
        );
    }

    #[test]
    fn fit_recovers_known_strengths_from_simulated_games() {
        // Simulate 3000 four-player games from known Plackett–Luce strengths
        // and check the fit recovers every player's true Elo. This is the
        // end-to-end accuracy test: sampling noise at 3000 games is ~10 Elo.
        let true_elo = [(0u32, 0.0f64), (10, 100.0), (20, 200.0), (30, 300.0)];
        let strengths: Vec<(u32, f64)> = true_elo
            .iter()
            .map(|&(g, e)| (g, 10f64.powf(e / 400.0)))
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let records: Vec<MatchRecord> = (0..3000)
            .map(|seq| game(seq, &sample_order(&strengths, &mut rng)))
            .collect();
        let pool: Vec<u32> = true_elo.iter().map(|&(g, _)| g).collect();
        let elo = fit_ratings(&pool, &records);
        for &(g, expected) in &true_elo {
            assert!(
                (elo[&g] - expected).abs() < 25.0,
                "gen {g}: fitted {:.1} vs true {expected}",
                elo[&g]
            );
        }
    }

    #[test]
    fn fit_is_converged_and_scale_invariant() {
        // A maximum-likelihood fit must not depend on the absolute number of
        // identical observations: doubling every record leaves the optimum in
        // place (only the fixed virtual ties shift relative weight, negligible
        // at this data size). A drifting result here means the MM iteration
        // ran out of its fixed iteration budget before converging.
        let strengths = [(0u32, 1.0f64), (10, 1.6), (20, 2.6), (30, 4.2)];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let records: Vec<MatchRecord> = (0..600)
            .map(|seq| game(seq, &sample_order(&strengths, &mut rng)))
            .collect();
        let doubled: Vec<MatchRecord> = records.iter().chain(records.iter()).cloned().collect();
        let pool = vec![0, 10, 20, 30];
        let once = fit_ratings(&pool, &records);
        let twice = fit_ratings(&pool, &doubled);
        for g in &pool {
            assert!(
                (once[g] - twice[g]).abs() < 3.0,
                "gen {g}: {:.2} vs doubled {:.2}",
                once[g],
                twice[g]
            );
        }
    }

    #[test]
    fn duplicate_seats_in_one_game_read_as_repeated_comparisons() {
        // Early league games seat the same player twice (pool smaller than the
        // table): two floodfill seats tying for first over two gen-0 seats.
        // That must fit as floodfill > gen 0 with every rating finite.
        let pool = vec![0, HEURISTIC_GEN];
        let records: Vec<MatchRecord> = (0..5)
            .map(|seq| {
                let mut r = game(seq, &[HEURISTIC_GEN, HEURISTIC_GEN, 0, 0]);
                r.placements[1].rank = 1; // both floodfill seats share first
                r
            })
            .collect();
        let elo = fit_ratings(&pool, &records);
        assert!(
            elo[&0].is_finite() && elo[&HEURISTIC_GEN].is_finite(),
            "{elo:?}"
        );
        assert!(
            elo[&HEURISTIC_GEN] > 100.0,
            "floodfill swept every game: {elo:?}"
        );
    }

    #[test]
    fn plackett_luce_rates_the_full_finishing_order_not_the_win_rate() {
        // A "win or die" player X against three identical steady opponents: X
        // finishes first in half its games and dead last in the rest, so it
        // wins 3x as often as any opponent (50% vs ~17%) — yet the analytic
        // Plackett–Luce MLE puts X ~74 Elo BELOW the steady players, because
        // finishing last loses all three elimination stages while a win only
        // takes the first. This is intended model semantics — league Elo
        // orders by full-ranking strength, not rank-1 rate — and it is exactly
        // why a high-win-rate bimodal player (the flood-fill baseline) can sit
        // mid-table under nets with fewer outright wins. If this test starts
        // failing, the rating semantics have changed; update the leaderboard
        // copy accordingly.
        let pool = vec![0, 10, 20, 99];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let records: Vec<MatchRecord> = (0..1200)
            .map(|seq| {
                let others = shuffled(&[0, 10, 20], &mut rng);
                let order: Vec<u32> = if rng.gen_bool(0.5) {
                    std::iter::once(99).chain(others).collect()
                } else {
                    others.into_iter().chain(std::iter::once(99)).collect()
                };
                game(seq, &order)
            })
            .collect();
        let elo = fit_ratings(&pool, &records);
        for g in [0, 10, 20] {
            assert!(
                elo[&99] < elo[&g] - 20.0,
                "win-or-die player must rate below steady equals: {elo:?}"
            );
        }
        assert!(
            elo[&99] > -160.0,
            "penalty should stay near the analytic ~-74: {elo:?}"
        );
    }

    #[test]
    fn fit_matches_bradley_terry_for_two_players() {
        let pool = vec![0, 5];
        let records = head_to_head(5, 0, 8, 2);
        let elo = fit_ratings(&pool, &records);
        assert_eq!(elo[&0], 0.0);
        assert!(elo[&5] > 100.0 && elo[&5] < 400.0, "{elo:?}");
    }

    #[test]
    fn fit_orders_multiplayer_rankings_transitively() {
        let pool = vec![0, 5, 10, 15];
        // Consistent finishing order across several 4-player games.
        let records: Vec<MatchRecord> = (0..6).map(|s| game(s, &[15, 10, 5, 0])).collect();
        let elo = fit_ratings(&pool, &records);
        assert_eq!(elo[&0], 0.0);
        assert!(elo[&5] > elo[&0], "{elo:?}");
        assert!(elo[&10] > elo[&5], "{elo:?}");
        assert!(elo[&15] > elo[&10], "{elo:?}");
    }

    #[test]
    fn fit_stays_finite_with_ties_and_perfect_records() {
        let pool = vec![0, 5, 10];
        let mut tied = game(0, &[10, 5, 0]);
        // Make 5 and 0 tie for second.
        tied.placements[2].rank = 2;
        let records = vec![tied, game(1, &[10, 5, 0])];
        let elo = fit_ratings(&pool, &records);
        for g in &pool {
            assert!(elo[g].is_finite() && elo[g].abs() < 2000.0, "{elo:?}");
        }
    }

    #[test]
    fn picks_distinct_players_and_prioritizes_the_unrated() {
        let pool = vec![0, 5, 10, 15, 20];
        // gen 20 has never played; the rest share several games.
        let records: Vec<MatchRecord> = (0..5).map(|s| game(s, &[15, 10, 5, 0])).collect();
        let ratings = fit_ratings(&pool, &records);
        let mut with_new = 0;
        for seq in 0..20 {
            let players = pick_players(&pool, &records, &HashMap::new(), &ratings, 4, seq);
            assert_eq!(players.len(), 4);
            let mut dedup = players.clone();
            dedup.sort_unstable();
            dedup.dedup();
            assert_eq!(dedup.len(), 4, "players must be distinct: {players:?}");
            if players.contains(&20) {
                with_new += 1;
            }
        }
        assert!(
            with_new > 12,
            "unrated entrant only drawn {with_new}/20 times"
        );
    }

    #[test]
    fn heuristic_baseline_plays_regularly() {
        let pool = vec![0, 5, 10, 15, HEURISTIC_GEN];
        // Give everyone (including the baseline) some history so the forced
        // inclusion — not just new-player uncertainty — is what keeps it in.
        let mut records: Vec<MatchRecord> = (0..5).map(|s| game(s, &[15, 10, 5, 0])).collect();
        records.extend((5..10).map(|s| game(s, &[15, HEURISTIC_GEN, 5, 0])));
        let ratings = fit_ratings(&pool, &records);
        let with_baseline = (0..60)
            .filter(|&seq| pick_players(&pool, &records, &HashMap::new(), &ratings, 4, seq).contains(&HEURISTIC_GEN))
            .count();
        assert!(
            with_baseline > 15,
            "floodfill baseline only drawn {with_baseline}/60 times"
        );
    }

    #[test]
    fn api_player_ids_are_stable_across_restarts_and_reorders() {
        let dir =
            std::env::temp_dir().join(format!("snek3-eval-registry-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut cfg = crate::config::RunConfig {
            league_api_players: vec![
                "alpha=http://10.0.0.1:8000".into(),
                "beta=http://10.0.0.2:8000/".into(),
                "broken-entry-no-url".into(),
            ],
            ..Default::default()
        };
        let (reg, errors) = load_external_registry(&cfg, &dir);
        assert_eq!(errors.len(), 1, "{errors:?}");
        let alpha = reg.active.iter().find(|p| p.name == "alpha").unwrap().gen;
        let beta = reg.active.iter().find(|p| p.name == "beta").unwrap().gen;
        assert!(is_external(alpha) && is_external(beta));
        assert_ne!(alpha, beta);
        assert_eq!(
            reg.active.iter().find(|p| p.name == "beta").unwrap().spec,
            "http://10.0.0.2:8000",
            "trailing slash trimmed"
        );
        // Built-ins always present.
        assert!(reg.active.iter().any(|p| p.gen == HEURISTIC_GEN));
        assert!(reg.active.iter().any(|p| p.gen == VORONOI_GEN));

        // Reorder + drop one: survivors keep their ids, the dropped player
        // stays known (for naming history) but not active.
        cfg.league_api_players = vec!["beta=http://10.0.0.2:9999".into()];
        let (reg2, _) = load_external_registry(&cfg, &dir);
        assert_eq!(
            reg2.active.iter().find(|p| p.name == "beta").unwrap().gen,
            beta
        );
        assert!(!reg2.active.iter().any(|p| p.name == "alpha"));
        assert_eq!(reg2.name(alpha), Some("alpha"));

        // Re-adding alpha restores its original id.
        cfg.league_api_players = vec![
            "beta=http://10.0.0.2:9999".into(),
            "alpha=http://10.0.0.1:8000".into(),
        ];
        let (reg3, _) = load_external_registry(&cfg, &dir);
        assert_eq!(
            reg3.active.iter().find(|p| p.name == "alpha").unwrap().gen,
            alpha
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn forced_external_rotates_between_baselines() {
        let pool = vec![0, 5, 10, 15, HEURISTIC_GEN, VORONOI_GEN];
        let mut records: Vec<MatchRecord> = (0..5).map(|s| game(s, &[15, 10, 5, 0])).collect();
        records.extend((5..10).map(|s| game(s, &[15, HEURISTIC_GEN, VORONOI_GEN, 0])));
        let ratings = fit_ratings(&pool, &records);
        let (mut with_ff, mut with_vor) = (0, 0);
        for seq in 0..120 {
            let players = pick_players(&pool, &records, &HashMap::new(), &ratings, 4, seq);
            with_ff += players.contains(&HEURISTIC_GEN) as u32;
            with_vor += players.contains(&VORONOI_GEN) as u32;
        }
        assert!(with_ff > 15, "floodfill only drawn {with_ff}/120");
        assert!(with_vor > 15, "voronoi only drawn {with_vor}/120");
    }

    #[test]
    fn small_pools_use_every_member() {
        let pool = vec![0, 5];
        let records = Vec::new();
        let ratings = fit_ratings(&pool, &records);
        let players = pick_players(&pool, &records, &HashMap::new(), &ratings, 4, 0);
        let mut sorted = players.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 5]);
    }

    #[test]
    fn prune_keeps_newest_game_files() {
        let dir =
            std::env::temp_dir().join(format!("snek3-eval-prune-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for seq in 0..10u64 {
            std::fs::write(dir.join(format!("game_{seq:06}.json")), b"{}").unwrap();
        }
        std::fs::write(dir.join("summary.jsonl"), b"").unwrap();
        prune_game_files(&dir, 3);
        let mut left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(
            left,
            vec![
                "game_000007.json",
                "game_000008.json",
                "game_000009.json",
                "summary.jsonl"
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
