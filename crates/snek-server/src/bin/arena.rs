//! Head-to-head evaluation arena: two nets, the exact `snek-core` rules engine,
//! and the same `serve_move_until` search the live server uses — all in-process,
//! no HTTP. Rules parity with the official Go engine is inherited from
//! `snek-core` (the same `Board::step_and_spawn` self-play and serving run on).
//!
//! Each side gets its own pool of worker threads pinned to disjoint CPU cores,
//! one worker (and one `Net`) per core, so the two players do not steal cycles
//! from each other or from a concurrently running training job. Parallel games
//! = min(cores per side); within a turn both sides search concurrently.
//!
//! Fairness/determinism:
//! - The default budget is a fixed simulation count per move (`--sims`): the
//!   serving search is strict-argmax DUCT with no noise, so every game is
//!   deterministic given its seed and hardware-independent. `--time-ms`
//!   switches to wall-clock budgets like live play (that is where core pinning
//!   matters most).
//! - Games run in mirrored pairs: each seed is played twice with seats swapped,
//!   cancelling spawn/seat asymmetry.
//! - CPU-only by default; `--gpu` opts in (don't use while training runs).
//!
//! Example:
//!   arena --a runs/X/models/gen_0300.pt --b runs/X/models/gen_0400.pt \
//!         --games 200 --sims 1000 --cores-a 0-3 --cores-b 4-7

use std::path::Path;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::Serialize;
use serde_json::json;
use snek_core::{standard_start, Board, Move};
use snek_server::{serve_move_until_diagnostics, Config, Net};

/// Stand-in "no deadline" horizon for fixed-sims mode (matches `serve_move`).
const SIMS_DEADLINE: Duration = Duration::from_secs(3600);

#[derive(Clone, Copy)]
enum Budget {
    Sims(usize),
    TimeMs(u64),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    A,
    B,
}

impl Side {
    fn label(self) -> &'static str {
        match self {
            Side::A => "A",
            Side::B => "B",
        }
    }
}

struct Args {
    model_a: String,
    model_b: String,
    name_a: Option<String>,
    name_b: Option<String>,
    games: usize,
    snakes: usize,
    budget: Budget,
    cores_a: Option<Vec<usize>>,
    cores_b: Option<Vec<usize>>,
    cores_per_side: usize,
    parallel: Option<usize>,
    board: i8,
    seed: u64,
    max_turns: u32,
    c_puct: f32,
    draw_value: f32,
    eval_chunk: usize,
    leaves_per_sim: usize,
    virtual_loss: f32,
    gpu: bool,
    out: Option<String>,
    record: Option<String>,
    record_gen: u32,
}

/// On-disk mirror of the recorded-game schema in `snek-train`'s `sample.rs`
/// (`GameFileJson` and friends), so eval games render through the exact same
/// frontend viewer as self-play sample games. Keep the two in sync.
#[derive(Serialize)]
struct GameFileJson {
    gen: u32,
    config: serde_json::Value,
    games: Vec<GameJson>,
}

#[derive(Serialize)]
struct GameJson {
    frames: Vec<FrameJson>,
    winner: Option<i32>,
    num_turns: u32,
}

#[derive(Serialize, Clone)]
struct FrameJson {
    turn: u32,
    width: i32,
    height: i32,
    food: Vec<[i32; 2]>,
    hazards: Vec<[i32; 2]>,
    snakes: Vec<SnakeJson>,
}

#[derive(Serialize, Clone)]
struct SnakeJson {
    alive: bool,
    body: Vec<[i32; 2]>,
    health: i32,
    chosen_move: u32,
    policy: Vec<f64>,
    play_policy: Vec<f64>,
    value: f64,
}

fn usage() -> ! {
    eprintln!(
        "arena: play two nets head to head with the in-process rules engine.

usage: arena --a <model> --b <model> [options]

required:
  --a PATH            side-A model weights (.safetensors/.pt VarStore)
  --b PATH            side-B model weights

match:
  --games N           total games, played as mirrored seat-swapped pairs (100)
  --snakes N          snakes per game; seats alternate A,B,A,B… and a side
                      wins when only its snakes remain (2)
  --sims N            fixed MCTS sims per move; deterministic (1000)
  --time-ms MS        wall-clock budget per move instead of --sims
  --board N           board side length (11)
  --seed N            base seed; pair p uses seed+p (1)
  --max-turns N       turn cutoff, counted as a draw (500)

cpu / pinning:
  --cores-a SPEC      cores for side A, e.g. 0-3 or 0,2,4 (auto)
  --cores-b SPEC      cores for side B, disjoint from --cores-a (auto)
  --cores-per-side N  cores per side when --cores-a/b not given (2)
  --parallel N        cap concurrent games (min of side core counts)
  --gpu               allow CUDA (default forces CPU so training is untouched)

search (defaults match live serving):
  --c-puct F          PUCT exploration constant (1.5)
  --draw-value F      terminal draw value (-0.25)
  --leaves-per-sim N  virtual-loss batch width (8)
  --virtual-loss F    virtual-loss magnitude (1.0)
  --eval-chunk N      max rows per net forward (4096)

output:
  --name-a / --name-b display names (default: model file stem)
  --out PATH          write full results as JSON
  --record PATH       record every game (frames + search readout) as a
                      viewer-compatible games file (same schema as the
                      trainer's games/gen_NNNN.json)
  --record-gen N      the generation label stamped into --record (0)"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut model_a = None;
    let mut model_b = None;
    let mut name_a = None;
    let mut name_b = None;
    let mut games = 100usize;
    let mut snakes = 2usize;
    let mut sims = 1000usize;
    let mut time_ms: Option<u64> = None;
    let mut cores_a = None;
    let mut cores_b = None;
    let mut cores_per_side = 2usize;
    let mut parallel = None;
    let mut board = 11i8;
    let mut seed = 1u64;
    let mut max_turns = 500u32;
    let mut c_puct = 1.5f32;
    let mut draw_value = -0.25f32;
    let mut eval_chunk = 4096usize;
    let mut leaves_per_sim = 8usize;
    let mut virtual_loss = 1.0f32;
    let mut gpu = false;
    let mut out = None;
    let mut record = None;
    let mut record_gen = 0u32;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| -> String {
            it.next().unwrap_or_else(|| {
                eprintln!("arena: {name} requires a value");
                std::process::exit(2);
            })
        };
        match arg.as_str() {
            "--a" => model_a = Some(val("--a")),
            "--b" => model_b = Some(val("--b")),
            "--name-a" => name_a = Some(val("--name-a")),
            "--name-b" => name_b = Some(val("--name-b")),
            "--games" => games = parse_num(&val("--games"), "--games"),
            "--snakes" => snakes = parse_num(&val("--snakes"), "--snakes"),
            "--sims" => sims = parse_num(&val("--sims"), "--sims"),
            "--time-ms" => time_ms = Some(parse_num(&val("--time-ms"), "--time-ms")),
            "--cores-a" => cores_a = Some(parse_core_spec(&val("--cores-a"))),
            "--cores-b" => cores_b = Some(parse_core_spec(&val("--cores-b"))),
            "--cores-per-side" => {
                cores_per_side =
                    parse_num::<usize>(&val("--cores-per-side"), "--cores-per-side").max(1)
            }
            "--parallel" => parallel = Some(parse_num(&val("--parallel"), "--parallel")),
            "--board" => board = parse_num(&val("--board"), "--board"),
            "--seed" => seed = parse_num(&val("--seed"), "--seed"),
            "--max-turns" => max_turns = parse_num(&val("--max-turns"), "--max-turns"),
            "--c-puct" => c_puct = parse_num(&val("--c-puct"), "--c-puct"),
            "--draw-value" => draw_value = parse_num(&val("--draw-value"), "--draw-value"),
            "--eval-chunk" => eval_chunk = parse_num(&val("--eval-chunk"), "--eval-chunk"),
            "--leaves-per-sim" => {
                leaves_per_sim = parse_num(&val("--leaves-per-sim"), "--leaves-per-sim")
            }
            "--virtual-loss" => virtual_loss = parse_num(&val("--virtual-loss"), "--virtual-loss"),
            "--gpu" => gpu = true,
            "--out" => out = Some(val("--out")),
            "--record" => record = Some(val("--record")),
            "--record-gen" => record_gen = parse_num(&val("--record-gen"), "--record-gen"),
            "--help" | "-h" => usage(),
            other => {
                eprintln!("arena: unknown argument {other}");
                usage();
            }
        }
    }
    let (Some(model_a), Some(model_b)) = (model_a, model_b) else {
        eprintln!("arena: --a and --b are required");
        usage();
    };
    if games == 0 {
        eprintln!("arena: --games must be > 0");
        std::process::exit(2);
    }
    if !(2..=snek_core::MAX_SNAKES).contains(&snakes) {
        eprintln!("arena: --snakes must be 2..={}", snek_core::MAX_SNAKES);
        std::process::exit(2);
    }
    Args {
        model_a,
        model_b,
        name_a,
        name_b,
        games,
        snakes,
        budget: match time_ms {
            Some(ms) => Budget::TimeMs(ms.max(1)),
            None => Budget::Sims(sims.max(1)),
        },
        cores_a,
        cores_b,
        cores_per_side,
        parallel,
        board,
        seed,
        max_turns,
        c_puct,
        draw_value,
        eval_chunk,
        leaves_per_sim,
        virtual_loss,
        gpu,
        out,
        record,
        record_gen,
    }
}

fn parse_num<T: std::str::FromStr>(s: &str, flag: &str) -> T {
    s.parse().unwrap_or_else(|_| {
        eprintln!("arena: invalid value {s:?} for {flag}");
        std::process::exit(2);
    })
}

/// "0-3", "0,2,4", "0-1,6-7" → sorted, deduped core id list.
fn parse_core_spec(spec: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = parse_num(lo, "core spec");
            let hi: usize = parse_num(hi, "core spec");
            if lo > hi {
                eprintln!("arena: bad core range {part:?}");
                std::process::exit(2);
            }
            out.extend(lo..=hi);
        } else {
            out.push(parse_num(part, "core spec"));
        }
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        eprintln!("arena: empty core spec {spec:?}");
        std::process::exit(2);
    }
    out
}

/// Resolve the core lists for both sides. Explicit specs win; otherwise carve
/// disjoint blocks of `cores_per_side` from the machine's available cores.
/// Returns `None` per side when pinning is unavailable (workers run unpinned).
fn resolve_cores(args: &Args) -> (Option<Vec<usize>>, Option<Vec<usize>>) {
    let available: Vec<usize> = core_affinity::get_core_ids()
        .unwrap_or_default()
        .into_iter()
        .map(|c| c.id)
        .collect();
    if available.is_empty() && (args.cores_a.is_none() || args.cores_b.is_none()) {
        eprintln!("arena: warning: cannot enumerate CPU cores; workers will be unpinned");
        return (args.cores_a.clone(), args.cores_b.clone());
    }
    let taken = |used: &[usize], n: usize| -> Vec<usize> {
        available
            .iter()
            .copied()
            .filter(|c| !used.contains(c))
            .take(n)
            .collect()
    };
    let a = match &args.cores_a {
        Some(a) => a.clone(),
        None => {
            let free = taken(args.cores_b.as_deref().unwrap_or(&[]), args.cores_per_side);
            if free.is_empty() {
                eprintln!("arena: no free cores for side A");
                std::process::exit(2);
            }
            free
        }
    };
    let b = match &args.cores_b {
        Some(b) => b.clone(),
        None => {
            let free = taken(&a, args.cores_per_side);
            if free.is_empty() {
                eprintln!("arena: no free cores for side B");
                std::process::exit(2);
            }
            free
        }
    };
    if a.iter().any(|c| b.contains(c)) {
        eprintln!("arena: warning: --cores-a and --cores-b overlap; the sides will contend");
    }
    (Some(a), Some(b))
}

struct MoveJob {
    board: Board,
    me: usize,
}

/// A worker's answer for one position: the move plus the search readout for
/// snake `me` (visit-count policy and root value), used by game recording.
struct MoveInfo {
    move_index: usize,
    policy: [f32; 4],
    value: f32,
}

/// One pinned thread owning one `Net`. Each match runner has a dedicated
/// worker per side, so plain channels (no locking) carry the per-turn jobs.
struct Worker {
    job_tx: mpsc::Sender<MoveJob>,
    move_rx: mpsc::Receiver<MoveInfo>,
}

fn spawn_worker(
    label: String,
    model: String,
    core: Option<usize>,
    cfg: Config,
    budget: Budget,
) -> (Worker, JoinHandle<()>) {
    let (job_tx, job_rx) = mpsc::channel::<MoveJob>();
    let (move_tx, move_rx) = mpsc::channel::<MoveInfo>();
    let handle = std::thread::Builder::new()
        .name(label.clone())
        .spawn(move || {
            if let Some(id) = core {
                if !core_affinity::set_for_current(core_affinity::CoreId { id }) {
                    eprintln!("arena: warning: failed to pin {label} to core {id}");
                }
            }
            let mut net = Net::load(&model)
                .unwrap_or_else(|e| panic!("arena: {label}: failed to load {model}: {e}"));
            while let Ok(job) = job_rx.recv() {
                let deadline = match budget {
                    Budget::Sims(_) => Instant::now() + SIMS_DEADLINE,
                    Budget::TimeMs(ms) => Instant::now() + Duration::from_millis(ms),
                };
                let d = serve_move_until_diagnostics(&mut net, &cfg, &job.board, job.me, deadline);
                let n = job.board.snakes.len();
                let mut policy = [0f32; 4];
                let mut value = 0f32;
                if d.diagnostics.root_policy.len() == n * 4 {
                    policy.copy_from_slice(&d.diagnostics.root_policy[job.me * 4..job.me * 4 + 4]);
                    value = d
                        .diagnostics
                        .root_values
                        .get(job.me)
                        .copied()
                        .unwrap_or(0.0);
                } else {
                    // Forced move / fallback paths skip the search: record the
                    // decision as a one-hot policy with no value estimate.
                    policy[d.move_index] = 1.0;
                }
                let info = MoveInfo {
                    move_index: d.move_index,
                    policy,
                    value,
                };
                if move_tx.send(info).is_err() {
                    break;
                }
            }
        })
        .expect("spawn arena worker");
    (Worker { job_tx, move_rx }, handle)
}

struct GameOutcome {
    game_index: usize,
    seed: u64,
    /// Seat parity: `true` means side A holds the even snake indices.
    a_first: bool,
    winner: Option<Side>,
    winner_snake: Option<usize>,
    turns: u32,
    reason: &'static str,
    wall_ms: u64,
    /// Recorded frames (pre-step, like self-play sample games); empty unless
    /// --record is set.
    frames: Vec<FrameJson>,
}

/// Which side plays snake `i` given the seat parity: seats alternate
/// A,B,A,B… (`a_first`) or B,A,B,A… — the mirrored half of a game pair.
fn side_of(i: usize, a_first: bool) -> Side {
    if i.is_multiple_of(2) == a_first {
        Side::A
    } else {
        Side::B
    }
}

#[allow(clippy::too_many_arguments)]
fn play_game(
    game_index: usize,
    seed: u64,
    a_first: bool,
    snakes: usize,
    board_size: i8,
    max_turns: u32,
    record: bool,
    a: &Worker,
    b: &Worker,
) -> GameOutcome {
    let start = Instant::now();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut board = standard_start(board_size, board_size, snakes, &mut rng);
    let mut frames = Vec::new();
    // The game ends for eval purposes as soon as one side has no snakes left —
    // a same-side endgame (two A snakes fighting) says nothing about A vs B.
    let alive_sides = |board: &Board| {
        let mut a_alive = false;
        let mut b_alive = false;
        for (i, s) in board.snakes.iter().enumerate() {
            if s.alive() {
                match side_of(i, a_first) {
                    Side::A => a_alive = true,
                    Side::B => b_alive = true,
                }
            }
        }
        (a_alive, b_alive)
    };
    while board.turn < max_turns {
        let (a_alive, b_alive) = alive_sides(&board);
        if !(a_alive && b_alive) {
            break;
        }
        let n = board.snakes.len();
        let mut moves = vec![Move::Up; n];
        let mut infos: Vec<Option<MoveInfo>> = (0..n).map(|_| None).collect();
        let mut pending: Vec<(usize, Side)> = Vec::new();
        // Fan out both sides' searches before collecting either, so A and B
        // compute the turn concurrently on their pinned cores. Multiple snakes
        // per side queue serially on that side's worker.
        for i in 0..n {
            if !board.snakes[i].alive() {
                continue;
            }
            let side = side_of(i, a_first);
            let worker = if side == Side::A { a } else { b };
            worker
                .job_tx
                .send(MoveJob {
                    board: board.clone(),
                    me: i,
                })
                .expect("arena worker exited");
            pending.push((i, side));
        }
        for (i, side) in pending {
            let worker = if side == Side::A { a } else { b };
            let info = worker.move_rx.recv().expect("arena worker exited");
            moves[i] = Move::from_index(info.move_index);
            infos[i] = Some(info);
        }
        if record {
            frames.push(frame_from_board(&board, &moves, &infos));
        }
        board.step_and_spawn(&moves, &mut rng);
    }
    let (a_alive, b_alive) = alive_sides(&board);
    let winner = match (a_alive, b_alive) {
        (true, false) => Some(Side::A),
        (false, true) => Some(Side::B),
        _ => None, // both alive (max_turns cutoff) or both dead: a draw
    };
    // For the recorded game the "winner" is a snake index (the viewer colors by
    // snake): the first surviving snake of the winning side.
    let winner_snake = winner.and_then(|side| {
        (0..board.snakes.len()).find(|&i| board.snakes[i].alive() && side_of(i, a_first) == side)
    });
    GameOutcome {
        game_index,
        seed,
        a_first,
        winner,
        winner_snake,
        turns: board.turn,
        reason: if a_alive && b_alive {
            "max_turns"
        } else {
            "elimination"
        },
        wall_ms: start.elapsed().as_millis() as u64,
        frames,
    }
}

/// Capture the pre-step state of the board as a recordable frame, mirroring
/// self-play's `frame_from_board`: every snake's body/health plus the search
/// readout for the snakes that moved this turn. Eval plays argmax (no
/// sampling), so `play_policy` is the one-hot of the chosen move.
fn frame_from_board(board: &Board, moves: &[Move], infos: &[Option<MoveInfo>]) -> FrameJson {
    let coord = |p: &snek_core::Point| [p.x as i32, p.y as i32];
    let snakes = (0..board.snakes.len())
        .map(|i| {
            let snake = &board.snakes[i];
            let (policy, play_policy, value) = match &infos[i] {
                Some(info) => {
                    let mut one_hot = vec![0.0f64; 4];
                    one_hot[info.move_index] = 1.0;
                    (
                        info.policy.iter().map(|&v| v as f64).collect(),
                        one_hot,
                        info.value as f64,
                    )
                }
                None => (Vec::new(), Vec::new(), 0.0),
            };
            SnakeJson {
                alive: snake.alive(),
                body: snake.body.iter().map(|p| coord(&p)).collect(),
                health: snake.health as i32,
                chosen_move: moves[i].index() as u32,
                policy,
                play_policy,
                value,
            }
        })
        .collect();
    FrameJson {
        turn: board.turn,
        width: board.width as i32,
        height: board.height as i32,
        food: board.food.iter().map(coord).collect(),
        hazards: board.hazards.iter().map(coord).collect(),
        snakes,
    }
}

/// Mean score for A (win 1, draw 0.5), its 95% CI half-width, and the Elo
/// difference implied by the mean and by the CI endpoints.
fn summarize(a_wins: usize, b_wins: usize, draws: usize) -> (f64, f64, f64, f64, f64) {
    let n = (a_wins + b_wins + draws) as f64;
    let score = (a_wins as f64 + 0.5 * draws as f64) / n;
    let var = (a_wins as f64 * (1.0 - score).powi(2)
        + draws as f64 * (0.5 - score).powi(2)
        + b_wins as f64 * score.powi(2))
        / (n - 1.0).max(1.0);
    let ci = 1.96 * (var / n).sqrt();
    let elo = |s: f64| {
        let s = s.clamp(1e-3, 1.0 - 1e-3);
        400.0 * (s / (1.0 - s)).log10()
    };
    (score, ci, elo(score), elo(score - ci), elo(score + ci))
}

fn model_name(explicit: &Option<String>, path: &str, fallback: &str) -> String {
    explicit.clone().unwrap_or_else(|| {
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| fallback.to_string())
    })
}

fn main() {
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);

    let args = parse_args();
    if !args.gpu && std::env::var("SNEK_CPU_ONLY").is_err() {
        std::env::set_var("SNEK_CPU_ONLY", "1");
    }

    let mut name_a = model_name(&args.name_a, &args.model_a, "A");
    let mut name_b = model_name(&args.name_b, &args.model_b, "B");
    // Trainer checkpoints are all called net.safetensors — when the stems
    // collide, the parent directory (the run id) is the distinguishing name.
    if name_a == name_b {
        let parent = |p: &str| {
            Path::new(p)
                .parent()
                .and_then(Path::file_name)
                .and_then(|s| s.to_str())
                .map(str::to_string)
        };
        if args.name_a.is_none() {
            name_a = parent(&args.model_a).unwrap_or(name_a);
        }
        if args.name_b.is_none() {
            name_b = parent(&args.model_b).unwrap_or(name_b);
        }
    }
    let (cores_a, cores_b) = resolve_cores(&args);
    let pairs = args.games.div_ceil(2);
    let slots = cores_a
        .as_ref()
        .map_or(args.cores_per_side, Vec::len)
        .min(cores_b.as_ref().map_or(args.cores_per_side, Vec::len));
    let parallel = args.parallel.unwrap_or(slots).min(slots).min(pairs).max(1);

    let cfg = Config {
        max_sims: match args.budget {
            Budget::Sims(n) => n,
            Budget::TimeMs(_) => usize::MAX,
        },
        c_puct: args.c_puct,
        draw_value: args.draw_value,
        eval_chunk: args.eval_chunk,
        leaves_per_sim: args.leaves_per_sim,
        virtual_loss: args.virtual_loss,
    };
    let budget_desc = match args.budget {
        Budget::Sims(n) => format!("{n} sims/move"),
        Budget::TimeMs(ms) => format!("{ms} ms/move"),
    };
    let core_desc = |c: &Option<Vec<usize>>| {
        c.as_ref()
            .map(|v| format!("{v:?}"))
            .unwrap_or_else(|| "unpinned".into())
    };
    eprintln!(
        "arena: A={name_a} ({}) cores={} | B={name_b} ({}) cores={}",
        args.model_a,
        core_desc(&cores_a),
        args.model_b,
        core_desc(&cores_b),
    );
    eprintln!(
        "arena: {games} games ({pairs} mirrored pairs), {snakes} snakes, {budget_desc}, board {board}x{board}, parallel {parallel}, {mode}",
        games = args.games,
        snakes = args.snakes,
        board = args.board,
        mode = if args.gpu { "gpu allowed" } else { "cpu only" }
    );

    // One worker (own Net, own core) per side per match slot.
    let core_for = |cores: &Option<Vec<usize>>, i: usize| cores.as_ref().map(|v| v[i]);
    let mut runners = Vec::new();
    let (result_tx, result_rx) = mpsc::channel::<GameOutcome>();
    for slot in 0..parallel {
        let (worker_a, _ha) = spawn_worker(
            format!("arena-a{slot}"),
            args.model_a.clone(),
            core_for(&cores_a, slot),
            cfg.clone(),
            args.budget,
        );
        let (worker_b, _hb) = spawn_worker(
            format!("arena-b{slot}"),
            args.model_b.clone(),
            core_for(&cores_b, slot),
            cfg.clone(),
            args.budget,
        );
        let tx = result_tx.clone();
        let record = args.record.is_some();
        let (games, base_seed, snakes, board, max_turns) = (
            args.games,
            args.seed,
            args.snakes,
            args.board,
            args.max_turns,
        );
        runners.push(std::thread::spawn(move || {
            let mut p = slot;
            while p < pairs {
                let seed = base_seed.wrapping_add(p as u64);
                for (k, a_first) in [true, false].into_iter().enumerate() {
                    let game_index = 2 * p + k;
                    if game_index >= games {
                        break;
                    }
                    let out = play_game(
                        game_index, seed, a_first, snakes, board, max_turns, record, &worker_a,
                        &worker_b,
                    );
                    if tx.send(out).is_err() {
                        return;
                    }
                }
                p += parallel;
            }
            // Dropping the workers' job senders shuts their threads down.
        }));
    }
    drop(result_tx);

    let started = Instant::now();
    let (mut a_wins, mut b_wins, mut draws) = (0usize, 0usize, 0usize);
    let mut games: Vec<GameOutcome> = Vec::with_capacity(args.games);
    while let Ok(out) = result_rx.recv() {
        match out.winner {
            Some(Side::A) => a_wins += 1,
            Some(Side::B) => b_wins += 1,
            None => draws += 1,
        }
        let done = a_wins + b_wins + draws;
        eprintln!(
            "arena: [{done:>4}/{}] game {:04} winner={} turns={} ({:.1}s) | {name_a} {a_wins} - {b_wins} {name_b}, {draws} draws",
            args.games,
            out.game_index,
            out.winner.map(Side::label).unwrap_or("draw"),
            out.turns,
            out.wall_ms as f64 / 1000.0,
        );
        games.push(out);
    }
    for r in runners {
        let _ = r.join();
    }

    if games.is_empty() {
        eprintln!("arena: no games completed");
        std::process::exit(1);
    }
    games.sort_by_key(|g| g.game_index);
    let (score, ci, elo, elo_lo, elo_hi) = summarize(a_wins, b_wins, draws);
    let avg_turns = games.iter().map(|g| g.turns as f64).sum::<f64>() / games.len() as f64;
    let wall = started.elapsed().as_secs_f64();
    println!(
        "arena: {name_a} vs {name_b} — {} games, {budget_desc}",
        games.len()
    );
    println!("  {name_a} {a_wins} wins, {name_b} {b_wins} wins, {draws} draws");
    println!("  score({name_a}) = {score:.3} ± {ci:.3} (95% CI)");
    println!("  elo({name_a} - {name_b}) = {elo:+.0} (95% CI {elo_lo:+.0} .. {elo_hi:+.0})");
    println!(
        "  avg turns {avg_turns:.0}, wall {:.1}m ({:.1} games/min)",
        wall / 60.0,
        games.len() as f64 / (wall / 60.0).max(1e-9)
    );

    if args.out.is_some() || args.record.is_some() {
        let doc = json!({
            "config": {
                "a": { "model": args.model_a, "name": name_a, "cores": cores_a },
                "b": { "model": args.model_b, "name": name_b, "cores": cores_b },
                "games": args.games,
                "snakes": args.snakes,
                "budget": budget_desc,
                "board": args.board,
                "seed": args.seed,
                "max_turns": args.max_turns,
                "parallel": parallel,
                "c_puct": args.c_puct,
                "draw_value": args.draw_value,
                "leaves_per_sim": args.leaves_per_sim,
                "virtual_loss": args.virtual_loss,
            },
            "summary": {
                "a_wins": a_wins,
                "b_wins": b_wins,
                "draws": draws,
                "score_a": score,
                "score_ci95": ci,
                "elo_a_minus_b": elo,
                "elo_ci95": [elo_lo, elo_hi],
                "avg_turns": avg_turns,
                "wall_seconds": wall,
            },
            "games": games.iter().map(|g| json!({
                "index": g.game_index,
                "seed": g.seed,
                "a_first": g.a_first,
                "winner": g.winner.map(Side::label),
                "winner_snake": g.winner_snake,
                "turns": g.turns,
                "reason": g.reason,
                "wall_ms": g.wall_ms,
            })).collect::<Vec<_>>(),
        });
        if let Some(path) = &args.out {
            match serde_json::to_string_pretty(&doc)
                .map_err(anyhow::Error::from)
                .and_then(|s| std::fs::write(path, s).map_err(anyhow::Error::from))
            {
                Ok(()) => eprintln!("arena: results written to {path}"),
                Err(e) => eprintln!("arena: failed to write {path}: {e}"),
            }
        }
        if let Some(path) = &args.record {
            // Same file shape as the trainer's games/gen_NNNN.json, with the
            // match summary in the `config` slot so the viewer can surface it.
            let file = GameFileJson {
                gen: args.record_gen,
                config: doc.clone(),
                games: games
                    .iter_mut()
                    .map(|g| GameJson {
                        frames: std::mem::take(&mut g.frames),
                        winner: g.winner_snake.map(|i| i as i32),
                        num_turns: g.turns,
                    })
                    .collect(),
            };
            let write = || -> anyhow::Result<()> {
                if let Some(parent) = Path::new(path).parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let tmp = format!("{path}.tmp");
                std::fs::write(&tmp, serde_json::to_vec(&file)?)?;
                std::fs::rename(&tmp, path)?;
                Ok(())
            };
            match write() {
                Ok(()) => eprintln!("arena: games recorded to {path}"),
                Err(e) => eprintln!("arena: failed to record {path}: {e}"),
            }
        }
    }
}
