//! Multi-net evaluation arena: N nets (one snake seat each, seats rotating
//! between games), the exact `snek-core` rules engine, and the same
//! `serve_move_until` search the live server uses — all in-process, no HTTP.
//! Rules parity with the official Go engine is inherited from `snek-core` (the
//! same `Board::step_and_spawn` self-play and serving run on).
//!
//! Each net gets its own pool of worker threads pinned to its own CPU cores,
//! one worker (and one `Net`) per core, so players never steal cycles from
//! each other or from a concurrently running training job. Within a turn every
//! living snake's search runs concurrently on its net's pool.
//!
//! Games are scored by elimination order: rank 1 for the survivor, ties for
//! simultaneous deaths, and the game ends the moment at most one net has a
//! living snake (a Plackett–Luce rating fit consumes the full ranking).
//!
//! Fairness/determinism:
//! - The default budget is a fixed simulation count per move (`--sims`): the
//!   serving search is strict-argmax DUCT with no noise, so every game is
//!   deterministic given its seed and hardware-independent. `--time-ms`
//!   switches to wall-clock budgets like live play.
//! - Seat assignment rotates by game index (`seat s` plays net `(s+g) % N`),
//!   and start spawns are seed-shuffled, so no net accrues a seat bias.
//! - CPU-only by default; `--gpu` opts in (don't use while training runs).
//!
//! Example (4 checkpoints, one core each):
//!   arena --nets g40.st,g35.st,g25.st,g10.st --games 20 --sims 64 --cores 12-15

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
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

struct Args {
    nets: Vec<String>,
    names: Vec<String>,
    games: usize,
    seats: usize,
    budget: Budget,
    cores: Option<Vec<usize>>,
    cores_per_net: usize,
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
    events: bool,
}

/// One machine-readable JSON line on stdout, for a parent process driving us
/// (per-game results always; per-turn progress with --events). stdout is
/// block-buffered when piped, so flush each line.
fn emit_event(value: serde_json::Value) {
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout, "{value}");
    let _ = stdout.flush();
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
        "arena: play N nets against each other with the in-process rules engine.

usage: arena --nets <m1,m2[,m3,m4…]> [options]
       arena --a <model> --b <model> [options]   (two-net shorthand)

nets & seats:
  --nets LIST         comma-separated model weights, one player per entry;
                      the literal entry 'heuristic' (or 'floodfill') seats the
                      built-in flood-fill MCTS baseline instead of a net
  --a / --b PATH      shorthand for --nets a,b
  --names LIST        display names (default: model file stem / parent dir)
  --seats N           snakes per game; seat s plays net (s+game)%N, so two
                      nets with --seats 4 alternate A,B,A,B (default: net count)

match:
  --games N           total games; seats rotate every game (100)
  --sims N            fixed MCTS sims per move; deterministic (1000)
  --time-ms MS        wall-clock budget per move instead of --sims
  --board N           board side length (11)
  --seed N            base seed; game g uses seed+g (1)
  --max-turns N       turn cutoff; survivors tie for rank 1 (500)

cpu / pinning:
  --cores SPEC        flat core list split between nets in order, e.g. 12-15
                      gives net 1 cores 12,13 and net 2 cores 14,15 at
                      --cores-per-net 2 (auto)
  --cores-per-net N   cores (worker threads) per net (1)
  --parallel N        concurrent games (default: workers / seats, min 1)
  --gpu               allow CUDA (default forces CPU so training is untouched)

search (defaults match live serving):
  --c-puct F          PUCT exploration constant (1.5)
  --draw-value F      terminal draw value (-0.25)
  --leaves-per-sim N  virtual-loss batch width (8)
  --virtual-loss F    virtual-loss magnitude (1.0)
  --eval-chunk N      max rows per net forward (4096)

output:
  --out PATH          write full results (placements per game) as JSON
  --record PATH       record every game (frames + search readout) as a
                      viewer-compatible games file (same schema as the
                      trainer's games/gen_NNNN.json)
  --record-gen N      the generation label stamped into --record (0)
  --events            stream per-turn progress as JSON lines on stdout (for
                      a parent process; per-game results always stream)"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut nets: Vec<String> = Vec::new();
    let mut model_a = None;
    let mut model_b = None;
    let mut names: Vec<String> = Vec::new();
    let mut games = 100usize;
    let mut seats: Option<usize> = None;
    let mut sims = 1000usize;
    let mut time_ms: Option<u64> = None;
    let mut cores = None;
    let mut cores_per_net = 1usize;
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
    let mut events = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| -> String {
            it.next().unwrap_or_else(|| {
                eprintln!("arena: {name} requires a value");
                std::process::exit(2);
            })
        };
        let list = |v: String| -> Vec<String> {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        };
        match arg.as_str() {
            "--nets" => nets = list(val("--nets")),
            "--a" => model_a = Some(val("--a")),
            "--b" => model_b = Some(val("--b")),
            "--names" => names = list(val("--names")),
            "--name-a" => names = vec![val("--name-a")],
            "--name-b" => {
                let name = val("--name-b");
                if names.len() < 2 {
                    names.resize(1, String::new());
                }
                names.push(name);
            }
            "--games" => games = parse_num(&val("--games"), "--games"),
            "--seats" | "--snakes" => seats = Some(parse_num(&val("--seats"), "--seats")),
            "--sims" => sims = parse_num(&val("--sims"), "--sims"),
            "--time-ms" => time_ms = Some(parse_num(&val("--time-ms"), "--time-ms")),
            "--cores" => cores = Some(parse_core_spec(&val("--cores"))),
            "--cores-per-net" => {
                cores_per_net =
                    parse_num::<usize>(&val("--cores-per-net"), "--cores-per-net").max(1)
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
            "--events" => events = true,
            "--help" | "-h" => usage(),
            other => {
                eprintln!("arena: unknown argument {other}");
                usage();
            }
        }
    }
    if nets.is_empty() {
        if let (Some(a), Some(b)) = (model_a, model_b) {
            nets = vec![a, b];
        }
    }
    if nets.len() < 2 {
        eprintln!("arena: need at least two nets (--nets or --a/--b)");
        usage();
    }
    if games == 0 {
        eprintln!("arena: --games must be > 0");
        std::process::exit(2);
    }
    let seats = seats.unwrap_or(nets.len()).max(nets.len());
    if !(2..=snek_core::MAX_SNAKES).contains(&seats) {
        eprintln!("arena: seats must be 2..={}", snek_core::MAX_SNAKES);
        std::process::exit(2);
    }
    // Fill in default display names: file stem, or the parent directory when
    // stems collide (trainer checkpoints are all net_NNNN.safetensors).
    while names.len() < nets.len() {
        let path = &nets[names.len()];
        names.push(default_name(path, names.len()));
    }
    Args {
        nets,
        names,
        games,
        seats,
        budget: match time_ms {
            Some(ms) => Budget::TimeMs(ms.max(1)),
            None => Budget::Sims(sims.max(1)),
        },
        cores,
        cores_per_net,
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
        events,
    }
}

fn default_name(path: &str, index: usize) -> String {
    if snek_heuristic::is_heuristic_spec(path) {
        return snek_heuristic::DISPLAY_NAME.to_string();
    }
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("net-{index}"))
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

struct MoveJob {
    board: Board,
    me: usize,
    reply: mpsc::Sender<MoveInfo>,
}

/// A worker's answer for one position: the move plus the search readout for
/// snake `me` (visit-count policy and root value), used by game recording.
struct MoveInfo {
    move_index: usize,
    policy: [f32; 4],
    value: f32,
}

/// One net's worker pool: one pinned thread (own `Net`) per core. Jobs carry
/// their own reply channel, so any number of concurrent games share the pool.
struct NetPool {
    name: String,
    senders: Vec<mpsc::Sender<MoveJob>>,
    next: AtomicUsize,
}

impl NetPool {
    fn submit(&self, board: Board, me: usize) -> mpsc::Receiver<MoveInfo> {
        let (reply, rx) = mpsc::channel();
        let k = self.next.fetch_add(1, Ordering::Relaxed) % self.senders.len();
        self.senders[k]
            .send(MoveJob { board, me, reply })
            .expect("arena worker exited");
        rx
    }
}

fn spawn_worker(
    label: String,
    model: String,
    core: Option<usize>,
    cfg: Config,
    budget: Budget,
) -> mpsc::Sender<MoveJob> {
    let (job_tx, job_rx) = mpsc::channel::<MoveJob>();
    std::thread::Builder::new()
        .name(label.clone())
        .spawn(move || {
            if let Some(id) = core {
                if !core_affinity::set_for_current(core_affinity::CoreId { id }) {
                    eprintln!("arena: warning: failed to pin {label} to core {id}");
                }
            }
            // A "heuristic"/"floodfill" model spec plays the fixed flood-fill
            // MCTS baseline (snek-heuristic) — no weights, no libtorch. Its
            // sims are ~100× cheaper than a net forward, so in fixed-sims mode
            // it keeps its own static budget (~200ms/move, still
            // deterministic) instead of the nets' --sims; in time mode the
            // shared per-move deadline binds it like everyone else.
            if snek_heuristic::is_heuristic_spec(&model) {
                let hcfg = snek_heuristic::HeuristicConfig {
                    max_sims: match budget {
                        Budget::Sims(_) => {
                            snek_heuristic::HeuristicConfig::default().max_sims
                        }
                        Budget::TimeMs(_) => usize::MAX,
                    },
                    draw_value: cfg.draw_value,
                    ..Default::default()
                };
                while let Ok(job) = job_rx.recv() {
                    let deadline = match budget {
                        Budget::Sims(_) => Instant::now() + SIMS_DEADLINE,
                        Budget::TimeMs(ms) => Instant::now() + Duration::from_millis(ms),
                    };
                    let d = snek_heuristic::heuristic_move_until(
                        &hcfg,
                        &job.board,
                        job.me,
                        deadline,
                    );
                    let _ = job.reply.send(MoveInfo {
                        move_index: d.move_index,
                        policy: d.policy,
                        value: d.value,
                    });
                }
                return;
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
                let _ = job.reply.send(MoveInfo {
                    move_index: d.move_index,
                    policy,
                    value,
                });
            }
        })
        .expect("spawn arena worker");
    job_tx
}

/// One seat's result: which net played it and where it finished. Rank 1 is
/// best; ties share a rank (simultaneous deaths, or survivors at the cutoff).
#[derive(Serialize, Clone)]
struct SeatPlacement {
    seat: usize,
    net: usize,
    rank: u32,
    death_turn: Option<u32>,
}

struct GameOutcome {
    game_index: usize,
    seed: u64,
    turns: u32,
    wall_ms: u64,
    placements: Vec<SeatPlacement>,
    /// Recorded frames (pre-step, like self-play sample games); empty unless
    /// --record is set.
    frames: Vec<FrameJson>,
}

/// Which net plays seat `s` of game `g`: rotation by game index means every
/// net cycles through every seat (two nets at four seats give A,B,A,B then
/// B,A,B,A — the classic mirrored pair).
fn net_for_seat(seat: usize, game_index: usize, n_nets: usize) -> usize {
    (seat + game_index) % n_nets
}

#[allow(clippy::too_many_arguments)]
fn play_game(
    game_index: usize,
    seed: u64,
    seats: usize,
    board_size: i8,
    max_turns: u32,
    record: bool,
    events: bool,
    pools: &[NetPool],
) -> GameOutcome {
    let start = Instant::now();
    let n_nets = pools.len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut board = standard_start(board_size, board_size, seats, &mut rng);
    let mut frames = Vec::new();
    let mut death_turn: Vec<Option<u32>> = vec![None; seats];
    // The game ends for rating purposes as soon as at most one net has a
    // living snake — the elimination order is complete at that point.
    let living_nets = |board: &Board| {
        let mut nets: Vec<usize> = board
            .snakes
            .iter()
            .enumerate()
            .filter(|(_, s)| s.alive())
            .map(|(i, _)| net_for_seat(i, game_index, n_nets))
            .collect();
        nets.sort_unstable();
        nets.dedup();
        nets
    };
    while board.turn < max_turns && living_nets(&board).len() > 1 {
        let n = board.snakes.len();
        let mut moves = vec![Move::Up; n];
        let mut infos: Vec<Option<MoveInfo>> = (0..n).map(|_| None).collect();
        // Fan out every living snake's search before collecting any: each job
        // lands on its net's worker pool, so all nets search concurrently.
        let mut pending: Vec<(usize, mpsc::Receiver<MoveInfo>)> = Vec::new();
        for i in 0..n {
            if !board.snakes[i].alive() {
                continue;
            }
            let pool = &pools[net_for_seat(i, game_index, n_nets)];
            pending.push((i, pool.submit(board.clone(), i)));
        }
        for (i, rx) in pending {
            let info = rx.recv().expect("arena worker exited");
            moves[i] = Move::from_index(info.move_index);
            infos[i] = Some(info);
        }
        let frame = (record || events).then(|| frame_from_board(&board, &moves, &infos));
        if record {
            frames.push(frame.clone().expect("frame built when recording"));
        }
        board.step_and_spawn(&moves, &mut rng);
        for (i, snake) in board.snakes.iter().enumerate() {
            if !snake.alive() && death_turn[i].is_none() {
                death_turn[i] = Some(board.turn);
            }
        }
        if events {
            // The frame is the pre-step position with each snake's decision
            // (policy/value/move), so a live viewer renders exactly what the
            // recorded replay will show for this turn.
            emit_event(json!({
                "event": "turn",
                "index": game_index,
                "turn": board.turn,
                "alive": board.alive_count(),
                "frame": frame,
            }));
        }
    }
    // Competition ranking by elimination order: outlasting a snake beats it,
    // survivors (including everyone at a max-turns cutoff) tie at the top.
    let score = |i: usize| death_turn[i].unwrap_or(u32::MAX);
    let placements: Vec<SeatPlacement> = (0..seats)
        .map(|i| {
            let rank = 1 + (0..seats).filter(|&j| score(j) > score(i)).count() as u32;
            SeatPlacement {
                seat: i,
                net: net_for_seat(i, game_index, n_nets),
                rank,
                death_turn: death_turn[i],
            }
        })
        .collect();
    GameOutcome {
        game_index,
        seed,
        turns: board.turn,
        wall_ms: start.elapsed().as_millis() as u64,
        placements,
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

fn main() {
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);

    let args = parse_args();
    if !args.gpu && std::env::var("SNEK_CPU_ONLY").is_err() {
        std::env::set_var("SNEK_CPU_ONLY", "1");
    }
    let n_nets = args.nets.len();
    // Disambiguate colliding default names via the parent directory.
    let mut names = args.names.clone();
    for i in 0..names.len() {
        if names.iter().filter(|n| **n == names[i]).count() > 1 {
            if let Some(parent) = Path::new(&args.nets[i])
                .parent()
                .and_then(Path::file_name)
                .and_then(|s| s.to_str())
            {
                names[i] = format!("{parent}/{}", names[i]);
            }
        }
    }

    // Allocate cores: an explicit --cores list is split between nets in order;
    // otherwise take nets × cores_per_net from the machine's available cores.
    let available: Vec<usize> = core_affinity::get_core_ids()
        .unwrap_or_default()
        .into_iter()
        .map(|c| c.id)
        .collect();
    let flat: Option<Vec<usize>> = match &args.cores {
        Some(list) => Some(list.clone()),
        None if available.is_empty() => None,
        None => Some(
            available
                .iter()
                .copied()
                .take(n_nets * args.cores_per_net)
                .collect(),
        ),
    };
    let per_net = flat
        .as_ref()
        .map(|f| (f.len() / n_nets).max(1))
        .unwrap_or(args.cores_per_net);
    let net_cores = |net: usize, k: usize| -> Option<usize> {
        flat.as_ref()
            .and_then(|f| f.get(net * per_net + k).copied())
    };
    if flat.is_none() {
        eprintln!("arena: warning: cannot enumerate CPU cores; workers will be unpinned");
    }

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

    let pools: Arc<Vec<NetPool>> = Arc::new(
        (0..n_nets)
            .map(|net| {
                let senders = (0..per_net)
                    .map(|k| {
                        spawn_worker(
                            format!("arena-n{net}w{k}"),
                            args.nets[net].clone(),
                            net_cores(net, k),
                            cfg.clone(),
                            args.budget,
                        )
                    })
                    .collect();
                NetPool {
                    name: names[net].clone(),
                    senders,
                    next: AtomicUsize::new(0),
                }
            })
            .collect(),
    );
    for (net, pool) in pools.iter().enumerate() {
        let cores: Vec<_> = (0..per_net).filter_map(|k| net_cores(net, k)).collect();
        eprintln!(
            "arena: net {net} {name} ({path}) cores={cores:?}",
            name = pool.name,
            path = args.nets[net],
        );
    }
    let parallel = args
        .parallel
        .unwrap_or_else(|| ((n_nets * per_net) / args.seats).max(1))
        .max(1)
        .min(args.games);
    eprintln!(
        "arena: {games} games, {seats} seats over {n_nets} nets (rotating), {budget_desc}, board {board}x{board}, parallel {parallel}, {mode}",
        games = args.games,
        seats = args.seats,
        board = args.board,
        mode = if args.gpu { "gpu allowed" } else { "cpu only" }
    );

    let mut runners = Vec::new();
    let (result_tx, result_rx) = mpsc::channel::<GameOutcome>();
    for slot in 0..parallel {
        let pools = Arc::clone(&pools);
        let tx = result_tx.clone();
        let record = args.record.is_some();
        let events = args.events;
        let (games, base_seed, seats, board, max_turns) = (
            args.games,
            args.seed,
            args.seats,
            args.board,
            args.max_turns,
        );
        runners.push(std::thread::spawn(move || {
            let mut g = slot;
            while g < games {
                let seed = base_seed.wrapping_add(g as u64);
                let out = play_game(g, seed, seats, board, max_turns, record, events, &pools);
                if tx.send(out).is_err() {
                    return;
                }
                g += parallel;
            }
        }));
    }
    drop(result_tx);

    let started = Instant::now();
    let mut games: Vec<GameOutcome> = Vec::with_capacity(args.games);
    // Per-net running tallies: games, rank-1 finishes, summed rank.
    let mut tally = vec![(0u32, 0u32, 0u64); n_nets];
    while let Ok(out) = result_rx.recv() {
        for p in &out.placements {
            let t = &mut tally[p.net];
            t.0 += 1;
            t.1 += (p.rank == 1) as u32;
            t.2 += p.rank as u64;
        }
        let mut order = out.placements.clone();
        order.sort_by_key(|p| p.rank);
        let ranking = order
            .iter()
            .map(|p| names[p.net].clone())
            .collect::<Vec<_>>()
            .join(" > ");
        eprintln!(
            "arena: [{done:>4}/{total}] game {idx:04} {ranking} · {turns} turns ({secs:.1}s)",
            done = games.len() + 1,
            total = args.games,
            idx = out.game_index,
            turns = out.turns,
            secs = out.wall_ms as f64 / 1000.0,
        );
        // Per-game event for whoever drives us as a child process (the
        // trainer's league updates ratings after every game).
        emit_event(json!({
            "event": "game",
            "index": out.game_index,
            "turns": out.turns,
            "placements": out.placements,
        }));
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
    let avg_turns = games.iter().map(|g| g.turns as f64).sum::<f64>() / games.len() as f64;
    let wall = started.elapsed().as_secs_f64();
    println!(
        "arena: {} games over {} nets, {budget_desc}, avg turns {avg_turns:.0}, wall {:.1}m",
        games.len(),
        n_nets,
        wall / 60.0
    );
    for (net, (played, firsts, rank_sum)) in tally.iter().enumerate() {
        println!(
            "  {name:<24} {played} games, {firsts} wins, avg rank {avg:.2}",
            name = names[net],
            avg = if *played > 0 {
                *rank_sum as f64 / *played as f64
            } else {
                0.0
            },
        );
    }

    if args.out.is_some() || args.record.is_some() {
        let doc = json!({
            "config": {
                "nets": (0..n_nets).map(|i| json!({
                    "model": args.nets[i],
                    "name": names[i],
                    "cores": (0..per_net).filter_map(|k| net_cores(i, k)).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "games": args.games,
                "seats": args.seats,
                "budget": budget_desc,
                "board": args.board,
                "seed": args.seed,
                "max_turns": args.max_turns,
                "parallel": parallel,
                "c_puct": args.c_puct,
                "draw_value": args.draw_value,
            },
            "summary": {
                "nets": (0..n_nets).map(|i| json!({
                    "name": names[i],
                    "games": tally[i].0,
                    "wins": tally[i].1,
                    "avg_rank": if tally[i].0 > 0 { tally[i].2 as f64 / tally[i].0 as f64 } else { 0.0 },
                })).collect::<Vec<_>>(),
                "avg_turns": avg_turns,
                "wall_seconds": wall,
            },
            "games": games.iter().map(|g| json!({
                "index": g.game_index,
                "seed": g.seed,
                "turns": g.turns,
                "wall_ms": g.wall_ms,
                "placements": g.placements,
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
                    .map(|g| {
                        let winners: Vec<_> = g.placements.iter().filter(|p| p.rank == 1).collect();
                        GameJson {
                            frames: std::mem::take(&mut g.frames),
                            winner: match winners.as_slice() {
                                [only] => Some(only.seat as i32),
                                _ => None,
                            },
                            num_turns: g.turns,
                        }
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
