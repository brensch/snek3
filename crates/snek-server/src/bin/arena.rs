//! Multi-player evaluation CLI over [`snek_server::arena::play_game`] — the
//! same in-process game runner the trainer's league calls directly. N player
//! slots run games concurrently off a shared counter (no core pinning; the OS
//! schedules), seats rotating by game index so no player accrues a seat bias.
//!
//! Rules parity with the official Go engine is inherited from `snek-core`
//! (the same `Board::step_and_spawn` self-play and serving run on). With the
//! default fixed-sims budget every game is deterministic given its seed.
//!
//! Example (4 checkpoints, 4 concurrent games):
//!   arena --nets g40.st,g35.st,g25.st,g10.st --games 20 --sims 64 --parallel 4

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use serde_json::json;
use snek_server::arena::{play_game, Budget, GameSettings, Outcome, Placement, PlayerSpec};
use snek_server::Config;

struct Args {
    nets: Vec<String>,
    names: Vec<String>,
    games: usize,
    seats: usize,
    budget: Budget,
    parallel: usize,
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

fn usage() -> ! {
    eprintln!(
        "arena: play N players against each other with the in-process rules engine.

usage: arena --nets <m1,m2[,m3,m4…]> [options]
       arena --a <model> --b <model> [options]   (two-player shorthand)

players & seats:
  --nets LIST         comma-separated players, one per entry. Each entry is a
                      model weights path, a built-in heuristic agent
                      (\"floodfill\", \"voronoi\") or an http(s):// Battlesnake
                      server url
  --names LIST        display names, same order as --nets (default: file stems)
  --seats N           snakes per game; seat s plays player (s+game)%N, so two
                      players with --seats 4 alternate A,B,A,B (default: player
                      count)

schedule:
  --games N           total games; seats rotate every game (100)
  --parallel N        concurrent games, each on its own thread with its own
                      copies of the nets (default: 4, capped at --games)
  --seed N            base seed; game g plays with seed+g (1)
  --board N           board width/height (11)
  --max-turns N       cutoff; survivors tie at rank 1 (500)

search:
  --sims N            fixed simulations per move — deterministic (1000)
  --time-ms N         wall-clock per move instead of fixed sims
  --c-puct X          PUCT exploration constant (1.5)
  --draw-value X      terminal draw value (-0.25)
  --leaves-per-sim N  leaves per batched selection round (8)
  --virtual-loss X    virtual loss for batched selection (1.0)
  --eval-chunk N      max rows per net forward (4096)
  --gpu               allow CUDA (don't use while training runs; default CPU)

output:
  --out PATH          write full results (placements per game) as JSON
  --record PATH       record every game (frames + search readout) as a
                      viewer-compatible games file (same schema as the
                      trainer's games/gen_NNNN.json)
  --record-gen N      the generation label stamped into --record (0)"
    );
    std::process::exit(2);
}

fn parse_num<T: std::str::FromStr>(s: &str, flag: &str) -> T {
    s.parse().unwrap_or_else(|_| {
        eprintln!("arena: invalid value {s:?} for {flag}");
        std::process::exit(2);
    })
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
    let mut parallel = 4usize;
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
            "--parallel" => parallel = parse_num(&val("--parallel"), "--parallel"),
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
    if nets.is_empty() {
        if let (Some(a), Some(b)) = (model_a, model_b) {
            nets = vec![a, b];
        }
    }
    if nets.len() < 2 {
        eprintln!("arena: need at least two players (--nets or --a/--b)");
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
        let spec = PlayerSpec::parse(&nets[names.len()]);
        names.push(spec.default_name(names.len()));
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
        parallel: parallel.max(1).min(games),
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

struct GameResult {
    index: usize,
    seed: u64,
    outcome: Outcome,
}

fn main() {
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);

    let args = parse_args();
    let device = if args.gpu && tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    let trunk_channels = snek_server::env_or("SNEK_TRUNK_CHANNELS", 96i64);
    let trunk_blocks = snek_server::env_or("SNEK_TRUNK_BLOCKS", 8i64);
    let n_players = args.nets.len();

    // Disambiguate colliding default names via the parent directory.
    let mut names = args.names.clone();
    for i in 0..names.len() {
        if names.iter().filter(|n| **n == names[i]).count() > 1 {
            if let Some(parent) = std::path::Path::new(&args.nets[i])
                .parent()
                .and_then(std::path::Path::file_name)
                .and_then(|s| s.to_str())
            {
                names[i] = format!("{parent}/{}", names[i]);
            }
        }
    }

    let settings = Arc::new(GameSettings {
        seats: args.seats,
        board_size: args.board,
        max_turns: args.max_turns,
        cfg: Config {
            max_sims: match args.budget {
                Budget::Sims(n) => n,
                Budget::TimeMs(_) => usize::MAX,
            },
            c_puct: args.c_puct,
            draw_value: args.draw_value,
            eval_chunk: args.eval_chunk,
            leaves_per_sim: args.leaves_per_sim,
            virtual_loss: args.virtual_loss,
        },
        budget: args.budget,
    });
    let budget_desc = match args.budget {
        Budget::Sims(n) => format!("{n} sims/move"),
        Budget::TimeMs(ms) => format!("{ms} ms/move"),
    };
    eprintln!(
        "arena: {games} games, {seats} seats over {n_players} players (rotating), {budget_desc}, board {board}x{board}, parallel {parallel}, {dev}",
        games = args.games,
        seats = args.seats,
        board = args.board,
        parallel = args.parallel,
        dev = if device == tch::Device::Cpu { "cpu" } else { "gpu" },
    );

    // One slot per thread, games claimed off a shared counter so a slot that
    // draws short games absorbs more of them (no straggler by construction).
    // Each slot loads its own copies of the nets: a search needs `&mut Net`.
    let next_game = Arc::new(AtomicUsize::new(0));
    let (result_tx, result_rx) = mpsc::channel::<GameResult>();
    let specs: Arc<Vec<String>> = Arc::new(args.nets.clone());
    let slot_names: Arc<Vec<String>> = Arc::new(names.clone());
    let record_frames = args.record.is_some();
    let mut slots = Vec::new();
    for slot in 0..args.parallel {
        let next_game = Arc::clone(&next_game);
        let tx = result_tx.clone();
        let specs = Arc::clone(&specs);
        let slot_names = Arc::clone(&slot_names);
        let settings = Arc::clone(&settings);
        let (games, base_seed) = (args.games, args.seed);
        slots.push(std::thread::spawn(move || {
            let mut players: Vec<_> = specs
                .iter()
                .enumerate()
                .map(|(i, spec)| {
                    snek_server::arena::Player::load(
                        &PlayerSpec::parse(spec),
                        device,
                        trunk_channels,
                        trunk_blocks,
                        settings.budget,
                        &slot_names[i],
                    )
                    .unwrap_or_else(|e| {
                        eprintln!("arena: slot {slot}: failed to load {spec}: {e}");
                        std::process::exit(1);
                    })
                })
                .collect();
            loop {
                let g = next_game.fetch_add(1, Ordering::Relaxed);
                if g >= games {
                    return;
                }
                let seed = base_seed.wrapping_add(g as u64);
                let outcome = play_game(&mut players, g, &settings, seed, &|| false, &mut |_| {})
                    .expect("no stop signal in CLI mode");
                let mut outcome = outcome;
                if !record_frames {
                    outcome.frames.clear();
                }
                if tx
                    .send(GameResult {
                        index: g,
                        seed,
                        outcome,
                    })
                    .is_err()
                {
                    return;
                }
            }
        }));
    }
    drop(result_tx);

    let started = Instant::now();
    let mut results: Vec<GameResult> = Vec::with_capacity(args.games);
    // Per-player running tallies: games, rank-1 finishes, summed rank.
    let mut tally = vec![(0u32, 0u32, 0u64); n_players];
    while let Ok(res) = result_rx.recv() {
        for p in &res.outcome.placements {
            let t = &mut tally[p.player as usize];
            t.0 += 1;
            t.1 += (p.rank == 1) as u32;
            t.2 += p.rank as u64;
        }
        let mut order: Vec<&Placement> = res.outcome.placements.iter().collect();
        order.sort_by_key(|p| p.rank);
        eprintln!(
            "arena: [{done:>4}/{total}] game {idx:04} {ranking} · {turns} turns ({secs:.1}s)",
            done = results.len() + 1,
            total = args.games,
            idx = res.index,
            ranking = order
                .iter()
                .map(|p| names[p.player as usize].clone())
                .collect::<Vec<_>>()
                .join(" > "),
            turns = res.outcome.turns,
            secs = res.outcome.wall_ms as f64 / 1000.0,
        );
        results.push(res);
    }
    for handle in slots {
        let _ = handle.join();
    }

    if results.is_empty() {
        eprintln!("arena: no games completed");
        std::process::exit(1);
    }
    results.sort_by_key(|r| r.index);
    let avg_turns =
        results.iter().map(|r| r.outcome.turns as f64).sum::<f64>() / results.len() as f64;
    let wall = started.elapsed().as_secs_f64();
    println!(
        "arena: {} games over {} players, {budget_desc}, avg turns {avg_turns:.0}, wall {:.1}m",
        results.len(),
        n_players,
        wall / 60.0
    );
    for (player, (played, firsts, rank_sum)) in tally.iter().enumerate() {
        println!(
            "  {name:<24} {played} games, {firsts} wins, avg rank {avg:.2}",
            name = names[player],
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
                "players": (0..n_players).map(|i| json!({
                    "model": args.nets[i],
                    "name": names[i],
                })).collect::<Vec<_>>(),
                "games": args.games,
                "seats": args.seats,
                "budget": budget_desc,
                "board": args.board,
                "seed": args.seed,
                "max_turns": args.max_turns,
                "parallel": args.parallel,
                "c_puct": args.c_puct,
                "draw_value": args.draw_value,
            },
            "summary": {
                "players": (0..n_players).map(|i| json!({
                    "name": names[i],
                    "games": tally[i].0,
                    "wins": tally[i].1,
                    "avg_rank": if tally[i].0 > 0 { tally[i].2 as f64 / tally[i].0 as f64 } else { 0.0 },
                })).collect::<Vec<_>>(),
                "avg_turns": avg_turns,
                "wall_seconds": wall,
            },
            "games": results.iter().map(|r| json!({
                "index": r.index,
                "seed": r.seed,
                "turns": r.outcome.turns,
                "wall_ms": r.outcome.wall_ms,
                "placements": r.outcome.placements,
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
            // run summary in the `config` slot so the viewer can surface it.
            let file = json!({
                "gen": args.record_gen,
                "config": doc,
                "games": results.iter().map(|r| {
                    let winners: Vec<_> = r.outcome.placements.iter().filter(|p| p.rank == 1).collect();
                    json!({
                        "frames": r.outcome.frames,
                        "winner": match winners.as_slice() {
                            [only] => Some(only.seat as i32),
                            _ => None,
                        },
                        "num_turns": r.outcome.turns,
                    })
                }).collect::<Vec<_>>(),
            });
            let write = || -> anyhow::Result<()> {
                if let Some(parent) = std::path::Path::new(path).parent() {
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
