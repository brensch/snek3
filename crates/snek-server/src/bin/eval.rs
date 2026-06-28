//! Offline Albatross-faithful evaluator.
//!
//! Plays the *deployed* agent — proxy ONNX + [`snek_server::serve_move`] (the
//! exact serving path) — as snake 0 against the fixed pool (flood-fill baseline,
//! CPU UCT) as the other snakes, then writes win-rates and full replay frames so
//! the dashboard can both chart strength and show the real games.
//!
//! It is meant to run *out of band*: the trainer exports the just-finished gen's
//! proxy to ONNX and spawns this binary, which uses otherwise-idle CPU (one game
//! per worker thread, each with its own `Net`) while the GPU trains the next gen.
//!
//! Usage:  snek-eval <model.onnx> <out.json>
//!
//! Config (env vars):
//!   SNEK_EVAL_GEN      generation label written into the artifact (default -1)
//!   SNEK_EVAL_GAMES    games per opponent (default 16)
//!   SNEK_EVAL_SEED     base RNG seed (default 0)
//!   SNEK_EVAL_WORKERS  parallel games (default = half the cores, rounded up)
//!   SNEK_ORT_THREADS   intra-op threads per CPU ONNX session (default 1)
//!   SNEK_EVAL_MAXTURNS per-game turn cap, 0 = uncapped w/ safety (default 0)
//!   SNEK_BOARD         board size (default 11)
//!   SNEK_SNAKES        snakes per game (default 2)
//!   SNEK_UCT_ITERS     UCT simulations for the UCT opponent (default 256)
//!   SNEK_DEPTH / SNEK_ITERS / SNEK_RESPONSE_TAU / SNEK_DRAW_VALUE / SNEK_EVAL_CHUNK
//!                      search params — same names/defaults as the server, so the
//!                      evaluator's search matches the deployed one exactly.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use snek_core::baseline::baseline_action;
use snek_core::{standard_start, Move};
use snek_infer::Net;
use snek_search::uct_actions;
use snek_server::{board_snapshot_value, env_or, serve_move, tau_grid, Config, GameState};

/// Hang-guard for uncapped games, matching SAFETY_TURNS in train_albatross.py.
const SAFETY_TURNS: u32 = 5000;
const UCT_C: f32 = 1.4;

#[derive(Clone, Copy)]
enum Opponent {
    Baseline,
    Uct,
}

impl Opponent {
    fn label(self) -> &'static str {
        match self {
            Opponent::Baseline => "baseline",
            Opponent::Uct => "uct",
        }
    }
}

struct Spec {
    opp: Opponent,
    seed: u64,
}

/// Static per-run game settings shared by every played game.
struct EvalCfg {
    board_size: i8,
    num_snakes: usize,
    uct_iters: usize,
    cap: u32,
}

struct GameResult {
    opp: Opponent,
    /// 0 = our agent won, 1.. = an opponent won, -1 = draw / cap reached.
    winner: i8,
    terminal: bool,
    frames: Vec<serde_json::Value>,
}

fn play_game(
    net: &mut Net,
    grid: &[f32; snek_server::TAU_GRID_LEN],
    cfg: &Config,
    spec: &Spec,
    ev: &EvalCfg,
) -> GameResult {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(spec.seed);
    let mut board = standard_start(ev.board_size, ev.board_size, ev.num_snakes, &mut rng);
    let ids: Vec<String> = (0..ev.num_snakes).map(|i| format!("s{i}")).collect();
    let mut gs = GameState::default();
    let mut frames = Vec::new();
    let mut turns = 0u32;

    loop {
        frames.push(board_snapshot_value(&board));
        if board.is_terminal() || turns >= ev.cap {
            break;
        }
        // Our deployed agent (snake 0): the faithful serving move.
        let m0 = serve_move(net, grid, cfg, &board, &ids, 0, &mut gs);
        let mut moves = vec![Move::Up; ev.num_snakes];
        moves[0] = Move::from_index(m0);
        // Opponents (snakes 1..n) from the fixed pool.
        match spec.opp {
            Opponent::Baseline => {
                for (i, mv) in moves.iter_mut().enumerate().skip(1) {
                    *mv = baseline_action(&board, i);
                }
            }
            Opponent::Uct => {
                let acts = uct_actions(std::slice::from_ref(&board), ev.uct_iters, UCT_C, spec.seed);
                for (i, mv) in moves.iter_mut().enumerate().skip(1) {
                    *mv = acts[0][i];
                }
            }
        }
        board.step_and_spawn(&moves, &mut rng);
        turns += 1;
    }

    let terminal = board.is_terminal();
    let winner = if terminal {
        board.winner().map(|w| w as i8).unwrap_or(-1)
    } else {
        -1
    };
    GameResult {
        opp: spec.opp,
        winner,
        terminal,
        frames,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| std::env::var("SNEK_MODEL").unwrap_or_else(|_| "model.onnx".into()));
    let out = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| std::env::var("SNEK_EVAL_OUT").unwrap_or_else(|_| "eval.json".into()));

    let gen: i64 = env_or("SNEK_EVAL_GEN", -1i64);
    let games: usize = env_or("SNEK_EVAL_GAMES", 16usize).max(1);
    let base_seed: u64 = env_or("SNEK_EVAL_SEED", 0u64);
    // Default to half the cores (rounded up) so the evaluator leaves headroom for
    // the rest of the machine instead of pinning every core at 100%. Each worker
    // runs a single-intra-op-thread CPU session (see Net::load), so worker count
    // is the dominant CPU knob. Override with SNEK_EVAL_WORKERS.
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let workers: usize = env_or("SNEK_EVAL_WORKERS", cores.div_ceil(2)).max(1);
    let max_turns: u32 = env_or("SNEK_EVAL_MAXTURNS", 0u32);
    let cap = if max_turns > 0 { max_turns } else { SAFETY_TURNS };
    let board_size: i8 = env_or("SNEK_BOARD", 11i8);
    let num_snakes: usize = env_or("SNEK_SNAKES", 2usize).max(2);
    let uct_iters: usize = env_or("SNEK_UCT_ITERS", 256usize);

    let cfg = Config {
        depth: env_or("SNEK_DEPTH", 2u32),
        iters: env_or("SNEK_ITERS", 120usize),
        response_tau: env_or("SNEK_RESPONSE_TAU", 12.0f32),
        draw_value: env_or("SNEK_DRAW_VALUE", -0.9f32),
        eval_chunk: env_or("SNEK_EVAL_CHUNK", 4096usize),
    };
    let grid = tau_grid();
    let ev = EvalCfg {
        board_size,
        num_snakes,
        uct_iters,
        cap,
    };

    // One spec per game; distinct seeds so games don't collide.
    let mut specs: Vec<Spec> = Vec::with_capacity(2 * games);
    for opp in [Opponent::Baseline, Opponent::Uct] {
        for g in 0..games {
            specs.push(Spec {
                opp,
                seed: base_seed.wrapping_add((opp.label().len() as u64) << 32).wrapping_add(g as u64),
            });
        }
    }

    eprintln!(
        "snek-eval: model={model} gen={gen} games/opp={games} workers={workers} \
         board={board_size} snakes={num_snakes} uct_iters={uct_iters} depth={} iters={}",
        cfg.depth, cfg.iters
    );
    let t0 = std::time::Instant::now();

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<GameResult>> = Mutex::new(Vec::with_capacity(specs.len()));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut net = match Net::load(&model) {
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("snek-eval: worker failed to load model: {e}");
                        return;
                    }
                };
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= specs.len() {
                        break;
                    }
                    let r = play_game(&mut net, &grid, &cfg, &specs[i], &ev);
                    results.lock().unwrap().push(r);
                }
            });
        }
    });

    let results = results.into_inner().unwrap();

    // Per-opponent win-rate: (wins + 0.5*draws) / decided, matching `wr` in
    // train_albatross.py. Cap-reached (non-terminal) games are excluded.
    let summarize = |opp_label: &str| {
        let g: Vec<&GameResult> = results.iter().filter(|r| r.opp.label() == opp_label).collect();
        let decided = g.iter().filter(|r| r.terminal).count();
        let wins = g.iter().filter(|r| r.terminal && r.winner == 0).count();
        let draws = g.iter().filter(|r| r.terminal && r.winner == -1).count();
        let losses = decided - wins - draws;
        let wr = if decided > 0 {
            (wins as f64 + 0.5 * draws as f64) / decided as f64
        } else {
            0.0
        };
        serde_json::json!({
            "win_rate": wr, "wins": wins, "losses": losses, "draws": draws,
            "decided": decided, "games": g.len(),
        })
    };
    let base = summarize("baseline");
    let uct = summarize("uct");

    let games_json: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "opponent": format!("vs-{}", r.opp.label()),
                "winner": r.winner,
                "num_turns": r.frames.len(),
                "frames": r.frames,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "gen": gen,
        "vs_base": base["win_rate"],
        "vs_uct": uct["win_rate"],
        "summary": { "baseline": base, "uct": uct },
        "games": games_json,
    });

    // Atomic write: render to a sibling tmp then rename, so a dashboard reading
    // the artifact never sees a half-written file.
    if let Some(parent) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = format!("{out}.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&payload).expect("serialize"))
        .unwrap_or_else(|e| panic!("write {tmp}: {e}"));
    std::fs::rename(&tmp, &out).unwrap_or_else(|e| panic!("rename {tmp} -> {out}: {e}"));

    eprintln!(
        "snek-eval: done in {:.1}s  vs_base={:.3} vs_uct={:.3}  -> {out}",
        t0.elapsed().as_secs_f64(),
        base["win_rate"].as_f64().unwrap_or(0.0),
        uct["win_rate"].as_f64().unwrap_or(0.0),
    );
}
