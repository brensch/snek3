//! Held-out strength eval for the Logit-Equilibrium net.
//!
//! The AZ checkpoint league is disabled in LE mode (it serves 14-ch nets through
//! the MCTS server), so this provides the real "is it getting stronger" signal:
//! the LE net plays — via its own equilibrium search at a fixed serve-τ — against
//! built-in heuristic baselines it never trains against. It's a *held-out,
//! diverse-opponent* measure by construction, exactly the metric the diagnosis
//! said we must trust (self-play score and low-sim league both lied).
//!
//! One net seat (rotated across games) plays the LE search's argmax move; the
//! other seats play the heuristic. Returns the net's win rate.

use crate::config::RunConfig;
use crate::eval::{is_external, player_name, MatchRecord, Placement};
use crate::sample::{frame_from_board, FrameJson};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use snek_core::{
    encode_into_temp, obs_side, standard_start, Board, Move, MAX_SNAKES, NUM_CHANNELS_TEMP,
};
use snek_heuristic::Baseline;
use snek_search::EqForest;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use tch::{no_grad, Device, Tensor};

/// One recorded held-out game, ready to be written to the eval viewer's files
/// (same `game_SSSSSS.json` + `summary.jsonl` format the league uses, so the
/// existing frontend recorded-games panel replays them with no changes).
pub struct RecordedEvalGame {
    pub frames: Vec<FrameJson>,
    pub placements: Vec<Placement>,
    pub turns: u32,
    pub winner: Option<i32>,
}

fn argmax4(p: &[f32; 4]) -> Move {
    let mut bi = 0usize;
    let mut bv = p[0];
    for (i, &v) in p.iter().enumerate().skip(1) {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    Move::from_index(bi)
}

fn forward_values(net: &snek_tch::AZNet, device: Device, obs: &[f32], rows: usize, side: i64) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    no_grad(|| {
        let x = Tensor::from_slice(obs)
            .reshape([rows as i64, NUM_CHANNELS_TEMP as i64, side, side])
            .to_device(device);
        let (_logits, value) = net.forward(&x);
        value.to_device(Device::Cpu).copy_data(&mut out, rows);
    });
    out
}

/// Play `games` LE-net-vs-`baseline` games (net at `serve_tau`, others at the
/// heuristic) and return the net's win rate (fraction of games the net's rotated
/// seat is the sole survivor). `baseline_sims` = 0 uses the 1-ply greedy.
pub fn eval_le_vs_baseline(
    net: &snek_tch::AZNet,
    device: Device,
    cfg: &RunConfig,
    baseline: Baseline,
    baseline_sims: usize,
    games: usize,
    serve_tau: f32,
    seed: u64,
    // Bumped once per finished game so the dashboard's ARENA meter advances
    // smoothly; the caller sets arena_target and resets it around the eval.
    progress: Option<&AtomicU32>,
    // Player ids for the recorded games' placements: `net_gen` is the net's
    // generation, `opponent_gen` a reserved id (HEURISTIC_GEN / VORONOI_GEN)
    // that renders as "floodfill"/"voronoi". The first `record_games` games
    // capture per-turn frames (with the net's LE policy) for the viewer.
    net_gen: u32,
    opponent_gen: u32,
    record_games: usize,
) -> (f64, Vec<RecordedEvalGame>) {
    let n = cfg.num_snakes;
    let board = cfg.board;
    let side = obs_side(board as usize);
    let l15 = NUM_CHANNELS_TEMP * side * side;
    let depth = cfg.le_depth.max(1);
    let iters = cfg.le_iters.max(1);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut boards: Vec<Board> = (0..games)
        .map(|_| standard_start(board, board, n, &mut rng))
        .collect();
    let net_seat: Vec<usize> = (0..games).map(|g| g % n).collect();
    let mut done = vec![false; games];
    let mut net_won = vec![false; games];

    // Per-turn frames for the audited games, and per-seat death turns (to rank
    // placements in the recorded game's summary).
    let record_games = record_games.min(games);
    let mut rec_frames: Vec<Vec<FrameJson>> = vec![Vec::new(); record_games];
    let mut death_turn: Vec<[Option<u32>; MAX_SNAKES]> = vec![[None; MAX_SNAKES]; games];

    // No turn cap — every game runs until the net dies (early-stop below) or a
    // natural terminal. Health/growth/finite-board guarantee termination.
    while done.iter().any(|&d| !d) {
        let live: Vec<usize> = (0..games).filter(|&g| !done[g]).collect();
        let live_boards: Vec<Board> = live.iter().map(|&g| boards[g].clone()).collect();
        let forest = EqForest::build(&live_boards, depth, cfg.draw_value);

        let num_eval = forest.eval_boards().len();
        let rows = num_eval * n;
        let vals = if rows > 0 {
            // cuDNN benchmark is off in LE mode (variable shapes), so no padding.
            let mut obs = vec![0.0f32; rows * l15];
            let eb = forest.eval_boards();
            obs.par_chunks_mut(n * l15).enumerate().for_each(|(e, chunk)| {
                for s in 0..n {
                    encode_into_temp(&eb[e], s, serve_tau, &mut chunk[s * l15..(s + 1) * l15]);
                }
            });
            forward_values(net, device, &obs, rows, side as i64)
        } else {
            Vec::new()
        };
        let tau_pg = vec![[serve_tau; MAX_SNAKES]; live_boards.len()];
        let roots = forest.backup(&vals, &tau_pg, iters);

        // Each game's move vector is independent, and the baseline's per-seat
        // search (voronoi at `baseline_sims`) dominates the turn's cost — so
        // compute every live game's joint action in parallel over an immutable
        // borrow of `boards`, then apply the steps sequentially. Without this the
        // held-out eval ran ~3×seats×games voronoi searches single-file per turn
        // and took minutes; the games themselves are the parallelism.
        let boards_ref = &boards;
        let moves: Vec<[Move; MAX_SNAKES]> = live
            .par_iter()
            .enumerate()
            .map(|(li, &g)| {
                let eq = &roots[li];
                let mut actions = [Move::Up; MAX_SNAKES];
                for s in 0..n {
                    if !boards_ref[g].snakes[s].alive() {
                        continue;
                    }
                    actions[s] = if s == net_seat[g] {
                        argmax4(&eq.policy[s])
                    } else if baseline_sims > 0 {
                        snek_heuristic::strong_move(
                            baseline,
                            &boards_ref[g],
                            s,
                            baseline_sims,
                            cfg.draw_value,
                        )
                    } else {
                        snek_heuristic::greedy_move(baseline, &boards_ref[g], s, cfg.draw_value)
                    };
                }
                actions
            })
            .collect();
        for (li, &g) in live.iter().enumerate() {
            // Capture the pre-step frame for audited games. Every seat gets its
            // LE equilibrium policy/value (what the search believed); the played
            // distribution is the mix for the net seat, one-hot for heuristics.
            if g < record_games {
                let eq = &roots[li];
                let mut policy = vec![0.0f32; n * 4];
                let mut values = vec![0.0f32; n];
                let mut play_pols = vec![[0.0f32; 4]; n];
                for s in 0..n {
                    policy[s * 4..s * 4 + 4].copy_from_slice(&eq.policy[s]);
                    values[s] = eq.value[s];
                    if s == net_seat[g] {
                        play_pols[s] = eq.policy[s];
                    } else {
                        play_pols[s][moves[li][s].index()] = 1.0;
                    }
                }
                rec_frames[g].push(frame_from_board(
                    &boards[g],
                    n,
                    &policy,
                    &values,
                    &play_pols,
                    &moves[li][..n],
                ));
            }

            let alive_before: [bool; MAX_SNAKES] =
                std::array::from_fn(|s| s < n && boards[g].snakes[s].alive());
            // Real play replenishes food; plain `step` (foodless) is search-only.
            boards[g].step_and_spawn(&moves[li][..n], &mut rng);
            for s in 0..n {
                if alive_before[s] && !boards[g].snakes[s].alive() && death_turn[g][s].is_none() {
                    death_turn[g][s] = Some(boards[g].turn);
                }
            }

            if boards[g].is_terminal() {
                done[g] = true;
                net_won[g] = boards[g].winner() == Some(net_seat[g]);
            } else if !boards[g].snakes[net_seat[g]].alive() {
                // Net seat is out: its verdict is decided (a "net win" needs the
                // net as sole survivor), so stop rather than simulate the
                // opponents' expensive voronoi endgame to a winner we never
                // score. This holds for recorded games too — once the net is
                // dead the rest of the replay says nothing about the net's play,
                // so the recording ends at (and includes) the net's death.
                done[g] = true;
                net_won[g] = false;
            }
            if done[g] {
                if let Some(p) = progress {
                    p.fetch_add(1, Ordering::Relaxed);
                }
                if g < record_games {
                    // Final terminal frame (dead snakes visible), no search readout.
                    let zero_pol = vec![0.0f32; n * 4];
                    let zero_val = vec![0.0f32; n];
                    let zero_play = vec![[0.0f32; 4]; n];
                    let noop = [Move::Up; MAX_SNAKES];
                    rec_frames[g].push(frame_from_board(
                        &boards[g],
                        n,
                        &zero_pol,
                        &zero_val,
                        &zero_play,
                        &noop[..n],
                    ));
                }
            }
        }
    }

    let win_rate = net_won.iter().filter(|&&w| w).count() as f64 / games.max(1) as f64;

    // Assemble the recorded games: rank seats by survival (alive-at-end best,
    // then later death first), naming the net seat by `net_gen` and every other
    // seat by `opponent_gen`.
    let recorded: Vec<RecordedEvalGame> = (0..record_games)
        .map(|g| {
            let survived_key = |s: usize| -> u32 {
                if boards[g].snakes[s].alive() {
                    u32::MAX
                } else {
                    death_turn[g][s].unwrap_or(0)
                }
            };
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&s| std::cmp::Reverse(survived_key(s)));
            let mut placements = vec![
                Placement {
                    gen: 0,
                    seat: 0,
                    rank: 0,
                    death_turn: None,
                };
                n
            ];
            let mut rank = 0u32;
            let mut prev_key: Option<u32> = None;
            for (i, &s) in order.iter().enumerate() {
                let key = survived_key(s);
                // Standard competition ranking: equal survival shares a rank.
                if prev_key != Some(key) {
                    rank = i as u32 + 1;
                    prev_key = Some(key);
                }
                placements[s] = Placement {
                    gen: if s == net_seat[g] { net_gen } else { opponent_gen },
                    seat: s as u32,
                    rank,
                    death_turn: death_turn[g][s],
                };
            }
            let winner = boards[g].winner().map(|s| s as i32);
            RecordedEvalGame {
                frames: std::mem::take(&mut rec_frames[g]),
                placements,
                turns: boards[g].turn,
                winner,
            }
        })
        .collect();

    (win_rate, recorded)
}

/// Write recorded held-out games into `<root>/eval/` in the same format the
/// league uses: one `game_SSSSSS.json` per game plus a `summary.jsonl` line, so
/// the frontend's recorded-games panel lists and replays them unchanged. Old
/// game files are pruned to a bounded window. `sims` is stored for display only.
pub fn write_eval_games(
    root: &Path,
    sims: u32,
    now_unix_ms: i64,
    games: &[RecordedEvalGame],
) -> anyhow::Result<()> {
    if games.is_empty() {
        return Ok(());
    }
    let eval_dir = root.join("eval");
    std::fs::create_dir_all(&eval_dir)?;
    let mut next_seq = crate::eval::read_summaries(root)
        .iter()
        .map(|m| m.seq + 1)
        .max()
        .unwrap_or(0);
    let mut summary = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(eval_dir.join("summary.jsonl"))?;
    for game in games {
        let seq = next_seq;
        next_seq += 1;
        let record = MatchRecord {
            seq,
            placements: game.placements.clone(),
            turns: game.turns,
            sims,
            wall_seconds: 0.0,
            finished_unix_ms: now_unix_ms,
        };
        writeln!(summary, "{}", serde_json::to_string(&record)?)?;
        write_eval_game_file(&eval_dir, &record, &game.frames, game.winner)?;
    }
    prune_eval_game_files(&eval_dir, 64);
    Ok(())
}

fn write_eval_game_file(
    eval_dir: &Path,
    record: &MatchRecord,
    frames: &[FrameJson],
    winner: Option<i32>,
) -> anyhow::Result<()> {
    let mut players = vec![0u32; record.placements.len()];
    for p in &record.placements {
        if (p.seat as usize) < players.len() {
            players[p.seat as usize] = p.gen;
        }
    }
    let file = serde_json::json!({
        "gen": players.iter().copied().filter(|&g| !is_external(g)).max().unwrap_or(0),
        "config": {
            "seq": record.seq,
            "players": players.iter().map(|&g| serde_json::json!({
                "gen": g,
                "name": player_name(g),
            })).collect::<Vec<_>>(),
            "placements": record.placements,
            "sims": record.sims,
        },
        "games": [{
            "frames": frames,
            "winner": winner,
            "num_turns": record.turns,
        }],
    });
    let path = eval_dir.join(format!("game_{:06}.json", record.seq));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&file)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Keep only the newest `keep` `game_*.json` files in the eval dir.
fn prune_eval_game_files(eval_dir: &Path, keep: usize) {
    let mut files: Vec<(u64, std::path::PathBuf)> = match std::fs::read_dir(eval_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter_map(|p| {
                let name = p.file_name()?.to_str()?.to_string();
                let seq: u64 = name.strip_prefix("game_")?.strip_suffix(".json")?.parse().ok()?;
                Some((seq, p))
            })
            .collect(),
        Err(_) => return,
    };
    if files.len() <= keep {
        return;
    }
    files.sort_by_key(|(seq, _)| *seq);
    let remove = files.len() - keep;
    for (_, path) in files.into_iter().take(remove) {
        let _ = std::fs::remove_file(path);
    }
}
