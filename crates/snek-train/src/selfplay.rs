use crate::config::RunConfig;
use crate::metrics::Metrics;
use crate::replay::Samples;
use crate::sample::{frame_from_board, FrameJson, GameJson};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use snek_core::{encode_into, obs_side, standard_start, Board, Move, NUM_CHANNELS};
use snek_search::{mask_obvious_immediate_deaths, MctsForest};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tch::{no_grad, Device, Kind, Tensor};

type GpuReq = Option<GpuJob>;
type FinishReq = Option<FinishedGame>;
type SearchReq = Option<SearchTask>;

/// Safety cap on frames buffered per watched slot, to bound memory if a game
/// runs pathologically long (e.g. `max_turns == 0` with near-perfect play).
const MAX_REC_FRAMES: usize = 4096;

struct GpuJob {
    obs: Vec<f32>,
    rows: usize,
    reply: Sender<GpuEval>,
}

struct GpuEval {
    pol: Vec<f32>,
    val: Vec<f32>,
}

struct SearchTask {
    sid: usize,
    boards: Vec<Board>,
    c_puct: f32,
    draw_value: f32,
    sims: usize,
    reply: Sender<anyhow::Result<(usize, Vec<f32>, Vec<f32>)>>,
}

struct WorkerNet(*const snek_tch::AZNet);

// tch tensors are not Sync, but during a self-play generation the training
// thread never touches the net while the GPU worker is alive. The scoped worker
// joins before `generate` returns, so this borrow cannot outlive that invariant.
unsafe impl Send for WorkerNet {}

impl WorkerNet {
    fn infer(
        &self,
        device: Device,
        obs: &[f32],
        rows: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let net = unsafe { &*self.0 };
        infer(net, device, obs, rows, c, h, w)
    }
}

pub struct SelfPlayNet<'a> {
    pub net: &'a snek_tch::AZNet,
    pub device: Device,
}

#[derive(Default)]
struct Slot {
    obs: Vec<f32>,
    pol: Vec<f32>,
    value: Vec<f32>,
    alive: Vec<bool>,
    steps: usize,
}

struct FinishedGame {
    slot: Slot,
    winner: Option<usize>,
}

struct SampleChunk {
    obs: Vec<f32>,
    pol: Vec<f32>,
    z: Vec<f32>,
}

pub fn generate(
    net: &SelfPlayNet<'_>,
    cfg: &RunConfig,
    seed: u64,
    metrics: &Metrics,
    stop: &AtomicBool,
) -> anyhow::Result<(Samples, Vec<GameJson>)> {
    let counters = metrics.counters();
    counters
        .samples_target
        .store(cfg.samples_per_gen as u32, Ordering::Relaxed);
    counters.samples_collected.store(0, Ordering::Relaxed);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut boards: Vec<Board> = (0..cfg.count)
        .map(|_| standard_start(cfg.board, cfg.board, cfg.num_snakes, &mut rng))
        .collect();
    let mut turns = vec![0usize; cfg.count];
    let mut slots = (0..cfg.count).map(|_| Slot::default()).collect::<Vec<_>>();
    let c = NUM_CHANNELS;
    let h = obs_side(cfg.board as usize);
    let w = obs_side(cfg.board as usize);
    let obs_len = c * h * w;
    let n = cfg.num_snakes;
    let mut out_obs = Vec::new();
    let mut out_pol = Vec::new();
    let mut out_z = Vec::new();
    let mut actions = vec![Move::Up; n];
    let mut play_pols = vec![[0.0f32; 4]; n];
    let mut games = 0usize;
    let mut turn_total = 0usize;

    // Record full frames so the viewer can replay a handful of games each
    // generation. A generation ends once the sample target is hit, which the
    // shortest-finishing games satisfy, so we watch every slot and keep the first
    // `sample_games` that finish — otherwise generations with few finishers (long
    // games) produce no file. `rec_frames[g]` accumulates the current game on slot
    // `g`; buffers are cleared once we have enough, to bound memory.
    let mut rec_frames: Vec<Vec<FrameJson>> = vec![Vec::new(); cfg.count];
    let mut recorded: Vec<GameJson> = Vec::new();

    std::thread::scope(|scope| -> anyhow::Result<(Samples, Vec<GameJson>)> {
        let (tx_req, rx_req) = std::sync::mpsc::channel::<GpuReq>();
        let (tx_search, rx_search) = std::sync::mpsc::channel::<SearchReq>();
        let (tx_finish, rx_finish) = std::sync::mpsc::channel::<FinishReq>();
        let (tx_chunk, rx_chunk) = std::sync::mpsc::channel::<SampleChunk>();
        let gpu_counters = Arc::clone(&counters);
        let gpu_net = WorkerNet(net.net as *const snek_tch::AZNet);
        let gpu_device = net.device;
        let gpu = scope.spawn(move || -> anyhow::Result<()> {
            while let Ok(msg) = rx_req.recv() {
                let Some(job) = msg else { break };
                let started = Instant::now();
                let (pol, val) = gpu_net.infer(gpu_device, &job.obs, job.rows, c, h, w)?;
                gpu_counters
                    .gpu_forward_us
                    .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                gpu_counters.gpu_requests.fetch_add(1, Ordering::Relaxed);
                gpu_counters
                    .gpu_rows
                    .fetch_add(job.rows as u64, Ordering::Relaxed);
                gpu_counters
                    .inferences
                    .fetch_add(job.rows as u64, Ordering::Relaxed);
                if job.reply.send(GpuEval { pol, val }).is_err() {
                    break;
                }
            }
            Ok(())
        });
        let search_worker_count = search_worker_count(cfg);
        let rx_search = Arc::new(Mutex::new(rx_search));
        let mut search_workers = Vec::with_capacity(search_worker_count);
        for _ in 0..search_worker_count {
            let rx_search = Arc::clone(&rx_search);
            let tx_req = tx_req.clone();
            search_workers.push(scope.spawn(move || -> anyhow::Result<()> {
                loop {
                    let msg = rx_search
                        .lock()
                        .map_err(|_| anyhow::anyhow!("search queue lock poisoned"))?
                        .recv();
                    let Ok(Some(task)) = msg else {
                        break;
                    };
                    let result = search_shard(
                        &tx_req,
                        &task.boards,
                        task.c_puct,
                        task.draw_value,
                        task.sims,
                        stop,
                    )
                    .map(|(pol, val)| (task.sid, pol, val));
                    let _ = task.reply.send(result);
                }
                Ok(())
            }));
        }
        let draw_value = cfg.draw_value;
        let bootstrap_value = cfg.bootstrap_value;
        let collector = scope.spawn(move || -> anyhow::Result<()> {
            while let Ok(msg) = rx_finish.recv() {
                let Some(done) = msg else { break };
                if tx_chunk
                    .send(materialize_finished_game(
                        done,
                        n,
                        obs_len,
                        draw_value,
                        bootstrap_value,
                    ))
                    .is_err()
                {
                    break;
                }
            }
            Ok(())
        });

        let run_result = (|| -> anyhow::Result<()> {
            let mut pending_samples = 0usize;
            while out_z.len() + pending_samples < cfg.samples_per_gen
                && !stop.load(Ordering::Relaxed)
            {
                drain_sample_chunks(
                    &rx_chunk,
                    &mut out_obs,
                    &mut out_pol,
                    &mut out_z,
                    &mut pending_samples,
                    cfg.samples_per_gen,
                );
                if out_z.len() >= cfg.samples_per_gen {
                    break;
                }
                let (root_pol, root_val) = search_sharded(&tx_search, &boards, cfg)?;
                for (g, board) in boards.iter().enumerate() {
                    let slot = &mut slots[g];
                    for s in 0..n {
                        let base = slot.obs.len();
                        slot.obs.resize(base + obs_len, 0.0);
                        encode_into(board, s, &mut slot.obs[base..base + obs_len]);
                        slot.alive.push(board.snakes[s].alive());
                    }
                    slot.pol
                        .extend_from_slice(&root_pol[g * n * 4..(g + 1) * n * 4]);
                    slot.value.extend_from_slice(&root_val[g * n..(g + 1) * n]);
                    slot.steps += 1;
                }

                for g in 0..boards.len() {
                    for s in 0..n {
                        let base = (g * n + s) * 4;
                        let play_pol =
                            mask_obvious_immediate_deaths(&boards[g], s, &root_pol[base..base + 4]);
                        play_pols[s].copy_from_slice(&play_pol);
                        actions[s] = sample_move(&play_pol, cfg.exploration_prob, &mut rng);
                    }
                    if recorded.len() < cfg.sample_games && rec_frames[g].len() < MAX_REC_FRAMES {
                        rec_frames[g].push(frame_from_board(
                            &boards[g],
                            n,
                            &root_pol[g * n * 4..(g + 1) * n * 4],
                            &root_val[g * n..(g + 1) * n],
                            &play_pols,
                            &actions,
                        ));
                    }
                    boards[g].step_and_spawn(&actions, &mut rng);
                    turns[g] += 1;
                    turn_total += 1;
                }

                for g in 0..boards.len() {
                    let overrun = cfg.max_turns > 0 && turns[g] >= cfg.max_turns;
                    if !boards[g].is_terminal() && !overrun {
                        continue;
                    }
                    let winner = boards[g].winner();
                    let slot = std::mem::take(&mut slots[g]);
                    if winner.is_none()
                        && !overrun
                        && cfg.skip_short_draw_turns > 0
                        && slot.steps <= cfg.skip_short_draw_turns
                    {
                        rec_frames[g].clear();
                        boards[g] = standard_start(cfg.board, cfg.board, n, &mut rng);
                        turns[g] = 0;
                        continue;
                    }
                    {
                        let mut frames = std::mem::take(&mut rec_frames[g]);
                        if recorded.len() < cfg.sample_games && !frames.is_empty() {
                            // Append the terminal state (post-final-move) so the
                            // recording ends on the actual outcome. No search runs
                            // on a finished board, so policy is zero and each value
                            // is the terminal value (+1 winner / -1 loser / draw).
                            let term_vals: Vec<f32> = (0..n)
                                .map(|s| {
                                    terminal_value(
                                        winner,
                                        s,
                                        boards[g].snakes[s].alive(),
                                        cfg.draw_value,
                                    )
                                })
                                .collect();
                            frames.push(frame_from_board(
                                &boards[g],
                                n,
                                &vec![0.0f32; n * 4],
                                &term_vals,
                                &vec![[0.0f32; 4]; n],
                                &actions,
                            ));
                            let num_turns = frames.len() as u32;
                            recorded.push(GameJson {
                                frames,
                                winner: winner.map(|w| w as i32),
                                num_turns,
                            });
                            // Enough recorded: drop every slot's buffer so we stop
                            // holding frames for the rest of the generation.
                            if recorded.len() == cfg.sample_games {
                                for buf in rec_frames.iter_mut() {
                                    buf.clear();
                                    buf.shrink_to_fit();
                                }
                            }
                        }
                    }
                    pending_samples += slot.alive.iter().filter(|&&alive| alive).count();
                    tx_finish.send(Some(FinishedGame { slot, winner }))?;
                    games += 1;
                    counters.completed_games.fetch_add(1, Ordering::Relaxed);
                    boards[g] = standard_start(cfg.board, cfg.board, n, &mut rng);
                    turns[g] = 0;
                }
                drain_sample_chunks(
                    &rx_chunk,
                    &mut out_obs,
                    &mut out_pol,
                    &mut out_z,
                    &mut pending_samples,
                    cfg.samples_per_gen,
                );
                counters
                    .samples_collected
                    .store(out_z.len() as u32, Ordering::Relaxed);
            }
            Ok(())
        })();
        for _ in 0..search_worker_count {
            let _ = tx_search.send(None);
        }
        for worker in search_workers {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("search worker panicked"))??;
        }
        let _ = tx_req.send(None);
        let _ = tx_finish.send(None);
        gpu.join()
            .map_err(|_| anyhow::anyhow!("GPU worker panicked"))??;
        collector
            .join()
            .map_err(|_| anyhow::anyhow!("sample collector panicked"))??;
        let mut pending_samples = usize::MAX;
        drain_sample_chunks(
            &rx_chunk,
            &mut out_obs,
            &mut out_pol,
            &mut out_z,
            &mut pending_samples,
            cfg.samples_per_gen,
        );
        run_result?;

        Ok((
            Samples {
                obs: out_obs,
                pol: out_pol,
                z: out_z,
                obs_shape: [c, h, w],
                turns: turn_total,
                games,
            },
            recorded,
        ))
    })
}

fn materialize_finished_game(
    done: FinishedGame,
    n: usize,
    obs_len: usize,
    draw_value: f32,
    bootstrap_value: bool,
) -> SampleChunk {
    let slot = done.slot;
    let kept = slot.alive.iter().filter(|&&alive| alive).count();
    let mut obs = Vec::with_capacity(kept * obs_len);
    let mut pol = Vec::with_capacity(kept * 4);
    let mut z = Vec::with_capacity(kept);
    let final_alive_base = slot.steps.saturating_sub(1) * n;
    for st in 0..slot.steps {
        for s in 0..n {
            if !slot.alive[st * n + s] {
                continue;
            }
            let oi = (st * n + s) * obs_len;
            obs.extend_from_slice(&slot.obs[oi..oi + obs_len]);
            let pi = (st * n + s) * 4;
            pol.extend_from_slice(&slot.pol[pi..pi + 4]);
            z.push(if bootstrap_value {
                slot.value[st * n + s]
            } else {
                terminal_value(done.winner, s, slot.alive[final_alive_base + s], draw_value)
            });
        }
    }
    SampleChunk { obs, pol, z }
}

fn drain_sample_chunks(
    rx: &Receiver<SampleChunk>,
    out_obs: &mut Vec<f32>,
    out_pol: &mut Vec<f32>,
    out_z: &mut Vec<f32>,
    pending_samples: &mut usize,
    target: usize,
) {
    while let Ok(chunk) = rx.try_recv() {
        *pending_samples = pending_samples.saturating_sub(chunk.z.len());
        let remaining = target.saturating_sub(out_z.len());
        if remaining == 0 {
            continue;
        }
        let take = remaining.min(chunk.z.len());
        let obs_len = chunk.obs.len() / chunk.z.len().max(1);
        out_obs.extend_from_slice(&chunk.obs[..take * obs_len]);
        out_pol.extend_from_slice(&chunk.pol[..take * 4]);
        out_z.extend_from_slice(&chunk.z[..take]);
    }
}

fn search_sharded(
    tx_search: &Sender<SearchReq>,
    boards: &[Board],
    cfg: &RunConfig,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let n = cfg.num_snakes;
    let shard_games = cfg.gpu_batch_games.max(1);
    let shard_count = boards.chunks(shard_games).count();
    let mut shard_out = std::iter::repeat_with(|| None)
        .take(shard_count)
        .collect::<Vec<Option<(Vec<f32>, Vec<f32>)>>>();
    let (tx_done, rx_done) =
        std::sync::mpsc::channel::<anyhow::Result<(usize, Vec<f32>, Vec<f32>)>>();

    for (sid, shard) in boards.chunks(shard_games).enumerate() {
        tx_search.send(Some(SearchTask {
            sid,
            boards: shard.to_vec(),
            c_puct: cfg.c_puct,
            draw_value: cfg.draw_value,
            sims: cfg.sims,
            reply: tx_done.clone(),
        }))?;
    }
    drop(tx_done);
    for result in rx_done {
        let (sid, pol, val) = result?;
        shard_out[sid] = Some((pol, val));
    }

    let mut all_pol = Vec::with_capacity(boards.len() * n * 4);
    let mut all_val = Vec::with_capacity(boards.len() * n);
    for shard in shard_out {
        let (pol, val) = shard.ok_or_else(|| anyhow::anyhow!("missing search shard result"))?;
        all_pol.extend_from_slice(&pol);
        all_val.extend_from_slice(&val);
    }
    Ok((all_pol, all_val))
}

fn search_shard(
    tx_req: &Sender<GpuReq>,
    boards: &[Board],
    c_puct: f32,
    draw_value: f32,
    sims: usize,
    stop: &AtomicBool,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let mut forest = MctsForest::new_with_draw_value(boards, c_puct, draw_value);
    let mut pending = Vec::new();
    for sim in 0..sims {
        // Abort the search mid-generation the instant a stop is requested so the
        // trainer can interrupt immediately rather than finishing the turn.
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let (obs, rows) = select_write(&mut forest, &mut pending);
        if rows == 0 {
            break;
        }
        let (tx_eval, rx_eval) = std::sync::mpsc::channel();
        tx_req.send(Some(GpuJob {
            obs,
            rows,
            reply: tx_eval,
        }))?;
        let eval = rx_eval.recv()?;
        forest.expand_backup(&pending, &eval.pol, &eval.val);
        if sim == 1 {
            forest.freeze_forced_roots();
        }
    }
    Ok(forest.root_targets_serial())
}

fn select_write(forest: &mut MctsForest, pending: &mut Vec<usize>) -> (Vec<f32>, usize) {
    *pending = forest.select_serial();
    let rows = pending.len() * forest.n_snakes;
    let mut obs = vec![0.0f32; rows * forest.obs_size()];
    forest.write_pending_obs_serial(pending, &mut obs);
    (obs, rows)
}

fn search_worker_count(cfg: &RunConfig) -> usize {
    let shards = cfg.count.div_ceil(cfg.gpu_batch_games.max(1)).max(1);
    let configured = cfg.search_threads.max(1);
    configured.min(shards).max(1)
}

fn infer(
    net: &snek_tch::AZNet,
    device: Device,
    obs: &[f32],
    rows: usize,
    c: usize,
    h: usize,
    w: usize,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let x = Tensor::from_slice(obs)
        .reshape([rows as i64, c as i64, h as i64, w as i64])
        .to_device(device);
    let (logits, value) = no_grad(|| net.forward(&x));
    let probs = logits.softmax(-1, Kind::Float).to_device(Device::Cpu);
    let value = value.to_device(Device::Cpu);
    let mut pol = vec![0.0f32; rows * 4];
    let mut val = vec![0.0f32; rows];
    let pol_len = pol.len();
    let val_len = val.len();
    probs.copy_data(&mut pol, pol_len);
    value.copy_data(&mut val, val_len);
    Ok((pol, val))
}

fn sample_move<R: Rng>(policy: &[f32], exploration_prob: f32, rng: &mut R) -> Move {
    let mut p = [0.0f32; 4];
    let legal = policy.iter().filter(|&&v| v > 0.0).count().max(1);
    for i in 0..4 {
        p[i] = if policy[i] > 0.0 {
            (1.0 - exploration_prob) * policy[i] + exploration_prob / legal as f32
        } else {
            0.0
        };
    }
    let idx = WeightedIndex::new(p)
        .map(|d| d.sample(rng))
        .unwrap_or_else(|_| rng.gen_range(0..4));
    Move::from_index(idx)
}

fn terminal_value(winner: Option<usize>, snake: usize, alive_final: bool, draw_value: f32) -> f32 {
    match winner {
        Some(w) if w == snake => 1.0,
        Some(_) => -1.0,
        None if alive_final => draw_value,
        None => -1.0,
    }
}
