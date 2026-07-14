//! Offline supervised pre-training on external GameFile archives (e.g. scraped
//! Battlesnake ladder games), selected with `--sl-pretrain <dir>`.
//!
//! This is deliberately a *diagnostic*, not a production path. Behavioural
//! cloning on a horizon-~280 game compounds error as O(ε·T²) (Ross & Bagnell),
//! and at S/params ≈ 1 the net can memorise, so the honest way to know "how
//! much" is a held-out split with early stopping. We therefore:
//!   1. load + materialise the archive exactly as self-play would (same encoder,
//!      same value target), splitting BY GAME so no frames leak across the split;
//!   2. train a FRESH net with early stopping on validation loss;
//!   3. report what it actually learned — held-out move-match accuracy, value
//!      sign accuracy, and policy entropy (the one-hot overconfidence telltale) —
//!      and then its *play* strength vs the built-in baselines, which is the only
//!      test that matters.

use crate::config::RunConfig;
use crate::replay::{ReplayBuffer, Samples};
use crate::sample::GameFileJson;
use crate::selfplay::materialize::{game_matches_shape, materialize_game};
use crate::train::{build_optimizer, lr_for, train_one};
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use snek_core::{obs_side, NUM_CHANNELS};
use snek_server::arena::{play_game, Budget, GameSettings, Player, PlayerSpec};
use snek_tch::AZNet;
use std::path::{Path, PathBuf};
use tch::{nn, Device, Kind, Reduction, Tensor};

pub struct SlArgs {
    pub dir: PathBuf,
    pub out: PathBuf,
    pub epochs: usize,
    pub val_frac: f64,
    pub eval_games: usize,
    pub eval_ms: u64,
}

/// Held-out metrics, all averaged over the validation samples.
struct ValMetrics {
    policy_ce: f64,     // cross-entropy of the net's policy vs the expert one-hot
    top1: f64,          // fraction where argmax(net policy) == the expert move
    value_mse: f64,     // MSE of the value head vs the game outcome
    value_sign: f64,    // fraction where sign(value) == sign(outcome)
    net_entropy: f64,   // H(net policy) in nats; ln(4)=1.386 is uniform, 0 is collapsed
}

pub fn run(a: &SlArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        tch::Cuda::is_available(),
        "sl-pretrain needs CUDA; check LD_PRELOAD/LD_LIBRARY_PATH"
    );
    let cfg = RunConfig::default(); // board 11, 4 snakes, 96ch/8blk — matches the live run
    let n = cfg.num_snakes;
    let side = obs_side(cfg.board as usize);
    let obs_len = NUM_CHANNELS * side * side;
    let obs_shape = [NUM_CHANNELS, side, side];

    // ---- load + materialise, splitting BY GAME (no frame leakage across split) ----
    let mut files: Vec<PathBuf> = std::fs::read_dir(&a.dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".json.zst"))
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no .json.zst games in {}", a.dir.display());

    let empty = || Samples {
        obs: Vec::new(),
        pol: Vec::new(),
        z: Vec::new(),
        turn: Vec::new(),
        temp: Vec::new(),
        obs_shape,
        turns: 0,
        games: 0,
    };
    let mut train = empty();
    let mut val = empty();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0A7_5EED);
    let (mut n_train_games, mut n_val_games, mut skipped) = (0usize, 0usize, 0usize);
    for path in &files {
        let Some(bytes) = std::fs::read(path)
            .ok()
            .and_then(|raw| zstd::decode_all(&*raw).ok())
        else {
            skipped += 1;
            continue;
        };
        let Ok(parsed) = serde_json::from_slice::<GameFileJson>(&bytes) else {
            skipped += 1;
            continue;
        };
        for g in &parsed.games {
            if !game_matches_shape(g, cfg.board, n) {
                skipped += 1;
                continue;
            }
            let is_val = rng.gen::<f64>() < a.val_frac;
            if is_val {
                materialize_game(g, n, obs_len, cfg.draw_value, &mut val);
                n_val_games += 1;
            } else {
                materialize_game(g, n, obs_len, cfg.draw_value, &mut train);
                n_train_games += 1;
            }
        }
    }
    train.turns = train.len();
    train.games = n_train_games;
    val.turns = val.len();
    val.games = n_val_games;
    println!(
        "loaded: train {} games / {} samples · val {} games / {} samples · {} skipped",
        n_train_games,
        train.len(),
        n_val_games,
        val.len(),
        skipped
    );
    anyhow::ensure!(
        train.len() > 0 && val.len() > 0,
        "empty train or val split — check the archive"
    );
    // Anchors so the metrics below have meaning: a uniform policy scores CE=ln4,
    // and predicting value 0 everywhere scores MSE=mean(z^2).
    let uniform_ce = 4f64.ln();
    let val_z_var = val.z.iter().map(|&z| (z as f64).powi(2)).sum::<f64>() / val.len() as f64;
    println!(
        "anchors: uniform-policy CE = {:.3} (ln4) · predict-zero value MSE = {:.3}",
        uniform_ce, val_z_var
    );

    // ---- fresh net ----
    let device = Device::Cuda(0);
    let mut vs = nn::VarStore::new(device);
    let net = AZNet::new(
        &vs.root(),
        NUM_CHANNELS as i64,
        cfg.trunk_channels,
        cfg.trunk_blocks,
    );
    snek_tch::init_orthogonal(&vs, 2f64.sqrt());
    let mut opt = build_optimizer(&vs, &cfg)?;

    // ---- train with early stopping on validation loss ----
    let batch = cfg.batch_size.min(train.len());
    let steps_per_epoch = (train.len() / batch).max(1);
    let mut buf = ReplayBuffer::new(train.len() + 1);
    buf.add(train); // static dataset; recency=1 => uniform draws
    if let Some(parent) = a.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut samples_seen: u64 = 0;
    let mut best_val = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut patience = 0usize;
    const PATIENCE: usize = 2;
    let mut step_rng = Xoshiro256PlusPlus::seed_from_u64(0x5EED_1234);
    println!(
        "training: {} steps/epoch × up to {} epochs (batch {}), early-stop patience {}",
        steps_per_epoch, a.epochs, batch, PATIENCE
    );
    for epoch in 0..a.epochs {
        let mut last = crate::train::TrainMetrics::default();
        for _ in 0..steps_per_epoch {
            let Some(b) = buf.sample_batch(batch, 1.0, &mut step_rng) else {
                break;
            };
            opt.set_lr(lr_for(&cfg, samples_seen));
            last = train_one(&net, device, &mut opt, &b, cfg.value_weight)?;
            samples_seen += batch as u64;
        }
        let v = eval_val(&net, device, &val, obs_shape);
        let val_total = v.policy_ce + cfg.value_weight * v.value_mse;
        println!(
            "epoch {:>2} | train ploss {:.3} netH {:.3} | val CE {:.3} top1 {:.1}% · vMSE {:.3} vsign {:.1}% · netH {:.3} | total {:.3}{}",
            epoch,
            last.policy_loss,
            last.net_entropy,
            v.policy_ce,
            v.top1 * 100.0,
            v.value_mse,
            v.value_sign * 100.0,
            v.net_entropy,
            val_total,
            if val_total < best_val { "  <- best (saved)" } else { "" }
        );
        if val_total < best_val - 1e-4 {
            best_val = val_total;
            best_epoch = epoch;
            patience = 0;
            vs.save(&a.out)?;
        } else {
            patience += 1;
            if patience >= PATIENCE {
                println!("early stop: val loss did not improve for {PATIENCE} epochs");
                break;
            }
        }
    }
    // Guarantee a saved net even in the degenerate never-improved case.
    if !a.out.exists() {
        vs.save(&a.out)?;
    }
    println!(
        "best val {:.3} at epoch {} -> {}",
        best_val,
        best_epoch,
        a.out.display()
    );

    // Free the training varstore + optimizer before the play eval loads its own copy.
    drop(opt);
    drop(vs);
    drop(net);

    // ---- play-strength: the SL net (1 seat) vs 3 baseline seats ----
    // "Fair" for one seat in a 4-player game is a 25% win rate; beating that vs
    // floodfill means the scraped data is a genuinely useful bootstrap.
    println!(
        "\nplay eval: SL net vs 3 baselines, {} games each, {}ms/move (fair share = {:.0}%)",
        a.eval_games,
        a.eval_ms,
        100.0 / n as f64
    );
    for opponent in ["floodfill", "voronoi"] {
        match play_eval(&a.out, &cfg, device, opponent, a.eval_games, a.eval_ms) {
            Ok((wr, played)) => println!(
                "  vs {:<9} win rate {:.1}% over {} games",
                opponent,
                wr * 100.0,
                played
            ),
            Err(e) => eprintln!("  vs {opponent}: eval failed: {e}"),
        }
    }
    Ok(())
}

/// Full-strength round-robin among net checkpoints: seat the given gens as the
/// N players of a game, play many games rotating seat order, and report each
/// gen's win rate + average rank + pairwise later-vs-earlier record. Answers
/// "is there a real skill trend the low-sim league hides?" independently of the
/// Elo system — a direct head-to-head at full search.
pub fn round_robin(
    runs_dir: &Path,
    run_id: &str,
    gens: &[u32],
    games: usize,
    sims: usize,
    conc: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(gens.len() >= 2, "need >=2 gens");
    let cfg = RunConfig::default();
    // CPU: play_game runs each game to completion with small, low-latency
    // forwards (no GPU kernel-launch serialization across seats).
    let device = Device::Cpu;
    let budget = Budget::Sims(sims.max(1));
    let nseats = gens.len();
    let settings = GameSettings {
        seats: nseats,
        board_size: cfg.board,
        // Cap game length: strong nets run to a natural terminal ~300+ turns,
        // and a rank read doesn't need the full endgame (survivors at the cap
        // tie at the top, weaker seats have already died). Halves wall-clock.
        max_turns: 250,
        cfg: snek_server::Config {
            max_sims: sims.max(1),
            c_puct: cfg.c_puct,
            draw_value: cfg.draw_value,
            eval_chunk: 4096,
            // Bigger leaf batches per forward (vs the league's 8) → ~4× fewer GPU
            // round-trips per move, which is what the arena is bottlenecked on.
            leaves_per_sim: 32,
            virtual_loss: 1.0,
        },
        budget,
        torch_threads: 1,
    };
    // Checkpoint paths (player index i == gens[i]).
    let paths: Vec<String> = gens
        .iter()
        .map(|&g| {
            runs_dir
                .join(run_id)
                .join("checkpoints")
                .join(format!("net_{g:04}.safetensors"))
                .display()
                .to_string()
        })
        .collect();
    for p in &paths {
        anyhow::ensure!(std::path::Path::new(p).exists(), "missing checkpoint {p}");
    }
    let _ = budget;
    // One game at a time leaves the GPU ~idle (tiny per-move forwards, no
    // cross-game batching). Instead play `concurrency` independent games at once
    // — each worker thread owns its own copy of the 4-net roster, and the many
    // concurrent forwards keep the GPU busy (the burst's matchmaking is the part
    // we don't want; its concurrency is). Seat rotation uses the *global* game
    // index so every net still plays every seat equally.
    let concurrency = conc.max(1).min(games.max(1));
    println!(
        "round-robin: gens {:?} · {} games · {} sims/move (league uses 64) · {} concurrent · seats rotated",
        gens, games, sims, concurrency
    );
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    let rank_sum = Mutex::new(vec![0u64; nseats]); // sum of ranks (1=best)
    let wins = Mutex::new(vec![0u64; nseats]); // rank-1 finishes (ties co-count)
    let beats = Mutex::new(vec![vec![0u64; nseats]; nseats]); // beats[i][j] = i outranked j
    let done = AtomicU64::new(0);
    std::thread::scope(|scope| {
        for w in 0..concurrency {
            let (paths, settings, cfg) = (&paths, &settings, &cfg);
            let (rank_sum, wins, beats, done) = (&rank_sum, &wins, &beats, &done);
            scope.spawn(move || {
                // Each worker loads its own roster (nets aren't shared across
                // concurrent games — a search needs `&mut` its net).
                let mut players: Vec<Player> = paths
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        Player::load(
                            &PlayerSpec::Net(p.clone()),
                            device,
                            cfg.trunk_channels,
                            cfg.trunk_blocks,
                            settings.budget,
                            &format!("gen{}", gens[i]),
                        )
                        .expect("load checkpoint")
                    })
                    .collect();
                let never = || false;
                let mut sink = |_f: &snek_server::arena::Frame| {};
                let (mut lrank, mut lwin) = (vec![0u64; nseats], vec![0u64; nseats]);
                let mut lbeat = vec![vec![0u64; nseats]; nseats];
                let mut g = w;
                while g < games {
                    let rotation = g % nseats;
                    let seed = (g as u64).wrapping_mul(2_654_435_761).wrapping_add(1);
                    if let Some(out) =
                        play_game(&mut players, rotation, settings, seed, &never, &mut sink)
                    {
                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        let mut rank_of = vec![0u32; nseats];
                        for p in &out.placements {
                            let pi = p.player as usize;
                            rank_of[pi] = p.rank;
                            lrank[pi] += p.rank as u64;
                            if p.rank == 1 {
                                lwin[pi] += 1;
                            }
                        }
                        for i in 0..nseats {
                            for j in 0..nseats {
                                if i != j && rank_of[i] < rank_of[j] {
                                    lbeat[i][j] += 1;
                                }
                            }
                        }
                        let winner = (0..nseats)
                            .find(|&i| rank_of[i] == 1)
                            .map(|i| gens[i])
                            .unwrap_or(0);
                        eprintln!(
                            "  game {d}/{games} done ({} turns): winner gen {winner}",
                            out.turns
                        );
                    }
                    g += concurrency;
                }
                let mut rs = rank_sum.lock().unwrap();
                let mut ws = wins.lock().unwrap();
                let mut bs = beats.lock().unwrap();
                for i in 0..nseats {
                    rs[i] += lrank[i];
                    ws[i] += lwin[i];
                    for j in 0..nseats {
                        bs[i][j] += lbeat[i][j];
                    }
                }
            });
        }
    });
    let rank_sum = rank_sum.into_inner().unwrap();
    let wins = wins.into_inner().unwrap();
    let beats = beats.into_inner().unwrap();
    let done = done.load(Ordering::Relaxed);
    anyhow::ensure!(done > 0, "no games completed");
    let d = done as f64;
    println!("\ncompleted {done} games:");
    println!("  {:>6}   win%   avg_rank", "gen");
    for (i, &g) in gens.iter().enumerate() {
        println!(
            "  {:>6}  {:>5.1}   {:.3}",
            g,
            100.0 * wins[i] as f64 / d,
            rank_sum[i] as f64 / d
        );
    }
    // Pairwise later-vs-earlier: for each pair with gj>gi, how often the LATER
    // gen outranked the earlier one. >50% => later is genuinely stronger.
    println!("\n  later-vs-earlier head-to-head (later gen outranks earlier, % of decisive games):");
    for i in 0..nseats {
        for j in (i + 1)..nseats {
            // gens are passed in ascending order; j is the later gen
            let later_wins = beats[j][i];
            let earlier_wins = beats[i][j];
            let decisive = later_wins + earlier_wins;
            let pct = if decisive > 0 {
                100.0 * later_wins as f64 / decisive as f64
            } else {
                f64::NAN
            };
            println!(
                "    gen {:>5} vs gen {:>5}: {:>5.1}%  ({} decisive)",
                gens[j], gens[i], pct, decisive
            );
        }
    }
    Ok(())
}

/// Play-strength only: skip training and measure an already-saved net vs the
/// baselines. Lets us re-evaluate any checkpoint without re-running SL.
pub fn eval_only(net_path: &Path, games: usize, eval_ms: u64) -> anyhow::Result<()> {
    anyhow::ensure!(tch::Cuda::is_available(), "eval needs CUDA");
    let cfg = RunConfig::default();
    let device = Device::Cuda(0);
    println!(
        "play eval (net {}): {} games/baseline, {}ms/move (fair share = {:.0}%)",
        net_path.display(),
        games,
        eval_ms,
        100.0 / cfg.num_snakes as f64
    );
    for opponent in ["floodfill", "voronoi"] {
        match play_eval(net_path, &cfg, device, opponent, games, eval_ms) {
            Ok((wr, played)) => println!(
                "  vs {:<9} win rate {:.1}% over {} games",
                opponent,
                wr * 100.0,
                played
            ),
            Err(e) => eprintln!("  vs {opponent}: eval failed: {e}"),
        }
    }
    Ok(())
}

/// Forward the whole validation set in no-grad batches and average the metrics.
fn eval_val(net: &AZNet, device: Device, val: &Samples, shape: [usize; 3]) -> ValMetrics {
    let [c, h, w] = shape;
    let obs_len = c * h * w;
    let total = val.len();
    let bs = 8192usize;
    let (mut ce, mut top1, mut vmse, mut vsign, mut ent) = (0.0, 0i64, 0.0, 0i64, 0.0);
    tch::no_grad(|| {
        let mut i = 0;
        while i < total {
            let end = (i + bs).min(total);
            let b = (end - i) as i64;
            let obs = Tensor::from_slice(&val.obs[i * obs_len..end * obs_len])
                .reshape([b, c as i64, h as i64, w as i64])
                .to_device(device);
            let tpol = Tensor::from_slice(&val.pol[i * 4..end * 4])
                .reshape([b, 4])
                .to_device(device);
            let tz = Tensor::from_slice(&val.z[i..end]).to_device(device);
            let (logits, value) = net.forward(&obs);
            let logp = logits.log_softmax(-1, Kind::Float);
            ce += (-(&tpol * &logp).sum_dim_intlist(&[1i64][..], false, Kind::Float))
                .sum(Kind::Float)
                .double_value(&[]);
            ent += (-(logp.exp() * &logp).sum_dim_intlist(&[1i64][..], false, Kind::Float))
                .sum(Kind::Float)
                .double_value(&[]);
            top1 += logits
                .argmax(-1, false)
                .eq_tensor(&tpol.argmax(-1, false))
                .sum(Kind::Int64)
                .int64_value(&[]);
            vmse += value.mse_loss(&tz, Reduction::Sum).double_value(&[]);
            vsign += value
                .sign()
                .eq_tensor(&tz.sign())
                .sum(Kind::Int64)
                .int64_value(&[]);
            i = end;
        }
    });
    let t = total as f64;
    ValMetrics {
        policy_ce: ce / t,
        top1: top1 as f64 / t,
        value_mse: vmse / t,
        value_sign: vsign as f64 / t,
        net_entropy: ent / t,
    }
}

/// Play `games` of [SL net, baseline, baseline, baseline] and return the net's
/// win rate (its seat placing rank 1) plus how many games actually completed.
fn play_eval(
    net_path: &Path,
    cfg: &RunConfig,
    device: Device,
    opponent: &str,
    games: usize,
    budget_ms: u64,
) -> anyhow::Result<(f64, usize)> {
    let budget = Budget::TimeMs(budget_ms);
    let settings = GameSettings {
        seats: cfg.num_snakes,
        board_size: cfg.board,
        max_turns: 400,
        cfg: snek_server::Config {
            max_sims: 100_000, // bounded by the TimeMs budget in practice
            c_puct: cfg.c_puct,
            draw_value: cfg.draw_value,
            eval_chunk: 4096,
            leaves_per_sim: 8,
            virtual_loss: 1.0,
        },
        budget,
        torch_threads: 1,
    };
    // Player 0 is the SL net (on GPU); the rest are the baseline. Loaded once and
    // reused across games — play_game only borrows them.
    let mut players: Vec<Player> = Vec::with_capacity(cfg.num_snakes);
    players.push(Player::load(
        &PlayerSpec::Net(net_path.display().to_string()),
        device,
        cfg.trunk_channels,
        cfg.trunk_blocks,
        budget,
        "sl",
    )?);
    for _ in 1..cfg.num_snakes {
        players.push(Player::load(
            &PlayerSpec::parse(opponent),
            device,
            cfg.trunk_channels,
            cfg.trunk_blocks,
            budget,
            opponent,
        )?);
    }
    let never = || false;
    let mut sink = |_f: &snek_server::arena::Frame| {};
    let (mut wins, mut played) = (0usize, 0usize);
    for game in 0..games {
        let rotation = game % players.len(); // rotate the net through every seat
        let seed = (game as u64).wrapping_mul(1_000_003).wrapping_add(0xABC);
        if let Some(out) = play_game(&mut players, rotation, &settings, seed, &never, &mut sink) {
            played += 1;
            if out.placements.iter().any(|p| p.player == 0 && p.rank == 1) {
                wins += 1;
            }
        }
    }
    Ok((wins as f64 / played.max(1) as f64, played))
}
