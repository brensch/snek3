"""Training loop: alternate self-play data generation and supervised updates to
the policy (cross-entropy to the search policy) and value (MSE to game outcome).

Usage:
    python -m azsnek.train --generations 50 --samples 20000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path


def _setup_ort_env():
    """Point the Rust self-play (snek.generate_selfplay -> onnxruntime/CUDA) at
    the venv's onnxruntime + CUDA libs. Self-contained; no launcher env needed.
    Must run before onnxruntime is loaded."""
    import glob
    import sys
    sp = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    so = glob.glob(os.path.join(sp, "onnxruntime", "capi", "libonnxruntime.so.*"))
    if not so:
        return
    os.environ.setdefault("ORT_DYLIB_PATH", so[0])
    libdirs = [os.path.join(sp, "onnxruntime", "capi")] + glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")


_setup_ort_env()

import numpy as np
import snek
import torch
import torch.nn.functional as F

from .evaluate import evaluate, evaluate_vs_net
from .net import AZNet, NetConfig, autocast as net_autocast, device_auto
from .recorder import record_games
from .runlog import RunWriter
from .selfplay import ReplayBuffer, Samples, SelfPlayConfig, generate
from .autotune import TuneLimits, TuneSettings, tune_next


PHASE_COLORS = {
    "SETUP": "\033[36m",
    "RESUME": "\033[36m",
    "PLAYING": "\033[35m",
    "TRAINING": "\033[34m",
    "EVALUATING": "\033[33m",
    "RELATIVE": "\033[33m",
    "RECORDING": "\033[32m",
    "SAVING": "\033[90m",
    "METRICS": "\033[36m",
    "ADAPTIVE": "\033[35m",
    "DONE": "\033[32m",
    "WARN": "\033[31m",
}
RESET = "\033[0m"


def setup_logger() -> logging.Logger:
    """Human-oriented run logger: timestamped phases, color on terminals only."""
    logger = logging.getLogger("azsnek.train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def _color_enabled() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def log_phase(logger: logging.Logger, phase: str, message: str) -> None:
    label = f"{phase:<10}"
    if _color_enabled():
        label = f"{PHASE_COLORS.get(phase, '')}{label}{RESET}"
    logger.info("%s | %s", label, message)


def train_on_samples(
    net: AZNet,
    opt: torch.optim.Optimizer,
    samples,
    device: torch.device,
    steps: int = 256,
    batch_size: int = 1024,
    value_weight: float = 1.0,
) -> dict:
    """Run `steps` SGD updates on minibatches drawn uniformly (with replacement)
    from `samples` — a replay buffer's worth of recent positions."""
    # Keep the replay window in CPU RAM. The old path copied the full replay
    # buffer to GPU every generation, so GPU memory grew with buffer size and
    # PyTorch's caching allocator retained the high-water mark. Only the current
    # minibatch needs to live on the GPU.
    obs = samples.obs
    pol = samples.pol
    z = samples.z
    n = obs.shape[0]

    net.train()
    pl = vl = 0.0
    for _ in range(steps):
        idx = np.random.randint(0, n, size=min(batch_size, n))
        obs_b = torch.from_numpy(obs[idx]).to(device, non_blocking=True)
        pol_b = torch.from_numpy(pol[idx]).to(device, non_blocking=True)
        z_b = torch.from_numpy(z[idx]).to(device, non_blocking=True)
        with net_autocast(device):
            logits, value = net(obs_b)
            logp = F.log_softmax(logits, dim=1)
            # Soft-target cross-entropy; illegal moves have target 0.
            policy_loss = -(pol_b * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, z_b)
            loss = policy_loss + value_weight * value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pl += float(policy_loss.item())
        vl += float(value_loss.item())
        del obs_b, pol_b, z_b, logits, value, logp, policy_loss, value_loss, loss
    return {"policy_loss": pl / steps, "value_loss": vl / steps}


def policy_target_stats(pol: np.ndarray) -> dict:
    """Information content of the search policy targets."""
    p = np.clip(pol, 1e-9, 1.0)
    entropy = -(pol * np.log(p)).sum(axis=1)
    return {
        "target_entropy": float(entropy.mean()),
        "target_max_prob": float(pol.max(axis=1).mean()),
    }


def summarize_completed_games(games: list[dict]) -> dict:
    """Aggregate Rust self-play game summaries for dashboard inspection."""
    if not games:
        return {
            "completed_games": 0,
            "games": [],
            "length_histogram": [],
        }
    turns = np.array([int(g.get("turns", 0)) for g in games], dtype=np.int32)
    wins = sum(1 for g in games if g.get("winner") == 0)
    losses = sum(1 for g in games if g.get("winner") == 1)
    draws = len(games) - wins - losses
    overruns = sum(1 for g in games if g.get("overrun"))
    short_draws = sum(1 for g in games if g.get("short_draw"))
    terminal_draws = max(0, draws - overruns)
    max_turn = int(turns.max())
    bucket = 10
    hist = []
    for start in range(0, max_turn + bucket, bucket):
        end = start + bucket - 1
        count = int(((turns >= start) & (turns <= end)).sum())
        if count:
            hist.append({"min": start, "max": end, "count": count})
    decisive = wins + losses
    return {
        "completed_games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "overrun_draws": overruns,
        "terminal_draws": terminal_draws,
        "short_draws": short_draws,
        "win_rate": round((wins + 0.5 * draws) / len(games), 4),
        "decisive_win_rate": round(wins / decisive, 4) if decisive else None,
        "total_samples": int(sum(int(g.get("samples", 0)) for g in games)),
        "turns": {
            "min": int(turns.min()),
            "max": max_turn,
            "mean": round(float(turns.mean()), 2),
            "p50": int(np.percentile(turns, 50)),
            "p90": int(np.percentile(turns, 90)),
            "p95": int(np.percentile(turns, 95)),
        },
        "length_histogram": hist,
        "games": games,
    }


@torch.no_grad()
def export_onnx(net, channels: int, board: int, device, path) -> None:
    """Export the current net to ONNX so the Rust self-play can run it on GPU."""
    import warnings

    net.eval()
    dummy = torch.zeros(1, channels, board, board, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net, dummy, str(path),
            input_names=["obs"], output_names=["policy_logits", "value"],
            dynamic_axes={"obs": {0: "batch"}, "policy_logits": {0: "batch"}, "value": {0: "batch"}},
            opset_version=17, dynamo=False,
        )


def main():
    logger = setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--samples", type=int, default=50_000)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=8192,
        help="leaf observations per neural-net eval chunk; lower reduces eval tensor memory",
    )
    ap.add_argument(
        "--search-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Rayon threads for Rust search/encoding (default: all visible CPUs; 0 leaves Rayon default)",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-steps", type=int, default=1024, help="SGD steps per generation")
    ap.add_argument("--batch-size", type=int, default=2048, help="SGD minibatch size")
    ap.add_argument("--buffer-size", type=int, default=500_000, help="replay buffer capacity (samples)")
    ap.add_argument("--max-turns", type=int, default=0, help="0 plays until terminal; positive values cap games as draws")
    # AlphaZero MCTS search.
    ap.add_argument("--sims", type=int, default=128, help="MCTS simulations per move")
    ap.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant")
    ap.add_argument("--exploration-prob", type=float, default=0.15, help="uniform-legal mix into the played action")
    ap.add_argument("--draw-value", type=float, default=-0.25, help="value/search target for all draws")
    ap.add_argument("--skip-short-draw-turns", type=int, default=0, help="drop terminal draw games up to this many turns from replay; 0 disables")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-games", type=int, default=32)
    ap.add_argument("--league-every", type=int, default=20, help="snapshot a league checkpoint every N gens")
    ap.add_argument("--league-keep", type=int, default=8, help="keep this many recent league checkpoints (plus the anchor)")
    ap.add_argument("--relative-every", type=int, default=5, help="measure self-play win_rate vs past checkpoints every N gens")
    ap.add_argument("--relative-games", type=int, default=64, help="games per relative (net-vs-past) eval")
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--ckpt-dir", type=str, default=None, help="serving weights dir (default: runs/<run-id>/ckpt)")
    ap.add_argument("--runs-dir", type=str, default="runs", help="dashboard run root")
    ap.add_argument("--run-id", type=str, default=None, help="run dir name (default: timestamp)")
    ap.add_argument("--sample-games", type=int, default=16, help="cheap self-play replay samples captured in Rust")
    ap.add_argument("--sample-every", type=int, default=1, help="capture Rust replay samples every N generations")
    ap.add_argument("--record-games", type=int, default=8, help="replays per opponent per recording")
    ap.add_argument("--record-every", type=int, default=1, help="record replays every N generations")
    ap.add_argument("--keep-games", type=int, default=40, help="keep this many recent game files")
    ap.add_argument("--adaptive", action="store_true", help="adapt samples/train_steps/eval_games/tau during this run")
    ap.add_argument("--adaptive-every", type=int, default=4, help="retune every N generations")
    ap.add_argument("--min-train-steps", type=int, default=64)
    ap.add_argument("--max-train-steps", type=int, default=512)
    ap.add_argument("--min-samples", type=int, default=24_000)
    ap.add_argument("--max-samples", type=int, default=120_000)
    ap.add_argument("--target-buffer-epochs", type=float, default=1.5)
    ap.add_argument("--max-new-sample-epochs", type=float, default=12.0)
    ap.add_argument("--fresh", action="store_true", help="ignore saved state and restart this run-id from scratch")
    ap.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)  # deprecated: resume is the default
    args = ap.parse_args()

    if args.search_threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.search_threads)
        search_threads_configured = snek.set_search_threads(args.search_threads)
    else:
        search_threads_configured = False

    device = device_auto()
    log_phase(logger, "SETUP", f"device={device}")
    if args.search_threads:
        status = "configured" if search_threads_configured else "already initialized"
        log_phase(logger, "SETUP", f"search_threads={args.search_threads} ({status})")

    sp = SelfPlayConfig(
        count=args.count,
        sims=args.sims,
        c_puct=args.c_puct,
        eval_batch_size=args.eval_batch_size,
        samples_per_gen=args.samples,
        max_turns=args.max_turns,
        exploration_prob=args.exploration_prob,
    )
    run = RunWriter(
        args.runs_dir,
        run_id=args.run_id,
        meta={
            "board": sp.board,
            "num_snakes": sp.num_snakes,
            "filters": args.filters,
            "blocks": args.blocks,
            "depth": args.depth,
            "tau": args.tau,
            "iters": args.iters,
            "eval_batch_size": args.eval_batch_size,
            "max_turns": args.max_turns,
            "exploration_prob": args.exploration_prob,
            "draw_value": args.draw_value,
            "skip_short_draw_turns": args.skip_short_draw_turns,
            "search_threads": args.search_threads,
            "generations": args.generations,
            "samples_per_gen": args.samples,
            "sample_games": args.sample_games,
            "sample_every": args.sample_every,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "adaptive": args.adaptive,
            "adaptive_every": args.adaptive_every,
            "device": str(device),
        },
    )
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else run.dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_phase(logger, "SETUP", f"run_id={run.run_id} ckpt_dir={ckpt_dir}")
    run.write_json("meta.json", {**run.read_json("meta.json"), "ckpt_dir": str(ckpt_dir)})

    # Resume automatically when this run-id has saved state, unless --fresh.
    resume = None
    if run.has_state() and not args.fresh:
        resume = torch.load(run.state_path, map_location=device, weights_only=False)
        cfg = NetConfig(**resume["net_cfg"])
    else:
        if args.fresh and run.has_state():
            run.reset()
            log_phase(logger, "RESUME", f"--fresh cleared previous progress in {run.dir}")
        cfg = NetConfig(channels=snek.CHANNELS, filters=args.filters, blocks=args.blocks)

    net = AZNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    start_gen, best_win = 0, -1.0

    if resume is not None:
        net.load_state_dict(resume["net"])
        opt.load_state_dict(resume["opt"])
        start_gen = resume["gen"] + 1
        best_win = resume["best_win"]
        try:  # best-effort RNG restore
            torch.set_rng_state(resume["torch_rng"].cpu())
            if resume.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in resume["cuda_rng"]])
            np.random.set_state(resume["np_rng"])
            random.setstate(resume["py_rng"])
        except Exception as e:  # noqa: BLE001
            log_phase(logger, "WARN", f"could not fully restore RNG state: {e}")
        log_phase(logger, "RESUME", f"resumed run {run.run_id} at generation {start_gen}")
    else:
        log_phase(logger, "RESUME", f"run_dir={run.dir} fresh_start=true")

    def save_state(gen: int):
        run.save_state(
            lambda p: torch.save(
                {
                    "gen": gen,
                    "best_win": best_win,
                    "net_cfg": asdict(cfg),
                    "net": net.state_dict(),
                    "opt": opt.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "np_rng": np.random.get_state(),
                    "py_rng": random.getstate(),
                },
                p,
            )
        )

    run.write_status(
        {"generation": start_gen - 1, "running": True, "total_generations": args.generations}
    )

    # Relative-skill "league": snapshot past nets and play current vs them at the
    # (shallow) training depth. Isolates net improvement from the fixed baseline.
    league_dir = ckpt_dir / "league"
    league_dir.mkdir(parents=True, exist_ok=True)
    opp_net = AZNet(cfg).to(device)

    def league_ckpts():
        return sorted(league_dir.glob("gen_*.pt"), key=lambda p: int(p.stem.split("_")[1]))

    def relative_winrates(gen: int) -> dict:
        ckpts = league_ckpts()
        if not ckpts:
            return {}
        out = {}
        opp_net.load_state_dict(torch.load(ckpts[0], map_location=device))  # anchor (phase start)
        out["self_vs_anchor"] = round(
            evaluate_vs_net(net, opp_net, device, games=args.relative_games, sims=args.sims,
                            c_puct=args.c_puct, eval_batch_size=args.eval_batch_size,
                            max_turns=args.max_turns, seed=3000 + gen), 3)
        if len(ckpts) >= 2:  # rolling: a checkpoint a couple league-steps back
            recent = ckpts[max(0, len(ckpts) - 3)]
            opp_net.load_state_dict(torch.load(recent, map_location=device))
            out["self_vs_recent"] = round(
                evaluate_vs_net(net, opp_net, device, games=args.relative_games, sims=args.sims,
                                c_puct=args.c_puct, eval_batch_size=args.eval_batch_size,
                                max_turns=args.max_turns, seed=5000 + gen), 3)
        return out

    buffer = ReplayBuffer(args.buffer_size)
    metrics_history = []
    if run.metrics_path.exists() and not args.fresh:
        for line in run.metrics_path.read_text().splitlines():
            if line.strip():
                try:
                    metrics_history.append(json.loads(line))
                except ValueError:
                    pass

    tune_limits = TuneLimits(
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        min_train_steps=args.min_train_steps,
        max_train_steps=args.max_train_steps,
        target_buffer_epochs=args.target_buffer_epochs,
        max_new_sample_epochs=args.max_new_sample_epochs,
    )

    def retune(gen: int):
        if not args.adaptive or not args.adaptive_every or (gen + 1) % args.adaptive_every != 0:
            return
        settings = TuneSettings(
            samples=args.samples,
            count=args.count,
            depth=args.depth,
            tau=args.tau,
            iters=args.iters,
            eval_batch_size=args.eval_batch_size,
            search_threads=args.search_threads,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            filters=args.filters,
            blocks=args.blocks,
            eval_games=args.eval_games,
            max_turns=args.max_turns,
            sample_games=args.sample_games,
            sample_every=args.sample_every,
            record_games=args.record_games,
            record_every=args.record_every,
        )
        tuned, reasons = tune_next(settings, tune_limits, metrics_history)
        args.samples = tuned.samples
        args.train_steps = tuned.train_steps
        args.eval_games = tuned.eval_games
        args.tau = tuned.tau
        sp.samples_per_gen = tuned.samples
        sp.tau = tuned.tau
        meta = run.read_json("meta.json")
        meta.update(
            {
                "tau": args.tau,
                "samples_per_gen": args.samples,
                "train_steps": args.train_steps,
                "eval_games": args.eval_games,
                "adaptive_last_gen": gen,
                "adaptive_last_reasons": reasons,
            }
        )
        run.write_json("meta.json", meta)
        log_phase(
            logger,
            "ADAPTIVE",
            f"samples={args.samples} train_steps={args.train_steps} "
            f"eval_games={args.eval_games} tau={args.tau} | " + "; ".join(reasons),
        )

    onnx_path = run.dir / "model.onnx"
    for gen in range(start_gen, args.generations):
        # ---- GENERATE: Rust MCTS + ONNX/CUDA inference (no Python round-trips) ----
        log_phase(
            logger,
            "PLAYING",
            f"gen={gen} count={args.count} sims={args.sims} target_samples={args.samples}",
        )
        t0 = time.time()
        export_onnx(net, cfg.channels, sp.board, device, onnx_path)
        t_export = time.time() - t0
        rust_sample_games = (
            args.sample_games
            if args.sample_games > 0 and args.sample_every and gen % args.sample_every == 0
            else 0
        )
        generated = snek.generate_selfplay(
            str(onnx_path), board=sp.board, num_snakes=sp.num_snakes,
            count=args.count, sims=args.sims, c_puct=args.c_puct,
            samples_per_gen=args.samples, seed=1000 + gen,
            exploration_prob=args.exploration_prob, max_turns=args.max_turns,
            draw_value=args.draw_value, skip_short_draw_turns=args.skip_short_draw_turns,
            record_games=rust_sample_games,
        )
        if len(generated) == 4:
            obs, pol, z, gen_stats = generated
            gen_stats = dict(gen_stats)
        else:
            obs, pol, z = generated
            gen_stats = {}
        samples = Samples(obs=obs, pol=pol, z=z, turns=int(z.shape[0]), games=0)
        target_stats = policy_target_stats(samples.pol)
        buffer.add(samples)
        t_gen = time.time() - t0
        n_samp = samples.obs.shape[0]
        sampled_games = []
        raw_sampled_games = gen_stats.pop("recorded_games_json", []) if gen_stats else []
        for raw_game in raw_sampled_games:
            try:
                sampled_games.append(json.loads(raw_game))
            except (TypeError, ValueError):
                log_phase(logger, "WARN", f"gen={gen} could not parse sampled replay JSON")
        completed_games = []
        raw_completed_games = gen_stats.pop("completed_games_json", []) if gen_stats else []
        for raw_game in raw_completed_games:
            try:
                completed_games.append(json.loads(raw_game))
            except (TypeError, ValueError):
                log_phase(logger, "WARN", f"gen={gen} could not parse completed game summary JSON")
        selfplay_summary = summarize_completed_games(completed_games)
        log_phase(
            logger,
            "PLAYING",
            f"gen={gen} done samples={n_samp:,} seconds={t_gen:.1f} "
            f"samples_per_sec={n_samp / max(t_gen, 1e-9):.0f}"
            + (
                f" inference_per_sec={gen_stats['inference_per_sec']:,.0f}"
                if gen_stats.get("inference_per_sec") is not None else ""
            )
            + (
                f" gpu_busy={gen_stats['gpu_busy_pct']:.1f}%"
                if gen_stats.get("gpu_busy_pct") is not None else ""
            )
            + (
                f" skipped_short_draws={gen_stats['skipped_short_draw_games']}"
                if gen_stats.get("skipped_short_draw_games") is not None else ""
            )
            + (
                f" sample_games={len(sampled_games)}"
                if sampled_games else ""
            )
            + (
                f" completed_games={selfplay_summary['completed_games']}"
                if selfplay_summary.get("completed_games") else ""
            )
            + f" onnx_export={t_export:.1f}s",
        )

        # ---- TRAIN: PyTorch SGD on a window of recent games ----
        log_phase(
            logger,
            "TRAINING",
            f"gen={gen} steps={args.train_steps} batch={args.batch_size} buffer={len(buffer):,}",
        )
        t1 = time.time()
        losses = train_on_samples(
            net, opt, buffer.dataset(), device,
            steps=args.train_steps, batch_size=args.batch_size,
        )
        t_train = time.time() - t1
        log_phase(
            logger,
            "TRAINING",
            f"gen={gen} done seconds={t_train:.1f} steps_per_sec={args.train_steps / max(t_train, 1e-9):.1f} "
            f"policy_loss={losses['policy_loss']:.4f} value_loss={losses['value_loss']:.4f}",
        )

        turns_per_sec = samples.turns / max(t_gen, 1e-9)
        games_per_sec = int(selfplay_summary.get("completed_games", 0)) / max(t_gen, 1e-9)
        metric = {
            "gen": gen,
            "samples": int(samples.obs.shape[0]),
            "buffer": len(buffer),
            "policy_loss": round(losses["policy_loss"], 4),
            "value_loss": round(losses["value_loss"], 4),
            "target_entropy": round(target_stats["target_entropy"], 4),
            "target_max_prob": round(target_stats["target_max_prob"], 4),
            "gen_seconds": round(t_gen, 1),
            "train_seconds": round(t_train, 1),
            "turns_per_sec": round(turns_per_sec, 0),
            "games_per_sec": round(games_per_sec, 2),
            "win_rate": None,
            "sample_games": len(sampled_games),
            "completed_games": int(selfplay_summary.get("completed_games", 0)),
        }
        if gen_stats:
            metric.update(
                inference_count=int(gen_stats.get("inference_count", 0)),
                inference_seconds=round(float(gen_stats.get("inference_seconds", 0.0)), 2),
                inference_per_sec=round(float(gen_stats.get("inference_per_sec", 0.0)), 0),
                gpu_busy_pct=round(float(gen_stats.get("gpu_busy_pct", 0.0)), 1),
                gpu_forward_seconds=round(float(gen_stats.get("gpu_forward_seconds", 0.0)), 2),
                gpu_idle_seconds=round(float(gen_stats.get("gpu_idle_seconds", 0.0)), 2),
                cpu_recv_wait_seconds=round(float(gen_stats.get("cpu_recv_wait_seconds", 0.0)), 2),
                cpu_mcts_seconds=round(float(gen_stats.get("cpu_mcts_seconds", 0.0)), 2),
                cpu_record_play_seconds=round(float(gen_stats.get("cpu_record_play_seconds", 0.0)), 2),
                skipped_short_draw_games=int(gen_stats.get("skipped_short_draw_games", 0)),
                skipped_short_draw_samples=int(gen_stats.get("skipped_short_draw_samples", 0)),
                recorded_game_candidates=int(gen_stats.get("recorded_game_candidates", 0)),
            )
        msg = (
            f"gen {gen:3d} | samples {metric['samples']:6d} "
            f"| pol {metric['policy_loss']:.4f} val {metric['value_loss']:.4f} "
            f"| Hπ {metric['target_entropy']:.4f} maxπ {metric['target_max_prob']:.3f} "
            f"| {turns_per_sec:5.0f} turns/s {games_per_sec:4.1f} games/s "
            f"| gen {t_gen:5.1f}s train {t_train:4.1f}s"
        )

        t_eval = t_relative = t_record = 0.0
        games_to_save = list(sampled_games)
        if args.eval_every and (gen + 1) % args.eval_every == 0:
            log_phase(logger, "EVALUATING", f"gen={gen} games={args.eval_games} sims={args.sims}")
            t_phase = time.time()
            res = evaluate(
                net, device, games=args.eval_games, sims=args.sims, c_puct=args.c_puct,
                eval_batch_size=args.eval_batch_size, max_turns=args.max_turns
            )
            t_eval = time.time() - t_phase
            log_phase(
                logger,
                "EVALUATING",
                f"gen={gen} done seconds={t_eval:.1f} win_rate={res['win_rate']:.3f} "
                f"wins={res['wins']} losses={res['losses']} draws={res['draws']} "
                f"unfinished={res.get('unfinished', 0)}",
            )
            metric.update(
                win_rate=round(res["win_rate"], 3),
                wins=res["wins"],
                losses_count=res["losses"],
                draws=res["draws"],
                unfinished=res.get("unfinished", 0),
            )
            msg += f" | win_rate {res['win_rate']:.3f} ({res['wins']}/{res['losses']}/{res['draws']})"
            torch.save(net.state_dict(), ckpt_dir / "latest.pt")
            if res["win_rate"] > best_win:
                best_win = res["win_rate"]
                torch.save(net.state_dict(), ckpt_dir / "best.pt")
                msg += " *best*"

        # League: snapshot a past-self, then measure relative skill at the shallow
        # training depth. Seed an anchor immediately so progress is measured from here.
        if args.relative_every:
            if not league_ckpts() or (gen % args.league_every == 0):
                torch.save(net.state_dict(), league_dir / f"gen_{gen:06d}.pt")
                extra = league_ckpts()[1:]  # keep anchor (oldest) + most recent league_keep
                for old in extra[: max(0, len(extra) - args.league_keep)]:
                    old.unlink(missing_ok=True)
            if gen % args.relative_every == 0:
                log_phase(logger, "RELATIVE", f"gen={gen} games={args.relative_games} sims={args.sims}")
                t_phase = time.time()
                rel = relative_winrates(gen)
                t_relative = time.time() - t_phase
                rel_text = " ".join(f"{k}={v:.3f}" for k, v in rel.items()) if rel else "no_ckpts"
                log_phase(logger, "RELATIVE", f"gen={gen} done seconds={t_relative:.1f} {rel_text}")
                metric.update(rel)
                if rel:
                    msg += " | self " + " ".join(f"{k.split('_')[-1]}={v:.2f}" for k, v in rel.items())

        # Record replays for the dashboard's live game stream. Rust self-play samples
        # are cheap enough to capture every gen; Python recorder remains for baseline games.
        if args.record_games > 0 and args.record_every and (gen % args.record_every == 0):
            log_phase(
                logger,
                "RECORDING",
                f"gen={gen} games={args.record_games} per_opponent opponents=baseline,net",
            )
            t_phase = time.time()
            recorded_games = record_games(
                net, device, board=sp.board, n_games=args.record_games,
                depth=args.depth, tau=args.tau, iters=args.iters,
                eval_batch_size=args.eval_batch_size, max_turns=args.max_turns,
                opponent="baseline", seed=7000 + gen,
            )
            recorded_games += record_games(
                net, device, board=sp.board, n_games=args.record_games,
                depth=args.depth, tau=args.tau, iters=args.iters,
                eval_batch_size=args.eval_batch_size, max_turns=args.max_turns,
                opponent="net", seed=9000 + gen,
            )
            games_to_save.extend(recorded_games)
            t_record = time.time() - t_phase
            log_phase(logger, "RECORDING", f"gen={gen} done seconds={t_record:.1f} games={len(recorded_games)}")

        if games_to_save or selfplay_summary.get("completed_games"):
            run.save_games(gen, games_to_save, summary=selfplay_summary)
            run.prune_games(keep=args.keep_games)

        metric.update(
            eval_seconds=round(t_eval, 1),
            relative_seconds=round(t_relative, 1),
            record_seconds=round(t_record, 1),
        )
        log_phase(logger, "SAVING", f"gen={gen} checkpoint=state metrics=status")
        t_phase = time.time()
        save_state(gen)  # full resumable state, every generation (atomic write)
        metric["save_seconds"] = round(time.time() - t_phase, 1)
        run.append_metric(metric)
        metrics_history.append(metric)
        run.write_status(
            {
                "generation": gen,
                "running": gen < args.generations - 1,
                "total_generations": args.generations,
                "best_win_rate": None if best_win < 0 else round(best_win, 3),
                "last": metric,
            }
        )
        log_phase(
            logger,
            "METRICS",
            msg.replace(f"gen {gen:3d} | ", f"gen={gen} "),
        )
        retune(gen)

    run.write_status(
        {
            "generation": args.generations - 1,
            "running": False,
            "total_generations": args.generations,
            "best_win_rate": None if best_win < 0 else round(best_win, 3),
            "last": metric if "metric" in dir() else None,
        }
    )
    log_phase(logger, "DONE", f"generations={args.generations}")


if __name__ == "__main__":
    main()
