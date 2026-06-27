"""Training loop: alternate self-play data generation and supervised updates to
the policy (cross-entropy to the search policy) and value (MSE to game outcome).

Usage:
    python -m azsnek.train --generations 50 --samples 20000
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
    obs = torch.from_numpy(samples.obs).to(device)
    pol = torch.from_numpy(samples.pol).to(device)
    z = torch.from_numpy(samples.z).to(device)
    n = obs.shape[0]

    net.train()
    pl = vl = 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        with net_autocast(device):
            logits, value = net(obs[idx])
            logp = F.log_softmax(logits, dim=1)
            # Soft-target cross-entropy; illegal moves have target 0.
            policy_loss = -(pol[idx] * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, z[idx])
            loss = policy_loss + value_weight * value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pl += float(policy_loss.item())
        vl += float(value_loss.item())
    return {"policy_loss": pl / steps, "value_loss": vl / steps}


def policy_target_stats(pol: np.ndarray) -> dict:
    """Information content of the search policy targets."""
    p = np.clip(pol, 1e-9, 1.0)
    entropy = -(pol * np.log(p)).sum(axis=1)
    return {
        "target_entropy": float(entropy.mean()),
        "target_max_prob": float(pol.max(axis=1).mean()),
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
    ap.add_argument("--exploration-prob", type=float, default=0.25, help="uniform-legal mix into the played action")
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
    print(f"device: {device}")
    if args.search_threads:
        status = "configured" if search_threads_configured else "already initialized"
        print(f"search threads: {args.search_threads} ({status})", flush=True)

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
            "search_threads": args.search_threads,
            "generations": args.generations,
            "samples_per_gen": args.samples,
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
    print(f"serving checkpoints: {ckpt_dir}", flush=True)
    run.write_json("meta.json", {**run.read_json("meta.json"), "ckpt_dir": str(ckpt_dir)})

    # Resume automatically when this run-id has saved state, unless --fresh.
    resume = None
    if run.has_state() and not args.fresh:
        resume = torch.load(run.state_path, map_location=device, weights_only=False)
        cfg = NetConfig(**resume["net_cfg"])
    else:
        if args.fresh and run.has_state():
            run.reset()
            print(f"--fresh: cleared previous progress in {run.dir}", flush=True)
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
            print(f"warning: could not fully restore RNG state: {e}")
        print(f"resumed run {run.run_id} at generation {start_gen}", flush=True)
    else:
        print(f"run dir: {run.dir} (fresh start)", flush=True)

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
        print(
            "adaptive tune | "
            f"samples {args.samples} train_steps {args.train_steps} "
            f"eval_games {args.eval_games} tau {args.tau} | "
            + "; ".join(reasons),
            flush=True,
        )

    onnx_path = run.dir / "model.onnx"
    # Persistent Rust self-play: games carry across generations, so a large
    # `count` (big GPU batches) wastes no inference on unfinished games.
    selfplay = snek.SelfPlay(board=sp.board, num_snakes=sp.num_snakes, count=args.count, seed=12345)
    for gen in range(start_gen, args.generations):
        # ---- GENERATE: Rust MCTS + ONNX/CUDA inference (no Python round-trips) ----
        print(
            f"gen {gen:3d} | GENERATING  {args.count} games x {args.sims} sims "
            f"-> {args.samples} samples ...",
            flush=True,
        )
        t0 = time.time()
        export_onnx(net, cfg.channels, sp.board, device, onnx_path)
        t_export = time.time() - t0
        obs, pol, z = snek.generate_selfplay(
            str(onnx_path), board=sp.board, num_snakes=sp.num_snakes,
            count=args.count, sims=args.sims, c_puct=args.c_puct,
            samples_per_gen=args.samples, seed=1000 + gen,
            exploration_prob=args.exploration_prob, max_turns=args.max_turns,
        )
        samples = Samples(obs=obs, pol=pol, z=z, turns=int(z.shape[0]), games=0)
        target_stats = policy_target_stats(samples.pol)
        buffer.add(samples)
        t_gen = time.time() - t0
        n_samp = samples.obs.shape[0]
        print(
            f"gen {gen:3d} |   generated {n_samp:6d} samples in {t_gen:5.1f}s = "
            f"{n_samp / max(t_gen, 1e-9):6.0f} samples/s  (onnx export {t_export:.1f}s)",
            flush=True,
        )

        # ---- TRAIN: PyTorch SGD on a window of recent games ----
        print(
            f"gen {gen:3d} | TRAINING    {args.train_steps} steps x batch "
            f"{args.batch_size}  (buffer {len(buffer)}) ...",
            flush=True,
        )
        t1 = time.time()
        losses = train_on_samples(
            net, opt, buffer.dataset(), device,
            steps=args.train_steps, batch_size=args.batch_size,
        )
        t_train = time.time() - t1
        print(
            f"gen {gen:3d} |   trained {args.train_steps} steps in {t_train:5.1f}s = "
            f"{args.train_steps / max(t_train, 1e-9):5.1f} steps/s | "
            f"pol {losses['policy_loss']:.4f} val {losses['value_loss']:.4f}",
            flush=True,
        )

        turns_per_sec = samples.turns / max(t_gen, 1e-9)
        games_per_sec = samples.games / max(t_gen, 1e-9)
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
        }
        msg = (
            f"gen {gen:3d} | samples {metric['samples']:6d} "
            f"| pol {metric['policy_loss']:.4f} val {metric['value_loss']:.4f} "
            f"| Hπ {metric['target_entropy']:.4f} maxπ {metric['target_max_prob']:.3f} "
            f"| {turns_per_sec:5.0f} turns/s {games_per_sec:4.1f} games/s "
            f"| gen {t_gen:5.1f}s train {t_train:4.1f}s"
        )

        if args.eval_every and (gen + 1) % args.eval_every == 0:
            res = evaluate(
                net, device, games=args.eval_games, sims=args.sims, c_puct=args.c_puct,
                eval_batch_size=args.eval_batch_size, max_turns=args.max_turns
            )
            metric.update(
                win_rate=round(res["win_rate"], 3),
                wins=res["wins"],
                losses_count=res["losses"],
                draws=res["draws"],
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
                rel = relative_winrates(gen)
                metric.update(rel)
                if rel:
                    msg += " | self " + " ".join(f"{k.split('_')[-1]}={v:.2f}" for k, v in rel.items())

        # Record replays for the dashboard's live game stream (every gen by default).
        if args.record_games > 0 and args.record_every and (gen % args.record_every == 0):
            games = record_games(
                net, device, board=sp.board, n_games=args.record_games,
                depth=args.depth, tau=args.tau, iters=args.iters,
                eval_batch_size=args.eval_batch_size, max_turns=args.max_turns,
                opponent="baseline", seed=7000 + gen,
            )
            games += record_games(
                net, device, board=sp.board, n_games=args.record_games,
                depth=args.depth, tau=args.tau, iters=args.iters,
                eval_batch_size=args.eval_batch_size, max_turns=args.max_turns,
                opponent="net", seed=9000 + gen,
            )
            run.save_games(gen, games)
            run.prune_games(keep=args.keep_games)

        run.append_metric(metric)
        metrics_history.append(metric)
        save_state(gen)  # full resumable state, every generation (atomic write)
        run.write_status(
            {
                "generation": gen,
                "running": gen < args.generations - 1,
                "total_generations": args.generations,
                "best_win_rate": None if best_win < 0 else round(best_win, 3),
                "last": metric,
            }
        )
        print(msg, flush=True)
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


if __name__ == "__main__":
    main()
