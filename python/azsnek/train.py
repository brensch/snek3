"""Training loop: alternate self-play data generation and supervised updates to
the policy (cross-entropy to the search policy) and value (MSE to game outcome).

Usage:
    python -m azsnek.train --generations 50 --samples 20000
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import snek
import torch
import torch.nn.functional as F

from .evaluate import evaluate
from .net import AZNet, NetConfig, autocast as net_autocast, device_auto
from .recorder import record_games
from .runlog import RunWriter
from .selfplay import ReplayBuffer, SelfPlayConfig, generate


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--count", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument(
        "--search-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Rayon threads for Rust search/encoding (default: all visible CPUs; 0 leaves Rayon default)",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-steps", type=int, default=256, help="SGD steps per generation")
    ap.add_argument("--buffer-size", type=int, default=150_000, help="replay buffer capacity (samples)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--runs-dir", type=str, default="runs", help="dashboard run root")
    ap.add_argument("--run-id", type=str, default=None, help="run dir name (default: timestamp)")
    ap.add_argument("--record-games", type=int, default=8, help="replays per opponent per recording")
    ap.add_argument("--record-every", type=int, default=1, help="record replays every N generations")
    ap.add_argument("--keep-games", type=int, default=40, help="keep this many recent game files")
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
        depth=args.depth,
        tau=args.tau,
        iters=args.iters,
        samples_per_gen=args.samples,
    )
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            "search_threads": args.search_threads,
            "generations": args.generations,
            "samples_per_gen": args.samples,
            "device": str(device),
        },
    )

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

    buffer = ReplayBuffer(args.buffer_size)

    for gen in range(start_gen, args.generations):
        t0 = time.time()
        samples = generate(net, device, sp, seed=1000 + gen)
        buffer.add(samples)
        t_gen = time.time() - t0

        t1 = time.time()
        # Train on a window of recent games, not just this generation's samples.
        losses = train_on_samples(net, opt, buffer.dataset(), device, steps=args.train_steps)
        t_train = time.time() - t1

        turns_per_sec = samples.turns / max(t_gen, 1e-9)
        games_per_sec = samples.games / max(t_gen, 1e-9)
        metric = {
            "gen": gen,
            "samples": int(samples.obs.shape[0]),
            "buffer": len(buffer),
            "policy_loss": round(losses["policy_loss"], 4),
            "value_loss": round(losses["value_loss"], 4),
            "gen_seconds": round(t_gen, 1),
            "train_seconds": round(t_train, 1),
            "turns_per_sec": round(turns_per_sec, 0),
            "games_per_sec": round(games_per_sec, 2),
            "win_rate": None,
        }
        msg = (
            f"gen {gen:3d} | samples {metric['samples']:6d} "
            f"| pol {metric['policy_loss']:.4f} val {metric['value_loss']:.4f} "
            f"| {turns_per_sec:5.0f} turns/s {games_per_sec:4.1f} games/s "
            f"| gen {t_gen:5.1f}s train {t_train:4.1f}s"
        )

        if args.eval_every and (gen + 1) % args.eval_every == 0:
            res = evaluate(
                net, device, games=args.eval_games, depth=args.depth, tau=args.tau, iters=args.iters
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

        # Record replays for the dashboard's live game stream (every gen by default).
        if args.record_games > 0 and args.record_every and (gen % args.record_every == 0):
            games = record_games(
                net, device, board=sp.board, n_games=args.record_games,
                depth=args.depth, tau=args.tau, iters=args.iters,
                opponent="baseline", seed=7000 + gen,
            )
            games += record_games(
                net, device, board=sp.board, n_games=args.record_games,
                depth=args.depth, tau=args.tau, iters=args.iters,
                opponent="net", seed=9000 + gen,
            )
            run.save_games(gen, games)
            run.prune_games(keep=args.keep_games)

        run.append_metric(metric)
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
