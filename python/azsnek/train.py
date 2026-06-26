"""Training loop: alternate self-play data generation and supervised updates to
the policy (cross-entropy to the search policy) and value (MSE to game outcome).

Usage:
    python -m azsnek.train --generations 50 --samples 20000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import snek
import torch
import torch.nn.functional as F

from .evaluate import evaluate
from .net import AZNet, NetConfig, device_auto
from .selfplay import SelfPlayConfig, generate


def train_on_samples(
    net: AZNet,
    opt: torch.optim.Optimizer,
    samples,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 1024,
    value_weight: float = 1.0,
) -> dict:
    obs = torch.from_numpy(samples.obs).to(device)
    pol = torch.from_numpy(samples.pol).to(device)
    z = torch.from_numpy(samples.z).to(device)
    n = obs.shape[0]

    net.train()
    last = {"policy_loss": 0.0, "value_loss": 0.0}
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits, value = net(obs[idx])
            logp = F.log_softmax(logits, dim=1)
            # Soft-target cross-entropy; illegal moves have target 0 (no penalty).
            policy_loss = -(pol[idx] * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, z[idx])
            loss = policy_loss + value_weight * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
            }
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--count", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--tau", type=float, default=6.0)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    args = ap.parse_args()

    device = device_auto()
    print(f"device: {device}")
    cfg = NetConfig(channels=snek.CHANNELS, filters=args.filters, blocks=args.blocks)
    net = AZNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    sp = SelfPlayConfig(
        count=args.count,
        depth=args.depth,
        tau=args.tau,
        iters=args.iters,
        samples_per_gen=args.samples,
    )
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_win = -1.0

    for gen in range(args.generations):
        t0 = time.time()
        samples = generate(net, device, sp, seed=1000 + gen)
        t_gen = time.time() - t0

        t1 = time.time()
        losses = train_on_samples(net, opt, samples, device, epochs=args.epochs)
        t_train = time.time() - t1

        msg = (
            f"gen {gen:3d} | samples {samples.obs.shape[0]:6d} "
            f"| pol {losses['policy_loss']:.4f} val {losses['value_loss']:.4f} "
            f"| gen {t_gen:5.1f}s train {t_train:4.1f}s"
        )

        if args.eval_every and (gen + 1) % args.eval_every == 0:
            res = evaluate(
                net, device, games=args.eval_games, depth=args.depth, tau=args.tau, iters=args.iters
            )
            msg += f" | vs baseline win_rate {res['win_rate']:.3f} ({res['wins']}/{res['losses']}/{res['draws']})"
            torch.save(net.state_dict(), ckpt_dir / "latest.pt")
            if res["win_rate"] > best_win:
                best_win = res["win_rate"]
                torch.save(net.state_dict(), ckpt_dir / "best.pt")
                msg += " *best*"

        print(msg, flush=True)


if __name__ == "__main__":
    main()
