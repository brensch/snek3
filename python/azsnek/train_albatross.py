"""Albatross training: temperature-conditioned PROXY + best-response RESPONSE.

Self-contained loop (kept separate from the MCTS-bootstrap `train.py`):

* PROXY  pi_P(o, tau): a temperature-conditioned net trained on logit-equilibrium
  targets across a temperature distribution (see selfplay.generate_proxy).
* RESPONSE pi_R(o, tau_opp): after a warmup, a second net trained to best-respond
  (at tau_R) to weak opponents at tau_opp, using the frozen proxy's leaf values
  (see selfplay.generate_response).

At test time the opponent's temperature is estimated by MLE under the proxy and
fed to the response net so it exploits the specific opponent.

Run with `python -m azsnek.train_albatross --generations ... [--fresh] [--run-id ...]`.
"""

from __future__ import annotations

import os

# Must be set before torch initializes CUDA: expandable segments cut allocator
# fragmentation and avoid spilling into (slow) shared GPU memory on WSL2.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import time
from dataclasses import asdict

import numpy as np
import snek
import torch

from .net import AZNet, NetConfig, device_auto
from .runlog import RunWriter
from .search import greedy_actions, run_search, run_search_hetero
from .selfplay import (
    ReplayBuffer,
    SelfPlayConfig,
    generate_proxy,
    generate_response,
)
from .train import _setup_ort_env, log_phase, policy_target_stats, setup_logger, train_on_samples

TAU_GRID = np.geomspace(0.25, 20.0, 24).astype(np.float32)
SAFETY_TURNS = 5000  # hang-guard for uncapped (max_turns=0) eval/recording loops


@torch.no_grad()
def estimate_opponent_tau(proxy, device, obs_hist, act_hist) -> float:
    """MLE of an opponent's temperature under the proxy: the tau maximizing the
    log-likelihood of its observed actions. `obs_hist` [T, C, H, W], `act_hist`
    [T] move indices. Returns the best tau from TAU_GRID. (For single-opponent
    online use, e.g. serving; the batched eval uses a representative tau.)"""
    if len(act_hist) == 0:
        return float(np.sqrt(TAU_GRID[0] * TAU_GRID[-1]))
    obs_t = torch.from_numpy(np.asarray(obs_hist, dtype=np.float32)).to(device)
    acts = np.asarray(act_hist, dtype=np.int64)
    best_tau, best_ll = float(TAU_GRID[0]), -np.inf
    for tau in TAU_GRID:
        temp = torch.full((obs_t.shape[0],), float(tau), device=device)
        logits, _ = proxy(obs_t, temp)
        logp = torch.log_softmax(logits.float(), dim=1).cpu().numpy()
        ll = float(logp[np.arange(len(acts)), acts].sum())
        if ll > best_ll:
            best_ll, best_tau = ll, float(tau)
    return best_tau


@torch.no_grad()
def _winrate_vs(agent_fn, opp_fn, board, num_snakes, games, seed, max_turns) -> float:
    """Snake 0 = our agent (`agent_fn(batch) -> [count] move idx`), snake 1 =
    pool opponent (`opp_fn(batch) -> [count] move idx`). Win rate (draws=half)."""
    batch = snek.GameBatch(board, board, num_snakes, count=games, seed=seed)
    cap = max_turns if max_turns > 0 else SAFETY_TURNS  # hang-guard when uncapped
    steps = 0
    while not np.all(batch.done()) and steps < cap:
        agent = agent_fn(batch)
        opp = opp_fn(batch)
        actions = np.stack([agent, opp], axis=1).astype(np.uint8)
        batch.step(actions)
        steps += 1
    winners = batch.winners()
    done = batch.done().astype(bool)
    wins = int(np.sum(winners == 0))
    draws = int(np.sum(done & (winners == -1)))
    decided = int(np.sum(done))
    return (wins + 0.5 * draws) / decided if decided else 0.0


@torch.no_grad()
def evaluate_albatross(proxy, response, device, cfg: SelfPlayConfig, games, seed,
                       eval_opp_tau, uct_iters):
    """Win rate of the proxy (near-optimal tau) and the response (best-responding
    at tau_R to an assumed-weak opponent at `eval_opp_tau`) against the opponent
    pool: the 1-ply flood-fill baseline and the CPU UCT agent (stronger)."""
    def proxy_action(batch):
        pol, _ = run_search(batch, proxy, device, cfg.depth, cfg.tau_max, cfg.iters,
                            cfg.eval_batch_size, return_root_values=True, temp=cfg.tau_max,
                            draw_value=cfg.draw_value)
        return greedy_actions(pol)[:, 0]

    def response_action(batch):
        # Leaf values from proxy; for serving, use the response net + per-opponent
        # MLE tau (estimate_opponent_tau) instead of a fixed eval_opp_tau.
        pol, _ = run_search_hetero(batch, proxy, device, cfg.depth,
                                   [cfg.response_tau, eval_opp_tau], cfg.iters,
                                   cfg.eval_batch_size, temp=eval_opp_tau,
                                   draw_value=cfg.draw_value)
        return greedy_actions(pol)[:, 0]

    # Opponent pool: name -> action fn for snake 1.
    pool = {
        "baseline": lambda b: b.baseline_actions()[:, 1],
        "uct": lambda b: np.asarray(b.heuristic_actions(iters=uct_iters, seed=seed))[:, 1],
    }
    agents = {"proxy": proxy_action}
    if response is not None:
        agents["response"] = response_action

    out = {}
    for aname, afn in agents.items():
        for oname, ofn in pool.items():
            out[f"{aname}_vs_{oname}"] = _winrate_vs(
                afn, ofn, cfg.board, cfg.num_snakes, games, seed, cfg.max_turns)
    return out


@torch.no_grad()
def record_albatross_games(proxy, response, device, cfg: SelfPlayConfig, n_games, seed, uct_iters):
    """Record replay games for the dashboard, snake 0 = first named player.
    Matchups (each a joint move fn -> [count, 2] actions): proxy vs {proxy,
    baseline, uct} and, when the response exists, response vs {proxy, baseline,
    uct}. proxy plays the LE search (near-optimal tau); response plays the
    heterogeneous-tau best-response; baseline/uct are the fixed CPU opponents.
    Returns a list of {opponent, winner, num_turns, frames} dicts."""
    def proxy_pol(batch):  # [count, 2] greedy moves for both snakes from the proxy LE
        pol, _ = run_search(batch, proxy, device, cfg.depth, cfg.tau_max, cfg.iters,
                            cfg.eval_batch_size, return_root_values=True, temp=cfg.tau_max,
                            draw_value=cfg.draw_value)
        return greedy_actions(pol)

    def response_a0(batch):  # responder (snake 0) best-response move
        pol, _ = run_search_hetero(batch, proxy, device, cfg.depth,
                                   [cfg.response_tau, cfg.tau_min], cfg.iters,
                                   cfg.eval_batch_size, temp=cfg.tau_min,
                                   draw_value=cfg.draw_value)
        return greedy_actions(pol)[:, 0]

    baseline_a1 = lambda b: b.baseline_actions()[:, 1]
    uct_a1 = lambda b: np.asarray(b.heuristic_actions(iters=uct_iters, seed=seed))[:, 1]
    j = lambda a0, a1: np.stack([a0, a1], axis=1).astype(np.uint8)

    matchups = [
        ("proxy-v-proxy", lambda b: proxy_pol(b).astype(np.uint8)),
        ("proxy-v-baseline", lambda b: j(proxy_pol(b)[:, 0], baseline_a1(b))),
        ("proxy-v-uct", lambda b: j(proxy_pol(b)[:, 0], uct_a1(b))),
    ]
    if response is not None:
        matchups += [
            ("response-v-proxy", lambda b: j(response_a0(b), proxy_pol(b)[:, 1])),
            ("response-v-baseline", lambda b: j(response_a0(b), baseline_a1(b))),
            ("response-v-uct", lambda b: j(response_a0(b), uct_a1(b))),
        ]

    games = []
    for label, joint_fn in matchups:
        batch = snek.GameBatch(cfg.board, cfg.board, cfg.num_snakes, count=n_games, seed=seed)
        frames = [[] for _ in range(n_games)]
        term = [False] * n_games
        steps = 0
        cap = cfg.max_turns if cfg.max_turns > 0 else SAFETY_TURNS
        while steps < cap:
            done = batch.done().astype(bool)
            for g in range(n_games):
                if not term[g]:
                    frames[g].append(json.loads(batch.snapshot(g)))
                    if done[g]:
                        term[g] = True
            if all(term):
                break
            batch.step(joint_fn(batch))
            steps += 1
        winners = batch.winners()
        for g in range(n_games):
            games.append({
                "opponent": label,
                "winner": int(winners[g]),
                "num_turns": len(frames[g]),
                "frames": frames[g],
            })
    return games


def build_args():
    ap = argparse.ArgumentParser(description="Albatross proxy/response training")
    ap.add_argument("--generations", type=int, default=100000)
    ap.add_argument("--board", type=int, default=11)
    ap.add_argument("--num-snakes", type=int, default=2)
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--tau-min", type=float, default=0.5)
    ap.add_argument("--tau-max", type=float, default=10.0)
    ap.add_argument("--response-tau", type=float, default=12.0)
    ap.add_argument("--draw-value", type=float, default=-0.9,
                    help="terminal value of a draw in the equilibrium search; negative kills mutual-suicide draws")
    ap.add_argument("--response-after", type=int, default=30,
                    help="start training the response net after this many proxy generations")
    ap.add_argument("--eval-opp-tau", type=float, default=1.0,
                    help="assumed opponent temperature for response eval vs baseline")
    ap.add_argument("--uct-iters", type=int, default=200,
                    help="UCB simulations for the CPU UCT pool opponent in eval")
    ap.add_argument("--exploration-prob", type=float, default=0.15)
    ap.add_argument("--max-turns", type=int, default=0,
                    help="0 = play until a snake dies (no cap); positive caps games")
    # Egocentric obs are 21x21 (3.6x the cells of 11x11), so conv activations are
    # ~3.6x larger; keep the leaf-eval batch modest to stay within dedicated VRAM.
    ap.add_argument("--eval-batch-size", type=int, default=2048)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--buffer-size", type=int, default=500000)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument("--record-games", type=int, default=2,
                    help="replay games to record per (agent,opponent) matchup; 0 disables")
    ap.add_argument("--record-every", type=int, default=1,
                    help="record replays every N generations")
    ap.add_argument("--keep-games", type=int, default=200,
                    help="keep replays for at most this many recent generations")
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    return ap.parse_args()


def main():
    args = build_args()
    logger = setup_logger()
    device = device_auto()
    log_phase(logger, "SETUP", f"device={device}")

    spcfg = SelfPlayConfig(
        board=args.board, num_snakes=args.num_snakes, count=args.count,
        eval_batch_size=args.eval_batch_size, samples_per_gen=args.samples,
        max_turns=args.max_turns, exploration_prob=args.exploration_prob,
        depth=args.depth, iters=args.iters, tau_min=args.tau_min, tau_max=args.tau_max,
        response_tau=args.response_tau, draw_value=args.draw_value,
    )

    run = RunWriter(args.runs_dir, run_id=args.run_id, meta={
        "mode": "albatross", "board": args.board, "num_snakes": args.num_snakes,
        "filters": args.filters, "blocks": args.blocks, "depth": args.depth,
        "iters": args.iters, "tau_min": args.tau_min, "tau_max": args.tau_max,
        "response_tau": args.response_tau, "response_after": args.response_after,
        "samples_per_gen": args.samples, "count": args.count, "max_turns": args.max_turns,
        "buffer_size": args.buffer_size, "generations": args.generations,
    })
    log_phase(logger, "SETUP", f"run_id={run.run_id}")

    cfg = NetConfig(channels=snek.CHANNELS, filters=args.filters, blocks=args.blocks,
                    temperature_input=True)
    proxy = AZNet(cfg).to(device)
    proxy_opt = torch.optim.Adam(proxy.parameters(), lr=args.lr, weight_decay=1e-4)
    response = AZNet(cfg).to(device)
    response_opt = torch.optim.Adam(response.parameters(), lr=args.lr, weight_decay=1e-4)

    start_gen = 0
    if run.has_state() and not args.fresh:
        st = torch.load(run.state_path, map_location=device, weights_only=False)
        proxy.load_state_dict(st["proxy"])
        proxy_opt.load_state_dict(st["proxy_opt"])
        if st.get("response") is not None:
            response.load_state_dict(st["response"])
            response_opt.load_state_dict(st["response_opt"])
        start_gen = st["gen"] + 1
        log_phase(logger, "RESUME", f"resumed at gen {start_gen}")
    elif args.fresh and run.has_state():
        run.reset()

    proxy_buf = ReplayBuffer(args.buffer_size)
    response_buf = ReplayBuffer(args.buffer_size)
    run.write_status({"generation": start_gen - 1, "running": True,
                      "total_generations": args.generations})

    cuda = device.type == "cuda"
    for gen in range(start_gen, args.generations):
        t0 = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        # --- proxy self-play + train ---
        log_phase(logger, "PLAYING", f"gen={gen} proxy self-play (LE, target {args.samples:,} samples)")
        prog = lambda c, t: log_phase(logger, "PLAYING", f"gen={gen} proxy {c:,}/{t:,} samples")
        ps = generate_proxy(proxy, device, spcfg, seed=1000 + gen, progress_cb=prog)
        proxy_buf.add(ps)
        log_phase(logger, "TRAINING", f"gen={gen} proxy steps={args.train_steps} buffer={len(proxy_buf):,}")
        pstats = train_on_samples(proxy, proxy_opt, proxy_buf.dataset(), device,
                                  steps=args.train_steps, batch_size=args.batch_size)
        ptgt = policy_target_stats(ps.pol)

        # --- response self-play + train (after warmup) ---
        train_response = gen >= args.response_after and args.num_snakes == 2
        rstats = {}
        if train_response:
            log_phase(logger, "PLAYING", f"gen={gen} response self-play (best-response vs proxy)")
            rprog = lambda c, t: log_phase(logger, "PLAYING", f"gen={gen} response {c:,}/{t:,} samples")
            rs = generate_response(response, proxy, device, spcfg, seed=5000 + gen, progress_cb=rprog)
            response_buf.add(rs)
            log_phase(logger, "TRAINING", f"gen={gen} response steps={args.train_steps}")
            rstats = train_on_samples(response, response_opt, response_buf.dataset(), device,
                                      steps=args.train_steps, batch_size=args.batch_size)

        gen_seconds = time.time() - t0

        proxy_mean_turns = ps.turns / max(ps.games, 1)
        metric = {
            "gen": gen, "samples": int(ps.obs.shape[0]),
            "proxy_policy_loss": round(pstats["policy_loss"], 4),
            "proxy_value_loss": round(pstats["value_loss"], 4),
            "target_entropy": round(ptgt["target_entropy"], 4),
            "gen_seconds": round(gen_seconds, 1),
            "proxy_games": ps.games,
            # Issue indicators: draw rate and game-length spike on BOTH collapse
            # modes (short mutual-death OR long timeout); len_frac = share of the
            # turn cap (1.0 = games never resolve).
            "proxy_mean_turns": round(proxy_mean_turns, 1),
            "proxy_draw_rate": round(ps.draws / max(ps.games, 1), 3),
        }
        # len_frac is only meaningful with a turn cap; with no cap, report raw turns.
        if args.max_turns > 0:
            metric["proxy_len_frac"] = round(min(1.0, proxy_mean_turns / args.max_turns), 3)
        if rstats:
            metric["response_policy_loss"] = round(rstats["policy_loss"], 4)
            metric["response_value_loss"] = round(rstats["value_loss"], 4)
        if cuda:
            metric["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

        # --- eval ---
        if args.eval_every and gen % args.eval_every == 0:
            log_phase(logger, "EVALUATING", f"gen={gen} vs pool (baseline, UCT) games={args.eval_games}")
            ev = evaluate_albatross(
                proxy, response if train_response else None, device, spcfg,
                games=args.eval_games, seed=7000 + gen, eval_opp_tau=args.eval_opp_tau,
                uct_iters=args.uct_iters)
            metric.update({k: round(v, 4) for k, v in ev.items()})

        # --- record replays for the dashboard ---
        if args.record_games > 0 and args.record_every and gen % args.record_every == 0:
            log_phase(logger, "RECORDING", f"gen={gen} games={args.record_games}/matchup")
            recorded = record_albatross_games(
                proxy, response if train_response else None, device, spcfg,
                n_games=args.record_games, seed=9000 + gen, uct_iters=args.uct_iters)
            run.save_games(gen, recorded)
            run.prune_games(keep=args.keep_games)

        log_phase(logger, "GEN", " ".join(f"{k}={v}" for k, v in metric.items()))
        run.append_metric(metric)
        run.write_status({"generation": gen, "running": True,
                          "total_generations": args.generations, "last": metric})
        run.save_state(lambda p: torch.save({
            "gen": gen, "net_cfg": asdict(cfg),
            "proxy": proxy.state_dict(), "proxy_opt": proxy_opt.state_dict(),
            "response": response.state_dict() if train_response else None,
            "response_opt": response_opt.state_dict() if train_response else None,
        }, p))

    run.write_status({"generation": args.generations - 1, "running": False,
                      "total_generations": args.generations})


if __name__ == "__main__":
    main()
