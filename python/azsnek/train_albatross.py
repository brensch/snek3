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
import copy
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import snek
import torch

from .net import AZNet, NetConfig, device_auto
from .runlog import RunWriter
from .selfplay import (
    ReplayBuffer,
    SelfPlayConfig,
    generate_proxy,
    generate_response,
)
from .train import (
    _setup_ort_env,
    export_onnx,
    log_phase,
    policy_target_stats,
    setup_logger,
    train_on_samples,
)


def build_args():
    ap = argparse.ArgumentParser(description="Albatross proxy/response training")
    ap.add_argument("--generations", type=int, default=0,
                    help="0 = run forever until stopped via the control API")
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
    ap.add_argument("--train-steps", type=int, default=128)
    ap.add_argument("--recency", type=float, default=2.0,
                    help="bias buffer sampling toward recent gens (1=uniform, >1=more recent)")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--buffer-size", type=int, default=500000)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=6,
                    help="games per opponent for the faithful eval (2 opponents, so "
                         "2x this many CPU games per eval); kept small as each game "
                         "is a full CPU search and the eval runs every --eval-every gens")
    ap.add_argument("--eval-workers", type=int, default=0,
                    help="parallel CPU games for the faithful eval (0 = half the "
                         "cores; raise to use more CPU, lower to use less)")
    ap.add_argument("--record-games", type=int, default=4,
                    help="self-play games to snapshot per phase (proxy+response) for replays; 0 disables")
    ap.add_argument("--record-every", type=int, default=1,
                    help="record replays every N generations")
    ap.add_argument("--keep-games", type=int, default=200,
                    help="keep replays for at most this many recent generations")
    ap.add_argument("--eval-bin", type=str, default=None,
                    help="path to the Rust snek-eval binary (default: auto-detect "
                         "crates/snek-server/target/{release,debug}/snek-eval)")
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    # Embedded live control + dashboard server.
    ap.add_argument("--serve", action=argparse.BooleanOptionalAction, default=True,
                    help="host the live API + dashboard in-process (--no-serve to disable)")
    ap.add_argument("--serve-host", type=str, default="0.0.0.0")
    ap.add_argument("--serve-port", type=int, default=8050)
    return ap.parse_args()


def _find_eval_bin(a):
    """Locate the Rust `snek-eval` binary (explicit --eval-bin, else the release
    then debug build dir). Returns a path string or None if not built."""
    if a.eval_bin:
        return a.eval_bin if Path(a.eval_bin).exists() else None
    base = Path(__file__).resolve().parents[2] / "crates" / "snek-server" / "target"
    for profile in ("release", "debug"):
        cand = base / profile / "snek-eval"
        if cand.exists():
            return str(cand)
    return None


def _spawn_eval(run, proxy, cfg, spcfg, device, gen, games, uct_iters, eval_bin, prev_proc, logger, workers=0):
    """Export the proxy to ONNX and launch `snek-eval` as a non-blocking, CPU-only
    subprocess that plays the *deployed* agent (proxy ONNX + serve_move search) vs
    the fixed pool and writes runs/<id>/eval/gen_XXXX.json. Skips (returning the
    previous handle) if a prior eval is still running or the binary is missing.
    Returns the live subprocess handle (or prev_proc when skipped)."""
    if eval_bin is None:
        return prev_proc
    if prev_proc is not None and prev_proc.poll() is None:
        log_phase(logger, "EVAL", f"gen={gen} skip: previous eval still running")
        return prev_proc

    # Export the current proxy weights (eval mode, restored after) atomically.
    onnx_path = run.eval_dir / "model.onnx"
    tmp_onnx = run.eval_dir / "model.onnx.tmp"
    was_training = proxy.training
    proxy.eval()
    with torch.no_grad():
        export_onnx(proxy, cfg.channels, spcfg.board, device, str(tmp_onnx))
    if was_training:
        proxy.train()
    os.replace(tmp_onnx, onnx_path)

    out_path = run.eval_artifact_path(gen)
    env = {**os.environ}  # inherits ORT_DYLIB_PATH/LD_LIBRARY_PATH from _setup_ort_env
    env["SNEK_CPU_ONLY"] = "1"  # don't steal the GPU the trainer is using
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update({
        "SNEK_EVAL_GEN": str(gen),
        "SNEK_EVAL_GAMES": str(games),
        "SNEK_EVAL_SEED": str(7000 + gen),
        "SNEK_UCT_ITERS": str(uct_iters),
        "SNEK_BOARD": str(spcfg.board),
        "SNEK_SNAKES": str(spcfg.num_snakes),
        "SNEK_DEPTH": str(spcfg.depth),
        "SNEK_ITERS": str(spcfg.iters),
        "SNEK_RESPONSE_TAU": str(spcfg.response_tau),
        "SNEK_DRAW_VALUE": str(spcfg.draw_value),
        "SNEK_EVAL_CHUNK": str(spcfg.eval_batch_size),
        # Give the eval its own turn cap (independent of self-play's max_turns, which
        # is often 0 = uncapped). Normal games end well under this; the cap only
        # bounds pathological circling games so one stuck game can't stall the whole
        # eval. Cap-reached games are excluded from the win-rate, so it stays honest.
        "SNEK_EVAL_MAXTURNS": str(spcfg.max_turns if spcfg.max_turns > 0 else 400),
    })
    # Cap CPU usage so the eval doesn't pin every core. Each worker uses a
    # single-intra-op-thread ONNX session, so the worker count bounds the load.
    # workers<=0 lets the Rust binary pick its default (half the cores).
    if workers and workers > 0:
        env["SNEK_EVAL_WORKERS"] = str(workers)
    logf = (run.eval_dir / "eval.log").open("a")
    proc = subprocess.Popen([eval_bin, str(onnx_path), str(out_path)],
                            env=env, stdout=logf, stderr=subprocess.STDOUT)
    run.prune_eval(keep=200)
    log_phase(logger, "EVAL", f"gen={gen} launched snek-eval ({games} games/opp, CPU) "
                              f"-> {out_path.name}")
    return proc


def train_one_run(a, state, device, logger) -> None:
    """Train a single run end to end. `a` is an args-like namespace fully
    describing the run; `state` is the live RunState (or None when --no-serve).
    Returns when the run finishes, is stopped, or a different run is requested
    (the caller inspects `state` to decide what to do next)."""
    spcfg = SelfPlayConfig(
        board=a.board, num_snakes=a.num_snakes, count=a.count,
        eval_batch_size=a.eval_batch_size, samples_per_gen=a.samples,
        max_turns=a.max_turns, exploration_prob=a.exploration_prob,
        depth=a.depth, iters=a.iters, tau_min=a.tau_min, tau_max=a.tau_max,
        response_tau=a.response_tau, draw_value=a.draw_value,
    )

    run = RunWriter(a.runs_dir, run_id=a.run_id, meta={
        "mode": "albatross", "board": a.board, "num_snakes": a.num_snakes,
        "filters": a.filters, "blocks": a.blocks, "depth": a.depth,
        "iters": a.iters, "tau_min": a.tau_min, "tau_max": a.tau_max,
        "response_tau": a.response_tau, "response_after": a.response_after,
        "samples_per_gen": a.samples, "count": a.count, "max_turns": a.max_turns,
        "buffer_size": a.buffer_size, "generations": a.generations,
    })
    log_phase(logger, "SETUP", f"run_id={run.run_id}")

    # Restore live params tuned via the dashboard in a previous session of this
    # run (persisted to params.json each gen). meta.json holds the original
    # config; params.json holds the latest tuned values and wins on resume.
    if state is not None and not a.fresh:
        pj = run.dir / "params.json"
        if pj.exists():
            from . import control
            try:
                saved = json.loads(pj.read_text())
            except (OSError, json.JSONDecodeError):
                saved = {}
            applied = {k: v for k, v in saved.items() if k in control.LIVE_PARAMS}
            for k, v in applied.items():
                setattr(a, k, v)
            if applied:
                log_phase(logger, "RESUME", f"restored tuned params: {applied}")

    cfg = NetConfig(channels=snek.CHANNELS, filters=a.filters, blocks=a.blocks,
                    temperature_input=True)
    proxy = AZNet(cfg).to(device)
    proxy_opt = torch.optim.Adam(proxy.parameters(), lr=a.lr, weight_decay=1e-4)
    response = AZNet(cfg).to(device)
    response_opt = torch.optim.Adam(response.parameters(), lr=a.lr, weight_decay=1e-4)

    start_gen = 0
    if run.has_state() and not a.fresh:
        st = torch.load(run.state_path, map_location=device, weights_only=False)
        proxy.load_state_dict(st["proxy"])
        proxy_opt.load_state_dict(st["proxy_opt"])
        if st.get("response") is not None:
            response.load_state_dict(st["response"])
            response_opt.load_state_dict(st["response_opt"])
        start_gen = st["gen"] + 1
        log_phase(logger, "RESUME", f"resumed at gen {start_gen}")
    elif a.fresh and run.has_state():
        run.reset()

    proxy_buf = ReplayBuffer(a.buffer_size)
    response_buf = ReplayBuffer(a.buffer_size)

    if state is not None:
        from . import control
        init_params = {k: getattr(a, k) for k in control.LIVE_PARAMS}
        history = []
        if run.metrics_path.exists():
            for ln in run.metrics_path.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        history.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
        state.begin_run(run.run_id, run.read_json("meta.json"), init_params, history,
                        persist=lambda p: run.write_json("params.json", p))

    run.write_status({"generation": start_gen - 1, "running": True,
                      "total_generations": a.generations or None})

    cuda = device.type == "cuda"
    gen = start_gen
    infinite = a.generations <= 0
    eval_bin = _find_eval_bin(a)
    eval_proc = None  # most recent async snek-eval subprocess (skip-if-busy)
    if eval_bin is None:
        log_phase(logger, "EVAL", "snek-eval binary not found; per-gen eval disabled "
                                  "(build it: make api-build)")
    while (infinite or gen < a.generations) and not (
            state and (state.stopping or state.pending_run or state.shutdown)):
        # pause gate: hold at the gen boundary while paused (server stays live).
        while state and state.paused and not (state.stopping or state.pending_run or state.shutdown):
            time.sleep(0.3)
        if state and (state.stopping or state.pending_run or state.shutdown):
            break
        # apply any live param updates for this generation.
        if state:
            p = state.params_snapshot()
            run.write_json("params.json", p)  # persist tuned params -> restored on resume
            spcfg = SelfPlayConfig(
                board=a.board, num_snakes=a.num_snakes, count=p["count"],
                eval_batch_size=a.eval_batch_size, samples_per_gen=p["samples"],
                max_turns=a.max_turns, exploration_prob=p["exploration_prob"],
                depth=a.depth, iters=p["iters"], tau_min=p["tau_min"], tau_max=p["tau_max"],
                response_tau=p["response_tau"], draw_value=p["draw_value"],
            )
            for opt in (proxy_opt, response_opt):
                for grp in opt.param_groups:
                    grp["lr"] = p["lr"]
            train_steps, batch_size, recency = p["train_steps"], p["batch_size"], p["recency"]
            eval_every, eval_games = p["eval_every"], p["eval_games"]
            record_games, record_every = p["record_games"], p["record_every"]
        else:
            train_steps, batch_size, recency = a.train_steps, a.batch_size, a.recency
            eval_every, eval_games = a.eval_every, a.eval_games
            record_games, record_every = a.record_games, a.record_every

        # Replays are sampled straight from this gen's self-play (no extra games);
        # `record` slots per phase, only on recording gens.
        do_record = record_games > 0 and record_every and gen % record_every == 0
        rec_n = record_games if do_record else 0

        t0 = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        # --- proxy self-play + train ---
        log_phase(logger, "PLAYING", f"gen={gen} proxy self-play (LE, target {spcfg.samples_per_gen:,} samples)")
        def prog(c, t, _g=gen):
            log_phase(logger, "PLAYING", f"gen={_g} proxy {c:,}/{t:,} samples")
            if state:
                state.set_progress("proxy self-play", c, t, _g)
        ps = generate_proxy(proxy, device, spcfg, seed=1000 + gen, progress_cb=prog, record=rec_n)
        proxy_buf.add(ps)
        log_phase(logger, "TRAINING", f"gen={gen} proxy steps={train_steps} buffer={len(proxy_buf):,}")
        if state:
            state.set_progress("training proxy", 0, 1, gen)
        pstats = train_on_samples(proxy, proxy_opt, proxy_buf.dataset(), device,
                                  steps=train_steps, batch_size=batch_size,
                                  recency=recency)
        ptgt = policy_target_stats(ps.pol)

        # --- response self-play + train (after warmup) ---
        train_response = gen >= a.response_after and a.num_snakes == 2
        rstats = {}
        rs = None
        if train_response:
            log_phase(logger, "PLAYING", f"gen={gen} response self-play (best-response vs proxy)")
            def rprog(c, t, _g=gen):
                log_phase(logger, "PLAYING", f"gen={_g} response {c:,}/{t:,} samples")
                if state:
                    state.set_progress("response self-play", c, t, _g)
            rs = generate_response(response, proxy, device, spcfg, seed=5000 + gen, progress_cb=rprog, record=rec_n)
            response_buf.add(rs)
            log_phase(logger, "TRAINING", f"gen={gen} response steps={train_steps}")
            rstats = train_on_samples(response, response_opt, response_buf.dataset(), device,
                                      steps=train_steps, batch_size=batch_size,
                                      recency=recency)

        gen_seconds = time.time() - t0

        proxy_mean_turns = ps.turns / max(ps.games, 1)
        # Real average length of games that actually FINISHED this gen (sum of
        # finished-game lengths / number finished). Unlike proxy_mean_turns
        # (total ongoing turn-activity / finished count) this is a true mean.
        proxy_game_len = ps.game_len_total / max(ps.games, 1)
        metric = {
            "gen": gen, "samples": int(ps.obs.shape[0]),
            "proxy_policy_loss": round(pstats["policy_loss"], 4),
            "proxy_value_loss": round(pstats["value_loss"], 4),
            "target_entropy": round(ptgt["target_entropy"], 4),
            "gen_seconds": round(gen_seconds, 1),
            "proxy_games": ps.games,
            "proxy_game_len": round(proxy_game_len, 1),
            # Issue indicators: draw rate and game-length spike on BOTH collapse
            # modes (short mutual-death OR long timeout); len_frac = share of the
            # turn cap (1.0 = games never resolve).
            "proxy_mean_turns": round(proxy_mean_turns, 1),
            "proxy_draw_rate": round(ps.draws / max(ps.games, 1), 3),
        }
        # len_frac is only meaningful with a turn cap; with no cap, report raw turns.
        if a.max_turns > 0:
            metric["proxy_len_frac"] = round(min(1.0, proxy_mean_turns / a.max_turns), 3)
        # Throughput / utilization (from proxy self-play timing).
        fwd, srch = ps.fwd_seconds, ps.search_seconds
        metric["inference_per_sec"] = round(ps.inferences / fwd) if fwd > 0 else 0
        metric["selfplay_gpu_pct"] = round(100 * fwd / (fwd + srch), 1) if (fwd + srch) > 0 else 0
        metric["samples_per_sec"] = round(int(ps.obs.shape[0]) / gen_seconds, 1) if gen_seconds > 0 else 0
        if rstats:
            metric["response_policy_loss"] = round(rstats["policy_loss"], 4)
            metric["response_value_loss"] = round(rstats["value_loss"], 4)
        if cuda:
            metric["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

        # --- self-play replays for the dashboard (sampled from the games above) ---
        if do_record:
            replays = (ps.replays or []) + (rs.replays if rs else [])
            if replays:
                run.save_games(gen, replays)
                run.prune_games(keep=a.keep_games)

        # --- async faithful eval (proxy ONNX + serve search vs the pool) ---
        # Runs out of band on idle CPU while the next gen trains on the GPU; the
        # Rust binary writes win-rates + real games to runs/<id>/eval/.
        if eval_every and gen % eval_every == 0:
            eval_proc = _spawn_eval(run, proxy, cfg, spcfg, device, gen,
                                    games=eval_games, uct_iters=a.uct_iters,
                                    eval_bin=eval_bin, prev_proc=eval_proc, logger=logger,
                                    workers=a.eval_workers)

        log_phase(logger, "GEN", " ".join(f"{k}={v}" for k, v in metric.items()))
        run.append_metric(metric)
        run.write_status({"generation": gen, "running": True,
                          "total_generations": a.generations or None, "last": metric})
        if state:
            state.add_metric(metric)
            state.set_status(generation=gen, running=True, paused=state.paused,
                             phase="paused" if state.paused else "idle", last=metric)
        run.save_state(lambda p: torch.save({
            "gen": gen, "net_cfg": asdict(cfg),
            "proxy": proxy.state_dict(), "proxy_opt": proxy_opt.state_dict(),
            "response": response.state_dict() if train_response else None,
            "response_opt": response_opt.state_dict() if train_response else None,
        }, p))
        gen += 1

    run.write_status({"generation": gen - 1, "running": False,
                      "total_generations": a.generations or None})
    log_phase(logger, "DONE", f"run {run.run_id} ended at gen {gen - 1}")


def _run_cfg(args, name: str, overrides: dict):
    """Build an args-like config for a dashboard-created run: base CLI args with
    the requested overrides, a fresh start, and run-forever unless overridden."""
    a = copy.copy(args)
    for k, v in (overrides or {}).items():
        setattr(a, k, v)
    a.run_id = name
    a.fresh = True
    a.generations = int((overrides or {}).get("generations", 0))
    return a


def _resume_cfg(args, name: str):
    """Build a config to RESUME an existing run from its checkpoint: take the
    run's own saved config (meta.json, so the net matches the weights) and resume
    (fresh=False). Live params (lr, train_steps, ...) default to the server's;
    they can be retuned live."""
    a = copy.copy(args)
    a.run_id = name
    a.fresh = False
    a.generations = 0
    try:
        meta = json.loads((Path(args.runs_dir) / name / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        meta = {}
    for attr, key in (("board", "board"), ("num_snakes", "num_snakes"),
                      ("filters", "filters"), ("blocks", "blocks"), ("depth", "depth"),
                      ("iters", "iters"), ("tau_min", "tau_min"), ("tau_max", "tau_max"),
                      ("response_tau", "response_tau"), ("response_after", "response_after"),
                      ("samples", "samples_per_gen"), ("count", "count"),
                      ("max_turns", "max_turns"), ("buffer_size", "buffer_size")):
        if key in meta:
            setattr(a, attr, meta[key])
    return a


def main():
    args = build_args()
    logger = setup_logger()
    device = device_auto()
    log_phase(logger, "SETUP", f"device={device}")
    cuda = device.type == "cuda"

    # --no-serve: headless single run.
    if not args.serve:
        train_one_run(args, None, device, logger)
        return

    from pathlib import Path as _Path
    from . import control

    state = control.RunState()
    state.set_base_spec({k: getattr(args, k) for k in control.NEW_RUN_PARAMS})
    static_dir = _Path(__file__).resolve().parent.parent / "dashboard" / "static"
    control.serve_in_thread(state, args.serve_host, args.serve_port,
                            _Path(args.runs_dir), static_dir)
    log_phase(logger, "SERVE", f"http://{args.serve_host}:{args.serve_port}")

    # Auto-start a run iff a run id was passed on the CLI; otherwise idle.
    pending = {"name": args.run_id, "overrides": {}, "cli": True} if args.run_id else None

    while not state.shutdown:
        if pending is None:
            state.go_idle()
            log_phase(logger, "IDLE", "server up — start a run from the dashboard")
            while not state.pending_run and not state.shutdown:
                time.sleep(0.3)
            if state.shutdown:
                break
            pending = state.take_new_run()
            if pending is None:
                continue

        if pending.get("cli"):
            a = args  # resume/start exactly as launched
        elif pending.get("resume"):
            a = _resume_cfg(args, pending["name"])
            log_phase(logger, "RESUME", f"{a.run_id} from checkpoint")
        else:
            a = _run_cfg(args, pending["name"], pending["overrides"])
            log_phase(logger, "NEWRUN", f"{a.run_id} overrides={pending['overrides']}")
        train_one_run(a, state, device, logger)

        # Did the run end because a different one was requested? If so, loop into
        # it; otherwise (stop / natural end) drop back to idle.
        pending = state.take_new_run()
        if cuda:
            torch.cuda.empty_cache()

    state.go_idle()
    log_phase(logger, "DONE", "shutdown")


if __name__ == "__main__":
    main()
