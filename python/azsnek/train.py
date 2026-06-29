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
import shutil
import signal
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

import copy
import itertools

from . import control
from .net import AZNet, NetConfig, autocast as net_autocast, device_auto
from .runlog import RunWriter
from .symmetry import augment_batch
from .selfplay import ReplayBuffer, Samples, SelfPlayConfig, generate, save_shard, prune_shards


DEFAULT_PARAMS_PATH = Path(__file__).with_name("default_params.json")
DEFAULT_SERVING_CKPT_DIR = Path("checkpoints")
META_ARG_ALIASES = {
    "samples_per_gen": "samples",
}
NON_PARAM_KEYS = {"run_id", "ckpt_dir", "device"}


def load_default_params() -> dict:
    return json.loads(DEFAULT_PARAMS_PATH.read_text())


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _coerce_param_value(key: str, value):
    if isinstance(value, bool):
        return value
    if key in {"lr", "c_puct", "exploration_prob", "draw_value"}:
        return float(value)
    if key == "arch":
        return str(value)
    return int(value)


def apply_params_to_args(args, params: dict) -> None:
    for key, value in params.items():
        if key in NON_PARAM_KEYS or not hasattr(args, key) or value is None:
            continue
        setattr(args, key, _coerce_param_value(key, value))


def run_config_params(runs_dir: str | Path, run_id: str) -> dict:
    run_dir = Path(runs_dir) / run_id
    meta = _read_json_if_exists(run_dir / "meta.json")
    saved = {}
    for key, value in meta.items():
        saved[META_ARG_ALIASES.get(key, key)] = value
    saved.update(_read_json_if_exists(run_dir / "params.json"))
    return saved


def _last_metric_gen(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return -1
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            gen = json.loads(line).get("gen")
        except json.JSONDecodeError:
            continue
        return int(gen) if gen is not None else -1
    return -1


def latest_resumable_run(runs_dir: str | Path) -> str | None:
    root = Path(runs_dir)
    if not root.exists():
        return None
    candidates = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        state_path = run_dir / "state.pt"
        meta_path = run_dir / "meta.json"
        if not state_path.exists() and not meta_path.exists():
            continue
        mtime = max(
            (p.stat().st_mtime for p in (state_path, run_dir / "metrics.jsonl", meta_path) if p.exists()),
            default=run_dir.stat().st_mtime,
        )
        gen = _last_metric_gen(run_dir / "metrics.jsonl")
        candidates.append((mtime, gen, run_dir.name))
    if not candidates:
        return None
    return max(candidates)[2]


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
    "STOP": "\033[31m",
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


def install_fast_sigint_handler():
    def _fast_exit(_signum, _frame):
        try:
            snek.request_cancel()
        except Exception:
            pass
        os._exit(130)

    signal.signal(signal.SIGINT, _fast_exit)


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
    recency: float = 1.0,
) -> dict:
    """Run `steps` SGD updates on minibatches drawn from `samples` (a replay
    buffer's worth of recent positions). `recency` > 1 biases sampling toward the
    most recent positions (buffer is oldest->newest); 1.0 = uniform."""
    # Keep the replay window in CPU RAM. The old path copied the full replay
    # buffer to GPU every generation, so GPU memory grew with buffer size and
    # PyTorch's caching allocator retained the high-water mark. Only the current
    # minibatch needs to live on the GPU.
    obs = samples.obs
    pol = samples.pol
    z = samples.z
    temp = samples.temp  # [n] per-sample temperature, or None
    use_temp = temp is not None and getattr(net.cfg, "temperature_input", False)
    n = obs.shape[0]

    net.train()
    aug_rng = np.random.default_rng()
    pl = vl = 0.0
    bs = min(batch_size, n)
    for _ in range(steps):
        if recency and recency != 1.0:
            # r**recency skews toward 0 -> index near n-1 (most recent); tail still
            # reaches old samples. recency=1 is uniform.
            r = np.random.random(bs) ** recency
            idx = (n - 1 - (r * (n - 1))).astype(np.int64)
        else:
            idx = np.random.randint(0, n, size=bs)
        # D4 symmetry augmentation: rotate/reflect the egocentric obs and remap
        # the policy target consistently (value targets and the scalar temperature
        # are symmetry-invariant).
        ob, pb = augment_batch(obs[idx], pol[idx], aug_rng)
        obs_b = torch.from_numpy(ob).to(device, non_blocking=True)
        pol_b = torch.from_numpy(pb).to(device, non_blocking=True)
        z_b = torch.from_numpy(z[idx]).to(device, non_blocking=True)
        temp_b = (
            torch.from_numpy(temp[idx]).to(device, non_blocking=True) if use_temp else None
        )
        with net_autocast(device):
            logits, value = net(obs_b, temp_b)
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
        "total_turns": int(turns.sum()),
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net.eval()
    side = board  # absolute board coords: obs_side(board) == board
    dummy = torch.zeros(1, channels, side, side, device=device)
    dyn = {"obs": {0: "batch"}, "policy_logits": {0: "batch"}, "value": {0: "batch"}}
    # Temperature-conditioned nets (Albatross) take a second [batch] `temp` input.
    if getattr(net.cfg, "temperature_input", False):
        args = (dummy, torch.zeros(1, device=device))
        names = ["obs", "temp"]
        dyn["temp"] = {0: "batch"}
    else:
        args = (dummy,)
        names = ["obs"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net, args, str(path),
            input_names=names, output_names=["policy_logits", "value"],
            dynamic_axes=dyn,
            opset_version=17, dynamo=False,
        )


def publish_serving_checkpoint(net, channels: int, board: int, device, state_path: Path, ckpt_dir: Path) -> None:
    """Publish the latest train state and ONNX model for the live Battlesnake API."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_pt = ckpt_dir / "latest.pt"
    latest_onnx = ckpt_dir / "latest.onnx"

    tmp_pt = latest_pt.with_suffix(".pt.tmp")
    shutil.copyfile(state_path, tmp_pt)
    tmp_pt.replace(latest_pt)

    tmp_onnx = latest_onnx.with_suffix(".onnx.tmp")
    export_onnx(net, channels, board, device, tmp_onnx)
    tmp_onnx.replace(latest_onnx)


def main():
    logger = setup_logger()
    install_fast_sigint_handler()
    defaults = load_default_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=defaults["generations"])
    ap.add_argument("--board", type=int, default=defaults["board"], help="board side (square)")
    ap.add_argument("--num-snakes", type=int, default=defaults["num_snakes"],
                    help="snakes per game; 4-player FFA subsumes 2-player as snakes die")
    ap.add_argument("--samples", type=int, default=defaults["samples"])
    ap.add_argument("--count", type=int, default=defaults["count"])
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=defaults["eval_batch_size"],
        help="leaf observations per neural-net eval chunk; lower reduces eval tensor memory",
    )
    ap.add_argument(
        "--search-threads",
        type=int,
        default=defaults["search_threads"],
        help="Rayon threads for Rust search/encoding (default: all visible CPUs; 0 leaves Rayon default)",
    )
    ap.add_argument("--lr", type=float, default=defaults["lr"])
    ap.add_argument("--train-steps", type=int, default=defaults["train_steps"], help="SGD steps per generation")
    ap.add_argument("--batch-size", type=int, default=defaults["batch_size"], help="SGD minibatch size")
    ap.add_argument("--buffer-size", type=int, default=defaults["buffer_size"], help="replay buffer capacity (samples)")
    ap.add_argument("--max-turns", type=int, default=defaults["max_turns"], help="0 plays until terminal; positive values cap games as draws")
    # AlphaZero MCTS search.
    ap.add_argument("--sims", type=int, default=defaults["sims"], help="MCTS simulations per move")
    ap.add_argument("--c-puct", type=float, default=defaults["c_puct"], help="PUCT exploration constant")
    ap.add_argument("--exploration-prob", type=float, default=defaults["exploration_prob"], help="uniform-legal mix into the played action")
    ap.add_argument("--draw-value", type=float, default=defaults["draw_value"], help="value/search target for all draws")
    ap.add_argument("--bootstrap-value", action="store_true",
                    help="value target = search root (equilibrium) value per state instead of the flat game outcome")
    ap.add_argument("--skip-short-draw-turns", type=int, default=defaults["skip_short_draw_turns"], help="drop terminal draw games up to this many turns from replay; 0 disables")
    # Network architecture (default = KataGo-style grid trunk; see net.py).
    ap.add_argument("--arch", type=str, default=defaults["arch"], choices=["grid", "pyramid"])
    ap.add_argument("--trunk-channels", type=int, default=defaults["trunk_channels"], help="grid trunk width")
    ap.add_argument("--trunk-blocks", type=int, default=defaults["trunk_blocks"], help="grid trunk depth")
    ap.add_argument("--gpool-every", type=int, default=defaults["gpool_every"],
                    help="grid: every Nth block gets a global-pooling bias")
    ap.add_argument("--ckpt-dir", type=str, default=None, help="serving weights dir (default: runs/<run-id>/ckpt)")
    ap.add_argument("--serve", action=argparse.BooleanOptionalAction, default=True,
                    help="host the live dashboard in-process (--no-serve for headless)")
    ap.add_argument("--serve-host", type=str, default="0.0.0.0")
    ap.add_argument("--serve-port", type=int, default=8050)
    ap.add_argument("--runs-dir", type=str, default="runs", help="dashboard run root")
    ap.add_argument("--run-id", type=str, default=None, help="run dir name (default: timestamp)")
    ap.add_argument("--sample-games", type=int, default=defaults["sample_games"],
                    help="self-play games serialised per gen (recorded internally during generation)")
    ap.add_argument("--sample-every", type=int, default=defaults["sample_every"], help="serialise self-play games every N generations")
    ap.add_argument("--keep-games", type=int, default=defaults["keep_games"], help="keep this many recent game files")
    ap.add_argument("--fresh", action="store_true", help="ignore saved state and restart this run-id from scratch")
    ap.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)  # deprecated: resume is the default
    args = ap.parse_args()

    device = device_auto()
    log_phase(logger, "SETUP", f"device={device}")

    # Headless single run, or serve the dashboard and run on request.
    if not args.serve:
        train_one_run(args, None, device, logger)
        return
    from pathlib import Path as _Path
    state = control.RunState()
    state.set_base_spec({k: defaults[k] for k in control.NEW_RUN_PARAMS if k in defaults})
    static_dir = _Path(__file__).resolve().parent.parent / "dashboard" / "static"
    control.serve_in_thread(state, args.serve_host, args.serve_port, _Path(args.runs_dir), static_dir)
    latest_run = None if args.run_id else latest_resumable_run(args.runs_dir)
    log_phase(logger, "SERVE",
              f"http://{args.serve_host}:{args.serve_port}  "
              + (f"auto-starting {args.run_id}" if args.run_id else
                 f"auto-resuming latest {latest_run}" if latest_run else
                 "idle — log in and start a run from the dashboard"))
    pending = (
        {"name": args.run_id, "cli": True, "resume": not args.fresh}
        if args.run_id else
        {"name": latest_run, "resume": True} if latest_run else
        None
    )
    while not state.shutdown:
        if pending is None:
            state.go_idle()
            while not state.pending_run and not state.shutdown:
                time.sleep(0.3)
            if state.shutdown:
                break
            pending = state.take_new_run()
            if pending is None:
                continue
        train_one_run(_args_for_pending(args, pending), state, device, logger)
        pending = state.take_new_run()  # a switch request ends the run; loop into it
        if device.type == "cuda":
            torch.cuda.empty_cache()
    state.go_idle()
    log_phase(logger, "DONE", "shutdown")


def _args_for_pending(base, pending):
    """Build an args namespace for a requested run: CLI run as launched; dashboard
    'resume' continues from checkpoint; otherwise a fresh named run with overrides."""
    a = copy.copy(base)
    a.run_id = pending["name"]
    a.fresh = not pending.get("resume", False)
    if not a.fresh:
        apply_params_to_args(a, run_config_params(a.runs_dir, a.run_id))
    if pending.get("cli"):
        return a
    for k, v in (pending.get("overrides") or {}).items():
        setattr(a, k, v)
    return a


def train_one_run(args, state, device, logger):
    if args.search_threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.search_threads)
        configured = snek.set_search_threads(args.search_threads)
        status = "configured" if configured else "already initialized"
        log_phase(logger, "SETUP", f"search_threads={args.search_threads} ({status})")
    sp = SelfPlayConfig(
        board=args.board,
        num_snakes=args.num_snakes,
        count=args.count,
        sims=args.sims,
        c_puct=args.c_puct,
        eval_batch_size=args.eval_batch_size,
        samples_per_gen=args.samples,
        max_turns=args.max_turns,
        exploration_prob=args.exploration_prob,
        draw_value=args.draw_value,
    )
    run = RunWriter(
        args.runs_dir,
        run_id=args.run_id,
        meta={
            "board": sp.board,
            "num_snakes": sp.num_snakes,
            "arch": args.arch,
            "trunk_channels": args.trunk_channels,
            "trunk_blocks": args.trunk_blocks,
            "count": args.count,
            "sims": args.sims,
            "c_puct": args.c_puct,
            "eval_batch_size": args.eval_batch_size,
            "max_turns": args.max_turns,
            "exploration_prob": args.exploration_prob,
            "draw_value": args.draw_value,
            "bootstrap_value": args.bootstrap_value,
            "skip_short_draw_turns": args.skip_short_draw_turns,
            "search_threads": args.search_threads,
            "generations": args.generations,
            "samples_per_gen": args.samples,
            "sample_games": args.sample_games,
            "sample_every": args.sample_every,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "lr": args.lr,
            "device": str(device),
        },
    )
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else DEFAULT_SERVING_CKPT_DIR
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
        cfg = NetConfig(channels=snek.CHANNELS, height=args.board, width=args.board,
                        arch=args.arch, trunk_channels=args.trunk_channels,
                        trunk_blocks=args.trunk_blocks, gpool_every=args.gpool_every)

    net = AZNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    start_gen = 0

    if resume is not None:
        net.load_state_dict(resume["net"])
        opt.load_state_dict(resume["opt"])
        start_gen = resume["gen"] + 1
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
    buffer_dir = run.dir / "buffer"
    if not args.fresh:
        restored = buffer.restore(buffer_dir)
        if restored:
            log_phase(logger, "RESUME", f"restored replay buffer: {restored:,} samples from {buffer_dir}")
    elif buffer_dir.exists():
        for f in buffer_dir.glob("gen_*_n*.npz"):
            f.unlink(missing_ok=True)  # --fresh: drop stale shards
    metrics_history = []
    if run.metrics_path.exists() and not args.fresh:
        for line in run.metrics_path.read_text().splitlines():
            if line.strip():
                try:
                    metrics_history.append(json.loads(line))
                except ValueError:
                    pass

    # Register this run with the already-running dashboard server (if serving).
    if state is not None:
        init_params = {k: getattr(args, k) for k in control.LIVE_PARAMS if hasattr(args, k)}
        run.write_json("params.json", init_params)
        state.begin_run(run.run_id, run.read_json("meta.json"), init_params, metrics_history,
                        persist=lambda p: run.write_json("params.json", p))

    onnx_path = run.dir / "model.onnx"
    selfplay_state_id = snek.create_selfplay_state(
        board=sp.board,
        num_snakes=sp.num_snakes,
        count=args.count,
        seed=10_000 + start_gen,
    )
    gen_iter = (
        itertools.count(start_gen) if (state is not None and args.generations == 0)
        else range(start_gen, args.generations if args.generations else start_gen + 1)
    )
    for gen in gen_iter:
        # pause/stop gate + live-param apply (dashboard control)
        if state is not None:
            while state.paused and not (state.stopping or state.shutdown or state.pending_run):
                time.sleep(0.3)
            if state.stopping or state.shutdown or state.pending_run:
                break
            p = state.params_snapshot()
            args.count = p.get("count", args.count)
            args.samples = p.get("samples", args.samples)
            args.sims = p.get("sims", args.sims)
            args.c_puct = p.get("c_puct", args.c_puct)
            args.train_steps = p.get("train_steps", args.train_steps)
            args.batch_size = p.get("batch_size", args.batch_size)
            args.exploration_prob = p.get("exploration_prob", args.exploration_prob)
            args.draw_value = p.get("draw_value", args.draw_value)
            args.max_turns = p.get("max_turns", args.max_turns)
            args.sample_games = p.get("sample_games", args.sample_games)
            args.sample_every = p.get("sample_every", args.sample_every)
            args.keep_games = p.get("keep_games", args.keep_games)
            args.skip_short_draw_turns = p.get("skip_short_draw_turns", args.skip_short_draw_turns)
            lr = p.get("lr", args.lr)
            for grp in opt.param_groups:
                grp["lr"] = lr
            sp.samples_per_gen = args.samples
            sp.max_turns = args.max_turns
            sp.exploration_prob = args.exploration_prob
            sp.draw_value = args.draw_value
            state.set_status(generation=gen, phase="self-play", running=True)
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
        try:
            generated = snek.generate_selfplay(
                str(onnx_path), board=sp.board, num_snakes=sp.num_snakes,
                count=args.count, sims=args.sims, c_puct=args.c_puct,
                samples_per_gen=args.samples, seed=1000 + gen,
                exploration_prob=args.exploration_prob, max_turns=args.max_turns,
                draw_value=args.draw_value, skip_short_draw_turns=args.skip_short_draw_turns,
                record_games=rust_sample_games, bootstrap_value=args.bootstrap_value,
                state_id=selfplay_state_id,
            )
        except Exception as e:
            msg = str(e).lower()
            cancelled = state is not None and (
                state.stopping or state.shutdown or state.pending_run
                or "cancelled" in msg or "interrupted" in msg
            )
            if not cancelled:
                raise
            log_phase(logger, "STOP", f"gen={gen} self-play cancelled; discarding partial generation")
            run.write_status(
                {
                    "generation": gen - 1,
                    "running": False,
                    "total_generations": args.generations or None,
                    "phase": "stopped",
                }
            )
            if state is not None:
                state.set_status(running=False, paused=False, phase="stopped", progress=None)
            break
        if len(generated) == 4:
            obs, pol, z, gen_stats = generated
            gen_stats = dict(gen_stats)
        else:
            obs, pol, z = generated
            gen_stats = {}
        samples = Samples(obs=obs, pol=pol, z=z, turns=int(z.shape[0]), games=0)
        target_stats = policy_target_stats(samples.pol)
        buffer.add(samples)
        # Persist this gen's samples so a restart keeps the recency window.
        save_shard(buffer_dir, gen, samples)
        prune_shards(buffer_dir, args.buffer_size)
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

        total_completed_turns = int(selfplay_summary.get("total_turns", 0))
        turns_per_sec = total_completed_turns / max(t_gen, 1e-9)
        samples_per_sec = samples.obs.shape[0] / max(t_gen, 1e-9)
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
            "samples_per_sec": round(samples_per_sec, 0),
            "turns_per_sec": round(turns_per_sec, 0),
            "games_per_sec": round(games_per_sec, 2),
            "avg_game_len": selfplay_summary.get("turns", {}).get("mean"),
            "draw_rate": round(
                float(selfplay_summary.get("draws", 0)) / max(1, int(selfplay_summary.get("completed_games", 0))),
                4,
            ),
            "terminal_draw_rate": round(
                float(selfplay_summary.get("terminal_draws", 0)) / max(1, int(selfplay_summary.get("completed_games", 0))),
                4,
            ),
            "decisive_win_rate": selfplay_summary.get("decisive_win_rate"),
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
                length_balanced_samples=bool(gen_stats.get("length_balanced_samples", False)),
                continuous_selfplay=bool(gen_stats.get("continuous_selfplay", False)),
                selected_length_buckets=list(gen_stats.get("selected_length_buckets", [])),
            )
        msg = (
            f"gen {gen:3d} | samples {metric['samples']:6d} "
            f"| pol {metric['policy_loss']:.4f} val {metric['value_loss']:.4f} "
            f"| Hπ {metric['target_entropy']:.4f} maxπ {metric['target_max_prob']:.3f} "
            f"| {samples_per_sec:5.0f} samples/s {turns_per_sec:5.0f} turns/s {games_per_sec:4.1f} games/s "
            f"| gen {t_gen:5.1f}s train {t_train:4.1f}s"
        )

        # Self-play already records + serialises real games internally (Rust
        # generate_selfplay -> sampled_games). No separate eval/relative/recording
        # passes — those just replay extra games and waste wall-clock.
        if sampled_games or selfplay_summary.get("completed_games"):
            run.save_games(gen, sampled_games, summary=selfplay_summary)
            run.prune_games(keep=args.keep_games)

        log_phase(logger, "SAVING", f"gen={gen} checkpoint=state metrics=status")
        t_phase = time.time()
        save_state(gen)  # full resumable state, every generation (atomic write)
        publish_serving_checkpoint(net, cfg.channels, sp.board, device, run.state_path, ckpt_dir)
        metric["save_seconds"] = round(time.time() - t_phase, 1)
        run.append_metric(metric)
        metrics_history.append(metric)
        run.write_status(
            {
                "generation": gen,
                "running": args.generations == 0 or gen < args.generations - 1,
                "total_generations": args.generations or None,
                "last": metric,
            }
        )
        if state is not None:
            state.add_metric(metric)
            state.set_status(generation=gen, phase="running", running=True, last=metric)
        log_phase(
            logger,
            "METRICS",
            msg.replace(f"gen {gen:3d} | ", f"gen={gen} "),
        )

    run.write_status(
        {
            "generation": args.generations - 1,
            "running": False,
            "total_generations": args.generations,
            "last": metric if "metric" in dir() else None,
        }
    )
    try:
        snek.drop_selfplay_state(selfplay_state_id)
    except Exception:
        pass
    log_phase(logger, "DONE", f"generations={args.generations}")


if __name__ == "__main__":
    main()
