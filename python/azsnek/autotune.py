"""Adaptive training controller.

Runs ``azsnek.train`` in short resumable chunks, reads the run metrics after
each chunk, and adjusts high-level knobs for the next chunk. This is an outer
loop: the trainer remains simple and resumable, while this controller can back
off over-training or gather more data when a run stalls.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TuneSettings:
    samples: int = 50_000
    count: int = 32
    depth: int = 2
    tau: float = 30.0
    iters: int = 120
    eval_batch_size: int = 8192
    search_threads: int = 0
    train_steps: int = 256
    batch_size: int = 2048
    buffer_size: int = 500_000
    filters: int = 64
    blocks: int = 6
    eval_games: int = 64
    max_turns: int = 0
    record_games: int = 8
    record_every: int = 5


@dataclass
class TuneLimits:
    min_samples: int = 24_000
    max_samples: int = 120_000
    min_train_steps: int = 64
    max_train_steps: int = 512
    min_tau: float = 12.0
    max_tau: float = 80.0
    min_eval_games: int = 64
    max_eval_games: int = 256
    target_buffer_epochs: float = 1.5
    max_new_sample_epochs: float = 12.0
    plateau_window: int = 8


def _round_steps(value: float) -> int:
    return max(16, int(round(value / 16)) * 16)


def _last_number(rows: list[dict], key: str):
    for row in reversed(rows):
        value = row.get(key)
        if value is not None:
            return value
    return None


def _win_values(rows: list[dict]) -> list[float]:
    return [float(r["win_rate"]) for r in rows if r.get("win_rate") is not None]


def _loss_regressing(rows: list[dict], window: int) -> bool:
    hist = rows[-window:]
    if len(hist) < 4:
        return False
    pol = [r.get("policy_loss") for r in hist if r.get("policy_loss") is not None]
    val = [r.get("value_loss") for r in hist if r.get("value_loss") is not None]
    policy_backing_up = len(pol) >= 4 and pol[-1] > min(pol[:-1]) + 0.02
    value_backing_up = len(val) >= 4 and val[-1] > min(val[:-1]) * 1.12
    return policy_backing_up or value_backing_up


def _win_plateaued(rows: list[dict], window: int) -> bool:
    wins = _win_values(rows)
    if len(wins) < window:
        return False
    recent = wins[-window // 2 :]
    previous = wins[-window : -window // 2]
    return max(recent) <= max(previous) + 0.03 and (
        sum(recent) / len(recent)
    ) <= (sum(previous) / len(previous)) + 0.01


def _cap_training_pressure(settings: TuneSettings, limits: TuneLimits, rows: list[dict]) -> None:
    buffer_size = int(_last_number(rows, "buffer") or settings.samples)
    cap_by_buffer = buffer_size * limits.target_buffer_epochs / settings.batch_size
    cap_by_new = settings.samples * limits.max_new_sample_epochs / settings.batch_size
    capped = min(settings.train_steps, cap_by_buffer, cap_by_new, limits.max_train_steps)
    settings.train_steps = max(limits.min_train_steps, _round_steps(capped))


def tune_next(settings: TuneSettings, limits: TuneLimits, rows: list[dict]) -> tuple[TuneSettings, list[str]]:
    """Return next settings and human-readable reasons."""
    next_settings = TuneSettings(**asdict(settings))
    reasons: list[str] = []

    if rows:
        _cap_training_pressure(next_settings, limits, rows)
        if next_settings.train_steps != settings.train_steps:
            reasons.append(
                f"cap train_steps to {next_settings.train_steps} to limit replay-buffer epochs"
            )

    if _loss_regressing(rows, limits.plateau_window):
        reduced = max(limits.min_train_steps, _round_steps(next_settings.train_steps * 0.6))
        if reduced < next_settings.train_steps:
            next_settings.train_steps = reduced
            reasons.append("policy/value loss is backing up; reduce optimization pressure")
        bumped_samples = min(limits.max_samples, int(next_settings.samples * 1.25))
        if bumped_samples > next_settings.samples:
            next_settings.samples = bumped_samples
            reasons.append("collect more fresh self-play before the next update")

    if _win_plateaued(rows, limits.plateau_window):
        next_settings.eval_games = min(limits.max_eval_games, max(next_settings.eval_games, 128))
        next_settings.samples = min(limits.max_samples, int(next_settings.samples * 1.2))
        reasons.append("win-rate has plateaued; increase eval confidence and fresh data")

    target_entropy = _last_number(rows, "target_entropy")
    target_max_prob = _last_number(rows, "target_max_prob")
    if target_entropy is not None and target_max_prob is not None:
        if target_entropy > 0.85 and target_max_prob < 0.55:
            next_settings.tau = min(limits.max_tau, round(next_settings.tau * 1.25, 3))
            reasons.append("targets are too soft; raise tau")
        elif target_entropy < 0.25 and target_max_prob > 0.9:
            next_settings.tau = max(limits.min_tau, round(next_settings.tau * 0.8, 3))
            reasons.append("targets are too sharp; lower tau")

    next_settings.samples = max(limits.min_samples, min(limits.max_samples, next_settings.samples))
    next_settings.train_steps = max(
        limits.min_train_steps, min(limits.max_train_steps, next_settings.train_steps)
    )
    next_settings.eval_games = max(
        limits.min_eval_games, min(limits.max_eval_games, next_settings.eval_games)
    )
    if not reasons:
        reasons.append("keep current settings")
    return next_settings, reasons


def read_metrics(runs_dir: Path, run_id: str) -> list[dict]:
    path = runs_dir / run_id / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_train_command(args, settings: TuneSettings, generations: int, fresh: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "azsnek.train",
        "--generations",
        str(generations),
        "--samples",
        str(settings.samples),
        "--count",
        str(settings.count),
        "--depth",
        str(settings.depth),
        "--tau",
        str(settings.tau),
        "--iters",
        str(settings.iters),
        "--eval-batch-size",
        str(settings.eval_batch_size),
        "--search-threads",
        str(settings.search_threads),
        "--train-steps",
        str(settings.train_steps),
        "--batch-size",
        str(settings.batch_size),
        "--buffer-size",
        str(settings.buffer_size),
        "--filters",
        str(settings.filters),
        "--blocks",
        str(settings.blocks),
        "--eval-every",
        "1",
        "--eval-games",
        str(settings.eval_games),
        "--max-turns",
        str(settings.max_turns),
        "--record-games",
        str(settings.record_games),
        "--record-every",
        str(settings.record_every),
        "--runs-dir",
        str(args.runs_dir),
        "--run-id",
        args.run_id,
    ]
    if fresh:
        cmd.append("--fresh")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--total-generations", type=int, default=2500)
    ap.add_argument("--chunk-generations", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--samples", type=int, default=50_000)
    ap.add_argument("--train-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--buffer-size", type=int, default=500_000)
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=8192)
    ap.add_argument("--search-threads", type=int, default=0)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--record-games", type=int, default=8)
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--min-train-steps", type=int, default=64)
    ap.add_argument("--max-train-steps", type=int, default=512)
    ap.add_argument("--min-samples", type=int, default=24_000)
    ap.add_argument("--max-samples", type=int, default=120_000)
    ap.add_argument("--target-buffer-epochs", type=float, default=1.5)
    ap.add_argument("--max-new-sample-epochs", type=float, default=12.0)
    args = ap.parse_args()

    args.runs_dir = Path(args.runs_dir)
    args.run_id = args.run_id or "adaptive-" + datetime.now().strftime("%Y%m%d-%H%M%S")
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
    limits = TuneLimits(
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        min_train_steps=args.min_train_steps,
        max_train_steps=args.max_train_steps,
        target_buffer_epochs=args.target_buffer_epochs,
        max_new_sample_epochs=args.max_new_sample_epochs,
    )

    print(f"adaptive run: {args.run_id}", flush=True)
    fresh = args.fresh
    while True:
        rows = read_metrics(args.runs_dir, args.run_id)
        current_gen = int(rows[-1]["gen"]) + 1 if rows else 0
        if current_gen >= args.total_generations:
            print(f"done: reached generation {current_gen}", flush=True)
            return

        settings, reasons = tune_next(settings, limits, rows)
        next_total = min(args.total_generations, current_gen + args.chunk_generations)
        print(
            json.dumps(
                {
                    "from_gen": current_gen,
                    "to_generation": next_total,
                    "settings": asdict(settings),
                    "reasons": reasons,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        cmd = build_train_command(args, settings, next_total, fresh=fresh)
        print("command:", " ".join(cmd), flush=True)
        if args.dry_run:
            return
        subprocess.run(cmd, check=True)
        fresh = False


if __name__ == "__main__":
    main()
