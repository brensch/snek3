"""Filesystem layout the trainer writes to and the dashboard reads from.

A run directory looks like:

    runs/<run_id>/
        meta.json              run config (board, net size, hyperparams)
        metrics.jsonl          one JSON object per generation (appended live)
        status.json            latest summary (overwritten each generation)
        models/gen_XXXX.pt     model snapshots per generation
        ckpt/latest.pt         serving weights from the latest eval
        ckpt/best.pt           serving weights from the best eval
        games/gen_XXXX.json    recorded replays at that generation

Everything is plain JSON so it can be inspected by hand or by the dashboard.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class RunWriter:
    def __init__(self, runs_root: str | Path, run_id: str | None = None, meta: dict | None = None):
        self.root = Path(runs_root)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = self.root / self.run_id
        self.games_dir = self.dir / "games"
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Out-of-band "real games": faithful (proxy ONNX + serve search) eval vs
        # the fixed pool, written here directly by the Rust `snek-eval` binary.
        self.eval_dir = self.dir / "eval"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.state_path = self.dir / "state.pt"  # full resumable training state
        self.started = time.time()
        if meta is not None:
            self.write_json("meta.json", {"run_id": self.run_id, **meta})

    def save_state(self, save_fn) -> None:
        """Atomically persist resumable state. `save_fn(path)` does the write
        (e.g. ``lambda p: torch.save(obj, p)``); the temp file is renamed into
        place so a reader/resumer never sees a half-written checkpoint."""
        tmp = self.state_path.with_suffix(".pt.tmp")
        save_fn(tmp)
        tmp.replace(self.state_path)

    def has_state(self) -> bool:
        return self.state_path.exists()

    def reset(self) -> None:
        """Clear a run's progress (metrics, status, state, replays) for a fresh
        restart under the same run id. Keeps meta.json."""
        for p in (self.metrics_path, self.state_path, self.dir / "status.json",
                  self.dir / "params.json"):
            p.unlink(missing_ok=True)
        for g in self.games_dir.glob("gen_*.json"):
            g.unlink(missing_ok=True)
        for m in self.models_dir.glob("gen_*.pt"):
            m.unlink(missing_ok=True)
        for e in self.eval_dir.glob("gen_*.json"):
            e.unlink(missing_ok=True)

    def save_model(self, gen: int, save_fn) -> Path:
        """Atomically persist a per-generation model snapshot."""
        path = self.models_dir / f"gen_{gen:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        save_fn(tmp)
        tmp.replace(path)
        return path

    def write_json(self, name: str, obj) -> None:
        path = self.dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj))
        tmp.replace(path)  # atomic, so the dashboard never reads a half-written file

    def read_json(self, name: str):
        return json.loads((self.dir / name).read_text())

    def append_metric(self, metric: dict) -> None:
        metric = {"wall_time": round(time.time() - self.started, 1), **metric}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(metric) + "\n")

    def write_status(self, status: dict) -> None:
        self.write_json(
            "status.json",
            {"updated": datetime.now(timezone.utc).isoformat(), **status},
        )

    def save_games(self, gen: int, games: list[dict], summary: dict | None = None) -> None:
        payload = {"gen": gen, "games": games}
        if summary is not None:
            payload["selfplay"] = summary
        self.write_json(f"games/gen_{gen:04d}.json", payload)

    def prune_games(self, keep: int) -> None:
        """Keep only the `keep` most recent game files to bound disk usage."""
        files = sorted(self.games_dir.glob("gen_*.json"))
        for f in files[: max(0, len(files) - keep)]:
            try:
                f.unlink()
            except OSError:
                pass

    def eval_artifact_path(self, gen: int) -> Path:
        """Where the Rust evaluator writes gen `gen`'s win-rates + real games."""
        return self.eval_dir / f"gen_{gen:04d}.json"

    def prune_eval(self, keep: int) -> None:
        """Keep only the `keep` most recent eval artifacts."""
        files = sorted(self.eval_dir.glob("gen_*.json"))
        for f in files[: max(0, len(files) - keep)]:
            try:
                f.unlink()
            except OSError:
                pass
