"""Resume continues a run from its saved state instead of restarting."""

import json
import os
import subprocess
import sys
from pathlib import Path


def _gens(run_dir: Path):
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line)["gen"] for line in lines if line.strip()]


def _train(runs_dir: Path, run_id: str, generations: int, resume: bool = False):
    cmd = [
        sys.executable, "-m", "azsnek.train",
        "--runs-dir", str(runs_dir), "--run-id", run_id,
        "--generations", str(generations), "--samples", "120", "--count", "8",
        "--depth", "1", "--iters", "12", "--blocks", "1", "--filters", "8",
        "--eval-every", "999", "--record-games", "0", "--ckpt-dir", str(runs_dir / "ckpt"),
    ]
    if resume:
        cmd.append("--resume")
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300, env=dict(os.environ))


def test_resume_continues_generation_numbering(tmp_path):
    _train(tmp_path, "r", generations=2)
    run_dir = tmp_path / "r"
    assert _gens(run_dir) == [0, 1]
    assert run_dir.joinpath("state.pt").exists(), "full training state is saved"

    # Resume and extend: generations must continue, not restart from 0.
    _train(tmp_path, "r", generations=4, resume=True)
    assert _gens(run_dir) == [0, 1, 2, 3]
