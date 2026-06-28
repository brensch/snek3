"""Resume continues a run from its saved state instead of restarting."""

import json
import os
import subprocess
import sys
from pathlib import Path


def _gens(run_dir: Path):
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line)["gen"] for line in lines if line.strip()]


def _train(runs_dir: Path, run_id: str, generations: int, fresh: bool = False):
    cmd = [
        sys.executable, "-m", "azsnek.train",
        "--runs-dir", str(runs_dir), "--run-id", run_id,
        "--generations", str(generations), "--samples", "40", "--count", "4",
        "--sims", "4", "--blocks", "1", "--filters", "8",
        "--train-steps", "1", "--batch-size", "16", "--eval-every", "0",
        "--relative-every", "0", "--league-every", "0", "--record-games", "0",
        "--sample-games", "0", "--ckpt-dir", str(runs_dir / "ckpt"),
    ]
    if fresh:
        cmd.append("--fresh")
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120, env=dict(os.environ))


def test_rerunning_same_run_id_auto_resumes(tmp_path):
    _train(tmp_path, "r", generations=2)
    run_dir = tmp_path / "r"
    assert _gens(run_dir) == [0, 1]
    assert run_dir.joinpath("state.pt").exists(), "full training state is saved"

    # Same run-id, no flags: must auto-resume and continue, not restart from 0.
    _train(tmp_path, "r", generations=4)
    assert _gens(run_dir) == [0, 1, 2, 3]

    # --fresh wipes prior progress and restarts numbering.
    _train(tmp_path, "r", generations=2, fresh=True)
    assert _gens(run_dir) == [0, 1]
