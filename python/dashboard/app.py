"""Dashboard server: serves the single-page UI and read-only JSON APIs over the
training run directories. The trainer writes runs/<id>/ live; this just reads it.

Run:
    SNEK_RUNS_DIR=runs uvicorn dashboard.app:app --port 8050
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

RUNS_DIR = Path(os.environ.get("SNEK_RUNS_DIR", "runs")).resolve()
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="snek3 dashboard")

# Vite build output: index.html + hashed assets under static/assets/.
# check_dir=False so the server still starts if the UI hasn't been built yet
# (run `make ui`); the API routes work regardless.
app.mount("/assets", StaticFiles(directory=STATIC / "assets", check_dir=False), name="assets")


def _safe_run(run: str) -> Path:
    """Resolve a run directory, refusing anything outside RUNS_DIR."""
    if "/" in run or "\\" in run or run in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="bad run id")
    p = (RUNS_DIR / run).resolve()
    if p.parent != RUNS_DIR or not p.is_dir():
        raise HTTPException(status_code=404, detail="run not found")
    return p


@app.get("/")
def index():
    # no-store so the browser always fetches the current UI (the dashboard is a
    # dev tool; stale cached HTML would be confusing).
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/run/{name}")
def index_run(name: str):
    # SPA deep-link: /run/<run_name> serves the same app so reloads/bookmarks
    # return to that run (the client reads the run name from the path).
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/runs")
def list_runs():
    if not RUNS_DIR.exists():
        return {"runs": []}
    runs = sorted(
        (p.name for p in RUNS_DIR.iterdir() if p.is_dir()),
        reverse=True,
    )
    return {"runs": runs}


@app.get("/api/runs/{run}/meta")
def run_meta(run: str):
    p = _safe_run(run) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/runs/{run}/status")
def run_status(run: str):
    p = _safe_run(run) / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/runs/{run}/metrics")
def run_metrics(run: str):
    p = _safe_run(run) / "metrics.jsonl"
    if not p.exists():
        return {"metrics": []}
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a torn final line during a live append
    return {"metrics": out}


@app.get("/api/runs/{run}/games")
def list_games(run: str):
    """Lightweight index of recorded replays (no frames)."""
    gdir = _safe_run(run) / "games"
    if not gdir.exists():
        return {"files": []}
    files = []
    for f in sorted(gdir.glob("gen_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        selfplay = data.get("selfplay") or {}
        selfplay_index = {k: v for k, v in selfplay.items() if k != "games"}
        files.append(
            {
                "file": f.name,
                "gen": data.get("gen"),
                "selfplay": selfplay_index,
                "games": [
                    {
                        "opponent": g.get("opponent"),
                        "winner": g.get("winner"),
                        "num_turns": g.get("num_turns"),
                    }
                    for g in data.get("games", [])
                ],
            }
        )
    return {"files": files}


@app.get("/api/runs/{run}/games/{name}")
def get_game_file(run: str, name: str):
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".json"):
        raise HTTPException(status_code=400, detail="bad file")
    p = _safe_run(run) / "games" / name
    if not p.is_file():
        raise HTTPException(status_code=404, detail="game file not found")
    return json.loads(p.read_text())


@app.get("/api/runs/{run}/eval")
def list_eval(run: str):
    """Index of faithful eval artifacts: per-gen win-rates + real-game list."""
    edir = _safe_run(run) / "eval"
    if not edir.exists():
        return {"files": []}
    files = []
    for f in sorted(edir.glob("gen_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        files.append(
            {
                "file": f.name,
                "gen": data.get("gen"),
                "vs_base": data.get("vs_base"),
                "vs_uct": data.get("vs_uct"),
                "summary": data.get("summary"),
                "games": [
                    {
                        "opponent": g.get("opponent"),
                        "winner": g.get("winner"),
                        "num_turns": g.get("num_turns"),
                    }
                    for g in data.get("games", [])
                ],
            }
        )
    return {"files": files}


@app.get("/api/runs/{run}/eval/{name}")
def get_eval_file(run: str, name: str):
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".json"):
        raise HTTPException(status_code=400, detail="bad file")
    p = _safe_run(run) / "eval" / name
    if not p.is_file():
        raise HTTPException(status_code=404, detail="eval file not found")
    return json.loads(p.read_text())
