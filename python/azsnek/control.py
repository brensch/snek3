"""Embedded live control + telemetry server for the Albatross trainer.

The trainer process hosts this in-process: a background uvicorn thread serves a
FastAPI app while the training loop runs in the main thread. The single source
of truth for the *live* run is `RunState` (in memory) -- the dashboard reads it
over REST and a Server-Sent-Events stream instead of polling files. Files are
still written by RunWriter for durability/resume and for browsing past runs.

Endpoints (live run is whatever this process is training):
    GET  /api/state              snapshot: run, meta, status, params, metrics
    GET  /api/stream             SSE: snapshot then metric/progress/status/params
    POST /api/params             patch live-tunable params (token)        -> next gen
    POST /api/control            {action: pause|resume|stop|checkpoint}   (token)
    GET  /api/runs               list run ids (live + archived on disk)
    GET  /api/runs/{id}/...      meta/metrics/games for any run (file-backed)
    GET  /  /run/{name}          dashboard SPA

Write endpoints require `Authorization: Bearer <token>` (or ?token=) when a
token is configured; read endpoints are open so the dashboard "just works".
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Params that can be changed live and take effect at the next generation
# boundary, with the coercion applied to incoming JSON values.
LIVE_PARAMS: dict[str, type] = {
    "count": int, "samples": int, "iters": int,
    "lr": float, "train_steps": int, "batch_size": int, "recency": float,
    "draw_value": float, "exploration_prob": float,
    "tau_min": float, "tau_max": float, "response_tau": float,
    "eval_every": int, "eval_games": int,
    "record_games": int, "record_every": int,
}
# Baked into the net / board at startup; cannot change without a fresh run.
LOCKED_PARAMS = ("filters", "blocks", "board", "num_snakes", "depth")

# Everything a dashboard-created run may override (live + locked + a few extras),
# with the coercion applied to incoming JSON values.
NEW_RUN_PARAMS: dict[str, type] = {
    **LIVE_PARAMS,
    "filters": int, "blocks": int, "board": int, "num_snakes": int, "depth": int,
    "response_after": int, "generations": int, "eval_opp_tau": float,
    "uct_iters": int, "max_turns": int, "eval_batch_size": int,
    "buffer_size": int, "keep_games": int,
}


class RunState:
    """Thread-safe live state shared between the training thread (writer) and the
    server thread (readers / SSE fan-out)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.run_id: str | None = None       # None == idle (server up, no run)
        self.meta: dict = {}
        self.params: dict = {}
        self.metrics: list[dict] = []
        self.status: dict = {"running": False, "paused": False, "generation": None,
                             "phase": "idle", "progress": None}
        self.base_spec: dict = {}            # default config for dashboard-created runs
        self._paused = False
        self._stop_run = False               # stop current run -> back to idle
        self._new_run: dict | None = None    # {"name":..., "overrides":{...}}
        self._shutdown = False               # exit the process entirely
        # set by the server thread once its event loop is up
        self.loop: asyncio.AbstractEventLoop | None = None
        self.subscribers: set[asyncio.Queue] = set()  # touched only on loop thread

    # ---- run lifecycle (called by the training thread) ----
    def set_base_spec(self, spec: dict) -> None:
        with self._lock:
            self.base_spec = dict(spec)

    def begin_run(self, run_id: str, meta: dict, params: dict, history: list[dict]) -> None:
        with self._lock:
            self.run_id = run_id
            self.meta = dict(meta)
            self.params = {k: params[k] for k in LIVE_PARAMS if k in params}
            self.metrics = list(history)
            self.status = {"running": True, "paused": False, "generation": None,
                           "phase": "starting", "progress": None}
            self._paused = False
            self._stop_run = False
        self.publish({"type": "snapshot", **self.snapshot()})

    def go_idle(self) -> None:
        with self._lock:
            self.run_id = None
            self.meta = {}
            self.params = {}
            self.metrics = []
            self.status = {"running": False, "paused": False, "generation": None,
                           "phase": "idle", "progress": None}
            self._paused = False
            self._stop_run = False
        self.publish({"type": "snapshot", **self.snapshot()})

    # ---- control flags (read by the training loop) ----
    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def stopping(self) -> bool:
        return self._stop_run

    @property
    def pending_run(self) -> bool:
        return self._new_run is not None

    @property
    def shutdown(self) -> bool:
        return self._shutdown

    def request_stop(self) -> None:
        self._stop_run = True
        self.set_status(phase="stopping")

    def request_new_run(self, name: str, overrides: dict) -> None:
        self._new_run = {"name": name, "overrides": dict(overrides or {})}
        if self.run_id:  # interrupt the active run to switch to the new one
            self.set_status(phase="switching")

    def take_new_run(self) -> dict | None:
        nr = self._new_run
        self._new_run = None
        return nr

    def request_shutdown(self) -> None:
        self._shutdown = True
        self._stop_run = True

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.set_status(paused=paused, phase="paused" if paused else "running")

    # ---- params ----
    def params_snapshot(self) -> dict:
        with self._lock:
            return dict(self.params)

    def update_params(self, patch: dict) -> dict:
        applied, rejected = {}, {}
        with self._lock:
            for k, v in patch.items():
                if k in LIVE_PARAMS:
                    try:
                        self.params[k] = LIVE_PARAMS[k](v)
                        applied[k] = self.params[k]
                    except (TypeError, ValueError):
                        rejected[k] = "bad value"
                elif k in LOCKED_PARAMS:
                    rejected[k] = "locked (needs a fresh run)"
                else:
                    rejected[k] = "unknown"
            snap = dict(self.params)
        if applied:
            self.publish({"type": "params", "params": snap})
        return {"applied": applied, "rejected": rejected, "params": snap}

    # ---- telemetry (called by the training thread) ----
    def add_metric(self, metric: dict) -> None:
        with self._lock:
            self.metrics.append(metric)
        self.publish({"type": "metric", "metric": metric})

    def set_status(self, **kw) -> None:
        with self._lock:
            self.status.update(kw)
            snap = dict(self.status)
        self.publish({"type": "status", "status": snap})

    def set_progress(self, phase: str, done: int, total: int, gen: int) -> None:
        self.set_status(phase=phase, generation=gen,
                        progress={"done": done, "total": total})

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run": self.run_id, "meta": self.meta, "status": dict(self.status),
                "params": dict(self.params), "metrics": list(self.metrics),
                "base_spec": dict(self.base_spec),
                "live_params": list(LIVE_PARAMS), "locked_params": list(LOCKED_PARAMS),
            }

    # ---- SSE fan-out (publish is thread-safe; _fanout runs on the loop) ----
    def publish(self, event: dict) -> None:
        loop = self.loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._fanout, event)
        except RuntimeError:
            pass  # loop shutting down

    def _fanout(self, event: dict) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def build_app(state: RunState, runs_dir: Path, static_dir: Path, token: str | None):
    app = FastAPI(title="snek3 trainer")
    runs_dir = Path(runs_dir).resolve()

    def require_token(request: Request) -> None:
        if not token:
            return
        auth = request.headers.get("authorization", "")
        supplied = auth[7:] if auth.lower().startswith("bearer ") else request.query_params.get("token")
        if supplied != token:
            raise HTTPException(status_code=401, detail="bad or missing token")

    def safe_run(run: str) -> Path:
        if "/" in run or "\\" in run or run in ("", ".", ".."):
            raise HTTPException(status_code=400, detail="bad run id")
        p = (runs_dir / run).resolve()
        if p.parent != runs_dir or not p.is_dir():
            raise HTTPException(status_code=404, detail="run not found")
        return p

    # ---------- live (in-memory) ----------
    @app.get("/api/state")
    def get_state():
        return state.snapshot()

    @app.get("/api/stream")
    async def stream(request: Request):
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        state.subscribers.add(q)

        async def gen():
            try:
                yield _sse({"type": "snapshot", **state.snapshot()})
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield _sse(ev)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                state.subscribers.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/params")
    async def set_params(request: Request):
        require_token(request)
        patch = await request.json()
        if not isinstance(patch, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        return state.update_params(patch)

    @app.post("/api/control")
    async def control(request: Request):
        require_token(request)
        body = await request.json()
        action = (body or {}).get("action")
        if action == "pause":
            state.set_paused(True)
        elif action == "resume":
            state.set_paused(False)
        elif action == "stop":
            state.request_stop()
        elif action == "shutdown":
            state.request_shutdown()
        else:
            raise HTTPException(status_code=400, detail="unknown action")
        return {"ok": True, "action": action}

    @app.post("/api/runs")
    async def create_run(request: Request):
        """Start (or switch to) a fresh named run. Stops the active run first
        (its checkpoint is saved) and begins the new one at the next boundary."""
        require_token(request)
        body = await request.json()
        name = str((body or {}).get("name", "")).strip()
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise HTTPException(status_code=400, detail="bad run name")
        if (runs_dir / name).exists():
            raise HTTPException(status_code=409, detail="a run with that name already exists")
        overrides = {}
        for k, v in ((body or {}).get("params") or {}).items():
            if k not in NEW_RUN_PARAMS:
                raise HTTPException(status_code=400, detail=f"unknown param: {k}")
            try:
                overrides[k] = NEW_RUN_PARAMS[k](v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"bad value for {k}")
        state.request_new_run(name, overrides)
        return {"ok": True, "name": name, "overrides": overrides}

    # ---------- runs / history (file-backed; for browsing any run) ----------
    @app.get("/api/runs")
    def list_runs():
        live = state.run_id
        names = set()
        if runs_dir.exists():
            names = {p.name for p in runs_dir.iterdir() if p.is_dir()}
        if live:
            names.add(live)
        ordered = ([live] if live else []) + sorted((n for n in names if n != live), reverse=True)
        return {"runs": ordered, "live": live}

    def _read_jsonl(p: Path) -> list[dict]:
        out = []
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out

    @app.get("/api/runs/{run}/meta")
    def run_meta(run: str):
        if run == state.run_id:
            return {"run_id": run, **state.meta}
        p = safe_run(run) / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @app.get("/api/runs/{run}/metrics")
    def run_metrics(run: str):
        if run == state.run_id:
            return {"metrics": state.snapshot()["metrics"]}
        return {"metrics": _read_jsonl(safe_run(run) / "metrics.jsonl")}

    @app.get("/api/runs/{run}/status")
    def run_status(run: str):
        if run == state.run_id:
            return state.snapshot()["status"]
        p = safe_run(run) / "status.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @app.get("/api/runs/{run}/games")
    def list_games(run: str):
        gdir = safe_run(run) / "games"
        if not gdir.exists():
            return {"files": []}
        files = []
        for f in sorted(gdir.glob("gen_*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            selfplay = data.get("selfplay") or {}
            files.append({
                "file": f.name, "gen": data.get("gen"),
                "selfplay": {k: v for k, v in selfplay.items() if k != "games"},
                "games": [{"opponent": g.get("opponent"), "winner": g.get("winner"),
                           "num_turns": g.get("num_turns")} for g in data.get("games", [])],
            })
        return {"files": files}

    @app.get("/api/runs/{run}/games/{name}")
    def get_game_file(run: str, name: str):
        if "/" in name or "\\" in name or ".." in name or not name.endswith(".json"):
            raise HTTPException(status_code=400, detail="bad file")
        p = safe_run(run) / "games" / name
        if not p.is_file():
            raise HTTPException(status_code=404, detail="game file not found")
        return json.loads(p.read_text())

    # ---------- dashboard SPA ----------
    if (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/run/{name}")
    def index_run(name: str):
        return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})

    @app.on_event("startup")
    async def _capture_loop():
        state.loop = asyncio.get_running_loop()

    return app


def serve_in_thread(state: RunState, host: str, port: int, runs_dir: Path,
                    static_dir: Path, token: str | None) -> threading.Thread:
    """Start uvicorn on a daemon thread (signal handlers disabled since we're not
    on the main thread) and return once it's accepting connections."""
    import uvicorn

    app = build_app(state, runs_dir, static_dir, token)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    t = threading.Thread(target=server.run, name="snek-control", daemon=True)
    t.start()
    for _ in range(100):  # wait up to ~5s for startup
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    return t
