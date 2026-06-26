"""Recorder produces replayable frames; the dashboard serves runs/metrics/games."""

from pathlib import Path

import snek
from fastapi.testclient import TestClient

from azsnek.net import AZNet, NetConfig, device_auto
from azsnek.recorder import record_games
from azsnek.runlog import RunWriter


def test_recorder_frames_are_wellformed():
    device = device_auto()
    net = AZNet(NetConfig(channels=snek.CHANNELS, filters=8, blocks=1)).to(device)
    games = record_games(net, device, n_games=1, depth=1, iters=20, opponent="baseline", seed=1)
    assert len(games) == 1
    g = games[0]
    assert g["num_turns"] == len(g["frames"]) and g["frames"]
    f0 = g["frames"][0]
    assert {"turn", "width", "height", "food", "snakes"} <= set(f0)
    assert len(f0["snakes"]) == 2
    assert f0["snakes"][0]["body"], "head-first body present"


def test_dashboard_serves_run_data(tmp_path):
    device = device_auto()
    net = AZNet(NetConfig(channels=snek.CHANNELS, filters=8, blocks=1)).to(device)
    games = record_games(net, device, n_games=1, depth=1, iters=20, opponent="baseline", seed=2)

    rw = RunWriter(tmp_path, run_id="testrun", meta={"board": 11, "filters": 8, "blocks": 1, "depth": 1})
    rw.append_metric({"gen": 0, "win_rate": 0.5, "policy_loss": 1.0, "value_loss": 0.3, "samples": 100})
    rw.write_status({"generation": 0, "running": False, "last": {"gen": 0, "win_rate": 0.5}})
    rw.save_games(0, games)

    import dashboard.app as dash

    dash.RUNS_DIR = Path(tmp_path).resolve()
    client = TestClient(dash.app)

    assert client.get("/").status_code == 200
    assert "testrun" in client.get("/api/runs").json()["runs"]
    metrics = client.get("/api/runs/testrun/metrics").json()["metrics"]
    assert metrics and metrics[0]["win_rate"] == 0.5
    files = client.get("/api/runs/testrun/games").json()["files"]
    assert files and files[0]["gen"] == 0
    full = client.get(f"/api/runs/testrun/games/{files[0]['file']}").json()
    assert full["games"][0]["frames"]
    # path traversal is rejected
    assert client.get("/api/runs/..%2F..%2Fetc/metrics").status_code in (400, 404)
