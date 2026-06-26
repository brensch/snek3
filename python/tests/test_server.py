"""Smoke-test the Battlesnake server: a /move request returns a legal move."""

import json

from fastapi.testclient import TestClient


def _payload():
    return {
        "game": {"id": "g"},
        "turn": 3,
        "board": {
            "width": 11,
            "height": 11,
            "food": [{"x": 5, "y": 6}],
            "hazards": [],
            "snakes": [
                {
                    "id": "me",
                    "health": 90,
                    "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
                },
                {
                    "id": "opp",
                    "health": 85,
                    "body": [{"x": 2, "y": 2}, {"x": 2, "y": 1}, {"x": 2, "y": 0}],
                },
            ],
        },
        "you": {
            "id": "me",
            "health": 90,
            "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
        },
    }


def test_info_and_move():
    from server.main import app

    client = TestClient(app)

    info = client.get("/").json()
    assert info["apiversion"] == "1"

    resp = client.post("/move", content=json.dumps(_payload()))
    assert resp.status_code == 200
    move = resp.json()["move"]
    assert move in {"up", "down", "left", "right"}
    # Moving Down reverses onto the neck at (5,4); the search masks it, so the
    # chosen move must never be the reversal.
    assert move != "down"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
