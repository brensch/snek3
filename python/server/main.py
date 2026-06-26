"""Battlesnake HTTP server.

On each `/move`, parse the request into a single-game search batch, run one
fixed-depth equilibrium search (a single batched net forward pass), and return
the controlled snake's equilibrium-policy argmax.

Environment:
    SNEK_CKPT     path to a saved net state_dict (optional; random net if unset)
    SNEK_FILTERS  trunk width  (must match the checkpoint; default 64)
    SNEK_BLOCKS   residual blocks (must match the checkpoint; default 6)
    SNEK_DEPTH    search depth (default 2)
    SNEK_TAU      equilibrium temperature (default 6.0)
    SNEK_ITERS    SFP iterations (default 120)

Run:
    SNEK_CKPT=checkpoints/best.pt uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

import numpy as np
import snek
import torch
from fastapi import FastAPI, Request

from azsnek.net import AZNet, NetConfig, device_auto
from azsnek.search import run_search

MOVES = ["up", "down", "left", "right"]

_device = device_auto()
_net = AZNet(
    NetConfig(
        channels=snek.CHANNELS,
        filters=int(os.environ.get("SNEK_FILTERS", "64")),
        blocks=int(os.environ.get("SNEK_BLOCKS", "6")),
    )
).to(_device)
_ckpt = os.environ.get("SNEK_CKPT")
if _ckpt:
    _net.load_state_dict(torch.load(_ckpt, map_location=_device))
_net.eval()

_DEPTH = int(os.environ.get("SNEK_DEPTH", "2"))
_TAU = float(os.environ.get("SNEK_TAU", "6.0"))
_ITERS = int(os.environ.get("SNEK_ITERS", "120"))

app = FastAPI()


@app.get("/")
def info():
    return {
        "apiversion": "1",
        "author": "brensch",
        "color": "#3366ff",
        "head": "default",
        "tail": "default",
        "version": "0.1.0",
    }


@app.post("/start")
async def start(_req: Request):
    return {}


@app.post("/end")
async def end(_req: Request):
    return {}


@app.post("/move")
async def move(req: Request):
    body = (await req.body()).decode("utf-8")
    batch, me = snek.GameBatch.from_request(body)
    policy = run_search(batch, _net, _device, _DEPTH, _TAU, _ITERS)
    probs = policy[0, me]
    if float(probs.sum()) <= 1e-8:
        # No legal/searched move (already lost): return any move.
        return {"move": "up"}
    return {"move": MOVES[int(np.argmax(probs))]}
