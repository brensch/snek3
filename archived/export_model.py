#!/usr/bin/env python3
"""Export a training checkpoint to ONNX for the Rust Battlesnake API.

The current AlphaZero trainer stores the policy+value network under ``st["net"]``
in ``runs/<id>/state.pt``. The Rust API then runs decoupled-PUCT MCTS over that
ONNX model, using the policy head as priors and the value head at leaves.

Usage:
    PYTHONPATH=python .venv/bin/python scripts/export_model.py \
        runs/<run-id>/state.pt runs/<run-id>/serve.onnx
"""

from __future__ import annotations

import argparse

import torch

import snek
from azsnek.net import AZNet, NetConfig
from azsnek.train import export_onnx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="path to runs/<id>/state.pt")
    ap.add_argument("out", help="output .onnx path")
    ap.add_argument("--which", default="net",
                    help="checkpoint key to export (default: net; legacy: proxy/response)")
    args = ap.parse_args()

    st = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.which not in st:
        raise SystemExit(f"checkpoint has no '{args.which}' net (keys: {list(st)})")

    cfg_dict = dict(st["net_cfg"])
    # layer_specs round-trips as lists; NetConfig/torch want tuples.
    if "layer_specs" in cfg_dict and cfg_dict["layer_specs"] is not None:
        cfg_dict["layer_specs"] = [tuple(s) for s in cfg_dict["layer_specs"]]
    cfg = NetConfig(**cfg_dict)
    if cfg.channels != snek.CHANNELS:
        raise SystemExit(f"channel mismatch: net_cfg={cfg.channels} snek={snek.CHANNELS}")

    net = AZNet(cfg)
    net.load_state_dict(st[args.which])
    net.eval()

    export_onnx(net, cfg.channels, cfg.width, torch.device("cpu"), args.out)
    print(f"exported {args.which} (gen {st.get('gen', '?')}) -> {args.out}  "
          f"[obs 1x{cfg.channels}x{cfg.height}x{cfg.width}, arch={cfg.arch}]")


if __name__ == "__main__":
    main()
