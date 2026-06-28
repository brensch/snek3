#!/usr/bin/env python3
"""Export the Albatross **proxy** net from a training checkpoint to ONNX.

The Rust serving binary (`crates/snek-server`) runs the proxy net for two things:
the per-opponent temperature MLE (policy head) and the heterogeneous-temperature
best-response search at the leaves (value + policy head). The *response* net is a
not-yet-used distillation; faithful serving only needs the proxy + the search, so
this exports the proxy.

The checkpoint (`runs/<id>/state.pt`) is the full training state; we pull
`net_cfg` (so the architecture matches the weights exactly) and `st["proxy"]`.

Usage:
    PYTHONPATH=python .venv/bin/python scripts/export_model.py \
        runs/albatross-resp0/state.pt model.onnx
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
    ap.add_argument("--which", choices=["proxy", "response"], default="proxy",
                    help="which net to export (default: proxy — what serving uses)")
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
    if not cfg.temperature_input:
        # The server always feeds a temperature; a non-conditioned net can't serve
        # Albatross. Bail loudly rather than export a model the server can't drive.
        raise SystemExit("net is not temperature-conditioned; not an Albatross proxy")

    net = AZNet(cfg)
    net.load_state_dict(st[args.which])
    net.eval()

    export_onnx(net, cfg.channels, cfg.width, torch.device("cpu"), args.out)
    side = 2 * cfg.width - 1
    print(f"exported {args.which} (gen {st.get('gen', '?')}) -> {args.out}  "
          f"[obs 1x{cfg.channels}x{side}x{side} + temp, board {cfg.width}]")


if __name__ == "__main__":
    main()
