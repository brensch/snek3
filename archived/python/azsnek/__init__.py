"""azsnek: network, search bridge, self-play, and training for the snek3 bot."""

import os as _os

# Set before torch initializes its CUDA allocator: expandable segments reduce
# fragmentation and avoid spilling into (very slow) shared GPU memory on WSL2.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .net import AZNet, NetConfig, device_auto

__all__ = ["AZNet", "NetConfig", "device_auto"]
