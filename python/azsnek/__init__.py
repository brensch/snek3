"""azsnek: network, search bridge, self-play, and training for the snek3 bot."""

from .net import AZNet, NetConfig, device_auto

__all__ = ["AZNet", "NetConfig", "device_auto"]
