"""AlphaZero-style network for snek3, ported from the Albatross paper's
`ResNetConfig11x11` (arXiv:2402.03136, github.com/ymahlau/albatross).

A downsampling ResNet pyramid (channels 32->512) over board feature planes,
GroupNorm + LeakyReLU, with a global-average-pooled 512-d latent feeding 1-layer
linear policy (4 logits) and value (scalar in [-1, 1], tanh) heads.

The value is **egocentric**: it predicts the expected game outcome for the snake
whose perspective produced the observation (+1 win, 0 draw, -1 loss). The search
backs these up through per-node equilibria.

The encoder feeds a 21x21 *head-centered* observation for the standard 11x11
board. The final `AdaptiveAvgPool2d(1)` keeps the heads independent of the exact
spatial size, which is useful for tests and non-standard board sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# Albatross `default_centered_11x11` channel pyramid. Each entry is
# (out_channels, padding); kernel is 3, one residual block per layer,
# GroupNorm(8) throughout. padding=1 keeps the spatial size, padding=0 shrinks
# it by 4 (two valid 3x3 convs), then avgpool collapses the final map.
DEFAULT_LAYER_SPECS: list[tuple[int, int]] = [
    (32, 1),
    (64, 0),
    (96, 1),
    (128, 0),
    (256, 1),
    (384, 1),
    (512, 1),
]

GROUP_NORM_GROUPS = 8  # Albatross uses a fixed 8 groups; all channels divisible by 8.


@dataclass
class NetConfig:
    channels: int = 9  # must match snek.CHANNELS
    height: int = 11
    width: int = 11
    # `filters`/`blocks` are retained for config-serialization/back-compat with
    # callers and resume state; the pyramid backbone is defined by `layer_specs`.
    filters: int = 64
    blocks: int = 6
    layer_specs: list[tuple[int, int]] = field(
        default_factory=lambda: [tuple(s) for s in DEFAULT_LAYER_SPECS]
    )
    policy_actions: int = 4
    # Albatross-style temperature conditioning: when True the network takes a
    # per-sample temperature and sees it as an extra broadcast input plane, so
    # policy/value can depend on the (own or opponent) rationality level.
    temperature_input: bool = False
    temperature_scale: float = 100.0  # plane value = temp / temperature_scale
    # Backbone selection:
    #   "pyramid" — the original Albatross downsampling pyramid (layer_specs).
    #   "grid"    — KataGo-style constant-resolution residual trunk with
    #               global-pooling-bias blocks; recommended for AlphaZero on a
    #               small board + cheap CPU inference (see net_grid below).
    arch: str = "pyramid"
    trunk_channels: int = 96   # grid trunk width
    trunk_blocks: int = 8      # grid trunk depth (residual blocks)
    gpool_every: int = 3       # grid: every Nth block gets a global-pooling bias


class ResNetBlock(nn.Module):
    """Albatross ResNetBlock: two 3x3 convs (GroupNorm + LeakyReLU) with a skip
    connection that projects channels (1x1 conv) and/or pools spatial resolution
    (AvgPool) when the block changes them."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"Only odd kernel sizes supported, got {kernel}")
        res_dec = kernel - 1 - 2 * padding  # per-conv spatial reduction
        total_dec = 2 * res_dec
        self.identity_resolution = (kernel - 1) == 2 * padding
        if not self.identity_resolution:
            self.pool = nn.AvgPool2d(kernel_size=total_dec + 1, stride=1, padding=0)
        self.identity_channel = in_ch == out_ch
        if not self.identity_channel:
            self.downsample = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=padding, bias=False)
        self.norm1 = nn.GroupNorm(GROUP_NORM_GROUPS, out_ch, affine=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel, stride=1, padding=padding, bias=False)
        self.norm2 = nn.GroupNorm(GROUP_NORM_GROUPS, out_ch, affine=True)
        self.act = nn.LeakyReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        skip = x
        if not self.identity_resolution:
            skip = self.pool(skip)
        if not self.identity_channel:
            skip = self.downsample(skip)
        return self.act(skip + y)


def _global_pool(x: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant board summary: concat of channel-wise mean and max
    over the spatial dims -> [B, 2C]. The aggregation that lets the net reason
    about board-global facts (who's longest, food scarcity, snake count) and, by
    pooling over the head-localized per-opponent planes, summarise all opponents
    regardless of how many there are."""
    return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)


class GPoolResBlock(nn.Module):
    """KataGo-style pre-activation residual block. When `gpool`, a global-pooled
    summary of the first conv's output is broadcast back as a per-channel bias,
    giving every cell access to board-global information at negligible cost."""

    def __init__(self, ch: int, gpool: bool = False):
        super().__init__()
        self.gpool = gpool
        self.norm1 = nn.GroupNorm(GROUP_NORM_GROUPS, ch, affine=True)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(GROUP_NORM_GROUPS, ch, affine=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.act = nn.LeakyReLU(inplace=False)
        if gpool:
            self.gp = nn.Linear(2 * ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act(self.norm1(x)))
        if self.gpool:
            y = y + self.gp(_global_pool(y))[:, :, None, None]
        y = self.conv2(self.act(self.norm2(y)))
        return x + y


class AZNet(nn.Module):
    """AlphaZero net. `arch="pyramid"` is the original Albatross downsampling
    backbone; `arch="grid"` is a KataGo-style constant-resolution residual trunk
    with global-pooling-bias blocks (recommended for snek)."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg
        in_channels = c.channels + (1 if c.temperature_input else 0)
        self.arch = c.arch

        if c.arch == "grid":
            self._build_grid(in_channels, c)
        elif c.arch == "pyramid":
            self._build_pyramid(in_channels, c)
        else:
            raise ValueError(f"unknown arch {c.arch!r}")

        self._initialize_weights()

    def _build_pyramid(self, in_channels: int, c: NetConfig) -> None:
        blocks = []
        cur = in_channels
        for out_ch, padding in c.layer_specs:
            blocks.append(ResNetBlock(cur, out_ch, kernel=3, padding=padding))
            cur = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.latent_size = cur
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.policy_head = nn.Linear(self.latent_size, c.policy_actions, bias=True)
        self.value_head = nn.Linear(self.latent_size, 1, bias=True)

    def _build_grid(self, in_channels: int, c: NetConfig) -> None:
        ch = c.trunk_channels
        self.stem = nn.Conv2d(in_channels, ch, 3, padding=1, bias=False)
        self.trunk = nn.ModuleList([
            GPoolResBlock(ch, gpool=((i + 1) % c.gpool_every == 0))
            for i in range(c.trunk_blocks)
        ])
        self.trunk_norm = nn.GroupNorm(GROUP_NORM_GROUPS, ch, affine=True)
        self.act = nn.LeakyReLU(inplace=False)
        # Heads read the permutation-invariant global pool [B, 2*ch].
        self.policy_head = nn.Linear(2 * ch, c.policy_actions, bias=True)
        self.value_head = nn.Sequential(
            nn.Linear(2 * ch, ch), nn.LeakyReLU(inplace=False), nn.Linear(ch, 1)
        )

    def _initialize_weights(self) -> None:
        """Orthogonal init (gain sqrt 2) for conv/linear, ones/zeros for norms."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, temp: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits [B,4], value [B] in [-1,1]).

        When the net is temperature-conditioned, `temp` (shape [B]) is required
        and appended as a broadcast input plane (value `temp / temperature_scale`).
        """
        if self.cfg.temperature_input:
            if temp is None:
                raise ValueError("temperature_input net requires a `temp` tensor")
            b, _, h_, w_ = x.shape
            plane = (temp.to(x.dtype).view(b, 1, 1, 1) / self.cfg.temperature_scale).expand(b, 1, h_, w_)
            x = torch.cat([x, plane], dim=1)

        if self.arch == "grid":
            h = self.stem(x)
            for block in self.trunk:
                h = block(h)
            h = self.act(self.trunk_norm(h))
            pooled = _global_pool(h)  # [B, 2*ch]
            p = self.policy_head(pooled)
            v = torch.tanh(self.value_head(pooled)).squeeze(1)
            return p, v

        h = self.backbone(x)
        latent = torch.flatten(self.avg_pool(h), 1)
        p = self.policy_head(latent)
        v = torch.tanh(self.value_head(latent)).squeeze(1)
        return p, v

    @torch.no_grad()
    def infer(
        self, obs: torch.Tensor, legal_mask: torch.Tensor | None = None,
        temp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper: returns (policy_probs [B,4], value [B]).

        `legal_mask` (optional, [B,4] of 0/1) zeroes out illegal actions before
        the softmax so they receive no probability mass.
        """
        self.eval()
        logits, value = self.forward(obs, temp)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, float("-inf"))
        probs = F.softmax(logits, dim=1)
        # Rows with no legal move (all -inf) become uniform to avoid NaNs.
        probs = torch.nan_to_num(probs, nan=1.0 / logits.shape[1])
        return probs, value


def device_auto() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # TF32 matmul/conv. NOTE: do NOT enable cudnn.benchmark here — the search
        # leaf count (and thus batch size) varies every step, so benchmark mode
        # would re-autotune constantly and get slower. bf16 autocast is the win.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return dev


# Dtype for autocast inference/training forward passes (bf16 is stable and needs
# no gradient scaler). Used by the search bridge and the trainer.
AUTOCAST_DTYPE = torch.bfloat16


def autocast(device: torch.device):
    """Mixed-precision context for a forward pass; a no-op on CPU."""
    return torch.autocast(device.type, dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda"))
