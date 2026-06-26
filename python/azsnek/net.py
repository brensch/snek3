"""AlphaZero-style network for snek3: a small ResNet over board feature planes
with a policy head (4 logits) and a value head (scalar in [-1, 1]).

The value is **egocentric**: it predicts the expected game outcome for the snake
whose perspective produced the observation (+1 win, 0 draw, -1 loss). The search
backs these up through per-node equilibria. There is no temperature conditioning
(we assume perfect play; see README).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetConfig:
    channels: int = 9  # must match snek.CHANNELS
    height: int = 11
    width: int = 11
    filters: int = 64
    blocks: int = 6
    policy_actions: int = 4


class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AZNet(nn.Module):
    """Shared trunk + policy and value heads."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg

        self.stem = nn.Sequential(
            nn.Conv2d(c.channels, c.filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(c.filters),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(c.filters) for _ in range(c.blocks)])

        # Policy head: 1x1 conv -> flatten -> linear to 4 logits.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(c.filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * c.height * c.width, c.policy_actions)

        # Value head: 1x1 conv -> flatten -> linear -> scalar in [-1, 1].
        self.value_conv = nn.Sequential(
            nn.Conv2d(c.filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(c.height * c.width, c.filters),
            nn.ReLU(inplace=True),
            nn.Linear(c.filters, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits [B,4], value [B] in [-1,1])."""
        h = self.tower(self.stem(x))
        p = self.policy_fc(self.policy_conv(h).flatten(1))
        v = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(1)
        return p, v

    @torch.no_grad()
    def infer(
        self, obs: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper: returns (policy_probs [B,4], value [B]).

        `legal_mask` (optional, [B,4] of 0/1) zeroes out illegal actions before
        the softmax so they receive no probability mass.
        """
        self.eval()
        logits, value = self.forward(obs)
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
