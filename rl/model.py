"""Afterstate value network: V(board after a move, pieces still in the tray)."""

import numpy as np
import torch
from torch import nn

from rl.fastenv import GRID_H, GRID_W, N_PIECES, STATIC, bits_to_grid, boards_to_bits, tray_vector


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.c2(torch.relu(self.c1(x))))


class ValueNet(nn.Module):
    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.config = {"ch": ch, "blocks": blocks}
        self.register_buffer("static", torch.from_numpy(STATIC))
        self.stem = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.reduce = nn.Conv2d(ch, 8, 1)
        self.tray = nn.Linear(N_PIECES, 64)
        self.head = nn.Sequential(nn.Linear(8 * GRID_H * GRID_W + 64, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, grid, tray):
        """grid: (N, 8, 15) occupancy, tray: (N, 26) piece counts -> (N,) value."""
        x = torch.cat([grid.unsqueeze(1), self.static.expand(len(grid), -1, -1, -1)], 1)
        x = torch.relu(self.stem(x))
        x = torch.relu(self.reduce(self.body(x))).flatten(1)
        t = torch.relu(self.tray(tray))
        return self.head(torch.cat([x, t], 1)).squeeze(1)


def encode(boards, trays):
    """boards: list of ints, trays: list of piece-id lists -> numpy (grid, tray) arrays."""
    grid = bits_to_grid(boards_to_bits(boards))
    tv = np.stack([tray_vector(t) for t in trays]) if trays else np.zeros((0, N_PIECES), np.float32)
    return grid, tv


def save(model, path, extra=None):
    torch.save({"config": model.config, "state": model.state_dict(), "extra": extra or {}}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ValueNet(**ck["config"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck.get("extra", {})
