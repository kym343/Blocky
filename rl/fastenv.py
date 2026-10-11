"""Fast bitboard version of the Blocky rules for RL.

The board is a 96-bit Python int (bit i = CELLS[i] filled). All placements of all
pieces are precomputed, so a move is a couple of integer AND/OR operations.
Rules are identical to blocky_core.Game (verified in rl/test_fastenv.py).
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blocky_core import CELLS, CELL_SET, LINES, PIECES  # noqa: E402

N_CELLS = len(CELLS)                  # 96
N_PIECES = len(PIECES)                # 26
TRAY = 3
IDX = {c: i for i, c in enumerate(CELLS)}
FULL = (1 << N_CELLS) - 1


def cells_mask(cells):
    m = 0
    for c in cells:
        m |= 1 << IDX[c]
    return m


LINE_MASKS = [cells_mask(v) for v in LINES.values()]
_CELL_LINE_MASKS = [[L for L in LINE_MASKS if L >> i & 1] for i in range(N_CELLS)]


class Placement:
    __slots__ = ("piece", "ox", "oy", "mask", "lines", "size")

    def __init__(self, piece, ox, oy, cells):
        self.piece, self.ox, self.oy = piece, ox, oy
        self.mask = cells_mask(cells)
        self.size = len(cells)
        lines = {L for c in cells for L in _CELL_LINE_MASKS[IDX[c]]}
        self.lines = tuple(lines)


def _build_placements():
    out = []
    for p, shape in enumerate(PIECES):
        lst = []
        for oy in range(-1, 9):
            for ox in range(-2, 16):
                if (ox + oy) % 2:
                    continue
                cells = [(x + ox, y + oy) for x, y in shape]
                if all(c in CELL_SET for c in cells):
                    lst.append(Placement(p, ox, oy, cells))
        out.append(lst)
    return out


PLACEMENTS = _build_placements()


def apply(board, pl):
    """Return (new_board, reward) for placing pl on board (assumed legal)."""
    b = board | pl.mask
    cleared = 0
    for L in pl.lines:
        if b & L == L:
            cleared |= L
    return b & ~cleared, pl.size + cleared.bit_count()


def legal(board, piece):
    return [pl for pl in PLACEMENTS[piece] if not board & pl.mask]


def has_move(board, piece):
    return any(not board & pl.mask for pl in PLACEMENTS[piece])


class Env:
    """One game. tray holds piece ids or None (used)."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.board = 0
        self.score = 0
        self.moves = 0
        self.new_tray()
        return self

    def new_tray(self):
        self.tray = [self.rng.randrange(N_PIECES) for _ in range(TRAY)]

    def remaining_after(self, slot):
        """Pieces left in the tray after using `slot` (empty -> tray will be refilled)."""
        return [p for i, p in enumerate(self.tray) if i != slot and p is not None]

    def candidates(self):
        """All distinct moves: list of (slot, placement, new_board, reward)."""
        out = []
        seen = set()
        for slot, piece in enumerate(self.tray):
            if piece is None or piece in seen:
                continue
            seen.add(piece)
            for pl in PLACEMENTS[piece]:
                if not self.board & pl.mask:
                    nb, r = apply(self.board, pl)
                    out.append((slot, pl, nb, r))
        return out

    def step(self, slot, pl):
        self.board, r = apply(self.board, pl)
        self.score += r
        self.moves += 1
        self.tray[slot] = None
        if all(p is None for p in self.tray):
            self.new_tray()
        return r

    def is_over(self):
        return not any(p is not None and has_move(self.board, p) for p in self.tray)


# ---------------------------------------------------------------- features

GRID_H, GRID_W = 8, 15
_GY = np.array([y for _, y in CELLS])
_GX = np.array([x for x, _ in CELLS])
# static channels: inside-board mask and triangle orientation (1 = up)
STATIC = np.zeros((2, GRID_H, GRID_W), np.float32)
STATIC[0, _GY, _GX] = 1
for x, y in CELLS:
    STATIC[1, y, x] = (x + y) % 2


def boards_to_bits(boards):
    """List of board ints -> (N, 96) uint8 array."""
    raw = b"".join(b.to_bytes(12, "little") for b in boards)
    arr = np.frombuffer(raw, np.uint8).reshape(len(boards), 12)
    return np.unpackbits(arr, axis=1, bitorder="little")[:, :N_CELLS]


def bits_to_grid(bits):
    """(N, 96) -> (N, 8, 15) float32 occupancy grid."""
    g = np.zeros((len(bits), GRID_H, GRID_W), np.float32)
    g[:, _GY, _GX] = bits
    return g


def tray_vector(pieces):
    v = np.zeros(N_PIECES, np.float32)
    for p in pieces:
        v[p] += 1
    return v
