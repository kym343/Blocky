"""Blocky game logic (no UI).

Board coordinates (see 2.png / 3.png):
  - x: 0..14 to the right, y: 0..7 downward.
  - (x + y) even -> down triangle (▽), odd -> up triangle (△).
  - Rows form a hexagon: 9, 11, 13, 15, 15, 13, 11, 9 triangles.

Geometry in lattice units (u = half edge length horizontally, line = row boundary):
  - up   (x, y): vertices (x+1, y), (x, y+1), (x+2, y+1)
  - down (x, y): vertices (x, y), (x+2, y), (x+1, y+1)
"""

import random

ROW_RANGES = [(3, 11), (2, 12), (1, 13), (0, 14), (0, 14), (1, 13), (2, 12), (3, 11)]
CELLS = [(x, y) for y, (a, b) in enumerate(ROW_RANGES) for x in range(a, b + 1)]
CELL_SET = frozenset(CELLS)


def is_up(x, y):
    return (x + y) % 2 == 1


def tri_vertices(x, y):
    """Triangle vertices in lattice units (u, line)."""
    if is_up(x, y):
        return [(x + 1, y), (x, y + 1), (x + 2, y + 1)]
    return [(x, y), (x + 2, y), (x + 1, y + 1)]


def neighbors(x, y):
    yield (x - 1, y)
    yield (x + 1, y)
    yield (x, y + 1) if is_up(x, y) else (x, y - 1)


def _build_lines():
    """Group cells into straight strips in the 3 directions: horizontal and two diagonals."""
    lines = {}
    for x, y in CELLS:
        for key in (("h", y), ("d1", (x - y) // 2), ("d2", (x + y + 1) // 2)):
            lines.setdefault(key, []).append((x, y))
    return {k: frozenset(v) for k, v in lines.items()}


LINES = _build_lines()
CELL_LINES = {c: [k for k, v in LINES.items() if c in v] for c in CELLS}


# ---------------------------------------------------------------- pieces

def normalize(cells):
    """Translate a shape so it touches x=0/y=0 while keeping triangle orientation."""
    min_y = min(y for _, y in cells)
    min_x = min(x for x, _ in cells)
    if (min_x + min_y) % 2:  # translation must have even x+y to keep ▽/△ parity
        min_x -= 1
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def _fixed_polyiamonds(max_size):
    shapes = [{normalize([(0, 0)]), normalize([(1, 0)])}]
    for _ in range(max_size - 1):
        grown = set()
        for shape in shapes[-1]:
            cells = set(shape)
            for c in shape:
                for n in neighbors(*c):
                    if n not in cells:
                        grown.add(normalize(cells | {n}))
        shapes.append(grown)
    return shapes


def _build_pieces():
    pieces = []
    for group in _fixed_polyiamonds(4):  # sizes 1..4: 2 + 3 + 6 + 14 shapes
        pieces.extend(sorted(group, key=lambda s: (len(s), s)))
    # hexagon of 6 triangles around a vertex: △▽△ / ▽△▽
    pieces.append(normalize([(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)]))
    return pieces


PIECES = _build_pieces()


# ---------------------------------------------------------------- game

class Game:
    TRAY_SIZE = 3

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.filled = {}  # cell -> piece index (used for color)
        self.score = 0
        self.tray = []
        self.refill_tray()

    def refill_tray(self):
        self.tray = [self.rng.randrange(len(PIECES)) for _ in range(self.TRAY_SIZE)]

    def cells_at(self, piece, ox, oy):
        return [(x + ox, y + oy) for x, y in PIECES[piece]]

    def can_place(self, piece, ox, oy):
        if (ox + oy) % 2:
            return False
        return all(c in CELL_SET and c not in self.filled for c in self.cells_at(piece, ox, oy))

    def valid_positions(self, piece):
        shape = PIECES[piece]
        w = max(x for x, _ in shape)
        h = max(y for _, y in shape)
        return [(ox, oy)
                for oy in range(0, 8 - h)
                for ox in range(-1, 15 - w)
                if self.can_place(piece, ox, oy)]

    def can_place_anywhere(self, piece):
        return bool(self.valid_positions(piece))

    def lines_cleared_by(self, new_cells):
        """Cells that would disappear if new_cells were filled."""
        occupied = set(self.filled) | set(new_cells)
        keys = {k for c in new_cells for k in CELL_LINES[c]}
        cleared = set()
        for k in keys:
            if LINES[k] <= occupied:
                cleared |= LINES[k]
        return cleared

    def place(self, slot, ox, oy):
        """Place tray[slot] at (ox, oy). Returns (gained score, cleared cells dict) or None."""
        piece = self.tray[slot]
        if piece is None or not self.can_place(piece, ox, oy):
            return None
        cells = self.cells_at(piece, ox, oy)
        cleared = self.lines_cleared_by(cells)
        for c in cells:
            self.filled[c] = piece
        removed = {c: self.filled.pop(c) for c in cleared}
        gained = len(cells) + len(cleared)
        self.score += gained
        self.tray[slot] = None
        if all(p is None for p in self.tray):
            self.refill_tray()
        return gained, removed

    def is_over(self):
        return not any(p is not None and self.can_place_anywhere(p) for p in self.tray)
