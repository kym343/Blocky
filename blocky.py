"""Blocky - triangle block puzzle (pygame UI).

Drag a piece from the tray onto the hexagon board.
Keys: R = restart, Esc = quit.
"""

import json
import math
import os
import random
import sys

import pygame
import pygame.gfxdraw

from blocky_core import CELLS, PIECES, Game, tri_vertices

W, H = 480, 820
EDGE = 52                         # triangle edge length on the board (px)
TRI_H = EDGE * math.sqrt(3) / 2
BOARD_X = (W - 8 * EDGE) / 2      # board spans 16 half-edges = 8 edges
BOARD_Y = 190
TRAY_Y = 690
SNAP_DIST = EDGE * 0.7
CLEAR_ANIM_MS = 300

GAP = 0.018                   # inward offset of each triangle, as a fraction of edge
ROUND = 0.05                      # corner radius, as a fraction of edge
ARC_STEPS = 6

BG = (30, 30, 30)
EMPTY = (62, 62, 62)
SCORE_C = (110, 214, 190)
BEST_C = (224, 110, 90)
CROWN_C = (245, 196, 70)
TEXT_C = (230, 230, 230)
DISABLED = (90, 90, 90)
PALETTE = [(232, 112, 130), (150, 222, 100), (110, 200, 150), (245, 196, 70),
           (110, 170, 240), (190, 130, 230), (240, 150, 80), (90, 210, 220),
           (240, 120, 200), (200, 220, 90)]

HIGHSCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highscore.json")


def load_best():
    try:
        with open(HIGHSCORE_FILE, encoding="utf-8") as f:
            return int(json.load(f)["best"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def save_best(best):
    try:
        with open(HIGHSCORE_FILE, "w", encoding="utf-8") as f:
            json.dump({"best": best}, f)
    except OSError:
        pass


def tri_points(x, y, ox, oy, edge, scale=1.0, gap=None):
    """Rounded, inset pixel polygon of triangle (x, y) with lattice origin at (ox, oy).

    gap: inward offset in px (default GAP * edge)."""
    th = edge * math.sqrt(3) / 2
    verts = [(ox + u * edge / 2, oy + line * th) for u, line in tri_vertices(x, y)]
    cx = sum(p[0] for p in verts) / 3
    cy = sum(p[1] for p in verts) / 3
    inradius = edge / (2 * math.sqrt(3))
    gap = GAP * edge if gap is None else gap
    r = ROUND * edge
    # corner arc centers: triangle offset inward by gap + r
    k = (inradius - gap - r) / inradius
    verts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    pts = []
    for vx, vy in verts:
        ax, ay = cx + (vx - cx) * k, cy + (vy - cy) * k
        a = math.atan2(vy - cy, vx - cx)
        for i in range(ARC_STEPS + 1):
            t = a - math.pi / 3 + (2 * math.pi / 3) * i / ARC_STEPS
            pts.append((ax + r * math.cos(t), ay + r * math.sin(t)))
    if scale != 1.0:
        pts = [(cx + (px - cx) * scale, cy + (py - cy) * scale) for px, py in pts]
    return pts


def fill_poly(surf, color, pts):
    pts = [(round(px), round(py)) for px, py in pts]
    pygame.gfxdraw.filled_polygon(surf, pts, color)
    pygame.gfxdraw.aapolygon(surf, pts, color)


def piece_bounds(piece, edge):
    """Pixel center of a piece relative to its lattice origin."""
    cells = PIECES[piece]
    u0 = min(x for x, _ in cells)
    u1 = max(x for x, _ in cells) + 2
    l0 = min(y for _, y in cells)
    l1 = max(y for _, y in cells) + 1
    th = edge * math.sqrt(3) / 2
    return (u0 + u1) / 2 * edge / 2, (l0 + l1) / 2 * th


def draw_piece(surf, piece, ox, oy, edge, color, gap=None):
    for x, y in PIECES[piece]:
        fill_poly(surf, color, tri_points(x, y, ox, oy, edge, gap=gap))


def lighten(c, k):
    return tuple(min(255, int(v + (255 - v) * k)) for v in c)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Blocky")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("arialblack,arial", 40, bold=True)
        self.font_mid = pygame.font.SysFont("arial", 26, bold=True)
        self.font_small = pygame.font.SysFont("arial", 18)
        self.rng = random.Random()
        self.best = load_best()
        self.new_game()

    def new_game(self):
        self.game = Game()
        self.cell_colors = {}     # board cell -> color
        self.new_tray_colors()
        self.drag = None          # (slot, grab_dx, grab_dy): cursor offset from piece origin
        self.mouse = (0, 0)
        self.fading = []          # (cell, color, start_ms)
        self.popups = []          # (text, x, y, start_ms)
        self.over = False

    def new_tray_colors(self):
        self.tray_colors = self.rng.sample(PALETTE, len(self.game.tray))

    # ------------------------------------------------------------ layout

    def slot_center(self, slot):
        return W * (slot + 0.5) / 3, TRAY_Y

    def slot_at(self, pos):
        for i in range(3):
            cx, cy = self.slot_center(i)
            if abs(pos[0] - cx) < W / 6 and abs(pos[1] - cy) < 80:
                return i
        return None

    def drag_origin(self):
        slot, gdx, gdy = self.drag
        return self.mouse[0] - gdx, self.mouse[1] - gdy

    def tray_origin(self, slot):
        """Pixel lattice origin of the tray piece in `slot` (drawn at board size, centered)."""
        cx, cy = self.slot_center(slot)
        bx, by = piece_bounds(self.game.tray[slot], EDGE)
        # integer origin keeps row lines on whole pixels (like the board); a half-pixel
        # offset makes the thin horizontal seams round away and the triangles merge
        return round(cx - bx), round(cy - by)

    def snap(self):
        """Nearest valid board placement for the dragged piece, or None."""
        piece = self.game.tray[self.drag[0]]
        px, py = self.drag_origin()
        fu = (px - BOARD_X) / (EDGE / 2)
        fl = (py - BOARD_Y) / TRI_H
        best, best_d = None, SNAP_DIST
        for oy in range(round(fl) - 1, round(fl) + 2):
            for ox in range(round(fu) - 2, round(fu) + 3):
                if not self.game.can_place(piece, ox, oy):
                    continue
                d = math.hypot(BOARD_X + ox * EDGE / 2 - px, BOARD_Y + oy * TRI_H - py)
                if d < best_d:
                    best, best_d = (ox, oy), d
        return best

    # ------------------------------------------------------------ events

    def on_down(self, pos):
        if self.over:
            self.new_game()
            return
        slot = self.slot_at(pos)
        if slot is None or self.game.tray[slot] is None:
            return
        # grab the piece where it was clicked so it doesn't jump
        ox, oy = self.tray_origin(slot)
        self.drag = (slot, pos[0] - ox, pos[1] - oy)
        self.mouse = pos

    def on_up(self):
        if not self.drag:
            return
        target = self.snap()
        slot = self.drag[0]
        self.drag = None
        if target is None:
            return
        color = self.tray_colors[slot]
        cells = self.game.cells_at(self.game.tray[slot], *target)
        gained, removed = self.game.place(slot, *target)
        for c in cells:
            self.cell_colors[c] = color
        now = pygame.time.get_ticks()
        self.fading += [(c, self.cell_colors.pop(c), now) for c in removed]
        self.tray_colors[slot] = None
        if all(c is None for c in self.tray_colors):  # game refilled the tray
            self.new_tray_colors()
        bx = BOARD_X + target[0] * EDGE / 2
        by = BOARD_Y + target[1] * TRI_H
        self.popups.append((f"+{gained}", bx, by, now))
        if self.game.score > self.best:
            self.best = self.game.score
            save_best(self.best)
        self.over = self.game.is_over()

    # ------------------------------------------------------------ drawing

    def draw_header(self):
        s = self.font_big.render(str(self.game.score), True, SCORE_C)
        self.screen.blit(s, (W * 0.25 - s.get_width() / 2, 40))
        b = self.font_big.render(str(self.best), True, BEST_C)
        self.screen.blit(b, (W * 0.75 - b.get_width() / 2, 40))
        # crown
        cx, cy = W / 2, 62
        pts = [(cx - 22, cy + 14), (cx - 24, cy - 12), (cx - 10, cy), (cx, cy - 18),
               (cx + 10, cy), (cx + 24, cy - 12), (cx + 22, cy + 14)]
        pygame.draw.polygon(self.screen, CROWN_C, pts, 4)

    def draw_board(self, preview_cells, preview_color, clear_cells):
        now = pygame.time.get_ticks()
        for cell in CELLS:
            if cell in clear_cells:
                color = lighten(preview_color, 0.45)
            elif cell in self.cell_colors:
                color = self.cell_colors[cell]
            elif cell in preview_cells:
                color = tuple(int(e * 0.45 + p * 0.55) for e, p in zip(EMPTY, preview_color))
            else:
                color = EMPTY
            fill_poly(self.screen, color, tri_points(*cell, BOARD_X, BOARD_Y, EDGE))

        alive = []
        for cell, color, t0 in self.fading:
            k = (now - t0) / CLEAR_ANIM_MS
            if k >= 1:
                continue
            alive.append((cell, color, t0))
            pts = tri_points(*cell, BOARD_X, BOARD_Y, EDGE, scale=1 - k)
            fill_poly(self.screen, lighten(color, 0.5 * (1 - k)), pts)
        self.fading = alive

    def draw_tray(self):
        for i, piece in enumerate(self.game.tray):
            if piece is None or (self.drag and self.drag[0] == i):
                continue
            color = self.tray_colors[i]
            if not self.game.can_place_anywhere(piece):
                color = DISABLED
            draw_piece(self.screen, piece, *self.tray_origin(i), EDGE, color)

    def draw_popups(self):
        now = pygame.time.get_ticks()
        alive = []
        for text, x, y, t0 in self.popups:
            k = (now - t0) / 800
            if k >= 1:
                continue
            alive.append((text, x, y, t0))
            s = self.font_mid.render(text, True, TEXT_C)
            s.set_alpha(int(255 * (1 - k)))
            self.screen.blit(s, (x, y - 40 * k))
        self.popups = alive

    def draw_game_over(self):
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 170))
        self.screen.blit(veil, (0, 0))
        for text, font, y in (("GAME OVER", self.font_big, H / 2 - 60),
                              (f"SCORE  {self.game.score}", self.font_mid, H / 2),
                              ("click to play again", self.font_small, H / 2 + 50)):
            s = font.render(text, True, TEXT_C)
            self.screen.blit(s, (W / 2 - s.get_width() / 2, y))

    def draw(self):
        self.screen.fill(BG)
        self.draw_header()

        preview, clear, color = [], set(), EMPTY
        if self.drag:
            piece = self.game.tray[self.drag[0]]
            color = self.tray_colors[self.drag[0]]
            target = self.snap()
            if target:
                preview = self.game.cells_at(piece, *target)
                clear = self.game.lines_cleared_by(preview)
        self.draw_board(set(preview), color, clear)
        self.draw_tray()
        if self.drag:
            draw_piece(self.screen, self.game.tray[self.drag[0]], *self.drag_origin(), EDGE, color)
        self.draw_popups()
        if self.over:
            self.draw_game_over()
        pygame.display.flip()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    self.new_game()
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.on_down(e.pos)
                elif e.type == pygame.MOUSEMOTION:
                    self.mouse = e.pos
                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    self.mouse = e.pos
                    self.on_up()
            self.draw()
            self.clock.tick(60)


if __name__ == "__main__":
    App().run()
    sys.exit()
