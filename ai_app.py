"""Blocky AI app: recommend moves for 3 chosen pieces, or watch the AI play.

Run: python ai_app.py [model.pt]      (default: runs/exp1/model_best.pt)

[추천 모드]
  - 보드 클릭: 칸 채우기/비우기 (실제 게임 보드와 똑같이 맞추기)
  - 오른쪽 블록 목록 클릭: 대기 블록으로 추가 (최대 3개), 아래 대기 블록 클릭: 제거
  - 블록을 고르면 AI가 순서와 위치를 바로 계산해 보드에 1·2·3 번호로 표시
  - ← / → : 전체 보기 ↔ 단계별 보기,  Enter: 추천대로 적용,  C: 보드 비우기,  Backspace: 선택 취소
[관전 모드]
  - Space: 일시정지,  + / - : 속도,  N: 새 게임
Tab: 모드 전환,  Esc: 종료
"""

import os
import random
import sys

import pygame

import blocky as ui
from blocky_core import CELLS, PIECES, tri_vertices
from rl.fastenv import IDX, Env

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H = 980, 820
LEFT_W = 480
PANEL_X = 510
PICK_COLS, PICK_CELL, PICK_EDGE = 6, 74, 24
PICK_Y = 120
MANUAL = (150, 150, 165)          # cells filled by hand in recommend mode
LABEL_BG = (250, 250, 250)
LABEL_FG = (30, 30, 30)
HINT_C = (160, 160, 160)


def mask_cells(mask):
    return [CELLS[i] for i in range(len(CELLS)) if mask >> i & 1]


def cell_center(cell, ox=ui.BOARD_X, oy=ui.BOARD_Y, edge=ui.EDGE):
    th = edge * 3 ** 0.5 / 2
    pts = [(ox + u * edge / 2, oy + line * th) for u, line in tri_vertices(*cell)]
    return sum(p[0] for p in pts) / 3, sum(p[1] for p in pts) / 3


def slot_center(slot):
    return LEFT_W * (slot + 0.5) / 3, ui.TRAY_Y


def tray_origin(slot, piece):
    cx, cy = slot_center(slot)
    bx, by = ui.piece_bounds(piece, ui.EDGE)
    return round(cx - bx), round(cy - by)


def point_in_tri(p, tri):
    (x, y), (a, b, c) = p, tri

    def s(p1, p2):
        return (x - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (y - p2[1])
    d1, d2, d3 = s(a, b), s(b, c), s(c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def board_cell_at(pos):
    th = ui.TRI_H
    for cell in CELLS:
        tri = [(ui.BOARD_X + u * ui.EDGE / 2, ui.BOARD_Y + line * th) for u, line in tri_vertices(*cell)]
        if point_in_tri(pos, tri):
            return cell
    return None


class AIApp:
    def __init__(self, model_path):
        pygame.init()
        pygame.display.set_caption("Blocky AI")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("arialblack,arial", 40, bold=True)
        self.font = pygame.font.SysFont("malgungothic", 18)
        self.font_b = pygame.font.SysFont("malgungothic", 20, bold=True)
        self.font_num = pygame.font.SysFont("arial", 18, bold=True)
        self.font_small = pygame.font.SysFont("arial", 11)
        self.rng = random.Random()
        self.agent, self.model_msg = None, ""
        try:
            from rl.agent import Agent
            self.agent = Agent.from_file(model_path)
            self.model_msg = f"모델: {os.path.relpath(model_path, ROOT)}"
        except Exception as e:  # noqa: BLE001 - show any load error in the UI
            self.model_msg = f"모델을 불러오지 못함: {e}"
        self.mode = "assist"
        self.fading = []
        self.assist_reset()
        self.watch_reset()

    # ================================================================ recommend mode

    def assist_reset(self):
        self.board = 0
        self.colors = {}
        self.sel = []
        self.sel_colors = []
        self.plan = None
        self.view = -1            # -1 = all steps, k = step k
        self.total = 0

    def set_selection(self, sel):
        self.sel = sel
        while len(self.sel_colors) < len(sel):
            used = set(self.sel_colors)
            self.sel_colors.append(self.rng.choice([c for c in ui.PALETTE if c not in used]))
        self.sel_colors = self.sel_colors[:len(sel)]
        self.replan()

    def replan(self):
        self.view = -1
        self.plan = None
        if self.sel and self.agent:
            steps, score, value = self.agent.plan(self.board, self.sel)
            # which selected slot each step uses (handles duplicate pieces)
            free = list(range(len(self.sel)))
            slots = []
            for piece, *_ in steps:
                s = next(i for i in free if self.sel[i] == piece)
                free.remove(s)
                slots.append(s)
            self.plan = {"steps": steps, "slots": slots, "score": score, "value": value}

    def plan_states(self):
        """Board/colors before each step: list of (board, colors) with len(steps)+1 entries."""
        states = [(self.board, dict(self.colors))]
        for (piece, pl, r, nb), slot in zip(self.plan["steps"], self.plan["slots"]):
            b, col = states[-1]
            col = dict(col)
            for c in mask_cells(pl.mask):
                col[c] = self.sel_colors[slot]
            for c in mask_cells((b | pl.mask) & ~nb):
                col.pop(c, None)
            states.append((nb, col))
        return states

    def apply_plan(self):
        if not self.plan or not self.plan["steps"]:
            return
        b, col = self.plan_states()[-1]
        self.board, self.colors = b, col
        self.total += self.plan["score"]
        self.sel, self.sel_colors, self.plan = [], [], None

    def assist_click(self, pos):
        cell = board_cell_at(pos)
        if cell:
            bit = 1 << IDX[cell]
            self.board ^= bit
            if self.board & bit:
                self.colors[cell] = MANUAL
            else:
                self.colors.pop(cell, None)
            self.replan()
            return
        for i, piece in enumerate(self.sel):
            cx, cy = slot_center(i)
            if abs(pos[0] - cx) < LEFT_W / 6 and abs(pos[1] - cy) < 80:
                self.sel_colors.pop(i)
                self.set_selection(self.sel[:i] + self.sel[i + 1:])
                return
        p = self.picker_at(pos)
        if p is not None and len(self.sel) < 3:
            self.set_selection(self.sel + [p])

    def picker_rect(self, p):
        r, c = divmod(p, PICK_COLS)
        return pygame.Rect(PANEL_X + c * PICK_CELL, PICK_Y + r * PICK_CELL, PICK_CELL - 6, PICK_CELL - 6)

    def picker_at(self, pos):
        for p in range(len(PIECES)):
            if self.picker_rect(p).collidepoint(pos):
                return p
        return None

    # ================================================================ watch mode

    def watch_reset(self):
        self.env = Env(self.rng.randrange(1 << 30))
        self.w_colors = {}
        self.w_tray_colors = self.rng.sample(ui.PALETTE, 3)
        self.w_last = []
        self.w_over = False
        self.paused = False
        self.delay = 400
        self.next_t = 0

    def watch_tick(self, now):
        if self.paused or self.w_over or not self.agent or now < self.next_t:
            return
        self.next_t = now + self.delay
        mv = self.agent.choose_many([self.env])[0]
        if mv is None:
            self.w_over = True
            return
        slot, pl, r, _ = mv
        before = self.env.board
        color = self.w_tray_colors[slot]
        self.env.step(slot, pl)
        placed = mask_cells(pl.mask)
        for c in placed:
            self.w_colors[c] = color
        removed = mask_cells((before | pl.mask) & ~self.env.board)
        self.fading += [(c, self.w_colors.pop(c), now) for c in removed]
        self.w_tray_colors[slot] = None
        if all(c is None for c in self.w_tray_colors):
            self.w_tray_colors = self.rng.sample(ui.PALETTE, 3)
        self.w_last = [c for c in placed if c not in removed]
        if self.env.is_over():
            self.w_over = True

    # ================================================================ drawing

    def text(self, s, pos, font=None, color=ui.TEXT_C, center=False):
        img = (font or self.font).render(s, True, color)
        x, y = pos
        if center:
            x -= img.get_width() / 2
        self.screen.blit(img, (x, y))
        return img.get_height()

    def draw_cells(self, board, colors, highlight=(), ghost=None, ghost_color=None, clear=()):
        for cell in CELLS:
            filled = board >> IDX[cell] & 1
            if cell in clear:
                color = ui.lighten(ghost_color, 0.55)
            elif ghost and cell in ghost:
                color = ghost_color
            elif filled:
                color = colors.get(cell, MANUAL)
                if cell in highlight:
                    color = ui.lighten(color, 0.35)
            else:
                color = ui.EMPTY
            ui.fill_poly(self.screen, color, ui.tri_points(*cell, ui.BOARD_X, ui.BOARD_Y, ui.EDGE))
        now = pygame.time.get_ticks()
        alive = []
        for cell, color, t0 in self.fading:
            k = (now - t0) / ui.CLEAR_ANIM_MS
            if k < 1:
                alive.append((cell, color, t0))
                pts = ui.tri_points(*cell, ui.BOARD_X, ui.BOARD_Y, ui.EDGE, scale=1 - k)
                ui.fill_poly(self.screen, ui.lighten(color, 0.5 * (1 - k)), pts)
        self.fading = alive

    def draw_label(self, cells, n):
        pts = [cell_center(c) for c in cells]
        x = sum(p[0] for p in pts) / len(pts)
        y = sum(p[1] for p in pts) / len(pts)
        pygame.draw.circle(self.screen, LABEL_BG, (x, y), 14)
        pygame.draw.circle(self.screen, LABEL_FG, (x, y), 14, 2)
        self.text(str(n), (x, y - 11), self.font_num, LABEL_FG, center=True)

    def draw_tray(self, pieces, colors):
        for i, piece in enumerate(pieces):
            if piece is None:
                continue
            ui.draw_piece(self.screen, piece, *tray_origin(i, piece), ui.EDGE, colors[i])

    def draw_assist(self):
        self.text(str(self.total), (LEFT_W * 0.25, 40), self.font_big, ui.SCORE_C, center=True)
        if self.plan:
            self.text(f"+{self.plan['score']}", (LEFT_W * 0.75, 40), self.font_big, ui.BEST_C, center=True)
        self.text("누적 점수", (LEFT_W * 0.25, 95), color=HINT_C, center=True)
        self.text("이번 추천 점수", (LEFT_W * 0.75, 95), color=HINT_C, center=True)

        steps = self.plan["steps"] if self.plan else []
        if not steps:
            self.draw_cells(self.board, self.colors)
        elif self.view < 0:
            # overview: every step drawn on the current board with its number
            self.draw_cells(self.board, self.colors)
            for k, ((piece, pl, r, nb), slot) in enumerate(zip(steps, self.plan["slots"])):
                cells = mask_cells(pl.mask)
                for c in cells:
                    ui.fill_poly(self.screen, self.sel_colors[slot],
                                 ui.tri_points(*c, ui.BOARD_X, ui.BOARD_Y, ui.EDGE))
            for k, (piece, pl, r, nb) in enumerate(steps):
                self.draw_label(mask_cells(pl.mask), k + 1)
        else:
            states = self.plan_states()
            b, col = states[self.view]
            piece, pl, r, nb = steps[self.view]
            cells = mask_cells(pl.mask)
            clear = set(mask_cells((b | pl.mask) & ~nb))
            self.draw_cells(b, col, ghost=set(cells), ghost_color=self.sel_colors[self.plan["slots"][self.view]],
                            clear=clear)
            self.draw_label(cells, self.view + 1)

        self.draw_tray(self.sel, self.sel_colors)
        if self.plan:
            for k, slot in enumerate(self.plan["slots"]):
                cx, cy = slot_center(slot)
                pygame.draw.circle(self.screen, LABEL_BG, (cx, cy + 92), 13)
                self.text(str(k + 1), (cx, cy + 81), self.font_num, LABEL_FG, center=True)

        # ---- right panel
        x = PANEL_X
        self.text("블록 선택 (클릭, 최대 3개)", (x, 85), self.font_b)
        for p in range(len(PIECES)):
            rect = self.picker_rect(p)
            pygame.draw.rect(self.screen, (44, 44, 44), rect, border_radius=8)
            bx, by = ui.piece_bounds(p, PICK_EDGE)
            color = (200, 200, 200) if len(self.sel) < 3 else (110, 110, 110)
            ui.draw_piece(self.screen, p, round(rect.centerx - bx), round(rect.centery - by), PICK_EDGE, color,
                          gap=1.2)
            self.text(str(p), (rect.x + 4, rect.y + 1), self.font_small, (115, 115, 115))
        y = PICK_Y + 5 * PICK_CELL + 10
        if self.plan:
            steps = self.plan["steps"]
            if len(steps) < len(self.sel):
                y += self.text(f"모든 블록을 놓을 수 없음 ({len(steps)}개만 가능)", (x, y), color=ui.BEST_C) + 4
            for k, (piece, pl, r, nb) in enumerate(steps):
                b = self.plan_states()[k][0]
                lines = self.count_lines(b, pl, nb)
                extra = f", {lines}줄 삭제" if lines else ""
                mark = "▶ " if self.view == k else "   "
                y += self.text(f"{mark}{k + 1}. 블록 {piece}  +{r}점{extra}", (x, y), self.font_b) + 4
            view = "전체" if self.view < 0 else f"{self.view + 1}단계"
            y += self.text(f"보기: {view}   (← → 로 전환)", (x, y + 6), color=HINT_C) + 12
        elif not self.agent:
            y += self.text(self.model_msg, (x, y), color=ui.BEST_C) + 8
        for line in ("보드 클릭: 칸 채우기/비우기 (실제 게임과 맞추기)",
                     "아래 대기 블록 클릭: 선택 취소",
                     "Enter: 추천대로 적용   C: 보드 비우기",
                     "Backspace: 선택 모두 취소   Tab: 관전 모드"):
            y += self.text(line, (x, y), color=HINT_C) + 2

    @staticmethod
    def count_lines(before, pl, after):
        from rl.fastenv import LINE_MASKS
        b = before | pl.mask
        return sum(1 for L in LINE_MASKS if b & L == L and not after & L)

    def draw_watch(self):
        self.text(str(self.env.score), (LEFT_W * 0.25, 40), self.font_big, ui.SCORE_C, center=True)
        self.text(str(self.env.moves), (LEFT_W * 0.75, 40), self.font_big, ui.BEST_C, center=True)
        self.text("점수", (LEFT_W * 0.25, 95), color=HINT_C, center=True)
        self.text("둔 수", (LEFT_W * 0.75, 95), color=HINT_C, center=True)
        self.draw_cells(self.env.board, self.w_colors, highlight=set(self.w_last))
        self.draw_tray(self.env.tray, self.w_tray_colors)
        x, y = PANEL_X, 85
        y += self.text("AI 관전 모드", (x, y), self.font_b) + 10
        state = "게임 오버" if self.w_over else ("일시정지" if self.paused else "진행 중")
        for line in (f"상태: {state}", f"속도: {self.delay} ms / 수", self.model_msg, "",
                     "Space: 일시정지   + / - : 속도", "N: 새 게임   Tab: 추천 모드"):
            y += self.text(line, (x, y), color=ui.TEXT_C if line.startswith("상태") else HINT_C) + 4

    def draw(self):
        self.screen.fill(ui.BG)
        pygame.draw.line(self.screen, (50, 50, 50), (LEFT_W + 10, 20), (LEFT_W + 10, H - 20), 1)
        if self.mode == "assist":
            self.draw_assist()
        else:
            self.draw_watch()
        pygame.display.flip()

    # ================================================================ loop

    def on_key(self, key):
        if key == pygame.K_TAB:
            self.mode = "watch" if self.mode == "assist" else "assist"
            self.fading = []
        elif self.mode == "assist":
            n = len(self.plan["steps"]) if self.plan else 0
            if key == pygame.K_RIGHT and n:
                self.view = self.view + 1 if self.view + 1 < n else -1
            elif key == pygame.K_LEFT and n:
                self.view = self.view - 1 if self.view >= 0 else n - 1
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.apply_plan()
            elif key == pygame.K_c:
                self.assist_reset()
            elif key == pygame.K_BACKSPACE:
                self.sel_colors = []
                self.set_selection([])
        else:
            if key == pygame.K_SPACE:
                self.paused = not self.paused
            elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.delay = max(20, int(self.delay * 0.6))
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.delay = min(3000, int(self.delay / 0.6))
            elif key == pygame.K_n:
                self.watch_reset()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN:
                    self.on_key(e.key)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.mode == "assist":
                    self.assist_click(e.pos)
            if self.mode == "watch":
                self.watch_tick(pygame.time.get_ticks())
            self.draw()
            self.clock.tick(60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "exp1", "model_best.pt")
    AIApp(path).run()
