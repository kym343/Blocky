/* Blocky AI web demo: game rules + afterstate value net (ONNX) + 3-piece planner.
   Game data (cells, pieces, lines, placements) is exported from Python (rl/export_web.py),
   so the rules are exactly those used for training. */
"use strict";

// ------------------------------------------------------------------ constants
const W = 480;
const EDGE = 52, TRI_H = EDGE * Math.sqrt(3) / 2;
const BOARD_X = (W - 8 * EDGE) / 2;
// canvas layouts: PC (tall header) and mobile (compact, so board + piece picker fit on one phone screen)
const LAYOUTS = {
  pc:     { H: 780, BOARD_Y: 150, TRAY_Y: 640, SCORE_Y: 48, LABEL_Y: 88, SCORE_SIZE: 38, BADGE_DY: 96, BUSY_Y: 758 },
  mobile: { H: 600, BOARD_Y: 66,  TRAY_Y: 500, SCORE_Y: 22, LABEL_Y: 48, SCORE_SIZE: 26, BADGE_DY: 82, BUSY_Y: null },
};
let H, BOARD_Y, TRAY_Y, SCORE_Y, LABEL_Y, SCORE_SIZE, BADGE_DY, BUSY_Y, layoutName;
const GAP = 0.018, ROUND = 0.05, ARC_STEPS = 6;
const BG = "#1e1e1e", EMPTY = "#3e3e3e", MANUAL = "#9696a5", TEXT = "#e6e6e6", MUTED = "#9a9a9a";
const SCORE_C = "#6ed6be", BEST_C = "#e06e5a";
const PALETTE = ["#e87082", "#96de64", "#6ec896", "#f5c446", "#6eaaf0", "#be82e6", "#f09650",
                 "#5ad2dc", "#f078c8", "#c8dc5a"];
const CHUNK = 2048;
const params = new URLSearchParams(location.search);

let G, RS, session, engineName = "";
const CX = [], CY = [];

// ------------------------------------------------------------------ rules
function isUp(x, y) { return (x + y) % 2 === 1; }
function triVerts(x, y) {
  return isUp(x, y) ? [[x + 1, y], [x, y + 1], [x + 2, y + 1]] : [[x, y], [x + 2, y], [x + 1, y + 1]];
}

/** Place pl on board -> {nb, r, cleared, nLines}. Same rules as rl/fastenv.apply. */
function applyMove(board, pl) {
  const nb = board.slice();
  for (const i of pl.cells) nb[i] = 1;
  const cleared = new Set();
  let nLines = 0;
  for (const l of pl.lines) {
    const cells = G.lines[l];
    if (cells.every(i => nb[i])) { nLines++; for (const i of cells) cleared.add(i); }
  }
  for (const i of cleared) nb[i] = 0;
  return { nb, r: pl.cells.length + cleared.size, cleared: [...cleared], nLines };
}
function fits(board, pl) { for (const i of pl.cells) if (board[i]) return false; return true; }
function hasMove(board, piece) { return G.placements[piece].some(pl => fits(board, pl)); }

// ------------------------------------------------------------------ model
/** Run the model on reference inputs; a backend that disagrees with Python is not used. */
async function verifySession(s, check) {
  const v = await runValues(s, check.boards.map(b => Uint8Array.from(b, ch => +ch)), check.trays);
  return Math.max(...check.values.map((e, i) => Math.abs(e - v[i])));
}

async function initModel(check) {
  const base = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.wasmPaths = base;
  // threads need cross-origin isolation (not available on GitHub Pages); proxy runs inference in a worker
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const want = params.get("ep");
  ort.env.wasm.proxy = want !== "webgpu" && !params.has("noproxy");
  // the lite model is small: plain WebAssembly beats WebGPU's per-call overhead, so GPU is opt-in (?ep=webgpu)
  const eps = want && want !== "wasm" ? [want, "wasm"] : ["wasm"];   // always fall back to wasm
  let lastErr;
  for (const ep of eps) {
    try {
      const s = await ort.InferenceSession.create("web/model.onnx", { executionProviders: [ep] });
      // correctness check: e.g. WebGPU produced wrong values (diff ~15) for this model in ort 1.20.1
      const diff = await verifySession(s, check);
      if (!(diff < 1e-2)) throw new Error(`${ep} 결과 불일치 (차이 ${diff.toFixed(3)})`);
      session = s;
      engineName = ep === "webgpu" ? "GPU (WebGPU)"
        : `CPU (WebAssembly, ${ort.env.wasm.numThreads}스레드${ort.env.wasm.proxy ? ", 워커" : ""})`;
      return;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

/** Afterstate values for boards (Uint8Array[96]) with remaining tray pieces. */
async function values(boards, trays) { return runValues(session, boards, trays); }

async function runValues(sess, boards, trays) {
  const n = boards.length, out = new Float32Array(n);
  for (let s = 0; s < n; s += CHUNK) {
    const m = Math.min(CHUNK, n - s);
    const g = new Float32Array(m * 120), t = new Float32Array(m * 26);
    for (let k = 0; k < m; k++) {
      const b = boards[s + k];
      for (let i = 0; i < 96; i++) if (b[i]) g[k * 120 + CY[i] * 15 + CX[i]] = 1;
      for (const p of trays[s + k]) t[k * 26 + p] += 1;
    }
    const r = await sess.run({ grid: new ort.Tensor("float32", g, [m, 8, 15]),
                                  tray: new ort.Tensor("float32", t, [m, 26]) });
    out.set(r.value.data, s);
  }
  return out;
}

/** Best order + positions for `pieces` on `board` (beam search over afterstate values). */
async function plan(board, pieces, beam) {
  let states = [{ score: 0, board, rem: pieces.slice(), steps: [] }];
  let best = states[0];
  for (let d = 0; d < pieces.length; d++) {
    const nxt = [];
    for (const st of states) {
      const seen = new Set();
      for (const p of st.rem) {
        if (seen.has(p)) continue;
        seen.add(p);
        const left = st.rem.slice();
        left.splice(left.indexOf(p), 1);
        for (const pl of G.placements[p]) {
          if (!fits(st.board, pl)) continue;
          const res = applyMove(st.board, pl);
          nxt.push({ score: st.score + res.r, board: res.nb, rem: left,
                     steps: [...st.steps, { piece: p, pl, before: st.board, ...res }] });
        }
      }
    }
    if (!nxt.length) break;
    const v = await values(nxt.map(s => s.board), nxt.map(s => s.rem));
    const order = nxt.map((s, i) => [s.score * RS + v[i], i]).sort((a, b) => b[0] - a[0]);
    states = order.slice(0, beam).map(o => nxt[o[1]]);
    best = states[0];
  }
  return best;
}

/** One greedy move for the watch mode: argmax r + V(afterstate). */
async function chooseMove(board, tray) {
  const cands = [];
  const seen = new Set();
  tray.forEach((p, slot) => {
    if (p === null || seen.has(p)) return;
    seen.add(p);
    const rem = tray.filter((q, i) => i !== slot && q !== null);
    for (const pl of G.placements[p]) if (fits(board, pl)) cands.push({ slot, pl, rem, ...applyMove(board, pl) });
  });
  if (!cands.length) return null;
  const v = await values(cands.map(c => c.nb), cands.map(c => c.rem));
  let bi = 0, bq = -Infinity;
  cands.forEach((c, i) => { const q = c.r * RS + v[i]; if (q > bq) { bq = q; bi = i; } });
  return cands[bi];
}

// ------------------------------------------------------------------ drawing
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");

function triPoints(x, y, ox, oy, edge, scale = 1, gap = null) {
  const th = edge * Math.sqrt(3) / 2;
  let v = triVerts(x, y).map(([u, l]) => [ox + u * edge / 2, oy + l * th]);
  const cx = (v[0][0] + v[1][0] + v[2][0]) / 3, cy = (v[0][1] + v[1][1] + v[2][1]) / 3;
  const inr = edge / (2 * Math.sqrt(3));
  const g = gap === null ? GAP * edge : gap, r = ROUND * edge;
  const k = (inr - g - r) / inr;
  v.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  const pts = [];
  for (const [vx, vy] of v) {
    const ax = cx + (vx - cx) * k, ay = cy + (vy - cy) * k, a = Math.atan2(vy - cy, vx - cx);
    for (let i = 0; i <= ARC_STEPS; i++) {
      const t = a - Math.PI / 3 + (2 * Math.PI / 3) * i / ARC_STEPS;
      pts.push([ax + r * Math.cos(t), ay + r * Math.sin(t)]);
    }
  }
  return scale === 1 ? pts : pts.map(([px, py]) => [cx + (px - cx) * scale, cy + (py - cy) * scale]);
}
function fillPoly(c, pts, color) {
  c.beginPath();
  c.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) c.lineTo(pts[i][0], pts[i][1]);
  c.closePath();
  c.fillStyle = color;
  c.fill();
}
function lighten(hex, k) {
  const n = parseInt(hex.slice(1), 16);
  const ch = [n >> 16, (n >> 8) & 255, n & 255].map(v => Math.round(v + (255 - v) * k));
  return `rgb(${ch[0]},${ch[1]},${ch[2]})`;
}
function pieceBounds(piece, edge) {
  const cells = G.pieces[piece], th = edge * Math.sqrt(3) / 2;
  const xs = cells.map(c => c[0]), ys = cells.map(c => c[1]);
  return [(Math.min(...xs) + Math.max(...xs) + 2) / 2 * edge / 2, (Math.min(...ys) + Math.max(...ys) + 1) / 2 * th];
}
function drawPiece(c, piece, ox, oy, edge, color, gap = null) {
  for (const [x, y] of G.pieces[piece]) fillPoly(c, triPoints(x, y, ox, oy, edge, 1, gap), color);
}
function slotCenter(slot) { return [W * (slot + 0.5) / 3, TRAY_Y]; }
function trayOrigin(slot, piece) {
  const [cx, cy] = slotCenter(slot), [bx, by] = pieceBounds(piece, EDGE);
  return [Math.round(cx - bx), Math.round(cy - by)];
}
function cellCenter(i) {
  const v = triVerts(CX[i], CY[i]).map(([u, l]) => [BOARD_X + u * EDGE / 2, BOARD_Y + l * TRI_H]);
  return [(v[0][0] + v[1][0] + v[2][0]) / 3, (v[0][1] + v[1][1] + v[2][1]) / 3];
}
function text(s, x, y, size, color, align = "center", weight = "700") {
  ctx.font = `${weight} ${size}px "Malgun Gothic", system-ui, sans-serif`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(s, x, y);
}
function label(cells, n) {
  const pts = cells.map(cellCenter);
  const x = pts.reduce((a, p) => a + p[0], 0) / pts.length, y = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  ctx.beginPath(); ctx.arc(x, y, 14, 0, 2 * Math.PI);
  ctx.fillStyle = "#fafafa"; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = "#1e1e1e"; ctx.stroke();
  text(String(n), x, y + 1, 17, "#1e1e1e");
}
function drawCells(board, colors, opts = {}) {
  for (let i = 0; i < 96; i++) {
    let color;
    if (opts.clear && opts.clear.has(i)) color = lighten(opts.ghostColor, 0.55);
    else if (opts.ghost && opts.ghost.has(i)) color = opts.ghostColor;
    else if (board[i]) color = colors[i] || MANUAL;
    else color = EMPTY;
    if (board[i] && opts.highlight && opts.highlight.has(i)) color = lighten(colors[i] || MANUAL, 0.35);
    fillPoly(ctx, triPoints(CX[i], CY[i], BOARD_X, BOARD_Y, EDGE), color);
  }
}

// ------------------------------------------------------------------ recommend mode state
const A = { board: new Uint8Array(96), colors: new Array(96).fill(null), sel: [], selColors: [],
            plan: null, view: -1, total: 0, busy: false, token: 0 };

function pickColor(used) {
  const free = PALETTE.filter(c => !used.includes(c));
  return free[Math.floor(Math.random() * free.length)];
}
async function replan() {
  A.view = -1;
  A.plan = null;
  const token = ++A.token;
  if (!A.sel.length) { A.busy = false; render(); return; }
  A.busy = true;
  if (!session) { render(); return; }      // engine still loading: main() re-plans once it is ready
  render();
  const t0 = performance.now();
  // progressive: a quick plan first (shown at once), then refine with the chosen beam
  const full = Number(document.getElementById("beam").value);
  for (const beam of full > 2 ? [2, full] : [full]) {
    const best = await plan(A.board, A.sel, beam);
    if (token !== A.token) return;        // selection changed while computing
    const free = A.sel.map((_, i) => i), slots = [];
    for (const st of best.steps) {
      const s = free.find(i => A.sel[i] === st.piece);
      free.splice(free.indexOf(s), 1);
      slots.push(s);
    }
    const keepView = A.plan && A.plan.steps.length === best.steps.length ? A.view : -1;
    A.plan = { steps: best.steps, slots, score: best.score, ms: performance.now() - t0, beam, final: beam === full };
    A.view = keepView;
    A.busy = beam !== full;
    render();
  }
}
function planColors() {
  // colors before each step, and after the last
  const out = [A.colors.slice()];
  A.plan.steps.forEach((st, k) => {
    const col = out[out.length - 1].slice();
    for (const i of st.pl.cells) col[i] = A.selColors[A.plan.slots[k]];
    for (const i of st.cleared) col[i] = null;
    out.push(col);
  });
  return out;
}
function applyPlan() {
  if (!A.plan || !A.plan.steps.length) return;
  const cols = planColors();
  A.board = A.plan.steps[A.plan.steps.length - 1].nb;
  A.colors = cols[cols.length - 1];
  A.total += A.plan.score;
  A.sel = []; A.selColors = []; A.plan = null;
  replan();
}
function setSelection(sel, colors) {
  A.sel = sel;
  A.selColors = colors;
  while (A.selColors.length < A.sel.length) A.selColors.push(pickColor(A.selColors));
  replan();
}

function renderAssist() {
  text(String(A.total), W * 0.25, SCORE_Y, SCORE_SIZE, SCORE_C);
  text("누적 점수", W * 0.25, LABEL_Y, 14, MUTED, "center", "400");
  text(A.plan ? `+${A.plan.score}` : "–", W * 0.75, SCORE_Y, SCORE_SIZE, BEST_C);
  text("이번 추천 점수", W * 0.75, LABEL_Y, 14, MUTED, "center", "400");

  const steps = A.plan ? A.plan.steps : [];
  if (!steps.length) {
    drawCells(A.board, A.colors);
  } else if (A.view < 0) {
    drawCells(A.board, A.colors);
    steps.forEach((st, k) => {
      for (const i of st.pl.cells) fillPoly(ctx, triPoints(CX[i], CY[i], BOARD_X, BOARD_Y, EDGE), A.selColors[A.plan.slots[k]]);
    });
    steps.forEach((st, k) => label(st.pl.cells, k + 1));
  } else {
    const st = steps[A.view], cols = planColors()[A.view];
    drawCells(st.before, cols, { ghost: new Set(st.pl.cells), ghostColor: A.selColors[A.plan.slots[A.view]],
                                 clear: new Set(st.cleared) });
    label(st.pl.cells, A.view + 1);
  }
  A.sel.forEach((p, i) => drawPiece(ctx, p, ...trayOrigin(i, p), EDGE, A.selColors[i]));
  if (A.plan) A.plan.slots.forEach((slot, k) => {
    const [cx, cy] = slotCenter(slot);
    ctx.beginPath(); ctx.arc(cx, cy + BADGE_DY, 13, 0, 2 * Math.PI); ctx.fillStyle = "#fafafa"; ctx.fill();
    text(String(k + 1), cx, cy + BADGE_DY + 1, 16, "#1e1e1e");
  });
  if (!A.sel.length) text(layoutName === "mobile" ? "아래 목록에서 블록을 고르세요" : "오른쪽에서 블록을 고르세요",
                          W / 2, TRAY_Y, 16, MUTED, "center", "400");
  if (A.busy && BUSY_Y) text(A.plan ? "정밀 계산 중…" : "AI 계산 중…", W / 2, BUSY_Y, 15, MUTED, "center", "400");

  // side panel
  const stepsEl = document.getElementById("steps");
  stepsEl.innerHTML = "";
  steps.forEach((st, k) => {
    const li = document.createElement("li");
    li.className = A.view === k ? "on" : "";
    li.innerHTML = `<span class="num">${k + 1}</span>블록 ${st.piece} &nbsp; <b>+${st.r}점</b>` +
                   (st.nLines ? ` · ${st.nLines}줄 삭제` : "");
    li.onclick = () => { A.view = A.view === k ? -1 : k; render(); };
    stepsEl.appendChild(li);
  });
  const status = document.getElementById("status");
  if (A.busy && A.plan) status.innerHTML = `<span class="muted">빠른 추천 표시 중 · 정밀 계산 중…</span>`;
  else if (A.busy) status.innerHTML = `<span class="muted">AI 계산 중…</span>`;
  else if (A.plan && steps.length < A.sel.length)
    status.innerHTML = `<span class="warn">모든 블록을 놓을 수 없습니다 (${steps.length}개만 가능)</span>`;
  else if (A.plan) status.innerHTML = `<span class="muted">계산 ${Math.round(A.plan.ms)} ms</span>`;
  else status.textContent = "";
  document.getElementById("apply").disabled = !steps.length;
  document.getElementById("prev").disabled = document.getElementById("next").disabled = !steps.length;
  document.getElementById("viewlabel").textContent = A.view < 0 ? "전체 보기" : `${A.view + 1}단계`;
  document.getElementById("picker").classList.toggle("full", A.sel.length >= 3);
}

// ------------------------------------------------------------------ watch mode
const Wt = { board: new Uint8Array(96), colors: new Array(96).fill(null), tray: [], trayColors: [],
             score: 0, moves: 0, over: false, running: false, last: new Set(), timer: null };

function newTray() {
  Wt.tray = [0, 1, 2].map(() => Math.floor(Math.random() * G.pieces.length));
  Wt.trayColors = [];
  for (let i = 0; i < 3; i++) Wt.trayColors.push(pickColor(Wt.trayColors));
}
function watchReset() {
  Object.assign(Wt, { board: new Uint8Array(96), colors: new Array(96).fill(null), score: 0, moves: 0,
                      over: false, last: new Set() });
  newTray();
  render();
}
async function watchStep() {
  if (!Wt.running || Wt.over) return;
  const mv = await chooseMove(Wt.board, Wt.tray);
  if (!mv) { Wt.over = true; Wt.running = false; render(); return; }
  for (const i of mv.pl.cells) Wt.colors[i] = Wt.trayColors[mv.slot];
  for (const i of mv.cleared) Wt.colors[i] = null;
  Wt.board = mv.nb;
  Wt.score += mv.r;
  Wt.moves += 1;
  Wt.last = new Set(mv.pl.cells.filter(i => !mv.cleared.includes(i)));
  Wt.tray[mv.slot] = null;
  if (Wt.tray.every(p => p === null)) newTray();
  if (!Wt.tray.some(p => p !== null && hasMove(Wt.board, p))) { Wt.over = true; Wt.running = false; }
  render();
  if (Wt.running) Wt.timer = setTimeout(watchStep, Number(document.getElementById("wspeed").value));
}
function renderWatch() {
  text(String(Wt.score), W * 0.25, SCORE_Y, SCORE_SIZE, SCORE_C);
  text("점수", W * 0.25, LABEL_Y, 14, MUTED, "center", "400");
  text(String(Wt.moves), W * 0.75, SCORE_Y, SCORE_SIZE, BEST_C);
  text("둔 수", W * 0.75, LABEL_Y, 14, MUTED, "center", "400");
  drawCells(Wt.board, Wt.colors, { highlight: Wt.last });
  Wt.tray.forEach((p, i) => { if (p !== null) drawPiece(ctx, p, ...trayOrigin(i, p), EDGE, Wt.trayColors[i]); });
  if (Wt.over) {
    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(0, 0, W, H);
    text("GAME OVER", W / 2, H / 2 - 20, 40, TEXT);
    text(`점수 ${Wt.score} · ${Wt.moves}수`, W / 2, H / 2 + 26, 20, TEXT, "center", "400");
  }
  document.getElementById("wplay").textContent = Wt.running ? "❚❚ 일시정지" : "▶ 시작";
  document.getElementById("wstatus").textContent = Wt.over ? "게임 오버 — 새 게임을 눌러 다시 시작" :
    (Wt.running ? "진행 중" : "일시정지");
}

// ------------------------------------------------------------------ main render + input
let mode = "assist";
function render() {
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);
  if (!G) { text("불러오는 중…", W / 2, H / 2, 18, MUTED, "center", "400"); return; }
  if (mode === "assist") renderAssist(); else renderWatch();
}

function pointInTri(p, t) {
  const s = (a, b) => (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1]);
  const d1 = s(t[0], t[1]), d2 = s(t[1], t[2]), d3 = s(t[2], t[0]);
  return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
}
canvas.addEventListener("click", e => {
  if (mode !== "assist" || !G) return;
  const r = canvas.getBoundingClientRect();
  const p = [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height];
  for (let i = 0; i < 96; i++) {
    const t = triVerts(CX[i], CY[i]).map(([u, l]) => [BOARD_X + u * EDGE / 2, BOARD_Y + l * TRI_H]);
    if (pointInTri(p, t)) {
      A.board = A.board.slice();
      A.board[i] ^= 1;
      A.colors[i] = A.board[i] ? MANUAL : null;
      replan();
      return;
    }
  }
  for (let i = 0; i < A.sel.length; i++) {
    const [cx, cy] = slotCenter(i);
    if (Math.abs(p[0] - cx) < W / 6 && Math.abs(p[1] - cy) < 85) {
      setSelection(A.sel.filter((_, j) => j !== i), A.selColors.filter((_, j) => j !== i));
      return;
    }
  }
});

function buildPicker() {
  const el = document.getElementById("picker");
  G.pieces.forEach((_, p) => {
    const c = document.createElement("canvas");
    c.width = c.height = 120;
    c.title = `블록 ${p}`;
    const cx = c.getContext("2d"), edge = 34;
    const [bx, by] = pieceBounds(p, edge);
    drawPiece(cx, p, Math.round(60 - bx), Math.round(62 - by), edge, "#c8c8c8", 1.8);
    cx.font = "20px system-ui"; cx.fillStyle = "#777"; cx.fillText(String(p), 8, 24);
    c.onclick = () => { if (A.sel.length < 3) setSelection([...A.sel, p], A.selColors); };
    el.appendChild(c);
  });
}

function setLayout(name, remember) {
  layoutName = name;
  ({ H, BOARD_Y, TRAY_Y, SCORE_Y, LABEL_Y, SCORE_SIZE, BADGE_DY, BUSY_Y } = LAYOUTS[name]);
  canvas.height = H;
  document.body.classList.toggle("mobile", name === "mobile");
  document.querySelectorAll("#layout button").forEach(b => b.classList.toggle("on", b.dataset.layout === name));
  if (remember) { try { localStorage.setItem("blocky-layout", name); } catch (e) { /* storage blocked */ } }
  render();
}
function initialLayout() {
  if (params.get("layout") in LAYOUTS) return params.get("layout");
  try { const saved = localStorage.getItem("blocky-layout"); if (saved in LAYOUTS) return saved; } catch (e) { /* ignore */ }
  return matchMedia("(max-width: 760px)").matches ? "mobile" : "pc";
}
document.querySelectorAll("#layout button").forEach(b => { b.onclick = () => setLayout(b.dataset.layout, true); });
setLayout(initialLayout(), false);

function setMode(m) {
  mode = m;
  document.getElementById("tab-assist").classList.toggle("on", m === "assist");
  document.getElementById("tab-watch").classList.toggle("on", m === "watch");
  document.getElementById("assist-panel").hidden = m !== "assist";
  document.getElementById("watch-panel").hidden = m !== "watch";
  if (m !== "watch") { Wt.running = false; clearTimeout(Wt.timer); }
  render();
}

function wire() {
  document.getElementById("tab-assist").onclick = () => setMode("assist");
  document.getElementById("tab-watch").onclick = () => setMode("watch");
  document.getElementById("apply").onclick = applyPlan;
  document.getElementById("prev").onclick = () => {
    const n = A.plan.steps.length; A.view = A.view >= 0 ? A.view - 1 : n - 1; render(); };
  document.getElementById("next").onclick = () => {
    const n = A.plan.steps.length; A.view = A.view + 1 < n ? A.view + 1 : -1; render(); };
  document.getElementById("clearsel").onclick = () => setSelection([], []);
  document.getElementById("clearboard").onclick = () => {
    A.board = new Uint8Array(96); A.colors = new Array(96).fill(null); A.total = 0; replan(); };
  document.getElementById("beam").onchange = replan;
  document.getElementById("wplay").onclick = () => {
    if (Wt.over) watchReset();
    Wt.running = !Wt.running;
    clearTimeout(Wt.timer);
    render();
    if (Wt.running) watchStep();
  };
  document.getElementById("wnew").onclick = () => { Wt.running = false; clearTimeout(Wt.timer); watchReset(); };
  document.addEventListener("keydown", e => {
    if (mode !== "assist" || !A.plan || !A.plan.steps.length) return;
    if (e.key === "ArrowRight") document.getElementById("next").click();
    else if (e.key === "ArrowLeft") document.getElementById("prev").click();
    else if (e.key === "Enter") applyPlan();
  });
}

// ------------------------------------------------------------------ self test (?selftest)
async function selftest() {
  const out = document.getElementById("selftest");
  const log = s => { out.textContent += s + "\n"; };
  const fx = await (await fetch("web/selftest.json")).json();
  let ok = 0, bad = 0;
  for (const g of fx.games) {               // rules: replay Python games move by move
    let b = new Uint8Array(96), score = 0;
    for (const [piece, idx, expScore, expBoard] of g) {
      const res = applyMove(b, G.placements[piece][idx]);
      b = res.nb; score += res.r;
      const bits = Array.from(b).join("");
      if (score === expScore && bits === expBoard) ok++; else bad++;
    }
  }
  log(`rules: ${ok} moves match, ${bad} mismatch`);
  let vmax = 0;                              // model: values vs PyTorch
  const v = await values(fx.values.boards.map(s => Uint8Array.from(s, ch => +ch)), fx.values.trays);
  fx.values.expected.forEach((e, i) => { vmax = Math.max(vmax, Math.abs(e - v[i])); });
  log(`values: max abs diff vs PyTorch = ${vmax.toExponential(2)}`);
  for (const p of fx.plans) {                // planner vs Python Agent.plan
    const t0 = performance.now();
    const best = await plan(Uint8Array.from(p.board, ch => +ch), p.pieces, p.beam);
    log(`plan pieces=${p.pieces} beam=${p.beam}: js score ${best.score} / py ${p.score}, ` +
        `same moves: ${JSON.stringify(best.steps.map(s => [s.piece, s.pl.ox, s.pl.oy])) === JSON.stringify(p.moves)}, ` +
        `same final board: ${Array.from(best.board).join("") === p.final}, ` +
        `${Math.round(performance.now() - t0)} ms`);
  }
  log(`engine: ${engineName}`);
  log("SELFTEST DONE");
  if (params.has("report")) fetch("/__report", { method: "POST", body: out.textContent }).catch(() => {});
}

// ------------------------------------------------------------------ boot
(async function main() {
  render();
  try {
    const [g, info] = await Promise.all([fetch("web/game.json").then(r => r.json()),
                                         fetch("web/model_info.json").then(r => r.json())]);
    G = g;
    RS = g.reward_scale;
    G.cells.forEach(([x, y]) => { CX.push(x); CY.push(y); });
    // the board and piece picker work before the AI engine is ready
    buildPicker();
    wire();
    newTray();
    if (params.has("pick")) {               // e.g. ?pick=14,25,3 preselects pieces (demo / sharing)
      const ps = params.get("pick").split(",").map(Number).filter(p => p >= 0 && p < G.pieces.length).slice(0, 3);
      setSelection(ps, []);
    }
    const engineEl = document.getElementById("engine");
    engineEl.textContent = "AI 엔진 불러오는 중… (처음 한 번만 몇 초 걸립니다)";
    render();
    await initModel(info.check);
    const score = info.plan3_mean !== undefined
      ? `3개 계획 평균 ${Math.round(info.plan3_mean)}점 (최고 ${info.plan3_max}) · 1수씩 평균 ${Math.round(info.greedy_mean)}점`
      : `평가 평균 ${info.eval_mean}점`;
    engineEl.textContent =
      `모델 v${info.version}${info.lite ? " 경량(웹용)" : ""} · ${score} · ${(info.train_steps / 1e6).toFixed(1)}M 수 학습 · 연산: ${engineName}`;
    replan();
    if (params.has("selftest")) await selftest();
  } catch (e) {
    document.getElementById("engine").innerHTML = `<span class="warn">모델을 불러오지 못했습니다: ${e}</span>`;
    console.error(e);
  }
})();
