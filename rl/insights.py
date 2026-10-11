"""Turn the trained model into human-teachable tips by testing hypotheses.

Run: python -m rl.insights runs/exp1/model_best.pt --games 200
"""

import argparse
import random
from collections import Counter

import numpy as np

from rl.agent import REWARD_SCALE, Agent
from rl.evaluate import eval_plan
from rl.fastenv import CELLS, CELL_SET, IDX, LINE_MASKS, N_PIECES, PIECES, Env, has_move
from blocky_core import neighbors

EDGE_CELLS = [c for c in CELLS if sum(n in CELL_SET for n in neighbors(*c)) < 3]
EDGE_MASK = sum(1 << IDX[c] for c in EDGE_CELLS)


def lines_of(before, pl, after):
    b = before | pl.mask
    return sum(1 for L in LINE_MASKS if b & L == L and not after & L)


def q_values(agent, env, cands):
    v = agent.values([nb for _, _, nb, _ in cands], [env.remaining_after(s) for s, _, _, _ in cands])
    return np.array([c[3] for c in cands]) * REWARD_SCALE + v


# ---- policies: (agent, env, cands, rng) -> index into cands

def p_model(agent, env, cands, rng):
    return int(q_values(agent, env, cands).argmax())


def p_force_clear(agent, env, cands, rng):
    """Model, but if any move clears a line, it must pick a clearing move."""
    q = q_values(agent, env, cands)
    clear = [i for i, (_, pl, nb, _) in enumerate(cands) if lines_of(env.board, pl, nb)]
    pool = clear or range(len(cands))
    return max(pool, key=lambda i: q[i])


def p_max_lines(agent, env, cands, rng):
    """Model, but always pick the move clearing the most lines (ties -> model)."""
    q = q_values(agent, env, cands)
    n = [lines_of(env.board, pl, nb) for _, pl, nb, _ in cands]
    m = max(n)
    return max((i for i in range(len(cands)) if n[i] == m), key=lambda i: q[i])


def p_greedy(agent, env, cands, rng):
    best = max(c[3] for c in cands)
    return rng.choice([i for i, c in enumerate(cands) if c[3] == best])


def play(agent, policy, games, seed0=3_000_000, max_moves=3000, stats=False):
    scores, moves = [], []
    st = Counter()
    occ = np.zeros(len(CELLS))
    occ_n = 0
    deaths = Counter()
    for s in range(games):
        env = Env(seed0 + s)
        rng = random.Random(s)
        while env.moves < max_moves:
            cands = env.candidates()
            if not cands:
                for p in env.tray:
                    if p is not None:
                        deaths[p] += 1
                break
            i = policy(agent, env, cands, rng)
            slot, pl, nb, r = cands[i]
            if stats:
                avail = [lines_of(env.board, c[1], c[2]) for c in cands]
                chosen = lines_of(env.board, pl, nb)
                st["moves"] += 1
                if max(avail) > 0:
                    st["clear_avail"] += 1
                    st["clear_taken"] += chosen > 0
                    st["took_max"] += chosen == max(avail)
                if max(avail) >= 2:
                    st["multi_avail"] += 1
                    st["multi_taken"] += chosen >= 2
                st["edge_place"] += (pl.mask & EDGE_MASK).bit_count()
                st["cells_place"] += pl.size
            env.step(slot, pl)
            if stats:
                bits = env.board
                occ += np.array([bits >> k & 1 for k in range(len(CELLS))])
                occ_n += 1
                st["hex_ok"] += has_move(bits, N_PIECES - 1)
                st["types_ok"] += sum(has_move(bits, p) for p in range(N_PIECES))
                st["filled"] += bits.bit_count()
        scores.append(env.score)
        moves.append(env.moves)
    out = {"mean": float(np.mean(scores)), "median": float(np.median(scores)), "moves": float(np.mean(moves))}
    if stats:
        m = st["moves"]
        out.update({
            "clear_avail_rate": st["clear_avail"] / m,
            "clear_taken_when_avail": st["clear_taken"] / max(1, st["clear_avail"]),
            "took_max_lines_when_avail": st["took_max"] / max(1, st["clear_avail"]),
            "multi_avail_rate": st["multi_avail"] / m,
            "multi_taken_when_avail": st["multi_taken"] / max(1, st["multi_avail"]),
            "edge_share_of_placed": st["edge_place"] / st["cells_place"],
            "edge_share_of_board": len(EDGE_CELLS) / len(CELLS),
            "hexagon_fits_rate": st["hex_ok"] / m,
            "piece_types_fit": st["types_ok"] / m,
            "avg_filled": st["filled"] / m,
            "edge_occupancy": float(occ[[IDX[c] for c in EDGE_CELLS]].mean() / occ_n),
            "center_occupancy": float(np.delete(occ, [IDX[c] for c in EDGE_CELLS]).mean() / occ_n),
            "deaths_by_size": dict(sorted(Counter(len(PIECES[p]) for p in deaths.elements()).items())),
        })
    return out


def plan_order(agent, samples=300, seed=5):
    """In 3-piece plans, which piece goes first? (big-first vs small-first, clear-first)."""
    rng = random.Random(seed)
    env = Env(seed)
    st = Counter()
    while st["plans"] < samples:
        if env.is_over():
            env.reset()
        pieces = [p for p in env.tray if p is not None]
        if len(pieces) == 3 and len({len(PIECES[p]) for p in pieces}) > 1:
            steps, _, _ = agent.plan(env.board, pieces)
            if len(steps) == 3:
                sizes = [len(PIECES[p]) for p, *_ in steps]
                st["plans"] += 1
                st["largest_first"] += sizes[0] == max(sizes)
                st["smallest_first"] += sizes[0] == min(sizes)
                st["smallest_last"] += sizes[-1] == min(sizes)
                b = env.board
                clears = []
                for p, pl, r, nb in steps:
                    clears.append(lines_of(b, pl, nb) > 0)
                    b = nb
                if any(clears):
                    st["plans_with_clear"] += 1
                    st["clear_not_first"] += not clears[0]
        mv = agent.choose_many([env])[0]
        if mv:
            env.step(mv[0], mv[1])
    n = st["plans"]
    return {k: (v / n if k not in ("plans", "plans_with_clear", "clear_not_first") else v) for k, v in st.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--plan-games", type=int, default=64)
    args = ap.parse_args()
    agent = Agent.from_file(args.model)

    print("## behaviour (model, one move at a time)")
    for name, pol in (("model", p_model), ("greedy", p_greedy)):
        r = play(agent, pol, args.games, stats=True)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    print("## rule experiments (same seeds)")
    for name, pol in (("model", p_model), ("force_clear", p_force_clear), ("max_lines", p_max_lines)):
        r = play(agent, pol, args.games)
        print(name, {k: round(v, 1) for k, v in r.items()})

    print("## one-at-a-time vs plan-3 (same seeds)")
    res = eval_plan(agent, args.plan_games, 3000, 64, seed0=3_000_000)
    print("plan3", {"mean": round(np.mean([s for s, _, _ in res]), 1),
                    "median": float(np.median([s for s, _, _ in res])),
                    "moves": round(np.mean([m for _, m, _ in res]), 1)})
    r = play(agent, p_model, args.plan_games)
    print("model(same games)", {k: round(v, 1) for k, v in r.items()})

    print("## plan order")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in plan_order(agent).items()})


if __name__ == "__main__":
    main()
