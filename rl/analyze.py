"""Behaviour statistics: what does a policy actually do? Compares random / greedy / trained models.

Run: python -m rl.analyze runs/exp1/model_best.pt [more models...] --games 200
"""

import argparse
import random
from collections import Counter

import numpy as np

from rl.agent import Agent
from rl.fastenv import LINE_MASKS, PIECES, Env


def count_lines(before, mask, after):
    b = before | mask
    return sum(1 for L in LINE_MASKS if b & L == L and not after & L)


def run(policy, games, seed0=2_000_000, max_moves=3000):
    st = Counter()
    lines_hist = Counter()
    fill_samples = []
    death_pieces = Counter()
    scores = []
    for s in range(games):
        env = Env(seed0 + s)
        rng = random.Random(s)
        while env.moves < max_moves:
            cands = env.candidates()
            if not cands:
                for p in env.tray:
                    if p is not None:
                        death_pieces[len(PIECES[p])] += 1
                break
            slot, pl, nb, r = policy(env, cands, rng)
            n = count_lines(env.board, pl.mask, nb)
            lines_hist[n] += 1
            st["place_pts"] += pl.size
            st["clear_pts"] += r - pl.size
            st["moves"] += 1
            env.step(slot, pl)
            fill_samples.append(env.board.bit_count())
        scores.append(env.score)
    moves = st["moves"]
    return {
        "score_mean": np.mean(scores),
        "moves_per_game": moves / games,
        "pts_per_move": (st["place_pts"] + st["clear_pts"]) / moves,
        "clear_share": st["clear_pts"] / (st["place_pts"] + st["clear_pts"]),
        "clear_move_rate": 1 - lines_hist[0] / moves,
        "multi_clear_rate": sum(v for k, v in lines_hist.items() if k >= 2) / moves,
        "lines_hist": dict(sorted(lines_hist.items())),
        "avg_filled": np.mean(fill_samples),
        "death_piece_sizes": dict(sorted(death_pieces.items())),
    }


def random_policy(env, cands, rng):
    return rng.choice(cands)


def greedy_policy(env, cands, rng):
    best = max(c[3] for c in cands)
    return rng.choice([c for c in cands if c[3] == best])


def model_policy(agent):
    def pol(env, cands, rng):
        mv = agent.choose_many([env])[0]
        slot, pl, r, (nb, _) = mv
        return slot, pl, nb, r
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--games", type=int, default=200)
    args = ap.parse_args()
    pols = [("random", random_policy), ("greedy", greedy_policy)]
    pols += [(m, model_policy(Agent.from_file(m))) for m in args.models]
    keys = ["score_mean", "moves_per_game", "pts_per_move", "clear_share", "clear_move_rate",
            "multi_clear_rate", "avg_filled", "lines_hist", "death_piece_sizes"]
    rows = [(name, run(p, args.games)) for name, p in pols]
    print("| metric | " + " | ".join(n for n, _ in rows) + " |")
    print("|---|" + "---:|" * len(rows))
    for k in keys:
        vals = []
        for _, r in rows:
            v = r[k]
            vals.append(f"{v:.3f}" if isinstance(v, float) and v < 1 else
                        (f"{v:.1f}" if isinstance(v, float) else str(v)))
        print(f"| {k} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
