"""Evaluate a trained model: greedy (one move at a time) vs planned (whole tray of 3 at once).

Run: python -m rl.evaluate runs/exp1/model_best.pt --games 200 --modes greedy plan
"""

import argparse
import time

import numpy as np

from rl.agent import Agent
from rl.fastenv import Env
from rl.train import evaluate as eval_greedy


def eval_plan(agent, games, max_moves, beam, seed0=1_000_000):
    out = []
    for s in range(games):
        env = Env(seed0 + s)
        while env.moves < max_moves:
            pieces = [p for p in env.tray if p is not None]
            steps, _, _ = agent.plan(env.board, pieces, beam=beam)
            if not steps:
                break
            for piece, pl, _, _ in steps:
                env.step(env.tray.index(piece), pl)
            if len(steps) < len(pieces) or env.is_over():
                break
        out.append((env.score, env.moves, env.moves >= max_moves))
    return out


def summary(name, res, secs):
    sc = np.array([s for s, _, _ in res])
    mv = np.array([m for _, m, _ in res])
    capped = sum(c for _, _, c in res)
    print(f"{name:8s} games {len(sc):4d}  mean {sc.mean():8.1f}  median {np.median(sc):7.0f}  "
          f"min {sc.min():5d}  max {sc.max():6d}  moves {mv.mean():7.1f}  capped {capped}  ({secs:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-moves", type=int, default=3000)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--modes", nargs="+", default=["greedy", "plan"])
    args = ap.parse_args()
    agent = Agent.from_file(args.model)
    for mode in args.modes:
        t = time.time()
        if mode == "greedy":
            res = eval_greedy(agent, args.games, args.max_moves)
        else:
            res = eval_plan(agent, args.games, args.max_moves, args.beam)
        summary(mode, res, time.time() - t)


if __name__ == "__main__":
    main()
