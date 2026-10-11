"""Fixture for the web demo self-test (docs/index.html?selftest): rules, model values and plans from Python.

Run: python -m rl.make_selftest models/v1.0.0/blocky_v1.0.0.pt
"""

import json
import os
import random
import sys

from rl.agent import Agent
from rl.fastenv import PLACEMENTS, Env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bitstr(board):
    return "".join(str(board >> i & 1) for i in range(96))


def main(ckpt):
    agent = Agent.from_file(ckpt, "cpu")
    games = []
    for s in range(20):                          # random games, every move recorded
        env, rng, g = Env(s), random.Random(s), []
        while not env.is_over():
            slot, pl, _, _ = rng.choice(env.candidates())
            env.step(slot, pl)
            g.append([pl.piece, PLACEMENTS[pl.piece].index(pl), env.score, bitstr(env.board)])
        games.append(g)

    rng = random.Random(1)
    boards, trays = [], []
    for _ in range(64):
        boards.append(rng.getrandbits(96) & rng.getrandbits(96))
        trays.append([rng.randrange(26) for _ in range(rng.randrange(3))])
    expected = [float(v) for v in agent.values(boards, trays)]

    plans = []
    env = Env(123)
    for k in range(6):                           # positions from an AI-played game
        for _ in range(9):
            mv = agent.choose_many([env])[0]
            if mv is None:
                env.reset()
                continue
            env.step(mv[0], mv[1])
        pieces = [rng.randrange(26) for _ in range(3)]
        beam = [4, 12, 32][k % 3]
        steps, score, _ = agent.plan(env.board, pieces, beam=beam)
        plans.append({"board": bitstr(env.board), "pieces": pieces, "beam": beam, "score": score,
                      "moves": [[p, pl.ox, pl.oy] for p, pl, _, _ in steps],
                      "final": bitstr(steps[-1][3]) if steps else bitstr(env.board)})

    path = os.path.join(ROOT, "docs", "web", "selftest.json")
    with open(path, "w") as f:
        json.dump({"games": games, "values": {"boards": [bitstr(b) for b in boards], "trays": trays,
                                              "expected": expected}, "plans": plans}, f, separators=(",", ":"))
    print("wrote", path, sum(len(g) for g in games), "moves")


if __name__ == "__main__":
    main(sys.argv[1])
