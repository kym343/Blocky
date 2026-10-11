"""Check fastenv matches blocky_core exactly, and measure speed + simple baselines.

Run: python -m rl.test_fastenv
"""

import random
import time

from blocky_core import Game
from rl.fastenv import IDX, PLACEMENTS, Env


def check_equivalence(games=200):
    for seed in range(games):
        rng = random.Random(seed)
        env = Env(seed)
        g = Game(random.Random(0))
        g.tray = list(env.tray)
        while True:
            assert env.is_over() == g.is_over()
            if env.is_over():
                break
            cands = env.candidates()
            # same legal move set
            ref = {(p, pos) for p in set(x for x in g.tray if x is not None) for pos in g.valid_positions(p)}
            assert {(pl.piece, (pl.ox, pl.oy)) for _, pl, _, _ in cands} == ref
            slot, pl, _, _ = rng.choice(cands)
            env_tray_before = list(env.tray)
            env.step(slot, pl)
            gained, _ = g.place(slot, pl.ox, pl.oy)
            if all(p is None for p in g.tray) or g.tray != env.tray:
                g.tray = list(env.tray)  # sync random refill
            board = sum(1 << IDX[c] for c in g.filled)
            assert board == env.board and g.score == env.score, (seed, env_tray_before)
    print(f"equivalence OK over {games} random games")


def play(policy, games, seed0=10_000):
    scores, moves = [], []
    t = time.perf_counter()
    for s in range(games):
        env = Env(seed0 + s)
        rng = random.Random(s)
        while not env.is_over():
            slot, pl, _, _ = policy(env.candidates(), rng)
            env.step(slot, pl)
        scores.append(env.score)
        moves.append(env.moves)
    dt = time.perf_counter() - t
    return scores, sum(moves) / dt


def random_policy(cands, rng):
    return rng.choice(cands)


def greedy_policy(cands, rng):
    best = max(c[3] for c in cands)
    return rng.choice([c for c in cands if c[3] == best])


if __name__ == "__main__":
    print("placements per piece:", [len(p) for p in PLACEMENTS])
    check_equivalence()
    for name, pol in (("random", random_policy), ("greedy", greedy_policy)):
        sc, sps = play(pol, 300)
        sc.sort()
        print(f"{name:7s} mean {sum(sc)/len(sc):7.1f}  median {sc[len(sc)//2]:5d}  max {sc[-1]:5d}  ({sps:,.0f} moves/s)")
