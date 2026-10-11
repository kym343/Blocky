"""Move selection with a trained afterstate value net: greedy play and 3-piece planning."""

import random

import numpy as np
import torch

from rl.fastenv import PLACEMENTS, apply
from rl.model import encode, load

REWARD_SCALE = 0.1


class Agent:
    def __init__(self, model, device=None):
        self.device = device or next(model.parameters()).device
        self.model = model

    @classmethod
    def from_file(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load(path, device)
        return cls(model, device)

    @torch.no_grad()
    def values(self, boards, trays, chunk=16384):
        if not boards:
            return np.zeros(0, np.float32)
        grid, tv = encode(boards, trays)
        out = []
        for i in range(0, len(boards), chunk):
            g = torch.from_numpy(grid[i:i + chunk]).to(self.device)
            t = torch.from_numpy(tv[i:i + chunk]).to(self.device)
            out.append(self.model(g, t).float().cpu().numpy())
        return np.concatenate(out)

    def choose_many(self, envs, eps=0.0, rng=None):
        """One move per env: list of (slot, placement, reward, afterstate) or None if game over."""
        all_c, boards, trays, owner = [], [], [], []
        for i, env in enumerate(envs):
            cands = env.candidates()
            all_c.append(cands)
            for slot, _, nb, _ in cands:
                boards.append(nb)
                trays.append(env.remaining_after(slot))
                owner.append(i)
        v = self.values(boards, trays)
        out, k = [], 0
        rng = rng or random
        for i, cands in enumerate(all_c):
            n = len(cands)
            if n == 0:
                out.append(None)
                continue
            if eps and rng.random() < eps:
                j = rng.randrange(n)
            else:
                q = np.array([c[3] for c in cands], np.float32) * REWARD_SCALE + v[k:k + n]
                j = int(q.argmax())
            slot, pl, nb, r = cands[j]
            out.append((slot, pl, r, (nb, trays[k + j])))
            k += n
        return out

    def plan(self, board, pieces, beam=64):
        """Best order + positions for placing all `pieces` (list of piece ids) on `board`.

        Returns (steps, total_reward, value) where steps = [(piece, placement, reward, board_after)].
        If not every piece fits, returns the plan placing as many pieces as possible.
        """
        # state: (score so far, board, remaining pieces, steps)
        states = [(0.0, board, list(pieces), [])]
        best_partial = states[0]
        for _ in range(len(pieces)):
            nxt, boards, trays = [], [], []
            for score, b, rem, steps in states:
                for piece in set(rem):
                    left = list(rem)
                    left.remove(piece)
                    for pl in PLACEMENTS[piece]:
                        if b & pl.mask:
                            continue
                        nb, r = apply(b, pl)
                        nxt.append((score + r, nb, left, steps + [(piece, pl, r, nb)]))
                        boards.append(nb)
                        trays.append(left)
            if not nxt:
                break
            v = self.values(boards, trays)
            ranked = np.argsort(-(np.array([s[0] for s in nxt]) * REWARD_SCALE + v))
            states = [nxt[i] for i in ranked[:beam]]
            best_partial = states[0]
        score, b, rem, steps = best_partial
        value = float(self.values([b], [rem])[0])
        return steps, int(score), value
