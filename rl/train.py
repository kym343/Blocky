"""Train the afterstate value network with TD learning + experience replay.

Run: python -m rl.train --name exp1 --steps 1000000
Outputs in runs/<name>/: train.csv, eval.csv, model_latest.pt, model_best.pt, config.json
"""

import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from rl.agent import REWARD_SCALE, Agent
from rl.fastenv import N_CELLS, N_PIECES, Env, bits_to_grid, boards_to_bits, tray_vector
from rl.model import ValueNet, save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Replay:
    def __init__(self, cap):
        self.cap, self.n, self.i = cap, 0, 0
        self.bits = np.zeros((cap, N_CELLS), np.uint8)
        self.tray = np.zeros((cap, N_PIECES), np.uint8)
        self.r = np.zeros(cap, np.float32)
        self.nbits = np.zeros((cap, N_CELLS), np.uint8)
        self.ntray = np.zeros((cap, N_PIECES), np.uint8)
        self.done = np.zeros(cap, np.float32)
        self.disc = np.zeros(cap, np.float32)

    def add_many(self, items):
        """items: list of ((board, tray), discounted return, (board, tray) or None, bootstrap discount).

        TD target for the first afterstate = return + (1 - done) * discount * V(next afterstate)."""
        if not items:
            return
        b = boards_to_bits([s[0] for s, _, _, _ in items])
        nb = boards_to_bits([(n[0] if n else 0) for _, _, n, _ in items])
        for k, (s, r, n, disc) in enumerate(items):
            j = self.i
            self.bits[j], self.nbits[j] = b[k], nb[k]
            self.tray[j] = tray_vector(s[1])
            self.ntray[j] = tray_vector(n[1]) if n else 0
            self.r[j] = r
            self.done[j] = n is None
            self.disc[j] = disc
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, size, device):
        idx = np.random.randint(0, self.n, size)
        t = lambda a: torch.from_numpy(a).to(device)
        return (t(bits_to_grid(self.bits[idx])), t(self.tray[idx].astype(np.float32)), t(self.r[idx]),
                t(bits_to_grid(self.nbits[idx])), t(self.ntray[idx].astype(np.float32)), t(self.done[idx]),
                t(self.disc[idx]))


def evaluate(agent, games, max_moves, seed0=1_000_000):
    """Greedy play on fixed seeds, all games batched. Returns list of (score, moves, capped)."""
    envs = [Env(seed0 + s) for s in range(games)]
    active = list(range(games))
    results = [None] * games
    while active:
        moves = agent.choose_many([envs[i] for i in active])
        still = []
        for i, mv in zip(active, moves):
            env = envs[i]
            if mv is None or env.moves >= max_moves:
                results[i] = (env.score, env.moves, mv is not None)
                continue
            env.step(mv[0], mv[1])
            still.append(i)
        active = still
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp1")
    ap.add_argument("--steps", type=int, default=1_000_000, help="total env moves")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--nstep", type=int, default=1, help="n-step TD returns (1 = plain TD(0))")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--updates", type=int, default=2, help="gradient steps per env step batch")
    ap.add_argument("--replay", type=int, default=500_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--eps-start", type=float, default=0.10)
    ap.add_argument("--eps-end", type=float, default=0.005)
    ap.add_argument("--eps-frac", type=float, default=0.3, help="fraction of steps to decay eps over")
    ap.add_argument("--target-every", type=int, default=1000, help="updates between target syncs")
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--max-moves", type=int, default=3000, help="episode cap (train and eval)")
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    args = ap.parse_args()

    out = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ValueNet(args.ch, args.blocks).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["state"])
    tgt = ValueNet(args.ch, args.blocks).to(dev)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent = Agent(net, dev)
    rb = Replay(args.replay)
    rng = random.Random(args.seed)

    envs = [Env(args.seed * 100_000 + i) for i in range(args.envs)]
    # per env: afterstates still waiting for n rewards -> [afterstate, [scaled rewards after it]]
    hist = [deque() for _ in range(args.envs)]
    g_pow = [args.gamma ** k for k in range(args.nstep + 1)]

    def ret(rs):
        return sum(g_pow[k + 1] * r for k, r in enumerate(rs))
    ep_scores, ep_moves = [], []
    steps = updates = 0
    next_eval = args.eval_every
    best_eval = -1.0
    losses, vmeans = [], []
    t0 = time.time()
    tlog = open(os.path.join(out, "train.csv"), "w", newline="")
    elog = open(os.path.join(out, "eval.csv"), "w", newline="")
    tw, ew = csv.writer(tlog), csv.writer(elog)
    tw.writerow(["steps", "episodes", "score_mean100", "score_max100", "moves_mean100", "loss", "v_mean", "eps", "sps", "minutes"])
    ew.writerow(["steps", "mean", "median", "min", "max", "moves_mean", "capped", "minutes"])
    last_log = 0

    while steps < args.steps:
        eps = args.eps_end + (args.eps_start - args.eps_end) * max(0.0, 1 - steps / (args.eps_frac * args.steps))
        net.eval()
        moves = agent.choose_many(envs, eps, rng)
        trans = []
        for i, mv in enumerate(moves):
            env = envs[i]
            h = hist[i]
            if mv is None:                       # game over: everything pending ends with V = 0
                trans += [(s, ret(rs), None, 0.0) for s, rs in h]
                h.clear()
                ep_scores.append(env.score)
                ep_moves.append(env.moves)
                env.reset()
                continue
            slot, pl, r, after = mv
            for item in h:
                item[1].append(r * REWARD_SCALE)
            if h and len(h[0][1]) == args.nstep:  # oldest has n rewards: bootstrap from this afterstate
                s, rs = h.popleft()
                trans.append((s, ret(rs), after, g_pow[args.nstep]))
            h.append([after, []])
            env.step(slot, pl)
            steps += 1
            if env.moves >= args.max_moves:      # truncate (no terminal signal): drop pending
                ep_scores.append(env.score)
                ep_moves.append(env.moves)
                env.reset()
                h.clear()
        rb.add_many(trans)

        if rb.n >= args.warmup:
            net.train()
            for _ in range(args.updates):
                g, t, r, ng, nt, d, disc = rb.sample(args.batch, dev)
                with torch.no_grad():
                    target = r + (1 - d) * disc * tgt(ng, nt)
                v = net(g, t)
                loss = F.smooth_l1_loss(v, target)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                updates += 1
                losses.append(loss.item())
                vmeans.append(v.mean().item())
                if updates % args.target_every == 0:
                    tgt.load_state_dict(net.state_dict())

        if steps - last_log >= 10_000:
            last_log = steps
            mins = (time.time() - t0) / 60
            s100, m100 = ep_scores[-100:], ep_moves[-100:]
            row = [steps, len(ep_scores), round(float(np.mean(s100)), 1) if s100 else 0, max(s100) if s100 else 0,
                   round(float(np.mean(m100)), 1) if m100 else 0, round(float(np.mean(losses[-200:])), 4) if losses else 0,
                   round(float(np.mean(vmeans[-200:])), 2) if vmeans else 0, round(eps, 4),
                   round(steps / (mins * 60)), round(mins, 2)]
            tw.writerow(row)
            tlog.flush()
            print("train", dict(zip(["steps", "eps_done", "score100", "max100", "moves100", "loss", "v", "eps", "sps", "minutes"], row)), flush=True)

        if steps >= next_eval or steps >= args.steps:
            next_eval += args.eval_every
            net.eval()
            res = evaluate(agent, args.eval_games, args.max_moves)
            sc = sorted(s for s, _, _ in res)
            mins = (time.time() - t0) / 60
            row = [steps, round(float(np.mean(sc)), 1), sc[len(sc) // 2], sc[0], sc[-1],
                   round(float(np.mean([m for _, m, _ in res])), 1), sum(c for _, _, c in res), round(mins, 2)]
            ew.writerow(row)
            elog.flush()
            print("EVAL ", dict(zip(["steps", "mean", "median", "min", "max", "moves", "capped", "minutes"], row)), flush=True)
            save(net, os.path.join(out, "model_latest.pt"), {"steps": steps, "eval_mean": row[1]})
            if row[1] > best_eval:
                best_eval = row[1]
                save(net, os.path.join(out, "model_best.pt"), {"steps": steps, "eval_mean": row[1]})

    tlog.close()
    elog.close()
    print("done. best eval mean", best_eval)


if __name__ == "__main__":
    main()
