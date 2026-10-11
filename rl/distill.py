"""Knowledge distillation: train a small "lite" value net that imitates a big one (for the web demo).

1. Play games with the teacher (with a little randomness) and record many afterstates it could reach,
   labelled with the teacher's value V(afterstate).
2. Train a small ValueNet (fewer channels / blocks) to predict those values (MSE).
3. Benchmark the student exactly like a release; store as models/vX/blocky_vX_lite.{pt,onnx}.

Run: python -m rl.distill models/v1.0.1/blocky_v1.0.1.pt --ch 32 --blocks 2
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from rl import export_web, release
from rl.agent import REWARD_SCALE, Agent
from rl.fastenv import N_PIECES, Env, bits_to_grid, boards_to_bits, tray_vector
from rl.model import ValueNet, save


def collect(agent, samples, envs=128, per_env=16, eps=0.05, seed=0):
    """Afterstates (bits, tray) + teacher values, from teacher-played games."""
    rng = random.Random(seed)
    games = [Env(seed * 1000 + i) for i in range(envs)]
    bits, trays, vals = [], [], []
    n = 0
    while n < samples:
        boards, tr, owner, cands_all = [], [], [], []
        for i, env in enumerate(games):
            cands = env.candidates()
            cands_all.append(cands)
            for slot, _, nb, _ in cands:
                boards.append(nb)
                tr.append(env.remaining_after(slot))
                owner.append(i)
        v = agent.values(boards, tr)
        k = 0
        keep = []
        for i, env in enumerate(games):
            cands = cands_all[i]
            m = len(cands)
            if m == 0:
                env.reset()
                continue
            q = np.array([c[3] for c in cands]) * REWARD_SCALE + v[k:k + m]
            j = rng.randrange(m) if rng.random() < eps else int(q.argmax())
            # keep the chosen move + a random subset of the alternatives (diverse, incl. bad boards)
            pick = {j} | set(rng.sample(range(m), min(per_env, m)))
            keep += [k + p for p in pick]
            env.step(cands[j][0], cands[j][1])
            if env.moves >= 3000:
                env.reset()
            k += m
        bits.append(boards_to_bits([boards[x] for x in keep]))
        trays.append(np.stack([tray_vector(tr[x]) for x in keep]).astype(np.uint8))
        vals.append(v[keep].astype(np.float32))
        n += len(keep)
    return np.concatenate(bits), np.concatenate(trays), np.concatenate(vals)


def train_student(bits, trays, vals, ch, blocks, epochs, dev, batch=4096, lr=2e-3):
    net = ValueNet(ch, blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(vals)
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:20000], perm[20000:]
    steps = epochs * (len(tr_idx) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)

    def tens(idx):
        return (torch.from_numpy(bits_to_grid(bits[idx])).to(dev), torch.from_numpy(trays[idx].astype(np.float32)).to(dev),
                torch.from_numpy(vals[idx]).to(dev))

    vg, vt, vv = tens(val_idx)
    for ep in range(epochs):
        net.train()
        np.random.shuffle(tr_idx)
        for s in range(0, len(tr_idx) - batch + 1, batch):
            g, t, y = tens(tr_idx[s:s + batch])
            loss = F.mse_loss(net(g, t), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            p = net(vg, vt)
            mae = (p - vv).abs().mean().item()
        print(f"epoch {ep + 1}/{epochs}  val MAE {mae:.3f}  (value std {vv.std().item():.2f})", flush=True)
    return net, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--publish", action="store_true", help="use this lite model in the web demo")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = Agent.from_file(args.teacher, dev)
    t = time.time()
    bits, trays, vals = collect(teacher, args.samples)
    print(f"collected {len(vals):,} afterstates in {time.time() - t:.0f}s", flush=True)
    net, mae = train_student(bits, trays, vals, args.ch, args.blocks, args.epochs, dev)

    out_dir = os.path.dirname(os.path.abspath(args.teacher))
    with open(os.path.join(out_dir, "model_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    v = info["version"]
    tag = f"lite_c{args.ch}b{args.blocks}"
    pt = os.path.join(out_dir, f"blocky_v{v}_{tag}.pt")
    save(net, pt, {"teacher": v, "distill_val_mae": round(mae, 4)})
    export_web.export_model(pt, os.path.join(out_dir, f"blocky_v{v}_{tag}.onnx"))
    bench = release.benchmark(pt)
    params = sum(p.numel() for p in net.parameters())
    info.setdefault("lite", {})[tag] = {"arch": {"ch": args.ch, "blocks": args.blocks}, "params": params,
                                        "distill_val_mae": round(mae, 4), "samples": int(len(vals)),
                                        "benchmark": bench}
    with open(os.path.join(out_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"{tag}: params {params:,}, plan3 mean {bench['plan3']['mean']} (teacher {info['benchmark']['plan3']['mean']})")
    if args.publish:
        release.publish_web(info, lite=tag)


if __name__ == "__main__":
    main()
