"""Plot training curves of one or more runs.

Run: python -m rl.plot exp1 [exp2 ...]   -> docs/img/<names>.png
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = {"random": 76.2, "greedy": 99.9}


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def main(names):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for name in names:
        d = os.path.join(ROOT, "runs", name)
        tr, ev = read(os.path.join(d, "train.csv")), read(os.path.join(d, "eval.csv"))
        s = [x / 1e6 for x in tr["steps"]]
        ax[0, 0].plot(s, tr["score_mean100"], label=f"{name} train (last 100 games)")
        if ev:
            es = [x / 1e6 for x in ev["steps"]]
            ax[0, 0].plot(es, ev["mean"], "o-", label=f"{name} eval mean (greedy, 64 games)")
            ax[0, 1].plot(es, ev["moves_mean"], "o-", label=name)
        ax[1, 0].plot(s, tr["loss"], label=name)
        ax[1, 1].plot(s, tr["v_mean"], label=name)
    for k, v in BASELINE.items():
        ax[0, 0].axhline(v, ls="--", c="gray", lw=1)
        ax[0, 0].text(0, v, f" {k} {v}", va="bottom", color="gray", fontsize=8)
    titles = [("Score", "score"), ("Game length (eval)", "moves per game"),
              ("TD loss (Huber)", "loss"), ("Mean predicted value V", "V")]
    for a, (t, y) in zip(ax.flat, titles):
        a.set_title(t)
        a.set_xlabel("env moves (millions)")
        a.set_ylabel(y)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "img")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "_".join(names) + ".png")
    fig.savefig(path, dpi=110)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1:] or ["exp1"])
