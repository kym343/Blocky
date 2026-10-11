"""Export a trained model (ONNX) and the game data (JSON) for the browser demo in docs/.

Run: python -m rl.export_web models/v1.0.0/blocky_v1.0.0.pt --version 1.0.0
Writes docs/web/{game.json, model.onnx, model_info.json} and a copy of the ONNX next to the checkpoint.
"""

import argparse
import datetime
import json
import os
import shutil

import numpy as np
import torch

from blocky_core import CELLS, PIECES
from rl.fastenv import IDX, LINE_MASKS, PLACEMENTS
from rl.model import encode, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "web")


def bits(mask):
    return [i for i in range(len(CELLS)) if mask >> i & 1]


def export_data(path):
    lines = [bits(L) for L in LINE_MASKS]
    line_id = {L: k for k, L in enumerate(LINE_MASKS)}
    data = {
        "cells": CELLS,                       # index -> [x, y]
        "pieces": [list(map(list, p)) for p in PIECES],
        "lines": lines,                       # line -> cell indices
        "placements": [[{"ox": pl.ox, "oy": pl.oy, "cells": bits(pl.mask),
                         "lines": sorted(line_id[L] for L in pl.lines)} for pl in plist]
                       for plist in PLACEMENTS],
        "reward_scale": 0.1,
    }
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def export_model(ckpt, path):
    model, extra = load(ckpt, "cpu")
    rng = np.random.default_rng(0)
    boards = [int(rng.integers(0, 2 ** 62)) << 34 | int(rng.integers(0, 2 ** 34)) for _ in range(8)]
    trays = [[0, 5], [], [25], [3, 3], [1], [], [7, 8], [24]]
    grid, tv = encode(boards, trays)
    g, t = torch.from_numpy(grid), torch.from_numpy(tv)
    torch.onnx.export(model, (g, t), path, input_names=["grid", "tray"], output_names=["value"],
                      dynamic_axes={"grid": {0: "n"}, "tray": {0: "n"}, "value": {0: "n"}},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    ref = model(g, t).detach().numpy()
    got = sess.run(None, {"grid": grid, "tray": tv})[0]
    print("onnx max abs diff:", float(np.abs(ref - got).max()), "| checkpoint:", extra)
    return extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    export_data(os.path.join(OUT, "game.json"))
    onnx_path = os.path.join(OUT, "model.onnx")
    extra = export_model(args.ckpt, onnx_path)
    info = {"version": args.version, "train_steps": extra.get("steps"), "eval_mean": extra.get("eval_mean"),
            "checkpoint": os.path.relpath(args.ckpt, ROOT).replace("\\", "/"),
            "exported": datetime.date.today().isoformat()}
    for path in (os.path.join(OUT, "model_info.json"),
                 os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "model_info.json")):
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
    shutil.copy(onnx_path, os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                        f"blocky_v{args.version}.onnx"))
    print("wrote", OUT, info)
