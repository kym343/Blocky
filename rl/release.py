"""Register a trained checkpoint as a new model version (only if it beats the current best).

Versioning rules (see models/README.md):
  patch  x.y.Z  same training setup, more training / better checkpoint of the same recipe
  minor  x.Y.0  new hyperparameters, network size or training technique (same inputs, app-compatible)
  major  X.0.0  new algorithm (e.g. search on top of the net), incompatible inputs, or game-rule change

Every version is scored on the same fixed benchmark:
  greedy : one move at a time, 200 games (seeds 1,000,000+)
  plan3  : whole tray planned at once (beam 64, what the app uses), 64 games (seeds 3,000,000+)
A release is accepted only if plan3 mean beats the best registered version (or with --force).

Run: python -m rl.release runs/exp1/model_best.pt --bump patch --notes "exp1 1.8M steps"
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import time

import numpy as np

from rl import export_web
from rl.agent import Agent
from rl.evaluate import eval_plan
from rl.train import evaluate as eval_greedy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


def parse(v):
    return tuple(int(x) for x in v.split("."))


def registry():
    """All registered versions, oldest first: list of info dicts."""
    infos = []
    for path in glob.glob(os.path.join(MODELS, "v*", "model_info.json")):
        with open(path, encoding="utf-8") as f:
            infos.append(json.load(f))
    return sorted(infos, key=lambda i: parse(i["version"]))


def bump(version, kind):
    a, b, c = parse(version)
    return {"major": f"{a + 1}.0.0", "minor": f"{a}.{b + 1}.0", "patch": f"{a}.{b}.{c + 1}"}[kind]


def stats(res, secs):
    sc = np.array([s for s, _, _ in res])
    mv = np.array([m for _, m, _ in res])
    return {"games": len(sc), "mean": round(float(sc.mean()), 1), "median": float(np.median(sc)),
            "min": int(sc.min()), "max": int(sc.max()), "moves_mean": round(float(mv.mean()), 1),
            "seconds": round(secs)}


def benchmark(ckpt):
    agent = Agent.from_file(ckpt)
    t = time.time()
    greedy = stats(eval_greedy(agent, 200, 3000, seed0=1_000_000), time.time() - t)
    print("greedy", greedy, flush=True)
    t = time.time()
    plan3 = stats(eval_plan(agent, 64, 3000, 64, seed0=3_000_000), time.time() - t)
    print("plan3 ", plan3, flush=True)
    return {"greedy": greedy, "plan3": plan3}


def write_readme(infos):
    rows = []
    for i in reversed(infos):
        b = i["benchmark"]
        rows.append(f"| v{i['version']} | {i['date']} | {i['bump']} | {b['plan3']['mean']} | {b['plan3']['median']:.0f} | "
                    f"{b['plan3']['max']} | {b['greedy']['mean']} | {b['greedy']['max']} | "
                    f"{(i['source']['train_steps'] or 0):,} | {i['notes']} |")
    text = f"""# Blocky 모델 버전

최신 버전이 항상 최고 성능입니다 (더 좋을 때만 새 버전 등록). `ai_app.py`와 웹 데모의 기본 모델은 최신 버전입니다.

## 버전 규칙 (MAJOR.MINOR.PATCH)

| 올리는 자리 | 언제 | 예 |
|---|---|---|
| PATCH `1.0.x` | 같은 학습 설정으로 더 학습했거나, 같은 실험의 더 좋은 체크포인트 | exp1을 더 오래 학습 |
| MINOR `1.x.0` | 하이퍼파라미터·신경망 크기·학습 기법 변경 (입력 형식과 앱 사용법은 동일) | γ 0.99→0.995, 채널 64→128 |
| MAJOR `x.0.0` | 알고리즘 변경, 입력 형식 변경(이전 앱과 비호환), 게임 규칙 변경 | MCTS 탐색 결합, 새 입력 특징 |

등록 조건: 아래 표준 벤치마크의 **plan3 평균**이 이전 최고 버전보다 높아야 함.

## 표준 벤치마크

- **plan3**: 대기 블록 3개를 한꺼번에 계획(beam 64, 앱·웹과 같은 방식), 고정 시드 64판
- **greedy**: 한 수씩 선택, 고정 시드 200판
- 한 판 최대 3000수

## 버전 목록

| 버전 | 날짜 | 변경 | plan3 평균 | plan3 중앙값 | plan3 최고 | greedy 평균 | greedy 최고 | 학습 수 | 메모 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

등록: `python -m rl.release <checkpoint> --bump patch|minor|major --notes "..."`
"""
    with open(os.path.join(MODELS, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--bump", choices=["patch", "minor", "major"])
    ap.add_argument("--version", help="explicit version (first release)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--force", action="store_true", help="register even if not better")
    args = ap.parse_args()

    infos = registry()
    if args.version:
        version, kind = args.version, args.bump or "initial"
    else:
        version, kind = bump(infos[-1]["version"], args.bump), args.bump
    if any(i["version"] == version for i in infos) and not args.force:
        raise SystemExit(f"v{version} already exists")

    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"benchmarking {args.ckpt} as v{version} ...", flush=True)
    bench = benchmark(args.ckpt)
    best = max((i for i in infos if i["version"] != version), key=lambda i: i["benchmark"]["plan3"]["mean"],
               default=None)
    if best and bench["plan3"]["mean"] <= best["benchmark"]["plan3"]["mean"] and not args.force:
        raise SystemExit(f"rejected: plan3 mean {bench['plan3']['mean']} <= v{best['version']} "
                         f"{best['benchmark']['plan3']['mean']}")

    extra = ck.get("extra", {})
    if extra.get("teacher") and not extra.get("steps"):   # distilled: inherit the teacher's training length
        teacher = next((i for i in infos if i["version"] == extra["teacher"]), {})
        extra["steps"] = (teacher.get("source") or {}).get("train_steps")
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg_path = os.path.join(run_dir, "config.json")
    train_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    out = os.path.join(MODELS, f"v{version}")
    os.makedirs(out, exist_ok=True)
    pt = os.path.join(out, f"blocky_v{version}.pt")
    if os.path.abspath(args.ckpt) != os.path.abspath(pt):
        shutil.copy(args.ckpt, pt)
    export_web.export_model(pt, os.path.join(out, f"blocky_v{version}.onnx"))
    info = {
        "version": version,
        "date": datetime.date.today().isoformat(),
        "bump": kind,
        "notes": args.notes,
        "source": {"run": train_cfg.get("name"), "checkpoint": os.path.relpath(args.ckpt, ROOT).replace("\\", "/"),
                   "train_steps": extra.get("steps"), "train_eval_mean": extra.get("eval_mean"),
                   "distilled_from": extra.get("teacher")},
        "arch": ck["config"],
        "train_config": train_cfg,
        "benchmark": bench,
    }
    with open(os.path.join(out, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    infos = registry()
    write_readme(infos)
    publish_web(infos[-1])
    print(f"registered v{version}: plan3 mean {bench['plan3']['mean']}, greedy mean {bench['greedy']['mean']}")


def publish_web(info, lite=None):
    """Point the web demo (docs/web) at the given version (optionally its distilled lite model)."""
    v = info["version"]
    src = os.path.join(MODELS, f"v{v}")
    os.makedirs(export_web.OUT, exist_ok=True)
    export_web.export_data(os.path.join(export_web.OUT, "game.json"))
    name = f"blocky_v{v}_{lite}.onnx" if lite else f"blocky_v{v}.onnx"
    shutil.copy(os.path.join(src, name), os.path.join(export_web.OUT, "model.onnx"))
    b = info["lite"][lite]["benchmark"] if lite else info["benchmark"]
    web_info = {"version": v, "lite": lite, "train_steps": info["source"]["train_steps"],
                "plan3_mean": b["plan3"]["mean"], "plan3_max": b["plan3"]["max"],
                "greedy_mean": b["greedy"]["mean"], "greedy_max": b["greedy"]["max"]}
    web_info["check"] = check_vector(os.path.join(export_web.OUT, "model.onnx"))
    with open(os.path.join(export_web.OUT, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(web_info, f, indent=2)


def check_vector(onnx_path, n=6, seed=7):
    """A few (board, tray) inputs + reference outputs; the web demo rejects backends that disagree."""
    import random
    import onnxruntime as ort
    from rl.model import encode
    rng = random.Random(seed)
    boards = [rng.getrandbits(96) & rng.getrandbits(96) for _ in range(n)]
    trays = [[rng.randrange(26) for _ in range(rng.randrange(3))] for _ in range(n)]
    grid, tv = encode(boards, trays)
    out = ort.InferenceSession(onnx_path).run(None, {"grid": grid, "tray": tv})[0]
    return {"boards": ["".join(str(b >> i & 1) for i in range(96)) for b in boards], "trays": trays,
            "values": [round(float(v), 5) for v in out]}


if __name__ == "__main__":
    main()
