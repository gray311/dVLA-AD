"""Larger ADE stability run. Picks N stratified samples and runs the current
SGLang template config across all of them, saves output + ADE per sample,
and prints distribution stats.

Usage:
  python scripts/run_n_waymo_ade.py 30   # 30 samples
  python scripts/run_n_waymo_ade.py 50   # 50 samples
"""
import json
import math
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, parse_filled

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
OUT_DIR = os.path.join(ROOT, "results", "waymo_ade_stability")


def _fix(p):
    return p.replace(PATH_FIX[0], PATH_FIX[1])


def pick_stratified(data, n):
    """Pick n samples spanning nav × speed buckets."""
    buckets = {
        "GO_LEFT": [],
        "GO_RIGHT": [],
        "GO_STRAIGHT_stopped": [],
        "GO_STRAIGHT_slow": [],
        "GO_STRAIGHT_mid": [],
        "GO_STRAIGHT_fast": [],
    }
    for i, s in enumerate(data):
        nav = s["navigation_command"]
        vx, vy = s["velocity"][-1]
        sp = math.hypot(vx, vy)
        if nav == "GO_LEFT":
            buckets["GO_LEFT"].append((i, sp))
        elif nav == "GO_RIGHT":
            buckets["GO_RIGHT"].append((i, sp))
        elif sp < 1.0:
            buckets["GO_STRAIGHT_stopped"].append((i, sp))
        elif sp < 5:
            buckets["GO_STRAIGHT_slow"].append((i, sp))
        elif sp < 15:
            buckets["GO_STRAIGHT_mid"].append((i, sp))
        else:
            buckets["GO_STRAIGHT_fast"].append((i, sp))
    # Distribute n across buckets proportionally with a floor of 1.
    per_bucket = max(1, n // len(buckets))
    picks = []
    for k, v in buckets.items():
        v_sorted = sorted(v, key=lambda x: x[1])
        if not v_sorted:
            continue
        # Evenly sample per_bucket items spread across speed range.
        step = max(1, len(v_sorted) // per_bucket)
        for j in range(0, len(v_sorted), step):
            picks.append(v_sorted[j][0])
            if sum(1 for p in picks if p in [x[0] for x in v]) >= per_bucket:
                break
    return sorted(set(picks))[:n]


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0:
        return None
    return sum(
        math.hypot(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1])
        for i in range(n)
    ) / n


def fde(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0:
        return None
    i = n - 1
    return math.hypot(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1])


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    os.makedirs(OUT_DIR, exist_ok=True)

    data = json.load(open(DATA))
    indices = pick_stratified(data, n_samples)
    picked = [data[i] for i in indices]
    print(f"Picked {len(picked)} samples:")
    for s in picked[:5]:
        vx, vy = s["velocity"][-1]
        print(f"  {s['sample_id'][:24]}  nav={s['navigation_command']:14}  speed={math.hypot(vx, vy):.1f}")
    if len(picked) > 5:
        print(f"  ... and {len(picked) - 5} more")

    print("\nLoading SGLang Fast-dVLM (template-fill mdm)...")
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    bundle = loader.load(algorithm="mdm")
    print("Warmup...")
    _, _ = loader.generate(
        bundle, [_fix(picked[0]['image'][1])], build_prompt_v3(picked[0]),
        temperature=0.0, nav_command=picked[0]['navigation_command'],
    )

    results = []
    for k, sample in enumerate(picked):
        try:
            text, latency = loader.generate(
                bundle, [_fix(sample['image'][1])], build_prompt_v3(sample),
                temperature=0.0, nav_command=sample['navigation_command'],
            )
            pred = parse_filled(text)
            gt = sample["future waypoints"][:5]
            a = ade(pred, gt)
            f = fde(pred, gt)
            vx, vy = sample["velocity"][-1]
            print(f"  [{k+1}/{len(picked)}] {sample['sample_id'][:8]} "
                  f"nav={sample['navigation_command']:14} v={math.hypot(vx,vy):5.1f}  "
                  f"ADE={a:5.2f}m  FDE={f:5.2f}m  lat={latency:.2f}s")
            results.append({
                "sample_id": sample["sample_id"],
                "nav": sample["navigation_command"],
                "speed": math.hypot(vx, vy),
                "gt_future_5": gt,
                "pred_5": pred[:5],
                "ade": a, "fde": f, "latency_s": latency, "output": text,
            })
        except Exception as e:
            traceback.print_exc()
            results.append({"sample_id": sample["sample_id"], "error": str(e)})

    loader.shutdown(bundle)

    # Save raw
    out_path = os.path.join(OUT_DIR, f"sglang_semantic_n{len(picked)}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Stats
    valid = [r for r in results if "ade" in r and r["ade"] is not None]
    if not valid:
        print("No valid samples!")
        return
    ades = sorted(r["ade"] for r in valid)
    fdes = sorted(r["fde"] for r in valid)
    lats = [r["latency_s"] for r in valid]
    n = len(ades)

    def pct(arr, p):
        k = max(0, min(len(arr) - 1, int(p * len(arr))))
        return arr[k]

    print("\n=== ADE / FDE distribution (semantic-trajectory) ===")
    print(f"N = {n}")
    print(f"ADE: mean={sum(ades)/n:.2f}m  median={pct(ades, 0.5):.2f}m  "
          f"p25={pct(ades, 0.25):.2f}m  p75={pct(ades, 0.75):.2f}m  "
          f"p90={pct(ades, 0.9):.2f}m  min={ades[0]:.2f}m  max={ades[-1]:.2f}m")
    print(f"FDE: mean={sum(fdes)/n:.2f}m  median={pct(fdes, 0.5):.2f}m  "
          f"p25={pct(fdes, 0.25):.2f}m  p75={pct(fdes, 0.75):.2f}m  "
          f"p90={pct(fdes, 0.9):.2f}m  min={fdes[0]:.2f}m  max={fdes[-1]:.2f}m")
    print(f"latency: mean={sum(lats)/n:.2f}s  min={min(lats):.2f}s  max={max(lats):.2f}s")

    # By nav
    print("\n=== By nav ===")
    by_nav = {}
    for r in valid:
        by_nav.setdefault(r["nav"], []).append(r["ade"])
    for nav, vs in sorted(by_nav.items()):
        print(f"  {nav:14}  n={len(vs):3}  ADE mean={sum(vs)/len(vs):.2f}m  "
              f"median={sorted(vs)[len(vs)//2]:.2f}m")

    # By speed bucket
    print("\n=== By speed ===")
    buckets = {"stop (<1)": [], "slow (1-5)": [], "mid (5-15)": [], "fast (>15)": []}
    for r in valid:
        sp = r["speed"]
        if sp < 1: buckets["stop (<1)"].append(r["ade"])
        elif sp < 5: buckets["slow (1-5)"].append(r["ade"])
        elif sp < 15: buckets["mid (5-15)"].append(r["ade"])
        else: buckets["fast (>15)"].append(r["ade"])
    for k, vs in buckets.items():
        if vs:
            print(f"  {k:14}  n={len(vs):3}  ADE mean={sum(vs)/len(vs):.2f}m  "
                  f"median={sorted(vs)[len(vs)//2]:.2f}m")


if __name__ == "__main__":
    main()
