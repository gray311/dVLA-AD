"""Sweep dllm_template_steps_per_chunk to see how latency vs ADE trade off
on SGLang. Compares to transformers baseline (3.55m ADE, 4.40s latency).
"""
import json
import math
import os
import sys

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, parse_filled

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
# Same 20 picks as compare_latency_sglang_vs_transformers.py
PICKED = [20, 1, 25, 107, 327, 14, 48, 281, 204, 175, 250, 300, 4, 67, 89,
          150, 200, 350, 400, 444]
OUT = os.path.join(ROOT, "results", "latency_compare")


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])
def _front(s): return _fix(s["image"][1])
def _ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0: return None
    return sum(math.hypot(pred[i][0]-gt[i][0], pred[i][1]-gt[i][1]) for i in range(n)) / n


def main():
    steps_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [4, 6, 8]
    data = json.load(open(DATA))
    picked = [data[i] for i in PICKED]

    from eval.loaders import fast_dvlm_sglang_v3 as loader
    print("Loading SGLang...")
    bundle = loader.load(algorithm="mdm")

    all_results = {}
    for n_steps in steps_list:
        print(f"\n=== steps_per_chunk = {n_steps} ===")
        # Warmup
        loader.generate(bundle, [_front(picked[0])], build_prompt_v3(picked[0]),
                         temperature=0.0, steps_per_chunk=n_steps)
        results = []
        for i, s in enumerate(picked):
            text, lat = loader.generate(
                bundle, [_front(s)], build_prompt_v3(s),
                temperature=0.0, steps_per_chunk=n_steps,
            )
            pred = parse_filled(text)
            gt = s["future waypoints"][:5]
            a = _ade(pred, gt)
            vx, vy = s["velocity"][-1]
            results.append({
                "sample_id": s["sample_id"], "nav": s["navigation_command"],
                "speed": math.hypot(vx, vy), "latency_s": lat, "ade": a,
            })
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(picked)}] lat={lat:.2f}s ADE={a:.2f}m")
        all_results[n_steps] = results
        lats = [r["latency_s"] for r in results]
        ades = [r["ade"] for r in results if r["ade"] is not None]
        print(f"  steps={n_steps}: mean lat={sum(lats)/len(lats):.2f}s, "
              f"mean ADE={sum(ades)/len(ades):.2f}m")

    loader.shutdown(bundle)
    with open(os.path.join(OUT, "sglang_steps_sweep.json"), "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved {OUT}/sglang_steps_sweep.json")

    print("\n=== Summary ===")
    print(f"{'steps':>5}  {'mean lat':>10}  {'mean ADE':>10}  {'min lat':>8}  {'max lat':>8}")
    for n_steps in steps_list:
        rs = all_results[n_steps]
        lats = [r["latency_s"] for r in rs]
        ades = [r["ade"] for r in rs if r["ade"] is not None]
        print(f"{n_steps:>5}  {sum(lats)/len(lats):>9.2f}s  {sum(ades)/len(ades):>9.2f}m  "
              f"{min(lats):>7.2f}s  {max(lats):>7.2f}s")
    print(f"transformers baseline: 4.40s, 3.55m (116 fwd/sample)")


if __name__ == "__main__":
    main()
