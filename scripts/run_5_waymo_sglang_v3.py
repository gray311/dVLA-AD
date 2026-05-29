"""Run SGLang Fast-dVLM with V3 template fill on 5 Waymo samples.
Saves to results/waymo_5_compare/fast_dvlm_sglang_v3_<algo>.json
"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PICKED_INDICES = [281, 204, 327, 14, 48]
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")


def _fix(p):
    return p.replace(PATH_FIX[0], PATH_FIX[1])


def _front_cam(sample):
    return _fix(sample["image"][1])


def main():
    algorithm = sys.argv[1] if len(sys.argv) > 1 else "mdm"
    data = json.load(open(DATA))
    picked = [data[i] for i in PICKED_INDICES]
    print(f"Algorithm: {algorithm}")
    print(f"\nPicked {len(picked)} samples:")
    for s in picked:
        vx, vy = s["velocity"][-1]
        print(f"  {s['sample_id'][:30]}  nav={s['navigation_command']}  speed={math.hypot(vx, vy):.1f}")

    print(f"\nLoading Fast-dVLM via SGLang (algorithm={algorithm}) ...")
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    bundle = loader.load(algorithm=algorithm)

    # Warmup with first sample
    print("\nWarmup...")
    warmup_sample = picked[0]
    _, warmup_lat = loader.generate(
        bundle, [_front_cam(warmup_sample)], build_prompt_v3(warmup_sample),
        temperature=0.0,
    )
    print(f"  Warmup done ({warmup_lat:.2f}s)")

    results = []
    for sample in picked:
        sid = sample["sample_id"]
        print(f"\n=== {sid[:30]} nav={sample['navigation_command']} ===")
        try:
            text, latency = loader.generate(
                bundle, [_front_cam(sample)], build_prompt_v3(sample),
                temperature=0.0, nav_command=sample["navigation_command"],
            )
            print(f"  done ({latency:.2f}s, {len(text)} chars)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            text, latency = f"ERROR: {e}", -1
        results.append({
            "sample_id": sid,
            "nav": sample["navigation_command"],
            "speed": (sample["velocity"][-1][0] ** 2 + sample["velocity"][-1][1] ** 2) ** 0.5,
            "image": _front_cam(sample),
            "gt_future_5_waypoints": sample["future waypoints"][:5],
            "fast_dvlm_sglang_v3": {
                "output": text,
                "latency_s": latency,
                "algorithm": algorithm,
            },
        })

    os.makedirs(f"{ROOT}/results/waymo_5_compare", exist_ok=True)
    out_path = f"{ROOT}/results/waymo_5_compare/fast_dvlm_sglang_v3_{algorithm}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")
    loader.shutdown(bundle)


if __name__ == "__main__":
    main()
