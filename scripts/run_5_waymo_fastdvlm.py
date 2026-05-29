"""Run Fast-dVLM-3B (V3 template) on the same 5 Waymo samples used by
run_5_waymo_compare.py. Saves to results/waymo_5_compare/fast_dvlm_raw.json
so it can be merged into the final comparison report alongside V3 and
dVLM-AD.
"""
import json
import os
import sys
import time
import traceback

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


def _run_fastdvlm(loader, bundle, sample, n_steps=2):
    prompt = build_prompt_v3(sample)
    img_path = _front_cam(sample)
    text, latency = loader.generate(bundle, [img_path], prompt,
                                     gen_length=512, steps=n_steps, temperature=0.0,
                                     block_size=32)
    return {"output": text, "latency_s": latency, "img": img_path, "prompt_len": len(prompt)}


def main():
    data = json.load(open(DATA))
    picked = [data[i] for i in PICKED_INDICES]

    import math
    print(f"Picked {len(picked)} samples:")
    for s in picked:
        vx, vy = s["velocity"][-1]
        print(f"  {s['sample_id'][:30]}  nav={s['navigation_command']}  speed={math.hypot(vx, vy):.1f}")

    print("\nLoading Fast-dVLM-3B...")
    from eval.loaders import fast_dvlm_v3 as fast_loader
    bundle = fast_loader.load()

    results = []
    for sample in picked:
        print(f"\n=== {sample['sample_id'][:30]} nav={sample['navigation_command']} ===")
        try:
            res = _run_fastdvlm(fast_loader, bundle, sample, n_steps=2)
            print(f"  Fast-dVLM done ({res['latency_s']:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            res = {"output": f"ERROR: {e}", "latency_s": -1}
        results.append({
            "sample_id": sample["sample_id"],
            "nav": sample["navigation_command"],
            "speed": (sample["velocity"][-1][0] ** 2 + sample["velocity"][-1][1] ** 2) ** 0.5,
            "image": _front_cam(sample),
            "gt_future_5_waypoints": sample["future waypoints"][:5],
            "fast_dvlm": res,
        })

    os.makedirs(f"{ROOT}/results/waymo_5_compare", exist_ok=True)
    out_path = f"{ROOT}/results/waymo_5_compare/fast_dvlm_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
