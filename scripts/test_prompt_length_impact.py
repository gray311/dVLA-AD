"""Test how prompt length affects SGLang Fast-dVLM latency.

Three prompt sizes, same image + same template:
  - SHORT:  bare task statement (~20 lines)
  - MEDIUM: V3 prompt without trajectory examples (~50 lines)
  - LONG:   full V3 prompt with per-sample worked examples (~90 lines)

Same sample, run 3 times each to average warm-state latency.
"""
import json
import math
import os
import sys
import time

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, build_template_v3

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def _joint_image(s):
    return _fix(s["image"][1]).replace("CAM_FRONT.jpg", "CAM_JOINT.jpg")


SHORT_PROMPT = """You are an autonomous driving assistant. Look at the image and predict the future
behavior + trajectory.

INPUT:
- Multi-view image (3 cams stitched)
- Ego state: speed={speed:.1f} m/s
- Driver instruction: {instruction}

Fill the JSON template below.

TEMPLATE:

{template}
"""


MEDIUM_PROMPT = """You are an autonomous driving assistant. Given the current driving scene, identify critical objects, explain your reasoning, then predict the future driving behavior and trajectory.

INPUT:
- Multi-view images: front-left, front, front-right
- Ego state: speed={speed:.1f} m/s, longitudinal acceleration={accel:.2f} m/s^2
- Driver instruction: {instruction}

TASK:
Fill in the masked positions. Format requirements:
- critical_objects: 12 categories × 2 tokens (e.g. "red car" or "none")
- complexity: "simple" or "complex"
- explanation: ~100 tokens of natural reasoning
- behavior: longitudinal in {{speed up, slow down, keep speed, stop now}}; lateral in {{keep lane, turn left, turn right, change left, change right}}
- trajectory: 10 waypoints @ 0.5 s, format "<t>s: forward=+XX.Xm, lateral=+YY.Ym"
  • lateral POSITIVE = LEFT, NEGATIVE = RIGHT
  • GO_LEFT → lateral grows positive; GO_RIGHT → lateral grows negative

TEMPLATE:

{template}
"""


def main():
    data = json.load(open(DATA))
    s = data[202]  # GO_LEFT 5.7 m/s
    img = _joint_image(s)
    vx, vy = s["velocity"][-1]
    sp = math.hypot(vx, vy)
    ax, _ = s["acceleration"][-1]

    prompts = {
        "SHORT":  SHORT_PROMPT.format(speed=sp, instruction=s["navigation_command"],
                                       template=build_template_v3()),
        "MEDIUM": MEDIUM_PROMPT.format(speed=sp, accel=ax,
                                        instruction=s["navigation_command"],
                                        template=build_template_v3()),
        "LONG":   build_prompt_v3(s),
    }
    for name, p in prompts.items():
        print(f"{name}: {len(p)} chars / ~{p.count(chr(10))} lines")

    print("\nLoading SGLang Fast-dVLM...")
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    bundle = loader.load(algorithm="mdm")

    # Warmup
    loader.generate(bundle, [img], prompts["MEDIUM"], temperature=0.0,
                     nav_command=s["navigation_command"])

    # Three runs each, average
    results = {}
    for name, prompt in prompts.items():
        lats = []
        for i in range(3):
            _, lat = loader.generate(bundle, [img], prompt, temperature=0.0,
                                       nav_command=s["navigation_command"])
            lats.append(lat)
        results[name] = lats
        print(f"  {name:>6} ({len(prompt):>5} chars): {[f'{l:.2f}' for l in lats]}  "
              f"mean={sum(lats)/len(lats):.2f}s")

    loader.shutdown(bundle)

    print("\n=== Summary ===")
    print(f"{'prompt':>8}  {'chars':>6}  {'avg lat':>9}  {'Δ vs SHORT':>10}")
    base = sum(results["SHORT"]) / 3
    for name, lats in results.items():
        avg = sum(lats) / len(lats)
        delta = avg - base
        print(f"{name:>8}  {len(prompts[name]):>6}  {avg:>8.2f}s  {delta:+9.2f}s")


if __name__ == "__main__":
    main()
