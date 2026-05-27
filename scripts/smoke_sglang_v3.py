"""Smoke test for SGLang Fast-dVLM with V3 template fill on 1 Waymo sample."""
import json
import math
import os
import sys
import time

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")


def _fix(p):
    return p.replace(PATH_FIX[0], PATH_FIX[1])


def main():
    algorithm = sys.argv[1] if len(sys.argv) > 1 else "mdm"
    data = json.load(open(DATA))
    sample = data[281]  # GO_LEFT, speed=1.2
    img = _fix(sample["image"][1])
    prompt = build_prompt_v3(sample)
    print(f"Sample: {sample['sample_id'][:24]}  nav={sample['navigation_command']}")
    print(f"Image: {img}")
    print(f"Prompt length: {len(prompt)} chars")

    print(f"\nLoading Fast-dVLM via SGLang (algorithm={algorithm}) ...")
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    bundle = loader.load(algorithm=algorithm)

    # Warmup
    print("\nWarmup...")
    _, lat_warmup = loader.generate(bundle, [img], prompt, temperature=0.0)
    print(f"  Warmup done ({lat_warmup:.2f}s)")

    print("\nRun 1...")
    text, lat = loader.generate(bundle, [img], prompt, temperature=0.0)
    print(f"  done ({lat:.2f}s, {len(text)} chars)")
    print("\n=== OUTPUT ===")
    print(text)
    print("=== END ===")

    loader.shutdown(bundle)


if __name__ == "__main__":
    main()
