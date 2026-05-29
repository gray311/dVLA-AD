"""Sweep Fast-dVLM configs to find <2s sweet spot with reasonable quality.
Runs on sample 0 (GO_LEFT, speed=1.2) only — single sample to compare configs fast.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")


def _fix(p):
    return p.replace(PATH_FIX[0], PATH_FIX[1])


def main():
    data = json.load(open(DATA))
    # Try TWO samples for robustness: one slow GO_LEFT, one fast GO_STRAIGHT
    samples = [data[281], data[327]]  # GO_LEFT 1.2m/s, GO_STRAIGHT 21.9m/s

    print("Loading Fast-dVLM-3B...")
    from eval.loaders import fast_dvlm_v3 as fast_loader
    bundle = fast_loader.load()

    # Warmup
    print("Warmup...")
    fast_loader.generate(bundle, [_fix(samples[0]["image"][1])],
                          build_prompt_v3(samples[0]),
                          gen_length=512, steps=2, temperature=0.0, block_size=32)

    configs = [
        # (block_size, steps, label)
        (16, 4, "16/4"),
        (16, 3, "16/3"),
        (16, 2, "16/2"),
        (32, 2, "32/2"),
        (32, 3, "32/3"),
        (8,  2, "8/2"),
        (8,  3, "8/3"),
    ]
    results = []
    for bs, st, label in configs:
        for sname, samp in zip(["GO_LEFT", "GO_STR_fast"], samples):
            prompt = build_prompt_v3(samp)
            img = [_fix(samp["image"][1])]
            text, lat = fast_loader.generate(bundle, img, prompt,
                                              gen_length=512, steps=st,
                                              temperature=0.0, block_size=bs)
            results.append({"config": label, "sample": sname, "latency": lat,
                            "output": text})
            print(f"[{label}] {sname}: {lat:.2f}s")

    out_path = f"{ROOT}/results/waymo_5_compare/fast_dvlm_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
