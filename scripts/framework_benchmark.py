"""Apples-to-apples framework benchmark: SAME denoise algorithm, count
forward passes and ms/fwd in both paths to verify SGLang's framework-level
speedup (CUDA Graph + flashinfer attention + KV cache pool).

Usage:
  DLLM_FWD_LOG=1 python scripts/framework_benchmark.py sglang
  python scripts/framework_benchmark.py transformers
  python scripts/framework_benchmark.py report
"""
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
# Just 5 samples — enough for ms/fwd statistic.
PICKED = [20, 1, 327, 281, 444]
OUT = os.path.join(ROOT, "results", "framework_bench")


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])
def _front(s): return _fix(s["image"][1])


def run_sglang():
    os.environ["DLLM_FWD_LOG"] = "1"
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    data = json.load(open(DATA))
    picked = [data[i] for i in PICKED]

    print("Loading SGLang (4 steps/chunk)...")
    bundle = loader.load(algorithm="mdm")
    # Warmup
    loader.generate(bundle, [_front(picked[0])], build_prompt_v3(picked[0]),
                     temperature=0.0, steps_per_chunk=4)

    results = []
    for i, s in enumerate(picked):
        text, lat = loader.generate(bundle, [_front(s)], build_prompt_v3(s),
                                     temperature=0.0, steps_per_chunk=4)
        results.append({"sample_id": s["sample_id"], "latency_s": lat, "output": text})
        print(f"  [{i+1}/{len(picked)}] lat={lat:.2f}s")

    loader.shutdown(bundle)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sglang_4steps.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT}/sglang_4steps.json")


def run_transformers():
    """Run transformers with fixed step count comparable to SGLang's 4 steps/chunk.
    Default transformers uses adaptive 1/2/4 → fix to constant by setting
    steps=4 (steps_per_block in BD3LM loop). The transformers loader already
    has an `steps` arg in its `generate()` function.
    """
    from eval.loaders import fast_dvlm_v3 as loader
    data = json.load(open(DATA))
    picked = [data[i] for i in PICKED]

    print("Loading transformers (4 steps/chunk)...")
    bundle = loader.load()
    # Warmup
    loader.generate(bundle, [_front(picked[0])], build_prompt_v3(picked[0]),
                     gen_length=512, steps=4, temperature=0.0)

    results = []
    for i, s in enumerate(picked):
        text, lat = loader.generate(bundle, [_front(s)], build_prompt_v3(s),
                                     gen_length=512, steps=4, temperature=0.0)
        results.append({"sample_id": s["sample_id"], "latency_s": lat, "output": text})
        print(f"  [{i+1}/{len(picked)}] lat={lat:.2f}s")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "transformers_4steps.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT}/transformers_4steps.json")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "sglang":
        run_sglang()
    elif cmd == "transformers":
        run_transformers()
    else:
        print("Usage: python scripts/framework_benchmark.py {sglang|transformers|report}")


if __name__ == "__main__":
    main()
