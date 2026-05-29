"""Worker: load engine once, run steps=16/rep=4 on 3 samples, print
explanation. Explanation slot length comes from env DVLA_N_EXPLANATION_TOKENS.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, N_EXPLANATION_TOKENS

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
SAMPLE_INDICES = [281, 327, 14]
KW = dict(steps_per_chunk=16, rep_penalty=4.0)


def _fix(p):
    return p.replace(PATH_FIX[0], PATH_FIX[1])


def _exp(text):
    try:
        return json.loads(text, strict=False).get("explanation", "<no-key>")
    except Exception as e:
        return f"<unparsed: {e}>"


def main():
    data = json.load(open(DATA))
    samples = [data[i] for i in SAMPLE_INDICES]
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    bundle = loader.load(algorithm="mdm")
    s0 = samples[0]
    loader.generate(bundle, [_fix(s0["image"][1])], build_prompt_v3(s0),
                    temperature=0.0, nav_command=s0["navigation_command"], **KW)
    print("=" * 92)
    print(f"EXPLANATION LENGTH = {N_EXPLANATION_TOKENS} masks   (steps=16, rep=4.0)")
    print("=" * 92)
    for sample in samples:
        img = _fix(sample["image"][1])
        text, lat = loader.generate(
            bundle, [img], build_prompt_v3(sample), temperature=0.0,
            nav_command=sample["navigation_command"], **KW,
        )
        print(f"\n--- {sample['sample_id'][:13]}  nav={sample['navigation_command']}  ({lat:.2f}s) ---")
        print(_exp(text))
    loader.shutdown(bundle)


if __name__ == "__main__":
    main()
