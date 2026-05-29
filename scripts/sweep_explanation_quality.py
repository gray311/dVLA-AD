"""Sweep steps_per_chunk and rep_penalty to see if explanation quality
improves. Loads the SGLang engine ONCE, runs several configs on a few
representative Waymo samples, and dumps the explanation field for each.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
# A subset of the picked-5 that showed the worst explanation collapse.
SAMPLE_INDICES = [281, 327, 14]  # GO_LEFT slow, GO_STRAIGHT highway, GO_STRAIGHT slow

# (label, kwargs)
CONFIGS = [
    ("baseline  steps=4  rep=2.0", dict(steps_per_chunk=4,  rep_penalty=2.0)),
    ("steps=8   rep=2.0",          dict(steps_per_chunk=8,  rep_penalty=2.0)),
    ("steps=16  rep=2.0",          dict(steps_per_chunk=16, rep_penalty=2.0)),
    ("steps=8   rep=4.0",          dict(steps_per_chunk=8,  rep_penalty=4.0)),
    ("steps=16  rep=4.0",          dict(steps_per_chunk=16, rep_penalty=4.0)),
]


def _fix(p):
    return p.replace(PATH_FIX[0], PATH_FIX[1])


def _explanation(text):
    try:
        return json.loads(text, strict=False).get("explanation", "<no-key>")
    except Exception as e:
        return f"<unparsed: {e}>"


def main():
    algorithm = sys.argv[1] if len(sys.argv) > 1 else "mdm"
    data = json.load(open(DATA))
    samples = [data[i] for i in SAMPLE_INDICES]

    from eval.loaders import fast_dvlm_sglang_v3 as loader
    print(f"Loading engine (algorithm={algorithm})...")
    bundle = loader.load(algorithm=algorithm)

    # Warmup once.
    s0 = samples[0]
    loader.generate(bundle, [_fix(s0["image"][1])], build_prompt_v3(s0),
                    temperature=0.0, nav_command=s0["navigation_command"])

    results = {}
    for si, sample in enumerate(samples):
        img = _fix(sample["image"][1])
        prompt = build_prompt_v3(sample)
        nav = sample["navigation_command"]
        sid = sample["sample_id"][:13]
        results[sid] = {"nav": nav}
        print("\n" + "#" * 92)
        print(f"# SAMPLE {sid}  nav={nav}")
        print("#" * 92)
        for label, kw in CONFIGS:
            text, lat = loader.generate(
                bundle, [img], prompt, temperature=0.0,
                nav_command=nav, **kw,
            )
            exp = _explanation(text)
            results[sid][label] = {"explanation": exp, "latency_s": lat}
            print(f"\n--- {label}   ({lat:.2f}s) ---")
            print(exp)

    out_path = f"{ROOT}/results/waymo_5_compare/explanation_sweep_{algorithm}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")
    loader.shutdown(bundle)


if __name__ == "__main__":
    main()
