"""Quick 20-case validation for fast algorithm iteration.
Same quality flags as test_100, on 20 stratified samples.
Usage: python scripts/test_20_quick.py
"""
import json, math, os, re, sys, time
ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))
from eval.template_v3 import build_prompt_v3, parse_filled
from scripts.test_100_ddrive_sglang import quality_flags, pick_stratified, extract_explanation, ade

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def main():
    data = json.load(open(DATA))
    indices = pick_stratified(data, 20)
    print(f"Picked {len(indices)} samples", flush=True)
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    bundle = loader.load(algorithm="mdm", engine_block_size=160)
    s0 = data[indices[0]]
    loader.generate(bundle, [_fix(s0['image'][1])], build_prompt_v3(s0),
                    temperature=0.0, block_size=160, section_align=True,
                    nav_command=s0['navigation_command'])
    results = []
    for k, idx in enumerate(indices):
        s = data[idx]
        text, lat = loader.generate(
            bundle, [_fix(s['image'][1])], build_prompt_v3(s),
            temperature=0.0, block_size=160, section_align=True,
            nav_command=s['navigation_command'])
        pred = parse_filled(text)
        gt = s["future waypoints"][:5]
        a = ade(pred, gt) if pred else None
        flags = quality_flags(text)
        expl = extract_explanation(text)
        results.append({"idx": idx, "ade": a, "latency": lat, "flags": flags, "explanation": expl, "text": text})
    loader.shutdown(bundle)

    flagged = [r for r in results if r["flags"]]
    valid = [r for r in results if r.get("ade") is not None]
    from collections import Counter
    fc = Counter()
    for r in flagged:
        for fl in r["flags"]:
            fc[fl.split("(")[0]] += 1
    print("\n==== 20-CASE QUICK ====")
    print(f"  flagged: {len(flagged)}/20   breakdown: {dict(fc)}")
    if valid:
        ades = sorted(r["ade"] for r in valid)
        print(f"  ADE median={ades[len(ades)//2]:.2f}m mean={sum(ades)/len(ades):.2f}m")
    lats=[r['latency'] for r in valid]
    print(f"  lat mean={sum(lats)/len(lats):.2f}s")
    print("  --- explanation lengths + sample ---")
    for r in results[:6]:
        print(f"    idx={r['idx']} len={len(r['explanation'])} flags={r['flags']}: {r['explanation'][:120]}")
    json.dump(results, open("/tmp/test20.json","w"), indent=2)


if __name__ == "__main__":
    main()
