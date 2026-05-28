"""Load Fast-dDrive weights via OUR SGLang fork (using our HierarchyBlock
template-mode + V3 schema). Measure latency.

Setup: dDrive weights converted to Fast-dVLM key naming
       (see convert_ddrive_to_dvlm.py).
"""
import json, math, os, sys, time

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, parse_filled

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
PICKED_IDX = [202, 244, 59]
DDRIVE_AS_DVLM = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dDrive_as_dVLM"


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0: return None
    return sum(math.hypot(pred[i][0]-gt[i][0], pred[i][1]-gt[i][1]) for i in range(n)) / n


def main():
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    print(f"Loading dDrive weights via SGLang from {DDRIVE_AS_DVLM} ...", flush=True)
    bundle = loader.load(
        model_path=DDRIVE_AS_DVLM,
        algorithm="mdm",
        engine_block_size=32,
    )

    data = json.load(open(DATA))
    samples = [(i, data[i]) for i in PICKED_IDX]

    print("Warmup...", flush=True)
    _, _ = loader.generate(
        bundle, [_fix(samples[0][1]['image'][1])],
        build_prompt_v3(samples[0][1]),
        temperature=0.0, block_size=32, section_align=False,
        nav_command=samples[0][1]['navigation_command'],
    )

    results = []
    for idx, s in samples:
        text, latency = loader.generate(
            bundle, [_fix(s['image'][1])], build_prompt_v3(s),
            temperature=0.0, block_size=32, section_align=False,
            nav_command=s['navigation_command'],
        )
        pred = parse_filled(text)
        gt = s["future waypoints"][:5]
        a = ade(pred, gt) if pred else None
        results.append({"idx": idx, "ade": a, "latency": latency, "text": text})
        print(f"  idx={idx:3} ADE={'N/A' if a is None else f'{a:5.2f}m'} lat={latency:.2f}s", flush=True)

    loader.shutdown(bundle)
    with open("/tmp/ddrive_via_sglang.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n--- summary ---")
    lats = [r['latency'] for r in results]
    ades = [r['ade'] for r in results if r['ade'] is not None]
    print(f"mean_lat = {sum(lats)/len(lats):.2f}s")
    if ades: print(f"mean_ADE = {sum(ades)/len(ades):.2f}m")


if __name__ == "__main__":
    main()
