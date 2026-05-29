"""A/B: explanation parallel top-K vs strict L2R/AR. Same samples, print
explanation + latency for each mode to see if L2R removes the glue artifacts."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))
from eval.template_v3 import build_prompt_v3
DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
fix = lambda p: p.replace("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
IDX = [281, 204, 327, 14]


def _exp(text):
    try:
        return json.loads(text, strict=False).get("explanation", "<no-key>")
    except Exception as e:
        return f"<unparsed: {e}>"


def main():
    data = json.load(open(DATA))
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    b = loader.load(algorithm="mdm")
    s0 = data[281]
    loader.generate(b, [fix(s0["image"][1])], build_prompt_v3(s0), temperature=0.0,
                    nav_command=s0["navigation_command"])  # warmup
    for idx in IDX:
        s = data[idx]
        img = fix(s["image"][1]); prompt = build_prompt_v3(s); nav = s["navigation_command"]
        print("\n" + "=" * 92)
        print(f"idx{idx}  nav={nav}")
        for label, l2r in (("PARALLEL", False), ("L2R/AR ", True)):
            text, lat = loader.generate(b, [img], prompt, temperature=0.0,
                                        nav_command=nav, explanation_l2r=l2r)
            print(f"\n--- {label}  ({lat:.2f}s) ---")
            print(_exp(text))
    loader.shutdown(b)


if __name__ == "__main__":
    main()
