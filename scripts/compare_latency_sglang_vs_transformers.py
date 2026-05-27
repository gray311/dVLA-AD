"""Compare latency + ADE: SGLang vs transformers paths on the SAME 20 samples
with the SAME template config (V3 semantic trajectory + complexity slot).

Both paths:
  - same Fast_dVLM_3B model
  - same V3 prompt (3s history @ 0.5s spacing)
  - same template scaffold (semantic trajectory, complexity slot, all gates)
  - nav_command injection DISABLED in both (transformers doesn't support it)
  - temperature=0
  - same picked sample list

Two passes in separate processes (Prometheus CollectorRegistry collides
when both engines load in the same process).

Usage:
  python scripts/compare_latency_sglang_vs_transformers.py sglang
  python scripts/compare_latency_sglang_vs_transformers.py transformers
  python scripts/compare_latency_sglang_vs_transformers.py report

Files:
  results/latency_compare/sglang.json
  results/latency_compare/transformers.json
  results/latency_compare/report.md
"""
import json
import math
import os
import sys
import traceback

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, parse_filled

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
# 20 stratified picks (subset of the 50-sample run for time).
PICKED = [20, 1, 25, 107, 327, 14, 48, 281, 204, 175, 250, 300, 4, 67, 89,
          150, 200, 350, 400, 444]
OUT_DIR = os.path.join(ROOT, "results", "latency_compare")


def _fix(p):
    return p.replace(PATH_FIX[0], PATH_FIX[1])


def _front(s):
    return _fix(s["image"][1])


def _ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0:
        return None
    return sum(
        math.hypot(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1])
        for i in range(n)
    ) / n


def run_sglang(picked, out_path):
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    print("Loading SGLang...")
    bundle = loader.load(algorithm="mdm")
    print("Warmup...")
    loader.generate(bundle, [_front(picked[0])], build_prompt_v3(picked[0]),
                     temperature=0.0)  # no nav_command for fairness
    print("Running...")
    out = []
    for i, s in enumerate(picked):
        text, lat = loader.generate(bundle, [_front(s)], build_prompt_v3(s),
                                     temperature=0.0)
        pred = parse_filled(text)
        gt = s["future waypoints"][:5]
        a = _ade(pred, gt)
        vx, vy = s["velocity"][-1]
        out.append({
            "sample_id": s["sample_id"], "nav": s["navigation_command"],
            "speed": math.hypot(vx, vy), "latency_s": lat, "ade": a,
            "output": text,
        })
        print(f"  [{i+1}/{len(picked)}] lat={lat:.2f}s ADE={a:.2f}m")
    loader.shutdown(bundle)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


def run_transformers(picked, out_path):
    from eval.loaders import fast_dvlm_v3 as loader
    print("Loading transformers...")
    bundle = loader.load()
    print("Warmup...")
    loader.generate(bundle, [_front(picked[0])], build_prompt_v3(picked[0]),
                     gen_length=512, steps=8, temperature=0.0)
    print("Running...")
    out = []
    for i, s in enumerate(picked):
        text, lat = loader.generate(bundle, [_front(s)], build_prompt_v3(s),
                                     gen_length=512, steps=8, temperature=0.0)
        pred = parse_filled(text)
        gt = s["future waypoints"][:5]
        a = _ade(pred, gt)
        vx, vy = s["velocity"][-1]
        out.append({
            "sample_id": s["sample_id"], "nav": s["navigation_command"],
            "speed": math.hypot(vx, vy), "latency_s": lat, "ade": a,
            "output": text,
        })
        print(f"  [{i+1}/{len(picked)}] lat={lat:.2f}s ADE={a:.2f}m")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


def report():
    sg = json.load(open(os.path.join(OUT_DIR, "sglang.json")))
    tr = json.load(open(os.path.join(OUT_DIR, "transformers.json")))
    sg_by = {r["sample_id"]: r for r in sg}
    tr_by = {r["sample_id"]: r for r in tr}
    common_ids = set(sg_by) & set(tr_by)
    common = sorted(common_ids)

    lines = []
    lines.append("# Latency comparison: SGLang vs transformers")
    lines.append("")
    lines.append("Both paths run Fast_dVLM_3B with the same V3 template "
                 "(semantic trajectory + complexity slot) and same prompt. "
                 "nav_command injection disabled in both for apples-to-apples.")
    lines.append("")
    lines.append("| # | sample | nav | speed | SGLang lat | transformers lat | speedup | SGLang ADE | transformers ADE |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    sg_lats, tr_lats, sg_ades, tr_ades = [], [], [], []
    for i, sid in enumerate(common):
        s = sg_by[sid]
        t = tr_by[sid]
        sg_lats.append(s["latency_s"]); tr_lats.append(t["latency_s"])
        if s["ade"] is not None: sg_ades.append(s["ade"])
        if t["ade"] is not None: tr_ades.append(t["ade"])
        speedup = t["latency_s"] / s["latency_s"] if s["latency_s"] > 0 else 0
        lines.append(
            f"| {i+1} | {sid[:8]} | {s['nav']} | {s['speed']:.1f}"
            f" | {s['latency_s']:.2f}s | {t['latency_s']:.2f}s | {speedup:.2f}x"
            f" | {s['ade']:.2f}m | {t['ade']:.2f}m |"
        )
    n = len(common)
    lines.append(f"| **mean** |  |  |  | **{sum(sg_lats)/n:.2f}s** | **{sum(tr_lats)/n:.2f}s**"
                 f" | **{(sum(tr_lats)/n) / (sum(sg_lats)/n):.2f}x** | **{sum(sg_ades)/len(sg_ades):.2f}m**"
                 f" | **{sum(tr_ades)/len(tr_ades):.2f}m** |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **N = {n}** samples")
    lines.append(f"- **SGLang**: mean latency {sum(sg_lats)/n:.2f}s, "
                 f"min {min(sg_lats):.2f}s, max {max(sg_lats):.2f}s")
    lines.append(f"- **transformers**: mean latency {sum(tr_lats)/n:.2f}s, "
                 f"min {min(tr_lats):.2f}s, max {max(tr_lats):.2f}s")
    speedup = (sum(tr_lats)/n) / (sum(sg_lats)/n)
    lines.append(f"- **SGLang speedup**: {speedup:.2f}x")
    lines.append("")
    lines.append(f"- **Mean ADE**: SGLang {sum(sg_ades)/len(sg_ades):.2f}m, "
                 f"transformers {sum(tr_ades)/len(tr_ades):.2f}m "
                 f"(same template ⇒ similar quality)")
    out_path = os.path.join(OUT_DIR, "report.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    os.makedirs(OUT_DIR, exist_ok=True)
    if cmd == "report":
        report()
        return
    data = json.load(open(DATA))
    picked = [data[i] for i in PICKED]
    print(f"Picked {len(picked)} samples")
    if cmd == "sglang":
        run_sglang(picked, os.path.join(OUT_DIR, "sglang.json"))
    elif cmd == "transformers":
        run_transformers(picked, os.path.join(OUT_DIR, "transformers.json"))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
