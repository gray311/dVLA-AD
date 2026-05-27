"""Sweep steps_per_chunk and threshold configs for latency-vs-ADE tradeoff.

Runs the same 3 samples under 4 configurations, each in its own subprocess
(to avoid SGLang re-init issues):
  A. steps=4 (current baseline)
  B. steps=3 (~25% fewer forwards)
  C. steps=2 (~50% fewer forwards)
  D. threshold=0.5, max_iter=3 (Fast-dDrive style, capped)

Usage: python scripts/smoke_steps_sweep.py
Internal: --phase {label} {steps_per_chunk} {threshold} {max_iter} {out_json}
"""
import json, math, os, subprocess, sys

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
PICKED_IDX = [202, 244, 59]


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0: return None
    return sum(math.hypot(pred[i][0]-gt[i][0], pred[i][1]-gt[i][1]) for i in range(n)) / n


def run_phase(label, steps_per_chunk, out_json):
    """In subprocess: load engine + run 3 samples."""
    from eval.template_v3 import build_prompt_v3, parse_filled
    from eval.loaders import fast_dvlm_sglang_v3 as loader

    data = json.load(open(DATA))
    samples = [(i, data[i]) for i in PICKED_IDX]

    print(f"[{label}] loading engine (steps_per_chunk={steps_per_chunk}, "
          f"thresh={os.environ.get('SGLANG_DLLM_THRESHOLD', '-')}, "
          f"max_iter={os.environ.get('SGLANG_DLLM_MAX_ITER', '-')})", flush=True)
    bundle = loader.load(algorithm="mdm")
    print(f"[{label}] warmup...", flush=True)
    _, _ = loader.generate(
        bundle, [_fix(samples[0][1]['image'][1])],
        build_prompt_v3(samples[0][1]),
        temperature=0.0, nav_command=samples[0][1]['navigation_command'],
        steps_per_chunk=steps_per_chunk,
    )

    results = []
    for idx, s in samples:
        text, latency = loader.generate(
            bundle, [_fix(s['image'][1])], build_prompt_v3(s),
            temperature=0.0, nav_command=s['navigation_command'],
            steps_per_chunk=steps_per_chunk,
        )
        pred = parse_filled(text)
        gt = s["future waypoints"][:5]
        a = ade(pred, gt) if pred else None
        results.append({"idx": idx, "ade": a, "latency": latency})
        print(f"  [{label}] idx={idx:3} ADE={'N/A' if a is None else f'{a:5.2f}m'} lat={latency:.2f}s", flush=True)

    loader.shutdown(bundle)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{label}] saved {out_json}", flush=True)


def main():
    # Subprocess entry: --phase {label} {steps} {thresh} {max_iter} {out_json}
    if len(sys.argv) > 1 and sys.argv[1] == "--phase":
        label = sys.argv[2]
        steps = int(sys.argv[3])
        run_phase(label, steps, sys.argv[6])
        return

    # Top-level: run 4 configs in subprocesses
    configs = [
        # (label,         steps, threshold, max_iter)
        ("A_steps4",      4,     "",       ""),
        ("B_steps3",      3,     "",       ""),
        ("C_steps2",      2,     "",       ""),
        ("D_thr0.5_mi3",  3,     "0.5",    "3"),  # steps_per_chunk only used in fixed mode; threshold overrides loop control
    ]

    all_results = {}
    for label, steps, thresh, max_iter in configs:
        env = dict(os.environ)
        if thresh:
            env["SGLANG_DLLM_THRESHOLD"] = thresh
            env["SGLANG_DLLM_MAX_ITER"] = max_iter
        else:
            env.pop("SGLANG_DLLM_THRESHOLD", None)
            env.pop("SGLANG_DLLM_MAX_ITER", None)
        env["DLLM_FWD_LOG"] = "1"
        out_json = f"/tmp/sweep_{label}.json"
        print(f"\n========== {label} (steps={steps} thresh={thresh or '-'} max_iter={max_iter or '-'}) ==========",
              flush=True)
        rc = subprocess.call(
            [sys.executable, "-u", __file__, "--phase", label, str(steps),
             thresh or "0", max_iter or "0", out_json],
            env=env,
        )
        if rc != 0:
            print(f"{label} failed (rc={rc})")
            continue
        all_results[label] = json.load(open(out_json))

    # Summary table
    print("\n========== SWEEP SUMMARY ==========")
    print(f"{'config':<16} | {'mean_lat':>9} | {'mean_ADE':>9} | per-sample ADE")
    for label, _, _, _ in configs:
        if label not in all_results:
            continue
        rs = all_results[label]
        lats = [r['latency'] for r in rs]
        ades = [r['ade'] for r in rs if r['ade'] is not None]
        mean_lat = sum(lats) / len(lats)
        mean_ade = sum(ades) / len(ades) if ades else float('nan')
        per_ade = ", ".join(f"{r['ade']:.2f}" if r['ade'] is not None else "N/A" for r in rs)
        print(f"{label:<16} | {mean_lat:>8.2f}s | {mean_ade:>8.2f}m | {per_ade}")


if __name__ == "__main__":
    main()
