"""Smoke test for SGLANG_DLLM_THRESHOLD mode.

Runs the SAME 3 Waymo samples under two SEPARATE subprocess invocations:
  1. Baseline (no env → fixed 4 steps/chunk)
  2. Threshold mode (SGLANG_DLLM_THRESHOLD set)

Two subprocesses (instead of two engine loads in one process) because
SGLang's prometheus / multiprocessing state can't cleanly tear down and
re-initialise in the same Python process.

Usage:
  python scripts/smoke_threshold_mode.py [threshold] [max_iter]

Internal: when invoked with `--phase {base,thr}`, runs ONE phase and
writes results to /tmp/smoke_phase_<name>.json.
"""
import json, math, os, subprocess, sys, time

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")

PICKED_IDX = [202, 244, 59]  # GO_LEFT, GO_RIGHT, GO_RIGHT (mixed)


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0:
        return None
    return sum(
        math.hypot(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1])
        for i in range(n)
    ) / n


def run_single_phase(phase_name, out_json_path):
    """Inside subprocess: load engine ONCE for this phase, run all samples."""
    from eval.template_v3 import build_prompt_v3, parse_filled
    from eval.loaders import fast_dvlm_sglang_v3 as loader

    data = json.load(open(DATA))
    samples = [(i, data[i]) for i in PICKED_IDX]

    mf = float(os.environ.get("SGLANG_MEM_FRACTION", "0.75"))
    disable_cg = os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "") not in ("", "0", "false")
    print(f"\n[{phase_name}] loading engine (mem_fraction={mf}, "
          f"disable_cg={disable_cg}, SGLANG_DLLM_THRESHOLD="
          f"{os.environ.get('SGLANG_DLLM_THRESHOLD', '(unset)')}, "
          f"SGLANG_DLLM_MAX_ITER={os.environ.get('SGLANG_DLLM_MAX_ITER', '(unset)')})",
          flush=True)
    bundle = loader.load(
        algorithm="mdm", mem_fraction_static=mf, disable_cuda_graph=disable_cg,
    )

    print(f"[{phase_name}] warmup...", flush=True)
    _, _ = loader.generate(
        bundle, [_fix(samples[0][1]['image'][1])],
        build_prompt_v3(samples[0][1]),
        temperature=0.0, nav_command=samples[0][1]['navigation_command'],
    )

    results = []
    for idx, s in samples:
        text, latency = loader.generate(
            bundle, [_fix(s['image'][1])], build_prompt_v3(s),
            temperature=0.0, nav_command=s['navigation_command'],
        )
        pred = parse_filled(text)
        gt = s["future waypoints"][:5]
        a = ade(pred, gt) if pred else None
        results.append({
            "idx": idx, "nav": s["navigation_command"],
            "ade": a, "latency": latency,
            "text": text,
        })
        print(f"  [{phase_name}] idx={idx:3} nav={s['navigation_command']:14} "
              f"ADE={'N/A' if a is None else f'{a:5.2f}m'} "
              f"lat={latency:.2f}s", flush=True)

    loader.shutdown(bundle)

    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{phase_name}] saved {out_json_path}", flush=True)


def main():
    # Subprocess entry: --phase {base,thr}
    if len(sys.argv) > 1 and sys.argv[1] == "--phase":
        phase = sys.argv[2]
        out_json = sys.argv[3]
        run_single_phase(phase, out_json)
        return

    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    print(f"\n=== Smoke test: threshold={thresh}, max_iter={max_iter} ===\n",
          flush=True)

    # Phase A: baseline (subprocess, env clean)
    base_env = dict(os.environ)
    base_env.pop("SGLANG_DLLM_THRESHOLD", None)
    base_env.pop("SGLANG_DLLM_MAX_ITER", None)
    base_env["DLLM_FWD_LOG"] = "1"
    base_out = "/tmp/smoke_phase_base.json"
    rc = subprocess.call(
        [sys.executable, "-u", __file__, "--phase", "BASE", base_out],
        env=base_env,
    )
    if rc != 0:
        print(f"BASE phase failed (rc={rc})")
        return

    # Phase B: threshold mode (subprocess, env set)
    thr_env = dict(os.environ)
    thr_env["SGLANG_DLLM_THRESHOLD"] = str(thresh)
    thr_env["SGLANG_DLLM_MAX_ITER"] = str(max_iter)
    thr_env["DLLM_FWD_LOG"] = "1"
    thr_out = "/tmp/smoke_phase_thr.json"
    rc = subprocess.call(
        [sys.executable, "-u", __file__, "--phase", "THR ", thr_out],
        env=thr_env,
    )
    if rc != 0:
        print(f"THR phase failed (rc={rc})")
        return

    base = json.load(open(base_out))
    thresh_res = json.load(open(thr_out))

    print(f"\n=== Comparison ===")
    print(f"{'idx':>4} | {'BASE_ade':>10} {'BASE_lat':>10} | {'THR_ade':>10} {'THR_lat':>10}")
    for b, t in zip(base, thresh_res):
        bade = f"{b['ade']:.2f}m" if b['ade'] is not None else "N/A"
        tade = f"{t['ade']:.2f}m" if t['ade'] is not None else "N/A"
        print(f"{b['idx']:>4} | {bade:>10} {b['latency']:>9.2f}s | "
              f"{tade:>10} {t['latency']:>9.2f}s")

    base_lats = [r['latency'] for r in base]
    thr_lats = [r['latency'] for r in thresh_res]
    print(f"\nmean lat: BASE={sum(base_lats)/len(base_lats):.2f}s  "
          f"THR={sum(thr_lats)/len(thr_lats):.2f}s")

    valid_b = [r['ade'] for r in base if r['ade'] is not None]
    valid_t = [r['ade'] for r in thresh_res if r['ade'] is not None]
    if valid_b and valid_t:
        print(f"mean ADE: BASE={sum(valid_b)/len(valid_b):.2f}m  "
              f"THR={sum(valid_t)/len(valid_t):.2f}m")


if __name__ == "__main__":
    main()
