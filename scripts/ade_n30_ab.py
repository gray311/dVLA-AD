"""A/B run: baseline (steps=4) vs threshold=0.5/max_iter=3, on N stratified samples.

Two independent subprocesses to avoid SGLang prometheus / mp re-init issues.

Usage:
  python scripts/ade_n30_ab.py [N=30]
"""
import json, math, os, subprocess, sys, time

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0: return None
    return sum(math.hypot(pred[i][0]-gt[i][0], pred[i][1]-gt[i][1]) for i in range(n)) / n


def fde(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0: return None
    return math.hypot(pred[n-1][0]-gt[n-1][0], pred[n-1][1]-gt[n-1][1])


def pick_stratified(data, n):
    """Same as run_n_waymo_ade.py."""
    buckets = {
        "GO_LEFT": [], "GO_RIGHT": [],
        "GO_STRAIGHT_stopped": [], "GO_STRAIGHT_slow": [],
        "GO_STRAIGHT_mid": [], "GO_STRAIGHT_fast": [],
    }
    for i, s in enumerate(data):
        nav = s["navigation_command"]
        vx, vy = s["velocity"][-1]
        sp = math.hypot(vx, vy)
        if nav == "GO_LEFT": buckets["GO_LEFT"].append((i, sp))
        elif nav == "GO_RIGHT": buckets["GO_RIGHT"].append((i, sp))
        elif sp < 1: buckets["GO_STRAIGHT_stopped"].append((i, sp))
        elif sp < 5: buckets["GO_STRAIGHT_slow"].append((i, sp))
        elif sp < 15: buckets["GO_STRAIGHT_mid"].append((i, sp))
        else: buckets["GO_STRAIGHT_fast"].append((i, sp))
    per_bucket = max(1, n // len(buckets))
    picks = []
    for k, v in buckets.items():
        v_sorted = sorted(v, key=lambda x: x[1])
        if not v_sorted: continue
        step = max(1, len(v_sorted) // per_bucket)
        added = 0
        for j in range(0, len(v_sorted), step):
            picks.append(v_sorted[j][0])
            added += 1
            if added >= per_bucket: break
    return sorted(set(picks))[:n]


def run_phase(label, indices_json, out_json):
    """Subprocess: load engine + run all picked samples."""
    from eval.template_v3 import build_prompt_v3, parse_filled
    from eval.loaders import fast_dvlm_sglang_v3 as loader

    indices = json.load(open(indices_json))
    data = json.load(open(DATA))
    samples = [(i, data[i]) for i in indices]

    print(f"[{label}] loading engine (thresh={os.environ.get('SGLANG_DLLM_THRESHOLD', '-')}, "
          f"max_iter={os.environ.get('SGLANG_DLLM_MAX_ITER', '-')})", flush=True)
    bundle = loader.load(algorithm="mdm")

    print(f"[{label}] warmup...", flush=True)
    _, _ = loader.generate(
        bundle, [_fix(samples[0][1]['image'][1])],
        build_prompt_v3(samples[0][1]),
        temperature=0.0, nav_command=samples[0][1]['navigation_command'],
    )

    results = []
    for k, (idx, s) in enumerate(samples):
        try:
            text, latency = loader.generate(
                bundle, [_fix(s['image'][1])], build_prompt_v3(s),
                temperature=0.0, nav_command=s['navigation_command'],
            )
            pred = parse_filled(text)
            gt = s["future waypoints"][:5]
            a = ade(pred, gt) if pred else None
            f_ = fde(pred, gt) if pred else None
            vx, vy = s["velocity"][-1]
            speed = math.hypot(vx, vy)
            results.append({
                "idx": idx, "nav": s["navigation_command"], "speed": speed,
                "ade": a, "fde": f_, "latency": latency,
            })
            if (k + 1) % 5 == 0 or k == 0:
                print(f"  [{label}] [{k+1}/{len(samples)}] idx={idx:3} "
                      f"ADE={'N/A' if a is None else f'{a:5.2f}m'} lat={latency:.2f}s",
                      flush=True)
        except Exception as e:
            print(f"  [{label}] idx={idx} ERROR: {e}", flush=True)
            results.append({"idx": idx, "error": str(e)})

    loader.shutdown(bundle)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{label}] saved {out_json}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--phase":
        run_phase(sys.argv[2], sys.argv[3], sys.argv[4])
        return

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    data = json.load(open(DATA))
    indices = pick_stratified(data, n)
    print(f"Picked {len(indices)} stratified samples\n", flush=True)
    idx_json = "/tmp/ade_n30_indices.json"
    with open(idx_json, "w") as f:
        json.dump(indices, f)

    configs = [("A_steps4", {}, "/tmp/ade_n30_A.json"),
               ("D_thr0.5_mi3",
                {"SGLANG_DLLM_THRESHOLD": "0.5", "SGLANG_DLLM_MAX_ITER": "3"},
                "/tmp/ade_n30_D.json")]

    all_results = {}
    for label, env_set, out_json in configs:
        env = dict(os.environ)
        env.pop("SGLANG_DLLM_THRESHOLD", None)
        env.pop("SGLANG_DLLM_MAX_ITER", None)
        env.update(env_set)
        env["DLLM_FWD_LOG"] = ""  # disable per-chunk print for cleaner log
        print(f"\n========== {label} ==========", flush=True)
        rc = subprocess.call(
            [sys.executable, "-u", __file__, "--phase", label, idx_json, out_json],
            env=env,
        )
        if rc != 0:
            print(f"{label} failed (rc={rc})"); continue
        all_results[label] = json.load(open(out_json))

    # Compare
    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'config':<16} | {'mean_lat':>9} | {'mean_ADE':>9} | {'median_ADE':>10} | {'p90_ADE':>8}", flush=True)
    for label, _, _ in configs:
        if label not in all_results: continue
        rs = [r for r in all_results[label] if r.get("ade") is not None]
        if not rs: continue
        ades = sorted(r["ade"] for r in rs)
        lats = [r["latency"] for r in rs]
        n_valid = len(ades)
        p90 = ades[max(0, int(0.9 * n_valid) - 1)]
        median = ades[n_valid // 2]
        print(f"{label:<16} | {sum(lats)/len(lats):>8.2f}s | {sum(ades)/n_valid:>8.2f}m | {median:>9.2f}m | {p90:>7.2f}m",
              flush=True)

    # Per-sample paired delta
    if "A_steps4" in all_results and "D_thr0.5_mi3" in all_results:
        A = {r["idx"]: r for r in all_results["A_steps4"]}
        D = {r["idx"]: r for r in all_results["D_thr0.5_mi3"]}
        common = sorted(set(A.keys()) & set(D.keys()))
        win = lose = tie = 0
        ade_deltas = []
        lat_deltas = []
        for i in common:
            a, d = A[i], D[i]
            if a.get("ade") is None or d.get("ade") is None: continue
            delta = d["ade"] - a["ade"]
            ade_deltas.append(delta)
            lat_deltas.append(d["latency"] - a["latency"])
            if delta < -0.1: win += 1
            elif delta > 0.1: lose += 1
            else: tie += 1
        print(f"\nD vs A paired: win={win} lose={lose} tie={tie} (n={len(ade_deltas)})", flush=True)
        if ade_deltas:
            print(f"  mean Δ ADE = {sum(ade_deltas)/len(ade_deltas):+.3f}m  "
                  f"mean Δ lat = {sum(lat_deltas)/len(lat_deltas):+.3f}s", flush=True)


if __name__ == "__main__":
    main()
