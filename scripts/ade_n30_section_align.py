"""N=30 A/B: baseline vs section_align bs=160.

Same stratified sampling as ade_n30_ab.py.
"""
import json, math, os, subprocess, sys

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


def run_phase(label, section_align, block_size, idx_json, out_json):
    from eval.template_v3 import build_prompt_v3, parse_filled
    from eval.loaders import fast_dvlm_sglang_v3 as loader

    indices = json.load(open(idx_json))
    data = json.load(open(DATA))
    samples = [(i, data[i]) for i in indices]

    print(f"[{label}] loading engine (section_align={section_align}, block_size={block_size})", flush=True)
    if section_align:
        bundle = loader.load(algorithm="mdm", engine_block_size=block_size)
    else:
        bundle = loader.load(algorithm="mdm")  # default bs=32

    gen_kwargs = dict(temperature=0.0)
    if section_align:
        gen_kwargs["section_align"] = True
        gen_kwargs["block_size"] = block_size

    print(f"[{label}] warmup...", flush=True)
    _, _ = loader.generate(
        bundle, [_fix(samples[0][1]['image'][1])],
        build_prompt_v3(samples[0][1]),
        nav_command=samples[0][1]['navigation_command'],
        **gen_kwargs,
    )

    results = []
    for k, (idx, s) in enumerate(samples):
        try:
            text, latency = loader.generate(
                bundle, [_fix(s['image'][1])], build_prompt_v3(s),
                nav_command=s['navigation_command'],
                **gen_kwargs,
            )
            pred = parse_filled(text)
            gt = s["future waypoints"][:5]
            a = ade(pred, gt) if pred else None
            f_ = fde(pred, gt) if pred else None
            results.append({"idx": idx, "nav": s["navigation_command"],
                            "ade": a, "fde": f_, "latency": latency})
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
        run_phase(sys.argv[2], sys.argv[3] == "true", int(sys.argv[4]), sys.argv[5], sys.argv[6])
        return

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    data = json.load(open(DATA))
    indices = pick_stratified(data, n)
    print(f"Picked {len(indices)} stratified samples\n", flush=True)
    idx_json = "/tmp/ade_n30_SA_indices.json"
    with open(idx_json, "w") as f:
        json.dump(indices, f)

    configs = [
        ("A_baseline",     "false", 32,  "/tmp/ade30_SA_A.json"),
        ("B_uniform160",   "true",  160, "/tmp/ade30_SA_B.json"),
    ]

    all_results = {}
    for label, sa, bs, out_json in configs:
        env = dict(os.environ)
        env["DLLM_FWD_LOG"] = ""
        print(f"\n========== {label} ==========", flush=True)
        rc = subprocess.call(
            [sys.executable, "-u", __file__, "--phase", label, sa, str(bs), idx_json, out_json],
            env=env,
        )
        if rc != 0:
            print(f"{label} failed (rc={rc})"); continue
        all_results[label] = json.load(open(out_json))

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'config':<16} | {'mean_lat':>9} | {'mean_ADE':>9} | {'median_ADE':>10} | {'p90_ADE':>8}", flush=True)
    for label, _, _, _ in configs:
        if label not in all_results: continue
        rs = [r for r in all_results[label] if r.get("ade") is not None]
        if not rs: continue
        ades = sorted(r["ade"] for r in rs)
        lats = [r["latency"] for r in rs]
        n_v = len(ades)
        p90 = ades[max(0, int(0.9 * n_v) - 1)]
        median = ades[n_v // 2]
        print(f"{label:<16} | {sum(lats)/len(lats):>8.2f}s | {sum(ades)/n_v:>8.2f}m | {median:>9.2f}m | {p90:>7.2f}m", flush=True)

    if "A_baseline" in all_results and "B_uniform160" in all_results:
        A = {r["idx"]: r for r in all_results["A_baseline"]}
        B = {r["idx"]: r for r in all_results["B_uniform160"]}
        common = sorted(set(A.keys()) & set(B.keys()))
        win = lose = tie = 0
        ade_d = []; lat_d = []
        for i in common:
            a, b = A[i], B[i]
            if a.get("ade") is None or b.get("ade") is None: continue
            delta = b["ade"] - a["ade"]
            ade_d.append(delta)
            lat_d.append(b["latency"] - a["latency"])
            if delta < -0.1: win += 1
            elif delta > 0.1: lose += 1
            else: tie += 1
        print(f"\nB vs A paired: win={win} lose={lose} tie={tie} (n={len(ade_d)})", flush=True)
        if ade_d:
            print(f"  mean Δ ADE = {sum(ade_d)/len(ade_d):+.3f}m  "
                  f"mean Δ lat = {sum(lat_d)/len(lat_d):+.3f}s", flush=True)


if __name__ == "__main__":
    main()
