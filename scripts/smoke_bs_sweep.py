"""Sweep block_size in section-aligned mode on 3 samples.

Each config = (engine_block_size, generate block_size=same).
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


def run_phase(label, bs, out_json):
    from eval.template_v3 import build_prompt_v3, parse_filled
    from eval.loaders import fast_dvlm_sglang_v3 as loader

    data = json.load(open(DATA))
    samples = [(i, data[i]) for i in PICKED_IDX]

    print(f"[{label}] loading engine bs={bs}...", flush=True)
    bundle = loader.load(algorithm="mdm", engine_block_size=bs)

    gen_kwargs = dict(temperature=0.0, section_align=True, block_size=bs)

    print(f"[{label}] warmup...", flush=True)
    _, _ = loader.generate(
        bundle, [_fix(samples[0][1]['image'][1])],
        build_prompt_v3(samples[0][1]),
        nav_command=samples[0][1]['navigation_command'],
        **gen_kwargs,
    )

    results = []
    for idx, s in samples:
        text, latency = loader.generate(
            bundle, [_fix(s['image'][1])], build_prompt_v3(s),
            nav_command=s['navigation_command'],
            **gen_kwargs,
        )
        pred = parse_filled(text)
        gt = s["future waypoints"][:5]
        a = ade(pred, gt) if pred else None
        results.append({"idx": idx, "ade": a, "latency": latency, "text": text})
        print(f"  [{label}] idx={idx:3} ADE={'N/A' if a is None else f'{a:5.2f}m'} lat={latency:.2f}s", flush=True)

    loader.shutdown(bundle)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--phase":
        run_phase(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        return

    for bs in (64, 96, 128, 160):
        env = dict(os.environ)
        env["DLLM_FWD_LOG"] = ""
        out_json = f"/tmp/svb_bs{bs}.json"
        print(f"\n========== bs={bs} ==========", flush=True)
        rc = subprocess.call(
            [sys.executable, "-u", __file__, "--phase", f"bs{bs}", str(bs), out_json],
            env=env,
        )
        if rc != 0:
            print(f"bs={bs} failed (rc={rc})"); continue

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'config':<8} | {'mean_lat':>9} | {'mean_ADE':>9} | per-sample ADE", flush=True)
    for bs in (64, 96, 128, 160):
        path = f"/tmp/svb_bs{bs}.json"
        if not os.path.exists(path):
            continue
        rs = json.load(open(path))
        lats = [r['latency'] for r in rs]
        ades = [r['ade'] for r in rs if r['ade'] is not None]
        mean_lat = sum(lats)/len(lats) if lats else 0
        mean_ade = sum(ades)/len(ades) if ades else float('nan')
        per = ", ".join(f"{r['ade']:.2f}" if r['ade'] is not None else "N/A" for r in rs)
        print(f"bs={bs:<5} | {mean_lat:>8.2f}s | {mean_ade:>8.2f}m | {per}", flush=True)


if __name__ == "__main__":
    main()
