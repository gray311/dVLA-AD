"""Smoke test: per-section variable chunk sizes (no padding waste).

Configs:
  A: baseline (block_size=32, no section_align)
  B: section_align=True, block_size=160 (uniform-padded sections)
  C: section_align="variable", max_chunk_size=192 — natural chunk sizes
  D: section_align="variable", max_chunk_size=256 — traj in 1 chunk
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


def run_phase(label, block_size, section_align, max_chunk_size, out_json):
    from eval.template_v3 import build_prompt_v3, parse_filled
    from eval.loaders import fast_dvlm_sglang_v3 as loader

    data = json.load(open(DATA))
    samples = [(i, data[i]) for i in PICKED_IDX]

    print(f"[{label}] loading engine (block_size={block_size}, "
          f"section_align={section_align}, max_chunk_size={max_chunk_size})", flush=True)
    # For variable mode, engine_block_size must be ≥ max chunk size (KV buffers
    # are pre-allocated for that shape). For uniform-160 mode, ≥ 160.
    if section_align == "variable":
        eng_bs = max_chunk_size
    elif section_align:
        eng_bs = block_size
    else:
        eng_bs = 32
    bundle = loader.load(algorithm="mdm", engine_block_size=eng_bs)

    gen_kwargs = dict(temperature=0.0, block_size=block_size)
    if section_align == "variable":
        gen_kwargs["section_align"] = "variable"
        gen_kwargs["max_chunk_size"] = max_chunk_size
    elif section_align:
        gen_kwargs["section_align"] = True

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
    print(f"[{label}] saved {out_json}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--phase":
        label = sys.argv[2]
        bs = int(sys.argv[3])
        sa = sys.argv[4]
        if sa == "false":
            sa = False
        elif sa == "true":
            sa = True
        mcs = int(sys.argv[5])
        out_json = sys.argv[6]
        run_phase(label, bs, sa, mcs, out_json)
        return

    configs = [
        # (label,         block_size, section_align,  max_chunk_size)
        ("A_baseline",      32,  "false",   0),
        ("B_uniform160",    160, "true",    0),
        ("C_variable192",   32,  "variable", 192),  # block_size unused in variable mode
        ("D_variable256",   32,  "variable", 256),
    ]

    all_results = {}
    for label, bs, sa, mcs in configs:
        env = dict(os.environ)
        env["DLLM_FWD_LOG"] = "1"
        out_json = f"/tmp/svar_{label}.json"
        print(f"\n========== {label} ==========", flush=True)
        rc = subprocess.call(
            [sys.executable, "-u", __file__, "--phase", label,
             str(bs), sa, str(mcs), out_json],
            env=env,
        )
        if rc != 0:
            print(f"{label} failed (rc={rc})"); continue
        all_results[label] = json.load(open(out_json))

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'config':<18} | {'mean_lat':>9} | {'mean_ADE':>9} | per-sample ADE", flush=True)
    for label, _, _, _ in configs:
        if label not in all_results: continue
        rs = all_results[label]
        lats = [r['latency'] for r in rs]
        ades = [r['ade'] for r in rs if r['ade'] is not None]
        mean_lat = sum(lats)/len(lats) if lats else 0
        mean_ade = sum(ades)/len(ades) if ades else float('nan')
        per = ", ".join(f"{r['ade']:.2f}" if r['ade'] is not None else "N/A" for r in rs)
        print(f"{label:<18} | {mean_lat:>8.2f}s | {mean_ade:>8.2f}m | {per}", flush=True)


if __name__ == "__main__":
    main()
