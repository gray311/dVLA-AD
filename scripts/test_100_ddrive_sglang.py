"""100-sample validation of SGLang + dDrive algorithm.

Runs the new SGLang template-fill (dDrive mdm_sample_deep_scaffold port) on
100 stratified Waymo val samples. Beyond ADE, auto-flags explanation
quality problems:
  - parse_fail:    JSON / trajectory unparseable
  - non_ascii:     Chinese/garbage collapse (>5% non-ASCII chars in explanation)
  - repetition:    mode collapse (any 2-gram repeated >4x)
  - too_short:     explanation < 30 chars
  - bpe_glue:      heuristic for fragment glue (lowercase-run >25 chars no space)
"""
import json, math, os, re, sys, time

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, parse_filled

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
OUT = "/tmp/test100_ddrive_sglang.json"


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0: return None
    return sum(math.hypot(pred[i][0]-gt[i][0], pred[i][1]-gt[i][1]) for i in range(n)) / n


def pick_stratified(data, n):
    buckets = {"L": [], "R": [], "S_stop": [], "S_slow": [], "S_mid": [], "S_fast": []}
    for i, s in enumerate(data):
        nav = s["navigation_command"]
        vx, vy = s["velocity"][-1]
        sp = math.hypot(vx, vy)
        if nav == "GO_LEFT": buckets["L"].append(i)
        elif nav == "GO_RIGHT": buckets["R"].append(i)
        elif sp < 1: buckets["S_stop"].append(i)
        elif sp < 5: buckets["S_slow"].append(i)
        elif sp < 15: buckets["S_mid"].append(i)
        else: buckets["S_fast"].append(i)
    per = max(1, n // len(buckets))
    picks = []
    for k, v in buckets.items():
        step = max(1, len(v) // per)
        picks.extend(v[::step][:per])
    return sorted(set(picks))[:n]


def extract_explanation(text):
    m = re.search(r'"explanation":\s*"([^"]*)"', text)
    return m.group(1) if m else ""


def quality_flags(text):
    """Return list of quality issues found."""
    flags = []
    expl = extract_explanation(text)
    if not expl:
        flags.append("no_expl")
        return flags
    if len(expl) < 30:
        flags.append("too_short")
    # Non-ASCII ratio (Chinese collapse)
    non_ascii = sum(1 for c in expl if ord(c) > 127)
    if non_ascii / max(len(expl), 1) > 0.05:
        flags.append(f"non_ascii({non_ascii})")
    # Repetition: any word repeated >5x consecutively-ish
    words = expl.split()
    if words:
        from collections import Counter
        wc = Counter(words)
        top_word, top_n = wc.most_common(1)[0]
        if top_n > max(6, len(words) * 0.25):
            flags.append(f"repeat('{top_word}'x{top_n})")
    # BPE glue: a run of >25 non-space word chars
    if re.search(r'[A-Za-z]{26,}', expl):
        flags.append("bpe_glue")
    return flags


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    data = json.load(open(DATA))
    indices = pick_stratified(data, n)
    print(f"Picked {len(indices)} stratified samples", flush=True)

    from eval.loaders import fast_dvlm_sglang_v3 as loader
    print("Loading SGLang engine (dDrive algorithm)...", flush=True)
    bundle = loader.load(algorithm="mdm", engine_block_size=160)

    print("Warmup...", flush=True)
    s0 = data[indices[0]]
    _, _ = loader.generate(
        bundle, [_fix(s0['image'][1])], build_prompt_v3(s0),
        temperature=0.0, block_size=160, section_align=True,
        nav_command=s0['navigation_command'],
    )

    results = []
    t_start = time.time()
    for k, idx in enumerate(indices):
        s = data[idx]
        try:
            text, latency = loader.generate(
                bundle, [_fix(s['image'][1])], build_prompt_v3(s),
                temperature=0.0, block_size=160, section_align=True,
                nav_command=s['navigation_command'],
            )
            pred = parse_filled(text)
            gt = s["future waypoints"][:5]
            a = ade(pred, gt) if pred else None
            flags = quality_flags(text)
            vx, vy = s["velocity"][-1]
            results.append({
                "idx": idx, "nav": s["navigation_command"],
                "speed": math.hypot(vx, vy),
                "ade": a, "latency": latency,
                "parse_ok": pred is not None and len(pred) >= 3,
                "flags": flags,
                "explanation": extract_explanation(text),
                "text": text,
            })
            if (k + 1) % 10 == 0:
                done = [r for r in results if r.get("ade") is not None]
                ade_so_far = sum(r["ade"] for r in done) / max(len(done), 1)
                nflag = sum(1 for r in results if r["flags"])
                print(f"  [{k+1}/{len(indices)}] ADE_mean={ade_so_far:.2f}m  "
                      f"flagged={nflag}  lat~{latency:.2f}s", flush=True)
        except Exception as e:
            import traceback
            results.append({"idx": idx, "error": str(e)})
            print(f"  [{k+1}] idx={idx} ERROR: {e}", flush=True)

    loader.shutdown(bundle)
    elapsed = time.time() - t_start

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    # Report
    valid = [r for r in results if r.get("ade") is not None]
    errored = [r for r in results if r.get("error")]
    parse_fail = [r for r in results if not r.get("error") and not r.get("parse_ok")]
    flagged = [r for r in results if r.get("flags")]

    print("\n" + "=" * 60)
    print(f"100-CASE VALIDATION — SGLang + dDrive algorithm")
    print("=" * 60)
    print(f"  total          : {len(results)}")
    print(f"  errored        : {len(errored)}")
    print(f"  parse_fail     : {len(parse_fail)}")
    print(f"  quality-flagged: {len(flagged)}")
    print(f"  elapsed        : {elapsed:.0f}s ({elapsed/max(len(results),1):.2f}s/sample)")
    if valid:
        ades = sorted(r["ade"] for r in valid)
        nv = len(ades)
        print(f"  ADE mean       : {sum(ades)/nv:.2f}m")
        print(f"  ADE median     : {ades[nv//2]:.2f}m")
        print(f"  ADE p90        : {ades[min(nv-1, int(0.9*nv))]:.2f}m")
    lats = [r["latency"] for r in valid]
    if lats:
        print(f"  latency mean   : {sum(lats)/len(lats):.2f}s")

    # Flag breakdown
    if flagged:
        print("\n  --- quality flags breakdown ---")
        from collections import Counter
        flag_counter = Counter()
        for r in flagged:
            for fl in r["flags"]:
                flag_counter[fl.split("(")[0]] += 1
        for fl, cnt in flag_counter.most_common():
            print(f"    {fl}: {cnt}")
        print("\n  --- sample flagged explanations (first 5) ---")
        for r in flagged[:5]:
            print(f"    idx={r['idx']} flags={r['flags']}")
            print(f"      {r['explanation'][:150]}")
    print("=" * 60)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
