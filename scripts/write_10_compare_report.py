"""Merge SGLang Fast-dVLM template-fill + dVLM-AD results into a single
side-by-side report. Run after both passes of run_10_waymo_compare.py.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
OUT_DIR = os.path.join(ROOT, "results", "waymo_10_compare")

LONG_VALID = {("speed", "up"), ("slow", "down"), ("keep", "speed"), ("stop", "now")}
LAT_VALID = {("keep", "lane"), ("turn", "left"), ("turn", "right"),
              ("change", "left"), ("change", "right")}
LAT_EXPECTED = {
    "GO_LEFT": ("turn", "left"),
    "GO_RIGHT": ("turn", "right"),
    "GO_STRAIGHT": ("keep", "lane"),
}
# dVLM-AD was finetuned with a different vocab ("left turn", "lane follow",
# "go straight" etc). Accept ANY of these legal variants per nav.
LAT_ACCEPTABLE = {
    "GO_LEFT": [("turn", "left"), ("left", "turn")],
    "GO_RIGHT": [("turn", "right"), ("right", "turn")],
    "GO_STRAIGHT": [("keep", "lane"), ("lane", "follow"), ("go", "straight")],
}
# Similarly: longitudinal vocab differs ("come to stop" vs "stop now").
LONG_ACCEPTABLE_EXTRA = {("come", "to", "stop"), ("speed", "down")}


def _clean(txt):
    """Strip dVLM-AD's <|mdm_start|> / <|mdm_end|> markers."""
    if not txt:
        return ""
    return re.sub(r"<\|mdm_(start|end)\|>", "", txt).strip()


def _field(txt, key):
    m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', txt)
    return m.group(1).strip() if m else ""


def _critical(txt):
    blob = re.search(r'"critical_objects"\s*:\s*\{(.*?)\}', txt, re.DOTALL)
    if not blob:
        return {}
    out = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', blob.group(1)):
        out[m.group(1)] = m.group(2).strip()
    return out


def _verb_tuple(s):
    return tuple(s.lower().split()) if s else ()


def _is_correct_lat(nav, lat_tup):
    return lat_tup in LAT_ACCEPTABLE.get(nav, [])


def _is_valid_long(long_tup):
    return long_tup in LONG_VALID or long_tup in LONG_ACCEPTABLE_EXTRA


def main():
    sg = json.load(open(os.path.join(OUT_DIR, "sglang.json")))
    ad = json.load(open(os.path.join(OUT_DIR, "dvlm_ad.json")))
    ad_by_id = {r["sample_id"]: r for r in ad}

    lines = []
    lines.append("# 10-Sample Waymo: SGLang Fast-dVLM (template-fill) vs dVLM-AD (finetuned)")
    lines.append("")
    lines.append("- **SGLang Fast-dVLM**: NVlabs Fast_dVLM_3B via modified SGLang fork "
                 "(template-fill mdm; vocab gates + JSON blacklist + rep penalty + "
                 "nav injection). Zero-shot — no driving fine-tune.")
    lines.append("- **dVLM-AD**: LLaDA-V-8B **finetuned on Waymo CoT** via data-file "
                 "template (data file `conversations[0]` as prompt, `conversations[1]` "
                 "with `<|mdm_mask|>` as response template). 64-step diffusion.")
    lines.append("")

    # Summary table first
    lines.append("## Summary")
    lines.append("")
    lines.append("| # | sample | nav | speed | SGLang lat | AD lat | SGLang long/lat | AD long/lat | SGLang lat-ok | AD lat-ok |")
    lines.append("|---|---|---|---:|---:|---:|---|---|:---:|:---:|")

    sg_lat_correct = 0
    ad_lat_correct = 0
    sg_long_valid = 0
    ad_long_valid = 0
    sg_lat_sum = 0
    ad_lat_sum = 0
    n = 0
    for i, r in enumerate(sg):
        sid = r["sample_id"]
        adr = ad_by_id.get(sid, {})
        sg_out = r["sglang_template"]["output"]
        ad_out = _clean(adr.get("dvlm_ad", {}).get("output", ""))
        sg_long_t = _verb_tuple(_field(sg_out, "longitudinal"))
        sg_lat_t = _verb_tuple(_field(sg_out, "lateral"))
        ad_long_t = _verb_tuple(_field(ad_out, "longitudinal"))
        ad_lat_t = _verb_tuple(_field(ad_out, "lateral"))
        sg_lat_ok = _is_correct_lat(r["nav"], sg_lat_t)
        ad_lat_ok = _is_correct_lat(r["nav"], ad_lat_t)
        if sg_lat_ok: sg_lat_correct += 1
        if ad_lat_ok: ad_lat_correct += 1
        if _is_valid_long(sg_long_t): sg_long_valid += 1
        if _is_valid_long(ad_long_t): ad_long_valid += 1
        sg_lat_sum += r["sglang_template"]["latency_s"]
        ad_lat_sum += adr.get("dvlm_ad", {}).get("latency_s", 0)
        n += 1
        lines.append(
            f"| {i+1} | {sid[:8]} | {r['nav']} | {r['speed']:.1f}"
            f" | {r['sglang_template']['latency_s']:.2f}s"
            f" | {adr.get('dvlm_ad', {}).get('latency_s', -1):.1f}s"
            f" | `{' '.join(sg_long_t) or '-'}` / `{' '.join(sg_lat_t) or '-'}`"
            f" | `{' '.join(ad_long_t) or '-'}` / `{' '.join(ad_lat_t) or '-'}`"
            f" | {'✓' if sg_lat_ok else '✗'} | {'✓' if ad_lat_ok else '✗'} |"
        )
    lines.append(f"| **avg** | | | | **{sg_lat_sum/n:.2f}s** | **{ad_lat_sum/n:.1f}s**"
                 f" | long {sg_long_valid}/{n} valid | long {ad_long_valid}/{n} valid"
                 f" | **{sg_lat_correct}/{n}** | **{ad_lat_correct}/{n}** |")
    lines.append("")
    lines.append(f"**Speed-up**: SGLang is **{(ad_lat_sum/n) / (sg_lat_sum/n):.1f}x** faster than dVLM-AD.")
    lines.append("")
    lines.append(f"**Behavior accuracy** (lateral matches nav):")
    lines.append(f"  - SGLang Fast-dVLM (zero-shot): {sg_lat_correct}/{n}")
    lines.append(f"  - dVLM-AD (finetuned): {ad_lat_correct}/{n}")
    lines.append("")

    # Per-sample full content
    lines.append("---")
    lines.append("")
    lines.append("## Per-sample details")
    lines.append("")
    for i, r in enumerate(sg):
        sid = r["sample_id"]
        adr = ad_by_id.get(sid, {})
        sg_out = r["sglang_template"]["output"]
        ad_out = _clean(adr.get("dvlm_ad", {}).get("output", ""))

        lines.append(f"### {i+1}. `{sid[:16]}` — nav=`{r['nav']}`, speed={r['speed']:.1f} m/s")
        lines.append("")
        lines.append(f"Image: `{r['image']}`")
        lines.append("")

        lines.append("**Behavior**:")
        lines.append("")
        lines.append("|  | longitudinal | lateral |")
        lines.append("|---|---|---|")
        lines.append(f"| SGLang | `{_field(sg_out, 'longitudinal')}` | `{_field(sg_out, 'lateral')}` |")
        lines.append(f"| dVLM-AD | `{_field(ad_out, 'longitudinal')}` | `{_field(ad_out, 'lateral')}` |")
        lines.append("")

        sg_co = _critical(sg_out)
        ad_co = _critical(ad_out)
        if sg_co or ad_co:
            keys = sorted(set(list(sg_co.keys()) + list(ad_co.keys())))
            lines.append("**Critical objects**:")
            lines.append("")
            lines.append("| category | SGLang | dVLM-AD |")
            lines.append("|---|---|---|")
            for k in keys:
                lines.append(f"| `{k}` | `{sg_co.get(k, '-')}` | `{ad_co.get(k, '-')}` |")
            lines.append("")

        sg_e = _field(sg_out, "explanation")
        ad_e = _field(ad_out, "explanation")
        lines.append("**Explanation**:")
        lines.append("")
        lines.append(f"_SGLang_ ({len(sg_e)} chars):")
        lines.append(f"> {sg_e[:500] or '(empty)'}")
        lines.append("")
        lines.append(f"_dVLM-AD_ ({len(ad_e)} chars):")
        lines.append(f"> {ad_e[:500] or '(empty)'}")
        lines.append("")

        sg_t = _field(sg_out, "trajectory")
        ad_t = _field(ad_out, "trajectory")
        lines.append("**Trajectory**:")
        lines.append("")
        lines.append(f"_SGLang_ ({r['sglang_template']['latency_s']:.2f}s):")
        lines.append(f"```\n{sg_t[:200]}\n```")
        lines.append(f"_dVLM-AD_ ({adr.get('dvlm_ad', {}).get('latency_s', -1):.1f}s):")
        lines.append(f"```\n{ad_t[:200]}\n```")
        lines.append(f"_GT first 5 waypoints_: `{r['gt_future_5_waypoints']}`")
        lines.append("")

    out_path = os.path.join(OUT_DIR, "comparison.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
