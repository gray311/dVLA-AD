"""4-way comparison: V3 vs Fast-dVLM transformers (template-fill) vs
Fast-dVLM SGLang (free-form gen) vs dVLM-AD.
"""
import json
import re

ROOT = "/weka/home/ext-yingzima/dVLA-AD"


def _field(txt, key):
    m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', txt)
    return m.group(1) if m else ""


def main():
    raw = json.load(open(f"{ROOT}/results/waymo_5_compare/raw.json"))
    fast_tf = json.load(open(f"{ROOT}/results/waymo_5_compare/fast_dvlm_raw.json"))
    fast_sg_spec = json.load(open(f"{ROOT}/results/waymo_5_compare/fast_dvlm_sglang_spec.json"))
    fast_sg_mdm = json.load(open(f"{ROOT}/results/waymo_5_compare/fast_dvlm_sglang_mdm.json"))

    by_id_tf = {r["sample_id"]: r for r in fast_tf}
    by_id_sg_spec = {r["sample_id"]: r for r in fast_sg_spec}
    by_id_sg_mdm = {r["sample_id"]: r for r in fast_sg_mdm}

    out = []
    out.append("# Waymo 5-Sample Comparison: 4 paths")
    out.append("")
    out.append("All paths use the SAME 5 Waymo samples + V3 prompt + ego state.")
    out.append("")
    out.append("- **V3** — DiffusionVL-3B + V3 template-fill (zero-shot).")
    out.append("- **Fast-dVLM transformers** — Fast_dVLM_3B + V3 template-fill via custom "
               "block-causal diffusion (our tuned 3-tier adaptive steps).")
    out.append("- **Fast-dVLM SGLang (spec)** — Fast_dVLM_3B via NVlabs vendored SGLang fork "
               "with `SpeculativeBlock` decoding (self-speculative). Free-form gen (no template fill).")
    out.append("- **Fast-dVLM SGLang (mdm)** — same engine with `HierarchyBlock` (MDM block-diffusion).")
    out.append("- **dVLM-AD** — LLaDA-V-8B finetuned on Waymo CoT + data file template.")
    out.append("")

    for i, r in enumerate(raw):
        sid = r["sample_id"]
        tf = by_id_tf.get(sid, {}).get("fast_dvlm", {})
        sg_s = by_id_sg_spec.get(sid, {}).get("fast_dvlm_sglang", {})
        sg_m = by_id_sg_mdm.get(sid, {}).get("fast_dvlm_sglang", {})

        out.append(f"---")
        out.append(f"")
        out.append(f"## {i+1}. {sid[:24]} — nav=`{r['nav']}`, speed={r['speed']:.1f} m/s")
        out.append(f"")

        v3_out = r["v3"]["output"]
        ad_out = r["dvlm_ad"]["output"]
        tf_out = tf.get("output", "")
        sg_s_out = sg_s.get("output", "")
        sg_m_out = sg_m.get("output", "")

        out.append("### Latency")
        out.append("")
        out.append("|  | latency |")
        out.append("|---|---:|")
        out.append(f"| V3 | {r['v3']['latency_s']:.2f}s |")
        out.append(f"| Fast-dVLM tf | {tf.get('latency_s', -1):.2f}s |")
        out.append(f"| **Fast-dVLM SGLang spec** | **{sg_s.get('latency_s', -1):.2f}s** |")
        out.append(f"| Fast-dVLM SGLang mdm | {sg_m.get('latency_s', -1):.2f}s |")
        out.append(f"| dVLM-AD | {r['dvlm_ad']['latency_s']:.2f}s |")
        out.append("")

        out.append("### Behavior")
        out.append("")
        out.append("|  | longitudinal | lateral |")
        out.append("|---|---|---|")
        out.append(f"| V3 | `{_field(v3_out, 'longitudinal')}` | `{_field(v3_out, 'lateral')}` |")
        out.append(f"| Fast-dVLM tf | `{_field(tf_out, 'longitudinal')}` | `{_field(tf_out, 'lateral')}` |")
        out.append(f"| Fast-dVLM SGLang spec | `{_field(sg_s_out, 'longitudinal')}` | `{_field(sg_s_out, 'lateral')}` |")
        out.append(f"| Fast-dVLM SGLang mdm | `{_field(sg_m_out, 'longitudinal')}` | `{_field(sg_m_out, 'lateral')}` |")
        out.append(f"| dVLM-AD | `{_field(ad_out, 'longitudinal')}` | `{_field(ad_out, 'lateral')}` |")
        out.append("")

        out.append("### Fast-dVLM SGLang (spec) output")
        out.append("```json")
        out.append(sg_s_out[:1800])
        out.append("```")
        out.append("")

        v3_expl = _field(v3_out, "explanation")
        tf_expl = _field(tf_out, "explanation")
        sg_s_expl = _field(sg_s_out, "explanation")
        out.append("### Explanation comparison")
        out.append("")
        out.append(f"**V3** ({len(v3_expl)} chars):")
        out.append("> " + (v3_expl[:400] or "(empty)"))
        out.append("")
        out.append(f"**Fast-dVLM tf** ({len(tf_expl)} chars):")
        out.append("> " + (tf_expl[:400] or "(empty)"))
        out.append("")
        out.append(f"**Fast-dVLM SGLang spec** ({len(sg_s_expl)} chars):")
        out.append("> " + (sg_s_expl[:400] or "(empty)"))
        out.append("")

    # Summary
    out.append("---")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| # | sample | nav | speed | V3 s | tf s | **SGLang spec s** | SGLang mdm s | AD s |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    avg_v3 = avg_tf = avg_sgs = avg_sgm = avg_ad = 0.0
    for i, r in enumerate(raw):
        sid = r["sample_id"]
        tf = by_id_tf.get(sid, {}).get("fast_dvlm", {})
        sg_s = by_id_sg_spec.get(sid, {}).get("fast_dvlm_sglang", {})
        sg_m = by_id_sg_mdm.get(sid, {}).get("fast_dvlm_sglang", {})
        avg_v3 += r["v3"]["latency_s"]; avg_tf += tf.get("latency_s", 0)
        avg_sgs += sg_s.get("latency_s", 0); avg_sgm += sg_m.get("latency_s", 0)
        avg_ad += r["dvlm_ad"]["latency_s"]
        out.append(
            f"| {i+1} | {sid[:8]} | {r['nav']} | {r['speed']:.1f} "
            f"| {r['v3']['latency_s']:.2f} | {tf.get('latency_s', -1):.2f} "
            f"| **{sg_s.get('latency_s', -1):.2f}** | {sg_m.get('latency_s', -1):.2f} "
            f"| {r['dvlm_ad']['latency_s']:.2f} |"
        )
    n = len(raw)
    out.append(f"| **avg** | | | | {avg_v3/n:.2f} | {avg_tf/n:.2f} | **{avg_sgs/n:.2f}** | {avg_sgm/n:.2f} | {avg_ad/n:.2f} |")
    out.append("")

    out_path = f"{ROOT}/results/waymo_5_compare/comparison_4way.md"
    with open(out_path, "w") as f:
        f.write("\n".join(out))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
