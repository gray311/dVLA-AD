"""Compare explanation quality + diversity + accuracy across the 10-sample
SGLang vs dVLM-AD run. Focuses on:

  Diversity:
    - per-explanation unique-token ratio (vocab / total)
    - repeated bigram / trigram fraction
    - cross-sample distinct-word set (do explanations look templated?)

  Accuracy / grounding:
    - mentions the actual speed value
    - mentions the nav direction (left / right / straight / lane)
    - mentions the longitudinal/lateral verb the model itself emitted
    - mentions critical-objects values (red/black/white/etc. + cone/car/etc.)
    - sentence count + average sentence length

Output: results/waymo_10_compare/explanation_compare.md
"""
import json
import os
import re
from collections import Counter

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
OUT_DIR = os.path.join(ROOT, "results", "waymo_10_compare")


def _clean(t):
    return re.sub(r"<\|mdm_(start|end)\|>", "", t or "").strip()


def _field(t, key):
    m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', t)
    return m.group(1).strip() if m else ""


def _crit(t):
    blob = re.search(r'"critical_objects"\s*:\s*\{(.*?)\}', t, re.DOTALL)
    if not blob:
        return {}
    out = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', blob.group(1)):
        out[m.group(1)] = m.group(2).strip()
    return out


def _tokens(text):
    return re.findall(r"\w+", text.lower())


def _bigrams(tokens):
    return list(zip(tokens, tokens[1:]))


def _trigrams(tokens):
    return list(zip(tokens, tokens[1:], tokens[2:]))


def diversity_metrics(text):
    toks = _tokens(text)
    if not toks:
        return {"n_tokens": 0, "unique_ratio": 0, "rep_bigram_frac": 0,
                "rep_trigram_frac": 0, "n_sentences": 0}
    unique = len(set(toks)) / max(1, len(toks))
    bg = _bigrams(toks)
    bg_counts = Counter(bg)
    rep_bg = sum(c for c in bg_counts.values() if c > 1) / max(1, len(bg))
    tg = _trigrams(toks)
    tg_counts = Counter(tg)
    rep_tg = sum(c for c in tg_counts.values() if c > 1) / max(1, len(tg))
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "n_tokens": len(toks),
        "unique_ratio": unique,
        "rep_bigram_frac": rep_bg,
        "rep_trigram_frac": rep_tg,
        "n_sentences": len(sentences),
    }


def grounding_signals(text, sample, model_output):
    """Check what concrete scene/state info the explanation references."""
    signals = {}
    lo = text.lower()
    # Speed mention (e.g., "1.2 m/s" or just "1.2" in context)
    speed = sample["speed"]
    speed_str = f"{speed:.1f}"
    signals["mentions_speed"] = (speed_str in text) or (f"{int(round(speed))}" in lo and "m/s" in lo)
    # Nav direction
    nav = sample["nav"]
    if nav == "GO_LEFT":
        signals["mentions_nav"] = "left" in lo
    elif nav == "GO_RIGHT":
        signals["mentions_nav"] = "right" in lo
    else:
        signals["mentions_nav"] = ("straight" in lo) or ("lane" in lo and "keep" in lo)
    # References behavior the model emitted
    long_v = _field(model_output, "longitudinal").lower()
    lat_v = _field(model_output, "lateral").lower()
    signals["mentions_own_long"] = bool(long_v) and (long_v in lo)
    signals["mentions_own_lat"] = bool(lat_v) and (lat_v in lo)
    # References critical objects content (non-"none" values)
    co = _crit(model_output)
    interesting = [v for v in co.values()
                   if v and v.lower() not in ("none", "none found", "none seen",
                                              "none visible", "none detected",
                                              "none here", "no", " found here",
                                              "no found")]
    matched_co = sum(1 for v in interesting if any(w in lo for w in v.lower().split() if len(w) > 2))
    signals["co_mentions"] = matched_co
    signals["co_total"] = len(interesting)
    # Specific object/color words
    keywords = ["car", "truck", "bus", "cyclist", "pedestrian", "cone", "light",
                "sign", "construction", "highway", "lane", "intersection",
                "weather", "sunny", "overcast", "smoke", "fire", "hazard"]
    signals["object_keywords"] = sum(1 for k in keywords if k in lo)
    return signals


def main():
    import sys
    sg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, "sglang.json")
    out_path_name = sys.argv[2] if len(sys.argv) > 2 else "explanation_compare.md"
    sg = json.load(open(sg_path))
    ad = json.load(open(os.path.join(OUT_DIR, "dvlm_ad.json")))
    ad_by = {r["sample_id"]: r for r in ad}

    lines = []
    lines.append("# Explanation Quality + Diversity: SGLang (zero-shot) vs dVLM-AD (finetuned)")
    lines.append("")
    lines.append("Comparing the `explanation` field of each model's response on the same "
                 "10 Waymo samples. Both fields are ~400-550 chars (roughly 100-130 tokens).")
    lines.append("")

    # ============ Per-sample side-by-side ============
    lines.append("## Per-sample explanations side-by-side")
    lines.append("")
    sg_metrics = []
    ad_metrics = []
    for i, r in enumerate(sg):
        sid = r["sample_id"]
        sg_out = r["sglang_template"]["output"]
        ad_out = _clean(ad_by[sid]["dvlm_ad"]["output"])
        e_sg = _field(sg_out, "explanation")
        e_ad = _field(ad_out, "explanation")
        m_sg = diversity_metrics(e_sg)
        m_ad = diversity_metrics(e_ad)
        g_sg = grounding_signals(e_sg, r, sg_out)
        g_ad = grounding_signals(e_ad, r, ad_out)
        sg_metrics.append((m_sg, g_sg, e_sg))
        ad_metrics.append((m_ad, g_ad, e_ad))

        lines.append(f"### {i+1}. `{sid[:16]}` — nav=`{r['nav']}`, speed={r['speed']:.1f} m/s")
        lines.append("")
        lines.append(f"**SGLang Fast-dVLM** ({m_sg['n_tokens']} tok, {m_sg['n_sentences']} sent, "
                     f"unique={m_sg['unique_ratio']:.2f}, rep-bg={m_sg['rep_bigram_frac']:.2f}, "
                     f"speed-ref={g_sg['mentions_speed']}, nav-ref={g_sg['mentions_nav']}, "
                     f"co-ref={g_sg['co_mentions']}/{g_sg['co_total']}, "
                     f"keywords={g_sg['object_keywords']}):")
        lines.append(f"> {e_sg or '(empty)'}")
        lines.append("")
        lines.append(f"**dVLM-AD** ({m_ad['n_tokens']} tok, {m_ad['n_sentences']} sent, "
                     f"unique={m_ad['unique_ratio']:.2f}, rep-bg={m_ad['rep_bigram_frac']:.2f}, "
                     f"speed-ref={g_ad['mentions_speed']}, nav-ref={g_ad['mentions_nav']}, "
                     f"co-ref={g_ad['co_mentions']}/{g_ad['co_total']}, "
                     f"keywords={g_ad['object_keywords']}):")
        lines.append(f"> {e_ad or '(empty)'}")
        lines.append("")

    # ============ Aggregate metrics ============
    def _avg(rows, key):
        vals = [m[key] for m, *_ in rows]
        return sum(vals) / len(vals)

    def _sum(rows, key):
        return sum(g[key] for _, g, _ in rows)

    lines.append("---")
    lines.append("")
    lines.append("## Aggregate metrics (avg over 10 samples)")
    lines.append("")
    lines.append("| metric | SGLang | dVLM-AD |")
    lines.append("|---|---:|---:|")
    lines.append(f"| avg tokens | {_avg(sg_metrics, 'n_tokens'):.1f} | {_avg(ad_metrics, 'n_tokens'):.1f} |")
    lines.append(f"| avg sentences | {_avg(sg_metrics, 'n_sentences'):.1f} | {_avg(ad_metrics, 'n_sentences'):.1f} |")
    lines.append(f"| **unique-token ratio** (higher = more diverse) | **{_avg(sg_metrics, 'unique_ratio'):.2f}** | **{_avg(ad_metrics, 'unique_ratio'):.2f}** |")
    lines.append(f"| **repeated bigram fraction** (lower = better) | **{_avg(sg_metrics, 'rep_bigram_frac'):.2f}** | **{_avg(ad_metrics, 'rep_bigram_frac'):.2f}** |")
    lines.append(f"| repeated trigram fraction | {_avg(sg_metrics, 'rep_trigram_frac'):.2f} | {_avg(ad_metrics, 'rep_trigram_frac'):.2f} |")
    lines.append(f"| **mentions speed value** | {_sum(sg_metrics, 'mentions_speed')}/10 | {_sum(ad_metrics, 'mentions_speed')}/10 |")
    lines.append(f"| **mentions nav direction** | {_sum(sg_metrics, 'mentions_nav')}/10 | {_sum(ad_metrics, 'mentions_nav')}/10 |")
    lines.append(f"| references own longitudinal verb | {_sum(sg_metrics, 'mentions_own_long')}/10 | {_sum(ad_metrics, 'mentions_own_long')}/10 |")
    lines.append(f"| references own lateral verb | {_sum(sg_metrics, 'mentions_own_lat')}/10 | {_sum(ad_metrics, 'mentions_own_lat')}/10 |")
    lines.append(f"| object-keyword count (sum across 10) | {_sum(sg_metrics, 'object_keywords')} | {_sum(ad_metrics, 'object_keywords')} |")
    lines.append("")

    # ============ Cross-sample distinct phrasing ============
    # How often do the SAME bigrams appear across all 10 samples?
    def _all_bigrams(rows):
        all_bg = []
        for _, _, txt in rows:
            all_bg += _bigrams(_tokens(txt))
        return all_bg

    sg_bg = _all_bigrams(sg_metrics)
    ad_bg = _all_bigrams(ad_metrics)
    sg_top = Counter(sg_bg).most_common(15)
    ad_top = Counter(ad_bg).most_common(15)

    lines.append("## Cross-sample top bigrams (templated phrasing detection)")
    lines.append("")
    lines.append("If the same bigrams repeatedly dominate across all 10 samples, the "
                 "explanations are templated — model is just shifting around boilerplate.")
    lines.append("")
    lines.append("**SGLang top 15 bigrams**:")
    lines.append("")
    for b, c in sg_top:
        lines.append(f"  - `{b[0]} {b[1]}` × {c}")
    lines.append("")
    lines.append("**dVLM-AD top 15 bigrams**:")
    lines.append("")
    for b, c in ad_top:
        lines.append(f"  - `{b[0]} {b[1]}` × {c}")
    lines.append("")

    # Unique bigrams across 10 explanations (= vocabulary breadth)
    sg_unique = len(set(sg_bg))
    ad_unique = len(set(ad_bg))
    lines.append(f"**Distinct bigrams across all 10 explanations**:")
    lines.append(f"  - SGLang: {sg_unique}")
    lines.append(f"  - dVLM-AD: {ad_unique}")
    lines.append("")

    out_path = os.path.join(OUT_DIR, out_path_name)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
