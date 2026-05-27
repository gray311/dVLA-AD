"""Compute Average Displacement Error (ADE) for trajectory predictions across
all SGLang-template runs on the same 10 Waymo samples. Saves a comparison
markdown to results/waymo_10_compare/ade_comparison.md.

ADE = mean L2 distance per waypoint between predicted and GT, then averaged
across samples. We use the first 5 GT future waypoints (the data file's
`future waypoints` field saved as `gt_future_5_waypoints`).
"""
import json
import math
import os
import sys

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, os.path.join(ROOT, "eval"))
from template_v3 import parse_filled

OUT_DIR = os.path.join(ROOT, "results", "waymo_10_compare")


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0:
        return None
    return sum(
        math.hypot(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1])
        for i in range(n)
    ) / n


def fde(pred, gt, k=5):
    """Final Displacement Error — L2 at the k-th waypoint."""
    n = min(len(pred), len(gt), k)
    if n == 0:
        return None
    i = n - 1
    return math.hypot(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1])


def main():
    runs = [
        ("loose (compact format)", "sglang_expl100_loose.json"),
        ("+ complexity (compact)", "sglang_complexity_after_objects.json"),
        ("+ semantic trajectory", "sglang_semantic_traj.json"),
    ]

    lines = []
    lines.append("# Trajectory ADE / FDE: compact vs semantic format")
    lines.append("")
    lines.append("Average Displacement Error (mean L2 over the first 5 waypoints) and "
                 "Final Displacement Error (L2 at the 5th waypoint, t=2.5s) on the 10 "
                 "stratified Waymo samples.")
    lines.append("")
    lines.append("| # | sample | nav | speed (m/s) | " +
                 " | ".join([f"{name} ADE" for name, _ in runs]) +
                 " | " + " | ".join([f"{name} FDE" for name, _ in runs]) + " |")
    sep = "|---|---|---|---:|" + "|".join(["---:"] * (len(runs) * 2)) + "|"
    lines.append(sep)

    loaded = {name: json.load(open(os.path.join(OUT_DIR, path))) for name, path in runs}
    n_samples = len(next(iter(loaded.values())))
    totals_ade = {name: [] for name, _ in runs}
    totals_fde = {name: [] for name, _ in runs}

    for i in range(n_samples):
        first = list(loaded.values())[0][i]
        sid = first["sample_id"]
        row = [str(i + 1), sid[:8], first["nav"], f"{first['speed']:.1f}"]
        for name, _ in runs:
            r = loaded[name][i]
            pred = parse_filled(r["sglang_template"]["output"])
            gt = r["gt_future_5_waypoints"]
            a = ade(pred, gt)
            totals_ade[name].append(a if a is not None else float("nan"))
            row.append(f"{a:.2f}" if a is not None else "—")
        for name, _ in runs:
            r = loaded[name][i]
            pred = parse_filled(r["sglang_template"]["output"])
            gt = r["gt_future_5_waypoints"]
            f = fde(pred, gt)
            totals_fde[name].append(f if f is not None else float("nan"))
            row.append(f"{f:.2f}" if f is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    # mean rows
    def _mean(vs):
        vs = [v for v in vs if not math.isnan(v)]
        return sum(vs) / max(1, len(vs))

    mean_row = ["**mean**", "", "", ""]
    for name, _ in runs:
        mean_row.append(f"**{_mean(totals_ade[name]):.2f}**")
    for name, _ in runs:
        mean_row.append(f"**{_mean(totals_fde[name]):.2f}**")
    lines.append("| " + " | ".join(mean_row) + " |")
    lines.append("")

    base_ade = _mean(totals_ade["loose (compact format)"])
    new_ade = _mean(totals_ade["+ semantic trajectory"])
    base_fde = _mean(totals_fde["loose (compact format)"])
    new_fde = _mean(totals_fde["+ semantic trajectory"])
    lines.append("## Key result")
    lines.append("")
    lines.append(f"- **Mean ADE**: {base_ade:.2f}m (compact) → "
                 f"**{new_ade:.2f}m** (semantic). "
                 f"**Δ = {new_ade - base_ade:+.2f}m ({100*(new_ade-base_ade)/base_ade:+.0f}%)**")
    lines.append(f"- **Mean FDE**: {base_fde:.2f}m → **{new_fde:.2f}m**. "
                 f"**Δ = {new_fde - base_fde:+.2f}m ({100*(new_fde-base_fde)/base_fde:+.0f}%)**")
    lines.append("")
    lines.append("Semantic per-waypoint format with `<t>s: forward=...m, lateral=...m` "
                 "anchors each predicted number to a concrete physical quantity. The "
                 "model now treats trajectory like proper distance-vs-time data instead "
                 "of unlabeled digit sequences.")
    lines.append("")

    out_path = os.path.join(OUT_DIR, "ade_comparison.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
