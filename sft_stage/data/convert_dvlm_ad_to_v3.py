#!/usr/bin/env python3
"""Convert dvlm-ad Waymo CoT data -> V3 schema (for Fast-dVLM SFT).

Source: dvlm-ad_waymo_training_cot.json (yes/no critical_objects, dvlm-ad
behavior verbs, [[+XX.XX,-XX.XX]] 5-wp@1s trajectory).

Target: V3 schema response (eval/template_v3.py), one JSON object per sample:
    {"critical_objects": {12 cats: <noun phrase|none>},
     "complexity": "<simple|complex>",
     "explanation": "<free text>",
     "future_meta_behavior": {"longitudinal": "<2-word verb>",
                               "lateral": "<2-word verb>"},
     "trajectory": "0.5s: forward=+XX.Xm, lateral=+YY.Ym\\n..."}   # 10 wp @ 0.5s

The human turn is rebuilt as a V3 task prompt that reuses the source's
"Input:" section (nav command + ego history) verbatim.

Usage:
    # write 10 examples to stdout for review (no output file)
    python convert_dvlm_ad_to_v3.py --n 10 --preview

    # convert the full set
    python convert_dvlm_ad_to_v3.py \
        --src /weka/.../dvlm-ad_waymo_training_cot.json \
        --out /weka/.../dvlm-ad_waymo_training_v3.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# V3 schema constants — mirror of eval/template_v3.py (inlined to avoid a
# fragile cross-module import; keep these in sync if template_v3.py changes).
CRITICAL_CATEGORIES = [
    "nearby_vehicle", "pedestrian", "cyclist", "construction",
    "traffic_element", "weather_condition", "road_hazard",
    "emergency_vehicle", "animal", "special_vehicle",
    "conflicting_vehicle", "door_opening_vehicle",
]
LONG_VERBS = ["speed up", "slow down", "keep speed", "stop now"]
LAT_VERBS = ["keep lane", "turn left", "turn right", "change left", "change right"]
COMPLEXITY_LABELS = ["simple", "complex"]
N_WAYPOINTS = 10           # trajectory waypoints (0.5 s spacing -> 5 s horizon)
TRAJECTORY_DT = 0.5

DEFAULT_SRC = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_training_cot.json"
DEFAULT_OUT = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_training_v3.json"

_MDM_START = "<|mdm_start|>"
_MDM_END = "<|mdm_end|>"


# ─── field mappings ──────────────────────────────────────────────────────────

# meta_speed -> V3 longitudinal verb (LONG_VERBS = speed up/slow down/keep speed/stop now)
SPEED_TO_LONG = {
    "keep_speed": "keep speed",
    "accelerate": "speed up",
    "accel": "speed up",
    "decelerate": "slow down",
    "decel": "slow down",
    "stop": "stop now",
    "stop_now": "stop now",
    "other": "keep speed",
}

# meta_direction -> V3 lateral verb (LAT_VERBS = keep lane/turn left/turn right/change left/change right)
DIR_TO_LAT = {
    "keep_lane": "keep lane",
    "left_turn": "turn left",
    "left_turn_in_lane": "turn left",
    "right_turn": "turn right",
    "right_turn_in_lane": "turn right",
    "left_lane_change": "change left",
    "right_lane_change": "change right",
    "other": "keep lane",
}

# critical_objects "yes" -> a category-appropriate noun phrase (V3 is open-vocab).
# "no" maps to "none".
CRITICAL_YES_PHRASE = {
    "nearby_vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
    "construction": "construction",
    "traffic_element": "traffic light",
    "weather_condition": "adverse weather",
    "road_hazard": "road hazard",
    "emergency_vehicle": "emergency vehicle",
    "animal": "animal",
    "special_vehicle": "special vehicle",
    "conflicting_vehicle": "conflicting vehicle",
    "door_opening_vehicle": "opening door",
}


def _strip_mdm(s: str) -> str:
    return (s or "").replace(_MDM_START, "").replace(_MDM_END, "").strip()


def map_longitudinal(meta_speed: str) -> str:
    v = SPEED_TO_LONG.get((meta_speed or "").strip().lower(), "keep speed")
    assert v in LONG_VERBS, f"long verb {v!r} not in {LONG_VERBS}"
    return v


def map_lateral(meta_direction: str) -> str:
    v = DIR_TO_LAT.get((meta_direction or "").strip().lower(), "keep lane")
    assert v in LAT_VERBS, f"lat verb {v!r} not in {LAT_VERBS}"
    return v


def map_critical_objects(src_co: dict) -> dict:
    """yes/no -> {noun phrase | 'none'} for all 12 V3 categories (fixed order)."""
    out = {}
    for cat in CRITICAL_CATEGORIES:
        raw = str(src_co.get(cat, "no")).strip().lower()
        out[cat] = CRITICAL_YES_PHRASE.get(cat, "object") if raw == "yes" else "none"
    return out


def derive_complexity(sample: dict, src_co: dict) -> str:
    """Heuristic simple/complex from metadata (no human labels).

    complex when the scene needs more planning attention:
      - >= 2 critical objects present, OR
      - a turn / lane-change maneuver (meta_direction != keep_lane), OR
      - non-normal maneuver (creep / wait / reverse), OR
      - overtake flagged, OR
      - large heading change / lane shift in meta_diag, OR
      - coming to a stop.
    """
    n_yes = sum(1 for cat in CRITICAL_CATEGORIES
                if str(src_co.get(cat, "no")).strip().lower() == "yes")
    md = (sample.get("meta_direction") or "").strip().lower()
    mm = (sample.get("meta_maneuver") or "").strip().lower()
    ms = (sample.get("meta_speed") or "").strip().lower()
    diag = sample.get("meta_diag", {}) or {}
    dir_d = diag.get("dir", {}) or {}
    overtake = bool((diag.get("overtake", {}) or {}).get("overtake", False))
    net_heading = abs(float(dir_d.get("net_heading_deg", 0.0) or 0.0))
    lane_shift = abs(float(dir_d.get("lane_shift_abs", 0.0) or 0.0))
    mode = (dir_d.get("mode") or "none").strip().lower()

    complex_signals = [
        n_yes >= 2,
        md not in ("keep_lane", "", "other"),
        mm not in ("normal", "stationary", ""),
        overtake,
        mode not in ("none", ""),
        net_heading > 15.0,
        lane_shift > 2.0,
        ms in ("stop", "stop_now"),
    ]
    return "complex" if any(complex_signals) else "simple"


def _fmt_coord(v: float) -> str:
    """Format one coordinate as <sign><2-digit int>.<1-digit frac>, clamped ±99.9."""
    v = max(-99.9, min(99.9, round(float(v), 1)))
    sign = "+" if v >= 0 else "-"
    av = abs(v)
    whole = int(av)
    frac = int(round((av - whole) * 10))
    if frac == 10:
        whole += 1
        frac = 0
    return f"{sign}{whole:02d}.{frac}"


def map_trajectory(future_waypoints, overflow_counter=None):
    """Subsample the source future waypoints (20 @ 0.25 s, index i -> (i+1)*0.25 s)
    to 10 @ 0.5 s (indices 1,3,..,19) and render the V3 semantic lines.

    Returns (traj_string, n_overflow) where n_overflow counts coords clamped to ±99.9.
    """
    # 0.5 s, 1.0 s, ..., 5.0 s == source indices 1,3,5,...,19
    want_idx = [2 * (w + 1) - 1 for w in range(N_WAYPOINTS)]  # 1,3,...,19
    lines = []
    n_overflow = 0
    for w, idx in enumerate(want_idx):
        t = (w + 1) * TRAJECTORY_DT
        if idx < len(future_waypoints):
            fwd, lat = future_waypoints[idx][0], future_waypoints[idx][1]
        else:
            fwd, lat = (future_waypoints[-1] if future_waypoints else (0.0, 0.0))
        if abs(fwd) > 99.9 or abs(lat) > 99.9:
            n_overflow += 1
        lines.append(f"{t:.1f}s: forward={_fmt_coord(fwd)}m, lateral={_fmt_coord(lat)}m")
    if overflow_counter is not None:
        overflow_counter[0] += n_overflow
    return "\n".join(lines), n_overflow


# ─── prompt rebuild ──────────────────────────────────────────────────────────

V3_TASK_HEADER = (
    "You are an expert autonomous driving agent. Analyze the scene and produce a "
    "single JSON object with these fields, in order:\n"
    "1. critical_objects: for each of "
    f"[{', '.join(CRITICAL_CATEGORIES)}] give a short noun phrase naming the "
    "relevant instance, or \"none\".\n"
    "2. complexity: \"simple\" or \"complex\" — whether the scene needs extra "
    "planning attention.\n"
    "3. explanation: a brief free-text rationale for the driving decision.\n"
    "4. future_meta_behavior: {longitudinal: one of "
    f"[{', '.join(LONG_VERBS)}], lateral: one of [{', '.join(LAT_VERBS)}]}}.\n"
    "5. trajectory: the next 5 s as 10 waypoints at 0.5 s spacing, each line "
    "\"<t>s: forward=<sign><XX.X>m, lateral=<sign><YY.Y>m\".\n"
)


def build_v3_prompt(src_prompt: str) -> str:
    """Reuse the source 'Input:' section (nav + ego history) verbatim, replacing
    the Task 1-4 spec with the V3 task header."""
    src_prompt = src_prompt.replace("<image>", "").strip()
    i = src_prompt.find("Input:")
    input_block = src_prompt[i:] if i >= 0 else ""
    return V3_TASK_HEADER + "\n" + input_block


# ─── per-sample conversion ───────────────────────────────────────────────────

def parse_src_response(resp_value: str) -> dict:
    """Parse the source assistant JSON (with mdm tags) into a dict."""
    obj = json.loads(resp_value)
    return obj


def convert_sample(sample: dict, overflow_counter=None) -> dict:
    src_resp = parse_src_response(sample["conversations"][1]["value"])
    src_co = src_resp.get("critical_objects", {}) or {}

    co = map_critical_objects(src_co)
    complexity = derive_complexity(sample, src_co)
    explanation = _strip_mdm(src_resp.get("explanation", ""))
    longitudinal = map_longitudinal(sample.get("meta_speed", ""))
    lateral = map_lateral(sample.get("meta_direction", ""))
    traj_str, _ = map_trajectory(sample.get("future waypoints", []) or [], overflow_counter)

    v3_response = {
        "critical_objects": co,
        "complexity": complexity,
        "explanation": explanation,
        "future_meta_behavior": {"longitudinal": longitudinal, "lateral": lateral},
        "trajectory": traj_str,
    }
    # Match build_template_v3 serialization (separators=(", ", ": ")).
    resp_str = json.dumps(v3_response, separators=(", ", ": "), ensure_ascii=False)

    v3_prompt = build_v3_prompt(sample["conversations"][0]["value"])

    out = {
        "image": sample.get("image"),
        "conversations": [
            {"from": "human", "value": v3_prompt},
            {"from": "gpt", "value": resp_str},
        ],
        # carry-through metadata useful for eval / debugging
        "navigation_command": sample.get("navigation_command"),
        "future waypoints": sample.get("future waypoints"),
        "meta_speed": sample.get("meta_speed"),
        "meta_direction": sample.get("meta_direction"),
        "sample_id": sample.get("sample_id"),
        "complexity": complexity,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=0, help="convert only first N (0 = all)")
    ap.add_argument("--preview", action="store_true",
                    help="print converted samples to stdout instead of writing a file")
    args = ap.parse_args()

    with open(args.src) as f:
        data = json.load(f)
    if args.n > 0:
        data = data[: args.n]

    overflow = [0]
    converted = []
    complexities = {"simple": 0, "complex": 0}
    expl_lens = []
    for s in data:
        c = convert_sample(s, overflow_counter=overflow)
        converted.append(c)
        complexities[c["complexity"]] = complexities.get(c["complexity"], 0) + 1
        expl_lens.append(len(c["conversations"][1]["value"]))

    if args.preview:
        for i, c in enumerate(converted):
            print(f"\n{'='*90}\n#{i}  sample_id={c['sample_id']}  nav={c['navigation_command']}  "
                  f"meta_speed={c['meta_speed']}  meta_direction={c['meta_direction']}  "
                  f"complexity={c['complexity']}")
            print(f"--- V3 PROMPT (first 400 chars) ---\n{c['conversations'][0]['value'][:400]}")
            print(f"--- V3 RESPONSE ---")
            # pretty-print the response JSON for readability
            resp = json.loads(c["conversations"][1]["value"])
            print(json.dumps(resp, indent=2, ensure_ascii=False))
    else:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(converted, f, ensure_ascii=False)
        print(f"wrote {len(converted)} samples -> {args.out}")

    print(f"\n=== summary ===")
    print(f"  samples: {len(converted)}")
    print(f"  complexity: {complexities}")
    print(f"  trajectory coord overflow (|v|>99.9, clamped): {overflow[0]}")
    if expl_lens:
        print(f"  explanation chars: mean {sum(expl_lens)//len(expl_lens)}, "
              f"min {min(expl_lens)}, max {max(expl_lens)}")


if __name__ == "__main__":
    main()
