#!/usr/bin/env python3
"""Convert dvlm-ad CoT data -> dVLA Stage-1 V3 template.

Prompt and template are the CANONICAL ones from ``eval/template_v3.py`` — this
script imports ``build_prompt_v3`` / ``build_template_v3`` directly so the
training data, eval scaffold and inference loader all share one definition.

Source schema (one object per sample, e.g. ``data/example/sample.json`` or
``dvlm-ad_waymo_training_cot.json``):
    {
      "image": [front_left, front, front_right],
      "conversations": [{"from":"human", ...}, {"from":"gpt", "value": "{...}"}],
      "future waypoints": [[fwd, lat] x20 @ 0.25s],   # 5 s horizon
      "history waypoints": [...], "velocity": [...], "acceleration": [...],
      "meta_speed": "...", "meta_direction": "...", "meta_diag": {...},
      "navigation_command": "...", "sample_id": "..."
    }

Target (V3, per ``eval/template_v3.py``):
    human : build_prompt_v3(sample)        # INPUT + TASK + worked example + masked TEMPLATE tail
    gpt   : --split train -> filled V3 template (GT values)
            --split eval  -> build_template_v3()  (every slot = <|mdm_mask|>)

V3 response slots (key order matches build_template_v3):
    critical_objects {12 cats: "<=2-tok phrase" | "none"}, complexity {simple|complex},
    explanation <free text>, future_meta_behavior {longitudinal, lateral},
    trajectory "<t>s: forward=<sign><tens><ones>.<frac>m, lateral=<...>m\\n..." (10 wp @ 0.5s)

Usage:
    python data_convert.py --src example/sample.json --n 2 --preview
    python data_convert.py --split train --src .../dvlm-ad_waymo_training_cot.json --out .../train_v3.json
    python data_convert.py --split eval  --src .../dvlm-ad_waymo_e2e_val_cot.json  --out .../val_v3.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Canonical prompt + template definition lives in the repo-root eval package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))   # .../dVLA-AD
sys.path.insert(0, os.path.join(_REPO, "eval"))
import template_v3 as T  # noqa: E402

CRITICAL_CATEGORIES = T.CRITICAL_CATEGORIES
LONG_VERBS = T.LONG_VERBS
LAT_VERBS = T.LAT_VERBS
N_WAYPOINTS = T.N_WAYPOINTS        # 10
N_CRITICAL_TOKENS = T.N_CRITICAL_TOKENS  # 2 — fixed mask/value budget per critical slot
SRC_DT = 0.25                      # source future-waypoint spacing (0.25 s)

# Tokenizer used to NULL-pad critical values to exactly N_CRITICAL_TOKENS tokens
# (so train GT matches the 2-token inference scaffold). Set in main() when
# --pad_critical is on; None disables padding (variable-length critical).
_PAD_TOK = None

DEFAULT_SRC = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_training_cot.json"
DEFAULT_OUT = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/data/dvla_sft/dvlm-ad_waymo_training_v3.json"

_MDM_START = "<|mdm_start|>"
_MDM_END = "<|mdm_end|>"

# ─── field mappings ──────────────────────────────────────────────────────────
# meta_speed (source) -> longitudinal verb (T.LONG_VERBS)
SPEED_TO_LONG = {
    "keep_speed": "keep speed", "accelerate": "speed up", "accel": "speed up",
    "decelerate": "slow down", "decel": "slow down",
    "stop": "stop now", "stop_now": "stop now", "other": "keep speed",
}
# meta_direction (source) -> lateral verb (T.LAT_VERBS)
DIR_TO_LAT = {
    "keep_lane": "keep lane", "left_turn": "turn left", "left_turn_in_lane": "turn left",
    "right_turn": "turn right", "right_turn_in_lane": "turn right",
    "left_lane_change": "change left", "right_lane_change": "change right", "other": "keep lane",
}
# critical_objects "yes" -> short (<=2-tok) referring phrase; "no" -> "none".
CRITICAL_YES_PHRASE = {
    "nearby_vehicle": "vehicle", "pedestrian": "person", "cyclist": "cyclist",
    "construction": "construction", "traffic_element": "traffic light",
    "weather_condition": "bad weather", "road_hazard": "road hazard",
    "emergency_vehicle": "emergency vehicle", "animal": "animal",
    "special_vehicle": "special vehicle", "conflicting_vehicle": "cross vehicle",
    "door_opening_vehicle": "open door",
}


# ─── slot mappers ────────────────────────────────────────────────────────────
def _strip_mdm(s: str) -> str:
    return (s or "").replace(_MDM_START, "").replace(_MDM_END, "").strip()


def map_longitudinal(meta_speed: str) -> str:
    return SPEED_TO_LONG.get((meta_speed or "").strip().lower(), "keep speed")


def map_lateral(meta_direction: str) -> str:
    return DIR_TO_LAT.get((meta_direction or "").strip().lower(), "keep lane")


def map_critical_objects(src_co: dict) -> dict:
    """yes/no -> {short phrase | 'none'} for all 12 categories, fixed order."""
    out = {}
    for cat in CRITICAL_CATEGORIES:
        raw = str(src_co.get(cat, "no")).strip().lower()
        out[cat] = CRITICAL_YES_PHRASE.get(cat, "object") if raw == "yes" else "none"
    return out


def pad_critical_to_budget(co: dict) -> dict:
    """NULL-pad each critical value to exactly N_CRITICAL_TOKENS tokens, so the
    train GT matches the fixed 2-token inference scaffold (1-token answers learn
    to emit '<word><|NULL|>'). No-op if _PAD_TOK is unset."""
    if _PAD_TOK is None:
        return co
    out = {}
    for k, v in co.items():
        n = len(_PAD_TOK.encode(v, add_special_tokens=False))
        if n < N_CRITICAL_TOKENS:
            v = v + "<|NULL|>" * (N_CRITICAL_TOKENS - n)
        elif n > N_CRITICAL_TOKENS:  # safety: shouldn't happen (all phrases <=2)
            v = _PAD_TOK.decode(_PAD_TOK.encode(v, add_special_tokens=False)[:N_CRITICAL_TOKENS])
        out[k] = v
    return out


def derive_complexity(sample: dict, src_co: dict) -> str:
    """Heuristic simple/complex from metadata (no human labels)."""
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
        mm not in ("normal", "stationary", "none", ""),
        overtake,
        mode not in ("none", ""),
        net_heading > 15.0,
        lane_shift > 2.0,
        ms in ("stop", "stop_now"),
    ]
    return "complex" if any(complex_signals) else "simple"


def _fmt_coord_v3(v: float) -> str:
    """Coordinate as ``<sign><tens><ones>.<frac>`` — mirrors ``template_v3._fwd``
    so GT exactly matches the prompt's worked example and the masked template's
    ``forward=<sign><tens><ones>.<frac>m`` scaffold. Magnitudes >=100 m extend
    the integer part to 3 digits (no clamp)."""
    sign = "+" if v >= 0 else "-"
    av = abs(v)
    return f"{sign}{int(av // 10):01d}{int(av % 10):01d}.{int((av * 10) % 10):01d}"


def subsample_waypoints(future_waypoints):
    """Source 20 wp @ 0.25 s (index i -> (i+1)*0.25 s) -> 10 wp @ 0.5 s
    (indices 1,3,..,19). Returns a list of (forward, lateral) tuples."""
    want_idx = [2 * (w + 1) - 1 for w in range(N_WAYPOINTS)]  # 1,3,...,19
    fw = future_waypoints or []
    pairs = []
    for idx in want_idx:
        if idx < len(fw):
            pairs.append((fw[idx][0], fw[idx][1]))
        else:
            pairs.append((fw[-1][0], fw[-1][1]) if fw else (0.0, 0.0))
    return pairs


def build_filled_template_v3(co: dict, complexity: str, explanation: str,
                             long_verb: str, lat_verb: str, traj_pairs) -> str:
    """GT-filled V3 response, structurally identical to ``T.build_template_v3``
    (same keys / order / separators) but with real values in place of masks."""
    traj_lines = [
        f"{(w + 1) * T.TRAJECTORY_DT:.1f}s: "
        f"forward={_fmt_coord_v3(fwd)}m, lateral={_fmt_coord_v3(lat)}m"
        for w, (fwd, lat) in enumerate(traj_pairs)
    ]
    obj = {
        "critical_objects": co,
        "complexity": complexity,
        "explanation": explanation,
        "future_meta_behavior": {"longitudinal": long_verb, "lateral": lat_verb},
        "trajectory": "\n".join(traj_lines),
    }
    return json.dumps(obj, separators=(", ", ": "), ensure_ascii=False)


# ─── per-sample conversion ───────────────────────────────────────────────────
def convert_sample(sample: dict) -> dict:
    """train split: V3 prompt + GT-filled V3 template."""
    convs = sample.get("conversations") or []
    if len(convs) < 2:
        raise ValueError("missing conversations[0/1]")
    src_resp = json.loads(convs[1]["value"])
    src_co = src_resp.get("critical_objects", {}) or {}

    co = pad_critical_to_budget(map_critical_objects(src_co))
    complexity = derive_complexity(sample, src_co)
    explanation = _strip_mdm(src_resp.get("explanation", ""))
    long_verb = map_longitudinal(sample.get("meta_speed", ""))
    lat_verb = map_lateral(sample.get("meta_direction", ""))
    traj_pairs = subsample_waypoints(sample.get("future waypoints", []) or [])

    resp_str = build_filled_template_v3(co, complexity, explanation,
                                        long_verb, lat_verb, traj_pairs)
    return {
        "image": sample.get("image"),
        "conversations": [
            {"from": "human", "value": T.build_prompt_v3(sample)},
            {"from": "gpt", "value": resp_str},
        ],
        "navigation_command": sample.get("navigation_command"),
        "future waypoints": sample.get("future waypoints"),
        "meta_speed": sample.get("meta_speed"),
        "meta_direction": sample.get("meta_direction"),
        "sample_id": sample.get("sample_id"),
        "complexity": complexity,
    }


def convert_sample_eval(sample: dict) -> dict:
    """eval split (val/test): V3 prompt + fully-masked V3 template.

    No GT slot labels / meta in val/test — only GT ``future waypoints`` (for
    ADE) and ids are carried through. The masked scaffold is the canonical
    ``T.build_template_v3()`` (every slot = ``<|mdm_mask|>``)."""
    convs = sample.get("conversations") or []
    if len(convs) < 1:
        raise ValueError("missing conversations[0]")
    return {
        "image": sample.get("image"),
        "conversations": [
            {"from": "human", "value": T.build_prompt_v3(sample)},
            {"from": "gpt", "value": T.build_template_v3()},
        ],
        "navigation_command": sample.get("navigation_command"),
        "future waypoints": sample.get("future waypoints"),
        "sample_id": sample.get("sample_id"),
        "timestamp_micros": sample.get("timestamp_micros"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="source JSON (list of samples)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output V3 JSON path")
    ap.add_argument("--split", choices=["train", "eval"], default="train",
                    help="train: GT-filled V3 (default). eval: masked V3 scaffold (val/test).")
    ap.add_argument("--n", type=int, default=0, help="convert only first N (0 = all)")
    ap.add_argument("--preview", action="store_true",
                    help="print converted samples to stdout instead of writing a file")
    ap.add_argument("--pad_critical", type=int, default=1,
                    help="1: NULL-pad critical values to 2 tokens (matches inference scaffold); 0: off")
    ap.add_argument("--tokenizer", default="/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/models/Fast_dVLM_3B_sasd",
                    help="tokenizer for critical NULL-padding (must have <|NULL|> as 1 token)")
    args = ap.parse_args()

    if args.split == "train" and args.pad_critical:
        global _PAD_TOK
        from transformers import AutoTokenizer
        _PAD_TOK = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        print(f"critical NULL-padding ON (budget={N_CRITICAL_TOKENS}, tokenizer={args.tokenizer})")

    with open(args.src) as f:
        data = json.load(f)
    if args.n > 0:
        data = data[: args.n]

    converted = []
    skipped = 0
    complexities = {"simple": 0, "complex": 0}
    resp_lens = []
    for idx, s in enumerate(data):
        try:
            c = convert_sample_eval(s) if args.split == "eval" else convert_sample(s)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [skip #{idx}] {type(e).__name__}: {e}")
            continue
        converted.append(c)
        if args.split == "train":
            complexities[c["complexity"]] = complexities.get(c["complexity"], 0) + 1
        resp_lens.append(len(c["conversations"][1]["value"]))

    if args.preview:
        for i, c in enumerate(converted):
            print(f"\n{'=' * 90}\n#{i}  sample_id={c.get('sample_id')}  "
                  f"nav={c.get('navigation_command')}  meta_speed={c.get('meta_speed')}  "
                  f"meta_direction={c.get('meta_direction')}  complexity={c.get('complexity')}")
            print(f"--- V3 PROMPT ---\n{c['conversations'][0]['value']}")
            print("--- V3 RESPONSE ---")
            print(json.dumps(json.loads(c["conversations"][1]["value"]),
                             indent=2, ensure_ascii=False))
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(converted, f, ensure_ascii=False)
        print(f"wrote {len(converted)} samples -> {args.out}")

    print("\n=== summary ===")
    print(f"  split         : {args.split}")
    print(f"  input samples : {len(data)}")
    print(f"  converted     : {len(converted)}")
    print(f"  skipped       : {skipped}")
    if args.split == "train":
        print(f"  complexity    : {complexities}")
    if resp_lens:
        print(f"  response chars: mean {sum(resp_lens) // len(resp_lens)}, "
              f"min {min(resp_lens)}, max {max(resp_lens)}")


if __name__ == "__main__":
    main()
