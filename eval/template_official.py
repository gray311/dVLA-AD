"""Official dVLM-AD prompt + template, aligned with `SaFo-Lab/dVLM-AD/eval/inference.py`.

Differences vs my V3:
- critical_objects: **1** mask token per category (yes/no), not 10
- meta-behavior verbs: official dictionary (speed ∈ {keep,accelerate,decelerate,stop,other},
  command ∈ {straight,yield,left_turn,right_turn,lane_follow,lane_change_left,
  lane_change_right,reverse,overtake,other})
- trajectory: 5 waypoints @ 1 s intervals (not 20 @ 0.25 s)
- historical ego state: 7 points over 3 s @ 0.5 s, with position + accel + velocity per point
- prompt is structured as Task 1 / 2 / 3 / 4
"""
from __future__ import annotations
import json
import math
import re
from typing import List, Tuple

MASK = "<|mdm_mask|>"

# Legacy structured categories (kept for backward compat with V2). For V3
# we use a single free-form `crucial_objects` list slot instead.
CRITICAL_CATEGORIES = [
    "lead_vehicle", "nearby_vehicle", "pedestrian",
    "traffic_element", "weather_condition", "road_hazard",
]

# Slot lengths — V3 schema (5 slots)
N_CRITICAL_TOKENS = 3            # legacy (unused in V3)
N_BEHAVIOR_TOKENS = 4
N_EXPLANATION_TOKENS = 80        # explanation free-form prose
N_TRAJECTORY_TOKENS = 100        # legacy
N_COUNTRY_TOKENS = 3
N_RISK_TOKENS = 2
N_CRUCIAL_LIST_TOKENS = 18       # free-form crucial-objects list

PROMPT_OFFICIAL_MINIMAL = (
    "You are a driving agent. Fill in the JSON template based on the image.\n"
    "Navigation: {nav}. Past ego state: {history}\n"
    "Output:"
)

PROMPT_OFFICIAL_HYBRID = (
    "You are a driving agent. Fill the JSON template based on the image.\n"
    "- country: country/region (e.g. \"USA\", \"China\").\n"
    "- risk_level ∈ {low, medium, high, critical}.\n"
    "- crucial_objects: comma-separated list of important objects/conditions visible "
    "(e.g. types + colors + positions). \"none\" if road is empty.\n"
    "- explanation: 1-2 sentences on visible objects and how to react.\n"
    "- speed ∈ {keep, accelerate, decelerate, stop}.\n"
    "- command must match nav: GO_LEFT→left_turn, GO_RIGHT→right_turn, GO_STRAIGHT→straight.\n\n"
    "Input:\n- <image>: front camera.\n- Navigation: {nav}\n- Past ego: {history}\n\nOutput:"
)

PROMPT_OFFICIAL_FULL = (
    "You are an expert autonomous driving agent.\n"
    "Task 1: Scene Context\n"
    "- country: infer the country/region from visual cues (road signs, plates, vehicles, "
    "lane markings). Examples: \"USA\", \"China Shanghai\", \"Germany\".\n"
    "- risk_level: one of {low, medium, high, critical}. Pick by severity of any visible "
    "threat. \"critical\" only for active fire / crash / pedestrian-in-path.\n"
    "Task 2: Critical Object Detection\n"
    "For each category, fill a SHORT keyword (max 3 words) describing any visible "
    "instance, OR \"none\".\n"
    "- lead_vehicle: the SINGLE vehicle directly ahead in OUR lane (the one we are "
    "following). Example: \"black sedan 8m\" / \"truck close\" / \"none\".\n"
    "- nearby_vehicle: other vehicles around (parked, opposite lane, side). \"none\" "
    "if no other car.\n"
    "- pedestrian: people on/near road. - cyclist / construction / weather_condition "
    "/ road_hazard / emergency_vehicle / animal / special_vehicle / conflicting_vehicle / "
    "door_opening_vehicle: same rules.\n"
    "- traffic_element: traffic light / stop sign / lane marking sign.\n"
    "Weather is \"none\" for clear sky; only \"rain\" / \"fog\" / \"snow\" / \"haze\".\n"
    "Task 3: Scene Reasoning\n"
    "Briefly explain (in English) what is visible that matters and how the ego vehicle "
    "should react in the next 3 s. Reference specific objects.\n"
    "Task 4: Meta-Behavior Prediction\n"
    "- speed ∈ {keep, accelerate, decelerate, stop, other}. Pick \"decelerate\" if a "
    "vehicle is slowing ahead, stop sign, red light, or hazard.\n"
    "- command ∈ {straight, lane_follow, lane_change_left, lane_change_right, "
    "left_turn, right_turn, yield, overtake, reverse, other}. Pick "
    "\"left_turn\"/\"right_turn\" for GO_LEFT/GO_RIGHT; \"straight\"/\"lane_follow\" "
    "for GO_STRAIGHT.\n"
    "Task 5: Trajectory Prediction\n"
    "5 future waypoints, 1 s apart, ego-frame meters (x = forward, y = left).\n\n"
    "Input:\n"
    "- <image>: three front-view frames (front-left, front, front-right).\n"
    "- High-level navigation command: {nav}\n"
    "- Historical ego state over last 3.0 s (0.5 s intervals):{history}\n\n"
    "Output:"
)

# Choose prompt: env PROMPT_MODE ∈ {full, minimal, hybrid}. Default = hybrid.
import os as _os
_mode = _os.environ.get("PROMPT_MODE", "hybrid").lower()
if _mode == "minimal":
    PROMPT_OFFICIAL = PROMPT_OFFICIAL_MINIMAL
elif _mode == "full":
    PROMPT_OFFICIAL = PROMPT_OFFICIAL_FULL
else:
    PROMPT_OFFICIAL = PROMPT_OFFICIAL_HYBRID


def build_template_ids_official(tokenizer, mask_id: int, structured_traj: bool = False,
                                  n_waypoints: int = 5, n_mask_per_number: int = 6,
                                  baseline_traj=None):
    """Return (ids: List[int], slots: List[(pos, kind)]) for the OFFICIAL template.

    Segment-by-segment encoding so the mask token is *always* a single id even
    on tokenizers that don't recognise `<|mdm_mask|>` (Qwen2.5-VL family).

    structured_traj=True: split the trajectory blob into per-waypoint scaffold
    where brackets/commas are FIXED text and only numeric values are mask.
    Each number gets `n_mask_per_number` mask tokens (room for "+12.34" etc.).
    Shape: "[[<m>×6, <m>×6], [<m>×6, <m>×6], …, [<m>×6, <m>×6]]"  with n_waypoints points.

    baseline_traj: optional list of (x, y) tuples (n_waypoints long) used to
    pre-fill the integer-digit portion of each number as FIXED scaffold.
    Only the sign + fractional digits stay as mask. Falls back to digit slots
    if None.
    """
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    ids: list[int] = []
    slots: list[tuple[int, str]] = []
    def add_masks(n: int, kind: str):
        for _ in range(n):
            slots.append((len(ids), kind))
            ids.append(mask_id)
    # V3 schema (5 slots): country / risk_level / crucial_objects(free list) / explanation / behavior / trajectory
    ids += enc('{"country": "')
    add_masks(N_COUNTRY_TOKENS, "country")
    ids += enc('", "risk_level": "')
    slots.append((len(ids), "risk_head"))
    ids.append(mask_id)
    add_masks(N_RISK_TOKENS - 1, "risk_tail")
    ids += enc('", "crucial_objects": "')
    add_masks(N_CRUCIAL_LIST_TOKENS, "crucial_list")
    ids += enc('", "explanation": "')
    add_masks(N_EXPLANATION_TOKENS, "explanation")
    ids += enc('", "future_meta_behavior": {"longitudinal": "')
    # First mask = verb head (gated to legal verb set)
    slots.append((len(ids), "long_head"))
    ids.append(mask_id)
    add_masks(N_BEHAVIOR_TOKENS - 1, "long_tail")
    ids += enc('", "lateral": "')
    slots.append((len(ids), "lat_head"))
    ids.append(mask_id)
    add_masks(N_BEHAVIOR_TOKENS - 1, "lat_tail")
    ids += enc('"}, "trajectory": "')
    if structured_traj:
        # Per number, character-level structure. Each mask = 1 token (Qwen treats
        # each digit as 1 token).
        # Without baseline: <sign> <tens> <ones> . <frac1> <frac2>          (5 masks)
        # With baseline:    <sign> [tens fixed] [ones fixed] . <frac1> <frac2> (3 masks)
        # The decimal point is FIXED scaffold either way.
        ids += enc('[')
        for w in range(n_waypoints):
            if w > 0:
                ids += enc(', ')
            ids += enc('[')
            # ---- x number ----
            if baseline_traj is not None and w < len(baseline_traj):
                bx = int(round(baseline_traj[w][0]))
                bx = max(-99, min(99, bx))
                ids += enc('+' if bx >= 0 else '-')              # FIXED sign
                int_str = f"{abs(bx):02d}"
                ids += enc(int_str)                              # FIXED int digits
            else:
                add_masks(1, "traj_sign")
                add_masks(1, "traj_int_tens")
                add_masks(1, "traj_int_ones")
            ids += enc('.')
            add_masks(1, "traj_frac_tenths")
            add_masks(1, "traj_frac_hundredths")
            ids += enc(',')
            # ---- y number ----
            if baseline_traj is not None and w < len(baseline_traj):
                by = int(round(baseline_traj[w][1]))
                by = max(-99, min(99, by))
                ids += enc('+' if by >= 0 else '-')
                int_str = f"{abs(by):02d}"
                ids += enc(int_str)
            else:
                add_masks(1, "traj_sign")
                add_masks(1, "traj_int_tens")
                add_masks(1, "traj_int_ones")
            ids += enc('.')
            add_masks(1, "traj_frac_tenths")
            add_masks(1, "traj_frac_hundredths")
            ids += enc(']')
        ids += enc(']')
    else:
        add_masks(N_TRAJECTORY_TOKENS, "trajectory")    # = 100 free masks
    ids += enc('"}')
    return ids, slots


def build_template_official() -> str:
    """Render the official JSON template (string with <|mdm_mask|> literals)."""
    co = {cat: MASK * N_CRITICAL_TOKENS for cat in CRITICAL_CATEGORIES}
    obj = {
        "critical_objects": co,
        "explanation": MASK * N_EXPLANATION_TOKENS,
        "future_meta_behavior": {
            "longitudinal": MASK * N_BEHAVIOR_TOKENS,
            "lateral": MASK * N_BEHAVIOR_TOKENS,
        },
        "trajectory": MASK * N_TRAJECTORY_TOKENS,
    }
    return json.dumps(obj, separators=(", ", ": "), ensure_ascii=False)


def _subsample_history(sample: dict, n_target: int = 7):
    """Subsample 7 ego-state points from the raw 16 history waypoints / velocity / accel.

    Raw data: history waypoints are at 10 Hz over 1.6 s (16 points). The official
    prompt format expects 7 points over 3 s @ 0.5 s. We linearly interpolate or
    just sample evenly from whatever history is available, keeping the official
    time labels (t-3.0s … t+0.0s).
    """
    hist = sample["history waypoints"]  # 16 × 2
    vel = sample["velocity"]              # 16 × 2
    acc = sample["acceleration"]          # 16 × 2
    n_raw = len(hist)
    # Take evenly-spaced indices (oldest → newest)
    idxs = [int(round(i * (n_raw - 1) / (n_target - 1))) for i in range(n_target)]
    parts = []
    times = [f"(t-{3.0 - 0.5 * i:.1f}s)" if i < n_target - 1 else "(t+0.0s)" for i in range(n_target)]
    for t_label, idx in zip(times, idxs):
        x, y = hist[idx]
        ax, ay = acc[idx]
        vx, vy = vel[idx]
        parts.append(f"{t_label} [{x:.2f}, {y:.2f}], "
                     f"Acceleration: X {ax:.2f}, Y {ay:.2f} m/s², "
                     f"Velocity: X {vx:.2f}, Y {vy:.2f} m/s,")
    return "; ".join(parts)


def build_prompt_official(sample: dict) -> str:
    history_str = _subsample_history(sample, n_target=7)
    # Use plain string replacement to avoid clashes with literal {…} in the prompt
    return (PROMPT_OFFICIAL
            .replace("{nav}", str(sample["navigation_command"]))
            .replace("{history}", history_str))


# === Trajectory parser (5 waypoints @ 1s) ===

_PAIR_RE = re.compile(
    r"([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*[,\s]\s*"
    r"([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
)


def _extract_trajectory_field(filled_text: str) -> str:
    try:
        i = filled_text.find("{")
        j = filled_text.rfind("}")
        if i >= 0 and j > i:
            data = json.loads(filled_text[i:j+1])
            tr = data.get("trajectory", "")
            return tr if isinstance(tr, str) else ""
    except Exception:
        pass
    m = re.search(r'"trajectory"\s*:\s*"([^"]*)"', filled_text, re.DOTALL)
    return m.group(1) if m else ""


def parse_filled_5wp(filled_text: str) -> List[Tuple[float, float]]:
    """Extract up to 5 (x, y) pairs from the filled trajectory field."""
    tr = _extract_trajectory_field(filled_text)
    if not tr:
        tr = filled_text
    tr = tr.replace(MASK, " ")
    pairs = _PAIR_RE.findall(tr)
    return [(float(x), float(y)) for x, y in pairs][:5]


def upsample_to_20(pred5: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Linearly interpolate 5 wp @ 1 s → 20 wp @ 0.25 s so scoring is comparable.

    Output indices i (0-based) correspond to t = 0.25 * (i+1).
    pred5 indices j (0-based) correspond to t = 1, 2, 3, 4, 5.
    We add t=0 → (0, 0) as anchor.
    """
    if not pred5:
        return [(0.0, 0.0)] * 20
    if len(pred5) < 5:
        pred5 = list(pred5) + [pred5[-1]] * (5 - len(pred5))
    anchors_t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    anchors = [(0.0, 0.0)] + list(pred5[:5])
    out = []
    for i in range(20):
        t = 0.25 * (i + 1)
        # find segment
        for k in range(len(anchors_t) - 1):
            if anchors_t[k] <= t <= anchors_t[k + 1]:
                t0, t1 = anchors_t[k], anchors_t[k + 1]
                (x0, y0), (x1, y1) = anchors[k], anchors[k + 1]
                a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                out.append((x0 + a * (x1 - x0), y0 + a * (y1 - y0)))
                break
        else:
            out.append(anchors[-1])
    return out
