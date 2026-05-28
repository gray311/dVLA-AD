"""PROMPT_V3 + TEMPLATE_V3 builders and trajectory parser.

For the trajectory task we adapt the template's trajectory horizon to our
Waymo-50 setup: 20 waypoints @ 0.25 s spacing (5 s horizon), not the
template's default 8 @ 0.5 s.

The mask token for this backbone is the literal `|<MASK>|` (single id 151665);
see the MASK constant below. After fill, we extract:
  - critical_objects (12 entries × 2 mask slots each)
  - complexity (1 mask slot)
  - explanation (N_EXPLANATION_TOKENS mask slots)
  - future_meta_behavior.{longitudinal, lateral} (2 + 2)
  - trajectory (10 waypoints, sign+digit gated slots)
"""
from __future__ import annotations
import math
import re
import json
from typing import List, Tuple

# Backbone mask token. Fast_dDrive_as_dVLM (model_type=fast_dvlm) tokenizes
# `|<MASK>|` to a single id (151665) — the same mask_id the finetuner uses.
# The legacy `<|mdm_mask|>` (LaViDa/DiffusionVL) is NOT a single token here, so
# it must not be used in scaffolds for this backbone.
MASK = "|<MASK>|"

CRITICAL_CATEGORIES = [
    "nearby_vehicle", "pedestrian", "cyclist", "construction",
    "traffic_element", "weather_condition", "road_hazard",
    "emergency_vehicle", "animal", "special_vehicle",
    "conflicting_vehicle", "door_opening_vehicle",
]

# Per-slot mask counts (V3 spec)
# critical_objects: 2 mask tokens per category — just enough for "none" or a
# short 2-token phrase like "black car" / "green light".
N_CRITICAL_TOKENS = 2
N_EXPLANATION_TOKENS = 100

# Trajectory: 10 waypoints @ 2 Hz (0.5 s spacing) covering 5 s horizon.
# Each waypoint = `<sign><tens><ones>.<frac>,<sign><tens><ones>.<frac>` with
# fixed scaffold `.`, `,`, `.`, `;` between mask slots → decoded form
# `+XX.X,+YY.Y;+XX.X,+YY.Y;...`. One decimal place per coordinate.
N_WAYPOINTS = 10
TRAJECTORY_DT = 0.5  # seconds between waypoints (2 Hz)
# N_TRAJECTORY_TOKENS retained for the legacy string-render `build_template_v3`
# (used as the human-readable copy at the tail of the prompt). The token-id
# builder uses the structured layout.
N_TRAJECTORY_TOKENS = N_WAYPOINTS * 8  # 8 mask slots per waypoint

# Behavior layout: 2 mask tokens per field — just the 2-word verb.
#     "verb_w1 verb_w2"  →  e.g. "speed up" / "turn left" / "keep lane".
# Each mask gets its own per-position vocab gate.
LONG_VERBS = ["speed up", "slow down", "keep speed", "stop now"]
LAT_VERBS = ["keep lane", "turn left", "turn right", "change left", "change right"]

# Scene-complexity tag: 1 mask token, gated to {"simple", "complex"}.
# "complex" when the scene has many agents (multiple vehicles / pedestrians /
# cyclists), an unfolding hazard (accident ahead, fire/smoke, blocked lane,
# emergency vehicle in path), or any other condition that the model should
# flag for downstream planning (e.g. budget more replanning compute).
COMPLEXITY_LABELS = ["simple", "complex"]


def build_template_v3() -> str:
    """JSON scaffold with the |<MASK>| token at every slot the model should fill
    (string-rendered version, for the human-readable TEMPLATE block at the tail
    of the prompt). The token-id-level builder is `build_template_ids_v3` below
    and is the source of truth used by loaders.
    """
    co = {cat: MASK * N_CRITICAL_TOKENS for cat in CRITICAL_CATEGORIES}
    # Behavior: 2 mask tokens (verb only) with fixed scaffold space between them.
    behavior_str = f"{MASK} {MASK}"
    # Trajectory: semantic per-waypoint lines (the actual token layout in
    # build_template_ids_v3 uses sign+digits gates; this string is just for
    # display in the human-readable TEMPLATE block at the tail of the prompt).
    traj_lines = []
    for w in range(N_WAYPOINTS):
        t = (w + 1) * TRAJECTORY_DT
        # 4 masks (sign,tens,ones,frac) for forward and another 4 for lateral.
        # The "." that splits ones from frac is scaffold, kept as a literal `.`.
        slot = MASK * 3 + "." + MASK
        traj_lines.append(f"{t:.1f}s: forward={slot}m, lateral={slot}m")
    traj_str = "\n".join(traj_lines)
    obj = {
        "critical_objects": co,
        "complexity": MASK,
        "explanation": MASK * N_EXPLANATION_TOKENS,
        "future_meta_behavior": {
            "longitudinal": behavior_str,
            "lateral": behavior_str,
        },
        "trajectory": traj_str,
    }
    return json.dumps(obj, separators=(", ", ": "), ensure_ascii=False)


def build_template_ids_v3(tokenizer, mask_id: int):
    """Build the V3 JSON-scaffold token ids with mask_id at every slot.

    Returns (ids, slots, critical_pairs) where critical_pairs is a list of
    (head_local_pos, [tail_local_pos, ...]) tuples — one per critical_objects
    category. The loader uses these to do "tail EOS pre-fill" once the head
    is committed, so the block exits the diffusion step loop earlier.
    """
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    ids: list[int] = []
    slots: list[tuple[int, str]] = []
    critical_pairs: list[tuple[int, list[int]]] = []

    def add_masks(n: int, kind: str):
        for _ in range(n):
            slots.append((len(ids), kind))
            ids.append(mask_id)

    ids += enc('{"critical_objects": {')
    for i, cat in enumerate(CRITICAL_CATEGORIES):
        if i > 0:
            ids += enc(", ")
        ids += enc(f'"{cat}": "')
        # First mask in each category = "critical_head"; rest are "critical_tail".
        head_pos = len(ids)
        slots.append((head_pos, "critical_head"))
        ids.append(mask_id)
        tail_positions = []
        for _ in range(N_CRITICAL_TOKENS - 1):
            tail_positions.append(len(ids))
            slots.append((len(ids), "critical_tail"))
            ids.append(mask_id)
        critical_pairs.append((head_pos, tail_positions))
        ids += enc('"')
    # Complexity tag (1 mask token, gated to {"simple","complex"}). Placed
    # right after critical_objects so the model can use the just-committed
    # category values to inform the complexity judgement, AND so a "complex"
    # commit cascades into the subsequent explanation + behavior decisions.
    # (Empirically this gives better behavior on hazard scenes vs putting
    # complexity at the template tail — verified on examples/test_image.png.)
    ids += enc('}, "complexity": "')
    slots.append((len(ids), "complexity"))
    ids.append(mask_id)
    ids += enc('", "explanation": "')
    add_masks(N_EXPLANATION_TOKENS, "explanation")
    # Behavior fields: each is just the 2-word verb "verb_w1 verb_w2" (2 masks
    # with a fixed scaffold space between them).
    ids += enc('", "future_meta_behavior": {"longitudinal": "')
    slots.append((len(ids), "long_w1")); ids.append(mask_id)
    ids += enc(' ')
    slots.append((len(ids), "long_w2")); ids.append(mask_id)
    ids += enc('", "lateral": "')
    slots.append((len(ids), "lat_w1")); ids.append(mask_id)
    ids += enc(' ')
    slots.append((len(ids), "lat_w2")); ids.append(mask_id)
    ids += enc('"}, "trajectory": "')
    # Semantic trajectory: 10 waypoints, each in the form
    #   `<t>s: forward=<sign><tens><ones>.<frac>m, lateral=<sign><tens><ones>.<frac>m`
    # newline-separated. Per-position gates:
    #   traj_sign → {+, -}
    #   traj_tens / traj_ones / traj_frac → digit 0-9
    # The semantic labels (`forward=`, `m`, `lateral=`, `<t>s:`) are SCAFFOLD
    # — committed before diffusion, so the mask positions see them as
    # context. Empirically this improves model grounding vs the older
    # compact `+XX.X,+YY.Y;...` format.
    for w in range(N_WAYPOINTS):
        t = (w + 1) * TRAJECTORY_DT  # 0.5, 1.0, ..., 5.0
        if w > 0:
            ids += enc('\n')
        # Scaffold: e.g. "0.5s: forward="
        ids += enc(f'{t:.1f}s: forward=')
        slots.append((len(ids), "traj_sign"));  ids.append(mask_id)
        slots.append((len(ids), "traj_tens"));  ids.append(mask_id)
        slots.append((len(ids), "traj_ones"));  ids.append(mask_id)
        ids += enc('.')
        slots.append((len(ids), "traj_frac"));  ids.append(mask_id)
        ids += enc('m, lateral=')
        slots.append((len(ids), "traj_sign"));  ids.append(mask_id)
        slots.append((len(ids), "traj_tens"));  ids.append(mask_id)
        slots.append((len(ids), "traj_ones"));  ids.append(mask_id)
        ids += enc('.')
        slots.append((len(ids), "traj_frac"));  ids.append(mask_id)
        ids += enc('m')
    ids += enc('"}')
    return ids, slots, critical_pairs


PROMPT_V3 = """You are an autonomous driving assistant. Reason like a driver in three stages: first PERCEIVE the scene, then PREDICT how it will change, then PLAN what to do.

INPUT:
- Multi-view images: front-left, front, front-right
- Ego state: speed={speed:.1f} m/s, longitudinal acceleration={accel:.2f} m/s^2
- Driver instruction: {instruction}
- Past 1.5 s of ego positions at 0.1 s spacing (x forward, y left, meters):
  {history}

OUTPUT:
- Return ONLY the completed JSON object. No commentary, no markdown fences.
- Every field is mandatory. Do not leave any mask empty.
- All planning fields (longitudinal, lateral, trajectory) must be mutually
  consistent and must obey the instruction below unless a hazard makes it unsafe.


FIELDS:
1. critical_objects — an object with these 12 keys. For each key, give a value
   that is either "none" (if absent) OR a short descriptor of AT MOST 2 words
   naming the most salient instance (e.g. "black car", "green light",
   "orange cone"):
     - nearby_vehicle      (any car/truck around: ahead, oncoming, or side)
     - pedestrian          (people on or near the road)
     - cyclist             (bicycle / motorcycle / scooter)
     - construction        (cone, barrier, work zone)
     - traffic_element     (traffic light, stop sign, lane-marking sign)
     - weather_condition   (lighting / weather, e.g. "clear day", "wet night")
     - road_hazard         (debris, pothole, fire, blocked lane, fallen object)
     - emergency_vehicle   (police, ambulance, fire truck)
     - animal
     - special_vehicle     (bus, trailer, construction truck)
     - conflicting_vehicle (vehicle whose path may cross yours)
     - door_opening_vehicle(parked car with a door opening)

2. complexity — exactly one word:
     "complex" if there are many vehicles/people OR any hazard
       (fire, accident, blocked lane, emergency vehicle);
     otherwise "simple".

3. explanation — plain words, NO numbers. Three short parts, in this order:
     Perceive: the road and lighting, plus the key vehicles, people, or hazards.
     Predict:  what those vehicles and people are likely to do next.
     Plan:     what you will therefore do, and why.

4. future_meta_behavior.longitudinal — choose ONE of:
     {{speed up, slow down, keep speed, stop now}}

5. future_meta_behavior.lateral — choose ONE of:
     {{keep lane, turn left, turn right, change left, change right}}
   Follow {instruction} unless a hazard makes it unsafe.

6. trajectory — 10 waypoints, one every 0.5 s (covering the next 5 s).
   Coordinate frame: origin (0, 0) = your current position; axes are ego-centric.
     - forward: meters ahead. Strictly increasing unless stopping. Spacing
       between consecutive points is about {step_m:.1f} m at current speed —
       larger if speeding up, smaller if slowing, approaching 0 if stopping.
     - lateral: meters to the side. + is LEFT, - is RIGHT, ~0 for straight.
       When turning or changing lane, magnitude grows smoothly over time.
   Both axes must vary smoothly (no jumps) and match the chosen longitudinal
   and lateral behaviors. Example:
{lateral_example}

{template}
"""


def _build_step_example(vx: float) -> tuple[int, int]:
    """Return (int_part_at_1s, frac_part_at_0.5s) for the prompt example."""
    step_m_05 = vx * 0.5  # forward distance per 0.5 s step
    step_m_10 = vx * 1.0  # forward distance at t=1.0 s
    frac_at_05 = int(round((step_m_05 - int(step_m_05)) * 10)) % 10
    int_at_10 = int(step_m_10) % 10
    return int_at_10, frac_at_05


def build_prompt_v3(sample: dict) -> str:
    # Waymo `history waypoints` has 16 points at 0.1 s spacing = 1.5 s back
    # to t=0. We use ALL points (no downsampling) so the model sees the
    # full motion at the data's native resolution; this matters for the
    # bidir attention to capture velocity / turn-onset signals.
    raw_hist = sample["history waypoints"]
    hist_str = "; ".join(f"({p[0]:.2f}, {p[1]:.2f})" for p in raw_hist)
    vx, vy = sample["velocity"][-1]
    speed = math.hypot(vx, vy)
    ax, ay = sample["acceleration"][-1]
    accel = ax  # longitudinal accel
    nav = sample["navigation_command"]
    step_int1, step_frac = _build_step_example(vx)
    # Per-sample worked trajectory example with BOTH forward & lateral, in
    # the EXACT format the model must emit. Showing the full per-waypoint
    # line context (not just lateral) gives the model a copyable pattern
    # that survives bidirectional diffusion's local-context bias.
    step = vx * 0.5  # forward step per 0.5 s
    def _fwd(t):
        v = vx * t
        sign = "+" if v >= 0 else "-"
        av = abs(v)
        return f"{sign}{int(av // 10):01d}{int(av % 10):01d}.{int((av * 10) % 10):01d}"
    if nav == "GO_RIGHT":
        # Right turn: lateral negative, magnitude grows.
        lateral_example = "\n".join([
            f"       0.5s: forward={_fwd(0.5)}m, lateral=-00.3m",
            f"       1.0s: forward={_fwd(1.0)}m, lateral=-00.8m",
            f"       1.5s: forward={_fwd(1.5)}m, lateral=-01.3m",
            f"       2.0s: forward={_fwd(2.0)}m, lateral=-01.8m",
            f"       2.5s: forward={_fwd(2.5)}m, lateral=-02.4m",
            "       (continue with NEGATIVE lateral, sign character '-' for all 10 lines)",
        ])
    elif nav == "GO_LEFT":
        lateral_example = "\n".join([
            f"       0.5s: forward={_fwd(0.5)}m, lateral=+00.3m",
            f"       1.0s: forward={_fwd(1.0)}m, lateral=+00.8m",
            f"       1.5s: forward={_fwd(1.5)}m, lateral=+01.3m",
            f"       2.0s: forward={_fwd(2.0)}m, lateral=+01.8m",
            f"       2.5s: forward={_fwd(2.5)}m, lateral=+02.4m",
            "       (continue with POSITIVE lateral, sign character '+' for all 10 lines)",
        ])
    else:  # GO_STRAIGHT
        lateral_example = "\n".join([
            f"       0.5s: forward={_fwd(0.5)}m, lateral=+00.0m",
            f"       1.0s: forward={_fwd(1.0)}m, lateral=+00.0m",
            f"       2.5s: forward={_fwd(2.5)}m, lateral=+00.0m",
            f"       5.0s: forward={_fwd(5.0)}m, lateral=+00.0m",
            "       (lateral stays 00.0 for all 10 lines)",
        ])
    return PROMPT_V3.format(
        speed=speed, accel=accel, instruction=nav, history=hist_str,
        step_m=vx * TRAJECTORY_DT, step_int1=step_int1, step_frac=step_frac,
        lateral_example=lateral_example,
        template=build_template_v3(),
    )


# --- parsing the filled template ---

# Semantic trajectory: each waypoint is
#   `<t>s: forward=<x>m, lateral=<y>m`
# (newlines or not, but `forward=...m` is the reliable anchor.)
_FORWARD_RE = re.compile(r"forward\s*=\s*([+\-]?\d+(?:\.\d+)?)\s*m", re.I)
_LATERAL_RE = re.compile(r"lateral\s*=\s*([+\-]?\d+(?:\.\d+)?)\s*m", re.I)
# Fallback for the OLD compact `+XX.X,+YY.Y;...` format (in case any caller
# still uses it).
_PAIR_RE = re.compile(
    r"([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*[,\s]\s*"
    r"([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
)


def _extract_trajectory_field(filled_text: str) -> str:
    """Pull the 'trajectory' string out of the filled JSON (best-effort)."""
    # Try strict JSON first
    try:
        # Trim to JSON-looking region: between first { and last }
        i = filled_text.find("{")
        j = filled_text.rfind("}")
        if i >= 0 and j > i:
            blob = filled_text[i:j+1]
            data = json.loads(blob)
            tr = data.get("trajectory", "")
            return tr if isinstance(tr, str) else ""
    except Exception:
        pass
    # Fallback: regex-extract the field
    m = re.search(r'"trajectory"\s*:\s*"([^"]*)"', filled_text, re.DOTALL)
    if m:
        return m.group(1)
    # Last fallback: anything after 'trajectory'
    m = re.search(r"trajectory[:\s\"]+(.*)", filled_text, re.DOTALL)
    return m.group(1) if m else ""


def parse_filled(filled_text: str) -> List[Tuple[float, float]]:
    """Extract up to N_WAYPOINTS (x, y) pairs from the filled trajectory field.
    Tries the semantic `forward=...m, lateral=...m` layout first, then falls
    back to the legacy compact `+XX.X,+YY.Y;...` form.
    """
    tr = _extract_trajectory_field(filled_text)
    if not tr:
        tr = filled_text  # last-ditch: scan the whole output
    # Strip mask tokens and pad placeholders
    tr = tr.replace(MASK, " ").replace("<|mdm_mask|>", " ").replace("|MASK|", " ")
    tr = tr.replace("pad", " ")
    # Preferred: semantic format
    forwards = [float(x) for x in _FORWARD_RE.findall(tr)]
    laterals = [float(y) for y in _LATERAL_RE.findall(tr)]
    if forwards and laterals:
        n = min(len(forwards), len(laterals), N_WAYPOINTS)
        return list(zip(forwards[:n], laterals[:n]))
    # Fallback: legacy compact format
    pairs = _PAIR_RE.findall(tr)
    return [(float(x), float(y)) for x, y in pairs][:N_WAYPOINTS]
