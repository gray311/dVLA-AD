"""PROMPT_V3 + TEMPLATE_V3 builders and trajectory parser.

For the trajectory task we adapt the template's trajectory horizon to our
Waymo-50 setup: 20 waypoints @ 0.25 s spacing (5 s horizon), not the
template's default 8 @ 0.5 s.

The template literal `<|mdm_mask|>` is the mask token of LaViDa / DiffusionVL /
Fast-dVLM. After fill, we extract:
  - critical_objects (12 entries × 10 mask slots)
  - explanation (100 mask slots)
  - future_meta_behavior.{longitudinal, lateral} (5 + 5)
  - trajectory (100 mask slots) — contains the actual "x.x,y.y;..." waypoints
"""
from __future__ import annotations
import math
import re
import json
from typing import List, Tuple

MASK = "<|mdm_mask|>"

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
    """JSON scaffold with <|mdm_mask|> at every slot the model should fill
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


PROMPT_V3 = """You are an autonomous driving assistant. Given the current driving scene, identify critical objects, explain your reasoning, then predict the future driving behavior and trajectory.

INPUT:
- Multi-view images: front-left, front, front-right
- Ego state: speed={speed:.1f} m/s, longitudinal acceleration={accel:.2f} m/s^2
- Driver instruction: {instruction}
- Past 3.0 s of ego positions at 0.5 s spacing (x forward, y left, meters):
  {history}

TASK:
Fill in the masked positions in the following JSON template.

OUTPUT FORMAT REQUIREMENTS:

1. critical_objects: For each of the 12 categories, fill EXACTLY 2 tokens:
   - If the category does NOT exist in the scene: fill "none" (or "none" + 1 pad).
   - If it DOES exist: fill a SHORT referring expression (max 2 tokens, e.g.
     "black car", "green light", "orange cone", "overcast sky"). Keep it tight —
     do NOT introduce new JSON keys inside the slot.

2. complexity: ONE token — exactly one of {{"simple", "complex"}}.
   - "simple": low-traffic, no hazards, predictable surroundings (empty
     residential street, clear highway, single lead car in calm flow).
   - "complex": ANY of — multiple interacting agents (≥3 nearby vehicles
     or pedestrians/cyclists in path), an unfolding hazard (accident
     ahead, fire / smoke, blocked lane, debris, oncoming emergency
     vehicle), construction zones with lane shifts, complex
     intersections with mixed traffic, or anything that should signal
     "give this frame extra planning attention".

3. explanation: ~100 tokens describing the scene, salient objects, and how
   they shape your planned action. Write naturally — no fixed template, no
   numbered headings. A useful explanation usually grounds in the visible
   scene (road type, weather, what other agents are doing) and ties at least
   one observed agent or hazard to the longitudinal / lateral choice you
   emit below. Avoid generic filler.

5. future_meta_behavior.longitudinal: format is exactly "verb_w1 verb_w2" — a
   2-word phrase (one mask token per word), separated by a single space.
   - verb (2 words, pick ONE phrase) in {{"speed up", "slow down", "keep speed", "stop now"}}
   - Pick based on current speed AND scene context:
     * current speed < 1 m/s and path clear → "speed up"
     * cruising and no hazard ahead          → "keep speed"
     * hazard / red light / vehicle slowing ahead → "slow down"
     * imminent collision / fire / pedestrian in path → "stop now"

6. future_meta_behavior.lateral: same 2-word format.
   - verb (2 words, pick ONE phrase) in {{"keep lane", "turn left", "turn right", "change left", "change right"}}
   - The Driver instruction is a HIGH-LEVEL hint, not a hard rule. The
     lateral verb should usually follow it (`GO_LEFT`→`turn left`,
     `GO_RIGHT`→`turn right`, `GO_STRAIGHT`→`keep lane`) — BUT if the
     current frame shows a hazard (pedestrian in turning path, oncoming
     vehicle, blocked lane, imminent collision, etc.) that makes
     following the nav unsafe within the next 5 s, pick the lateral
     verb that the EGO ACTUALLY NEEDS to execute instead. Local scene
     safety overrides the high-level nav.

7. trajectory: 10 future ego waypoints at 0.5 s spacing (t = 0.5, 1.0, ...
   5.0 s) — one line per waypoint in the form
       `<t>s: forward=<sign><tens><ones>.<frac>m, lateral=<sign><tens><ones>.<frac>m`
   • `forward` is the ego-frame +x distance (meters, +forward / -reverse)
   • `lateral` is the ego-frame +y offset (meters, +left / -right)
   • each coordinate has a sign (`+`/`-`), two integer digits, ONE decimal
   Example output (current speed {speed:.1f} m/s, going straight, no turn):
       0.5s: forward=+00.{step_frac:01d}m, lateral=+00.0m
       1.0s: forward=+0{step_int1:01d}.{step_frac:01d}m, lateral=+00.0m
       ...
   At {speed:.1f} m/s the forward distance grows by ~{step_m:.2f} m per
   0.5 s step. Lateral offset stays near 0 when going straight; grows
   negative on a right turn, positive on a left turn.

TEMPLATE (fill the <|mdm_mask|> positions only — keep all other characters
verbatim):

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
    # Waymo stores history at 0.1 s spacing across 1.5 s (16 points,
    # t=-1.5..0). The V3 spec wants 3 s of history at 0.5 s spacing
    # (7 points, t=-3..0). When only 16 points (1.5 s) are available we
    # downsample to every-0.5 s (every 5th point) → 4 points covering 1.5 s.
    # When the sample supplies a longer history (e.g. 30+ points covering
    # 3 s), we still take every 5th tail point and cap at 7.
    raw_hist = sample["history waypoints"]
    n = len(raw_hist)
    # Step = 5 points per 0.5 s (Waymo's 0.1 s spacing). Take the last
    # 7 such samples (most recent 3 s) including t=0.
    step = 5
    end = n - 1
    idxs = list(range(end, -1, -step))[:7][::-1]
    hist = [raw_hist[i] for i in idxs]
    hist_str = "; ".join(f"({p[0]:.2f}, {p[1]:.2f})" for p in hist)
    vx, vy = sample["velocity"][-1]
    speed = math.hypot(vx, vy)
    ax, ay = sample["acceleration"][-1]
    accel = ax  # longitudinal accel
    nav = sample["navigation_command"]
    step_int1, step_frac = _build_step_example(vx)
    return PROMPT_V3.format(
        speed=speed, accel=accel, instruction=nav, history=hist_str,
        step_m=vx * TRAJECTORY_DT, step_int1=step_int1, step_frac=step_frac,
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
