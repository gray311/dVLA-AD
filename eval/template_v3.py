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
N_EXPLANATION_TOKENS = 50

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


def build_template_v3() -> str:
    """JSON scaffold with <|mdm_mask|> at every slot the model should fill
    (string-rendered version, for the human-readable TEMPLATE block at the tail
    of the prompt). The token-id-level builder is `build_template_ids_v3` below
    and is the source of truth used by loaders.
    """
    co = {cat: MASK * N_CRITICAL_TOKENS for cat in CRITICAL_CATEGORIES}
    # Behavior: 2 mask tokens (verb only) with fixed scaffold space between them.
    behavior_str = f"{MASK} {MASK}"
    obj = {
        "critical_objects": co,
        "explanation": MASK * N_EXPLANATION_TOKENS,
        "future_meta_behavior": {
            "longitudinal": behavior_str,
            "lateral": behavior_str,
        },
        "trajectory": MASK * N_TRAJECTORY_TOKENS,
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
    ids += enc('}, "explanation": "')
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
    # Structured trajectory: 20 waypoints, each as `<sign><tens><ones>.<frac>,<sign><tens><ones>.<frac>`
    # separated by `;`. Mask slot kinds carry constraint info:
    #   traj_sign      → {+, -}
    #   traj_tens      → digit 0-9
    #   traj_ones      → digit 0-9
    #   traj_frac      → digit 0-9
    for w in range(N_WAYPOINTS):
        if w > 0:
            ids += enc(';')
        # x = <sign><tens><ones>.<frac>
        slots.append((len(ids), "traj_sign"));  ids.append(mask_id)
        slots.append((len(ids), "traj_tens"));  ids.append(mask_id)
        slots.append((len(ids), "traj_ones"));  ids.append(mask_id)
        ids += enc('.')
        slots.append((len(ids), "traj_frac"));  ids.append(mask_id)
        ids += enc(',')
        # y = <sign><tens><ones>.<frac>
        slots.append((len(ids), "traj_sign"));  ids.append(mask_id)
        slots.append((len(ids), "traj_tens"));  ids.append(mask_id)
        slots.append((len(ids), "traj_ones"));  ids.append(mask_id)
        ids += enc('.')
        slots.append((len(ids), "traj_frac"));  ids.append(mask_id)
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

2. explanation: ~50 tokens of natural-language reasoning in EXACTLY 3 stages.
   Do not just list the critical_objects values — instead reason in this order:

   (a) SCENE DESCRIPTION — one sentence on the overall scene context
       (road type, lane layout, ego situation, weather, time of day).
       Example: "Two-lane city road at an intersection in overcast weather."

   (b) CRUCIAL-OBJECT BEHAVIOR PREDICTION — for each critical object that is
       NOT "none", predict what that object is about to do in the next 3 s.
       Example: "The red sedan ahead is slowing for the red light;
                 the pedestrian on the right curb intends to cross."

   (c) EGO–OBJECT INTERACTION — explain HOW the predicted object behaviors
       constrain the ego's own next action. Tie this to the meta-behavior
       (longitudinal/lateral) you will emit.
       Example: "Because the sedan is decelerating in our lane, ego must
                 slow down and keep the lane; we therefore output
                 'slow down' + 'keep lane'."

   Keep it ~50 tokens total — be terse. No filler sentences like "the road
   is clear" if it isn't load-bearing. Never repeat critical_objects values
   verbatim — rephrase as agent behaviors.

3. future_meta_behavior.longitudinal: format is exactly "verb_w1 verb_w2" — a
   2-word phrase (one mask token per word), separated by a single space.
   - verb (2 words, pick ONE phrase) in {{"speed up", "slow down", "keep speed", "stop now"}}
   - Pick based on current speed AND scene context:
     * current speed < 1 m/s and path clear → "speed up"
     * cruising and no hazard ahead          → "keep speed"
     * hazard / red light / vehicle slowing ahead → "slow down"
     * imminent collision / fire / pedestrian in path → "stop now"

4. future_meta_behavior.lateral: same 2-word format.
   - verb (2 words, pick ONE phrase) in {{"keep lane", "turn left", "turn right", "change left", "change right"}}
   - The Driver instruction is a HIGH-LEVEL hint, not a hard rule. The
     lateral verb should usually follow it (`GO_LEFT`→`turn left`,
     `GO_RIGHT`→`turn right`, `GO_STRAIGHT`→`keep lane`) — BUT if the
     current frame shows a hazard (pedestrian in turning path, oncoming
     vehicle, blocked lane, imminent collision, etc.) that makes
     following the nav unsafe within the next 5 s, pick the lateral
     verb that the EGO ACTUALLY NEEDS to execute instead. Local scene
     safety overrides the high-level nav.

5. trajectory: 10 future ego waypoints at 0.5 s spacing (t = 0.5, 1.0, ... 5.0 s),
   in meters in the ego frame (x = forward, y = left). Each coordinate has a
   sign (`+`/`-`), two integer digits, and ONE decimal place — e.g.
   "+05.0,+00.0;+10.0,+00.0;...+50.0,+00.0". 10 waypoints separated by `;`.
   At the current speed of {speed:.1f} m/s going straight, x grows by about
   {step_m:.2f} m per step.

TEMPLATE (fill the <|mdm_mask|> positions only — keep all other characters
verbatim):

{template}
"""


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
    return PROMPT_V3.format(
        speed=speed, accel=accel, instruction=nav, history=hist_str,
        step_m=vx * TRAJECTORY_DT, template=build_template_v3(),
    )


# --- parsing the filled template ---

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
    """Extract up to 20 (x, y) pairs from the filled trajectory field."""
    tr = _extract_trajectory_field(filled_text)
    if not tr:
        tr = filled_text  # last-ditch: scan the whole output
    # Strip mask tokens and pad placeholders
    tr = tr.replace(MASK, " ").replace("<|mdm_mask|>", " ").replace("|MASK|", " ")
    tr = tr.replace("pad", " ")
    pairs = _PAIR_RE.findall(tr)
    return [(float(x), float(y)) for x, y in pairs][:20]
