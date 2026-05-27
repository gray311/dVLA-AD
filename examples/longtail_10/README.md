# 10 Longtail Waymo Examples + Burning-Car test_image

Hand-picked interesting / longtail samples from Waymo CoT val + one
out-of-distribution hazard scene (`test_image_burning_car/`).

Each sample folder contains both **SGLang Fast-dVLM (zero-shot)** and
**dVLM-AD (finetuned on Waymo CoT)** inputs/outputs side by side.

## Per-sample file layout

| file | model | content |
|---|---|---|
| `cam_joint.jpg` | SGLang | stitched 3-cam panorama (2818×1079) |
| `cam_front_left.jpg` / `cam_front.jpg` / `cam_front_right.jpg` | dVLM-AD | individual 3-cam inputs (training format) |
| `prompt.txt` | SGLang | full V3 prompt (12-cat critical_objects + complexity + explanation + behavior + semantic trajectory) |
| `output.json` | SGLang | model output text + parsed 10 waypoints |
| `dvlm_ad_prompt.txt` | dVLM-AD | data-file `conversations[0]` verbatim (Waymo Task 1-4 prompt format) |
| `dvlm_ad_template.txt` | dVLM-AD | data-file `conversations[1]` (222-mask scaffold: 12×1 yes/no + ~100 explanation + ~10 behavior + ~100 trajectory) |
| `dvlm_ad_output.json` | dVLM-AD | 64-step diffusion fill output (raw + cleaned text) |
| `meta.json` | both | idx, nav, speed, accel, GT trajectory, latencies |

For the Waymo samples, `dvlm_ad_prompt.txt` and `dvlm_ad_template.txt`
are copied **verbatim** from the data file's `conversations[0/1]` fields
(i.e., exactly what dVLM-AD was finetuned on).

For `test_image_burning_car/` (OOD, not from Waymo), we borrow the
Waymo template/prompt schema verbatim and substitute in a mock 7-point
history at 0.5 s spacing — so dVLM-AD still receives input in its
training distribution.

## Picked samples

| dir | idx | nav | speed (m/s) | accel | GT max\|lat\| | notes |
|---|---:|---|---:|---:|---:|---|
| 059_…_GO_RIGHT | 59 | GO_RIGHT | 4.6 | +0.30 | 8.4 m | large right turn |
| 066_…_GO_STRAIGHT | 66 | GO_STRAIGHT | 7.6 | **+0.72** | 3.8 m | strong acceleration |
| 086_…_GO_STRAIGHT | 86 | GO_STRAIGHT | 4.0 | +0.24 | 4.2 m | lateral drift |
| 107_…_GO_RIGHT | 107 | GO_RIGHT | 3.5 | +0.20 | 6.0 m | |
| 142_…_GO_RIGHT | 142 | GO_RIGHT | 3.2 | +0.15 | 5.9 m | |
| 143_…_GO_LEFT | 143 | GO_LEFT | 8.4 | +0.01 | 3.2 m | mid-speed left |
| 202_…_GO_LEFT | 202 | GO_LEFT | 5.7 | -0.18 | **9.4 m** | largest lateral turn |
| 244_…_GO_RIGHT | 244 | GO_RIGHT | 5.3 | +0.43 | 8.3 m | accelerating right turn |
| 327_…_GO_STRAIGHT | 327 | GO_STRAIGHT | **21.9** | -0.02 | 4.6 m | highway cruise + drift |
| 374_…_GO_STRAIGHT | 374 | GO_STRAIGHT | **25.2** | +0.00 | 2.1 m | highest speed |
| **test_image_burning_car** | — | GO_STRAIGHT (mock) | 15 (mock) | -0.5 (mock) | — | **OOD hazard**: burning car + orange cones on highway |

## Schema difference (SGLang vs dVLM-AD)

The two models use **different output schemas** — both stored in each folder:

| field | SGLang Fast-dVLM (V3, zero-shot) | dVLM-AD (Waymo CoT, finetuned) |
|---|---|---|
| critical_objects | 12 categories × 2-token open-vocab phrases (e.g. `"black car"`, `"orange cone"`) | 12 categories × `"yes"`/`"no"` only |
| complexity | 1-mask `{simple, complex}` tag | — (not in dVLM-AD schema) |
| explanation | ~100 tokens of natural reasoning | ~100 tokens of natural reasoning |
| behavior | 2-word verbs from V3 vocab: `{speed up, slow down, keep speed, stop now}` × `{turn left, turn right, keep lane, change left/right}` | dVLM-AD vocab: `{keep, accelerate, decelerate, stop, other}` × `{straight, left_turn, right_turn, lane_follow, lane_change_*, yield, reverse, overtake, other}` |
| trajectory | 10 waypoints @ 0.5 s, format `<t>s: forward=±XX.Xm, lateral=±YY.Ym` | 5-7 waypoints @ 1 s, format `[[x,y], ...]` |

SGLang's schema is richer (open-vocab perception, complexity tag, finer
1 s grid). dVLM-AD's schema is what its finetune was trained on.

## Latency

- SGLang uses CAM_JOINT (stitched 3-cam, 2818×1079): **~3.3 s/sample**.
  The vision tower processes ~4× more tokens than CAM_FRONT alone
  (which would give ~2.0 s).
- dVLM-AD uses 3 individual cams: **~33 s/sample** (HuggingFace
  transformers path, 64 diffusion steps over 222 masks).
- Test image: dVLM-AD ~21 s (same template, single image fed to all 3
  cam slots).

→ SGLang is **~10× faster** at comparable behavior accuracy.

## test_image_burning_car — OOD hazard highlight

The burning-car / orange-cones highway image is the only non-Waymo sample.
Both models still produce coherent output on this OOD scene:

**SGLang** (zero-shot, V3 schema, ~1.3 s):
```
critical_objects:
  nearby_vehicle:      "none car"          (burnt vehicle treated as non-active)
  road_hazard:         "fire smoke"        ✓ explicit hazard detection
  traffic_element:     "orange cone"       ✓
  weather_condition:   "over sky"
  others:              mostly "none"
complexity:    "complex"                   ✓
behavior:      "slow down" / "keep lane"   ✓
trajectory:    10 waypoints @ 0.5 s
```

**dVLM-AD** (finetuned, 222-mask Waymo schema, ~21 s):
```
critical_objects:
  nearby_vehicle:      "yes"               ← burning car as nearby vehicle
  traffic_element:     "yes"               ← cones / signs
  others:              "no"
explanation: "There is a nearby vehicle directly ahead in the same lane,
              visible in the front camera, which will influence the ego
              vehicle's speed... Additionally, a traffic sign is present
              on the right side..."
behavior:      "slow down" / "lane follow" ✓
trajectory:    [+14.5,+27.5,+39.0,+49.5,+59.5,+69.5,+79.5] @ 1 s spacing
               (decel from 14.5 → ~10 m/s — correct cascade from slow-down decision)
```

Both models choose **slow down** — the critical safety behavior.
SGLang produces richer perception detail (`"fire smoke"` road_hazard,
`"orange cone"` traffic_element), dVLM-AD produces cleaner prose but
abstracts the burning car as just a "nearby vehicle".

## Quick-start: load one sample

```python
import json
sample_dir = "examples/longtail_10/202_90f0dd7e6049d2d4_GO_LEFT"

meta       = json.load(open(f"{sample_dir}/meta.json"))
sglang     = json.load(open(f"{sample_dir}/output.json"))
dvlm_ad    = json.load(open(f"{sample_dir}/dvlm_ad_output.json"))

print("GT trajectory (10 wp):", meta["future_waypoints_10"])
print("\nSGLang output:");  print(sglang["model_output_text"])
print("\ndVLM-AD output:"); print(dvlm_ad["clean_output_text"])
```

## Reproduce

```bash
# 10 longtail Waymo samples
python scripts/save_longtail_examples.py     # SGLang side
python scripts/save_longtail_dvlm_ad.py      # dVLM-AD side

# Burning-car OOD sample
python scripts/save_test_image_to_longtail.py both
```
