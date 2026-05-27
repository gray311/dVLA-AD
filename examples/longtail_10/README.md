# 10 Longtail Waymo Examples (joint view)

Hand-picked interesting / longtail samples from the Waymo CoT validation set,
scored by lateral motion magnitude, acceleration events, and keyword presence
(construction, cone, emergency, hazard, cyclist, ped, intersection, etc.).

Each sample folder contains:
- `cam_joint.jpg`  — stitched front-left + front + front-right view (2818×1079)
- `prompt.txt`     — full V3 prompt sent to the model
- `output.json`    — model output text + parsed waypoints + ADE
- `meta.json`      — sample metadata (idx, nav, speed, accel, GT trajectory, latency)

## Picked samples

| dir | idx | nav | speed | accel | GT max\|lat\| | ADE | latency |
|---|---:|---|---:|---:|---:|---:|---:|
| 059_…_GO_RIGHT | 59 | GO_RIGHT | 4.6 | +0.30 | 8.4m (large turn) | 3.80m | 3.32s |
| 066_…_GO_STRAIGHT | 66 | GO_STRAIGHT | 7.6 | +0.72 (strong accel) | 3.8m | 14.45m | 3.29s |
| 086_…_GO_STRAIGHT | 86 | GO_STRAIGHT | 4.0 | +0.24 | 4.2m (drift) | 5.30m | 3.31s |
| 107_…_GO_RIGHT | 107 | GO_RIGHT | 3.5 | +0.20 | 6.0m | 2.73m | 3.29s |
| 142_…_GO_RIGHT | 142 | GO_RIGHT | 3.2 | +0.15 | 5.9m | 2.07m | 3.30s |
| 143_…_GO_LEFT | 143 | GO_LEFT | 8.4 | +0.01 | 3.2m | 6.12m | 3.29s |
| 202_…_GO_LEFT | 202 | GO_LEFT | 5.7 | -0.18 | **9.4m (largest lat turn)** | 4.60m | 2.19s |
| 244_…_GO_RIGHT | 244 | GO_RIGHT | 5.3 | +0.43 | 8.3m | 4.17m | 3.29s |
| 327_…_GO_STRAIGHT | 327 | GO_STRAIGHT | **21.9 (highway)** | -0.02 | 4.6m | 15.66m | 3.30s |
| 374_…_GO_STRAIGHT | 374 | GO_STRAIGHT | **25.2 (highest)** | +0.00 | 2.1m | 8.89m | 3.30s |

## Config

- Model: `Fast_dVLM_3B` (NVlabs) — zero-shot, no driving finetune
- Engine: SGLang fork at `third_party/sglang` with V3 template-fill (`HierarchyBlock`,
  4 diffusion steps per chunk)
- Image: **CAM_JOINT** (stitched 3-cam panoramic), 2818×1079
- Prompt: V3 with 3 s history at 0.5 s spacing + per-sample worked
  trajectory example
- Template fields: critical_objects (12 cat × 2 mask) + complexity + explanation
  (100 mask) + future_meta_behavior (long+lat) + semantic trajectory
  (10wp × `forward=±XX.Xm, lateral=±YY.Ym`)

## Notes

- Joint image is ~4× wider than CAM_FRONT alone → vision tower processes more
  tokens → per-sample latency ~3.3 s (vs ~2 s with front-only).
- All 10 samples emit `"complexity": "simple"` — these are interesting
  geometric edge cases (large turns / high speed) but not hazard scenes; the
  complexity tag fires only on accidents/fires/multi-agent danger.
- Lateral sign: model always emits `+` even for GO_RIGHT (token-prior
  limitation, see `5ba15be` commit message). The magnitude direction is
  usually right; just the sign is wrong on right turns.
