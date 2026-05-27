# 10 Longtail Waymo Examples + Burning-Car test_image

Hand-picked interesting / longtail samples from Waymo CoT val + one
out-of-distribution scene (`test_image_burning_car/`).

Each sample folder contains both **SGLang Fast-dVLM (zero-shot)** and
**dVLM-AD (finetuned)** inputs/outputs:

| file | source | description |
|---|---|---|
| `cam_joint.jpg` | SGLang | stitched 3-cam panorama (2818×1079) |
| `cam_front_left.jpg` / `cam_front.jpg` / `cam_front_right.jpg` | dVLM-AD | individual 3-cam inputs |
| `prompt.txt` | SGLang | full V3 prompt |
| `output.json` | SGLang | model output text + parsed waypoints |
| `dvlm_ad_prompt.txt` | dVLM-AD | data-file `conversations[0]` verbatim |
| `dvlm_ad_template.txt` | dVLM-AD | data-file `conversations[1]` (scaffold with mask markers) |
| `dvlm_ad_output.json` | dVLM-AD | 64-step diffusion fill output |
| `meta.json` | both | sample id, nav, speed, accel, GT trajectory, latency |

## Picked samples

| dir | idx | nav | speed | accel | GT max\|lat\| | notes |
|---|---:|---|---:|---:|---:|---|
| 059_…_GO_RIGHT | 59 | GO_RIGHT | 4.6 | +0.30 | 8.4m | large right turn |
| 066_…_GO_STRAIGHT | 66 | GO_STRAIGHT | 7.6 | **+0.72** | 3.8m | strong acceleration |
| 086_…_GO_STRAIGHT | 86 | GO_STRAIGHT | 4.0 | +0.24 | 4.2m | lateral drift |
| 107_…_GO_RIGHT | 107 | GO_RIGHT | 3.5 | +0.20 | 6.0m | |
| 142_…_GO_RIGHT | 142 | GO_RIGHT | 3.2 | +0.15 | 5.9m | |
| 143_…_GO_LEFT | 143 | GO_LEFT | 8.4 | +0.01 | 3.2m | mid-speed left |
| 202_…_GO_LEFT | 202 | GO_LEFT | 5.7 | -0.18 | **9.4m** | largest lateral turn |
| 244_…_GO_RIGHT | 244 | GO_RIGHT | 5.3 | +0.43 | 8.3m | accelerating right turn |
| 327_…_GO_STRAIGHT | 327 | GO_STRAIGHT | **21.9** | -0.02 | 4.6m | highway cruise + drift |
| 374_…_GO_STRAIGHT | 374 | GO_STRAIGHT | **25.2** | +0.00 | 2.1m | highest speed |
| **test_image_burning_car** | — | GO_STRAIGHT (mock) | 15 (mock) | -0.5 (mock) | — | OOD hazard scene (burning car + orange cones on highway) |

## Latency / cost notes

- SGLang uses CAM_JOINT (one stitched image, 2818×1079): ~3.3s/sample with
  the joint view because the vision tower processes ~4x more tokens than
  CAM_FRONT alone (which gives ~2.0s).
- dVLM-AD uses three individual cams (front-left, front, front-right):
  ~33s/sample (HuggingFace transformers path, 64 diffusion steps).

## Quick-start: load one sample

```python
import json, os
sample_dir = "examples/longtail_10/202_90f0dd7e6049d2d4_GO_LEFT"
meta = json.load(open(f"{sample_dir}/meta.json"))
sglang_out = json.load(open(f"{sample_dir}/output.json"))
dvlm_ad_out = json.load(open(f"{sample_dir}/dvlm_ad_output.json"))

print("GT trajectory:", meta["future_waypoints_10"])
print("\nSGLang output:")
print(sglang_out["model_output_text"])
print("\ndVLM-AD output:")
print(dvlm_ad_out["clean_output_text"])
```
