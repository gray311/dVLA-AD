"""Run SGLang Fast-dVLM on 10 interesting/longtail Waymo samples using the
JOINT camera view (stitched front-left + front + front-right). Save each
sample's image, prompt, template, model output, and GT to examples/longtail_10/.
"""
import json
import math
import os
import shutil
import sys

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, parse_filled

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
OUT_DIR = os.path.join(ROOT, "examples", "longtail_10")

# Chosen by interestingness score: large lateral motion, hazards, varied speed
# and nav. Mix of turns (LEFT/RIGHT), highway cruise, multi-agent scenes.
PICKED = [202, 244, 59, 327, 107, 142, 66, 86, 374, 143]


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def _joint_image(sample):
    """Convert the CAM_FRONT path to CAM_JOINT (stitched 3-cam view)."""
    front = _fix(sample["image"][1])  # e.g., .../149_CAM_FRONT.jpg
    return front.replace("CAM_FRONT.jpg", "CAM_JOINT.jpg")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = json.load(open(DATA))
    picked = [(i, data[i]) for i in PICKED]

    print(f"Saving {len(picked)} longtail samples to {OUT_DIR}")
    for idx, s in picked:
        vx, vy = s["velocity"][-1]
        sp = math.hypot(vx, vy)
        ax, _ = s["acceleration"][-1]
        my = max(abs(p[1]) for p in s["future waypoints"][:10])
        print(f"  idx={idx}  nav={s['navigation_command']:14}  v={sp:5.1f}  "
              f"ax={ax:+.2f}  max|y|={my:.2f}  id={s['sample_id'][:16]}")

    print("\nLoading SGLang Fast-dVLM...")
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    bundle = loader.load(algorithm="mdm")
    print("Warmup...")
    loader.generate(bundle, [_joint_image(picked[0][1])],
                     build_prompt_v3(picked[0][1]), temperature=0.0,
                     nav_command=picked[0][1]["navigation_command"])

    for idx, s in picked:
        vx, vy = s["velocity"][-1]
        sp = math.hypot(vx, vy)
        ax, _ = s["acceleration"][-1]
        sid = s["sample_id"][:16]
        sample_dir = os.path.join(OUT_DIR, f"{idx:03d}_{sid}_{s['navigation_command']}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n=== idx={idx} nav={s['navigation_command']} v={sp:.1f} ===")

        # Copy JOINT image
        joint_path = _joint_image(s)
        if os.path.exists(joint_path):
            shutil.copy(joint_path, os.path.join(sample_dir, "cam_joint.jpg"))
        else:
            print(f"  ! joint image missing: {joint_path}")

        prompt = build_prompt_v3(s)
        with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        # Inference (using joint image)
        text, latency = loader.generate(
            bundle, [joint_path], prompt, temperature=0.0,
            nav_command=s["navigation_command"],
        )
        pred = parse_filled(text)
        gt = s["future waypoints"][:10]
        # ADE on first 5 waypoints
        if pred and gt:
            n = min(len(pred), len(gt), 5)
            ade = sum(math.hypot(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1])
                       for i in range(n)) / n
        else:
            ade = None

        meta = {
            "sample_idx": idx,
            "sample_id": s["sample_id"],
            "navigation_command": s["navigation_command"],
            "speed_m_s": sp,
            "longitudinal_accel_m_s2": ax,
            "image_joint": joint_path,
            "image_joint_local": "cam_joint.jpg",
            "future_waypoints_10": gt,
            "ade_first5_m": ade,
            "latency_s": latency,
        }
        with open(os.path.join(sample_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(sample_dir, "output.json"), "w") as f:
            json.dump({
                "model_output_text": text,
                "predicted_waypoints_10": pred[:10],
                "ade_first5_m": ade,
            }, f, indent=2)
        print(f"  done lat={latency:.2f}s ADE={ade:.2f}m → {sample_dir}")

    loader.shutdown(bundle)

    # Top-level summary
    summary = {
        "n_samples": len(picked),
        "picked_indices": [i for i, _ in picked],
        "config": {
            "model": "Fast_dVLM_3B (zero-shot)",
            "image": "CAM_JOINT (stitched 3-cam)",
            "template": "V3 (critical_objects + complexity + explanation + "
                         "behavior + semantic trajectory)",
            "algorithm": "SGLang HierarchyBlock, section-aligned bs=160, 4 steps/chunk",
        },
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
