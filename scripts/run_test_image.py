"""Run test_image.png through SGLang Fast-dVLM (free-form + V3 template)
and dump everything to examples/test_image_run/ as a self-contained example.

Uses mock ego state appropriate for the scene (highway, slowing down).
"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

IMG = os.path.join(ROOT, "examples/test_image.png")
OUTDIR = os.path.join(ROOT, "examples/test_image_run")

# Mock ego state — highway approaching a smoke/cone hazard ahead.
# History = 3 s of cruise at 15 m/s, sampled every 0.5 s → 7 points
# (t = -3.0, -2.5, ..., 0.0). build_prompt_v3 downsamples to every 5th
# point, so we provide 31 points at 0.1 s spacing covering the same 3 s.
_n_hist_pts = 31  # 3.0 s at 0.1 s steps + endpoint
_v_mock = 15.0
_dt_hist = 0.1
MOCK_SAMPLE = {
    "sample_id": "test_image_highway_smoke",
    "navigation_command": "GO_STRAIGHT",
    "velocity": [(_v_mock, 0.0)],          # 15 m/s = 54 km/h highway cruise
    "acceleration": [(-0.5, 0.0)],          # slight deceleration (driver sees smoke)
    "image": ["", IMG, ""],
    "history waypoints": [
        (-_v_mock * (_n_hist_pts - 1 - i) * _dt_hist, 0.0)
        for i in range(_n_hist_pts)
    ],
    "future waypoints": [],
}


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    from eval.template_v3 import build_prompt_v3
    from eval.loaders import fast_dvlm_sglang_v3 as tpl_loader
    from eval.loaders import fast_dvlm_sglang as free_loader

    prompt = build_prompt_v3(MOCK_SAMPLE)

    # Dump prompt + meta
    with open(f"{OUTDIR}/prompt_full.txt", "w") as f:
        f.write(prompt)
    from eval.loaders.fast_dvlm_sglang_v3 import _strip_template_from_prompt, _inject_nav_into_template
    from eval.template_v3 import build_template_ids_v3
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B",
        trust_remote_code=True,
    )
    mask_id = tok.encode("|<MASK>|")[0]
    with open(f"{OUTDIR}/prompt_to_model.txt", "w") as f:
        f.write(_strip_template_from_prompt(prompt))
    ids, slot_info, _ = build_template_ids_v3(tok, mask_id)
    ids_nav, _ = _inject_nav_into_template(
        ids, slot_info, tok, mask_id, MOCK_SAMPLE["navigation_command"],
    )
    with open(f"{OUTDIR}/template.txt", "w") as f:
        f.write(tok.decode(ids_nav, skip_special_tokens=False))

    meta = {
        "image": IMG,
        "scenario": "Highway, vehicle on fire ahead, orange cones diverting traffic. "
                     "Ego cruising at 15 m/s, lightly decelerating, nav=GO_STRAIGHT.",
        "mock_speed_m_s": 15.0,
        "mock_accel_m_s2": -0.5,
        "mock_nav": "GO_STRAIGHT",
        "note": "Mock ego state (test_image.png is not a Waymo sample).",
    }
    with open(f"{OUTDIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Run SGLang template-fill (mdm)
    print("Loading SGLang for template-fill mdm...")
    bundle = tpl_loader.load(algorithm="mdm")
    # warmup
    print("Warmup...")
    _, lat_warm = tpl_loader.generate(
        bundle, [IMG], prompt, temperature=0.0,
        nav_command=MOCK_SAMPLE["navigation_command"],
    )
    print(f"  warmup {lat_warm:.2f}s")
    print("Run template-fill...")
    text_tpl, lat_tpl = tpl_loader.generate(
        bundle, [IMG], prompt, temperature=0.0,
        nav_command=MOCK_SAMPLE["navigation_command"],
    )
    print(f"  template-fill {lat_tpl:.2f}s, {len(text_tpl)} chars")
    json.dump({
        "path": "Fast_dVLM_3B via modified SGLang fork; template-fill mdm "
                "(vocab gates + JSON blacklist + rep penalty + nav injection)",
        "latency_s": lat_tpl,
        "output_text": text_tpl,
    }, open(f"{OUTDIR}/output_sglang_template.json", "w"), indent=2)
    tpl_loader.shutdown(bundle)

    # Note: free-form path requires a second sgl.Engine in the same process,
    # which collides on Prometheus' global CollectorRegistry. Run that
    # separately (scripts/run_5_waymo_sglang.py spec) if needed.
    print(f"\nSaved everything to {OUTDIR}")


if __name__ == "__main__":
    main()
