"""Run Fast-dDrive (NVlabs, xiwenyoumu/Fast-dDrive) on our 3 longtail samples
and compare explanation quality against our SGLang V3 outputs.

Fast-dDrive is Qwen2.5-VL-3B finetuned on Waymo CoT, with built-in section
diffusion (`mdm_sample_deep_scaffold`). Output uses their schema
(critical_objects with yes/no, explanation, future_meta_behavior, trajectory).

Caches model under $HF_HOME (set to scratch).
"""
import json, math, os, sys, time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
PICKED_IDX = [202, 244, 59]

MODEL_ID = "xiwenyoumu/Fast-dDrive"


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def _joint_image(sample):
    front = _fix(sample["image"][1])
    return front.replace("CAM_FRONT.jpg", "CAM_JOINT.jpg")


def main():
    print(f"Loading Fast-dDrive from {MODEL_ID} (downloads to $HF_HOME if missing)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    processor.tokenizer = tokenizer

    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])
    print(f"mask_id={mask_id}", flush=True)

    data = json.load(open(DATA))

    # Use their canonical prompt asking for the deep-scaffold output.
    prompt_template = (
        "You are an expert autonomous driving agent.\n"
        "Analyze the driving scene and produce:\n"
        "  - critical_objects (12 categories yes/no)\n"
        "  - explanation (natural language reasoning)\n"
        "  - future_meta_behavior (longitudinal / lateral verbs)\n"
        "  - trajectory (5 waypoints over 5 seconds)\n\n"
        "Ego state: speed={speed:.1f} m/s, acceleration={accel:.2f} m/s^2\n"
        "Driver instruction: {nav}\n"
    )

    results = {}
    for idx in PICKED_IDX:
        s = data[idx]
        img_path = _joint_image(s)
        vx, vy = s["velocity"][-1]
        speed = math.hypot(vx, vy)
        ax, _ = s["acceleration"][-1]
        nav = s["navigation_command"]
        prompt = prompt_template.format(speed=speed, accel=ax, nav=nav)

        print(f"\n=== idx={idx} nav={nav} v={speed:.1f} ===", flush=True)
        if not os.path.exists(img_path):
            print(f"  ! image missing: {img_path}")
            continue
        image = Image.open(img_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")

        kwargs = dict(
            input_ids=inputs.input_ids,
            tokenizer=tokenizer,
            block_size=32,
            max_tokens=512,
            mask_id=mask_id,
            threshold=0.9,  # paper default for section_diffusion
        )
        if getattr(inputs, "pixel_values", None) is not None:
            kwargs["pixel_values"] = inputs.pixel_values
        if getattr(inputs, "image_grid_thw", None) is not None:
            kwargs["image_grid_thw"] = inputs.image_grid_thw

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.mdm_sample_deep_scaffold(**kwargs)
        torch.cuda.synchronize()
        latency = time.time() - t0

        trimmed = out[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(trimmed, skip_special_tokens=True)
        print(f"  latency={latency:.2f}s, response_len={len(response)} chars", flush=True)
        print(f"  --- Response ---")
        print(f"  {response[:1200]}")

        results[idx] = {"latency": latency, "response": response, "nav": nav, "speed": speed}

    out_json = "/tmp/fast_ddrive_compare.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
