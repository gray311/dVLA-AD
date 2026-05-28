"""Run Fast-dVLM-3B (our zero-shot ckpt) using Fast-dDrive's model class +
section-diffusion decoding algorithm.

Isolates the DECODING ALGORITHM from the model weights:
- Same checkpoint as our SGLang setup (xiwenyoumu/Fast_dVLM_3B equivalents)
- But uses Fast-dDrive's mdm_sample_deep_scaffold (deep JSON scaffold +
  section-aware diffusion with confidence threshold).

Weights remap: Fast-dVLM uses `model.language_model.X` / `model.visual.X`;
Fast-dDrive uses `model.X` / `visual.X`. We strip the inner prefix.

Outputs Fast-dDrive's schema (not our V3). Saves to /tmp/dvlm_with_ddrive.json.
"""
import glob, json, math, os, sys, time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from safetensors.torch import load_file

DVLM_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"
DDRIVE_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/transformers/models--xiwenyoumu--Fast-dDrive/snapshots/ddadfbbd31014fa0d6c3bbf457070d499ec19241"

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
PICKED_IDX = [202, 244, 59]


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def _joint_image(sample):
    return _fix(sample["image"][1]).replace("CAM_FRONT.jpg", "CAM_JOINT.jpg")


def remap_key(k):
    # Fast-dDrive's live state_dict uses `model.language_model.X` and
    # `model.visual.X` (transformers auto-converts during from_pretrained).
    # Fast-dVLM safetensors already use this naming — NO remap needed.
    return k


def main():
    print(f"Loading Fast-dDrive model class from {DDRIVE_DIR} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        DDRIVE_DIR, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    ).eval()

    print(f"\nReplacing weights with Fast-dVLM ckpt from {DVLM_DIR} ...", flush=True)
    state_dict = {}
    for sf in sorted(glob.glob(f"{DVLM_DIR}/model-*.safetensors")):
        print(f"  loading {os.path.basename(sf)}", flush=True)
        state_dict.update(load_file(sf))
    print(f"  raw keys: {len(state_dict)}", flush=True)

    remapped = {remap_key(k): v for k, v in state_dict.items()}
    print(f"  remapped keys: {len(remapped)}", flush=True)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"  missing keys: {len(missing)}", flush=True)
    print(f"  unexpected keys: {len(unexpected)}", flush=True)
    if missing[:5]:
        print(f"  sample missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  sample unexpected: {unexpected[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(DDRIVE_DIR, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(DDRIVE_DIR, use_fast=False)
    processor.tokenizer = tokenizer
    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])
    print(f"\nmask_id={mask_id}", flush=True)

    data = json.load(open(DATA))
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
            print(f"  ! image missing: {img_path}"); continue
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
            threshold=0.9,
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
        print(f"  {response[:1500]}")
        results[idx] = {"latency": latency, "response": response, "nav": nav, "speed": speed}

    with open("/tmp/dvlm_with_ddrive.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved /tmp/dvlm_with_ddrive.json")


if __name__ == "__main__":
    main()
