"""Add the burning-car test_image to examples/longtail_10/ with both
SGLang Fast-dVLM and dVLM-AD outputs. Uses mock ego state from
examples/test_image_run/meta.json. test_image.png is not from Waymo,
so:
  - SGLang: uses the single image as if it were the joint view
  - dVLM-AD: receives the same image 3 times (front-left/front/right
    slots all share the same view; degraded input but lets the
    finetuned model still answer)
"""
import json
import math
import os
import re
import shutil
import sys
import time
import traceback

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, parse_filled, build_template_v3, PROMPT_V3, TRAJECTORY_DT

OUT_DIR = os.path.join(ROOT, "examples", "longtail_10", "test_image_burning_car")
IMG = os.path.join(ROOT, "examples", "test_image.png")

# Mock ego state for the burning-car OOD scene (test_image.png is NOT a
# Waymo sample, so we synthesize a plausible state).
MOCK_META = {
    "image": IMG,
    "scenario": "Highway, vehicle on fire ahead, orange cones diverting "
                "traffic. Ego cruising at 15 m/s, lightly decelerating, "
                "nav=GO_STRAIGHT.",
    "mock_speed_m_s": 15.0,
    "mock_accel_m_s2": -0.5,
    "mock_nav": "GO_STRAIGHT",
    "note": "test_image.png is not a Waymo sample; ego state is synthesized "
            "for an OOD hazard scene.",
}


def _build_mock_sample(meta):
    """Make a Waymo-like dict so build_prompt_v3 and the loaders can read it."""
    speed = meta["mock_speed_m_s"]
    nav = meta["mock_nav"]
    # Synthesize 1.5 s of cruise history at 0.1 s steps (16 points).
    dt = 0.1
    n = 16
    hist = [(-speed * (n - 1 - i) * dt, 0.0) for i in range(n)]
    return {
        "sample_id": "test_image_burning_car",
        "navigation_command": nav,
        "velocity": [(speed, 0.0)],
        "acceleration": [(meta["mock_accel_m_s2"], 0.0)],
        "image": [IMG, IMG, IMG],  # same image for all 3 cams (dVLM-AD path)
        "history waypoints": hist,
        "future waypoints": [],  # no GT
        # dVLM-AD expects conversations[0/1] from the data file. We'll
        # synthesize a Waymo-style prompt + template ourselves.
        "conversations": None,
    }


def _build_dvlm_ad_prompt_template(sample):
    """Compose conversations[0/1] for the mock sample using the EXACT
    schema dVLM-AD was finetuned on. We borrow the prompt + template
    verbatim from a Waymo sample (any one of the 479 — they all share
    the same template structure, only the historical state differs);
    then we substitute in our mock ego state in the prompt text.

    Result: dVLM-AD receives input in its training distribution.
    """
    DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
    canon = json.load(open(DATA))[0]  # any sample works for the template
    template_str = canon["conversations"][1]["value"]

    # Build a mock historical-state block in the data-file's exact format:
    # "(t-3.0s) [x, y], Acceleration: X ax, Y ay m/s², Velocity: X vx, Y vy m/s,; ..."
    speed = sample["velocity"][-1][0]
    accel = sample["acceleration"][-1][0]
    nav = sample["navigation_command"]
    # 7 history points at 0.5 s spacing (3.0 s back to now), straight cruise.
    history_lines = []
    for k, t in enumerate([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]):
        x = speed * t        # ego frame: ego currently at (0,0), past x < 0
        history_lines.append(
            f"(t{t:+.1f}s) [{x:.2f}, 0.00], "
            f"Acceleration: X {accel:.2f}, Y 0.00 m/s², "
            f"Velocity: X {speed:.2f}, Y 0.00 m/s,"
        )
    hist_str = "; ".join(history_lines)

    user_text = (
        "You are an expert autonomous driving agent.\n"
        "Task 1: Critical Object Detection\n"
        "For each class below, answer \"yes\" or \"no\" to indicate whether it "
        "affects the ego vehicle’s behavior or future trajectory:\n"
        "[nearby_vehicle, pedestrian, cyclist, construction, traffic_element, "
        "weather_condition, road_hazard, emergency_vehicle, animal, "
        "special_vehicle, conflicting_vehicle, door_opening_vehicle]\n"
        "Task 2: Scene Reasoning\n"
        "Predict the future behavior of the identified critical objects and "
        "explain how the identified critical objects or conditions affect the "
        "ego vehicle’s next 3-second trajectory.\n"
        "Task 3: Meta-Behavior Prediction\n"
        "Predict the ego vehicle’s future meta-driving behavior:\n"
        "- speed ∈ {keep, accelerate, decelerate, stop, other}\n"
        "- command ∈ {straight, yield, left_turn, right_turn, lane_follow, "
        "lane_change_left, lane_change_right, reverse, overtake, other}\n"
        "Task 4: Trajectory Prediction\n"
        "Predict the optimal 5-second future trajectory (5 waypoints, "
        "1 s intervals).\n"
        "\n"
        "Input:\n"
        "- <image>: three front-view frames from left front, center front, "
        "right front cameras. \n"
        f"- High-level navigation command: {nav}\n"
        f"- Historical ego state: Provided are the previous ego vehicle "
        f"status recorded over the last 3.0 seconds (at 0.5-second intervals)."
        f"{hist_str}"
    )
    return user_text, template_str


def _build_template_ids_from_string(tokenizer, mask_id, template_str):
    if template_str.startswith('"') and template_str.endswith('"'):
        template_str = template_str[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    parts = template_str.split("<|mdm_mask|>")
    ids = []
    for i, part in enumerate(parts):
        if part:
            ids += tokenizer.encode(part, add_special_tokens=False)
        if i < len(parts) - 1:
            ids.append(mask_id)
    return ids


def run_sglang(sample, out_dir):
    """SGLang Fast-dVLM template-fill."""
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    bundle = loader.load(algorithm="mdm")

    prompt = build_prompt_v3(sample)
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    # Warmup
    loader.generate(bundle, [IMG], prompt, temperature=0.0,
                     nav_command=sample["navigation_command"])
    # Real run
    text, latency = loader.generate(
        bundle, [IMG], prompt, temperature=0.0,
        nav_command=sample["navigation_command"],
    )
    pred = parse_filled(text)
    with open(os.path.join(out_dir, "output.json"), "w") as f:
        json.dump({
            "model": "Fast_dVLM_3B (zero-shot via SGLang HierarchyBlock)",
            "model_output_text": text,
            "predicted_waypoints_10": pred[:10],
            "latency_s": latency,
        }, f, indent=2)
    print(f"  SGLang done  lat={latency:.2f}s, {len(text)} chars")
    loader.shutdown(bundle)


def run_dvlm_ad(sample, out_dir):
    """dVLM-AD with the same image fed to all 3 cam slots."""
    import torch, torch.nn.functional as F, copy
    from PIL import Image

    from eval.loaders import dvlm_ad
    bundle = dvlm_ad.load()
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    image_processor = bundle["image_processor"]
    device = bundle["device"]
    mask_id = bundle["mask_id"]

    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    user_text, template_str = _build_dvlm_ad_prompt_template(sample)
    with open(os.path.join(out_dir, "dvlm_ad_prompt.txt"), "w") as f:
        f.write(user_text)
    with open(os.path.join(out_dir, "dvlm_ad_template.txt"), "w") as f:
        f.write(template_str)

    images = [Image.open(IMG).convert("RGB") for _ in range(3)]
    image_tensor = process_images(images, image_processor, model.config)
    target_dtype = next(model.parameters()).dtype
    image_tensor = [t.to(dtype=target_dtype, device=device) for t in image_tensor]
    image_sizes = [img.size for img in images]

    user_text_no_img = user_text.replace("<image>", "").strip()
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + user_text_no_img)
    conv.append_message(conv.roles[1], None)
    chat_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        chat_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(device)
    (_, _, _, _, prompt_embeds, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, image_tensor, ["image"],
        image_sizes=image_sizes,
    )
    L_prompt = prompt_embeds.shape[1]

    template_ids_list = _build_template_ids_from_string(tokenizer, mask_id, template_str)
    template_ids = torch.tensor(template_ids_list, dtype=torch.long, device=device)
    L_template = template_ids.numel()
    is_mask = (template_ids == mask_id)
    n_mask = int(is_mask.sum().item())
    steps = 64
    base = max(1, n_mask // steps)
    remainder = n_mask % steps
    budget = [base + (1 if i < remainder else 0) for i in range(steps)]
    forbidden_ids = [126081, 126080, 126346, 126347, mask_id]
    embed_layer = model.get_model().embed_tokens
    template_embeds = embed_layer(template_ids).unsqueeze(0)

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for step in range(steps):
            if not is_mask.any():
                break
            full_embeds = torch.cat([prompt_embeds, template_embeds], dim=1)
            out = model.get_model()(
                inputs_embeds=full_embeds, attention_mask=None,
                position_ids=None, use_cache=False, return_dict=True,
            )
            logits = model.lm_head(out.last_hidden_state)[0, L_prompt:L_prompt + L_template].float()
            for bid in forbidden_ids:
                if 0 <= bid < logits.shape[-1]:
                    logits[:, bid] = float("-inf")
            argmax = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)
            conf = probs.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(is_mask, conf, torch.tensor(float("-inf"), device=device))
            k = min(budget[step], int(is_mask.sum().item()))
            if k <= 0:
                break
            _, top_idx = torch.topk(conf, k)
            template_ids[top_idx] = argmax[top_idx]
            is_mask[top_idx] = False
            new_emb = embed_layer(argmax[top_idx])
            template_embeds[0, top_idx] = new_emb
    torch.cuda.synchronize()
    latency = time.time() - t0
    text = tokenizer.decode(template_ids.tolist(), skip_special_tokens=False)
    text = text.replace("<|endoftext|>", "")
    clean = re.sub(r"<\|mdm_(start|end)\|>", "", text).strip()
    with open(os.path.join(out_dir, "dvlm_ad_output.json"), "w") as f:
        json.dump({
            "model": "dVLM-AD_waymo (LLaDA-V-8B finetuned)",
            "raw_output_text": text,
            "clean_output_text": clean,
            "latency_s": latency,
            "n_mask_positions": n_mask,
            "n_diffusion_steps": steps,
            "note": "test_image.png is single-view; dVLM-AD was fed the SAME image to all 3 cam slots.",
        }, f, indent=2)
    print(f"  dVLM-AD done lat={latency:.2f}s, {len(clean)} chars")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Copy image
    shutil.copy(IMG, os.path.join(OUT_DIR, "test_image.png"))

    # Write meta
    meta = MOCK_META
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    sample = _build_mock_sample(meta)
    print(f"Scenario: {meta['scenario']}")
    print(f"Output dir: {OUT_DIR}\n")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "both"
    if cmd in ("sglang", "both"):
        print("Running SGLang...")
        run_sglang(sample, OUT_DIR)
    if cmd in ("ad", "both"):
        print("\nRunning dVLM-AD...")
        run_dvlm_ad(sample, OUT_DIR)


if __name__ == "__main__":
    main()
