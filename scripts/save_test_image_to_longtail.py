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
META_SRC = os.path.join(ROOT, "examples", "test_image_run", "meta.json")


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
    """Compose a Waymo-style conversations[0/1] pair for the mock sample
    (since test_image isn't a Waymo sample so we can't pull from data
    file). Mirrors the dVLM-AD training format: prompt asks for the same
    fields, template has <|mdm_mask|> markers."""
    speed = sample["velocity"][-1][0]
    nav = sample["navigation_command"]
    user_text = (
        "<image><image><image>\n"
        "You are an autonomous driving assistant. Given the three front "
        "camera views and the ego state, identify critical objects, "
        "explain reasoning, predict behavior and trajectory.\n\n"
        f"Ego speed: {speed:.1f} m/s. Navigation: {nav}.\n\n"
        "Output a single JSON with critical_objects (12 keys), explanation, "
        "future_meta_behavior {longitudinal, lateral}, trajectory."
    )
    # 12 critical categories, similar to V3
    cats = ["nearby_vehicle", "pedestrian", "cyclist", "construction",
             "traffic_element", "weather_condition", "road_hazard",
             "emergency_vehicle", "animal", "special_vehicle",
             "conflicting_vehicle", "door_opening_vehicle"]
    mask = "<|mdm_mask|>"
    co_lines = ", ".join([f'"{c}": "<|mdm_start|>{mask * 10}<|mdm_end|>"' for c in cats])
    template_str = (
        '{"critical_objects": {' + co_lines + '}, '
        f'"explanation": "<|mdm_start|>{mask * 100}<|mdm_end|>", '
        '"future_meta_behavior": {'
        f'"longitudinal": "<|mdm_start|>{mask * 5}<|mdm_end|>", '
        f'"lateral": "<|mdm_start|>{mask * 5}<|mdm_end|>"}}, '
        f'"trajectory": "<|mdm_start|>{mask * 100}<|mdm_end|>"}}'
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

    # Copy meta
    meta = json.load(open(META_SRC))
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
