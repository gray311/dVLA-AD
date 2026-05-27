"""Run SGLang Fast-dVLM (template-fill, mdm) AND dVLM-AD on 10 stratified
Waymo samples. Saves outputs to results/waymo_10_compare/.

Two passes so each model gets a fresh process for GPU memory:
  - pass=sglang : Fast_dVLM_3B via modified SGLang fork (V3 template)
  - pass=ad     : dVLM-AD_waymo (LLaDA-V finetuned) with data-file template

Usage:
  python scripts/run_10_waymo_compare.py sglang
  python scripts/run_10_waymo_compare.py ad
  python scripts/run_10_waymo_compare.py both     # sequential, same proc

Then write a merged report via write_10_waymo_compare_report.py.
"""
import json
import math
import os
import sys
import time
import traceback

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PICKED_INDICES = [20, 375, 25, 107, 30, 1, 135, 217, 420, 177]
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
OUT_DIR = os.path.join(ROOT, "results", "waymo_10_compare")


def _fix(p):
    return p.replace(PATH_FIX[0], PATH_FIX[1])


def _front_cam(sample):
    return _fix(sample["image"][1])


def _all_cams(sample):
    return [_fix(p) for p in sample["image"]]


def _build_template_ids_from_string(tokenizer, mask_id, template_str):
    """Tokenize the data file's template string (contains literal <|mdm_mask|>)."""
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


def run_sglang(picked):
    from eval.template_v3 import build_prompt_v3
    from eval.loaders import fast_dvlm_sglang_v3 as loader
    print("\nLoading SGLang Fast-dVLM (template-fill mdm)...")
    bundle = loader.load(algorithm="mdm")
    print("Warmup...")
    _, lat_warm = loader.generate(
        bundle, [_front_cam(picked[0])], build_prompt_v3(picked[0]),
        temperature=0.0, nav_command=picked[0]["navigation_command"],
    )
    print(f"  warmup {lat_warm:.2f}s")

    results = []
    for sample in picked:
        sid = sample["sample_id"]
        print(f"\n=== {sid[:30]} nav={sample['navigation_command']} ===")
        try:
            text, lat = loader.generate(
                bundle, [_front_cam(sample)], build_prompt_v3(sample),
                temperature=0.0, nav_command=sample["navigation_command"],
            )
            print(f"  done ({lat:.2f}s, {len(text)} chars)")
        except Exception as e:
            traceback.print_exc()
            text, lat = f"ERROR: {e}", -1
        vx, vy = sample["velocity"][-1]
        results.append({
            "sample_id": sid,
            "nav": sample["navigation_command"],
            "speed": math.hypot(vx, vy),
            "image": _front_cam(sample),
            "gt_future_5_waypoints": sample["future waypoints"][:5],
            "sglang_template": {"output": text, "latency_s": lat},
        })
    loader.shutdown(bundle)

    out_path = os.path.join(OUT_DIR, "sglang.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


def run_dvlm_ad(picked):
    import torch
    import torch.nn.functional as F
    import copy
    from PIL import Image

    # Load dVLM-AD first — dvlm_ad.load() sets up sys.path so llava
    # (from dLLM-RL/sample) can be imported.
    from eval.loaders import dvlm_ad
    print("\nLoading dVLM-AD (LLaDA-V finetuned)...")
    bundle = dvlm_ad.load()

    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    image_processor = bundle["image_processor"]
    device = bundle["device"]
    mask_id = bundle["mask_id"]

    steps = 64
    results = []
    for sample in picked:
        sid = sample["sample_id"]
        print(f"\n=== {sid[:30]} nav={sample['navigation_command']} ===")
        try:
            img_paths = _all_cams(sample)
            images = [Image.open(p).convert("RGB") for p in img_paths]
            image_tensor = process_images(images, image_processor, model.config)
            target_dtype = next(model.parameters()).dtype
            image_tensor = [t.to(dtype=target_dtype, device=device) for t in image_tensor]
            image_sizes = [img.size for img in images]

            # Prompt from conversations[0]
            user_text = sample["conversations"][0]["value"]
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

            template_str = sample["conversations"][1]["value"]
            template_ids_list = _build_template_ids_from_string(tokenizer, mask_id, template_str)
            template_ids = torch.tensor(template_ids_list, dtype=torch.long, device=device)
            L_template = template_ids.numel()
            is_mask = (template_ids == mask_id)
            n_mask = int(is_mask.sum().item())

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
                    conf = torch.where(is_mask, conf,
                                       torch.tensor(float("-inf"), device=device))
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
            print(f"  done ({latency:.2f}s, {len(text)} chars)")
        except Exception as e:
            traceback.print_exc()
            text, latency = f"ERROR: {e}", -1

        vx, vy = sample["velocity"][-1]
        results.append({
            "sample_id": sid,
            "nav": sample["navigation_command"],
            "speed": math.hypot(vx, vy),
            "image_front": _front_cam(sample),
            "gt_future_5_waypoints": sample["future waypoints"][:5],
            "dvlm_ad": {"output": text, "latency_s": latency},
        })

    out_path = os.path.join(OUT_DIR, "dvlm_ad.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


def main():
    pass_ = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.makedirs(OUT_DIR, exist_ok=True)

    data = json.load(open(DATA))
    picked = [data[i] for i in PICKED_INDICES]
    print(f"Picked {len(picked)} samples (indices {PICKED_INDICES}):")
    for s in picked:
        vx, vy = s["velocity"][-1]
        print(f"  {s['sample_id'][:30]}  nav={s['navigation_command']:14} speed={math.hypot(vx, vy):5.1f}")

    if pass_ in ("sglang", "both"):
        run_sglang(picked)
    if pass_ in ("ad", "both"):
        run_dvlm_ad(picked)


if __name__ == "__main__":
    main()
