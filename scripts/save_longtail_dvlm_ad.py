"""Run dVLM-AD (LLaDA-V finetuned on Waymo CoT) on the same 10 longtail
samples and save its prompt/template/output alongside the SGLang outputs
in examples/longtail_10/.

dVLM-AD uses 3 individual cams (front-left, front, front-right), NOT the
joint view, because it was trained on that input format. We save all
three cam images plus the dVLM-AD-specific prompt/template/output.
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

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
OUT_DIR = os.path.join(ROOT, "examples", "longtail_10")

PICKED = [202, 244, 59, 327, 107, 142, 66, 86, 374, 143]


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


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


def main():
    data = json.load(open(DATA))
    picked = [(i, data[i]) for i in PICKED]

    import torch
    import torch.nn.functional as F
    import copy
    from PIL import Image
    from eval.loaders import dvlm_ad
    print("Loading dVLM-AD (LLaDA-V finetuned)...")
    bundle = dvlm_ad.load()
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    image_processor = bundle["image_processor"]
    device = bundle["device"]
    mask_id = bundle["mask_id"]

    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    steps = 64

    for idx, s in picked:
        sid = s["sample_id"][:16]
        sample_dir = os.path.join(OUT_DIR, f"{idx:03d}_{sid}_{s['navigation_command']}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n=== idx={idx} nav={s['navigation_command']} ===")

        # Copy the 3 individual cam images
        img_paths = [_fix(p) for p in s["image"]]
        names = ["cam_front_left.jpg", "cam_front.jpg", "cam_front_right.jpg"]
        for src, dst_name in zip(img_paths, names):
            if os.path.exists(src):
                shutil.copy(src, os.path.join(sample_dir, dst_name))

        # Save dVLM-AD prompt + template (verbatim from data file)
        user_text = s["conversations"][0]["value"]
        template_str = s["conversations"][1]["value"]
        with open(os.path.join(sample_dir, "dvlm_ad_prompt.txt"), "w") as f:
            f.write(user_text)
        with open(os.path.join(sample_dir, "dvlm_ad_template.txt"), "w") as f:
            f.write(template_str)

        try:
            # Load 3 cams
            images = [Image.open(p).convert("RGB") for p in img_paths]
            image_tensor = process_images(images, image_processor, model.config)
            target_dtype = next(model.parameters()).dtype
            image_tensor = [t.to(dtype=target_dtype, device=device) for t in image_tensor]
            image_sizes = [img.size for img in images]

            # Build chat-templated input
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
            # Strip dVLM-AD's <|mdm_start|>/<|mdm_end|> for cleanliness
            clean_text = re.sub(r"<\|mdm_(start|end)\|>", "", text).strip()

            out_data = {
                "model": "dVLM-AD_waymo (LLaDA-V-8B finetuned)",
                "raw_output_text": text,
                "clean_output_text": clean_text,
                "latency_s": latency,
                "n_mask_positions": n_mask,
                "n_diffusion_steps": steps,
            }
            with open(os.path.join(sample_dir, "dvlm_ad_output.json"), "w") as f:
                json.dump(out_data, f, indent=2)
            print(f"  done lat={latency:.2f}s, {len(clean_text)} chars → {sample_dir}")
        except Exception as e:
            traceback.print_exc()
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
