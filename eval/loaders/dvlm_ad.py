"""dVLM-AD_waymo (official SFT'd LLaDA-V checkpoint) loader.

Uses the official prompt + template (1 mask per critical_object, 5-wp trajectory
@ 1 s, Task 1/2/3/4 prompt structure). Reuses our diffusion fill loop (same
algorithm as the paper's `fast_dllm_hook`, just without the KV-cache speedup).

This is the test of: "if I align template+prompt to official, does my pipeline
produce sensible output on the SFT'd model?" If yes, my pipeline code is sound.
"""
from __future__ import annotations
import copy, os, sys, time
from PIL import Image
import torch

DEFAULT_PATH = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/dVLM-AD_waymo"
DLLMRL_SAMPLE = "/weka/home/ext-yingzima/dLLM-RL/sample"
MASK_ID = 126336  # LLaDA-V <|mdm_mask|>

# Use dLLM-RL's llava package (same as our lladav.py loader uses)
if DLLMRL_SAMPLE not in sys.path:
    sys.path.insert(0, DLLMRL_SAMPLE)
# Drop LaViDa's llava if it was loaded earlier
for p in list(sys.path):
    if "/LaViDa" in p:
        sys.path.remove(p)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from template_official import build_template_official, build_prompt_official, build_template_ids_official


def load(model_path=None, device="cuda", dtype="bfloat16"):
    # Clean any previously imported llava (LaViDa) so we get dLLM-RL's version
    for k in list(sys.modules):
        if k == "llava" or k.startswith("llava."):
            del sys.modules[k]
    # Monkey-patch AutoTokenizer.from_pretrained to default to use_fast=True
    # because some checkpoints (LaViDa-llada) don't ship a slow tokenizer and
    # use_fast=False silently returns False on those.
    from transformers import AutoTokenizer
    _orig = AutoTokenizer.from_pretrained
    def _patched(*args, **kwargs):
        if "use_fast" in kwargs and kwargs["use_fast"] is False:
            kwargs["use_fast"] = True
        return _orig(*args, **kwargs)
    AutoTokenizer.from_pretrained = staticmethod(_patched)

    from llava.model.builder import load_pretrained_model
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    path = model_path or DEFAULT_PATH
    tokenizer, model, image_processor, _ = load_pretrained_model(
        path, None, "llava_llada", attn_implementation="sdpa", device_map=device,
        torch_dtype=dtype,
    )
    # Restore
    AutoTokenizer.from_pretrained = _orig
    model.eval()
    special = {"additional_special_tokens": [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]}
    n_added = tokenizer.add_special_tokens(special)
    if n_added > 0:
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
    return {"model": model, "tokenizer": tokenizer, "image_processor": image_processor,
            "device": device, "dtype": dtype, "mask_id": MASK_ID}


def _build_prompt_chat(question: str):
    from llava.conversation import conv_templates
    from llava.constants import DEFAULT_IMAGE_TOKEN
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def generate(bundle, image_paths, question, gen_length=512, steps=128, temperature=0.0):
    """Run constrained diffusion fill on dVLM-AD_waymo with the OFFICIAL template.

    Note: `question` here is the **OFFICIAL prompt text** built by
    `template_official.build_prompt_official(sample)`. We ignore the generic
    `--gen_length 256` from run_traj_v3 and use the official 512 / 128 steps.
    """
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    image_processor = bundle["image_processor"]
    device = bundle["device"]
    mask_id = bundle["mask_id"]

    # 1. Images — use ALL three front cams (official uses 3)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_tensor = process_images(images, image_processor, model.config)
    target_dtype = next(model.parameters()).dtype
    image_tensor = [t.to(dtype=target_dtype, device=device) for t in image_tensor]
    image_sizes = [img.size for img in images]

    # 2. Build the chat-templated input_ids
    user_text = question.replace("<image>", "").strip()
    chat_prompt = _build_prompt_chat(user_text)
    input_ids = tokenizer_image_token(chat_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors="pt").unsqueeze(0).to(device)

    # 3. Merge image embeds via the LlavaMetaForCausalLM helper
    (_, _, _, _, prompt_embeds, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, image_tensor, ["image"], image_sizes=image_sizes,
    )
    L_prompt = prompt_embeds.shape[1]

    # 4. Build the official template token IDs (segment-by-segment so mask_id
    #    is guaranteed to be a single token regardless of tokenizer behaviour).
    template_ids_list, _slots = build_template_ids_official(tokenizer, mask_id)
    template_ids = torch.tensor(template_ids_list, dtype=torch.long, device=device)
    L_template = template_ids.numel()

    is_mask = (template_ids == mask_id)
    n_mask = int(is_mask.sum().item())
    assert n_mask == 222, f"Expected 222 mask tokens (official), got {n_mask}"

    # Per-step budget
    base = max(1, n_mask // steps)
    remainder = n_mask % steps
    budget = [base + (1 if i < remainder else 0) for i in range(steps)]

    # Forbidden tokens (from official fast_dllm_hook)
    forbidden_ids = [126081, 126080, 126346, 126347, mask_id]

    # 5. Pre-compute template_embeds
    embed_layer = model.get_model().embed_tokens
    template_embeds = embed_layer(template_ids).unsqueeze(0)  # [1, L_template, H]

    if device.startswith("cuda"):
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
            if temperature <= 0:
                argmax = logits.argmax(dim=-1)
            else:
                p_ = torch.softmax(logits / max(temperature, 1e-3), dim=-1)
                argmax = torch.multinomial(p_, 1).squeeze(-1)
            probs = torch.softmax(logits, dim=-1)
            conf = probs.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(is_mask, conf, torch.tensor(float("-inf"), device=device))
            k = min(budget[step], int(is_mask.sum().item()))
            if k <= 0:
                break
            _, top_idx = torch.topk(conf, k)
            template_ids[top_idx] = argmax[top_idx]
            new_emb = embed_layer(argmax[top_idx])
            template_embeds[0, top_idx] = new_emb.to(template_embeds.dtype)
            is_mask[top_idx] = False

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency = time.time() - t0

    text_out = tokenizer.decode(template_ids.tolist(), skip_special_tokens=False)
    return text_out, latency
