"""Fast-dVLM inference using Fast-dDrive's algorithm + V3 template.

Replaces the old SGLang template-fill loader. Now we follow Fast-dDrive's
pattern directly:

  1. Load Fast-dDrive's model class (`Fast_dDriveForConditionalGeneration`)
     — has built-in `mdm_sample_deep_scaffold` / `scaffold_speculative_sample`
     decoding methods attached via `generation_utils.py`.
  2. Replace its finetuned weights with **our** Fast-dVLM-3B (zero-shot) ckpt.
  3. Monkey-patch `section_utils.build_deep_json_scaffold` so it returns OUR
     V3 schema (12-cat open-vocab critical_objects + complexity + 64-token
     explanation + 2-word verb behavior + semantic-format trajectory)
     instead of Fast-dDrive's yes/no schema.
  4. Use V3 prompt; dispatch through `mdm_sample_deep_scaffold` (HF
     transformers forward path — no SGLang Engine here).

Public API (compat with old loader so scripts don't need rewriting):

    bundle = load(model_path=None, algorithm="mdm", ...)
    text, latency = generate(bundle, image_paths, prompt,
                              max_new_tokens=None, temperature=0.0,
                              nav_command=..., **kwargs)
    shutdown(bundle)

Notes:
  * algorithm: "mdm" → mdm_sample_deep_scaffold  (paper threshold=0.9)
               "ss"  → scaffold_speculative_sample (paper canonical, faster)
               "ar"  → free-form HF generate (no template-fill)
  * `kwargs.get('threshold', 0.9)` overrides confidence threshold.
"""
from __future__ import annotations

import glob
import os
import sys
import time
import re
from typing import Optional

# Path setup so we can import eval.template_v3
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _HERE]
sys.path.insert(0, os.path.join(_HERE, ".."))


DEFAULT_DVLM_PATH = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"
DEFAULT_DDRIVE_PATH = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/transformers/models--xiwenyoumu--Fast-dDrive/snapshots/ddadfbbd31014fa0d6c3bbf457070d499ec19241"

_SECTION_OF_KIND = {
    "critical_head": "crit_cmplx", "critical_tail": "crit_cmplx",
    "complexity": "crit_cmplx",
    "explanation": "expl",
    "long_w1": "beh", "long_w2": "beh", "lat_w1": "beh", "lat_w2": "beh",
    "traj_sign": "traj", "traj_tens": "traj",
    "traj_ones": "traj", "traj_frac": "traj",
}

_DDRIVE_SECTION_NAMES = {
    "crit_cmplx": "critical_objects",
    "expl":       "explanation",
    "beh":        "future_meta_behavior",
    "traj":       "trajectory",
}


def _compute_section_spans(ids, slot_info):
    """Walk slot_info to find contiguous V3 section spans. Scaffold tokens
    inherit the section of the next mask; trailing scaffold inherits the
    previous mask's section."""
    n = len(ids)
    section_per_pos = [None] * n
    for pos, kind in slot_info:
        section_per_pos[pos] = _SECTION_OF_KIND.get(kind, kind)
    # Backward-fill from next mask.
    last = None
    for i in range(n - 1, -1, -1):
        if section_per_pos[i] is not None:
            last = section_per_pos[i]
        else:
            section_per_pos[i] = last
    # Forward-fill any trailing positions.
    last = None
    for i in range(n):
        if section_per_pos[i] is not None:
            last = section_per_pos[i]
        elif last is not None:
            section_per_pos[i] = last

    spans = []
    cur_sec, cur_start = section_per_pos[0], 0
    for i in range(1, n):
        if section_per_pos[i] != cur_sec:
            spans.append((cur_sec, cur_start, i))
            cur_sec, cur_start = section_per_pos[i], i
    spans.append((cur_sec, cur_start, n))
    return spans


def _inject_nav_into_template(template_ids, slot_info, tokenizer, mask_id, nav_command):
    """Splice `"navigation_command": "<nav>", ` immediately BEFORE the
    `"future_meta_behavior":` key. Updates slot_info positions."""
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    long_w1_pos = None
    for local_pos, kind in slot_info:
        if kind == "long_w1":
            long_w1_pos = local_pos
            break
    if long_w1_pos is None:
        return template_ids, slot_info
    future_tok = enc("future")[0]
    future_pos = None
    for i in range(long_w1_pos - 1, -1, -1):
        if template_ids[i] == future_tok:
            future_pos = i
            break
    if future_pos is None:
        return template_ids, slot_info
    inject = enc(f'navigation_command": "{nav_command}", "')
    new_template = template_ids[:future_pos] + inject + template_ids[future_pos:]
    n_inj = len(inject)
    new_slot_info = [
        (p + n_inj if p >= future_pos else p, k) for p, k in slot_info
    ]
    return new_template, new_slot_info


def _build_v3_scaffold_for_ddrive(tokenizer, mask_id, nav_command=None):
    """Drop-in replacement for Fast-dDrive's `build_deep_json_scaffold`.

    Returns the same triple but using OUR V3 schema:
      scaffold_tokens   : List[int]
      section_ranges    : Dict[name, (start, end)]
      scaffold_mask     : List[int]  (1 = scaffold, 0 = value/mask position)
    """
    from template_v3 import build_template_ids_v3

    ids, slot_info, _ = build_template_ids_v3(tokenizer, mask_id)
    if nav_command:
        ids, slot_info = _inject_nav_into_template(
            ids, slot_info, tokenizer, mask_id, nav_command,
        )

    spans = _compute_section_spans(ids, slot_info)
    section_ranges = {}
    for sec, start, end in spans:
        name = _DDRIVE_SECTION_NAMES.get(sec, sec)
        if name in section_ranges:
            prev_start, _ = section_ranges[name]
            section_ranges[name] = (prev_start, end)
        else:
            section_ranges[name] = (start, end)

    mask_positions = {pos for pos, _ in slot_info}
    scaffold_mask = [0 if i in mask_positions else 1 for i in range(len(ids))]
    return ids, section_ranges, scaffold_mask


# ───────────────────────────────────────────────────────────────────────────
#  Public API
# ───────────────────────────────────────────────────────────────────────────

def _strip_template_from_prompt(prompt: str) -> str:
    """Drop the TEMPLATE (...) tail from the V3 prompt — `mdm_sample_deep_scaffold`
    constructs its own scaffold from `build_deep_json_scaffold` (which we've
    patched to return the V3 scaffold). The prompt only needs the instruction
    block."""
    i = prompt.find("TEMPLATE (")
    return prompt[:i].rstrip() if i > 0 else prompt


def load(
    model_path=None,
    algorithm: str = "mdm",
    dvlm_weights_path: Optional[str] = None,
    ddrive_model_path: Optional[str] = None,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    **_kwargs,
):
    """Load Fast-dDrive class with Fast-dVLM weights, patch V3 scaffold.

    Args:
      model_path: ignored (back-compat with old SGLang loader API).
        Use `dvlm_weights_path` / `ddrive_model_path` instead.
      algorithm: "mdm" / "ss" / "ar" (see module docstring).
      dvlm_weights_path: path to Fast-dVLM-3B safetensors dir.
        Defaults to {DEFAULT_DVLM_PATH}.
      ddrive_model_path: path to Fast-dDrive snapshot (model code).
        Defaults to {DEFAULT_DDRIVE_PATH}.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from safetensors.torch import load_file

    dvlm_path = dvlm_weights_path or DEFAULT_DVLM_PATH
    ddrive_path = ddrive_model_path or DEFAULT_DDRIVE_PATH

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}.get(torch_dtype, torch.bfloat16)

    print(f"[Fast-dVLM+dDrive] loading Fast-dDrive class from {ddrive_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        ddrive_path, torch_dtype=dtype, device_map=device,
        trust_remote_code=True,
    ).eval()

    print(f"[Fast-dVLM+dDrive] replacing weights with Fast-dVLM from {dvlm_path}",
          flush=True)
    state_dict = {}
    for sf in sorted(glob.glob(f"{dvlm_path}/model-*.safetensors")):
        state_dict.update(load_file(sf))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ! missing keys: {len(missing)}  first: {missing[:3]}")
    if unexpected:
        print(f"  ! unexpected keys: {len(unexpected)}  first: {unexpected[:3]}")

    tokenizer = AutoTokenizer.from_pretrained(ddrive_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(ddrive_path, use_fast=False)
    processor.tokenizer = tokenizer
    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])

    # Monkey-patch Fast-dDrive's scaffold builder to use V3 schema.
    # The module path includes the snapshot hash; resolve dynamically.
    import importlib
    snapshot = os.path.basename(ddrive_path.rstrip("/"))
    su_mod = importlib.import_module(
        f"transformers_modules.{snapshot}.section_utils"
    )
    # The current nav_command travels via a closure cell mutated per-request.
    nav_cell = {"value": None}

    def patched_builder(tok, **kwargs):
        return _build_v3_scaffold_for_ddrive(
            tok, mask_id=mask_id, nav_command=nav_cell["value"],
        )

    su_mod.build_deep_json_scaffold = patched_builder
    print(f"[Fast-dVLM+dDrive] patched section_utils.build_deep_json_scaffold "
          f"→ V3 scaffold (algorithm={algorithm})", flush=True)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "mask_id": mask_id,
        "algorithm": algorithm,
        "device": device,
        "nav_cell": nav_cell,
    }


def generate(
    bundle,
    image_paths,
    question: str,
    max_new_tokens: int = None,
    temperature: float = 0.0,
    nav_command: Optional[str] = None,
    **kwargs,
):
    """Run V3 template-fill via Fast-dDrive's algorithm.

    Returns (text, latency_s). Same signature as the old SGLang loader.

    Extra kwargs:
      threshold (float, 0.9): confidence threshold for commit.
      max_tokens (int, 1024): max output tokens.
    """
    import torch
    from PIL import Image

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    processor = bundle["processor"]
    mask_id = bundle["mask_id"]
    algorithm = bundle.get("algorithm", "mdm")
    device = bundle.get("device", "cuda:0")
    nav_cell = bundle["nav_cell"]

    # Inject nav into the patched scaffold builder via the closure cell.
    nav_cell["value"] = nav_command

    image_path = image_paths[0] if image_paths else None
    user_text = _strip_template_from_prompt(question)

    # Build chat input.
    if image_path:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]}]
    else:
        image = None
        messages = [{"role": "user", "content": user_text}]
    text_in = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if image is not None:
        inputs = processor(text=[text_in], images=[image], return_tensors="pt").to(device)
    else:
        inputs = processor(text=[text_in], return_tensors="pt").to(device)

    threshold = float(kwargs.get("threshold", 0.9))
    max_tokens = int(kwargs.get("max_tokens", max_new_tokens or 1024))
    block_size = int(kwargs.get("block_size", 32))

    if algorithm == "ar":
        # Plain HF AR generation, no template-fill.
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
            )
        torch.cuda.synchronize()
        latency = time.time() - t0
        trimmed = out[0, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(trimmed, skip_special_tokens=True)
        return text, latency

    # Build mdm_sample_deep_scaffold args.
    mdm_kwargs = dict(
        input_ids=inputs.input_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        max_tokens=max_tokens,
        mask_id=mask_id,
        threshold=threshold,
    )
    if getattr(inputs, "pixel_values", None) is not None:
        mdm_kwargs["pixel_values"] = inputs.pixel_values
    if getattr(inputs, "image_grid_thw", None) is not None:
        mdm_kwargs["image_grid_thw"] = inputs.image_grid_thw

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        if algorithm == "ss":
            out = model.scaffold_speculative_sample(**mdm_kwargs)
        else:  # default "mdm"
            out = model.mdm_sample_deep_scaffold(**mdm_kwargs)
    torch.cuda.synchronize()
    latency = time.time() - t0

    trimmed = out[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(trimmed, skip_special_tokens=True)
    return text, latency


def shutdown(bundle):
    """No-op for HF-based loader (model is GC'd when bundle is dropped)."""
    if bundle:
        bundle.pop("model", None)
    import torch
    torch.cuda.empty_cache()
