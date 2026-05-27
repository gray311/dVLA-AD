"""Fast-dVLM via SGLang with V3 template fill.

Uses the modified SGLang fork's `dllm_template_token_ids` sampling param to
inject a pre-positioned scaffold + mask layout (V3 schema: critical_objects,
explanation, behavior, trajectory). The dllm algorithm refines only mask
positions; scaffold tokens stay intact across the block-by-block diffusion.

Requires the patched SGLang at:
  /weka/home/ext-yingzima/dVLA-AD/third_party/sglang
(installed via `pip install -e third_party/sglang/python`).

API:
  bundle = load(algorithm="mdm" | "spec")
  text, latency = generate(bundle, image_paths, prompt, max_new_tokens=...)
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _HERE]
sys.path.insert(0, os.path.join(_HERE, ".."))

from template_v3 import build_template_ids_v3


def _build_v3_position_gates(tokenizer, slot_info, template_len):
    """Map each template position to an optional list of allowed token ids.

    Mirrors the per-slot vocab gates from the transformers loader:
      - traj_sign: {'+', '-'}
      - traj_tens/ones/frac: '0'..'9'
      - long/lat verb words: pre-selected vocab per word position
    Other slot kinds (critical_*, explanation, structural padding) get None
    (unrestricted).
    """
    def first_tok(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else -1

    digit_ids = sorted({first_tok(c) for c in "0123456789" if first_tok(c) >= 0})
    sign_ids = sorted({first_tok(c) for c in "+-" if first_tok(c) >= 0})

    long_w1_ids = sorted({first_tok(w) for w in ["speed", "slow", "keep", "stop"] if first_tok(w) >= 0})
    long_w2_ids = sorted({first_tok(w) for w in ["up", "down", "speed", "now"] if first_tok(w) >= 0})
    lat_w1_ids = sorted({first_tok(w) for w in ["keep", "turn", "change"] if first_tok(w) >= 0})
    lat_w2_ids = sorted({first_tok(w) for w in ["lane", "left", "right"] if first_tok(w) >= 0})

    kind_to_allowed = {
        "traj_sign": sign_ids,
        "traj_tens": digit_ids,
        "traj_ones": digit_ids,
        "traj_frac": digit_ids,
        "long_w1":   long_w1_ids,
        "long_w2":   long_w2_ids,
        "lat_w1":    lat_w1_ids,
        "lat_w2":    lat_w2_ids,
    }

    gates = [None] * template_len
    for local_pos, kind in slot_info:
        allowed = kind_to_allowed.get(kind)
        if allowed:
            gates[local_pos] = list(allowed)
    return gates


def _build_json_special_blacklist(tokenizer):
    """Token IDs whose decoded form contains JSON metacharacters; never commit
    these at any mask slot (would corrupt the JSON scaffold)."""
    vocab = tokenizer.get_vocab()
    bad = set()
    bad_chars = ('"', '{', '}', '\\', '“', '”', '‘', '’', '`')
    for _tok_str, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            continue
        if any(c in decoded for c in bad_chars):
            bad.add(tok_id)
    return sorted(bad)

DEFAULT_PATH = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"
PROCESSOR_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

ALGO_MAP = {
    "mdm": "HierarchyBlock",
    "spec": "SpeculativeBlock",
}


def _strip_template_from_prompt(prompt: str) -> str:
    """Drop the TEMPLATE (...) tail from the V3 prompt — we now provide the
    template scaffold via SGLang's dllm_template_token_ids API instead."""
    i = prompt.find("TEMPLATE (")
    return prompt[:i].rstrip() if i > 0 else prompt


def load(model_path=None, algorithm="mdm", mem_fraction_static=0.75,
          quantization=None, max_running_requests=1, chunked_prefill_size=16384):
    """Load Fast-dVLM as sgl.Engine. ~30-60s for model load + CUDA graph capture."""
    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    import sglang as sgl
    from transformers import AutoProcessor, AutoTokenizer

    path = model_path or DEFAULT_PATH
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    dllm_algo = ALGO_MAP[algorithm]
    engine_kwargs = dict(
        model_path=path,
        trust_remote_code=True,
        dtype="bfloat16",
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        chunked_prefill_size=chunked_prefill_size,
        dllm_algorithm=dllm_algo,
        disable_cuda_graph=False,
        log_level="warning",
        enable_metrics=True,
        mm_attention_backend="triton_attn",
    )
    # Larger sub_block_size (16 instead of default 8) → half the sub-block
    # iterations per chunk → ~half the forward count. Quality is preserved
    # because scaffold positions are already committed (visible bidirectional)
    # and only mask positions get refined.
    import yaml, tempfile
    algo_config = {"sub_block_size": 16, "debug": False}
    cfg_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(algo_config, cfg_file)
    cfg_file.close()
    engine_kwargs["dllm_algorithm_config"] = cfg_file.name
    if quantization:
        engine_kwargs["quantization"] = quantization

    engine = sgl.Engine(**engine_kwargs)
    mask_id = tokenizer.encode("|<MASK>|")[0]
    # Cache the JSON-meta blacklist once (vocab scan is slow).
    json_blacklist = _build_json_special_blacklist(tokenizer)
    return {
        "engine": engine,
        "processor": processor,
        "tokenizer": tokenizer,
        "algorithm": algorithm,
        "model_path": path,
        "mask_id": mask_id,
        "json_blacklist": json_blacklist,
    }


def _build_input_ids(processor, image_path, prompt_text):
    """Build prompt input_ids (no template — template is passed separately)."""
    from qwen_vl_utils import process_vision_info
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    return inputs.input_ids[0].tolist()


def generate(bundle, image_paths, question, max_new_tokens=None, temperature=0.0,
              block_size=32):
    """V3 template-fill via SGLang dllm engine. Returns (text, latency_s).

    The template is built from `template_v3.build_template_ids_v3` (the same
    tokenizer used by the engine) and passed via `sampling_params.dllm_template_token_ids`.
    """
    import torch
    engine = bundle["engine"]
    processor = bundle["processor"]
    tokenizer = bundle["tokenizer"]
    mask_id = bundle["mask_id"]

    image_path = image_paths[0] if image_paths else None

    # Strip TEMPLATE (...) from prompt — we provide it via sampling_params now.
    user_text = _strip_template_from_prompt(question)
    input_ids = _build_input_ids(processor, image_path, user_text)

    # Build V3 template (the same builder used by the transformers loader).
    template_ids_list, slot_info, _critical_pairs = build_template_ids_v3(tokenizer, mask_id)

    # Pad template to multiple of block_size — SGLang will chunk it into
    # block_size pieces.
    pad_token = 151643  # Qwen <|endoftext|> — stripped by skip_special_tokens=True
    padded = list(template_ids_list)
    while len(padded) % block_size != 0:
        padded.append(pad_token)
    n_template = len(padded)

    # Build per-position vocab gates. Length matches padded template.
    gates = _build_v3_position_gates(tokenizer, slot_info, n_template)
    json_bad = list(bundle.get("json_blacklist") or [])
    # Also exclude JSON-meta tokens from gated allowlists in case any sneak in.
    json_bad_set = set(json_bad)
    for i, allowed in enumerate(gates):
        if allowed is not None:
            gates[i] = [a for a in allowed if a not in json_bad_set]

    if max_new_tokens is None:
        max_new_tokens = n_template

    sampling = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "dllm_template_token_ids": padded,
        "dllm_template_position_gates": gates,
        # Global blacklist applied at EVERY mask position (incl. critical
        # values, explanation prose): no token containing `"`, `}`, `\`, etc.
        # Without this, free-text slots emit `","` mid-value and corrupt JSON.
        "dllm_template_forbidden_token_ids": json_bad,
    }

    torch.cuda.synchronize()
    t0 = time.time()
    out = engine.generate(
        input_ids=input_ids,
        image_data=[image_path] if image_path else None,
        sampling_params=sampling,
    )
    torch.cuda.synchronize()
    latency = time.time() - t0

    if isinstance(out, list):
        out = out[0]
    text = out.get("text", "") if isinstance(out, dict) else str(out)
    return text, latency


def shutdown(bundle):
    engine = bundle.get("engine")
    if engine is not None:
        engine.shutdown()
