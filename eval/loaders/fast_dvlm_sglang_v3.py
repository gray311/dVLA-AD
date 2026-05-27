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
    # Complexity tag — single-token gate restricted to {"simple", "complex"}.
    complexity_ids = sorted({first_tok(w) for w in ["simple", "complex"] if first_tok(w) >= 0})

    kind_to_allowed = {
        "traj_sign": sign_ids,
        "traj_tens": digit_ids,
        "traj_ones": digit_ids,
        "traj_frac": digit_ids,
        "long_w1":   long_w1_ids,
        "long_w2":   long_w2_ids,
        "lat_w1":    lat_w1_ids,
        "lat_w2":    lat_w2_ids,
        "complexity": complexity_ids,
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
          quantization=None, max_running_requests=1, chunked_prefill_size=16384,
          max_total_tokens=None, disable_cuda_graph=False, engine_block_size=160):
    """Load Fast-dVLM as sgl.Engine. ~30-60s for model load + CUDA graph capture.

    engine_block_size: chunk size SGLang uses to feed the V3 template to the
        diffusion algorithm. Default 160 matches the section-aligned decoding
        path (each JSON section becomes its own block: crit_objects+complexity
        = 1 block of 160, explanation = 1, behavior = 1, trajectory = 2). For
        the legacy 32-token mechanical chunking, pass 32.

    max_total_tokens: optional cap on KV pool size (in tokens). When set, the
    KV pool is sized for max_total_tokens regardless of mem_fraction_static —
    useful when a shared GPU has limited free memory.
    """
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
        disable_cuda_graph=disable_cuda_graph,
        log_level="warning",
        enable_metrics=False,
        mm_attention_backend="triton_attn",
    )
    if max_total_tokens is not None:
        engine_kwargs["max_total_tokens"] = max_total_tokens
    # Larger sub_block_size (16 instead of default 8) → half the sub-block
    # iterations per chunk → ~half the forward count. Quality is preserved
    # because scaffold positions are already committed (visible bidirectional)
    # and only mask positions get refined.
    import yaml, tempfile
    algo_config = {"sub_block_size": 32, "debug": False, "block_size": engine_block_size}
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


_SECTION_OF_KIND = {
    "critical_head": "crit_cmplx", "critical_tail": "crit_cmplx",
    "complexity": "crit_cmplx",
    "explanation": "expl",
    "long_w1": "beh", "long_w2": "beh", "lat_w1": "beh", "lat_w2": "beh",
    "traj_sign": "traj", "traj_tens": "traj",
    "traj_ones": "traj", "traj_frac": "traj",
}


def _compute_section_spans(ids, slot_info):
    """Group token positions into contiguous V3 sections (crit_cmplx, expl,
    beh, traj). Scaffold tokens are assigned to the NEXT mask's section.
    Returns list of (section_name, start, end).
    """
    n = len(ids)
    section_per_pos = [None] * n
    for pos, kind in slot_info:
        section_per_pos[pos] = _SECTION_OF_KIND.get(kind, kind)
    # Backward-fill: scaffold before a mask inherits that mask's section.
    last = None
    for i in range(n - 1, -1, -1):
        if section_per_pos[i] is not None:
            last = section_per_pos[i]
        else:
            section_per_pos[i] = last
    # Forward-fill: any trailing positions after the LAST mask (no next mask
    # to inherit from) inherit the PREVIOUS mask's section.
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


def _section_align_template(ids, slot_info, block_size, pad_token):
    """Uniform-block section alignment: pad each section's tail to a multiple
    of block_size. Returns (padded_ids, remapped_slot_info).
    """
    spans = _compute_section_spans(ids, slot_info)
    old_to_new = [0] * len(ids)
    new_ids = []
    for sec, start, end in spans:
        for i, old_pos in enumerate(range(start, end)):
            old_to_new[old_pos] = len(new_ids) + i
        new_ids.extend(ids[start:end])
        while len(new_ids) % block_size != 0:
            new_ids.append(pad_token)
    new_slot_info = [(old_to_new[pos], kind) for pos, kind in slot_info]
    return new_ids, new_slot_info


def _section_variable_template(ids, slot_info, max_chunk_size, pad_token):
    """Section-aligned VARIABLE-size chunks. Each section becomes its own
    chunk (or split into N chunks of max_chunk_size when oversize). No
    padding inserted between sections — only at the end of the last chunk.

    Returns (ids_out, remapped_slot_info, chunk_sizes) where chunk_sizes
    is a list whose sum == len(ids_out).
    """
    spans = _compute_section_spans(ids, slot_info)
    old_to_new = [0] * len(ids)
    new_ids = []
    chunk_sizes = []
    for sec, start, end in spans:
        sec_len = end - start
        for i, old_pos in enumerate(range(start, end)):
            old_to_new[old_pos] = len(new_ids) + i
        new_ids.extend(ids[start:end])
        # Split this section into one or more chunks, each ≤ max_chunk_size.
        if sec_len <= max_chunk_size:
            chunk_sizes.append(sec_len)
        else:
            # Even split — e.g. traj 223 with max=128 → [112, 111] is bad
            # because of mask-boundary issues, so split at section_size // n.
            n_chunks = (sec_len + max_chunk_size - 1) // max_chunk_size
            base = sec_len // n_chunks
            rem = sec_len % n_chunks
            for k in range(n_chunks):
                chunk_sizes.append(base + (1 if k < rem else 0))
    # Pad final chunk to keep CUDA graph happy if needed — but with variable
    # chunk sizes, we don't pad. Just return as-is.
    new_slot_info = [(old_to_new[pos], kind) for pos, kind in slot_info]
    assert sum(chunk_sizes) == len(new_ids), \
        f"sum(chunk_sizes)={sum(chunk_sizes)} != len(new_ids)={len(new_ids)}"
    return new_ids, new_slot_info, chunk_sizes


def _inject_nav_into_template(template_ids, slot_info, tokenizer, mask_id, nav_command):
    """Splice `"navigation_command": "<nav>", ` into the token stream right
    BEFORE the `"future_meta_behavior":` sequence. This puts the nav target
    in immediate bidir-attention context of the behavior masks, biasing the
    lateral verb without conditional gating.

    We search for the `future` token (id 21055 in Qwen2.5-VL tokenizer) and
    insert just BEFORE the leading ` "` token. Falls back to no-op if the
    marker isn't found.
    """
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    # Find the position of the first long_w1 slot — behavior block starts there.
    long_w1_pos = None
    for local_pos, kind in slot_info:
        if kind == "long_w1":
            long_w1_pos = local_pos
            break
    if long_w1_pos is None:
        return template_ids, slot_info

    # Scan backwards from long_w1 for the token id of 'future' (21055 in
    # Qwen2.5-VL). We need to splice BEFORE the leading ` "` token that
    # precedes 'future' — that's the comma+space+quote separator.
    target_future_str = enc("future")
    if not target_future_str:
        return template_ids, slot_info
    future_tok = target_future_str[0]
    future_pos = None
    for i in range(long_w1_pos - 1, -1, -1):
        if template_ids[i] == future_tok:
            future_pos = i
            break
    if future_pos is None:
        return template_ids, slot_info

    # The token sequence at future_pos-1 should be ` "` (id 330). Insert right
    # after that quote-opener so the new key starts cleanly.
    insert_at = future_pos  # we'll insert immediately before 'future'
    # Build the injection: `navigation_command": "<nav>", "` — note we
    # already have ` "` from the original scaffold at future_pos-1, and we
    # want to end with `, "` so the next `future` token continues cleanly.
    inject = enc(f'navigation_command": "{nav_command}", "')
    n_inject = len(inject)

    new_ids = list(template_ids[:insert_at]) + list(inject) + list(template_ids[insert_at:])
    new_slot = []
    for local_pos, kind in slot_info:
        if local_pos >= insert_at:
            new_slot.append((local_pos + n_inject, kind))
        else:
            new_slot.append((local_pos, kind))
    return new_ids, new_slot


def generate(bundle, image_paths, question, max_new_tokens=None, temperature=0.0,
              block_size=160, nav_command=None, **kwargs):
    """V3 template-fill via SGLang dllm engine. Returns (text, latency_s).

    Default decoding: **section-aligned, block_size=160** (Fast-dDrive style).
    The V3 JSON schema has 4 sections (critical_objects+complexity,
    explanation, behavior, trajectory); each section gets its own 160-token
    block (trajectory uses 2). vs the legacy 32-token chunking, this is
    ~2x faster (1.30s → 0.69s on N=30 Waymo val) and ~6% better mean ADE.

    Caller overrides via kwargs:
      section_align (bool, default True): align chunk boundaries to V3 sections
      block_size (int, default 160): bytes per chunk in section-aligned mode
      steps_per_chunk (int, default 4): inner diffusion steps per chunk
      section_align="variable" + max_chunk_size=N: experimental variable-size
        chunks (one per section, no padding). Requires matching
        engine_block_size and currently fails due to SGLang buffer pre-alloc.

    `nav_command` (optional): when set, injects a literal
    `"navigation_command": "<nav>"` key-value right BEFORE the behavior block
    in the template. Biases the model's lateral verb choice toward the correct
    nav-implied direction (without conditional gates).
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

    if nav_command:
        template_ids_list, slot_info = _inject_nav_into_template(
            template_ids_list, slot_info, tokenizer, mask_id, nav_command,
        )

    # Pad template — three modes:
    #   section_align=False  (default): pad once at END to multiple of block_size.
    #   section_align=True:            pad EACH section to multiple of block_size
    #                                   (uniform chunks, some padding waste).
    #   section_align="variable":      each section becomes its own chunk (or
    #                                   split into max_chunk_size pieces). No
    #                                   inter-section padding. Sends `chunk_sizes`
    #                                   to engine so the chunker uses per-chunk
    #                                   sizes instead of global block_size.
    pad_token = 151643  # Qwen <|endoftext|> — stripped by skip_special_tokens=True
    # Section-aligned decoding is ON by default (each V3 section → its own block).
    # Set section_align=False to use legacy 32-token mechanical chunking.
    section_align = kwargs.get("section_align", True)
    chunk_sizes = None
    if section_align == "variable":
        max_chunk_size = int(kwargs.get("max_chunk_size", 192))
        padded, slot_info, chunk_sizes = _section_variable_template(
            template_ids_list, slot_info, max_chunk_size, pad_token,
        )
    elif section_align:
        padded, slot_info = _section_align_template(
            template_ids_list, slot_info, block_size, pad_token,
        )
    else:
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

    # Mark explanation positions for rep-penalty. Without this, the
    # 100-mask explanation slot mode-collapses into "a a a a..." or
    # "the the the..." at high mask density.
    rep_positions = [
        local_pos for local_pos, kind in slot_info if kind == "explanation"
    ]

    sampling = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "dllm_template_token_ids": padded,
        "dllm_template_position_gates": gates,
        # Global blacklist applied at EVERY mask position (incl. critical
        # values, explanation prose): no token containing `"`, `}`, `\`, etc.
        # Without this, free-text slots emit `","` mid-value and corrupt JSON.
        "dllm_template_forbidden_token_ids": json_bad,
        # Cross-chunk repetition tracking at explanation positions. Each
        # committed token is penalized at remaining explanation masks by
        # rep_penalty * count_so_far. Plus within-step dedup ensures top-K
        # commits in one step pick distinct tokens.
        "dllm_template_rep_penalty_positions": rep_positions,
        "dllm_template_rep_penalty": 2.0,
        # Steps per chunk (caller override → default 4). 4 = fast (1.7-2s)
        # but coarser ADE; 8 = closer to transformers loader's ADE at 2-3s.
        "dllm_template_steps_per_chunk": kwargs.get("steps_per_chunk", 4),
    }
    if chunk_sizes is not None:
        sampling["dllm_template_chunk_sizes"] = chunk_sizes

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
