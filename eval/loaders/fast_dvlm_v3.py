"""Fast-dVLM-3B with V3 templated constrained fill (block-causal path).

Processes the prompt+template block-by-block at the model's trained block_size
(32). For each template block, multi-step bidirectional diffusion refines
the predictions before committing the clean block to KV cache.

NOTE on the off-by-one shift used in the model's native `generate()`:
  Native generate uses `[logits[:, :1], logits[:, :-1]]` to shift logits right
  by 1 (so position i's prediction comes from causal-style "next-token" at
  position i-1). For our use case — bidirectional diffusion fill of a fixed
  template — we do NOT apply this shift. The unshifted MLM-style mapping
  (`logits[i]` predicts token at position `i`) is what gives distinct
  predictions across consecutive mask positions; applying the shift forces
  adjacent mask positions to share the same logit row, producing duplicated
  tokens like `" car car"` / `" on on"`.

V3 features ported from `diffusionvl_v3.py`:
  - per-position vocab gates (digit/sign for trajectory, verb word lists)
  - JSON-meta blacklist (Unicode-aware + backtick)
  - EOS-on-filler at critical_head/critical_tail (only at commit time)
  - Tail-EOS pre-fill (when head commits to terminator, force tails to EOS)
"""
from __future__ import annotations

import os
import sys
import time
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from template_v3 import build_template_ids_v3

DEFAULT_PATH = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"
MASK_ID = 151665
EOS_TOKEN_ID = 151643  # <|endoftext|>
SOFT_END_SLOT_KINDS = ("critical_head", "critical_tail")
# Slots where we apply a repetition penalty (suppress duplicates across the
# entire slot). Explanation has 100 mask positions and is prone to "a a a"
# mode collapse during high-noise diffusion; penalty forces diversity.
REPETITION_PENALTY_KINDS = ("explanation",)
REPETITION_PENALTY = 2.0  # subtracted from logit per already-committed copy
                          # (applied per-occurrence so first repeat is gentle,
                          # 5+ repeats fully suppressed)

_FORBIDDEN_TOKEN_IDS = (
    151644, 151645, 151646, 151647,
    151648, 151649, 151650, 151651, 151652, 151653,
    151654, 151655, 151656, 151657, 151658, 151659,
    151660, 151661, 151662, 151663, 151664,
    151665, 151666, 151667, 151668, 151669, 151670, 151671,
)


def _to_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _build_json_special_blacklist(tokenizer):
    vocab = tokenizer.get_vocab()
    bad = set()
    bad_chars = ('"', '{', '}', '\\',
                 '“', '”', '‘', '’',
                 '`')
    for tok_str, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            continue
        if any(c in decoded for c in bad_chars):
            bad.add(tok_id)
    return sorted(bad)


def _build_filler_token_set(tokenizer):
    vocab = tokenizer.get_vocab()
    fillers = set()
    filler_chars = set(",;.-_ \n\t")
    for tok_str, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            continue
        if not decoded:
            fillers.add(tok_id)
            continue
        stripped = decoded.strip()
        if not stripped:
            fillers.add(tok_id)
        elif all(c in filler_chars for c in decoded):
            fillers.add(tok_id)
    return fillers


def _build_terminator_token_set(tokenizer):
    vocab = tokenizer.get_vocab()
    terms = set()
    target = {"none", "absent", "n/a", "nil"}
    for tok_str, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            continue
        if decoded.strip().lower() in target:
            terms.add(tok_id)
    return terms


def _build_v3_vocab_gates(tokenizer):
    def first_tok(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else -1

    def no_space_first_toks(words):
        return sorted({first_tok(w) for w in words if first_tok(w) >= 0})

    digit_ids = sorted({first_tok(c) for c in "0123456789" if first_tok(c) >= 0})
    sign_ids  = sorted({first_tok(c) for c in "+-" if first_tok(c) >= 0})

    long_w1_words = ["speed", "slow", "keep", "stop"]
    long_w2_words = ["up", "down", "speed", "now"]
    lat_w1_words = ["keep", "turn", "change"]
    lat_w2_words = ["lane", "left", "right"]

    return {
        "traj_sign":  sign_ids,
        "traj_tens":  digit_ids,
        "traj_ones":  digit_ids,
        "traj_frac":  digit_ids,
        "long_w1":    no_space_first_toks(long_w1_words),
        "long_w2":    no_space_first_toks(long_w2_words),
        "lat_w1":     no_space_first_toks(lat_w1_words),
        "lat_w2":     no_space_first_toks(lat_w2_words),
    }


def load(model_path=None, device="cuda", dtype="bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    path = model_path or DEFAULT_PATH
    torch_dtype = _to_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    processor = AutoProcessor.from_pretrained(path, use_fast=False)
    processor.tokenizer = tokenizer
    json_blacklist = _build_json_special_blacklist(tokenizer)
    filler_token_set = _build_filler_token_set(tokenizer)
    terminator_token_set = _build_terminator_token_set(tokenizer)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "device": device,
        "dtype": torch_dtype,
        "mask_id": MASK_ID,
        "json_blacklist": json_blacklist,
        "filler_token_set": filler_token_set,
        "terminator_token_set": terminator_token_set,
    }


def _strip_template_from_prompt(prompt: str) -> str:
    i = prompt.find("TEMPLATE (")
    return prompt[:i].rstrip() if i > 0 else prompt


@torch.no_grad()
def _fastdvlm_block_fill(
    model,
    prompt_input_ids: torch.LongTensor,
    template_ids: torch.LongTensor,
    pixel_values,
    image_grid_thw,
    pos_allowlists: dict,
    soft_end_positions: set,
    critical_pairs_global,
    rep_penalty_positions: set,  # global positions where repetition penalty applies
    mask_id: int,
    terminator_mask: torch.BoolTensor,
    filler_mask: torch.BoolTensor,
    forbidden_mask: torch.BoolTensor,
    block_size: int = 32,
    steps_per_block: int = 8,
    temperature: float = 0.0,
    fwd_counter=None,
):
    """Block-by-block diffusion fill. NO logit shift in the draft (see header)."""
    device = prompt_input_ids.device
    vocab_size = forbidden_mask.shape[0]
    minus_inf = torch.tensor(-float("inf"), device=device)
    eos_prefill_enabled = os.environ.get("EOS_PREFILL", "1") != "0"

    # === Step 1: prompt prefill (causal, update KV cache) ===
    prefill_out = model.forward(
        input_ids=prompt_input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=True,
        update_kv_cache=True,
    )
    if fwd_counter is not None:
        fwd_counter[0] += 1
    past_key_values = prefill_out.past_key_values
    L_prompt = prompt_input_ids.shape[1]
    L_template = template_ids.shape[1]

    full_ids = torch.cat([prompt_input_ids, template_ids], dim=1)

    pos_gate_keep = {}
    for gpos, allow_ids in pos_allowlists.items():
        keep = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for aid in allow_ids:
            if 0 <= aid < vocab_size:
                keep[aid] = True
        pos_gate_keep[gpos] = keep

    head_to_tails = {h: tails for h, tails in critical_pairs_global}
    tail_to_head = {}
    for h, tails in critical_pairs_global:
        for t in tails:
            tail_to_head[t] = h

    # Per-token global tally of how many times this token has been committed
    # to a repetition-penalty slot (across the whole template, not per block,
    # so a single "a" anywhere in explanation suppresses "a" everywhere else).
    rep_token_counts = torch.zeros(vocab_size, dtype=torch.float, device=device)

    n_blocks = (L_template + block_size - 1) // block_size
    for block_idx in range(n_blocks):
        b_start_local = block_idx * block_size
        b_end_local = min(b_start_local + block_size, L_template)
        b_len = b_end_local - b_start_local
        b_start_global = L_prompt + b_start_local
        b_end_global = L_prompt + b_end_local

        block_ids = full_ids[:, b_start_global:b_end_global].clone()

        # === Pre-compute per-block lookups (once per block, not per step) ===
        # Vectorized membership of slot kinds within this block.
        rep_local_list = [lp for lp in range(b_len)
                          if (b_start_global + lp) in rep_penalty_positions]
        rep_local_positions = torch.tensor(rep_local_list, dtype=torch.long, device=device) \
            if rep_local_list else torch.empty(0, dtype=torch.long, device=device)
        rep_local_pyset = set(rep_local_list)

        # Gated positions: stack into [n_gated] tensor + [n_gated, V] keep mask
        gated_positions_list = []
        gated_keep_stack = []
        for lp in range(b_len):
            keep = pos_gate_keep.get(b_start_global + lp)
            if keep is not None:
                gated_positions_list.append(lp)
                gated_keep_stack.append(keep)
        if gated_positions_list:
            gated_positions = torch.tensor(gated_positions_list, dtype=torch.long, device=device)
            gated_keep = torch.stack(gated_keep_stack, dim=0)  # [n_gated, V]
        else:
            gated_positions = torch.empty(0, dtype=torch.long, device=device)
            gated_keep = None

        # Soft-end positions in this block as a tensor
        soft_end_local_list = [lp for lp in range(b_len)
                               if (b_start_global + lp) in soft_end_positions]
        soft_end_local = torch.tensor(soft_end_local_list, dtype=torch.long, device=device) \
            if soft_end_local_list else torch.empty(0, dtype=torch.long, device=device)

        # Same-block head→tail pairs (for within-step pre-fill)
        same_block_head_tails = []
        for lp in range(b_len):
            gp = b_start_global + lp
            tails = head_to_tails.get(gp)
            if not tails:
                continue
            for t in tails:
                if b_start_global <= t < b_end_global:
                    same_block_head_tails.append((lp, t - b_start_global))

        # Adaptive steps based on block content:
        #   - explanation block (has rep-penalty positions): 4 steps for refinement
        #   - structured block (has per-position gate: trajectory/behavior): 2 steps
        #   - free-text block (critical_objects: 2-mask phrases): 1 step is enough
        has_gates_in_block = any(
            (b_start_global + lp) in pos_gate_keep for lp in range(b_len)
        )
        if rep_local_positions.numel() > 0:
            block_steps = max(steps_per_block, 4)
        elif has_gates_in_block:
            block_steps = max(steps_per_block, 2)
        else:
            block_steps = 1

        # Cross-block tail pre-fill: heads in earlier blocks committed to terminator
        if eos_prefill_enabled:
            for local_pos in range(b_len):
                gpos = b_start_global + local_pos
                if block_ids[0, local_pos] != mask_id:
                    continue
                head_g = tail_to_head.get(gpos)
                if head_g is None or head_g >= b_start_global:
                    continue
                head_id = int(full_ids[0, head_g].item())
                if head_id == mask_id:
                    continue
                if bool(terminator_mask[head_id].item()):
                    block_ids[0, local_pos] = EOS_TOKEN_ID
                    if fwd_counter is not None and len(fwd_counter) > 1:
                        fwd_counter[1] += 1

        is_mask_block = (block_ids[0] == mask_id)
        n_mask_block = int(is_mask_block.sum().item())
        if n_mask_block > 0:
            base = max(1, n_mask_block // block_steps)
            remainder = n_mask_block % block_steps
            budget = [base + (1 if i < remainder else 0) for i in range(block_steps)]
        else:
            budget = []

        for step in range(block_steps):
            if not bool(is_mask_block.any().item()):
                break
            out = model.forward(
                input_ids=block_ids,
                use_cache=True,
                past_key_values=past_key_values,
                update_kv_cache=False,
                eval_bd_size=block_size,
            )
            if fwd_counter is not None:
                fwd_counter[0] += 1
            logits = out.logits[0].float()
            # Off-by-one shift to match native generate's convention. The
            # model's lm_head was trained AR-style (next-token), so logits
            # at position i predict position i+1; shifting right makes
            # `new[i]` predict token at position i for i>=1. Position 0
            # uses its own logits (degenerate but unavoidable without prior
            # context from KV cache).
            logits = torch.cat([logits[:1, :], logits[:-1, :]], dim=0)

            logits[:, forbidden_mask] = minus_inf
            # Vectorized gate application: forbid any token NOT in keep mask
            # at each gated position (only if still masked).
            if gated_positions.numel() > 0:
                gated_is_mask = is_mask_block[gated_positions]  # [n_gated]
                if gated_is_mask.any():
                    # apply ~keep -> -inf for masked gated positions
                    rows = logits[gated_positions]  # [n_gated, V]
                    rows = torch.where(gated_keep, rows, minus_inf)
                    # only overwrite the still-masked rows
                    rows = torch.where(
                        gated_is_mask.unsqueeze(-1),
                        rows,
                        logits[gated_positions],
                    )
                    logits[gated_positions] = rows

            # === Repetition penalty (global, across template) for slots in
            # `rep_penalty_positions`. Subtract REPETITION_PENALTY * count
            # from each token's logit at those positions. Prevents "a a a a"
            # cascade once "a" is committed once. ===
            if rep_local_positions.numel() > 0 and rep_token_counts.max() > 0:
                logits[rep_local_positions] -= REPETITION_PENALTY * rep_token_counts.unsqueeze(0)

            if temperature <= 0:
                argmax = logits.argmax(dim=-1)
                probs = F.softmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / max(temperature, 1e-3), dim=-1)
                argmax = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(is_mask_block, conf, torch.full_like(conf, float("-inf")))
            k = min(budget[step], int(is_mask_block.sum().item()))
            if k <= 0:
                break
            _, top_local = torch.topk(conf, k)
            top_local_set = set(top_local.tolist())

            # === Within-step uniqueness at rep-penalty positions ===
            # Use pyset for O(1) membership; do a single tolist() instead of
            # nested .item() calls. Build sorted list on CPU then dedup.
            if rep_local_pyset:
                top_list = top_local.tolist()
                rep_in_top = [lp for lp in top_list if lp in rep_local_pyset]
                if len(rep_in_top) > 1:
                    # Fetch confs for sorting in one go
                    rep_conf = conf[rep_in_top].tolist()
                    rep_argmax = argmax[rep_in_top].tolist()
                    pairs = sorted(
                        zip(rep_in_top, rep_conf, rep_argmax),
                        key=lambda x: -x[1],
                    )
                    claimed = set()
                    for lp, _, ag in pairs:
                        if ag not in claimed:
                            claimed.add(ag)
                            continue
                        row = logits[lp].clone()
                        for t in claimed:
                            row[t] = minus_inf
                        new_arg = int(row.argmax().item())
                        argmax[lp] = new_arg
                        claimed.add(new_arg)

            # Soft-end EOS swap (vectorized over soft_end positions)
            if soft_end_local.numel() > 0:
                # only swap if the position is in top_local AND its argmax is filler
                in_top = torch.zeros(b_len, dtype=torch.bool, device=device)
                in_top[top_local] = True
                soft_in_top = in_top[soft_end_local]
                soft_argmax = argmax[soft_end_local]
                is_filler_at_soft = filler_mask[soft_argmax]
                to_swap = soft_in_top & is_filler_at_soft
                if to_swap.any():
                    swap_positions = soft_end_local[to_swap]
                    argmax[swap_positions] = EOS_TOKEN_ID

            block_ids[0, top_local] = argmax[top_local]
            is_mask_block[top_local] = False

            # Update repetition counts: vectorized intersection of top_local
            # and rep_local_positions.
            if rep_local_positions.numel() > 0:
                in_top_mask = torch.zeros(b_len, dtype=torch.bool, device=device)
                in_top_mask[top_local] = True
                committed_rep_mask = in_top_mask[rep_local_positions]
                if committed_rep_mask.any():
                    committed_rep_pos = rep_local_positions[committed_rep_mask]
                    committed_tokens = block_ids[0, committed_rep_pos]
                    rep_token_counts.scatter_add_(
                        0, committed_tokens,
                        torch.ones_like(committed_tokens, dtype=torch.float),
                    )

            # Within-step head→tail pre-fill (same block). Use pre-computed
            # same_block_head_tails list; only iterate pairs where head is
            # in top_local AND head's committed token is a terminator.
            if eos_prefill_enabled and same_block_head_tails:
                in_top = torch.zeros(b_len, dtype=torch.bool, device=device)
                in_top[top_local] = True
                for h_local, t_local in same_block_head_tails:
                    if not in_top[h_local]:
                        continue
                    head_id = int(block_ids[0, h_local].item())
                    if not bool(terminator_mask[head_id].item()):
                        continue
                    if not is_mask_block[t_local]:
                        continue
                    block_ids[0, t_local] = EOS_TOKEN_ID
                    is_mask_block[t_local] = False
                    if fwd_counter is not None and len(fwd_counter) > 1:
                        fwd_counter[1] += 1

        full_ids[:, b_start_global:b_end_global] = block_ids
        # Commit forward: only needed if a later block will reference this
        # one's content via KV cache. The very last template block has no
        # follower, so skip its commit. Saves 1 fwd.
        if block_idx < n_blocks - 1:
            commit_out = model.forward(
                input_ids=block_ids,
                use_cache=True,
                past_key_values=past_key_values,
                update_kv_cache=True,
                eval_bd_size=block_size,
            )
            if fwd_counter is not None:
                fwd_counter[0] += 1
            past_key_values = commit_out.past_key_values

    return full_ids[:, L_prompt:L_prompt + L_template]


def generate(bundle, image_paths, question, gen_length=512, steps=8, temperature=0.0,
              block_size=32):
    from qwen_vl_utils import process_vision_info
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    processor = bundle["processor"]
    device = bundle["device"]
    dtype = bundle["dtype"]
    mask_id = bundle["mask_id"]

    user_text = _strip_template_from_prompt(question).replace("<image>", "").strip()
    images = [Image.open(p).convert("RGB") for p in image_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)
    prompt_input_ids = inputs.input_ids
    pixel_values = inputs.get("pixel_values")
    if torch.is_tensor(pixel_values):
        pixel_values = pixel_values.to(dtype)
    image_grid_thw = inputs.get("image_grid_thw")
    L_prompt = prompt_input_ids.shape[1]

    template_ids_list, slot_info, critical_pairs_local = build_template_ids_v3(tokenizer, mask_id)
    template_ids = torch.tensor(template_ids_list, dtype=torch.long, device=device).unsqueeze(0)

    gates = _build_v3_vocab_gates(tokenizer)
    pos_allowlists = {}
    soft_end_positions = set()
    rep_penalty_positions = set()
    for local_pos, kind in slot_info:
        gpos = L_prompt + local_pos
        if kind in gates:
            pos_allowlists[gpos] = gates[kind]
        if kind in SOFT_END_SLOT_KINDS:
            soft_end_positions.add(gpos)
        if kind in REPETITION_PENALTY_KINDS:
            rep_penalty_positions.add(gpos)
    critical_pairs_global = [
        (L_prompt + h, [L_prompt + t for t in ts])
        for h, ts in critical_pairs_local
    ]

    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.weight.shape[0]
    forbidden_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    forbidden_mask[mask_id] = True
    for bid in _FORBIDDEN_TOKEN_IDS:
        if 0 <= bid < vocab_size:
            forbidden_mask[bid] = True
    for bid in bundle.get("json_blacklist") or []:
        if 0 <= bid < vocab_size:
            forbidden_mask[bid] = True

    filler_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for fid in bundle.get("filler_token_set") or set():
        if 0 <= fid < vocab_size:
            filler_mask[fid] = True

    terminator_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tid in bundle.get("terminator_token_set") or set():
        if 0 <= tid < vocab_size:
            terminator_mask[tid] = True

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    fwd_counter = [0, 0]
    filled = _fastdvlm_block_fill(
        model=model,
        prompt_input_ids=prompt_input_ids,
        template_ids=template_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pos_allowlists=pos_allowlists,
        soft_end_positions=soft_end_positions,
        critical_pairs_global=critical_pairs_global,
        rep_penalty_positions=rep_penalty_positions,
        mask_id=mask_id,
        terminator_mask=terminator_mask,
        filler_mask=filler_mask,
        forbidden_mask=forbidden_mask,
        block_size=block_size,
        steps_per_block=steps,
        temperature=temperature,
        fwd_counter=fwd_counter,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency = time.time() - t0
    print(f"[fast_dvlm_v3] {fwd_counter[0]} fill forwards "
          f"(prefill_triggered={fwd_counter[1]}), {latency:.2f}s")

    text_out = tokenizer.decode(filled[0].tolist(), skip_special_tokens=True)
    return text_out, latency
