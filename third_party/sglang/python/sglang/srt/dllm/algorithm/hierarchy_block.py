from typing import Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class HierarchyBlock(DllmAlgorithm):
    """Fast dLLM v2 hierarchical block decoding with token inheritance.

    Attention type switching (matches original HF implementation):
      - Prefill (EXTEND mode):          DECODER (causal)   — handled by model default
      - Denoising iterations:            ENCODER_ONLY (bidirectional) — can use CUDA Graph
      - Final forward (KV cache write):  DECODER (causal)   — always eager (no CUDA Graph)
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.sub_block_size = config.algorithm_config.get("sub_block_size", 8)
        self.token_shift = config.algorithm_config.get("token_shift", 1)
        self.debug = config.algorithm_config.get("debug", True)
        self.use_AR_for_first_token = config.algorithm_config.get("use_AR_for_first_token", True)

        self.last_inherited_token = None
        self.last_block_end_position = None
        self.tokenizer = None
        # Cross-chunk repetition counts for template-mode rep-penalty
        # positions. Reset on new request.
        self.rep_token_counts = None
        # Forward counter for benchmarking. Increments every time
        # model_runner.forward is called in template mode.
        self.fwd_count = 0

    @staticmethod
    def _set_attention_type(model_runner: ModelRunner, attn_type: AttentionType):
        """Switch attention type on all language model layers."""
        model = model_runner.model
        # FastDVLMForConditionalGeneration.model is Qwen2Model
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        if layers is None:
            return
        for layer in layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn'):
                layer.self_attn.attn.attn_type = attn_type

    def _decode_tokens(self, token_ids, model_runner):
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                if hasattr(model_runner, 'model_config') and hasattr(model_runner.model_config, 'hf_config'):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_runner.model_config.hf_config._name_or_path,
                        trust_remote_code=True
                    )
            except Exception as e:
                logger.warning(f"[HierarchyBlock] Failed to load tokenizer: {e}")
                self.tokenizer = False

        if self.tokenizer and self.tokenizer is not False:
            try:
                return self.tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception as e:
                return f"<decode_error: {e}>"
        return None

    def _fill_template_l2r(
        self, model_runner, forward_batch, rep_chunk_pos_list,
        rep_penalty, n_steps, chunk_gates, forbidden_ids,
    ):
        """Template-mode chunk fill where the explanation (rep-penalty)
        positions are committed strictly left-to-right, ONE token per forward
        (autoregressive), while structured (non-rep) masks still fill in
        parallel via confidence top-K.

        Committing explanation tokens one-at-a-time, leftmost first, lets each
        token condition on every token already committed to its left, which is
        what removes the parallel-commit BPE-boundary glue ("cyclistsists",
        "becu", "animalscominging") that confidence-ordered top-K produces on a
        long free-text run. Returns can_run_cuda_graph from the last forward.
        """
        device = forward_batch.input_ids.device
        rep_set = set(rep_chunk_pos_list)
        state = {"forbidden": None, "keep": None, "active": None, "can_run": False}

        def _logits():
            self.fwd_count += 1
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            state["can_run"] = out.can_run_graph
            full_logits = out.logits_output.full_logits
            assert full_logits is not None
            if self.token_shift > 0:
                shifted = torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
            else:
                shifted = full_logits
            vocab_size = shifted.shape[-1]
            if forbidden_ids and state["forbidden"] is None:
                fm = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                for bid in forbidden_ids:
                    if 0 <= bid < vocab_size:
                        fm[bid] = True
                state["forbidden"] = fm
            if chunk_gates is not None and state["keep"] is None:
                bs = forward_batch.input_ids.shape[0]
                keep = torch.ones((bs, vocab_size), dtype=torch.bool, device=device)
                active = torch.zeros(bs, dtype=torch.bool, device=device)
                for lp, allowed in enumerate(chunk_gates):
                    if allowed is None or lp >= bs:
                        continue
                    active[lp] = True
                    row = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                    for aid in allowed:
                        if 0 <= aid < vocab_size:
                            row[aid] = True
                    keep[lp] = row
                state["keep"], state["active"] = keep, active
            if state["forbidden"] is not None:
                shifted[:, state["forbidden"]] = float("-inf")
            if state["active"] is not None and state["active"].any():
                block_minf = torch.full_like(shifted, float("-inf"))
                restricted = torch.where(state["keep"], shifted, block_minf)
                shifted = torch.where(state["active"].unsqueeze(-1), restricted, shifted)
            if rep_penalty > 0 and rep_chunk_pos_list:
                if self.rep_token_counts is None:
                    self.rep_token_counts = torch.zeros(
                        vocab_size, dtype=torch.float, device=device,
                    )
                if self.rep_token_counts.numel() == vocab_size:
                    rep_pos_t = torch.tensor(
                        rep_chunk_pos_list, dtype=torch.long, device=device,
                    )
                    shifted[rep_pos_t] -= rep_penalty * self.rep_token_counts.unsqueeze(0)
            return shifted

        def _masked():
            return (forward_batch.input_ids == self.mask_id).nonzero(
                as_tuple=True,
            )[0].tolist()

        # ---- Phase A: structured (non-rep) masks, parallel top-K ----
        struct0 = [p for p in _masked() if p not in rep_set]
        if struct0:
            # ceil so all structured masks are committed within n_steps.
            base = max(1, (len(struct0) + n_steps - 1) // n_steps)
            for _ in range(n_steps):
                struct = [p for p in _masked() if p not in rep_set]
                if not struct:
                    break
                shifted = _logits()
                preds = shifted.argmax(dim=-1)
                probs = F.softmax(shifted, dim=-1)
                conf = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
                struct_t = torch.tensor(struct, dtype=torch.long, device=device)
                kk = min(base, len(struct))
                top = torch.topk(conf[struct_t], kk).indices
                commit = struct_t[top]
                forward_batch.input_ids[commit] = preds[commit]

        # ---- Phase B: explanation (rep) masks, strict L2R, 1 per forward ----
        while True:
            rep_masked = sorted(p for p in _masked() if p in rep_set)
            if not rep_masked:
                break
            leftmost = rep_masked[0]
            shifted = _logits()
            tok = int(shifted[leftmost].argmax().item())
            forward_batch.input_ids[leftmost] = tok
            if (rep_penalty > 0 and self.rep_token_counts is not None
                    and 0 <= tok < self.rep_token_counts.shape[0]):
                self.rep_token_counts[tok] += 1.0

        return state["can_run"]

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        total_len = len(forward_batch.input_ids)
        block_mask = forward_batch.input_ids == self.mask_id
        num_masked = block_mask.sum().item()
        num_sub_blocks = total_len // self.sub_block_size

        # Template-fill mode: the chunk is a user-supplied scaffold (with
        # interleaved mask positions), not an [AR_token, mask*N] block.
        is_template_mode = bool(
            getattr(forward_batch, "dllm_template_modes", None)
            and forward_batch.dllm_template_modes[0]
        )
        if is_template_mode:
            # All chunk positions are NEW response content — scaffold tokens
            # must be returned to the caller (they weren't already added to
            # output_ids during prefill).
            block_start = 0
        else:
            block_start = total_len - num_masked

        # Detect new request (positions start from 0) and clear inheritance
        is_new_request = False
        if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
            if forward_batch.positions[0] == 0:
                is_new_request = True
                self.last_inherited_token = None
                self.last_block_end_position = None
                self.rep_token_counts = None
                self.fwd_count = 0  # reset for benchmarking

        # Handle token inheritance for all-mask blocks (skip in template mode —
        # template scaffold at position 0 must stay intact; if position 0 IS a
        # mask we want the model to predict it fresh from bidir context).
        first_token_is_mask = forward_batch.input_ids[0] == self.mask_id
        if (not is_template_mode
                and first_token_is_mask
                and self.last_inherited_token is not None
                and not is_new_request):
            forward_batch.input_ids[0] = self.last_inherited_token

        # === Denoising iterations: use bidirectional attention ===
        # These can run with CUDA Graph since attention type stays constant.
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)

        # Pre-build keep-mask tensors for any gated chunk positions. We do
        # this once per block (not per inner iteration) since gates are
        # determined by template position, not by current fill state.
        chunk_gates = getattr(forward_batch, "dllm_template_chunk_gates", None)
        forbidden_ids = getattr(forward_batch, "dllm_template_forbidden_token_ids", None)
        # Per-block pre-built (lazy on first forward when we know vocab_size):
        # gate_block_keep: bool tensor [block_size, vocab_size]; rows for
        #                  ungated positions are all True (no restriction).
        # gate_block_active: bool tensor [block_size]; True = position has a gate.
        # forbidden_mask: bool tensor [vocab_size]; True = forbidden everywhere.
        gate_block_keep = None
        gate_block_active = None
        forbidden_mask = None

        # ============================================================
        # TEMPLATE-MODE PATH: fixed-step chunk-level diffusion.
        # Matches the transformers loader: N steps, top-K budget, plus
        # cross-step rep penalty + within-step token dedup at rep positions.
        # Skips sub-block iteration — Fast-dVLM's bidir attention within
        # block already gives all positions full visibility.
        # ============================================================
        if is_template_mode:
            rep_chunk_pos_list = getattr(
                forward_batch, "dllm_template_rep_penalty_chunk_positions", None,
            ) or []
            rep_penalty = getattr(forward_batch, "dllm_template_rep_penalty", 0.0)
            n_steps = max(1, getattr(forward_batch, "dllm_template_steps_per_chunk", 4))
            explanation_l2r = bool(
                getattr(forward_batch, "dllm_template_explanation_l2r", False)
            )

            chunk_mask = forward_batch.input_ids == self.mask_id
            n_mask_chunk = int(chunk_mask.sum().item())

            if n_mask_chunk > 0:
                rep_chunk_pos_set = set(rep_chunk_pos_list)

                if explanation_l2r and rep_chunk_pos_set:
                    # Explanation positions fill strictly L2R, one per forward
                    # (AR); structured masks fill in parallel. Then skip the
                    # parallel budget loop below (budget=[]).
                    can_run_cuda_graph = self._fill_template_l2r(
                        model_runner, forward_batch, rep_chunk_pos_list,
                        rep_penalty, n_steps, chunk_gates, forbidden_ids,
                    )
                    budget = []
                else:
                    # Pre-distribute commit budget across n_steps so all masks
                    # get committed by the last step. Same as transformers loader.
                    base = max(1, n_mask_chunk // n_steps)
                    remainder = n_mask_chunk % n_steps
                    budget = [base + (1 if i < remainder else 0) for i in range(n_steps)]
                    # Trim trailing zero-budget steps.
                    while budget and budget[-1] == 0:
                        budget.pop()

                for step_idx, k in enumerate(budget):
                    cur_mask = forward_batch.input_ids == self.mask_id
                    if not cur_mask.any():
                        break
                    self.fwd_count += 1
                    out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                    logits_output, can_run_cuda_graph = (
                        out.logits_output, out.can_run_graph,
                    )
                    full_logits = logits_output.full_logits
                    assert full_logits is not None

                    if self.token_shift > 0:
                        shifted = torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
                    else:
                        shifted = full_logits

                    vocab_size = shifted.shape[-1]
                    device = shifted.device

                    # Build masks lazily (first step) and reuse across steps.
                    if forbidden_ids and forbidden_mask is None:
                        forbidden_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                        for bid in forbidden_ids:
                            if 0 <= bid < vocab_size:
                                forbidden_mask[bid] = True
                    if chunk_gates is not None and gate_block_keep is None:
                        block_size = forward_batch.input_ids.shape[0]
                        gate_block_keep = torch.ones(
                            (block_size, vocab_size), dtype=torch.bool, device=device,
                        )
                        gate_block_active = torch.zeros(
                            block_size, dtype=torch.bool, device=device,
                        )
                        for lp, allowed in enumerate(chunk_gates):
                            if allowed is None or lp >= block_size:
                                continue
                            gate_block_active[lp] = True
                            keep = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                            for aid in allowed:
                                if 0 <= aid < vocab_size:
                                    keep[aid] = True
                            gate_block_keep[lp] = keep

                    # Apply forbidden mask globally.
                    if forbidden_mask is not None:
                        shifted[:, forbidden_mask] = float("-inf")
                    # Apply per-position gates.
                    if gate_block_active is not None and gate_block_active.any():
                        block_minf = torch.full_like(shifted, float("-inf"))
                        restricted = torch.where(gate_block_keep, shifted, block_minf)
                        shifted = torch.where(
                            gate_block_active.unsqueeze(-1), restricted, shifted,
                        )

                    # Cross-step rep penalty at rep positions.
                    if (rep_penalty > 0 and rep_chunk_pos_list
                            and self.rep_token_counts is not None
                            and self.rep_token_counts.numel() == vocab_size):
                        rep_pos_t = torch.tensor(
                            rep_chunk_pos_list, dtype=torch.long, device=device,
                        )
                        shifted[rep_pos_t] -= rep_penalty * self.rep_token_counts.unsqueeze(0)
                    if (rep_penalty > 0 and rep_chunk_pos_list
                            and self.rep_token_counts is None):
                        self.rep_token_counts = torch.zeros(
                            vocab_size, dtype=torch.float, device=device,
                        )

                    preds = shifted.argmax(dim=-1)
                    probs = F.softmax(shifted, dim=-1)
                    conf = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
                    # Restrict transfer candidates to currently-masked positions.
                    conf = torch.where(
                        cur_mask, conf, torch.tensor(-np.inf, device=device),
                    )
                    k_actual = min(k, int(cur_mask.sum().item()))
                    if k_actual <= 0:
                        continue
                    topk_idx = torch.topk(conf, k_actual).indices

                    # Within-step dedup at rep positions: ensure no two
                    # rep-position commits share the same token in this step.
                    if rep_chunk_pos_set and topk_idx.numel() > 1:
                        rep_in_top = [
                            int(p) for p in topk_idx.tolist()
                            if int(p) in rep_chunk_pos_set
                        ]
                        if len(rep_in_top) > 1:
                            # Sort by confidence descending.
                            rep_in_top.sort(
                                key=lambda lp: -float(conf[lp].item()),
                            )
                            claimed = set()
                            minus_inf = float("-inf")
                            for lp in rep_in_top:
                                cur_tok = int(preds[lp].item())
                                if cur_tok not in claimed:
                                    claimed.add(cur_tok)
                                    continue
                                # Pick the best alternative not in claimed.
                                row = shifted[lp].clone()
                                for t in claimed:
                                    row[t] = minus_inf
                                new_tok = int(row.argmax().item())
                                preds[lp] = new_tok
                                claimed.add(new_tok)

                    # Commit.
                    forward_batch.input_ids[topk_idx] = preds[topk_idx]

                    # Update rep counts for newly-committed rep positions.
                    if rep_penalty > 0 and rep_chunk_pos_set and self.rep_token_counts is not None:
                        for p in topk_idx.tolist():
                            if int(p) in rep_chunk_pos_set:
                                tok = int(forward_batch.input_ids[int(p)].item())
                                if 0 <= tok < self.rep_token_counts.shape[0]:
                                    self.rep_token_counts[tok] += 1.0

                # If any masks remain (shouldn't happen with correct budget),
                # force-commit them in a final pass.
                cur_mask = forward_batch.input_ids == self.mask_id
                if cur_mask.any():
                    self.fwd_count += 1
                    out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                    logits_output, can_run_cuda_graph = (
                        out.logits_output, out.can_run_graph,
                    )
                    full_logits = logits_output.full_logits
                    if self.token_shift > 0:
                        shifted = torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
                    else:
                        shifted = full_logits
                    if forbidden_mask is not None:
                        shifted[:, forbidden_mask] = float("-inf")
                    if gate_block_active is not None and gate_block_active.any():
                        block_minf = torch.full_like(shifted, float("-inf"))
                        restricted = torch.where(gate_block_keep, shifted, block_minf)
                        shifted = torch.where(
                            gate_block_active.unsqueeze(-1), restricted, shifted,
                        )
                    preds = shifted.argmax(dim=-1)
                    forward_batch.input_ids = torch.where(
                        cur_mask, preds, forward_batch.input_ids,
                    )
            else:
                # All scaffold, no masks — single forward to advance KV cache.
                self.fwd_count += 1
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = (
                    out.logits_output, out.can_run_graph,
                )
        else:
            # === Original non-template path: sub-block iteration. ===
            pass

        # Process sub-blocks (non-template path only)
        for sub_idx in range(num_sub_blocks if not is_template_mode else 0):
            rel_start = sub_idx * self.sub_block_size
            rel_end = rel_start + self.sub_block_size

            while True:
                sub_mask = forward_batch.input_ids[rel_start:rel_end] == self.mask_id
                if sub_mask.sum().item() == 0:
                    break

                self.fwd_count += 1
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
                full_logits = logits_output.full_logits
                assert full_logits is not None

                # Token shift: [L0, L0, L1, ..., LN-2]
                if self.token_shift > 0:
                    shifted_full = torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
                else:
                    shifted_full = full_logits

                sub_logits = shifted_full[rel_start:rel_end, :]

                # Apply per-position vocab allowlists (gates) and global
                # forbidden token blacklist. Lazy-build masks once vocab_size
                # is known. Vectorized: one mask op per sub-block, not per
                # position.
                if chunk_gates is not None or forbidden_ids:
                    vocab_size = sub_logits.shape[-1]
                    device = sub_logits.device
                    if forbidden_ids and forbidden_mask is None:
                        forbidden_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                        for bid in forbidden_ids:
                            if 0 <= bid < vocab_size:
                                forbidden_mask[bid] = True
                    if chunk_gates is not None and gate_block_keep is None:
                        # Default: all True (no restriction). Then for gated
                        # positions, replace with the allowlist mask.
                        block_size = forward_batch.input_ids.shape[0]
                        gate_block_keep = torch.ones(
                            (block_size, vocab_size), dtype=torch.bool, device=device,
                        )
                        gate_block_active = torch.zeros(
                            block_size, dtype=torch.bool, device=device,
                        )
                        for lp, allowed in enumerate(chunk_gates):
                            if allowed is None or lp >= block_size:
                                continue
                            gate_block_active[lp] = True
                            keep = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                            for aid in allowed:
                                if 0 <= aid < vocab_size:
                                    keep[aid] = True
                            gate_block_keep[lp] = keep
                    # Global forbidden — applied to ALL sub_logits rows in one op.
                    if forbidden_mask is not None:
                        sub_logits[:, forbidden_mask] = float("-inf")
                    # Per-position allowlist — applied via vectorized where.
                    if gate_block_active is not None:
                        sub_keep = gate_block_keep[rel_start:rel_end]  # [SB, V]
                        sub_active = gate_block_active[rel_start:rel_end]  # [SB]
                        if sub_active.any():
                            # Restrict: where active AND not-keep -> -inf
                            block_minf = torch.full_like(sub_logits, float("-inf"))
                            # Apply only to active positions
                            restricted = torch.where(sub_keep, sub_logits, block_minf)
                            sub_logits = torch.where(
                                sub_active.unsqueeze(-1), restricted, sub_logits,
                            )

                # Compute predictions and confidence
                preds = sub_logits.argmax(dim=-1)
                probs = F.softmax(sub_logits, dim=-1)
                conf = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
                conf = torch.where(sub_mask, conf, torch.tensor(-np.inf, device=conf.device))

                # Confidence-based unmask. In template-fill mode, use top-K
                # commits sized by mask density of the sub-block:
                #   - mostly-masked (free-text like explanation): k = ceil(N/4)
                #     — gradual fill prevents "a-l-l-l-l-l" mode-collapse on
                #     long all-mask runs where adjacent positions land on the
                #     same high-prob filler.
                #   - mostly-committed (sparse-mask in JSON scaffold blocks):
                #     k = ceil(N/2) — aggressive since context is well-defined.
                # Also commit any position with conf > threshold (handles the
                # easy positions in a single pass).
                if is_template_mode:
                    sb_total = sub_mask.shape[0]
                    num_masked_sub = int(sub_mask.sum().item())
                    if num_masked_sub > 0:
                        # Density-aware top-K:
                        #   mostly-masked sub-block (free-text like
                        #     explanation): commit 1 per iteration —
                        #     prevents adjacent positions from landing on
                        #     the same high-prob filler (",", "the", "a")
                        #   moderately-masked: ceil(N/3) — gradual fill
                        #   lightly-masked (JSON scaffold blocks): ceil(N/2)
                        if num_masked_sub > (sb_total * 2 // 3):
                            # Heavily-masked: k=1 to avoid adjacent-position
                            # filler cascade ("a a a a"). Stays at k=1
                            # until enough clean context exists to anchor
                            # diverse predictions.
                            k = 1
                        elif num_masked_sub > (sb_total // 2):
                            # Majority-masked: k=2.
                            k = 2
                        else:
                            # Minority-masked: aggressive.
                            k = max(1, (num_masked_sub + 1) // 2)
                        topk_idx = torch.topk(conf, k).indices
                        unmask = torch.zeros_like(sub_mask)
                        unmask[topk_idx] = True
                        unmask = unmask & sub_mask
                    else:
                        unmask = torch.zeros_like(sub_mask)
                else:
                    unmask = conf > self.threshold
                    if unmask.sum().item() == 0:
                        unmask[conf.argmax()] = True
                    unmask = unmask & sub_mask

                # Update tokens
                forward_batch.input_ids[rel_start:rel_end] = torch.where(
                    unmask, preds, forward_batch.input_ids[rel_start:rel_end]
                )

        # === Final forward: switch to causal attention for KV cache write ===
        # Must bypass CUDA Graph here because attention type changed.
        # Use forward_extend directly (eager mode) instead of forward()
        # which would route through the CUDA Graph runner.
        self._set_attention_type(model_runner, AttentionType.DECODER)

        self.fwd_count += 1
        logits_output = model_runner.forward_extend(forward_batch, pp_proxy_tensors=None)
        if isinstance(logits_output, tuple):
            logits_output = logits_output[0]
        full_logits = logits_output.full_logits

        if full_logits is not None:
            self.last_inherited_token = full_logits[-1].argmax().item()

        # Update position tracking
        if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
            self.last_block_end_position = forward_batch.positions[-1].item()

        # Restore ENCODER_ONLY for next block's denoising iterations
        # so CUDA Graph (captured with ENCODER_ONLY) stays valid.
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)

        # Debug print for benchmarking — counts ALL model_runner.forward calls
        # since the last chunk boundary (printed per chunk).
        if is_template_mode:
            import os as _os
            if _os.environ.get("DLLM_FWD_LOG"):
                print(f"[HierarchyBlock] chunk done — fwd_count={self.fwd_count}", flush=True)

        return logits_output, forward_batch.input_ids[block_start:], can_run_cuda_graph


Algorithm = HierarchyBlock
