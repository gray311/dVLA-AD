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
        # Fast-dDrive `mdm_sample_deep_scaffold` style: the logit at the LAST
        # position of the prior chunk's final-commit forward. Used as the
        # predictor for position 0 of the next chunk (proper off-by-one
        # alignment for next-token AR-style lm_head). Reset on new request.
        self.prev_last_logit = None  # [1, V] tensor or None
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
                self._logged_mode_this_req = False
                self.prev_last_logit = None  # reset for dDrive off-by-one shift

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
        # TEMPLATE-MODE PATH: Fast-dDrive `mdm_sample_deep_scaffold`
        # algorithm (port of generation_utils.py:mdm_sample_deep_scaffold).
        #
        # Per chunk, iterate up to (n_masks + 5) times:
        #   1. Bidir forward over current chunk (encoder-only attention).
        #   2. Token shift: position i predicted by logit at position i-1.
        #      Position 0's predictor = self.prev_last_logit (captured from
        #      prior chunk's final commit forward). For the very first chunk
        #      we fall back to full_logits[:1] (slight off-by-one, but
        #      position 0 of first chunk is usually scaffold not mask).
        #   3. Argmax + softmax confidence at the shifted-logit row.
        #   4. Commit positions whose conf > threshold (default 0.9).
        #   5. If none cleared the threshold, fallback: commit the single
        #      highest-conf masked position.
        #
        # After the per-chunk loop, the final commit forward (below) writes
        # this chunk's K/V to cache AND captures self.prev_last_logit for
        # the next chunk.
        #
        # NO per-position gates / forbidden / rep-penalty / dedup — those
        # are our additions that don't match dDrive's recipe. To match
        # dDrive's "explanation as clean as no-template" behavior we use the
        # clean algorithm.
        # ============================================================
        if is_template_mode:
            import os as _os
            threshold = float(getattr(forward_batch, "dllm_template_threshold", 0.9) or 0.9)
            rep_penalty = float(
                getattr(forward_batch, "dllm_template_rep_penalty", 0.0) or 0.0
            )
            if _os.environ.get("DLLM_FWD_LOG"):
                if not getattr(self, "_logged_mode_this_req", False):
                    print(
                        f"[HierarchyBlock] template-mode path: dDrive mdm "
                        f"threshold={threshold}",
                        flush=True,
                    )
                    self._logged_mode_this_req = True

            chunk_mask = forward_batch.input_ids == self.mask_id
            n_mask_chunk = int(chunk_mask.sum().item())

            # Minimal forbidden list (JSON-meta + newline/tab). Applied at
            # EVERY mask commit so the model can't (a) break JSON with a
            # quote/brace, or (b) "give up" early by padding the explanation
            # tail with newline tokens. Built lazily once vocab_size is known.
            forbidden_ids = getattr(forward_batch, "dllm_template_forbidden_token_ids", None)
            forbidden_mask = None

            if n_mask_chunk > 0:
                max_iter = n_mask_chunk + 5
                for _it in range(max_iter):
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
                    # [B, V]

                    # Proper off-by-one shift: position i's predictor is the
                    # logit at position i-1. For position 0, that's the LAST
                    # logit from the prior chunk's final commit forward,
                    # stored in self.prev_last_logit. If unavailable (first
                    # chunk), use this chunk's full_logits[:1] as
                    # approximation.
                    if self.prev_last_logit is not None:
                        shifted = torch.cat(
                            [self.prev_last_logit.to(full_logits.device,
                                                     dtype=full_logits.dtype),
                             full_logits[:-1]],
                            dim=0,
                        )
                    elif self.token_shift > 0:
                        shifted = torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
                    else:
                        shifted = full_logits

                    # Apply minimal forbidden mask (JSON-meta + newline/tab) at
                    # all positions — keeps JSON valid and prevents newline
                    # "give-up" padding of the explanation tail.
                    if forbidden_ids:
                        if forbidden_mask is None:
                            vocab_size = shifted.shape[-1]
                            forbidden_mask = torch.zeros(
                                vocab_size, dtype=torch.bool, device=shifted.device,
                            )
                            for bid in forbidden_ids:
                                if 0 <= bid < vocab_size:
                                    forbidden_mask[bid] = True
                        shifted[:, forbidden_mask] = float("-inf")

                    # Repetition penalty: discourage each position from copying
                    # the committed token 1 or 2 slots to its left. Diffusion
                    # template-fill has a strong immediate-repeat tendency that
                    # survives neighbor-deferral ("straight straight", "the the",
                    # ",,", "::", "ing"+"ing", "Predict"+"Predict") because the
                    # off-by-one predictor at the fixed left neighbor still puts
                    # high mass on repeating it. We subtract a soft penalty from
                    # the left-neighbor token id(s) at each position. Only
                    # COMMITTED neighbors are penalized (skip mask_id); the
                    # penalty is soft so a strongly-justified repeat (e.g. a
                    # trajectory "00") can still win.
                    if rep_penalty > 0.0:
                        ids_row = forward_batch.input_ids  # [B]
                        B = shifted.shape[0]
                        rows = torch.arange(B, device=shifted.device)
                        for off, scale in ((1, 1.0), (2, 0.5)):
                            if B > off:
                                neigh = torch.cat(
                                    [ids_row[:off], ids_row[:-off]], dim=0,
                                )  # token at position i-off
                                valid = neigh != self.mask_id
                                shifted[rows[valid], neigh[valid]] -= rep_penalty * scale

                    # Argmax + confidence at each position
                    preds = shifted.argmax(dim=-1)  # [B]
                    probs = F.softmax(shifted, dim=-1)
                    conf = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
                    # Mask out positions that are no longer masked.
                    conf = torch.where(
                        cur_mask, conf, torch.tensor(-np.inf, device=conf.device),
                    )

                    # Commit positions with conf > threshold — but NEVER commit
                    # two CONSECUTIVE mask positions in the same step. Diffusion
                    # predicts each mask independently from the same context, so
                    # finalizing adjacent masks together produces doubling
                    # artifacts ("keepkeep", "PPredict", "straight straight").
                    # Deferring the right member of each adjacent run lets it
                    # re-predict next step with its left neighbor now fixed in
                    # context, which removes the doubling. A scaffold token
                    # between two masks separates their positions by >=2, so
                    # legitimate adjacent slots (e.g. trajectory digits) still
                    # both commit — they just take one extra iteration. Progress
                    # is guaranteed: the first candidate is always kept, so >=1
                    # mask commits per step (or 1 via the fallback below).
                    unmask = conf > threshold
                    if unmask.any():
                        idxs = torch.nonzero(unmask, as_tuple=False).flatten().tolist()
                        keep, prev = [], -2
                        for p in idxs:  # ascending position order
                            if p == prev + 1:
                                continue  # adjacent to a kept commit → defer
                            keep.append(p)
                            prev = p
                        keep_t = torch.tensor(
                            keep, dtype=torch.long,
                            device=forward_batch.input_ids.device,
                        )
                        forward_batch.input_ids[keep_t] = preds[keep_t]
                    else:
                        # Fallback: commit single highest-conf masked position.
                        best_pos = int(conf.argmax().item())
                        forward_batch.input_ids[best_pos] = int(preds[best_pos].item())
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
            # dDrive-style: capture the last position's logit for the next
            # chunk's position-0 prediction (proper off-by-one alignment).
            # detach() so we don't keep gradient state across forwards.
            if is_template_mode:
                self.prev_last_logit = full_logits[-1:].detach().clone()

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
