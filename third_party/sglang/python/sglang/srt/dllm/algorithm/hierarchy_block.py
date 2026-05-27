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

        # Process sub-blocks
        for sub_idx in range(num_sub_blocks):
            rel_start = sub_idx * self.sub_block_size
            rel_end = rel_start + self.sub_block_size

            while True:
                sub_mask = forward_batch.input_ids[rel_start:rel_end] == self.mask_id
                if sub_mask.sum().item() == 0:
                    break

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

        return logits_output, forward_batch.input_ids[block_start:], can_run_cuda_graph


Algorithm = HierarchyBlock
