"""Speculative block decoding for dLLM  (single CUDA-graph, custom-mask).

One CUDA graph is captured (ENCODER_ONLY).  The ragged wrapper is
created with a ``custom_mask_buf`` so the attention pattern can be
switched between bidirectional and causal by writing to the buffer:

  Draft:   mask = all-1  (bidirectional)  →  CUDA Graph replay
  Verify:  mask = tril   (causal / AR)    →  same CUDA Graph replay

After AR verification the longest matching prefix is accepted and
rejected KV slots are freed.
"""

from typing import Optional, Tuple, Union
import logging

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class SpeculativeBlock(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.token_shift = config.algorithm_config.get("token_shift", 1)
        self.debug = config.algorithm_config.get("debug", False)
        self.last_inherited_token = None
        self.last_block_end_position = None

        # Pre-compute causal mask (lower triangular) for the block
        B = self.block_size
        self._causal_mask = torch.tril(
            torch.ones(B, B, dtype=torch.uint8)
        ).flatten()
        self._bidir_mask = torch.ones(B * B, dtype=torch.uint8)

    # ------------------------------------------------------------------
    def _write_mask(self, model_runner: ModelRunner, causal: bool):
        """Write bidirectional or causal mask into the ragged custom_mask buffer."""
        buf = model_runner.attn_backend.dllm_ragged_custom_mask
        if buf is None:
            return
        src = self._causal_mask if causal else self._bidir_mask
        src = src.to(buf.device, non_blocking=True)
        n = src.numel()
        buf[:n].copy_(src, non_blocking=True)

    # ------------------------------------------------------------------
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

        # Template-fill mode: chunk is user-supplied scaffold + interleaved
        # masks. AR-verify would reject scaffold positions (the scaffold is
        # not the model's natural AR continuation), so we accept the full
        # draft. block_start=0 returns all positions to the caller.
        is_template_mode = bool(
            getattr(forward_batch, "dllm_template_modes", None)
            and forward_batch.dllm_template_modes[0]
        )
        if is_template_mode:
            block_start = 0
        else:
            block_start = total_len - num_masked

        # --- detect new request & clear state ---
        is_new_request = False
        if hasattr(forward_batch, "positions") and forward_batch.positions is not None:
            if forward_batch.positions[0] == 0:
                is_new_request = True
                self.last_inherited_token = None
                self.last_block_end_position = None

        # --- place inherited token at position 0 of an all-mask block
        # (skip in template mode — scaffold at position 0 must stay).
        if (
            not is_template_mode
            and forward_batch.input_ids[0] == self.mask_id
            and self.last_inherited_token is not None
            and not is_new_request
        ):
            forward_batch.input_ids[0] = self.last_inherited_token

        # ==============================================================
        # Phase 1 – Draft  (bidirectional mask, CUDA Graph)
        # ==============================================================
        self._write_mask(model_runner, causal=False)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        can_run_cuda_graph = out.can_run_graph
        draft_logits = out.logits_output.full_logits
        assert draft_logits is not None

        # token-shift: shifted[i] predicts token at position i
        if self.token_shift > 0:
            shifted = torch.cat([draft_logits[:1], draft_logits[:-1]], dim=0)
        else:
            shifted = draft_logits

        # Template-fill: apply per-position gates and global forbidden mask
        # so structural slots (digits/signs/verbs) commit to valid tokens and
        # free-text slots don't emit JSON-breaking chars. Without this the
        # single-shot draft picks BPE artifacts like " down" / `,"` / `.0`.
        if is_template_mode:
            chunk_gates = getattr(forward_batch, "dllm_template_chunk_gates", None)
            forbidden_ids = getattr(forward_batch, "dllm_template_forbidden_token_ids", None)
            if chunk_gates is not None or forbidden_ids:
                vocab_size = shifted.shape[-1]
                device = shifted.device
                if forbidden_ids:
                    fmask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                    for bid in forbidden_ids:
                        if 0 <= bid < vocab_size:
                            fmask[bid] = True
                    # Apply to MASK positions only (don't trash committed scaffold)
                    mp = forward_batch.input_ids == self.mask_id
                    shifted[mp.unsqueeze(-1) & fmask.unsqueeze(0)] = float("-inf")
                if chunk_gates is not None:
                    minus_inf = float("-inf")
                    for lp, allowed in enumerate(chunk_gates):
                        if allowed is None or lp >= shifted.shape[0]:
                            continue
                        if forward_batch.input_ids[lp] != self.mask_id:
                            continue
                        keep = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                        for aid in allowed:
                            if 0 <= aid < vocab_size:
                                keep[aid] = True
                        shifted[lp] = torch.where(keep, shifted[lp], torch.tensor(minus_inf, device=device))

        # materialize predictions before the output buffer is overwritten
        draft_preds = shifted.argmax(dim=-1)

        # fill mask positions
        mask_pos = forward_batch.input_ids == self.mask_id
        forward_batch.input_ids[mask_pos] = draft_preds[mask_pos]

        # ==============================================================
        # Phase 2 – Verify  (causal mask, same CUDA Graph)
        # ==============================================================
        self._write_mask(model_runner, causal=True)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output = out.logits_output
        verify_logits = logits_output.full_logits
        assert verify_logits is not None

        ar_tokens = verify_logits.argmax(dim=-1)

        if is_template_mode:
            # In template mode, the user has fixed the scaffold + the draft
            # has filled mask positions. AR-verify would reject scaffold (the
            # scaffold is not the model's natural AR continuation), shrinking
            # the accepted prefix to almost nothing. Skip verify entirely and
            # accept the full draft.
            accepted_num = total_len
        else:
            # --- AR comparison: ar[i] should equal block[i+1] ---
            accepted_num = 0
            for i in range(total_len - 1):
                if ar_tokens[i] == forward_batch.input_ids[i + 1]:
                    accepted_num += 1
                else:
                    break
            accepted_num += 1  # correction / next-token prediction
            accepted_num = min(accepted_num, total_len)

        # --- determine output tokens ---
        if accepted_num >= total_len:
            keep_positions = total_len
            output_count = total_len - block_start
            self.last_inherited_token = ar_tokens[-1].item()
        else:
            keep_positions = accepted_num
            output_count = max(accepted_num - block_start, 0)
            self.last_inherited_token = ar_tokens[accepted_num - 1].item()

        # ==============================================================
        # Phase 3 – KV cache cleanup is handled by the scheduler
        # ==============================================================
        # Do NOT free rejected KV slots here.  The scheduler will either:
        #   - free them via truncation (request continues), or
        #   - free all slots via release_kv_cache (request finishes).
        # Freeing here would cause a double-free.

        # --- restore bidirectional mask for next block's first (draft) call ---
        self._write_mask(model_runner, causal=False)

        # --- position tracking ---
        if hasattr(forward_batch, "positions") and forward_batch.positions is not None:
            end_idx = block_start + output_count - 1 if output_count > 0 else 0
            self.last_block_end_position = forward_batch.positions[end_idx].item()

        next_token_ids = forward_batch.input_ids[block_start : block_start + output_count]

        if self.debug:
            logger.info(
                f"[SpeculativeBlock] total={total_len} blk_start={block_start} "
                f"accepted={accepted_num} output={output_count} "
                f"keep={keep_positions}"
            )

        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = SpeculativeBlock
