# Template-Fill Denoise Algorithm (SGLang Fast-dVLM)

How the V3 JSON template is filled by the diffusion model, served through
the modified NVlabs SGLang fork.

**Current algorithm (since the dDrive port): Fast-dDrive's
`mdm_sample_deep_scaffold` — confidence-threshold commit, ported into
SGLang's `HierarchyBlock` template-mode.** This replaced our earlier
top-K + per-position-gates + rep-penalty fill, which produced word-salad
explanations on the zero-shot model. The dDrive algorithm yields
explanations as clean as the no-SGLang (HF transformers) path.

Section 6 keeps the superseded top-K design for history.

---

## 0. Top-level flow

```
prompt + image → SGLang prefill (1 forward, causal) → KV cache built
                                ↓
template (~470 tokens) split into block_size chunks (section-aligned)
                                ↓
for each chunk: confidence-threshold diffusion fill
   (≤ n_masks+5 forwards; commit conf>0.9 each step, top-1 fallback)
   + 1 final causal commit forward (writes chunk KV, captures prev_last_logit)
                                ↓
return all filled positions, decode to text, parse JSON
```

---

## 1. Loader side (`eval/loaders/fast_dvlm_sglang_v3.py`)

```python
# Build V3 template scaffold + mask positions
template_ids, slot_info, _ = build_template_ids_v3(tokenizer, mask_id)
# kind ∈ {critical_head, critical_tail, complexity, explanation,
#          long_w1, long_w2, lat_w1, lat_w2,
#          traj_sign, traj_tens, traj_ones, traj_frac}

# Inject "navigation_command": "<nav>" just before "future_meta_behavior"
template_ids, slot_info = _inject_nav_into_template(...)

# Section-align padding: each V3 section padded to a multiple of
# block_size (=160) so chunk boundaries fall on section transitions.
padded, slot_info = _section_align_template(template_ids, slot_info, 160, pad=newline)

engine.generate(
    input_ids=prompt_ids,
    image_data=[img],
    sampling_params={
        "max_new_tokens":            len(padded),
        "temperature":               0.0,
        "dllm_template_token_ids":   padded,
        # dDrive algorithm: confidence threshold + (no gates / rep-penalty).
        "dllm_template_threshold":   0.9,   # commit conf>0.9 per step
        "dllm_template_rep_penalty": 0.0,   # OFF (was 2.0; caused BPE glue)
        # Below are still plumbed but IGNORED by the dDrive algorithm path:
        "dllm_template_position_gates":      gates,   # inert
        "dllm_template_forbidden_token_ids": json_bad,  # inert
    },
)
```

Notable settings vs the old path:
- `N_EXPLANATION_TOKENS = 64` (was 100) — shorter explanation, less drift.
- `engine_block_size = 160` — section-aligned chunks (crit+cmplx / expl /
  beh = 1 chunk each, trajectory = 2). Set via
  `algorithm_config["block_size"]` in `load()`.
- gates / forbidden / rep-penalty are no longer used by the algorithm
  (kept in the plumbing for the legacy path / future re-enable).

---

## 2. SGLang server side

### 2.1 Request init — `Req._init_fill_ids_for_dllm()`
- **Round 1**: `dllm_ids = prompt_ids` → causal prompt prefill (1 forward
  over prompt+image, KV cache built).
- **Round 2..N**: append the next `block_size`-sized chunk of
  `template_ids`; record `_dllm_template_chunk_offset`.

### 2.2 ForwardBatch construction
`ScheduleBatch → ModelWorkerBatch → ForwardBatch` propagates:
```
dllm_template_modes        [True]
dllm_template_threshold    0.9
... (gates / forbidden / rep-penalty fields still present but unused)
```

### 2.3 `HierarchyBlock.run()` — dDrive template-mode

Port of `generation_utils.py:mdm_sample_deep_scaffold`. Input:
`forward_batch.input_ids[B]` = current chunk (scaffold + mask interleaved).

```python
threshold = forward_batch.dllm_template_threshold      # 0.9
n_mask = (input_ids == mask_id).sum()
max_iter = n_mask + 5

self._set_attention_type(ENCODER_ONLY)   # bidir within chunk; causal across via KV

for _ in range(max_iter):
    cur_mask = input_ids == mask_id
    if not cur_mask.any():
        break

    full_logits = model_runner.forward(forward_batch).full_logits   # [B, V]

    # OFF-BY-ONE SHIFT (the key fix vs the old path):
    #   position i is predicted by the logit at position i-1.
    #   position 0's predictor is prev_last_logit — the LAST logit of the
    #   PRIOR chunk's final commit forward — NOT this chunk's logit[0]
    #   (which actually predicts position 1).
    if self.prev_last_logit is not None:
        shifted = cat([self.prev_last_logit, full_logits[:-1]], dim=0)
    else:  # very first chunk: approximate with logit[0]
        shifted = cat([full_logits[:1], full_logits[:-1]], dim=0)

    preds = shifted.argmax(-1)
    conf  = softmax(shifted).gather(-1, preds[:,None]).squeeze(-1)
    conf  = where(cur_mask, conf, -inf)          # only still-masked positions

    unmask = conf > threshold
    if unmask.any():
        input_ids[unmask] = preds[unmask]        # commit ALL above threshold
    else:
        best = conf.argmax()                     # fallback: 1 highest-conf
        input_ids[best] = preds[best]

# Final commit forward (causal, DECODER) — writes this chunk's K/V to cache
self._set_attention_type(DECODER)
logits = model_runner.forward_extend(forward_batch).full_logits
self.last_inherited_token = logits[-1].argmax()
self.prev_last_logit = logits[-1:].detach().clone()   # → next chunk's pos-0 predictor
self._set_attention_type(ENCODER_ONLY)                # restore for CUDA graph
```

### 2.4 Cross-request state (cleared when `positions[0] == 0`)
```python
self.last_inherited_token = None
self.prev_last_logit      = None   # [1, V]; dDrive off-by-one predictor
```

---

## 3. Key parameters

| Parameter | Value | Effect |
|---|---|---|
| `block_size` (engine) | 160 | section-aligned chunk size |
| `threshold` | 0.9 | commit positions with conf > 0.9 each step |
| `max_iter` per chunk | n_masks + 5 | safety cap; converges earlier when conf is high |
| `temperature` | 0 | argmax, deterministic |
| `rep_penalty` | 0.0 | OFF (2.0 caused BPE-fragment fallback) |
| per-position gates | OFF | dDrive relies on the model, not vocab masks |
| latency | ~1.2 s/sample | CUDA Graph + flashinfer; 2.7× faster than HF dDrive |

---

## 4. Why dDrive's algorithm fixes explanation quality

Same Fast-dVLM zero-shot weights, two algorithms:

| | old top-K bidir | dDrive threshold |
|---|---|---|
| commit policy | fixed K-per-step by confidence | all conf>0.9, else top-1 |
| pos-0 predictor | chunk's own logit[0] (off-by-one) | prev chunk's last logit (correct) |
| gates/rep-penalty | yes (caused BPE glue / fragment fallback) | none |
| explanation | word-salad, "obant", mode-collapse | full English, scene-grounded |

The threshold policy lets high-confidence positions commit together and
hard ones wait; combined with the correct off-by-one shift, the model
fills each slot the way it would in free-form AR generation.

---

## 5. End-to-end forward count

```
1   prompt + image prefill                (causal, update KV cache)
─────────────────────────────────────────
per chunk: k diffusion forwards (k ≤ n_masks+5, usually << because conf
           clears 0.9 fast) + 1 final causal commit forward
─────────────────────────────────────────
≈ 1.2 s / sample on a single H100 (section-aligned bs=160, ~5 chunks)
```

---

## 6. Superseded design (history): top-K + gates + rep-penalty

The earlier template-mode used a fixed `steps_per_chunk=4` top-K commit
with per-position vocab gates (digits/signs/verb-words), a global JSON
forbidden list, and a cross-step repetition penalty + within-step dedup at
explanation slots. It kept trajectory digits clean (gates) but produced
word-salad explanations because:
- bidir attention to other masks injected noise into each slot's logit;
- top-K committed high-confidence filler ("a"/"the"/",") first, biasing
  the rest;
- rep_penalty=2.0 pushed common words below rare BPE fragments → "obant",
  "trafficnottraffic"; rep_penalty=0 caused "a a a is is" mode collapse.

These knobs are still plumbed through SamplingParams for the legacy path
but are inert under the dDrive algorithm. The trajectory section's digit
gates may be selectively re-enabled later to tighten high-speed-sample
ADE (currently the p90 tail) without touching explanation quality.

---

## 7. Validation (96 stratified Waymo val samples)

`scripts/test_100_ddrive_sglang.py`:
```
errored=0   parse_fail=0   quality-flagged=0
ADE  median 2.18m   mean 5.12m   p90 20.3m
latency 1.20 s/sample
```
Quality auto-checks (non-ASCII >5%, word repeat >25%, BPE run >25 chars,
explanation <30 chars) flagged 0 of 96. High-ADE samples are high-speed
highway trajectory-magnitude errors, not explanation defects.
