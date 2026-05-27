# Template-Fill Denoise Algorithm (SGLang Fast-dVLM)

Complete description of the diffusion algorithm that fills the V3 template
in the modified NVlabs SGLang fork.

## Top-level flow

```
prompt + image → SGLang prefill (1 forward, causal) → KV cache built
                                ↓
template (390–499 tokens) split into 32-token chunks (~13–16 chunks)
                                ↓
for each chunk: bidir-diffusion fill (4 fixed steps + 1 final force-commit)
                                ↓
return all filled positions, decode to text, parse JSON
```

---

## 1. Loader side (`eval/loaders/fast_dvlm_sglang_v3.py`)

```python
# Build template scaffold + mask positions
template_ids, slot_info, _ = build_template_ids_v3(tokenizer, mask_id)
# slot_info: [(local_pos, kind), ...]
# kind ∈ {critical_head, critical_tail, complexity, explanation,
#          long_w1, long_w2, lat_w1, lat_w2,
#          traj_sign, traj_tens, traj_ones, traj_frac}

# Inject "navigation_command": "<nav>"  just before "future_meta_behavior":
template_ids, slot_info = _inject_nav_into_template(...)

# Pad to multiple of block_size=32 with EOS (151643)
while len(template_ids) % 32 != 0:
    template_ids.append(EOS)

# Build per-position gates (vocab allowlists)
gates = [None] * len(template_ids)  # None = unrestricted
for local_pos, kind in slot_info:
    if kind == "traj_sign":  gates[local_pos] = [plus_id, minus_id]
    elif kind in {"traj_tens", "traj_ones", "traj_frac"}: gates[local_pos] = digit_ids
    elif kind == "long_w1":  gates[local_pos] = [speed_id, slow_id, keep_id, stop_id]
    elif kind == "long_w2":  gates[local_pos] = [up_id, down_id, speed_id, now_id]
    elif kind == "lat_w1":   gates[local_pos] = [keep_id, turn_id, change_id]
    elif kind == "lat_w2":   gates[local_pos] = [lane_id, left_id, right_id]
    elif kind == "complexity": gates[local_pos] = [simple_id, complex_id]
    # critical_*, explanation, padding → None

# Global forbidden (applied to every masked position)
json_blacklist = [token_ids containing '"', '{', '}', '\\', backtick, curly quotes]

# Rep-penalty positions (only "explanation" slots)
rep_positions = [pos for pos, kind in slot_info if kind == "explanation"]

# Pass to SGLang engine
engine.generate(
    input_ids=prompt_ids,
    image_data=[img],
    sampling_params={
        "max_new_tokens":   len(template_ids),  # auto-cap; also auto sets ignore_eos=True
        "temperature":      0.0,
        "dllm_template_token_ids":             template_ids,
        "dllm_template_position_gates":        gates,
        "dllm_template_forbidden_token_ids":   json_blacklist,
        "dllm_template_rep_penalty_positions": rep_positions,
        "dllm_template_rep_penalty":           2.0,
        "dllm_template_steps_per_chunk":       4,
    },
)
```

---

## 2. SGLang server side

### 2.1 Request init
`Req._init_fill_ids_for_dllm()`:

- **Round 1**: `dllm_ids = prompt_ids` → triggers causal prompt prefill
  (1 forward over prompt+image, KV cache built)
- **Round 2..N**: split `template_ids` into `block_size=32` chunks; each
  round appends one chunk to `dllm_ids` and records
  `_dllm_template_chunk_offset` (used to slice the gates window)

### 2.2 ForwardBatch construction
`ScheduleBatch → ModelWorkerBatch → ForwardBatch` propagates:

```
dllm_template_modes                       [True]
dllm_template_chunk_gates                 gates[offset : offset+32]
dllm_template_forbidden_token_ids         json_blacklist
dllm_template_rep_penalty_chunk_positions [local positions in this chunk]
dllm_template_rep_penalty                 2.0
dllm_template_steps_per_chunk             4
```

### 2.3 HierarchyBlock.run() — template-mode path

Input: `forward_batch.input_ids[32]` = current chunk (scaffold + mask
interleaved).

```python
# 1. Setup
n_mask_chunk = (input_ids == mask_id).sum()
n_steps = 4
budget = [n_mask_chunk // 4 + (1 if i < n_mask_chunk % 4 else 0)
          for i in range(4)]
# e.g., 8-mask chunk → budget = [2, 2, 2, 2]
#       32-mask chunk → budget = [8, 8, 8, 8]

# Lazy-build masks (once per chunk, vocab_size = 151936)
forbidden_mask:    bool[V]      # True for JSON metacharacters
gate_block_keep:   bool[32, V]  # per-position allowlist (all True if no gate)
gate_block_active: bool[32]     # True for positions WITH a gate

# 2. Iterate 4 diffusion steps
for step in range(4):
    if not (input_ids == mask_id).any():
        break

    # FORWARD with bidirectional (ENCODER_ONLY) attention over the 32-token
    # chunk; attention to prior chunks via KV cache (causal).
    logits = model_runner.forward(forward_batch).logits  # [32, V]

    # Off-by-one shift: shifted[i] predicts the token AT position i
    # (matches Fast-dVLM training's AR-style lm_head loss).
    shifted = torch.cat([logits[:1], logits[:-1]], dim=0)  # [32, V]

    # Apply global forbidden (mask out JSON-breaking tokens)
    shifted[:, forbidden_mask] = -inf

    # Apply per-position gates
    restricted = torch.where(gate_block_keep, shifted, -inf)
    shifted = torch.where(gate_block_active.unsqueeze(-1), restricted, shifted)

    # Apply cross-step REPETITION PENALTY at explanation positions
    # (penalize tokens already committed at any prior explanation slot —
    # prevents "a a a a" / "the the the" cascade on long mask runs).
    if rep_chunk_positions and self.rep_token_counts is not None:
        rep_pos_t = torch.tensor(rep_chunk_positions)
        shifted[rep_pos_t] -= 2.0 * self.rep_token_counts.unsqueeze(0)

    # Sample
    preds = shifted.argmax(dim=-1)               # [32]
    probs = softmax(shifted)                     # [32, V]
    conf  = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)  # [32]
    conf  = where(cur_mask, conf, -inf)          # only consider still-masked

    # Pick top-K by confidence
    k = budget[step]
    topk_idx = torch.topk(conf, k).indices       # [k]

    # WITHIN-STEP DEDUP at explanation positions:
    # If two explanation positions both want the same token this step,
    # the higher-confidence one keeps it; the other gets its 2nd-best
    # token (logits at the row, claimed tokens set to -inf, re-argmax).
    if rep_in_top := [p for p in topk_idx if p in rep_chunk_positions]:
        sort by conf desc
        claimed = set()
        for lp, conf_lp, predicted_tok in sorted_pairs:
            if predicted_tok in claimed:
                row = shifted[lp].clone()
                row[list(claimed)] = -inf
                preds[lp] = row.argmax()
            claimed.add(preds[lp])

    # COMMIT: write predictions back to input_ids at top-K positions
    input_ids[topk_idx] = preds[topk_idx]

    # Update rep_token_counts for newly-committed explanation tokens
    for p in topk_idx if p in rep_chunk_positions:
        self.rep_token_counts[input_ids[p]] += 1.0

# 3. Force-commit any leftover masks (rare; safety net)
if (input_ids == mask_id).any():
    one final forward + argmax over remaining masks, commit all in one shot

# 4. Switch attention to DECODER (causal) for KV cache write
model_runner.forward_extend(forward_batch)
# This commits the now-clean chunk's K/V to the cache so the next chunk
# can attend to it causally.
```

### 2.4 Cross-request state

```python
# Cleared on new request (forward_batch.positions[0] == 0):
self.last_inherited_token = None
self.rep_token_counts     = None  # vocab-size float tensor; accumulates
                                  # across chunks within a request
```

`rep_token_counts` accumulates **over the entire template** (across all
chunks). So if chunk 1 wrote `"the"`, the penalty active in chunk 3
discourages picking `"the"` at explanation positions there too.

---

## 3. Key parameters

| Parameter | Value | Effect |
|---|---|---|
| `block_size` | 32 | tokens per chunk |
| `steps_per_chunk` | 4 | fixed-step budget; 4 forwards per chunk |
| `temperature` | 0 | argmax, deterministic |
| `token_shift` | 1 | model `lm_head` is next-token; shift 1 to align |
| `rep_penalty` | 2.0 | logit subtract per existing commit at rep slots |
| total forwards/sample | 80 | 16 chunks × (4 draft + 1 commit) + 1 prefill |
| ms/forward | ~25 ms | CUDA Graph + flashinfer-accelerated |
| total latency | ~2 s | 80 × 25 ms |

---

## 4. Three layers of constraint

| Mechanism | Where it applies | Why |
|---|---|---|
| **Per-position gates** | structured slots (digits, signs, verb words, complexity) | Prevent BPE-boundary artifacts like `"slow  down"` (double space) or `"00..0"` (digit followed by `.`-bearing token); enforce schema vocabulary |
| **Global forbidden** | every masked position | Protect JSON integrity — no `"`, `}`, `\\`, backtick, curly quotes anywhere |
| **Rep penalty + within-step dedup** | explanation slot positions only | Prevent mode collapse on the 100-mask explanation run (model defaults to filler `"a"`/`"the"`/`","` at high noise); force diverse token choices |

---

## 5. Diffs from the vendored upstream `HierarchyBlock`

NVlabs' original `HierarchyBlock` is "draft + AR verify" — 2 forwards per
block. The template-mode path in our fork modifies this in 6 ways:

1. **Detect template mode** via `forward_batch.dllm_template_modes`
2. **Fixed-step budget** (4 steps) instead of `while sub_mask.sum() > 0`
   — converges in a predictable forward count, not dependent on a
   confidence threshold
3. **`block_start = 0`** in template mode (return the full chunk, not
   just the mask suffix) so scaffold tokens reach the output
4. **Skip AR-token override** at chunk position 0 (template scaffold
   must stay intact; if position 0 is a mask, the model predicts it
   fresh from bidir context)
5. **Apply rep penalty + within-step dedup** at rep-penalty positions
   inside the logits/argmax pipeline
6. **Apply per-position gates + global forbidden** to logits before
   argmax (every step)

Code is at:
`third_party/sglang/python/sglang/srt/dllm/algorithm/hierarchy_block.py`
— the `if is_template_mode:` branch (~200 lines).

---

## 6. End-to-end forward count for one request

```
1   prompt + image prefill        (causal, update KV cache)
─────────────────────────────────────
16  chunks × 4 diffusion drafts   (bidir, ENCODER_ONLY; CUDA-graphed)
16  chunks × 1 commit forward     (causal, DECODER; update KV cache)
─────────────────────────────────────
≈ 80 model forward passes per sample (variable: 65-85 depending on
template padding and early-exit when no masks remain)
```

At ~25 ms/forward, that's ~2 s end-to-end. The 1 commit per chunk is
why we can't go below ~2 s without restructuring the algorithm or
reducing the number of chunks.

---

## 7. Section-aligned decoding (default since v2)

Adopted after benchmarking against Fast-dDrive's `mdm_sample_deep_scaffold`,
which chunks per JSON section instead of per fixed-size window. We
implement a uniform-block variant: each V3 section is padded to a multiple
of `block_size=160` with EOS, so every chunk that SGLang processes contains
tokens from exactly one section.

### Section partition

```
crit_cmplx (140 tokens, 25 masks)  → 1 chunk × 160
explanation (112 tokens, 100 masks) → 1 chunk × 160
future_meta_behavior (24 tokens, 4 masks) → 1 chunk × 160
trajectory (223 tokens, 80 masks) → 2 chunks × 160
─────────────────────────────────────────────────────
Total: 5 chunks (vs 16 in legacy bs=32 path)
```

Scaffold tokens between sections inherit the *next* mask's section so JSON
key/quote groups stay grouped with their value masks.

### Engine block_size

The SGLang scheduler chunks `dllm_template_token_ids` by
`dllm_config.block_size`. We override the default 32 via
`algorithm_config["block_size"] = 160` in `loader.load(engine_block_size=160)`.
This sizes the CUDA-graph input buffer to 160 so 160-token forwards fit.

### Trade-off measured on N=30 stratified Waymo val

| Mode | mean lat | mean ADE | median ADE | p90 ADE |
|---|---:|---:|---:|---:|
| Legacy bs=32 | 1.30s | 5.59m | 2.47m | 11.41m |
| Section bs=160 | 0.69s | 5.25m | 1.61m | 15.97m |

- Latency: 47% reduction (fewer chunks, even though each chunk's forward
  is ~2× more expensive at bs=160 vs bs=32; net wins because chunk count
  drops 16→5).
- Mean ADE: 6% better.
- Median ADE: 35% better — most typical samples improve.
- p90 ADE: 40% worse — hard samples regress (some traj 80-mask chunk
  commits land in less-diverse patterns vs 7-chunk gradual fill).

### Variable per-section chunk sizes (not adopted)

Tried exposing `dllm_template_chunk_sizes: List[int]` so each chunk
matches its section's natural length (no EOS pad). Fails at
`input_buffers.py:156`: SGLang pre-allocates a buffer at engine
`block_size`, so chunks smaller than that buffer trigger a tensor-shape
mismatch. Implementing variable sizes properly needs SGLang changes to
the input-buffer allocation path. Stub left in
`_section_variable_template` (`section_align="variable"`) for future use.

### Knobs

- `loader.load(engine_block_size=160)` — SGLang scheduler chunk size
- `loader.generate(section_align=True, block_size=160)` — match the engine
- `section_align=False, block_size=32` — restore legacy mechanical chunking
