# dVLA-AD

Fast-dVLM zero-shot driving CoT with V3 template fill, served via a modified
NVlabs Fast-dLLM SGLang fork.

## What's in here

```
eval/
  template_v3.py                     V3 schema: 12 critical_objects × 2 mask,
                                     100 explanation mask, 2+2 behavior verb
                                     mask, 10wp × 8 trajectory mask
  loaders/
    fast_dvlm_v3.py                  transformers path (block-causal fill
                                     written from scratch, ~1.3s/sample)
    fast_dvlm_sglang.py              SGLang free-form (no template, JSON
                                     generated as continuation, ~1.2s)
    fast_dvlm_sglang_v3.py           SGLang template-fill (uses modified
                                     SGLang fork APIs, ~2.0s)

scripts/
  smoke_sglang_v3.py                 1-sample sanity check
  run_5_waymo_sglang_v3.py           5-sample Waymo eval (mdm|spec)
  run_5_waymo_sglang.py              free-form baseline runner
  run_5_waymo_fastdvlm.py            transformers baseline runner
  write_template_vs_freeform_report.py    quality + latency comparison

third_party/sglang/                  NVlabs vendored SGLang fork
                                     (modified — see "Modifications" below)

results/waymo_5_compare/             benchmark outputs and reports
```

## Setup

```bash
# In an env with torch 2.9.x and Qwen2.5-VL deps:
pip install -e third_party/sglang/python --no-deps

# Sanity check:
python scripts/smoke_sglang_v3.py mdm

# Full 5-sample run:
python scripts/run_5_waymo_sglang_v3.py mdm
```

## SGLang fork modifications

The vendored SGLang fork adds three new `SamplingParams` fields for
template-fill of structured outputs:

### `dllm_template_token_ids: List[int]`

Response scaffold with `mask_id` at fill slots. When set, the engine feeds
this token sequence to the dllm algorithm in `block_size`-sized chunks
instead of auto-generating fresh `[mask] * block_size` blocks. Scaffold
positions stay intact across diffusion (only `== mask_id` positions get
refined). `max_new_tokens` is automatically capped to `len(template)`, and
`ignore_eos=True` is forced so soft-end EOS commits don't truncate the
response.

### `dllm_template_position_gates: List[Optional[List[int]]]`

Per-position vocab allowlists. Same length as `dllm_template_token_ids`.
`None` at a position = unrestricted; a list at a position restricts that
position's prediction to those token ids. Used for structured slots
(trajectory `+XX.X,+YY.Y` digits/signs, behavior verb words) to avoid
BPE-boundary artifacts like `"slow  down"` (double space) or `"00..0"`
(digit followed by `.`-bearing token).

### `dllm_template_forbidden_token_ids: List[int]`

Global blacklist applied at every MASKED position. Used for JSON-meta
chars (`"`, `}`, `\`, backtick) so even free-text slots like
`critical_object` values can't emit JSON-breaking tokens.

### Algorithm changes

`HierarchyBlock` (MDM) — block-by-block diffusion with template support:
- bypass AR-token override at chunk position 0 (template scaffold must
  stay intact)
- return ALL block positions (not just mask suffix) so scaffold tokens
  reach the output
- apply gates + forbidden masks before argmax/topk
- **3-tier density-aware top-K**: k=1 for >2/3 masked sub-blocks (prevents
  filler cascade on long explanation runs), k=2 for >1/2 masked, k=N/2
  otherwise

`SpeculativeBlock` — same template handling; AR-verify is skipped in
template mode (scaffold doesn't follow the model's AR continuation, so
verify would shrink the accepted prefix to almost nothing).

`Req._init_fill_ids_for_dllm` — when template is set, appends template
chunks instead of fresh mask blocks. Each round emits `block_size` tokens
of template; last chunk is padded with `<|endoftext|>` (Qwen) if shorter.

`scheduler_output_processor_mixin.process_batch_result_dllm_prefill` —
skips the `ar_token` append to `output_ids` in template mode (would
duplicate / fight with `template[0]`).

## Results (5-sample Waymo, AD CoT)

| Path | Avg latency | Schema | Behavior accuracy | Explanation |
|---|---:|---|---|---|
| **SGLang template (mdm)** | **2.08s** | ✓ 12/12 categories | **5/5 lat, 5/5 long** | 3-stage CoT, real content |
| SGLang free-form (spec) | 1.19s | variable 5-12 (some loops) | 4/5 lat (1 missing) | variable; some AR loops |
| transformers template | 1.3s | ✓ 12/12 | 4/5 lat | partial CoT with artifacts |
| dVLM-AD (LLaDA-V finetuned) | 16.3s | data file template (12×yes/no) | 5/5 lat | trained CoT |

The SGLang template path **matches** dVLM-AD on behavior accuracy and
**beats** the free-form SGLang baseline on schema compliance + V3-CoT
explanation, at ~13% over the 2s latency target.

## Nav injection (the lateral-accuracy fix)

The loader splices `"navigation_command": "<nav>", ` into the template
right before `"future_meta_behavior":`. This puts the nav target in the
behavior block's immediate bidir-attention context, biasing the lateral
verb without conditional gating.

Effect: lateral correctness 4/5 → 5/5 (sample 2 GO_RIGHT, which would
otherwise emit `turn left`, now correctly emits `turn right`).
