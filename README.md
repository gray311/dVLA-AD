# dVLA-AD

Zero-shot driving CoT (perception → complexity → explanation → meta-behavior
→ trajectory) with **Fast-dVLM-3B** served via a modified **NVlabs SGLang
fork** that supports template-fill of structured JSON responses.

## Repo layout

```
eval/
  template_v3.py                     V3 schema + prompt builder + parser
  loaders/
    fast_dvlm_sglang_v3.py           production loader (SGLang template-fill)
    fast_dvlm_v3.py                  transformers baseline (block-causal)
    fast_dvlm_sglang.py              free-form baseline (no template)
    dvlm_ad.py                       dVLM-AD_waymo (finetuned LLaDA-V) loader

scripts/
  run_10_waymo_compare.py            10-sample SGLang vs dVLM-AD runner
  run_n_waymo_ade.py                 N-sample ADE sweep
  write_10_compare_report.py         renders the comparison.md report
  save_longtail_examples.py          dumps SGLang outputs for 10 longtail samples
  save_longtail_dvlm_ad.py           dumps dVLM-AD outputs for same 10 samples
  save_test_image_to_longtail.py     dumps SGLang + dVLM-AD on the burning-car image

third_party/sglang/                  vendored SGLang fork with template-fill APIs

examples/longtail_10/                10 longtail Waymo samples + test_image_burning_car/
                                       — each has prompts, templates, outputs,
                                       and images for both SGLang and dVLM-AD

results/waymo_10_compare/            comparison.md, ade_comparison.md, raw JSONs
```

## Setup

```bash
# In an env with torch 2.9.x and Qwen2.5-VL deps:
pip install -e third_party/sglang/python --no-deps

# 10-sample SGLang vs dVLM-AD comparison
python scripts/run_10_waymo_compare.py sglang
python scripts/run_10_waymo_compare.py ad
python scripts/write_10_compare_report.py
```

## V3 schema

```
{"critical_objects": {12 categories × 2 mask},   <- detect objects
 "complexity": "<simple|complex>",               <- 1-mask judgement
 "explanation": "<100 mask>",                    <- ~100 token CoT
 "navigation_command": "<runtime-inject>",       <- nav hint
 "future_meta_behavior": {                       <- 2-word verbs
   "longitudinal": "<m> <m>",                       speed up / slow down / keep speed / stop now
   "lateral":      "<m> <m>"                        turn left / turn right / keep lane / change left|right
 },
 "trajectory": "<semantic per-waypoint lines>"   <- 10 wp × 0.5s
}
```

Trajectory format (each waypoint one line):

```
0.5s: forward=+05.0m, lateral=+00.0m
1.0s: forward=+10.0m, lateral=+00.0m
...
5.0s: forward=+50.0m, lateral=+00.0m
```

## SGLang fork modifications

Adds three new `SamplingParams` fields for structured-response template-fill:

- **`dllm_template_token_ids: List[int]`** — response scaffold containing
  `mask_id` at fill slots. Engine feeds this in `block_size`-sized chunks
  instead of auto-generating fresh mask blocks. Scaffold positions stay
  intact across diffusion. `max_new_tokens` is auto-capped to template
  length and `ignore_eos=True` is forced.

- **`dllm_template_position_gates: List[Optional[List[int]]]`** — per-position
  vocab allowlist. Used for structured slots (trajectory digits/signs,
  behavior verb words, complexity tag) to avoid BPE-boundary artifacts.

- **`dllm_template_forbidden_token_ids: List[int]`** — global blacklist
  applied at every masked position (JSON-meta chars `"`, `}`, `\`, backtick).

Algorithm changes in `HierarchyBlock` (`third_party/sglang/.../dllm/algorithm/
hierarchy_block.py`):
- detects template mode via `forward_batch.dllm_template_modes`
- bypasses AR-token override at chunk position 0
- returns ALL block positions (not just mask suffix)
- applies gates + forbidden masks before argmax/topk
- fixed-step path (default 4 steps/chunk) with rep penalty + within-step
  dedup at rep-penalty positions (explanation slot)

Other patches: `Req._init_fill_ids_for_dllm`, `scheduler_output_processor_mixin
.process_batch_result_dllm_prefill`, plus the per-req flag plumbing through
`ScheduleBatch → ModelWorkerBatch → ForwardBatch`.

## Nav injection

The loader splices `"navigation_command": "<nav>", ` into the template right
before `"future_meta_behavior":`. This puts the nav target in the behavior
block's immediate bidir-attention context, biasing the lateral verb without
conditional gating. Effect on the 10-sample test: lateral correctness
goes from 7/10 (without injection) to 9/10 (with).

## Final benchmark (10-sample Waymo)

| Model | Avg latency | ADE (mean L2, 5 wp) | Lateral acc | Behavior validity |
|---|---:|---:|:---:|:---:|
| **SGLang Fast-dVLM (zero-shot)** | **~1.7s** (CAM_FRONT) / 3.3s (CAM_JOINT) | 4.28m (48-sample) | 9/10 | 10/10 |
| dVLM-AD (finetuned on Waymo CoT) | 33s | — | 9/10 | 10/10 |

SGLang Fast-dVLM matches the finetuned baseline on behavior accuracy at
**~19× less latency**.

See `results/waymo_10_compare/` for raw JSONs + reports and
`examples/longtail_10/` for per-sample browsable outputs.
