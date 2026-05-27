# Sample example: Waymo `9ac9e56e...`, `GO_LEFT`, 1.2 m/s

A canonical end-to-end example: image → V3 prompt + template → outputs from
five inference paths.

## Inputs

| file | what it is |
|---|---|
| `cam_front.jpg` | front camera (the only image used by Fast-dVLM; dVLM-AD uses 3 cams) |
| `meta.json` | sample id, navigation_command, speed, ego state, GT future waypoints |
| `prompt_full.txt` | the full V3 prompt that `build_prompt_v3()` returns (includes TEMPLATE block at tail) |
| `prompt_to_model.txt` | prompt actually sent to model in the SGLang path (TEMPLATE block stripped — template is passed via `sampling_params.dllm_template_token_ids` instead) |
| `template.txt` | the response template decoded from token ids, with `\|<MASK>\|` markers and `"navigation_command": "GO_LEFT"` injection. 390 tokens, 208 mask + 182 scaffold. |

## Outputs (sample_id = `9ac9e56e...`, all greedy temp=0)

| file | path | latency |
|---|---|---:|
| `output_fast_dvlm_sglang_template.json` | **Fast_dVLM_3B via MODIFIED SGLang fork; template-fill mdm (this repo's main path)** | **1.83s** |
| `output_fast_dvlm_sglang_freeform.json` | Fast_dVLM_3B via unmodified NVlabs SGLang fork; free-form spec, no template | 0.78s |
| `output_fast_dvlm_transformers_template.json` | Fast_dVLM_3B + V3 template via custom transformers loader (block-causal fill) | 1.72s |
| `output_diffusionvl_template.json` | DiffusionVL-3B + V3 template via custom transformers loader (BD3LM block-causal) | 5.47s |
| `output_dvlm_ad_finetuned.json` | dVLM-AD (LLaDA-V-8B finetuned on Waymo CoT) + the data file's template | 15.91s |

## Key takeaways from this example

1. **Template-fill (SGLang or transformers) gives the cleanest schema**:
   all 12 critical_objects categories, structured trajectory in
   `+XX.X,+YY.Y` format, 2-word verb behavior. Free-form often emits a
   different (model-chosen) set of categories.

2. **Template-fill SGLang explanation matches V3 3-stage CoT**:
   "scene description → crucial object behavior prediction → ego-object
   interaction". Free-form explanation is shorter and sometimes loops on
   `"the the the"` mid-paragraph.

3. **All paths got behavior right on this sample**:
   `slow down / turn left` (matches `GO_LEFT` and the slow ego speed).

4. **Per-token latency**: free-form spec ≈ 1ms/char output; mdm template
   ≈ 1.5ms/char (multi-step diffusion overhead). dVLM-AD is 10× slower
   because it's LLaDA-V (8B) instead of Qwen2.5-VL (3B) backbone.
