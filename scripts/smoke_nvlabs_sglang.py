"""Run Fast-dVLM weights through NVlabs SGLang (vanilla — no template-mode)
with V3 prompt + V3 template as free-form text.

Setup:
- Framework: NVlabs Fast-dLLM SGLang fork (third_party/sglang/python)
- Model:     Fast-dVLM-3B weights (our zero-shot ckpt)
- Schema:    V3 schema description in prompt (no template-fill support in
             vanilla SGLang, so this is free-form generation)
- Decoding:  HierarchyBlock (mdm) — vanilla, no scaffold

This is the closest analog to "their SGLang launches dDrive inference"
since NVlabs's repo has no public Fast-dDrive SGLang code path.
"""
import json, math, os, sys, time

os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"

# CRITICAL: prepend NVlabs SGLang path BEFORE any sglang import
NVLABS_SGL = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/Fast-dLLM_ref/Fast-dLLM/third_party/sglang/python"
sys.path.insert(0, NVLABS_SGL)

# Now imports of sglang resolve to NVlabs version
import sglang as sgl
print(f"Using sglang from: {sgl.__file__}", flush=True)

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, build_template_ids_v3, parse_filled

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
PICKED_IDX = [202, 244, 59]
DVLM_PATH = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0: return None
    return sum(math.hypot(pred[i][0]-gt[i][0], pred[i][1]-gt[i][1]) for i in range(n)) / n


def _build_input_ids(processor, image_path, prompt_text):
    from PIL import Image
    if image_path:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
    else:
        messages = [{"role": "user", "content": prompt_text}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image_path:
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")
    return inputs.input_ids[0].tolist()


def main():
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(DVLM_PATH, trust_remote_code=True)
    processor.tokenizer = tokenizer

    print(f"Launching NVlabs SGLang Engine with Fast-dVLM-3B ...", flush=True)
    engine = sgl.Engine(
        model_path=DVLM_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        mem_fraction_static=0.75,
        max_running_requests=1,
        chunked_prefill_size=16384,
        dllm_algorithm="HierarchyBlock",
        disable_cuda_graph=False,
        log_level="warning",
        enable_metrics=False,
        mm_attention_backend="triton_attn",
    )

    data = json.load(open(DATA))
    samples = [(i, data[i]) for i in PICKED_IDX]

    # Build prompt: full V3 prompt + describe template as text (no mask fill since
    # vanilla SGLang has no template-mode).
    mask_id = tokenizer.encode("|<MASK>|", add_special_tokens=False)[0]
    template_ids, _, _ = build_template_ids_v3(tokenizer, mask_id)
    template_str = tokenizer.decode(template_ids, skip_special_tokens=False)

    print("Warmup...", flush=True)
    s_warm = samples[0][1]
    prompt_warm = build_prompt_v3(s_warm) + f"\n\nTEMPLATE (fill the |<MASK>| positions only):\n\n{template_str}\n\nFilled JSON:"
    input_ids = _build_input_ids(processor, _fix(s_warm['image'][1]), prompt_warm)
    _ = engine.generate(
        input_ids=input_ids,
        image_data=[_fix(s_warm['image'][1])],
        sampling_params={"max_new_tokens": 512, "temperature": 0.0},
    )

    results = []
    for idx, s in samples:
        prompt = build_prompt_v3(s) + f"\n\nTEMPLATE (fill the |<MASK>| positions only):\n\n{template_str}\n\nFilled JSON:"
        img = _fix(s['image'][1])
        input_ids = _build_input_ids(processor, img, prompt)

        import torch
        torch.cuda.synchronize(); t0 = time.time()
        out = engine.generate(
            input_ids=input_ids,
            image_data=[img],
            sampling_params={"max_new_tokens": 600, "temperature": 0.0},
        )
        torch.cuda.synchronize(); latency = time.time() - t0
        text = out[0]["text"] if isinstance(out, list) else out["text"]
        pred = parse_filled(text)
        gt = s["future waypoints"][:5]
        a = ade(pred, gt) if pred else None
        results.append({"idx": idx, "ade": a, "latency": latency, "text": text})
        print(f"  idx={idx:3} ADE={'N/A' if a is None else f'{a:5.2f}m'} lat={latency:.2f}s", flush=True)

    engine.shutdown()
    with open("/tmp/nvlabs_sgl.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n--- summary ---")
    lats = [r['latency'] for r in results]
    ades = [r['ade'] for r in results if r['ade'] is not None]
    print(f"mean_lat = {sum(lats)/len(lats):.2f}s")
    if ades: print(f"mean_ADE = {sum(ades)/len(ades):.2f}m")


if __name__ == "__main__":
    main()
