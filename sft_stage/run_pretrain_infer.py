"""Pre-training inference sanity for dVLA Stage-1.

Loads the (untrained) backbone the same way the finetuner does
(trust_remote_code, bf16) and runs the section-diffusion decode
(`mdm_sample_deep_scaffold`) on a real V3 sample, printing the prompt the model
sees and the JSON it fills in BEFORE any driving finetuning. Per plan §1.2 we
expect fluent explanation but poor/degenerate trajectory at this point.

Usage:
  python run_pretrain_infer.py                       # SASD backbone, example case 0
  python run_pretrain_infer.py --model_path .../Fast_dVLM_3B --case 1
"""
import argparse, json, os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_SC = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=f"{_SC}/models/Fast_dVLM_3B_sasd")
    ap.add_argument("--sample_json", default=f"{_HERE}/data/example/v3_train_sample.json")
    ap.add_argument("--image_root", default=_HERE)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=1024)  # > scaffold len (~565)
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    sample = json.load(open(args.sample_json))[args.case]
    images = [Image.open(os.path.join(args.image_root, p)).convert("RGB")
              for p in sample["image"]]
    full_prompt = sample["conversations"][0]["value"]
    gt = sample["conversations"][1]["value"]
    # The V3 prompt embeds the masked TEMPLATE at its tail; mdm_sample_deep_scaffold
    # appends its OWN scaffold, so feed only the instruction part (before TEMPLATE).
    cut = full_prompt.find("TEMPLATE (")
    prompt = full_prompt[:cut].rstrip() if cut > 0 else full_prompt

    print(f"Loading {args.model_path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False,
                                              max_pixels=200704, min_pixels=200704)
    processor.tokenizer = tokenizer

    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda:0")

    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])
    kwargs = dict(input_ids=inputs.input_ids, tokenizer=tokenizer,
                  block_size=args.block_size, max_tokens=args.max_tokens,
                  mask_id=mask_id, threshold=args.threshold)
    if getattr(inputs, "pixel_values", None) is not None:
        kwargs["pixel_values"] = inputs.pixel_values
    if getattr(inputs, "image_grid_thw", None) is not None:
        kwargs["image_grid_thw"] = inputs.image_grid_thw

    print(f"\n{'='*90}\nINPUT PROMPT (sample_id={sample.get('sample_id')}):\n{'='*90}\n{prompt}")
    print(f"\n{'='*90}\nDecoding (section_diffusion, threshold={args.threshold}) ...\n{'='*90}", flush=True)
    with torch.inference_mode():
        out = model.mdm_sample_deep_scaffold(**kwargs)
    resp = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print("\n--- MODEL OUTPUT (untrained) ---\n" + resp)
    print("\n--- GROUND TRUTH (V3 target) ---\n" + gt)


if __name__ == "__main__":
    main()
