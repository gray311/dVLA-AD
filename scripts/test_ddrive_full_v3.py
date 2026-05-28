"""Fast-dDrive FULL finetuned weights + OUR V3 template.

Unlike run_dvlm_ddrive_v3_template.py (which swaps in Fast-dVLM zero-shot
weights), this keeps Fast-dDrive's finetuned weights. Patches
build_deep_json_scaffold → V3 scaffold.

Two decoding modes (CLI arg, default 'mdm'):
  mdm : mdm_sample_deep_scaffold  (section diffusion, threshold=0.9) — NO self-spec
  ss  : scaffold_speculative_sample (self-speculative, paper canonical)

Usage: python scripts/test_ddrive_full_v3.py [mdm|ss]
"""
import json, math, os, re, sys, time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import build_prompt_v3, build_template_ids_v3, parse_filled
from eval.loaders.fast_dvlm_ddrive_v3 import (
    _inject_nav_into_template, _compute_section_spans, _DDRIVE_SECTION_NAMES,
)

DDRIVE_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/transformers/models--xiwenyoumu--Fast-dDrive/snapshots/ddadfbbd31014fa0d6c3bbf457070d499ec19241"
DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
PICKED_IDX = [202, 244, 59]


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])
def _joint(s): return _fix(s["image"][1]).replace("CAM_FRONT.jpg", "CAM_JOINT.jpg")


def ade(pred, gt, k=5):
    n = min(len(pred), len(gt), k)
    if n == 0: return None
    return sum(math.hypot(pred[i][0]-gt[i][0], pred[i][1]-gt[i][1]) for i in range(n)) / n


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "mdm"
    print(f"Mode: {mode} ({'section diffusion, NO self-spec' if mode=='mdm' else 'self-speculative'})", flush=True)

    print("Loading Fast-dDrive FULL finetuned weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        DDRIVE_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(DDRIVE_DIR, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(DDRIVE_DIR, use_fast=False)
    processor.tokenizer = tokenizer
    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])

    # Patch scaffold builder → V3.
    import importlib
    snapshot = os.path.basename(DDRIVE_DIR.rstrip("/"))
    su = importlib.import_module(f"transformers_modules.{snapshot}.section_utils")
    nav_cell = {"v": None}

    def patched(tok, **kwargs):
        ids, slots, _ = build_template_ids_v3(tok, mask_id)
        if nav_cell["v"]:
            ids, slots = _inject_nav_into_template(ids, slots, tok, mask_id, nav_cell["v"])
        spans = _compute_section_spans(ids, slots)
        section_ranges = {}
        for sec, start, end in spans:
            name = _DDRIVE_SECTION_NAMES.get(sec, sec)
            if name in section_ranges:
                section_ranges[name] = (section_ranges[name][0], end)
            else:
                section_ranges[name] = (start, end)
        mask_pos = {p for p, _ in slots}
        scaffold_mask = [0 if i in mask_pos else 1 for i in range(len(ids))]
        return ids, section_ranges, scaffold_mask

    su.build_deep_json_scaffold = patched
    print("Patched build_deep_json_scaffold → V3 scaffold", flush=True)

    data = json.load(open(DATA))
    results = []
    for idx in PICKED_IDX:
        s = data[idx]
        nav_cell["v"] = s["navigation_command"]
        img = _joint(s)
        prompt = build_prompt_v3(s)
        image = Image.open(img).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_in], images=[image], return_tensors="pt").to("cuda:0")

        kw = dict(input_ids=inputs.input_ids, tokenizer=tokenizer, block_size=32,
                  max_tokens=1024, mask_id=mask_id, threshold=0.9)
        if getattr(inputs, "pixel_values", None) is not None:
            kw["pixel_values"] = inputs.pixel_values
        if getattr(inputs, "image_grid_thw", None) is not None:
            kw["image_grid_thw"] = inputs.image_grid_thw

        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode():
            if mode == "ss":
                out = model.scaffold_speculative_sample(**kw)
            else:
                out = model.mdm_sample_deep_scaffold(**kw)
        torch.cuda.synchronize(); lat = time.time() - t0

        resp = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = parse_filled(resp)
        gt = s["future waypoints"][:5]
        a = ade(pred, gt) if pred else None
        m = re.search(r'"explanation":\s*"([^"]*)"', resp)
        expl = m.group(1) if m else "(none)"
        print(f"\n=== idx={idx} nav={s['navigation_command']} ADE={'N/A' if a is None else f'{a:.2f}m'} lat={lat:.2f}s ===", flush=True)
        print(f"EXPL: {expl[:400]}", flush=True)
        results.append({"idx": idx, "ade": a, "latency": lat, "mode": mode, "text": resp})

    with open(f"/tmp/ddrive_full_v3_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)
    lats = [r["latency"] for r in results]
    ades = [r["ade"] for r in results if r["ade"] is not None]
    print(f"\nmean lat={sum(lats)/len(lats):.2f}s  mean ADE={sum(ades)/len(ades):.2f}m" if ades else "")


if __name__ == "__main__":
    main()
