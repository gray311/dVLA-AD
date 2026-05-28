"""Run test_image.png (burning car OOD) on 3 configs:
  A. SGLang Fast-dVLM (current default — bs=32 legacy + rep=0.5 + N=64 + fixes)
  B. dDrive code + Fast-dVLM weights + V3 template
  C. Fast-dDrive full (finetuned model + their schema)

Saves explanations side by side for comparison.
"""
import glob, json, math, os, subprocess, sys, time

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))

IMG = os.path.join(ROOT, "examples", "test_image.png")

MOCK_META = dict(
    speed=15.0, accel=-0.5, nav="GO_STRAIGHT",
    note="OOD: burning car + orange cones on highway",
)


def _build_mock_sample():
    dt, n = 0.1, 16
    hist = [(-MOCK_META["speed"] * (n - 1 - i) * dt, 0.0) for i in range(n)]
    return {
        "sample_id": "test_image_burning_car",
        "navigation_command": MOCK_META["nav"],
        "velocity": [(MOCK_META["speed"], 0.0)],
        "acceleration": [(MOCK_META["accel"], 0.0)],
        "image": [IMG, IMG, IMG],
        "history waypoints": hist,
        "future waypoints": [],
    }


def run_sglang_v3():
    """Config A: our SGLang current default."""
    from eval.template_v3 import build_prompt_v3, parse_filled
    from eval.loaders import fast_dvlm_sglang_v3 as loader

    s = _build_mock_sample()
    print("[A SGLang] loading engine...", flush=True)
    bundle = loader.load(algorithm="mdm", engine_block_size=32)
    print("[A SGLang] warmup...", flush=True)
    _, _ = loader.generate(
        bundle, [IMG], build_prompt_v3(s),
        temperature=0.0, block_size=32, section_align=False,
        nav_command=s['navigation_command'],
    )
    text, latency = loader.generate(
        bundle, [IMG], build_prompt_v3(s),
        temperature=0.0, block_size=32, section_align=False,
        nav_command=s['navigation_command'],
    )
    loader.shutdown(bundle)
    return text, latency


def run_ddrive_code_dvlm_v3():
    """Config B: Fast-dDrive code + Fast-dVLM weights + V3 template."""
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from safetensors.torch import load_file
    from eval.template_v3 import build_prompt_v3
    from eval.loaders.fast_dvlm_sglang_v3 import (
        _inject_nav_into_template, _SECTION_OF_KIND, _compute_section_spans,
    )
    from eval.template_v3 import build_template_ids_v3

    DVLM_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"
    DDRIVE_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/transformers/models--xiwenyoumu--Fast-dDrive/snapshots/ddadfbbd31014fa0d6c3bbf457070d499ec19241"

    print("[B dDrive+dVLM+V3] loading...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        DDRIVE_DIR, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    ).eval()
    state_dict = {}
    for sf in sorted(glob.glob(f"{DVLM_DIR}/model-*.safetensors")):
        state_dict.update(load_file(sf))
    model.load_state_dict(state_dict, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(DDRIVE_DIR, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(DDRIVE_DIR, use_fast=False)
    processor.tokenizer = tokenizer
    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])

    # Patch the scaffold builder
    import transformers_modules.ddadfbbd31014fa0d6c3bbf457070d499ec19241.section_utils as su
    SECT_NAME_MAP = {
        "crit_cmplx": "critical_objects", "expl": "explanation",
        "beh": "future_meta_behavior", "traj": "trajectory",
    }
    _NAV = {"v": None}
    def patched(tok, **kwargs):
        ids, slots, _ = build_template_ids_v3(tok, mask_id)
        if _NAV["v"]:
            ids, slots = _inject_nav_into_template(ids, slots, tok, mask_id, _NAV["v"])
        spans = _compute_section_spans(ids, slots)
        section_ranges = {}
        for sec, start, end in spans:
            name = SECT_NAME_MAP.get(sec, sec)
            if name in section_ranges:
                section_ranges[name] = (section_ranges[name][0], end)
            else:
                section_ranges[name] = (start, end)
        mask_positions = {pos for pos, _ in slots}
        scaffold_mask = [0 if i in mask_positions else 1 for i in range(len(ids))]
        return ids, section_ranges, scaffold_mask
    su.build_deep_json_scaffold = patched

    s = _build_mock_sample()
    _NAV["v"] = s["navigation_command"]
    prompt = build_prompt_v3(s)

    image = Image.open(IMG).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_in], images=[image], return_tensors="pt").to("cuda:0")
    kwargs = dict(input_ids=inputs.input_ids, tokenizer=tokenizer,
                  block_size=32, max_tokens=1024, mask_id=mask_id, threshold=0.9)
    if getattr(inputs, "pixel_values", None) is not None:
        kwargs["pixel_values"] = inputs.pixel_values
    if getattr(inputs, "image_grid_thw", None) is not None:
        kwargs["image_grid_thw"] = inputs.image_grid_thw

    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = model.mdm_sample_deep_scaffold(**kwargs)
    torch.cuda.synchronize(); latency = time.time() - t0
    response = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response, latency


def run_ddrive_full():
    """Config C: Fast-dDrive full finetuned model + their schema."""
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    print("[C dDrive full] loading...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "xiwenyoumu/Fast-dDrive", torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("xiwenyoumu/Fast-dDrive", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("xiwenyoumu/Fast-dDrive", use_fast=False)
    processor.tokenizer = tokenizer
    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])

    prompt = (
        "You are an expert autonomous driving agent.\n"
        f"Ego state: speed={MOCK_META['speed']:.1f} m/s, accel={MOCK_META['accel']:.2f} m/s^2\n"
        f"Driver instruction: {MOCK_META['nav']}\n"
        "Analyze the scene and produce critical_objects, explanation, "
        "future_meta_behavior, trajectory (5 waypoints, 1 s intervals)."
    )
    image = Image.open(IMG).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_in], images=[image], return_tensors="pt").to("cuda:0")
    kwargs = dict(input_ids=inputs.input_ids, tokenizer=tokenizer,
                  block_size=32, max_tokens=1024, mask_id=mask_id, threshold=0.9)
    if getattr(inputs, "pixel_values", None) is not None:
        kwargs["pixel_values"] = inputs.pixel_values
    if getattr(inputs, "image_grid_thw", None) is not None:
        kwargs["image_grid_thw"] = inputs.image_grid_thw

    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = model.mdm_sample_deep_scaffold(**kwargs)
    torch.cuda.synchronize(); latency = time.time() - t0
    response = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response, latency


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "all"
    out_json = "/tmp/testimg_3way.json"

    if config == "a":
        text, lat = run_sglang_v3()
        results = {"A": {"text": text, "latency": lat}}
    elif config == "b":
        text, lat = run_ddrive_code_dvlm_v3()
        results = {"B": {"text": text, "latency": lat}}
    elif config == "c":
        text, lat = run_ddrive_full()
        results = {"C": {"text": text, "latency": lat}}
    else:
        # Run all 3 in separate subprocesses (avoids SGLang/HF process conflicts)
        results = {}
        for c, label in [("a", "A_sglang"), ("b", "B_ddrive_code"), ("c", "C_ddrive_full")]:
            print(f"\n========== {label} ==========", flush=True)
            rc = subprocess.call(
                [sys.executable, "-u", __file__, c],
                env=dict(os.environ),
            )
            if rc != 0:
                print(f"{label} failed (rc={rc})"); continue
            tmp = json.load(open(out_json))
            key = next(iter(tmp))
            results[label] = tmp[key]
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        for label, r in results.items():
            print(f"\n========== {label} lat={r['latency']:.2f}s ==========")
            print(r['text'][:1500])
        return

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
