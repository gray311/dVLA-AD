"""Run Fast-dVLM weights + Fast-dDrive decoding algorithm + OUR V3 template.

Combines:
- Fast-dVLM-3B weights (our zero-shot ckpt)
- Fast-dDrive's `mdm_sample_deep_scaffold` algorithm (section diffusion +
  block-causal hybrid attention + confidence threshold)
- Our V3 schema (12-cat open-vocab critical_objects + complexity +
  100-token explanation + 2-word behavior verbs + semantic trajectory)
- Our V3 prompt (with nav injection)

Done by monkey-patching `section_utils.build_deep_json_scaffold` to return
our V3 scaffold instead of Fast-dDrive's yes/no scaffold.
"""
import glob, json, math, os, sys, time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from safetensors.torch import load_file

ROOT = "/weka/home/ext-yingzima/dVLA-AD"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval"))

from eval.template_v3 import (
    build_template_ids_v3, build_prompt_v3, parse_filled,
)
from eval.loaders.fast_dvlm_sglang_v3 import (
    _inject_nav_into_template, _SECTION_OF_KIND, _compute_section_spans,
)

DVLM_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"
DDRIVE_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/transformers/models--xiwenyoumu--Fast-dDrive/snapshots/ddadfbbd31014fa0d6c3bbf457070d499ec19241"

DATA = "/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_waymo_e2e_val_cot.json"
PATH_FIX = ("/weka/home/xliu316/", "/weka/home/ext-yingzima/")
PICKED_IDX = [202, 244, 59]


def _fix(p): return p.replace(PATH_FIX[0], PATH_FIX[1])
def _joint_image(sample): return _fix(sample["image"][1]).replace("CAM_FRONT.jpg", "CAM_JOINT.jpg")


def build_v3_scaffold_for_ddrive(tokenizer, mask_id, nav_command=None,
                                  null_id=151666, **_):
    """Drop-in replacement for Fast-dDrive's `build_deep_json_scaffold`.

    Returns the same triple but using our V3 schema instead of Fast-dDrive's:
      scaffold_tokens   : List[int]  (mask_id at value positions)
      section_ranges    : Dict[name, (start, end)]
      scaffold_mask     : List[int]  (1 = scaffold, 0 = value/mask)
    """
    ids, slot_info, _ = build_template_ids_v3(tokenizer, mask_id)
    if nav_command:
        ids, slot_info = _inject_nav_into_template(
            ids, slot_info, tokenizer, mask_id, nav_command,
        )

    # Compute section spans using the same logic as our SGLang loader.
    # Spans use our 4-section partition (crit_cmplx, expl, beh, traj).
    spans = _compute_section_spans(ids, slot_info)

    # Map our section names → Fast-dDrive's SECTION_KEYS names.
    SECT_NAME_MAP = {
        "crit_cmplx": "critical_objects",
        "expl": "explanation",
        "beh": "future_meta_behavior",
        "traj": "trajectory",
    }
    section_ranges = {}
    for sec, start, end in spans:
        name = SECT_NAME_MAP.get(sec, sec)
        # If we encountered multiple spans for the same section name
        # (e.g., trailing scaffold), merge with previous.
        if name in section_ranges:
            prev_start, _ = section_ranges[name]
            section_ranges[name] = (prev_start, end)
        else:
            section_ranges[name] = (start, end)

    # scaffold_mask: 0 at value positions (masks), 1 at scaffold positions.
    mask_positions = {pos for pos, _ in slot_info}
    scaffold_mask = [0 if i in mask_positions else 1 for i in range(len(ids))]

    return ids, section_ranges, scaffold_mask


def main():
    print(f"Loading Fast-dDrive model class from {DDRIVE_DIR} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        DDRIVE_DIR, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    ).eval()

    print(f"Replacing weights with Fast-dVLM ckpt ...", flush=True)
    state_dict = {}
    for sf in sorted(glob.glob(f"{DVLM_DIR}/model-*.safetensors")):
        state_dict.update(load_file(sf))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  loaded — missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(DDRIVE_DIR, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(DDRIVE_DIR, use_fast=False)
    processor.tokenizer = tokenizer
    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])

    # Monkey-patch the section scaffold builder.
    import transformers_modules.ddadfbbd31014fa0d6c3bbf457070d499ec19241.section_utils as su

    _CURRENT_NAV = {"value": None}
    def patched_builder(tok, **kwargs):
        return build_v3_scaffold_for_ddrive(
            tok, mask_id=mask_id, nav_command=_CURRENT_NAV["value"],
        )
    su.build_deep_json_scaffold = patched_builder
    print("Patched section_utils.build_deep_json_scaffold → V3 scaffold", flush=True)

    data = json.load(open(DATA))
    results = {}

    for idx in PICKED_IDX:
        s = data[idx]
        img_path = _joint_image(s)
        nav = s["navigation_command"]
        _CURRENT_NAV["value"] = nav  # threaded into patched builder
        vx, vy = s["velocity"][-1]
        speed = math.hypot(vx, vy)
        prompt = build_prompt_v3(s)

        print(f"\n=== idx={idx} nav={nav} v={speed:.1f} ===", flush=True)
        if not os.path.exists(img_path):
            print(f"  ! image missing"); continue
        image = Image.open(img_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")

        kwargs = dict(
            input_ids=inputs.input_ids, tokenizer=tokenizer,
            block_size=32, max_tokens=1024, mask_id=mask_id, threshold=0.9,
        )
        if getattr(inputs, "pixel_values", None) is not None:
            kwargs["pixel_values"] = inputs.pixel_values
        if getattr(inputs, "image_grid_thw", None) is not None:
            kwargs["image_grid_thw"] = inputs.image_grid_thw

        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode():
            out = model.mdm_sample_deep_scaffold(**kwargs)
        torch.cuda.synchronize(); latency = time.time() - t0

        trimmed = out[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(trimmed, skip_special_tokens=True)
        print(f"  latency={latency:.2f}s, response_len={len(response)} chars")
        print(f"  --- Response ---")
        print(f"  {response[:1800]}")
        results[idx] = {"latency": latency, "response": response, "nav": nav, "speed": speed}

    with open("/tmp/dvlm_ddrive_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved /tmp/dvlm_ddrive_v3.json")


if __name__ == "__main__":
    main()
