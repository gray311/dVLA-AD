"""Batch pre-training inference sanity over several V3 samples.

Loads the model ONCE and runs section-diffusion decode on N samples, then checks
each output for: valid JSON, all 5 sections, 12 critical keys, fmb structure,
trajectory waypoint count, and obvious garbage (token repetition, bad vocab).
"""
import argparse, json, os, re, collections
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_SC = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy"
LONG_VERBS = {"speed up", "slow down", "keep speed", "stop now"}
LAT_VERBS = {"keep lane", "turn left", "turn right", "change left", "change right"}
CRIT_KEYS = ["nearby_vehicle", "pedestrian", "cyclist", "construction", "traffic_element",
             "weather_condition", "road_hazard", "emergency_vehicle", "animal",
             "special_vehicle", "conflicting_vehicle", "door_opening_vehicle"]


def first_json(s):
    """Extract the first balanced {...} object string."""
    i = s.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return None


def max_token_run(s):
    """Longest run of a repeated whitespace-delimited token (garbage detector)."""
    toks = s.split()
    best = cur = 1
    for a, b in zip(toks, toks[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def check(resp_raw):
    out = {"json_valid": False, "sections_ok": False, "crit12": False, "fmb_ok": False,
           "n_traj": 0, "max_rep": max_token_run(resp_raw), "issues": []}
    blob = first_json(resp_raw)
    if blob is None:
        out["issues"].append("no JSON object")
        return out
    try:
        obj = json.loads(blob)
        out["json_valid"] = True
    except Exception as e:
        out["issues"].append(f"json parse fail: {str(e)[:40]}")
        # still try structural regex checks on raw
        out["n_traj"] = len(re.findall(r"forward=", blob))
        return out
    need = ["critical_objects", "complexity", "explanation", "future_meta_behavior", "trajectory"]
    miss = [k for k in need if k not in obj]
    out["sections_ok"] = not miss
    if miss:
        out["issues"].append("missing sections: " + ",".join(miss))
    co = obj.get("critical_objects", {})
    out["crit12"] = isinstance(co, dict) and all(k in co for k in CRIT_KEYS)
    if not out["crit12"]:
        out["issues"].append("critical_objects keys != 12")
    fmb = obj.get("future_meta_behavior", {})
    if isinstance(fmb, dict) and "longitudinal" in fmb and "lateral" in fmb:
        out["fmb_ok"] = True
        lo = str(fmb["longitudinal"]).replace("<|NULL|>", "").strip()
        la = str(fmb["lateral"]).replace("<|NULL|>", "").strip()
        if lo not in LONG_VERBS:
            out["issues"].append(f"long verb bad: {lo!r}")
        if la not in LAT_VERBS:
            out["issues"].append(f"lat verb bad: {la!r}")
    else:
        out["issues"].append("fmb missing long/lat")
    out["n_traj"] = len(re.findall(r"forward=", str(obj.get("trajectory", ""))))
    cx = str(obj.get("complexity", "")).replace("<|NULL|>", "").strip()
    if cx not in ("simple", "complex"):
        out["issues"].append(f"complexity bad: {cx!r}")
    if out["max_rep"] >= 6:
        out["issues"].append(f"token repetition x{out['max_rep']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=f"{_SC}/models/Fast_dVLM_3B_sasd")
    ap.add_argument("--sample_json", default=f"{_SC}/data/dvla_sft/dvlm-ad_waymo_training_v3_joint.json")
    ap.add_argument("--image_root", default="/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/waymo")
    ap.add_argument("--indices", default="0,4000,8000,12000,16000,20000,24000,28000")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--block_size", type=int, default=32)
    args = ap.parse_args()

    data = json.load(open(args.sample_json))
    idxs = [int(x) for x in args.indices.split(",")]

    print(f"Loading {args.model_path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False,
                                              max_pixels=200704, min_pixels=200704)
    processor.tokenizer = tokenizer
    mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])

    agg = collections.Counter()
    for n, i in enumerate(idxs):
        s = data[i]
        images = [Image.open(os.path.join(args.image_root, p)).convert("RGB") for p in s["image"]]
        full = s["conversations"][0]["value"]
        cut = full.find("TEMPLATE (")
        prompt = full[:cut].rstrip() if cut > 0 else full
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
        kw = dict(input_ids=inp.input_ids, tokenizer=tokenizer, block_size=args.block_size,
                  max_tokens=args.max_tokens, mask_id=mask_id, threshold=args.threshold)
        if getattr(inp, "pixel_values", None) is not None:
            kw["pixel_values"] = inp.pixel_values
        if getattr(inp, "image_grid_thw", None) is not None:
            kw["image_grid_thw"] = inp.image_grid_thw
        with torch.inference_mode():
            out = model.mdm_sample_deep_scaffold(**kw)
        resp = tokenizer.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
        c = check(resp)
        for k in ("json_valid", "sections_ok", "crit12", "fmb_ok"):
            agg[k] += int(c[k])
        # Extract explanation (robust to invalid JSON): between the two boundaries.
        m = re.search(r'"explanation":\s*"(.*?)",\s*"future_meta_behavior"', resp, re.DOTALL)
        expl = m.group(1).replace("<|NULL|>", "").strip() if m else "(not found)"
        gt_obj = json.loads(s["conversations"][1]["value"])
        gt_expl = gt_obj.get("explanation", "")
        print(f"\n{'='*100}\n[{n}] idx={i} id={s.get('sample_id')} nav={s.get('navigation_command')} "
              f"GTcomplexity={s.get('complexity')}")
        print(f"  json_valid={c['json_valid']} sections_ok={c['sections_ok']} fmb_ok={c['fmb_ok']} "
              f"n_traj_wp={c['n_traj']}")
        print(f"  --- EXPLANATION (model, {len(expl.split())} words) ---\n  {expl}")
        print(f"  --- EXPLANATION (GT) ---\n  {gt_expl}")

    N = len(idxs)
    print(f"\n{'#'*100}\nSUMMARY over {N} samples:")
    for k in ("json_valid", "sections_ok", "crit12", "fmb_ok"):
        print(f"  {k}: {agg[k]}/{N}")


if __name__ == "__main__":
    main()
