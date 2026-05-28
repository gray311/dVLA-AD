"""TrainerCallback: every N steps, run section-diffusion inference on a few
GT-labelled cases and report per-section quality (JSON validity, complexity acc,
fmb verb acc, critical exact-match, trajectory ADE) + one qualitative example.

Generation is forward-only (no collectives) so it is safe under ZeRO-2 where
params are replicated; it runs on every rank but only rank 0 prints.
"""
import json, os, re, sys
import torch
from transformers.trainer_callback import TrainerCallback

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_EVAL_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "eval"))
import template_v3 as T  # noqa: E402

LONG_VERBS = set(T.LONG_VERBS)
LAT_VERBS = set(T.LAT_VERBS)
CRIT = T.CRITICAL_CATEGORIES


def _strip(s):
    return str(s).replace("<|NULL|>", "").strip()


def _first_json(s):
    i = s.find("{")
    if i < 0:
        return None
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return s[i:j + 1]
    return None


def _unwrap(m):
    """Get the underlying Fast_dDrive model exposing mdm_sample_deep_scaffold
    (LoRA layers stay in-place, so its forward still applies the adapter)."""
    seen = 0
    while not hasattr(m, "mdm_sample_deep_scaffold") and seen < 6:
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        elif hasattr(m, "module"):
            m = m.module
        elif hasattr(m, "base_model"):
            m = m.base_model
        elif hasattr(m, "model"):
            m = m.model
        else:
            break
        seen += 1
    return m


def _ade(pred_pairs, gt_pairs):
    n = min(len(pred_pairs), len(gt_pairs))
    if n == 0:
        return None
    d = [((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2) ** 0.5
         for p, g in zip(pred_pairs[:n], gt_pairs[:n])]
    return sum(d) / n


class SectionEvalCallback(TrainerCallback):
    def __init__(self, model, processor, tokenizer, dataset_json, image_root,
                 indices, every=50, threshold=0.85, max_tokens=1024, block_size=32):
        from PIL import Image
        self.proc = processor
        self.tok = tokenizer
        self.every = every
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.block_size = block_size
        self.mask_id = int(tokenizer.encode("|<MASK>|", add_special_tokens=False)[0])
        data = json.load(open(dataset_json))
        self.cases = []
        for i in indices:
            s = data[i]
            imgs = [Image.open(os.path.join(image_root, p)).convert("RGB") for p in s["image"]]
            full = s["conversations"][0]["value"]
            cut = full.find("TEMPLATE (")
            prompt = full[:cut].rstrip() if cut > 0 else full
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": prompt})
            text = processor.apply_chat_template([{"role": "user", "content": content}],
                                                 tokenize=False, add_generation_prompt=True)
            inp = processor(text=[text], images=imgs, return_tensors="pt")
            gt = json.loads(s["conversations"][1]["value"])
            gt_traj = T.parse_filled(gt.get("trajectory", ""))
            self.cases.append({"inp": inp, "gt": gt, "gt_traj": gt_traj,
                               "sid": s.get("sample_id"), "nav": s.get("navigation_command")})

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        model = _unwrap(kwargs["model"])
        if not hasattr(model, "mdm_sample_deep_scaffold"):
            if state.is_world_process_zero:
                print(f"[seceval step {state.global_step}] could not unwrap model; skip", flush=True)
            return
        dev = next(model.parameters()).device
        was_training = model.training
        model.eval()
        rows = []
        with torch.inference_mode():
            for c in self.cases:
                inp = c["inp"]
                kw = dict(input_ids=inp["input_ids"].to(dev), tokenizer=self.tok,
                          block_size=self.block_size, max_tokens=self.max_tokens,
                          mask_id=self.mask_id, threshold=self.threshold)
                if "pixel_values" in inp:
                    kw["pixel_values"] = inp["pixel_values"].to(dev)
                if "image_grid_thw" in inp:
                    kw["image_grid_thw"] = inp["image_grid_thw"].to(dev)
                try:
                    out = model.mdm_sample_deep_scaffold(**kw)
                    resp = self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                except Exception as e:
                    rows.append({"err": str(e)[:60]})
                    continue
                rows.append(self._score(resp, c))
        if was_training:
            model.train()
        if state.is_world_process_zero:
            self._report(state.global_step, rows)

    def _score(self, resp, c):
        r = {"resp": resp, "json": False, "sec": False, "cx": None, "lo": None, "la": None,
             "crit_hit": 0, "ade": None, "expl": ""}
        blob = _first_json(resp)
        obj = None
        if blob:
            try:
                obj = json.loads(blob); r["json"] = True
            except Exception:
                obj = None
        gt = c["gt"]
        if obj:
            r["sec"] = all(k in obj for k in ("critical_objects", "complexity", "explanation",
                                              "future_meta_behavior", "trajectory"))
            r["cx"] = (_strip(obj.get("complexity")) == _strip(gt.get("complexity")))
            fo, go = obj.get("future_meta_behavior", {}), gt.get("future_meta_behavior", {})
            if isinstance(fo, dict):
                r["lo"] = (_strip(fo.get("longitudinal")) == _strip(go.get("longitudinal")))
                r["la"] = (_strip(fo.get("lateral")) == _strip(go.get("lateral")))
            co, gco = obj.get("critical_objects", {}), gt.get("critical_objects", {})
            if isinstance(co, dict):
                r["crit_hit"] = sum(1 for k in CRIT if _strip(co.get(k, "")) == _strip(gco.get(k, "")))
            r["expl"] = _strip(obj.get("explanation", ""))[:160]
        # trajectory ADE (works even if JSON invalid — regex parse)
        pred_traj = T.parse_filled(resp)
        r["ade"] = _ade(pred_traj, c["gt_traj"])
        return r

    def _report(self, step, rows):
        n = len(rows)
        ok = [r for r in rows if "err" not in r]
        jv = sum(r["json"] for r in ok)
        sc = sum(r["sec"] for r in ok)
        cx = [r["cx"] for r in ok if r["cx"] is not None]
        lo = [r["lo"] for r in ok if r["lo"] is not None]
        la = [r["la"] for r in ok if r["la"] is not None]
        crit = sum(r["crit_hit"] for r in ok)
        ades = [r["ade"] for r in ok if r["ade"] is not None]
        def acc(x):
            return f"{sum(x)}/{len(x)}" if x else "n/a"
        print(f"\n{'#'*80}\n[SECEVAL step {step}] n={n}", flush=True)
        print(f"  json_valid={jv}/{n}  sections_ok={sc}/{n}  complexity_acc={acc(cx)}  "
              f"fmb_long_acc={acc(lo)}  fmb_lat_acc={acc(la)}  "
              f"critical_exact={crit}/{12*n}  "
              f"traj_ADE={sum(ades)/len(ades):.2f}({len(ades)})" if ades else
              f"  json_valid={jv}/{n} sections_ok={sc}/{n} complexity_acc={acc(cx)} "
              f"fmb_long={acc(lo)} fmb_lat={acc(la)} critical_exact={crit}/{12*n} traj_ADE=n/a")
        # one qualitative example
        ex = ok[0] if ok else None
        if ex:
            print(f"  e.g. complexity={ex['cx']} fmb=({ex['lo']},{ex['la']}) crit_hit={ex['crit_hit']}/12 "
                  f"ade={ex['ade']}")
            print(f"       explanation: {ex['expl']}")
        print('#'*80, flush=True)
