# dVLA Stage-1 SFT — Resume / Reproduction Guide

How to reproduce or continue Stage-1 SFT, from data → backbone → training → eval.
Big artifacts (data, model weights, checkpoints) live on **weka scratch**, not in
git. Shorthands used below:

```
SC=/weka/home/ext-yingzima/scratchaszalay1_ssci/yy          # our scratch (models, data, ckpts)
SRC=/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/dvlm    # source CoT JSONs
WAYMO=/weka/home/ext-yingzima/scratchcxiao13/yingzi/workspace/waymo # source images
PYNAV=$SC/conda_envs/navsim/bin/python   # has transformers (for data/tokenizer steps)
DLLM=/home/ext-yingzima/miniconda3/envs/dllm/bin   # training env (torch/deepspeed)
```

---

## 0. Environment

- **Training env**: conda `dllm` (torch 2.9.1+cu128, transformers 4.57.1, deepspeed 0.18.5).
  `torchao` was upgraded 0.9.0 → **0.17.0** (PEFT 0.19 LoRA needs ≥0.16). sglang pins
  torchao==0.9 but still imports under 0.17.
- `pkg_resources` was removed in setuptools 82 → `lmflow/utils/versioning.py` patched to
  use `importlib.metadata`.
- **GPUs: L40S ×4. GPU 1 is hardware-degraded** (uncorrectable ECC, Xid 48/64, ECC
  row-remapper exhausted). **Skip it** — launchers accept `INCLUDE_GPUS="0,2,3"`
  (deepspeed `--num_gpus` ignores `CUDA_VISIBLE_DEVICES`, so use `--include`).

---

## 1. Data  ← start here

### 1a. Source schema
- `$SRC/dvlm-ad_waymo_training_cot.json` (29,550, full GT), `…_e2e_val_cot.json` (479,
  masked eval scaffold + GT future waypoints), `…_e2e_test_cot.json` (1505, no GT).
- Images: `$WAYMO/{train,val}/<scene>/<idx>_CAM_{FRONT_LEFT,FRONT,FRONT_RIGHT}.jpg`
  plus a **pre-made `<idx>_CAM_JOINT.jpg`** panorama (front_left|front|front_right).

### 1b. Convert → V3 template  (`data/data_convert.py`)
Prompt + template come from the canonical **`eval/template_v3.py`** (imported, single
source of truth). Run with a python that has `transformers` (for critical NULL-padding):

```bash
# train: GT-filled V3, critical values NULL-padded to 2 tokens (--pad_critical default on)
$PYNAV data/data_convert.py --split train \
  --src $SRC/dvlm-ad_waymo_training_cot.json \
  --out $SC/data/dvla_sft/dvlm-ad_waymo_training_v3.json
# val / test: masked V3 scaffold (no GT slots; GT future waypoints carried for ADE)
$PYNAV data/data_convert.py --split eval --src $SRC/dvlm-ad_waymo_e2e_val_cot.json  --out $SC/data/dvla_sft/dvlm-ad_waymo_e2e_val_v3.json
$PYNAV data/data_convert.py --split eval --src $SRC/dvlm-ad_waymo_e2e_test_cot.json --out $SC/data/dvla_sft/dvlm-ad_waymo_e2e_test_v3.json
```

V3 response slots (key order = critical_objects, complexity, explanation,
future_meta_behavior, trajectory):
- `critical_objects`: 12 categories, **≤2-token phrase | "none"**, NULL-padded to exactly
  2 tokens (matches the 2-token inference scaffold; pedestrian phrase = "person").
- `complexity`: heuristic simple/complex (1 tok). *(train skewed ~35/65, threshold tuning TBD)*
- `explanation`: from source CoT (loss-free slot content).
- `future_meta_behavior`: {longitudinal, lateral} 2-word verbs.
- `trajectory`: **semantic** `"<t>s: forward=<sign><tens><ones>.<frac>m, lateral=<...>m\n…"`,
  10 wp @ 0.5 s (5 s). forward signed, ≥100 m → 3 int digits.
- Mask token = **`|<MASK>|`** (single id 151665; NOT the legacy `<|mdm_mask|>`).

### 1c. Single joined image  (`data/join_images.py`)
Switch the 3 front views → one `_CAM_JOINT.jpg` (already exist on disk; this just rewrites
the `image` field). Tradeoff: ~1/3 the image tokens; raise `MAX_PIXELS` to keep detail.

```bash
for n in training:_training e2e_val:_e2e_val e2e_test:_e2e_test; do :; done   # see below
$PYNAV data/join_images.py --json $SC/data/dvla_sft/dvlm-ad_waymo_training_v3.json \
  --out $SC/data/dvla_sft/dvlm-ad_waymo_training_v3_joint.json --image_root $WAYMO --workers 16
# repeat for e2e_val / e2e_test
```

**Training data = `$SC/data/dvla_sft/dvlm-ad_waymo_training_v3_joint.json`** (single image).
`data/example/` has a few committed sample cases + images for smoke tests.

---

## 2. Backbone  (`$SC/models/Fast_dVLM_3B_sasd`)

Base = **Fast-dVLM 3B** (`Efficient-Large-Model/Fast_dVLM_3B`, block-diffusion VLM from
Qwen2.5-VL-3B — NOT the driving-finetuned Fast-dDrive). Downloaded to `$SC/models/Fast_dVLM_3B`.

The **SASD backbone** `$SC/models/Fast_dVLM_3B_sasd` reuses Fast-dDrive's proven SASD
modeling with Fast-dVLM base weights:
- Fast-dVLM weights **reverse key-renamed** to Fast-dDrive layout
  (`model.language_model.*`→`model.*`, `model.visual.*`→`visual.*`; 825 keys match) →
  single `model.safetensors`.
- Fast-dDrive `modeling.py` / `configuration.py` / `generation_utils.py` + tokenizer.
- **`section_utils.py` rewritten for V3** (5 sections incl. complexity; semantic-trajectory;
  `build_deep_json_scaffold` 2-tok critical slots). Two local patches in its `modeling.py`:
  `fused_flex_attention` passes smaller `kernel_options` (L40S/Ampere shared-mem limit), and
  it is the model the launchers point to.

Verify: `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` → 0 missing/0 unexpected.

---

## 3. Train

```bash
PATH=$DLLM:$PATH \
INCLUDE_GPUS="0,2,3" MASTER_PORT=11011 \
NUM_TRAIN_EPOCHS=2 LEARNING_RATE=1e-5 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=4 \
LOGGING_STEPS=10 SAVE_STEPS=500 SAVE_TOTAL_LIMIT=4 \
EVAL_EVERY=50 EVAL_INDICES="0,3000,6000,9000,12000,15000,18000,21000,24000,27000" \
OUTPUT_DIR=$SC/checkpoints/dvla_sft_stage1_sasd \
bash train_scripts/train_v3_sasd.sh
```

- `train_v3_sasd.sh` = SASD (deep scaffold + section loss weights + per-section Beta noise);
  `train_v3_mdm.sh` = plain MDM (no sections). Both default to the joint train, LoRA r32,
  frozen vision, bd_size 32, block_size 4096.
- **Section weights** (configurable via `SECTION_LOSS_WEIGHTS`): trajectory 3.0 / fmb 2.0 /
  critical 1.5 / complexity 2.0 / explanation 1.0. To switch to the plan's **selective
  supervision**, set explanation/critical_objects to 0 — no code change needed.
- **Noise** (`SECTION_NOISE_SCHEDULE`): per-section Beta(α,β) — traj 2,1 / fmb 1,1.5 /
  critical 1,2 / complexity 1,2 / expl 1,1.
- Loss = MDM (masked response value tokens, section-weighted) + AR/causal (clean stream,
  all response tokens, ×1.0; preserves AR for Scaffold-Spec). Mask ratio is per-block from
  the section Beta; **one noise draw per example** (plan's multi-ρ not implemented).
- Resume: re-run the same command (HF Trainer picks up the latest checkpoint in OUTPUT_DIR;
  pass `--resume_from_checkpoint` if needed).

### 50-step per-section eval (`train_scripts/section_eval_callback.py`)
`EVAL_EVERY=50` runs section-diffusion inference on `EVAL_INDICES` cases every 50 steps and
logs: json_valid, sections_ok, complexity_acc, fmb_long/lat_acc, critical_exact, **trajectory
ADE**, + one explanation example. Watch: `grep -A6 "SECEVAL step" $OUTPUT_DIR/train.log`.

---

## 4. Eval / inference (pre-train sanity)

```bash
# single case
PATH=$DLLM:$PATH python run_pretrain_infer.py --sample_json data/example/v3_train_sample_joint.json --case 0
# batch + per-section validity/garbage checks + explanation-vs-GT
PATH=$DLLM:$PATH python run_multi_infer.py --indices 0,8000,20000
```
Both use section-diffusion decode, `threshold=0.85`, `max_tokens=1024`. Note:
`mdm_sample_deep_scaffold` keeps frozen scaffold tokens fixed but **strips leftover
MASK/NULL** in post-process — incomplete denoising (low confidence / step cap) collapses
structure; raise max_tokens / lower threshold so denoising completes.

---

## 5. Status & next

- **Real SASD training was launched** (GPUs 0,2,3, 2 epochs) and confirmed healthy
  (deepspeed init + steps + 50-step SECEVAL). Resume with the §3 command.
- Pre-training the untrained model: fluent **explanation** (base strength), but
  trajectory/structure degenerate — exactly the skill SFT must teach.
- Open items: (a) **fix GPU 1** (hardware); (b) complexity label balance (~35/65);
  (c) optional **selective supervision** (expl/critical weight 0) + **multi-ρ** (not yet
  implemented) to reach the full plan §2–3 recipe; (d) migrate `eval/` ADE/RFS scoring to
  the semantic-trajectory format (`template_v3.parse_filled` already handles it).

See `plan.md` for the full Stage-1 design.
