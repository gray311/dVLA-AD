# dVLA Stage 1 — SFT Plan

**Scope**: Stage 1 only. Planner refinement (Stage 2) and reasoning-action RL (Stage 3) are referenced but not designed here.

---

## 0. Where Stage 1 sits

The full dVLA pipeline is three stages:

| Stage | What it produces | What it solves |
|---|---|---|
| **1 — dVLM SFT** | A dVLM that emits reasoning + textual trajectory + structured slots | Get a reasoning-capable driving VLA *without reasoning annotation* |
| 2 — Planner | Continuous-precision trajectory from the frozen dVLM | Bridge the 10 cm textual grid to cm-level waypoints |
| 3 — Joint RL | Reasoning aligned to driving outcome | Make reasoning *faithful* and *outcome-relevant* |

Stage 1 ends with a model that **has reasoning** (inherited from the base dVLM) and **can describe a trajectory in text** (newly learned). It does not yet produce cm-precise trajectories (Stage 2) and its reasoning is diverse but not yet outcome-faithful (Stage 3).

---

## 1. Motivation

### 1.1 The data asymmetry in driving VLAs

Every driving dataset ships **trajectory ground truth** (waypoints from the ego log). Almost none ship **reasoning ground truth** — there is no human-labeled "why" behind each maneuver. The standard reasoning-VLA recipe (Alpamayo-R1, AutoVLA) papers over this by generating CoT labels with a large LLM (GPT-4o), then supervising them. This imports the labeler's biases, costs heavily at scale (~700 K samples), and trains the model to *imitate a style* rather than to reason.

### 1.2 The capability asymmetry of the base model

A modern dVLM (Fast-dVLM 3B, inherited from a large vision-language pretrain) is **not uniformly ignorant** across the output template:

| Output | Base zero-shot capability |
|---|---|
| Scene description / explanation | **Strong** — fluent, diverse, grounded |
| Object identification | Moderate |
| Driving meta-decisions (speed, command) | Weak — no driving-specific vocabulary |
| Scene complexity assessment | Weak — task-specific |
| **Trajectory waypoints** | **None** — the model has never seen `0.5s: forward=0.5m` |

The bottleneck is *not* reasoning generation. The base model already describes driving scenes well. The bottleneck is (a) mapping understanding to driving-specific outputs, and (b) the trajectory representation, which is entirely new.

### 1.3 The Stage 1 idea

**Supervise only what the base model cannot already do. Freeze what it can.**

- Trajectory, complexity, meta_speed, meta_command → **supervised** (base can't do these)
- Explanation, critical_objects → **unsupervised**; use the base model's own zero-shot output as a *fixed training context*, not a loss target (base already does these well; updating them risks catastrophic forgetting and diversity collapse)

This is a previously underexplored point on the CoT-SFT spectrum: **frozen-teacher rationale as fixed context, zero loss on the rationale**. It is uniquely viable in driving because the base model's reasoning is already useful — unlike math reasoning, where omitting rationale loss precludes acquiring a skill the base model lacks.

### 1.4 Why a diffusion VLM enables this

Selective slot supervision is architecturally clean in a mask-diffusion LM and awkward in an AR LM:

- **AR LM**: the chain-rule objective ∏ p(yₜ | y_<t) requires a well-defined prefix; dropping the loss on intermediate tokens leaves their distribution unconstrained, and downstream tokens condition on an undefined prefix.
- **Mask-diffusion LM**: every position starts from the same [MASK] state and contributes an independent position-wise loss. Zeroing the loss on some slots is well-defined — those slots still produce hidden states and still participate in bidirectional attention, but receive no direct gradient.

LLaDA already exploits a coarse version of this (prompt visible, only response masked + supervised). We extend it to **slot-level** selective supervision.

### 1.5 What Stage 1 wants: adaptive routing

The same model should behave like a standard VLA on easy scenes (few denoising steps, no reasoning needed) and like a reasoning VLA on hard scenes (more steps, full reasoning), with the **model itself deciding** which via a `complexity` slot. Routing is implicit — cross-attention falls back to vision when reasoning slots are still [MASK], and uses reasoning once they fill.

---

## 2. Architecture (Stage 1 scope)

### 2.1 Single dVLM

```
Input:   [P]  [E]  [V]  +  template (mostly [MASK] at inference)
         prompt ego  vision

dVLM:    K iterative bidirectional forwards → reasoning + textual trajectory
```

Backbone: **Fast-dVLM 3B** (NVlabs Fast-dLLM fork), full bidirectional, SigLIP vision encoder frozen, LoRA rank 32 on the LM. Fast-dVLM 3B inherits AR + diffusion dual-mode from its AR base model; Stage 1 fine-tunes the diffusion behavior while preserving AR capability (needed for Scaffold Speculative Decoding at inference).

The planner is **not trained in Stage 1**. Stage 1 trajectory output is the parsed textual waypoints directly (10 cm grid). Planner refinement is Stage 2.

### 2.2 Template

```
complexity:   simple | complex

critical_objects:  <12 structured fields>
meta_speed:        keep | accel | decel | stop
meta_command:      lane_follow | lane_change_L | turn_L | ...

explanation:  <free-form scene description, ~64 tok>

trajectory:
0.5s: forward=0.5m, lateral=0.0m
1.0s: forward=1.0m, lateral=0.0m
1.5s: forward=1.5m, lateral=-0.3m
2.0s: forward=2.0m, lateral=-0.8m
2.5s: forward=2.5m, lateral=-1.5m
3.0s: forward=3.0m, lateral=-2.0m
```

Trajectory uses semantic-rich textual form — time-indexed, named axes (`forward`/`lateral`), units (`m`), 1-decimal precision. Digit/keyword tokens carry pretrain semantic priors, which lets the trajectory align with the natural-language explanation through shared vocabulary. ~90 tokens.

### 2.3 Supervision and content-source policy

The key table. Note the two distinct columns — **what fills the slot during training** vs **whether it contributes loss**.

| Slot | Training content source | Loss | Mask tier |
|---|---|---|---|
| complexity | heuristic label (agent count + metadata) | **yes** | always mask |
| meta_speed | derived from trajectory geometry | **yes** | always mask |
| meta_command | derived from trajectory geometry | **yes** | tier 2 bounded |
| critical_objects | detection-derived label (from 3D boxes) | **no** | tier 2 bounded |
| explanation | **base dVLM zero-shot generation** (frozen, pre-computed) | **no** | tier 3 direct |
| trajectory | ego-log GT, textual format | **yes** | always mask |

The unsupervised slots (explanation, critical_objects) are **filled with content during training** — not left as [MASK]. This is the fix for the train-inference gap: if explanation were always [MASK] in training, the trajectory would never learn to attend to explanation content, and the K-large reasoning path would be inert at inference. By filling explanation with the base model's own zero-shot text, the trajectory learns to condition on realistic explanation content, and because there is no loss on explanation, the fine-tuned model's explanation distribution stays close to the base — matching what the trajectory saw in training.

### 2.4 Three-tier mask schedule

$$\text{tier 1 (always): mask all tokens, independent of } \rho$$
$$\text{tier 2 (bounded): } p_{\text{mask}} = \text{floor} + (1-\text{floor})\cdot\rho$$
$$\text{tier 3 (direct): } p_{\text{mask}} = \rho$$

`floor`: meta_command 0.5, critical_objects 0.3. Tier 1 = inference-critical outputs (always start from MASK at inference). Tier 2 floors prevent small slots from learning trivial copy. Tier 3 is standard mask diffusion.

This is a discrete special case of Fast-dDrive's section-adaptive Beta noise schedule; an ablation in §6 compares the discrete tiers against parameterized Beta(α_s, β_s).

### 2.5 Complexity slot — no attention isolation

The complexity slot routes K and is **not** attention-isolated: it attends to reasoning slots so its prediction evolves as reasoning fills across denoising steps. With isolation, complexity confidence would be fixed at step 0 and the dynamic halting in §4 would collapse to a static decision. Risk (complexity cheating from full-content reasoning at train time) is mitigated by multi-ρ training covering the full mask range, including ρ = 1 (reasoning fully masked).

---

## 3. Training

### 3.1 Pre-computation: base explanations and labels

One-time preprocessing over `dvlm-ad_v1.2.json`:

```python
for sample in dataset:
    # base zero-shot explanation, frozen model, prompt forbids future prediction
    sample.explanation = base_dvlm.generate(
        sample.image,
        prompt="Describe the current driving scene: objects and events relevant "
               "to driving decisions. Do NOT predict future actions.",
        max_tokens=64, temperature=0.7)

    sample.critical_objects = derive_from_detection(sample.boxes)   # 12 fields
    sample.complexity       = heuristic_complexity(sample.metadata) # simple/complex
    sample.meta_speed, sample.meta_command = derive_from_trajectory(sample.traj)
    sample.trajectory_text  = format_textual(sample.traj)           # 0.5s: ...
```

Two filters on base explanations (STaR-style quality control, no self-training):
- **Grounding**: regenerate if explanation is not grounded in the image.
- **No leakage**: regenerate if explanation predicts future actions (would make the trajectory loss trivial).

Output: `dvlm-ad_v1.2_prepared.json`. Compute cost is one-time; explanation depends only on (P, E, V), not on ρ, so it is fixed across the M multi-ρ passes.

### 3.2 Multi-ρ stratified sampling

$$L(x) = \frac{1}{M}\sum_{m=1}^{M}\ell(x;\rho_m), \qquad \rho_m \sim \text{Uniform}\!\left(\left[\tfrac{m-1}{M},\tfrac{m}{M}\right]\right)$$

Each sample draws M stratified ρ values; the M dVLM forwards share the frozen SigLIP encoding (effective cost ≈ 0.85 M ×). Corner emphasis: ρ = 0 and ρ = 1 each oversampled ~25 % so the K = 1 (no reasoning) and K = large (full reasoning) regimes are both trained.

Three benefits: variance ↓ (1/M per ρ position); train-inference alignment (model sees the full mask spectrum it will traverse at inference); zero-extra-compute coverage of the difficulty range.

| Substage | M |
|---|---|
| 1a | 2 |
| 1b, 1c | 4 |

### 3.3 Training step

```python
def step(sample):
    vision = siglip(sample.image)               # frozen, cached across M passes
    template = {
        'complexity':       sample.complexity,        # supervised
        'meta_speed':       sample.meta_speed,        # supervised
        'meta_command':     sample.meta_command,      # supervised
        'critical_objects': sample.critical_objects,  # content yes, loss no
        'explanation':      sample.explanation,       # base-generated, loss no
        'trajectory':       sample.trajectory_text,   # supervised
    }
    L = 0
    for rho in stratified_rho(M):
        masked = apply_tiered_mask(template, rho)
        H = dvlm(P, E, V, masked, vision_cache=vision)   # no complexity isolation

        L_dvlm = ( w_cx  * ce(H.complexity,      sample.complexity)
                 + w_sp  * ce(H.meta_speed,      sample.meta_speed)
                 + w_cm  * ce_masked(H.command,  sample.meta_command)
                 + w_tau * ce_masked(H.trajectory, sample.trajectory_text) )
        # NO loss on explanation or critical_objects
        L += L_dvlm
    return L / M
```

Loss weights normalized by token count. Optionally, a small KL anchor on explanation logits against the frozen base (weight ≤ 0.02) can be switched on if monitoring (§5) shows diversity collapse — off by default.

### 3.4 Schedule

`dvlm-ad_v1.2.json` is a unified internal corpus pre-formatted to the template. Substages load different shards:

| Substage | Data | M | Weeks |
|---|---|---|---|
| 1a | v1.2 initial split (sanity) | 2 | 3–4 |
| 1b | v1.2 full | 4 | 6–8 |
| 1c | + NAVSIM-aligned subset, self-distilled complexity labels | 4 | 6–8 |

8 × H100, ~16–20 weeks. The 3B backbone and the single curated JSON (no per-dataset adapters) keep this shorter than an 8B multi-corpus plan.

### 3.5 Pre-flight checklist

- `dvlm-ad_v1.2_prepared.json` built; schema covers all template slots
- Base-explanation generation run + audited (grounding, no future-leakage, diversity)
- Heuristic complexity labels balanced ~50/50
- Tokenizer sanity check on textual trajectory (`0.5s: forward=0.5m, lateral=0.0m`)
- Tiered mask function unit-tested on 100 samples
- Multi-ρ stratified sampling implemented
- AR capability of base Fast-dVLM 3B verified (needed for SSD downstream)
- Loss weights normalized by token count
- Eval pipeline: textual-trajectory ADE, complexity accuracy, supervised-slot accuracy, emergence metrics for unsupervised slots
- Halting-margin calibration script

---

## 4. Inference — Dynamic Halting

K is not fixed in advance. After each denoising step the model re-reads the complexity slot and halts when the simple–complex margin clears a threshold.

```python
def infer(P, E, V, K_max=10, halt_threshold=0.4):
    vision = siglip(image)
    L = [MASK] * L_len
    H = dvlm(P, E, V, L, vision_cache=vision)

    for k in range(1, K_max + 1):
        L = unmask_step(L, H, rho=schedule(k, K_max))
        H = dvlm(P, E, V, L, vision_cache=vision)
        p = softmax(lm_head(H.complexity))
        if p[simple_id] - p[complex_id] > halt_threshold:
            break

    tau_text = decode(lm_head(H.trajectory))
    return parse_waypoints(tau_text)     # Stage 1 stops here; Stage 2 planner refines
```

**Routing is a continuous spectrum**, not a binary switch: K = 1–2 ≈ standard-VLA (reasoning still mostly [MASK]); K = K_max ≈ reasoning-VLA (full reasoning conditions the trajectory). Training's multi-ρ coverage is what makes the trajectory valid at any K.

**Cost** = K dVLM forwards, K ∈ {1, …, K_max}. `halt_threshold` calibrated on val to target E[K] ≈ 3–4.

Compatibility with Scaffold Speculative Decoding: complexity and trajectory slots must be committed **last** so the complexity margin keeps updating across rounds and the trajectory is unaffected until halting decides.

---

## 5. Stage 1 Success Criteria and Monitoring

Stage 1 is judged on four axes, checked from substage 1a onward.

### 5.1 Trajectory quality (the supervised target)
- Textual-trajectory ADE on val (parsed waypoints, pre-planner)
- Target: within reach of a fixed-K = 10 baseline; the 10 cm grid caps precision — Stage 2 closes the rest

### 5.2 Routing behaves
- Per-complexity-bucket K usage: simple scenes should halt early (K ≈ 1–2), complex late (K ≈ 8–10)
- If both buckets use similar K → routing inert → revisit complexity loss weight or label quality

### 5.3 Unsupervised slots do not collapse
The central risk of zero-loss explanation. Monitor every checkpoint:
- explanation distinct-n, self-BLEU (template-collapse detector)
- LLM-judge coherence on 200 samples
- **task-relevance**: JS divergence of explanation bag-of-words between simple and complex buckets; JS > 0.3 indicates explanation content varies with scene
- critical_objects emergence accuracy vs detection GT (random ≈ 5 %)

Decision rule:
- coherent + reasonably diverse → proceed (expected outcome, since explanation tracks the frozen base)
- template collapse → switch on KL anchor (weight 0.02)
- garbage → escalate to weak GPT-4o supervision on a 50 K subset

### 5.4 No catastrophic forgetting
- AR generation sanity on 50 held-out prompts (also guards SSD acceptance rate)
- General VQA / scene-description quality vs the base model — should not regress

---

## 6. Stage 1 Ablations (run at 1c)

- **Trajectory representation**: semantic-rich textual (default) vs codebook (2 token, 64×64) vs raw coordinates — impact on explanation alignment and ADE
- **Explanation content source**: base zero-shot (default) vs all-[MASK] vs GPT-4o pseudo-label — validates that base-as-context is necessary and sufficient
- **Multi-ρ**: M = 1 vs 2 vs 4 — variance-reduction claim
- **Mask schedule**: discrete tiers (default) vs Beta(α_s, β_s) per slot — vs Fast-dDrive formalism
- **Complexity attention**: no isolation (default, enables dynamic halting) vs isolated (static K)
- **Halting**: dynamic margin (default) vs fixed K = 5 vs static step-0 decision
- **Selective supervision**: drop each slot's loss one at a time → trajectory ADE / explanation diversity / critical_objects accuracy. This is a self-contained study of *which slots benefit from supervision* — a question with no systematic prior work in the mask-diffusion setting.

---

## 7. Risks (Stage 1)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Explanation collapses despite base-as-context | Low–Med | Per-epoch diversity monitor; KL anchor fallback |
| Trajectory ignores explanation (Mode A) | Med | Hard-case mix audit; counterfactual loss option in 1c; truly resolved in Stage 3 RL |
| Base explanation leaks future → trivial trajectory loss | Med | Prompt forbids prediction; leakage filter in preprocessing |
| Complexity cheats from content at train time | Med | Multi-ρ covers ρ = 1; verify complexity accuracy at ρ = 1 |
| Halting threshold mis-calibrated → wrong E[K] | Med | Val-set calibration; per-bucket K reporting |
| AR capability lost under LoRA → SSD degrades | Med | AR sanity check per checkpoint; aux AR loss (0.05) if needed |
| Train-inference gap on explanation distribution | Low | By construction: base-as-context training matches near-base inference output |
| Bidirectional dVLM serving slow | High | Report FLOPs primarily; wall-clock secondary with infra disclaimer |

---

## 8. Stage 1 Decisions Locked

| Decision | Choice |
|---|---|
| Backbone | Fast-dVLM 3B, LoRA rank 32, SigLIP frozen |
| Trajectory representation | Semantic-rich textual `0.5s: forward=0.5m, lateral=0.0m` |
| Explanation / critical_objects | Unsupervised; base zero-shot / detection-derived content as fixed training context |
| Supervised slots | complexity, meta_speed, meta_command, trajectory |
| Mask schedule | Three-tier (always / bounded / direct) |
| ρ sampling | Multi-ρ stratified, M = 2 → 4, corner emphasis |
| Complexity attention | No isolation (enables dynamic halting) |
| K decision | Dynamic margin halting on complexity slot |
| Planner | Not in Stage 1 — deferred to Stage 2 |
| RL | Not in Stage 1 — deferred to Stage 3 |
| Training data | `dvlm-ad_v1.2.json`, pre-formatted to template |

---

**Version**: Stage 1 plan v1.0  
**Boundaries**: Stage 2 (planner refinement) and Stage 3 (reasoning-action RL) are separate documents.