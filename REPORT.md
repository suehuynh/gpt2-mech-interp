# GPT-2 Mechanistic Interpretability on SCAN

A mechanistic interpretability study investigating how small transformer 
models fail at compositional generalization, using the SCAN dataset as 
a controlled benchmark.

---

## Research Question

**Which internal mechanisms cause transformer models to fail at 
compositional generalization?**

> Specifically: why do transformer models systematically fail to generate 
> the correct number of repeated actions — a failure pattern observed 
> consistently across all SCAN splits in 
> [T5 Compositional Analysis](https://github.com/suehuynh/scan-compositional-generalization)?

---

## Motivation

T5 exhibits a systematic failure across all SCAN splits: it consistently 
generates the wrong number of repeated actions, regardless of split 
difficulty. This suggests the failure is not about generalization to novel 
compositions, but about a deeper inability to track and reproduce action 
repetitions correctly.

Rather than interpreting T5 directly (intractable at scale), we use the 
**model organism methodology**: train a minimal GPT-2 style model on SCAN, 
replicate the failure pattern, and use TransformerLens to mechanistically 
investigate which internal components are responsible.

---

## Approach
T5 failure results (Project 2)
↓
Motivation & framing
↓
Train small GPT-2 on SCAN (2L, 4H, 128 d_model)

Trained on simple split
Evaluated on simple, length, addprim splits
↓
TransformerLens analysis (3 notebooks)
01: Logit Lens    → WHERE does failure happen?
02: Attention     → WHICH heads track modifiers?
03: Patching      → WHICH circuits cause failure?
↓
Findings + write-up

---

## Model

Small GPT-2 style transformer trained from scratch:

| Hyperparameter | Value |
|---------------|-------|
| Layers | 2 |
| Heads | 4 |
| d_model | 128 |
| d_mlp | 512 |
| d_head | 32 |
| d_vocab | 25 |
| max_seq_len | 128 |
| Trained on | Simple split only |

---

## Results

### Evaluation (Seq2Seq Generation)

| Split | Exact Match | Token Accuracy |
|-------|-------------|----------------|
| Simple | 33.9% | 85.1% |
| Length | 32.2% | 86.5% |
| AddPrim | 38.6% | 86.6% |

---

## Phase 3: Mechanistic Interpretability Findings

### Notebook 01: Logit Lens Analysis

**Question:** Where in the model does the failure happen?

| Split | Early Failure (L1 Dissolution) | Late Failure (L2 Overwrite) | Mean Error Onset |
|-------|-------------------------------|----------------------------|-----------------|
| Simple | 61.3% | 38.7% | step 9.19 |
| Length | 55.3% | 44.7% | step 18.50 |
| AddPrim | 61.0% | 39.0% | step 9.64 |

**Key Finding:** Two distinct failure modes identified:
- **Layer 1 Dissolution (~60%):** Failure originates in early layers — 
  foundational features fail to preserve structural context
- **Layer 2 Overwrite (~40%):** Layer 1 computes correctly but Layer 2 
  overwrites with wrong prediction
- **Length split** shows delayed failure onset (step 18.50) — model 
  holds on longer before collapsing on longer sequences

---

### Notebook 02: Attention Pattern Analysis

**Question:** Which heads track modifier tokens ('twice', 'thrice')?

**Modifier Attention Scores (Simple Split):**

| Head | Success | Failure | Difference |
|------|---------|---------|------------|
| L0H0 | 0.0691 | 0.0724 | -0.0033 |
| L0H1 | 0.0186 | 0.0219 | -0.0033 |
| L0H2 | 0.0777 | 0.0637 | **+0.0140** |
| L0H3 | 0.0513 | 0.0540 | -0.0027 |
| L1H0 | 0.0094 | 0.0086 | -0.0008 |
| **L1H1** | **0.1150** | **0.1155** | -0.0005 |
| L1H2 | 0.0072 | 0.0062 | +0.0010 |
| L1H3 | 0.0318 | 0.0347 | -0.0029 |

**Cross-Split L1H1 Modifier Attention:**

| Split | Success | Failure | Difference |
|-------|---------|---------|------------|
| Simple | 0.1150 | 0.1155 | -0.0005 |
| Length | 0.0954 | 0.1129 | -0.0175 |
| AddPrim | 0.1261 | 0.1169 | +0.0092 |

**Key Findings:**
- **H1 ✅ Confirmed:** L1H1 is the primary modifier-tracking head — 
  consistently highest modifier attention across all splits
- **H2 ❌ Rejected:** Modifier attention is preserved in failure cases —
  the model attends to 'twice'/'thrice' equally in success and failure
- **H3 ✅ Partially confirmed:** L1H1 shows diagonal attention pattern 
  suggesting previous-token tracking for repetition counting
- **New finding:** L0H2 shows largest attention drop in AddPrim failures 
  (+0.0226), suggesting primitive-specific structural encoding
- **New finding:** Length split shows L1H1 attention HIGHER in failures — 
  model compensates on longer sequences but still fails

**Core insight:** The model correctly SEES modifiers but fails to USE 
them — failure is downstream of attention.

---

### Notebook 03: Activation Patching & MLP Ablation

**Question:** Which downstream MLP circuits fail to translate modifier 
attention into correct repetition counts?

#### Section 3: L1H1 Ablation

| Split | Baseline | Ablated (L1H1) | Delta | Ablated (L0H2) | Delta |
|-------|----------|----------------|-------|----------------|-------|
| Simple | 0.00% | 1.92% | -0.019 | 3.44% | -0.034 |
| Length | 0.04% | 2.63% | -0.026 | — | — |
| AddPrim | 0.02% | 1.65% | -0.016 | — | — |

#### Section 4: MLP Activation Patching

| Split | Baseline | L0 Patched | L1 Patched |
|-------|----------|------------|------------|
| Simple | 69.76% | 31.97% | 50.29% |
| Length | 75.05% | 33.12% | 52.26% |
| AddPrim | 70.28% | 30.68% | 49.83% |

#### Section 5: Recovery Rates

| Split | L0 Recovery | L1 Recovery |
|-------|-------------|-------------|
| Simple | 0.1% | 3.1% |
| Length | 0.0% | 4.3% |
| AddPrim | 0.1% | 2.8% |

**Key Findings:**
- **H4 ⚠️ Partially confirmed:** Ablating L1H1 changes behavior 
  across all splits — negative delta (slight improvement) suggests 
  L1H1 contributes to structural tracking but is not the sole 
  repetition-counting circuit. L0H2 shows larger ablation effect.
- **H5 ❌ Rejected:** Patching average success MLP activations 
  DECREASES token accuracy (~20% drop for L1, ~40% for L0) — 
  MLP activations are highly context-specific and non-transferable
- **H6 ⚠️ Partially supported:** L1 MLP consistently more 
  transferable than L0 MLP across all splits, but neither recovers 
  failures meaningfully
- **Core finding:** Failures are localized (70-75% tokens correct) 
  but MLP activations are too context-specific to patch — the 
  failure mechanism is distributed rather than localized to one circuit

---

## Summary of Hypotheses

| Hypothesis | Status | Key Evidence |
|------------|--------|--------------|
| H1: Certain heads attend strongly to modifiers | ✅ Confirmed | L1H1 dominant modifier-tracking head across all splits |
| H2: Layer 2 Overwrite = failed modifier attention | ❌ Rejected | Modifier attention preserved in failure cases |
| H3: Some heads track previous actions | ✅ Partial | L1H1 diagonal pattern suggests repetition tracking |
| H4: Ablating L1H1 increases failure rate | ⚠️ Partial | Changes behavior but negative delta — not primary circuit |
| H5: L1 MLP patching restores behavior | ❌ Rejected | Patching hurts (~20% drop) — activations context-specific |
| H6: L1 MLP is primary failure circuit | ⚠️ Partial | L1 more transferable than L0 but recovery near-zero |

---

## Core Research Finding

> **The model correctly attends to modifier tokens (L1H1, ~70-75% 
> token accuracy on failure cases) but fails to translate this 
> attention into correct repetition counts at specific generation 
> steps. MLP activations are highly context-specific — averaged 
> success activations corrupt rather than fix failure computations. 
> The failure mechanism appears distributed across circuits rather 
> than localized to a single component.**

---

## Setup

```bash
git clone https://github.com/suehuynh/gpt2-mech-interp
cd gpt2-mech-interp
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `transformer_lens`, `datasets`, 
`wandb`, `matplotlib`, `circuitsvis`, `pandas`, `seaborn`

---

## Repository Structure
gpt2-mech-interp/
├── config/
│   └── config.py              # Model and training configuration
├── notebooks/
│   ├── 01_logit_lens_analysis.ipynb      # Phase 3: Logit lens
│   ├── 02_attention_patterns.ipynb       # Phase 3: Attention analysis
│   └── 03_activation_patching.ipynb      # Phase 3: MLP ablation
├── src/
│   ├── data.py                # SCAN data loading and tokenization
│   ├── model.py               # GPT-2 implementation from scratch
│   ├── train.py               # Training pipeline with W&B logging
│   └── evaluate.py            # Seq2seq evaluation with generation
├── results/
│   ├── model_final_small.pt   # Trained model weights
│   ├── small_failure_cases_simple.txt
│   ├── small_failure_cases_length.txt
│   └── small_failure_cases_addprim_jump.txt
└── README.md
---

## Related Work

- **Project 2:** T5 failure analysis on SCAN — 
  [scan-compositional-generalization](https://github.com/suehuynh/scan-compositional-generalization)
- Keysers et al. (2019) — Measuring Compositional Generalization
- Nanda et al. (2023) — Progress measures for grokking via 
  mechanistic interpretability
- Conmy et al. (2023) — Towards Automated Circuit Discovery for 
  Mechanistic Interpretability

---

## Experiment Tracking

All training runs logged at: 
[wandb.ai/suehuynh/gpt2-mech-interp](https://wandb.ai/suehuynh/gpt2-mech-interp)

---

## Citation

```bibtex
@misc{huynh2026gpt2scan,
  title={Mechanistic Interpretability of Compositional 
         Generalization Failure in GPT-2 on SCAN},
  author={Huynh, Sue},
  year={2026},
  url={https://github.com/suehuynh/gpt2-mech-interp}
}
```