# GPT-2 Mechanistic Interpretability on SCAN

Training a GPT-2 style model on SCAN to mechanistically investigate how transformers fail at compositional generalization using TransformerLens.

---

## Research Question

**Which internal mechanisms cause transformer models to fail at compositional generalization?**
> Specifically: why do transformer models systematically fail to generate the correct number of repeated actions — a failure pattern observed consistently across all SCAN splits in [T5](https://github.com/suehuynh/scan-compositional-generalization)?
---

## Motivation

T5 exhibits a systematic failure across all SCAN splits: it consistently generates the wrong number of repeated actions, regardless of split difficulty (see [T5 Compositional Analysis Project](https://github.com/suehuynh/scan-compositional-generalization)). This suggests the failure is not about generalization to novel compositions, but about a deeper inability to track and reproduce action repetitions correctly.

Rather than interpreting T5 directly (intractable at scale), we use it as a diagnostic tool — its failure pattern motivates a precise mechanistic question. We train a minimal GPT-2 style model and use TransformerLens to investigate: which internal components are responsible for tracking action repetitions, and why do they fail?

This follows the **model organism methodology**: use a tractable, transparent model to study a phenomenon mechanistically, then reason about what this implies for larger systems.

---

## Approach

```
T5 failure results (Project 2)
        ↓
  Motivation & framing
        ↓
Train small GPT-2 on SCAN
  - Succeeds on simple split
  - Fails on length + addprim splits
        ↓
TransformerLens analysis
  - Logit lens (where does the model commit to wrong token?)
  - Attention patterns (which heads track compositional structure?)
  - Activation patching (which layers/heads are causally responsible?)
        ↓
Causal intervention
  - Patch success activations into failure cases
  - Ablate candidate circuits
        ↓
Findings + write-up
```

---

## Setup

```bash
git clone https://github.com/your-username/gpt2-mech-interp
cd gpt2-mech-interp
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `transformer_lens`, `datasets`, `wandb`, `matplotlib`

---

## Results (In Progress)

| Split | Accuracy |
|-------|----------|
| Simple | TBD |
| Length | TBD |
| AddPrim | TBD |

---

## Related Work

- **Project 2:** T5 failure analysis on SCAN — [https://github.com/suehuynh/scan-compositional-generalization]
- Keysers et al. (2019) — Measuring Compositional Generalization
- Nanda et al. (2023) — Progress measures for grokking via mechanistic interpretability
- Conmy et al. (2023) — Towards Automated Circuit Discovery for Mechanistic Interpretability

---
