# GPT-2 Mechanistic Interpretability of Compositional Generalization Failure

A research repository investigating how and why a small transformer
model fails at compositional generalization on the SCAN benchmark,
using mechanistic interpretability tools (logit lens, attention
analysis, activation patching) to identify the specific failure modes
and their underlying circuits.

---

## Research Question

**What causes a minimal transformer to fail at compositional
generalization on SCAN, and is the dominant failure mode a
repetition-counting error, or something else?**

Prior self-led project (and some literature) frames transformer failures on
SCAN as primarily a repetition-counting problem — the model gets the
right actions but the wrong number of repeats. This project tests
that assumption directly, using a controlled model organism, before
building any mechanistic explanation on top of it.

---

## Key Finding

Across all three SCAN splits and five random seeds, the dominant
failure mode is **not** repetition-counting. Failures decompose as:

| Error Type | Description | Simple | Length | AddPrim |
|---|---|---|---|---|
| A1 — Repetition-count | Right actions, right order, wrong count | ~27% | ~29% | ~24% |
| **B — Semantic substitution** | **Wrong action generated for a clause** | **~69%** | **~68%** | **~72%** |
| C — Clause omission | Trailing clause dropped entirely | ~3% | ~2% | ~3% |
| D — Over-generation | Extra trailing tokens beyond target | ~1% | <1% | ~1% |

**Action-substitution errors dominate**, roughly 2.5x more common than
clean repetition-counting errors, across every split tested. See
`notebooks/00_error_pattern_analysis.ipynb` for the full
classification methodology and validation against hand-labeled
ground truth.

This finding directly shaped the mechanistic analysis in this
repository: rather than assuming a "repetition-counting circuit" is
the primary object of study, later notebooks investigate why the
model's attention correctly identifies task-relevant tokens (modifier
words like *twice*, *thrice*) while still frequently generating the
wrong action.

---

## Model

A GPT-2 style decoder-only transformer trained from scratch on SCAN's
simple split:

| Hyperparameter | Value |
|---|---|
| Layers | 2 |
| Attention heads | 4 |
| Model dimension | 128 |
| MLP dimension | 512 |
| Head dimension | 32 |
| Vocabulary size | 25 |
| Context length | 128 |
| Total parameters | 419,609 |

Trained with AdamW (lr=1e-4, weight decay=1e-2, batch size=32, 10
epochs) across **5 random seeds** (42, 101, 345, 2834, 10101) for
cross-seed validation of all mechanistic claims below.

---

## Evaluation

Autoregressive generation (`IN: [command] OUT:` → generated action
tokens), evaluated by exact sequence match and token-level accuracy.
See `results/eval_metrics.json` for the exact numbers behind Table 2
of the accompanying paper.

---

## Mechanistic Analysis — Notebooks

Findings below are validated across all 3 SCAN splits (simple,
length, addprim_jump) and all 5 seeds unless noted.

### `00_error_pattern_analysis.ipynb`
Classifies every failure case into one of four error types (A1/B/C/D
above) using a grammar-aware, per-clause comparison method. Validated
against 33 hand-labeled examples (27/33 exact match; remaining 5 are
documented boundary-ambiguity limitations, see notebook for detail).
Establishes the empirical premise for all subsequent analysis.

### `01_logit_lens_analysis.ipynb`
Applies the logit lens during autoregressive generation to localize
*where* in the network failures originate. Identifies two failure
modes:
- **Layer 0 Dissolution** (~55–76% of failures): the correct token
  never appears at Layer 0's output
- **Layer 1 Overwrite** (~20–45%): Layer 0 predicts correctly, Layer
  1 overwrites with the wrong token

### `02_attention_patterns.ipynb`
Analyzes modifier-token attention (*twice*, *thrice*, *around*,
*after*, *opposite*) across all 8 attention heads. Key findings:
- **No single head is a fixed "modifier tracker" across random
  seeds** — the top-attending head varies (an earlier single-seed
  finding pointing to one specific head did not replicate)
- Layer 0 attends to modifiers more than Layer 1 in 4 of 5 seeds
- Modifier attention is **preserved, and often higher, in failure
  cases** than in success cases — the model is not failing because
  it ignores relevant tokens
- L0H0 shows the most consistent failure-associated attention shift
  across all seeds and splits

### `03_activation_patching.ipynb`
Tests causal hypotheses via head ablation and MLP activation
patching:
- Ablating L0H0 produces a small, consistent effect on failure-case
  accuracy across all seeds/splits (interpreted as behavioral
  disruption, not genuine improvement — see notebook discussion)
- Patching averaged success-case MLP activations into failure cases
  **decreases** accuracy in every condition tested — MLP
  computations are highly context-specific and do not transfer via
  simple averaging
- L1 MLP patching is consistently less disruptive than L0 MLP
  patching, suggesting L0 encodes more input-specific computation

---

## Repository Structure

├── config/
│ └── config.py # Model, training, and seed configuration
├── notebooks/
│ ├── 00_error_pattern_analysis.ipynb
│ ├── 01_logit_lens_analysis.ipynb
│ ├── 02_attention_patterns.ipynb
│ └── 03_activation_patching.ipynb
├── src/
│ ├── data.py # SCAN loading and tokenization
│ ├── model.py # GPT-2 implementation from scratch
│ ├── train.py # Multi-seed training pipeline
│ └── evaluate.py # Seq2seq evaluation with generation
├── results/
│ ├── model_seed{42,101,345,2834,10101}.pt
| ├── eval_metrics/
│ ├── failure_cases/
│ ├── failure_patterns/
│ ├── ablation/
| ├── patching/ 
│ └── logit_lens_cross_seed_summary.json
├── figures/
└── README.md
---

## Reproducing Results

```bash
pip install -r requirements.txt
python src/train.py         # trains all 5 seeds (see config.py)
python src/evaluate.py       # generates results/eval_metrics.json and per-seed failure logs
jupyter notebook notebooks/00_error_pattern_analysis.ipynb
jupyter notebook notebooks/01_logit_lens_analysis.ipynb
jupyter notebook notebooks/02_attention_patterns.ipynb
jupyter notebook notebooks/03_activation_patching.ipynb
```

All model checkpoints for the 5 seeds used in this analysis are
included in `results/` for exact reproducibility without retraining.

---

## Limitations

- The grammar-based clause parser (Notebook 00) has one documented
  edge case: compound "after"-clauses combining "around" and
  "thrice" modifiers can produce a token-count mismatch
  (`PARSE_MISMATCH`), empirically rare (0% across all seeds/splits
  tested here, but not proven to be zero in general — see notebook
  for detail).
- MLP activation patching uses averaged activations across success
  cases; this is a coarse intervention that conflates "MLP
  computation is context-specific" with "wholesale averaging
  destroys signal." Position-specific patching is a natural next
  step (see paper's Future Work).
- Whole-Layer-0 ablation was tested as a sanity check only; given the
  model's minimal 2-layer depth, this intervention is too blunt to
  isolate any specific circuit's causal role (see appendix in
  accompanying paper).

---

## Experiment Tracking

Training runs logged at:
[wandb.ai/suehuynh/gpt2-mech-interp](https://wandb.ai/suehuynh/gpt2-mech-interp)