# GPT-2 Mechanistic Interpretability of Compositional Generalization Failure

A research repository investigating how and why a small transformer
model fails at compositional generalization on the SCAN benchmark,
using mechanistic interpretability tools (logit lens, attention
analysis, activation patching) to identify the specific failure modes
and their underlying circuits.

---

**Abstract.** A minimal transformer trained on SCAN fails predominantly through semantic
action substitution, not repetition miscounting, the assumption this line of work often
starts from. The model's attention correctly identifies the modifier tokens relevant to
avoiding these failures, and does so at least as strongly in failure cases as in success
cases, yet no single tested intervention, head ablation or MLP activation patching,
restores correct behavior. Validated across three SCAN splits and five independently
trained seeds, these results indicate the failure is neither a retrieval problem nor
attributable to a single circuit.

📄 **Full write-up:** [Paper (Google Drive)](https://drive.google.com/file/d/1wChfmvwN-EkWDqjVe6DuJVpCH-PfalK7/view?usp=sharing)
Written up as a paper submitted to the NewInML Workshop @ NeurIPS 2026 (non-archival,
decision pending).

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

## Contributions

- A **validated, grammar-aware error taxonomy** showing that semantic substitution errors
  dominate over repetition-counting errors, roughly two to three times as common across
  every split and seed tested.
- Cross-seed evidence that **single-head circuit claims in small transformers can be
  fragile**: a head identified as the primary modifier-tracking circuit in one training run
  does not replicate as such across four additional independently seeded runs.
- A **null result on attention magnitude as a causal signal**: the head most consistently
  associated with failure under ablation is not the head with the highest attention score.
- A **null result on coarse activation patching**: substituting averaged success-case MLP
  activations into failure cases decreases accuracy in every condition tested, rather than
  restoring it.

---

## Key Finding

Across all three SCAN splits and five random seeds, the dominant
failure mode is **not** repetition-counting. Failures decompose as:

| Error Type | Description | Simple | Length | AddPrim |
|---|---|---|---|---|
| A — Repetition-count | Right actions, right order, wrong count | ~27% | ~29% | ~24% |
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
Classifies every failure case into one of four error types (A/B/C/D
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
```
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
```
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

- This model organism is a minimal, from-scratch transformer trained specifically to
exhibit compositional generalization failure on SCAN. No claim is made that its internal
computation corresponds to the failure mechanisms of any larger or differently trained
model.

- No single attention head's association with the target behavior replicates as a fixed
circuit across seeds; an initial single-seed analysis identified one head as the primary
modifier-tracking circuit, and this claim did not hold once checked against four
additional independently seeded models. This is treated as a central result of the
project, not a caveat to a different result, and it motivates the cross-seed validation
applied throughout.

- The MLP activation patching intervention substitutes a single activation, averaged across
all verified success cases and an entire sequence, into a failure case's forward pass.
This conflates whether the relevant computation is transferable at all with whether a
single averaged representation can stand in for it; the reported result speaks only to the
latter. Position-specific patching, restricted to the token positions where a substitution
error occurs, is a natural next step.

- Only single-head ablation and whole-layer MLP patching are tested. Neither restores
correct behavior, but this does not establish that no finer-grained intervention would.

---

## Experiment Tracking

Training runs logged at:
[wandb.ai/suehuynh/gpt2-mech-interp](https://wandb.ai/suehuynh/gpt2-mech-interp)