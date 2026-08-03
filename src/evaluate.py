import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace
from jaxtyping import Float, Int
from config.config import Config
from src.data import SCANDataModule
from src.model import Transformer
import json
import statistics
import torch
from torch import Tensor

class Evaluator:
    def __init__(self, cfg: Config):
        """Initialize evaluator with trained model"""
        self.cfg = cfg
        self.dm = SCANDataModule(cfg)
        self.model = Transformer(cfg)
        self.model.load_state_dict(torch.load(f"{self.cfg.results_dir}/model_seed{self.cfg.seed}.pt"))
        self.model.eval()
        self.model.to(cfg.device)
        print("Model loaded!")

    def generate(self, command: str, max_new_tokens: int = 100) -> str:
        """
        Feed command prefix and autoregressively generate actions
        command: already contains <sos>, e.g. "<sos> IN: jump OUT:"
        """
        input_ids = torch.tensor(
            self.dm.tokenizer.encode(command),
            dtype=torch.long
        ).unsqueeze(0).to(self.cfg.device)

        pred_ids = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if exceeds max_seq_len
                input_truncated = input_ids[:, -self.cfg.max_seq_len:]

                logits = self.model(input_truncated)[:, -1, :]
                next_token = torch.argmax(logits, dim=-1)

                if next_token.item() == self.dm.tokenizer.token2id["<eos>"]:
                    break

                pred_ids.append(next_token.item())
                input_ids = torch.cat(
                    [input_ids, next_token.unsqueeze(-1)], dim=-1
                )

        return self.dm.tokenizer.decode(pred_ids)

    def evaluate_split(self, split: str, max_samples: int = None) -> tuple[float, float]:
        """
        Evaluate model on a specific SCAN split using proper generation
        Returns: exact_match_accuracy, token_level_accuracy

        max_samples: if set, stop after evaluating this many examples
        (used for a quick smoke test before running the full split).
        """
        test_loader = self.dm.get_test_loader(split)
        exact_matches = 0
        token_correct = 0
        token_total = 0
        total = 0

        os.makedirs(self.cfg.failure_cases_dir, exist_ok=True)
        failure_path = f"{self.cfg.failure_cases_dir}/failure_cases_{split}_seed{self.cfg.seed}.txt"
        with open(failure_path, "w") as f:
            for batch in test_loader:
                token_ids = batch["input_ids"]

                for ex_idx in range(token_ids.shape[0]):
                    if max_samples is not None and total >= max_samples:
                        break

                    # Decode full sequence
                    full_sequence = self.dm.tokenizer.decode(
                        token_ids[ex_idx].tolist()
                    )

                    if "OUT:" not in full_sequence:
                        continue

                    # Split into command and target actions
                    parts = full_sequence.split("OUT:")
                    command = parts[0].strip() + " OUT:"  # "<sos> IN: jump ... OUT:"
                    target_actions = parts[1].strip()

                    # Clean target — remove <eos> and <pad>
                    target_tokens = [
                        t for t in target_actions.split()
                        if t not in ["<eos>", "<pad>"]
                    ]
                    target_actions = " ".join(target_tokens)

                    # Generate actions
                    pred_actions = self.generate(command)
                    pred_tokens = pred_actions.split()


                    matches = sum(
                        p == t for p, t in zip(pred_tokens, target_tokens)
                    )
                    token_correct += matches
                    token_total += len(target_tokens)

                    # Exact match
                    if pred_actions.strip() == target_actions.strip():
                        exact_matches += 1
                    else:
                        f.write(
                            f"[{split}] COMMAND: {command} | "
                            f"TARGET: {target_actions} | "
                            f"PRED: {pred_actions}\n"
                        )
                    total += 1

                if max_samples is not None and total >= max_samples:
                    break

        # Guard against a silently truncated/short split (e.g. a bad
        # network fetch in data.py, or an interrupted run) poisoning the
        # accuracy numbers without any indication something's wrong.
        if max_samples is None:
            expected_total = len(self.dm.test_splits[split])
            if total != expected_total:
                raise RuntimeError(
                    f"Evaluated {total}/{expected_total} examples for split "
                    f"'{split}' (seed={self.cfg.seed}) — the split appears "
                    "truncated or the run was interrupted. Re-run before "
                    "trusting these metrics."
                )

        exact_acc = exact_matches / total if total > 0 else 0
        token_acc = token_correct / token_total if token_total > 0 else 0
        return exact_acc, token_acc

    def evaluate_all(self, max_samples: int = None) -> dict[str, dict[str, float]]:
        """Evaluate on all splits, return both metrics.

        max_samples: if set, evaluate only this many examples per split
        (smoke test mode) instead of the full test split.
        """
        results = {}
        for split in self.cfg.scan_split.keys():
            print(f"Evaluating {split} (seed={self.cfg.seed})...")
            exact_acc, token_acc = self.evaluate_split(split, max_samples=max_samples)
            print(f"{split}: exact_match = {exact_acc:.4f} | token_accuracy = {token_acc:.4f}")
            results[split] = {
                "exact_match": exact_acc,
                "token_accuracy": token_acc
            }

        os.makedirs(self.cfg.eval_metrics_dir, exist_ok=True)
        out_path = f"{self.cfg.eval_metrics_dir}/eval_metrics_seed{self.cfg.seed}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {out_path}")
        return results


def aggregate_seed_results(per_seed_results: dict[int, dict]) -> dict:
    """
    Aggregate per-seed eval_metrics dicts (as returned by evaluate_all())
    into mean/std per split per metric across seeds.

    per_seed_results: {seed: {split: {metric: value}}}
    Returns: {split: {metric: {"mean": ..., "std": ..., "seeds": {seed: value}}}}
    """
    seeds = sorted(per_seed_results.keys())
    splits = next(iter(per_seed_results.values())).keys()
    metrics = next(iter(next(iter(per_seed_results.values())).values())).keys()

    summary = {}
    for split in splits:
        summary[split] = {}
        for metric in metrics:
            values = [per_seed_results[seed][split][metric] for seed in seeds]
            summary[split][metric] = {
                "mean": statistics.mean(values),
                # std of a single seed is undefined; report 0.0 rather than error
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "seeds": {seed: per_seed_results[seed][split][metric] for seed in seeds},
            }
    return summary


def run_all_seeds(base_cfg: Config, max_samples: int = None) -> dict:
    """Evaluate every seed in base_cfg.seeds on every split, then write
    the per-seed JSONs plus one aggregated mean+/-std summary JSON."""
    per_seed_results = {}
    for seed in base_cfg.seeds:
        cfg = replace(base_cfg, seed=seed)
        evaluator = Evaluator(cfg)
        per_seed_results[seed] = evaluator.evaluate_all(max_samples=max_samples)

    summary = aggregate_seed_results(per_seed_results)
    os.makedirs(base_cfg.eval_metrics_dir, exist_ok=True)
    summary_path = f"{base_cfg.eval_metrics_dir}/eval_metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")
    return summary

# ========= EXECUTION =========

if __name__ == "__main__":
    base_cfg = Config()

    # --- Smoke test: 2 samples/split, sanity-check the pipeline before
    # print("=" * 60)
    # print("SMOKE TEST (2 samples per split, all seeds)")
    # print("=" * 60)
    # smoke_summary = run_all_seeds(base_cfg, max_samples=2)
    # print(json.dumps(smoke_summary, indent=2))

    # --- Full run across all splits, all seeds ---
    print("=" * 60)
    print("FULL EVALUATION RUN")
    print("=" * 60)
    full_summary = run_all_seeds(base_cfg, max_samples=None)
    print(json.dumps(full_summary, indent=2))