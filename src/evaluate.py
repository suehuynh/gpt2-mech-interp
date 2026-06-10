# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import torch
# from config.config import Config
# from data import SCANDataModule
# from model import Transformer

# class Evaluator:
#     def __init__(self, cfg: Config):
#         """Initialize evaluator with trained model"""
#         self.cfg = cfg
#         self.dm = SCANDataModule(cfg)
#         self.model = Transformer(cfg)
#         self.model.load_state_dict(torch.load("results/model_final.pt"))
#         self.model.eval()
#         self.model.to(cfg.device)

#     def generate(self, command: str, max_new_tokens: int = 100) -> str:
#         """
#         Feed command prefix and autoregressively generate actions
#         command: raw string e.g. "IN: jump around left twice OUT:"
#         """
#         input_ids = torch.tensor(
#             self.dm.tokenizer.encode(f"<sos> {command}"),
#             dtype=torch.long
#         ).unsqueeze(0).to(self.cfg.device)

#         pred_ids = []

#         with torch.no_grad():
#             for _ in range(max_new_tokens):
#                 # Truncate if exceeds max_seq_len
#                 input_truncated = input_ids[:, -self.cfg.max_seq_len:]
                
#                 logits = self.model(input_truncated)[:, -1, :]
#                 next_token = torch.argmax(logits, dim=-1)

#                 if next_token.item() == self.dm.tokenizer.token2id["<eos>"]:
#                     break

#                 pred_ids.append(next_token.item())
#                 input_ids = torch.cat(
#                     [input_ids, next_token.unsqueeze(-1)], dim=-1
#                 )

#         return self.dm.tokenizer.decode(pred_ids)

#     def evaluate_split(self, split: str) -> float:
#         """
#         Evaluate model on a specific SCAN split using proper generation
#         Returns accuracy
#         """
#         test_loader = self.dm.get_test_loader(split)
#         correct = 0
#         total = 0

#         with open(f"results/failure_cases_{split}.txt", "w") as f:
#             for batch in test_loader:
#                 token_ids = batch["input_ids"]

#                 for ex_idx in range(token_ids.shape[0]):
#                     full_sequence = self.dm.tokenizer.decode(
#                         token_ids[ex_idx].tolist()
#                     )

#                     if "OUT:" not in full_sequence:
#                         continue

#                     parts = full_sequence.split("OUT:")
#                     command = parts[0].strip() + " OUT:"
#                     target_actions = parts[1].strip()
#                     target_actions = " ".join([
#                         t for t in target_actions.split()
#                         if t not in ["<sos>", "<eos>", "<pad>"]
#                     ])

#                     pred_actions = self.generate(command)

#                     if pred_actions.strip() == target_actions.strip():
#                         correct += 1
#                     else:
#                         f.write(
#                             f"[{split}] COMMAND: {command} | "
#                             f"TARGET: {target_actions} | "
#                             f"PRED: {pred_actions}\n"
#                         )
#                     total += 1

#         accuracy = correct / total if total > 0 else 0
#         return accuracy

#     def evaluate_all(self) -> dict[str, float]:
#         """Evaluate on all splits"""
#         results = {}
#         for split in self.cfg.scan_split.keys():
#             print(f"Evaluating {split}...")
#             accuracy = self.evaluate_split(split)
#             print(f"{split}: accuracy = {accuracy:.4f}")
#             results[split] = accuracy
#         return results

# # ========= EXECUTION =========

# if __name__ == "__main__":
#     cfg = Config()
#     evaluator = Evaluator(cfg)
#     results = evaluator.evaluate_all()

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxtyping import Float, Int
from config.config import Config
from src.data import SCANDataModule
from src.model import Transformer

import torch
from torch import Tensor

class Evaluator:
    def __init__(self, cfg: Config):
        """Initialize evaluator with trained model"""
        self.cfg = cfg
        self.model_name = model_name
        self.dm = SCANDataModule(cfg)
        self.model = Transformer(cfg)
        self.model.load_state_dict(torch.load(f"results/model_final_{self.model_name}.pt"))
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

    def evaluate_split(self, split: str) -> tuple[float, float]:
        """
        Evaluate model on a specific SCAN split using proper generation
        Returns: exact_match_accuracy, token_level_accuracy
        """
        test_loader = self.dm.get_test_loader(split)
        exact_matches = 0
        token_correct = 0
        token_total = 0
        total = 0

        with open(f"results/{self.model_name}_failure_cases_{split}.txt", "w") as f:
            for batch in test_loader:
                token_ids = batch["input_ids"]

                for ex_idx in range(token_ids.shape[0]):
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

        exact_acc = exact_matches / total if total > 0 else 0
        token_acc = token_correct / token_total if token_total > 0 else 0
        return exact_acc, token_acc

    def evaluate_all(self) -> dict[str, dict[str, float]]:
        """Evaluate on all splits, return both metrics"""
        results = {}
        for split in self.cfg.scan_split.keys():
            print(f"Evaluating {split}...")
            exact_acc, token_acc = self.evaluate_split(split)
            print(f"{split}: exact_match = {exact_acc:.4f} | token_accuracy = {token_acc:.4f}")
            results[split] = {
                "exact_match": exact_acc,
                "token_accuracy": token_acc
            }
        return results

# ========= EXECUTION =========

if __name__ == "__main__":
    cfg = Config()
    model_name = "small" 
    evaluator = Evaluator(cfg)
    results = evaluator.evaluate_all()