import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxtyping import Float, Int
from config.config import *
from data import *
from model import Transformer

import torch
from torch import Tensor
import torch.nn as nn
import wandb

class Trainer:
    def __init__(self, cfg: Config):
        """Initialize trainer with config"""
        wandb.init(project="gpt2-mech-interp", config=cfg.__dict__)
        self.cfg = cfg
        self.model = Transformer(self.cfg)
        self.model = self.model.to(device=self.cfg.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.dm = SCANDataModule(self.cfg)
        self.train_loader = self.dm.get_train_loader()
        self.test_loader = self.dm.get_test_loader()
    
    def compute_loss(
            self,
            logits: Float[Tensor, "batch seq_len vocab_size"],
            token_ids: Int[Tensor, "batch seq_len"]
        ) -> Float[Tensor, ""]:
        """
        Compute cross-entropy loss
        Shift logits and labels by 1 (predict next token)
        """
        logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]) # [batch*(seq_len-1), vocab_size]
        targets = token_ids[:, 1:].reshape(-1) # [batch*(seq_len-1)]
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss
    
    def train_epoch(self) -> Float[Tensor, ""]:
        """
        Single training epoch
        Returns: average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            token_ids = batch["input_ids"].to(self.cfg.device)
            logits = self.model(token_ids)
            loss = self.compute_loss(logits, token_ids)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 5 == 0:  # log every 5 batches
                wandb.log({"loss": loss.item()})

        return total_loss / len(self.train_loader)
    
    def evaluate(self, split) -> Float[Tensor, ""]:
        """
        Evaluate model on a specific SCAN split (simple, length, addprim)
        Returns: accuracy and average loss
        """
        self.model.eval()
        test_loader = self.dm.get_test_loader(split)
        with torch.no_grad():
            accuracy = []
            total_loss = []
            all_preds = []
            all_targets = []
            for batch in test_loader:
                token_ids = batch["input_ids"].to(self.cfg.device)
                logits = self.model(token_ids)
                loss = self.compute_loss(logits, token_ids)
                total_loss.append(loss)
                preds = torch.argmax(logits[:, :-1, :], dim=-1)  # [batch, seq_len-1]

                all_preds.append(preds)
                all_targets.append(token_ids[:, 1:])

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)

            # Mask padding tokens
            pad_id = self.dm.tokenizer.token2id["<pad>"]
            mask = (all_targets != pad_id)
            accuracy = (all_preds[mask] == all_targets[mask]).float().mean()
            avg_loss = torch.stack(total_loss).mean()
            # accuracy = (all_preds == all_targets).float().mean()
            # avg_loss = torch.stack(total_loss).mean()
        return accuracy, avg_loss
    
    def log_failure_cases(self) -> None:
        """
        Log failure cases to results/ folder
        Format: input | predicted output | expected output
        """
        self.model.eval()
        test_loaders = self.dm.get_all_test_loaders()
        for split_name, test_loader in test_loaders.items():
            with open(f"results/failure_cases_{split_name}.txt", "w") as f:
                for batch in test_loader:
                    token_ids = batch["input_ids"].to(self.cfg.device)
                    logits = self.model(token_ids)

                    preds = torch.argmax(logits[:, :-1, :], dim=-1) # [batch, seq_len-1]
                    targets = token_ids[:, 1:] # [batch, seq_len-1]
                    failed_mask = (preds != targets)  # [batch, seq_len-1]
                    for ex_idx in range(failed_mask.shape[0]):  
                        if failed_mask[ex_idx].any():  # if this example has failures
                            input_text = self.dm.tokenizer.decode(token_ids[ex_idx])
                            pred_text = self.dm.tokenizer.decode(preds[ex_idx])  
                            target_text = self.dm.tokenizer.decode(targets[ex_idx]) 
                            
                            f.write(f"[{split_name}] INPUT: {input_text} | TARGET: {target_text} | PRED: {pred_text}\n")

    def train(self) -> None:
        """
        Main training pipeline
        """
        for epoch in range(self.cfg.epochs):
            epoch_loss = self.train_epoch()
            for split_name in self.cfg.scan_split.keys():
                accuracy, avg_loss = self.evaluate(split_name,)
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    f"eval_accuracy_{split_name}": accuracy,
                    f"eval_loss_{split_name}": avg_loss,
                })
            if epoch % 5 == 0 or epoch == self.cfg.epochs - 1:  # save every 5 + final
                torch.save(self.model.state_dict(), f"results/checkpoint_epoch_{epoch}.pt")
        self.log_failure_cases()
        torch.save(self.model.state_dict(), f"results/model_final.pt")

if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()