import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import requests
from torch.utils.data import Dataset, DataLoader
from config.config import Config

# ========= LOAD DATASET FROM GITHUB =========

config = Config()

def fetch_scan_split(folder: str, split_name: str, train: bool) -> list[str]:
    prefix = "train" if train else "test"
    url = f"{config.base_url}/{folder}/tasks_{prefix}_{split_name}.txt"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # A truncated download (network hiccup, proxy cutoff, etc.) would
    # otherwise silently yield a short split with no error, poisoning
    # every downstream accuracy/failure-count number computed from it.
    # Content-Length only describes the on-the-wire (possibly
    # compressed) size, so this check only applies to uncompressed
    # responses — for gzip/deflate transfers, requests/urllib3 already
    # raise on a truncated stream during decompression, so no separate
    # check is needed there.
    content_length = response.headers.get("Content-Length")
    if (
        content_length is not None
        and "Content-Encoding" not in response.headers
        and len(response.content) != int(content_length)
    ):
        raise RuntimeError(
            f"Truncated download for {url}: expected {content_length} bytes, "
            f"got {len(response.content)}"
        )

    lines = response.text.strip().split("\n")
    if not lines or lines == [""]:
        raise RuntimeError(f"Empty response body for {url}")

    return lines

# ========= TOKENIZE COMMANDS AND ACTIONS =========

class SCANTokenizer:
    def __init__(self, dataset):
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "unk": 3}
        for row in dataset:
            for token in row.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.token2id = vocab
        self.id2token = {v: k for k, v in vocab.items()}
    
    def encode(self, text):
        return [self.token2id.get(token, 3) for token in text.split()]

    def decode(self, ids):
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        return " ".join(self.id2token.get(i, "<unk>") for i in ids)

    def __len__(self):
        return len(self.token2id)

# ========= LOAD DATASET =========

class SCANDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer: SCANTokenizer, max_seq_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        line = self.examples[idx]
        tokens = (
            [self.tokenizer.token2id["<sos>"]]
            + self.tokenizer.encode(line)
            + [self.tokenizer.token2id["<eos>"]]
        )
        tokens = tokens[:self.max_seq_len]
        tokens += [self.tokenizer.token2id["<pad>"]] * (self.max_seq_len - len(tokens))
        return {"input_ids": torch.tensor(tokens, dtype=torch.long)}

# ========= LOAD TRAIN AND TEST DATA =========
     
class SCANDataModule:
    def __init__(self, config):
        # Tokenize from ONLY simple train split
        folder, split_name = config.scan_split["simple"]
        train_examples = fetch_scan_split(folder, split_name, train=True)
        self.tokenizer = SCANTokenizer(train_examples)
        self.train_dataset = SCANDataset(train_examples, self.tokenizer, config.max_seq_len)
        self.test_splits = {}
        for split_key, (folder, split_name) in config.scan_split.items():
            test_examples = fetch_scan_split(folder, split_name, train=False)
            self.test_splits[split_key]  = SCANDataset(
                test_examples,  
                self.tokenizer, 
                config.max_seq_len
                )
        self.config = config

    def get_train_loader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def get_test_loader(self, split: list = "simple") -> DataLoader:
        return DataLoader(self.test_splits[split], batch_size=self.config.batch_size, shuffle=False)
    
    def get_all_test_loaders(self) -> dict[str, DataLoader]:
        """Get all test loaders for evaluation"""
        return {
            split: self.get_test_loader(split) 
            for split in self.test_splits.keys()
        }
# ========= EXECUTION =========

if __name__ == "__main__":

    config = Config()
    dm = SCANDataModule(config)

    print(f"Vocab size: {len(dm.tokenizer)}")
    print(f"Train size: {len(dm.train_dataset)}")

    test_size = 0
    for split_key in dm.test_splits:
        test_size += len(dm.test_splits[split_key])
    print(f"Test size: {test_size} for {len(dm.test_splits)} splits")

    batch = next(iter(dm.get_train_loader()))
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Sample decoded: {dm.tokenizer.decode(batch['input_ids'][0].tolist())}")