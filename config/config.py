from dataclasses import dataclass
import torch
@dataclass
class Config:
    # Data
    base_url: str = "https://raw.githubusercontent.com/brendenlake/SCAN/master"

    scan_split = {
    "simple": ("simple_split","simple"),
    "length": ("length_split","length"),
    "addprim_jump": ("add_prim_split", "addprim_jump"),
    }
    max_seq_len: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    d_model: int = 768
    d_head: int = 64
    d_mlp: int = 3072
    n_layers: int = 12
    n_heads: int = 12
    init_range: float = 0.02
    layer_norm_eps: float = 1e-5
    n_ctx: int = 1024

    # Training
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 10
    weight_decay: float = 1e-2

    # Logging
    wandb_project: str = "gpt2-mech-interp"