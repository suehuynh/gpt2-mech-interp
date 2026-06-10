import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from jaxtyping import Float, Int
from config.config import *
from src.data import *

import torch
from torch import Tensor
import torch.nn as nn
import einops
from transformer_lens.utils import gelu_new

# config = Config()
# dm = SCANDataModule(config)
# d_vocab = len(dm.tokenizer)

# ========= EMBEDDING =========

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.d_vocab, cfg.d_model) # [batch_size, seq_len, d_model]
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) # [seq_len, d_model]
    def forward(self, token_ids: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        seq_len = token_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=self.cfg.device)
        return self.token_embed(token_ids) + self.pos_emb(pos_ids)  # [batch_size, seq_len, d_model]

# ========= LAYER NORM ========
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model)) # [d_model]
        self.b = nn.Parameter(torch.zeros(cfg.d_model)) # [d_model]

    def forward(self, 
                resid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        resid_mean = resid.mean(dim=-1, keepdim=True) # [batch, posn, 1]
        resid_std = (resid.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt() # [batch, posn, 1]

        resid = (resid - resid_mean) / resid_std # [batch, posn, 1]
        return resid * self.w + self.b # [batch, posn, d_model]
    
# ========= ATTENTION ========

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", torch.tensor(float("-inf"), dtype=torch.float32, device=cfg.device))
        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.b_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head))
        self.b_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head))
        self.b_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
    
    def forward(self, 
                normalized_resid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        q = (
            einops.einsum(
                normalized_resid, 
                self.W_Q, 
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid,
                self.W_K,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid,
                self.W_V,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head"
            )
            + self.b_V
        )

        # Calculate attention_scores
        attention_scores = einops.einsum(
            q,
            k,
            "batch posn_Q n_heads d_head, batch posn_K n_heads d_head -> batch n_heads posn_Q posn_K"
        )
        # Scale
        attention_scores = attention_scores / self.cfg.d_head**0.5
        # Apply casual mask
        attention_scores = self.apply_casual_mask(attention_scores)
        # Softmax
        attention_pattern = attention_scores.softmax(-1)

        # Weighted average of values
        z = einops.einsum(
            v,
            attention_pattern,
            "batch posn_K n_heads d_head, batch n_heads posn_Q posn_K -> batch posn_Q n_heads d_head"
        )

        # Ouput
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q n_heads d_head, n_heads d_head d_model -> batch posn_Q  d_model"
                )
                + self.b_O
            )
        return attn_out

    def apply_casual_mask(
            self,
            attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        The casual mask function in the Attention class will mask attention scores matrix so the model can only attend to previous position
        """
        all_ones = torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=self.cfg.device)
        # True to all probability we want set to 0
        mask = torch.triu(all_ones, diagonal=1).bool()
        attn_scores_masked = attn_scores.masked_fill(mask, self.IGNORE)
        return attn_scores_masked

# ========= MLP ========

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))
        self.b_in = nn.Parameter(torch.empty(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.empty(cfg.d_model))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
    
    def forward(self, 
                normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        pre = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
                ) 
                + self.b_in
            )
        post = gelu_new(pre)

        mlp_out = (
            einops.einsum(
                post,
                self.W_out,
                "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
            ) + self.b_out
        )
        return mlp_out
    
# ========= TRANSFORMER BLOCK ========
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        normalized_resid = self.ln1(resid_pre)
        attn_resid = self.attn(normalized_resid)
        resid_mid = attn_resid + resid_pre

        normalized_resid_mid = self.ln2(resid_mid)
        mlp_resid = self.mlp(normalized_resid_mid)
        resid_post = mlp_resid + resid_mid

        return resid_post

# ========= UNEMBEDDING =========
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros(cfg.d_vocab), requires_grad=True)
    
    def forward(self, normalized_resid_post: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_vocab"]:
        u = (
            einops.einsum(
                normalized_resid_post,
                self.W_U,
                "batch posn d_model, d_model d_vocab -> batch posn d_vocab"
            )
            + self.b_U
        )
        return u

# ========= TRANSFORMER =========
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.tfblocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_vocab"]:
        resid = self.embed(tokens) 
        for block in self.tfblocks:
            resid = block(resid)
        normalized_resid_final = self.ln(resid)
        logits = self.unembed(normalized_resid_final)
        return logits
        
# ========= EXECUTION =========

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TRANSFORMER COMPONENTS")
    print("=" * 60)
    config = Config()
    dm = SCANDataModule(config)

    # Load a batch
    batch = next(iter(dm.get_train_loader()))
    token_ids = batch["input_ids"].to(config.device)
    batch_size, seq_len = token_ids.shape
    
    print(f"\nInput shape: {token_ids.shape} (batch_size={batch_size}, seq_len={seq_len})")
    
    # Test Embed
    print("\n[1] Testing Embed...")
    embed = Embed(config)
    embed_out = embed(token_ids)
    assert embed_out.shape == (batch_size, seq_len, config.d_model)
    print(f"Embed output shape: {embed_out.shape}")
    
    # Test LayerNorm
    print("\n[2] Testing LayerNorm...")
    ln = LayerNorm(config)
    ln_out = ln(embed_out)
    assert ln_out.shape == embed_out.shape
    print(f"LayerNorm output shape: {ln_out.shape}")
    
    # Test Attention
    print("\n[3] Testing Attention...")
    attn = Attention(config)
    attn_out = attn(ln_out)
    assert attn_out.shape == (batch_size, seq_len, config.d_model)
    print(f"Attention output shape: {attn_out.shape}")
    
    # Test MLP
    print("\n[4] Testing MLP...")
    mlp = MLP(config)
    mlp_out = mlp(ln_out)
    assert mlp_out.shape == (batch_size, seq_len, config.d_model)
    print(f"MLP output shape: {mlp_out.shape}")
    
    # Test TransformerBlock
    print("\n[5] Testing TransformerBlock...")
    block = TransformerBlock(config)
    block_out = block(embed_out)
    assert block_out.shape == embed_out.shape
    print(f"TransformerBlock output shape: {block_out.shape}")
    
    # Test Unembed
    print("\n[6] Testing Unembed...")
    unembed = Unembed(config)
    unembed_out = unembed(ln_out)
    assert unembed_out.shape == (batch_size, seq_len, config.d_vocab)
    print(f"Unembed output shape: {unembed_out.shape}")
    
    # Test Full Transformer
    print("\n[7] Testing Full Transformer...")
    model = Transformer(config)
    logits = model(token_ids)
    assert logits.shape == (batch_size, seq_len, config.d_vocab)
    print(f"Transformer output shape: {logits.shape}")
    
    # Test gradient flow
    print("\n[8] Testing Gradient Flow...")
    loss = logits.sum()
    loss.backward()
    has_grad = model.embed.token_embed.weight.grad is not None
    assert has_grad
    print(f"Gradients flowing through model")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)