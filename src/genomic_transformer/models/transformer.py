"""Small pre-LN transformer encoder classifier for tokenized DNA windows.

Vocabulary is the prepare_ctcf encoding: A=0, C=1, G=2, T=3, N=4. The model
embeds tokens, adds a learned positional embedding, runs a stack of pre-norm
{attention, feed-forward} blocks, applies a final LayerNorm, mean-pools over
positions, and projects to a single logit for binary classification.
"""

from dataclasses import dataclass
from typing import Any, Union

import torch
from torch import nn

VOCAB_SIZE = 5
PAD_TOKEN = 4  # N doubles as the padding token; windows are fixed-length


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 256
    dropout: float = 0.1
    pooling: str = "mean"
    max_len: int = 1000

    @classmethod
    def from_dict(cls, config: Union[dict[str, Any], "TransformerConfig"]) -> "TransformerConfig":
        if isinstance(config, cls):
            return config
        return cls(**config)


class EncoderBlock(nn.Module):
    """Pre-LN block: attention sublayer then feed-forward sublayer."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class SmallTransformer(nn.Module):
    """Configurable small encoder; forward(input_ids [B, L]) -> logit [B]."""

    def __init__(self, config: Union[dict[str, Any], TransformerConfig]) -> None:
        super().__init__()
        config = TransformerConfig.from_dict(config)
        if config.pooling != "mean":
            raise ValueError(f"pooling must be 'mean', got {config.pooling!r}")
        self.config = config
        self.token_embed = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)
        self.blocks = nn.ModuleList(
            EncoderBlock(config) for _ in range(config.n_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, 1)
        self.pooling = config.pooling

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [batch, length], got shape {tuple(input_ids.shape)}"
            )
        batch, length = input_ids.shape
        if length > self.config.max_len:
            raise ValueError(
                f"sequence length {length} exceeds max_len {self.config.max_len}"
            )
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # mean-pool over positions
        return self.head(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
