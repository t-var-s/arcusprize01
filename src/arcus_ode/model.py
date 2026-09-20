"""Minimal pre-LayerNorm GPT implementation matching the released checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.size()
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head
        query = query.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        attention = attention.masked_fill(
            self.bias[:, :, :sequence_length, :sequence_length] == 0, float("-inf")
        )
        attention = self.attn_dropout(F.softmax(attention, dim=-1))
        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.resid_dropout(self.c_proj(output))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self, token_ids: torch.Tensor, *, capture_residuals: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        _, sequence_length = token_ids.size()
        if sequence_length > self.config.block_size:
            raise ValueError(f"sequence length {sequence_length} exceeds block size {self.config.block_size}")
        positions = torch.arange(sequence_length, dtype=torch.long, device=token_ids.device)
        residual = self.transformer["drop"](
            self.transformer["wte"](token_ids) + self.transformer["wpe"](positions)
        )
        residuals: list[torch.Tensor] = []
        for block in self.transformer["h"]:
            residual = block(residual)
            if capture_residuals:
                residuals.append(residual.detach())
        logits = self.lm_head(self.transformer["ln_f"](residual))
        return logits, residuals


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, dict]:
    """Load weights without executing arbitrary pickle globals."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    fields = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "bias")
    config = GPTConfig(**{field: model_config[field] for field in fields})
    model = GPT(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model.to(device), checkpoint

