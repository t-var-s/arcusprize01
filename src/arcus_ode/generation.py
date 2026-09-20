"""Small deterministic generation probe for reproducing the paper-jam clue."""

from __future__ import annotations

import torch

from .model import GPT
from .tokenizer import decode, encode


@torch.inference_mode()
def generate_greedy(
    model: GPT,
    prompt: str,
    device: str,
    *,
    max_new_tokens: int = 100,
    stop_at: str | None = None,
) -> str:
    token_ids = encode(prompt)
    for _ in range(max_new_tokens):
        context = token_ids[-model.config.block_size :]
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(inputs)
        token_ids.append(int(torch.argmax(logits[0, -1]).item()))
        text = decode(token_ids)
        if stop_at is not None and stop_at in text:
            return text[: text.index(stop_at) + len(stop_at)]
    return decode(token_ids)

