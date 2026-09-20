"""Capture, render, and decode the activation-level Aztec carrier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zxingcpp
from PIL import Image, ImageDraw

from .constants import CARRIER_PROMPT, CODE_LAYERS, HIDDEN_DIMS, TOKEN_ROWS
from .model import GPT
from .tokenizer import encode


@torch.inference_mode()
def capture_carrier(model: GPT, device: str) -> list[torch.Tensor]:
    token_ids = encode(CARRIER_PROMPT)
    if len(token_ids) != 77:
        raise AssertionError(f"carrier prompt encoded to {len(token_ids)} tokens, expected 77")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    _, residuals = model(inputs, capture_residuals=True)
    return [residuals[layer][0, TOKEN_ROWS, HIDDEN_DIMS].float().cpu() for layer in CODE_LAYERS]


def render_crop(crop: torch.Tensor, *, scale: int = 4, quiet_zone: int = 32) -> Image.Image:
    """Render the raw-sign readout used in the organisers' reference solver."""
    bits = crop.numpy() > 0
    pixels = (np.kron(bits, np.ones((scale, scale), dtype=np.uint8)) * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    return Image.fromarray(np.pad(np.asarray(image), quiet_zone, constant_values=255), mode="L")


def decode_crop(image: Image.Image) -> str:
    hits = zxingcpp.read_barcodes(np.asarray(image), formats=zxingcpp.BarcodeFormat.Aztec)
    if not hits:
        raise RuntimeError("Aztec decoder found no symbol")
    return hits[0].text


def save_contact_sheet(images: list[Image.Image], labels: list[str], path: str | Path) -> None:
    margin, label_height, gap, columns = 20, 38, 16, 4
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "L",
        (
            margin * 2 + columns * tile_width + (columns - 1) * gap,
            margin * 2 + rows * (label_height + tile_height) + (rows - 1) * gap,
        ),
        255,
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        row, column = divmod(index, columns)
        left = margin + column * (tile_width + gap)
        top = margin + row * (label_height + tile_height + gap)
        draw.text((left, top), label, fill=0)
        canvas.paste(image, (left, top + label_height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def solve(model: GPT, device: str, output_dir: str | Path) -> tuple[list[str], Path]:
    crops = capture_carrier(model, device)
    images = [render_crop(crop) for crop in crops]
    chunks = [decode_crop(image) for image in images]
    sheet_path = Path(output_dir) / "activation-crops.png"
    save_contact_sheet(images, [f"h{i}.resid -> {chunk}" for i, chunk in enumerate(chunks)], sheet_path)
    return chunks, sheet_path
