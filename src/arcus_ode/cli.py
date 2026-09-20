"""Command-line interface for artifact inspection and reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifact import download_artifact, verify_artifact
from .constants import ACCEPTED_FLAG, ARTIFACT_SHA256, CARRIER_PROMPT, EXPECTED_CHUNKS
from .generation import generate_greedy
from .model import load_checkpoint
from .solver import solve
from .tokenizer import encode


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def inspect_checkpoint(path: Path) -> None:
    verify_artifact(path)
    _, checkpoint = load_checkpoint(path)
    config = checkpoint["model_config"]
    public_config = {
        key: config[key]
        for key in ("n_layer", "n_head", "n_embd", "vocab_size", "block_size", "dropout", "bias")
    }
    print(json.dumps({"sha256": ARTIFACT_SHA256, "model_config": public_config}, indent=2))


def reproduce(path: Path, device: str, output_dir: Path) -> None:
    verify_artifact(path)
    selected_device = select_device(device)
    model, _ = load_checkpoint(path, selected_device)
    chunks, sheet_path = solve(model, selected_device, output_dir)
    joined = "".join(chunks)
    for layer, chunk in enumerate(chunks):
        print(f"h{layer}.resid -> {chunk}")
    print(f"\njoined -> {joined}")
    print(f"image  -> {sheet_path}")
    if tuple(chunks) != EXPECTED_CHUNKS or joined != ACCEPTED_FLAG:
        raise RuntimeError("decoded output differs from the documented result")


def generate_jam(path: Path, device: str, max_new_tokens: int) -> None:
    verify_artifact(path)
    selected_device = select_device(device)
    model, _ = load_checkpoint(path, selected_device)
    print(
        generate_greedy(
            model,
            "<|alvaro_de_campos|>",
            selected_device,
            max_new_tokens=max_new_tokens,
            stop_at="[EPSON W-02]",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and verify the public ode.pt")
    download.add_argument("--output", type=Path, default=Path("ode.pt"))

    inspect = subparsers.add_parser("inspect", help="print verified checkpoint metadata")
    inspect.add_argument("checkpoint", type=Path)

    generate = subparsers.add_parser("generate", help="greedily reproduce the paper-jam clue")
    generate.add_argument("checkpoint", type=Path)
    generate.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    generate.add_argument("--max-new-tokens", type=int, default=100)

    reproduce_parser = subparsers.add_parser("solve", help="decode the residual-stream Aztec symbols")
    reproduce_parser.add_argument("checkpoint", type=Path)
    reproduce_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    reproduce_parser.add_argument("--output-dir", type=Path, default=Path("output"))

    subparsers.add_parser("prompt", help="print the 77-token teacher-forced carrier prompt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        path = download_artifact(args.output)
        print(f"verified {path} ({ARTIFACT_SHA256})")
    elif args.command == "inspect":
        inspect_checkpoint(args.checkpoint)
    elif args.command == "generate":
        generate_jam(args.checkpoint, args.device, args.max_new_tokens)
    elif args.command == "solve":
        reproduce(args.checkpoint, args.device, args.output_dir)
    elif args.command == "prompt":
        print(CARRIER_PROMPT)
        print(f"\ntoken count: {len(encode(CARRIER_PROMPT))}")


if __name__ == "__main__":
    main()
