"""Download and verify the public challenge artifact."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from .constants import ARTIFACT_SHA256, ARTIFACT_URL


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: str | Path) -> str:
    actual = sha256_file(path)
    if actual != ARTIFACT_SHA256:
        raise ValueError(f"unexpected SHA-256 for {path}:\nexpected {ARTIFACT_SHA256}\nactual   {actual}")
    return actual


def download_artifact(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            verify_artifact(destination)
            return destination
        except ValueError:
            pass
    partial = destination.with_suffix(destination.suffix + ".part")

    def progress(blocks: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, blocks * block_size * 100 // total_size)
            print(f"\rDownloading ode.pt: {percent:3d}%", end="", file=sys.stderr, flush=True)

    try:
        urllib.request.urlretrieve(ARTIFACT_URL, partial, reporthook=progress)
        print(file=sys.stderr)
        verify_artifact(partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination
