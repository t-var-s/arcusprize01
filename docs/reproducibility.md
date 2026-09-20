# Reproducibility

## Research question

Can the accepted Arcus Ode Triunfal flag be recovered deterministically from the released `ode.pt` checkpoint by inspecting post-block residual activations?

## Inputs

| Input | Pinned value |
|---|---|
| Checkpoint | GitHub release `ode-triunfal-v1/ode.pt` |
| Size | 199,981,173 bytes |
| SHA-256 | `b54373efba6b89e38bdd56f031ca63b7bf49f9024dea254c21227acc3dacb6ab` |
| Carrier prompt length | 77 token IDs |
| Residual hooks | post-block `h0` through `h7` |
| Crop | token rows `0:77`, hidden dimensions `281:358` |
| Readout | value `> 0` becomes white |
| Decoder | ZXing-C++ restricted to Aztec |

The artifact URL and digest are encoded in [`constants.py`](../src/arcus_ode/constants.py). `arcus-ode download` writes to a temporary `.part` file, checks SHA-256, and only then moves it into place.

## Environment

Python 3.11 or newer is required. Runtime packages are declared in `pyproject.toml`:

- PyTorch for checkpoint loading and inference;
- NumPy for binary image construction;
- Pillow for image output;
- zxing-cpp for Aztec detection.

`torch.load(..., weights_only=True)` prevents arbitrary checkpoint globals from executing. Exact artifact verification happens before loading.

### uv

```bash
uv sync --extra dev
uv run arcus-ode download --output ode.pt
```

### Standard virtual environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
arcus-ode download --output ode.pt
```

## Procedure

### 1. Inspect the authenticated checkpoint

```bash
uv run arcus-ode inspect ode.pt
```

Expected model configuration:

```json
{
  "n_layer": 10,
  "n_head": 8,
  "n_embd": 640,
  "vocab_size": 262,
  "block_size": 1024,
  "dropout": 0.1,
  "bias": false
}
```

### 2. Reproduce the generated clue

```bash
uv run arcus-ode generate ode.pt --device cpu
```

Generation is greedy and stops when it reaches `[EPSON W-02]`. Its output is the trigger followed by the jam shown in the main README.

### 3. Check the teacher-forced input

```bash
uv run arcus-ode prompt
```

This prints the exact trigger plus jam and must end with `token count: 77`.

### 4. Capture and decode

```bash
uv run arcus-ode solve ode.pt --device cpu --output-dir output
```

Expected standard output:

```text
h0.resid -> flag
h1.resid -> {wit
h2.resid -> hin_
h3.resid -> laye
h4.resid -> rs_i
h5.resid -> nter
h6.resid -> link
h7.resid -> ed_}

joined -> flag{within_layers_interlinked_}
image  -> output/activation-crops.png
```

The command fails if any decoded chunk differs from the documented sequence.

### 5. Run invariant tests

```bash
uv run pytest
```

The fast tests do not need the checkpoint. They cover tokenizer round-tripping, greedy underscore aliases, the 77-token/77-dimension square, and chunk concatenation.

## Device considerations

Inference is deterministic for this extraction because no sampling occurs. CPU is the reference route. MPS and CUDA only change tensor execution; crops are moved to CPU before thresholding and decoding. If a platform produces a barcode-decoding discrepancy around values extremely close to zero, rerun with `--device cpu` and record the PyTorch/platform versions.

## Independent checks

Several observations reduce the chance of a coincidental visualization:

1. eight separate crops are recognized by a standard Aztec library;
2. each yields exactly four characters;
3. block order produces a grammatical, closed flag;
4. the result was accepted during the challenge and later published by the organisers;
5. the organisers independently published a sign-based reference solver;
6. during the challenge, the same fixed extraction tracked a payload change across public checkpoint versions.

## Limitations

- The 191 MiB checkpoint is hosted externally; a future removal would require an archival copy whose digest matches the pinned value.
- This repository reproduces inference and extraction, not the organisers' original base-model training or carrier fine-tuning.
- The historical intermediate refresh is documented from contemporaneous observations but is not required, distributed, or reproduced here.
- The live verifier is no longer part of the experiment. Agreement with the organisers' public answer now supplies the external validation.

## Citation and source boundaries

Primary public sources:

- [Augusta Labs' official challenge write-up](https://arcus.augustalabs.ai/)
- [Public checkpoint release](https://github.com/augustalabs/arcus-artifacts/releases/tag/ode-triunfal-v1)
- [Ode Triunfal at Arquivo Pessoa](http://arquivopessoa.net/textos/1458)

The architecture, solver behaviour, tensor shapes, and final decode are directly reproducible. Statements about the 84-book corpus, fine-tuning loss, held-out drift, challenge intent, and submission statistics are attributed to the official write-up rather than presented as our measurements.
