# Reviewer guide

This guide separates a quick installation check from the full numerical
reproduction. Commands are intended to be copied from the repository root.

## A. Five-minute installation and smoke check

Prerequisites: Git, `make`, and [uv](https://docs.astral.sh/uv/) 0.9.18.

```bash
uv python install 3.12.13
uv sync --locked --extra ml --extra dev
make reviewer-smoke
```

The command runs the complete test suite, then executes mapping, modelled
energy, and non-ideality stages with deterministic CPU fixtures. It should end
with a verification report containing `"valid": true`. Use a new output
directory or remove the disposable `runs/reviewer-smoke/` directory before
repeating it; run directories are deliberately write-once.

This step works on macOS or Linux. It confirms software usability but does not
reproduce the canonical numerical evidence.

## B. Full canonical reproduction

Additional requirements:

- an Apple Silicon Mac;
- PyTorch MPS built and available;
- network access while preparing the pinned Imagenette archive and timm
  weights;
- several GB of free space for the environment, downloads, and outputs.

Confirm MPS before downloading inputs:

```bash
uv run python -c "import torch; assert torch.backends.mps.is_built(); assert torch.backends.mps.is_available(); print('MPS ready:', torch.__version__)"
```

Do not set `PYTORCH_ENABLE_MPS_FALLBACK=1`. The canonical profile records the
device and fails instead of silently mixing CPU and MPS execution.

### 1. Prepare exact inputs

```bash
make reviewer-prepare
```

This downloads pinned third-party resources, checks their SHA-256 values,
selects the fixed 1,024-image subset, and writes a per-image hash ledger under
`runs/prepared/paper_v1/`. Dataset images and weights stay untracked.

### 2. Run and verify

```bash
make reviewer-run
make reviewer-verify
```

The long stage runs under `caffeinate -dimsu`, keeps warmup separate, and
synchronizes MPS before timing. The manifest records versions, seeds, dtype,
backend, fallback status, input hashes, commands, and every output hash.

Observed on the release machine, prepared inputs occupied about 259 MB and the
complete run about 136 MB. The canonical run took about 14 minutes based on
recorded output timestamps; preparation/download time was not recorded and
network or hardware differences can dominate. These figures are planning
estimates, not performance claims.

### 3. Download, verify, and compare the reference

```bash
make reviewer-reference
make reviewer-compare
```

`reviewer-reference` downloads `hpat-artifact-v1.0.1.tar.gz` from the GitHub
Release and checks every packaged file against its sanitized manifest.
`reviewer-compare` then compares the full reproduced canonical CSV/NPZ
inventory and digest against the frozen full-run reference.

The machine-readable result is written to
`runs/paper_v1/reference_comparison_report.json`. A successful reproduction
has `"status": "Green"` and `"match": true`. A changed, missing, or unexpected
canonical file produces a non-zero exit status and a `file_differences` list.

## C. Direct CLI equivalents

```bash
uv run hpat-repro prepare-data --profile paper --output-dir runs/prepared/paper_v1
HPAT_CAFFEINATE_USED=1 caffeinate -dimsu uv run hpat-repro run \
  --profile paper --device mps \
  --prepared-dir runs/prepared/paper_v1 \
  --output-dir runs/paper_v1
uv run hpat-repro verify --run-dir runs/paper_v1
uv run hpat-repro verify-reference \
  --reference artifacts/downloads/hpat-artifact-v1.0.1.tar.gz
uv run hpat-repro compare \
  --run-dir runs/paper_v1 \
  --reference artifacts/downloads/hpat-artifact-v1.0.1.tar.gz
```

## D. Interpreting limitations

An exact match reproduces this local, modelled Apple Silicon experiment. It
does not establish fabricated-silicon behavior, calibrated silicon energy,
author-measured edge/mobile deployment, full ImageNet robustness, or
cross-platform superiority. CPU and CUDA runs may be retained in separate
directories for comparison, but cannot replace or overwrite the MPS reference.

If the smoke path fails, include the failing command and JSON error in an
Issue. If a full run verifies locally but does not match, attach only
`verification_report.json` and `reference_comparison_report.json`; do not
upload datasets, weights, logits, or machine-identifying device logs.
