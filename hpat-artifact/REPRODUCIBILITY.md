# Reproducibility and release protocol

## Scope

The canonical workflow reproduces HPAT mapping, modelled energy, and
fixed-slice non-ideality diagnostics. The run of record is local PyTorch MPS
on Apple Silicon. CPU and CUDA runs are comparisons only.

The repository contains no manuscript or publication artwork. The compatibility
identifiers `paper` and `paper_v1` refer only to the frozen experiment profile.

## Reviewer workflow

The shortest instructions are in [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md).
The underlying protocol is summarized here.

### 1. Locked environment

```bash
uv python install 3.12.13
uv sync --locked --extra ml --extra dev
```

Python is locked to 3.12.13, uv to 0.9.18, and core dependencies in `uv.lock`.
The smoke profile can run on CPU; the canonical profile requires MPS.

### 2. CPU smoke

```bash
make reviewer-smoke
```

Generated metadata-free fixtures exercise the complete pipeline without data
or weight downloads. A smoke run validates installation, isolation,
serialization, and verification, but not the canonical numbers.

### 3. Fixed inputs

```bash
make reviewer-prepare
```

The command verifies pinned archive and weight hashes, then selects 1,024
Imagenette validation samples without replacement using seed `20260706`.
Each of ten sorted synsets receives 102 samples; the first four receive one
additional sample. The ledger records relative paths, labels, and file hashes.

### 4. Canonical run

```bash
make reviewer-run
make reviewer-verify
```

The long stage uses `caffeinate -dimsu`. Warmup is separate from measured
iterations and MPS is synchronized before timing. The five seeds are
`20260706` through `20260710`. The manifest records the commit, exact inputs,
backend, OS, packages, dtype, synchronization, commands, outputs, evidence
tier, and fallback status.

Do not set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Unsupported operations must fail
visibly rather than create mixed MPS/CPU evidence.

### 5. Reference package and exact comparison

```bash
make reviewer-reference
make reviewer-compare
```

The first command downloads and self-verifies the sanitized v1.0.1 Release
asset. The second requires a release-ready full paper run and compares its
canonical output digest and file inventory with the frozen full-run digest.
Success requires `"status": "Green"` and `"match": true`.

The cross-run digest includes canonical CSV and NPZ results. The declared
per-sample top-k margin detail is integrity-checked within each full run but
excluded from the comparison digest because sub-decision MPS values may vary
slightly. Categorical changes and canonical tables remain covered.

## Output contract

Every run directory is write-once. It contains:

- a locked configuration snapshot;
- an input subset ledger for the canonical profile;
- stage manifests and machine-readable tables;
- raw diagnostics required by the declared output policy;
- `repro_manifest.json` with hashes for every output;
- `verification_report.json` after verification;
- `reference_comparison_report.json` after comparison.

Run and verify never update tracked reference tables. Changing the backend,
OS, package set, or input hashes requires a separate output directory and
cannot overwrite the reference.

## Maintainer-only release operations

The following commands are not part of reviewer reproduction.

### Promote tracked tables

```bash
uv run hpat-repro promote --run-dir runs/paper_v1
```

Promotion updates only the allowlisted canonical tables and sample ledger.
It must follow a release-ready full run.

### Freeze v1.0.1 assets

```bash
uv run hpat-repro freeze \
  --run-dir runs/paper_v1 \
  --output-dir artifacts/reference/v1.0.1
```

The freeze output contains table/schema/claim/anonymity audits, commit binding,
checksums, and `release_assets/hpat-artifact-v1.0.1.tar.gz`. The archive has a
self-consistent sanitized manifest and a separate immutable full-run digest.
Dataset images, weights, logits, device traces, host logs, PDF, TeX, and
publication artwork remain excluded.

### Audit the public tree

```bash
make maintainer-release-audit
```

The audit enforces the 20 MiB Git limit, allowlist, required files, relative
paths, identity scan, and explicit rejection of manuscript/PDF/TeX material.
Before tagging, also inspect a fresh `git archive` and verify the generated
Release package with:

```bash
uv run hpat-repro verify-reference \
  --reference artifacts/reference/v1.0.1/release_assets/hpat-artifact-v1.0.1.tar.gz
```

## Evidence boundary

An exact reproduction remains local/modelled diagnostic evidence. It does not
establish silicon timing or energy, calibrated layout/device energy,
author-measured mobile deployment, production speedup, full-ImageNet
robustness, or cross-platform superiority. P0-E1, P0-E2, and P0-E4 remain
blocked.
