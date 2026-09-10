# HPAT experimental reproducibility artifact

This repository contains only the code, locked configuration, tests, and
machine-readable reference results for the bounded HPAT experiments. It does
not contain a manuscript, submission PDF, bibliography, or publication art.

The results are local architecture/model diagnostics. They are **not**
fabricated-silicon measurements, measured HPAT timing or energy, edge/mobile
deployment results, or full-ImageNet validation.

## Reviewer quick start

The smoke path checks a fresh installation, all tests, the three-stage
experiment pipeline, output isolation, and manifest verification. It uses
small generated fixtures and does not download data or model weights.

```bash
uv python install 3.12.13
uv sync --locked --extra ml --extra dev
make reviewer-smoke
```

Expected result: the test suite passes and the final JSON report contains
`"valid": true`. A smoke result is an installation check, not a reproduction
of the reference numbers.

## Reproduce and compare the reference results

The numerical run of record requires an Apple Silicon Mac with PyTorch MPS.
The pipeline refuses silent CPU fallback.

```bash
make reviewer-prepare
make reviewer-run
make reviewer-reference
make reviewer-compare
```

The last command verifies the downloaded Release archive and compares the
complete reproduced run with the immutable full-run digest. Success is
reported as:

```json
{
  "match": true,
  "status": "Green"
}
```

Equivalent commands, troubleshooting, resource expectations, and output
interpretation are in [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md).

## Locked experiment

[`experiments/config/paper_v1.json`](experiments/config/paper_v1.json) fixes:

- MobileViT variants and the reasonable-high mapping boundary;
- component-visible modelled energy assumptions;
- PD/TIA noise, WDM crosstalk, MRR variation, and thermal drift;
- precision, preprocessing, output policy, and five seeds;
- a deterministic 1,024-sample Imagenette-160 slice selected with seed
  `20260706`.

Ten classes contribute 102 samples each, with one extra sample from the first
four lexically sorted synsets. The generated ledger records relative paths,
labels, and per-file SHA-256 hashes. Exact third-party data and pretrained
weights are downloaded from pinned sources and verified; they are not
redistributed.

The identifiers `paper` and `paper_v1` are compatibility names for this frozen
experiment protocol. They do not refer to or include a manuscript.

## Repository map

| Path | Reviewer-facing purpose |
| --- | --- |
| `experiments/config/` | Locked experiment, data, and release contracts |
| `experiments/src/hpat_eval/` | Readable library code and the `hpat-repro` CLI |
| `experiments/scripts/` | Individual experiment stages used by the CLI |
| `experiments/tests/` | Unit, smoke, determinism, provenance, and anonymity tests |
| `tables/` | Canonical machine-readable reference tables |
| `docs/REVIEWER_GUIDE.md` | End-to-end reviewer instructions |
| `docs/claim_boundary.md` | Evidence scope and prohibited claim upgrades |

Reviewers should normally use `hpat-repro` or the `make reviewer-*` commands,
not invoke files in `experiments/scripts/` directly. See
[`experiments/README.md`](experiments/README.md) for the internal layout.

## Release policy

The `v1.0.1` Release asset `hpat-artifact-v1.0.1.tar.gz` is a deterministic,
self-verifying package of sanitized experiment outputs. It excludes datasets,
weights, logits, device traces, host logs, PDFs, TeX, and publication figures.
Its manifest also retains the full canonical digest needed to compare a
complete reviewer run.

P0-E1, P0-E2, and P0-E4 remain blocked. The complete MPS collection is a local
release gate; GitHub Actions runs CPU smoke tests and package audits without
claiming cross-platform performance equivalence.

Maintainer-only promotion, freeze, and release-audit commands are documented
separately in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Licensing and anonymous citation

Software is Apache-2.0. Original documentation and tabular experiment data are
CC BY 4.0. Third-party payloads are not included; see [`NOTICE`](NOTICE) and
[`LICENSES/`](LICENSES/). During double-blind review, use the anonymous
[`CITATION.cff`](CITATION.cff) entry.
