# Contributing

This repository accepts changes only to the anonymous experimental
reproducibility artifact.

## Allowed scope

- experiment source code and explicitly versioned configuration;
- deterministic data-selection and input-hash ledgers;
- smoke, unit, provenance, and release-audit tests;
- bounded machine-readable reference tables;
- documentation needed to install, run, verify, and freeze the experiments.

Manuscripts, submission PDFs, TeX, bibliographies, publication artwork,
reviewer correspondence, internal roadmaps, datasets, model weights, device
logs, and host-specific metadata are outside the repository scope.

## Before proposing a change

1. Keep all normal outputs inside an explicit output directory.
2. Require an explicit `promote` or `freeze` command before updating
   reference assets.
3. Preserve the declared evidence tier and claim boundary.
4. Run:

   ```bash
   uv sync --locked --extra ml --extra dev
   uv run pytest -q
   make release-audit
   ```

5. Confirm that a clean smoke run leaves the Git worktree unchanged.

Do not add personal names, email addresses, affiliations, user-home paths,
device identifiers, dataset images, weights, or third-party payloads. During
double-blind review, commits must use the anonymous artifact identity.
