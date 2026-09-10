# Experiment code layout

The public entry point is the `hpat-repro` command in
`src/hpat_eval/repro_cli.py`. It provides isolated, validated workflows for
reviewers and keeps internal stage details out of the quick-start path.

| Directory | Contents |
| --- | --- |
| `config/` | Frozen experiment settings, hashes, schemas, and release policy |
| `src/hpat_eval/` | Orchestration, validation, manifests, and reviewer CLI |
| `scripts/` | Readable single-purpose mapping, energy, sensitivity, and audit stages |
| `tests/` | Unit fixtures plus end-to-end smoke, tamper, and release tests |

The CLI launches existing stage scripts as subprocesses so every command,
return code, and output remains visible in the run manifest. Stage scripts
write only to the requested output directory unless a maintainer explicitly
uses the separate promotion boundary.

New or modified public functions should include an English docstring. Comments
are used where they explain scientific boundaries, deterministic serialization,
or non-obvious backend behavior; straightforward Python is left uncluttered.
