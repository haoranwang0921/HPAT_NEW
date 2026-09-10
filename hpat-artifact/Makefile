UV ?= uv
CAFFEINATE ?= caffeinate -dimsu
DEVICE ?= mps
SMOKE_RUN_DIR ?= runs/smoke
PREPARED_DIR ?= runs/prepared/paper_v1
RUN_DIR ?= runs/paper_v1
REFERENCE_DIR ?= artifacts/reference/v1.0.1
REFERENCE_ASSET ?= artifacts/downloads/hpat-artifact-v1.0.1.tar.gz
REPOSITORY_URL ?= $(patsubst %.git,%,$(shell git config --get remote.origin.url))
REFERENCE_URL ?= $(REPOSITORY_URL)/releases/download/v1.0.1/hpat-artifact-v1.0.1.tar.gz
RELEASE_ROOT ?= .
RELEASE_AUDIT ?= artifacts/release_audit.json
RELEASE_ALLOWLIST ?= experiments/config/release_spec_v1.json

.PHONY: setup reviewer-smoke reviewer-prepare reviewer-run reviewer-verify \
	reviewer-reference reviewer-compare maintainer-promote maintainer-freeze \
	maintainer-release-audit smoke reproduce verify freeze release-audit

setup:
	$(UV) sync --locked --extra ml --extra dev

reviewer-smoke:
	$(UV) run pytest -p no:cacheprovider -q
	$(UV) run hpat-repro run --profile smoke --device cpu --output-dir runs/reviewer-smoke
	$(UV) run hpat-repro verify --run-dir runs/reviewer-smoke

reviewer-prepare:
	$(UV) run hpat-repro prepare-data --profile paper --output-dir $(PREPARED_DIR)

reviewer-run:
	HPAT_CAFFEINATE_USED=1 $(CAFFEINATE) $(UV) run hpat-repro run --profile paper --device $(DEVICE) --prepared-dir $(PREPARED_DIR) --output-dir $(RUN_DIR)

reviewer-verify:
	$(UV) run hpat-repro verify --run-dir $(RUN_DIR)

reviewer-reference:
	mkdir -p artifacts/downloads
	curl --fail --location --output $(REFERENCE_ASSET) $(REFERENCE_URL)
	$(UV) run hpat-repro verify-reference --reference $(REFERENCE_ASSET)

reviewer-compare:
	$(UV) run hpat-repro compare --run-dir $(RUN_DIR) --reference $(REFERENCE_ASSET)

maintainer-promote: reviewer-verify
	$(UV) run hpat-repro promote --run-dir $(RUN_DIR)

maintainer-freeze: reviewer-verify
	$(UV) run hpat-repro freeze --run-dir $(RUN_DIR) --output-dir $(REFERENCE_DIR)

maintainer-release-audit:
	@if [ "$(RELEASE_ROOT)" = "." ]; then \
		audit_tree="$$(mktemp -d)"; \
		trap 'rm -rf "$$audit_tree"' EXIT; \
		git archive --format=tar HEAD | tar -xf - -C "$$audit_tree"; \
	else \
		audit_tree="$(RELEASE_ROOT)"; \
	fi; \
	$(UV) run python experiments/scripts/audit_public_release.py --root "$$audit_tree" --output $(RELEASE_AUDIT) --max-file-mib 20 --allowlist $(RELEASE_ALLOWLIST)

# Compatibility aliases retained for existing automation.
smoke:
	$(UV) run hpat-repro run --profile smoke --device cpu --output-dir $(SMOKE_RUN_DIR)
	$(UV) run hpat-repro verify --run-dir $(SMOKE_RUN_DIR)

reproduce:
	$(UV) run hpat-repro prepare-data --profile paper --output-dir $(PREPARED_DIR)
	HPAT_CAFFEINATE_USED=1 $(CAFFEINATE) $(UV) run hpat-repro run --profile paper --device $(DEVICE) --prepared-dir $(PREPARED_DIR) --output-dir $(RUN_DIR)

verify:
	$(UV) run hpat-repro verify --run-dir $(RUN_DIR)

freeze: maintainer-freeze

release-audit: maintainer-release-audit
