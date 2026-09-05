"""
LPWM operator trace reader and validator (Phase F1).

Reads a JSON-lines trace file exported by joint_sim/lpwm_trace_exporter.py
and produces validated OperatorRecord objects for LLMCompass simulation.
"""

import json
from typing import List, Optional


class TraceReader:
    """Read and validate LPWM operator traces.

    Parameters
    ----------
    trace_path : str
        Path to a .jsonl file where each line is a serialized OperatorRecord.
    """

    def __init__(self, trace_path: str):
        self.trace_path = trace_path
        self._records: List[dict] = []
        self._by_phase: dict = {}
        self._by_role: dict = {}
        self._unique_weight_ids: set = set()

    # ------------------------------------------------------------------
    # Load / validate
    # ------------------------------------------------------------------

    def load(self) -> List[dict]:
        """Parse the trace file and return validated records.

        The first line may be a manifest header (JSON with metadata like
        trace_version, total_operators, etc.).  Subsequent lines are
        OperatorRecord objects.
        """
        self._records = []
        self._manifest = {}
        with open(self.trace_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at {self.trace_path}:{line_no}: {e}"
                    )
                # First line may be a manifest header (has total_operators, no op_id)
                if line_no == 1 and "op_id" not in rec and "total_operators" in rec:
                    self._manifest = rec
                    continue
                self._validate_record(rec, line_no)
                self._records.append(rec)
        self._build_indices()
        return self._records

    @property
    def manifest(self) -> dict:
        return self._manifest

    def _validate_record(self, rec: dict, line_no: int):
        required = [
            "op_id", "order", "module_path", "op_type", "op_role",
            "phase", "block_kind", "call_index", "input_shapes",
            "output_shape", "M", "K", "N", "dtype", "input_bits",
            "weight_bits", "output_bits", "weight_static",
        ]
        for key in required:
            if key not in rec:
                raise ValueError(
                    f"Missing required key '{key}' at line {line_no}"
                )

    def _build_indices(self):
        self._by_phase = {}
        self._by_role = {}
        self._unique_weight_ids = set()
        for rec in self._records:
            phase = rec.get("phase", "unknown")
            self._by_phase.setdefault(phase, []).append(rec)
            role = rec.get("op_role", "unknown")
            self._by_role.setdefault(role, []).append(rec)
            wid = rec.get("weight_id")
            if wid:
                self._unique_weight_ids.add(wid)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[dict]:
        return self._records

    @property
    def operator_count(self) -> int:
        return len(self._records)

    @property
    def weight_count(self) -> int:
        return len(self._unique_weight_ids)

    def by_phase(self, phase: str) -> List[dict]:
        return self._by_phase.get(phase, [])

    def by_role(self, role: str) -> List[dict]:
        return self._by_role.get(role, [])

    def static_weight_linears(self) -> List[dict]:
        """Return only fixed-weight Linear ops eligible for photonic."""
        return [
            r for r in self._records
            if r.get("op_type") == "Linear" and r.get("weight_static")
        ]

    def dynamic_ops(self) -> List[dict]:
        """Return ops that must run on electronic hardware."""
        return [
            r for r in self._records
            if not (r.get("op_type") == "Linear" and r.get("weight_static"))
        ]

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def unique_mkn_signatures(self) -> set:
        """Return the set of unique (M, K, N, input_bits, weight_bits) tuples."""
        sigs = set()
        for r in self.static_weight_linears():
            sigs.add((
                r.get("M", 0),
                r.get("K", 0),
                r.get("N", 0),
                r.get("input_bits", 8),
                r.get("weight_bits", 8),
            ))
        return sigs

    def summary(self) -> dict:
        return {
            "trace_path": self.trace_path,
            "total_operators": self.operator_count,
            "static_weight_linears": len(self.static_weight_linears()),
            "dynamic_ops": len(self.dynamic_ops()),
            "unique_weights": self.weight_count,
            "unique_mkn": len(self.unique_mkn_signatures()),
            "phases": {
                p: len(ops) for p, ops in self._by_phase.items()
            },
        }
