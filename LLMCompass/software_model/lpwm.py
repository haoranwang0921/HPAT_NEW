"""
LPWM workload constructed from an operator trace (Phase F1).

Reads a validated trace via TraceReader and builds a sequence of
operators that can be simulated in the hybrid photonic-electronic
framework. Fixed Linear ops become PhotonicMatmul instances (routed
to SimPhony), while dynamic/nonlinear ops remain on electronic.
"""

from typing import List, Optional

from software_model.workload_trace import TraceReader
from software_model.photonic_matmul import PhotonicMatmul


class LPWMWorkload:
    """An LPWM inference workload reconstructed from an operator trace.

    Parameters
    ----------
    trace_path : str
        Path to the JSON-lines trace file.
    """

    def __init__(self, trace_path: str):
        self.reader = TraceReader(trace_path)
        self.reader.load()
        self._operators: List = []
        self._build_operators()

    # ------------------------------------------------------------------
    # Build operator graph from trace
    # ------------------------------------------------------------------

    def _build_operators(self):
        """Instantiate operator objects from trace records."""
        for rec in self.reader.records:
            op = self._make_operator(rec)
            self._operators.append((rec, op))

    def _make_operator(self, rec: dict):
        op_type = rec.get("op_type", "")

        if op_type == "Linear" and rec.get("weight_static"):
            return PhotonicMatmul(
                M=rec.get("M", 0) or 0,
                K=rec.get("K", 0) or 0,
                N=rec.get("N", 0) or 0,
                op_id=rec.get("op_id"),
                op_role=rec.get("op_role", "unknown"),
                weight_static=True,
                weight_id=rec.get("weight_id"),
                photonic_candidate=True,
                input_bits=rec.get("input_bits", 8),
                weight_bits=rec.get("weight_bits", 8) or 8,
                output_bits=rec.get("output_bits", 8),
            )
        else:
            # Electronic operator — stored as a plain dict for later
            # routing to LLMCompass electronic models
            return {
                "op_id": rec.get("op_id"),
                "op_type": op_type,
                "op_role": rec.get("op_role", "unknown"),
                "M": rec.get("M"),
                "K": rec.get("K"),
                "N": rec.get("N"),
                "weight_static": rec.get("weight_static", False),
                "electronic": True,
            }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def operator_count(self) -> int:
        return len(self._operators)

    @property
    def photonic_ops(self) -> List:
        return [
            (rec, op) for rec, op in self._operators
            if isinstance(op, PhotonicMatmul) and op.photonic_candidate
        ]

    @property
    def electronic_ops(self) -> List:
        return [
            (rec, op) for rec, op in self._operators
            if not (isinstance(op, PhotonicMatmul) and op.photonic_candidate)
        ]

    def unique_weight_ids(self) -> set:
        return set(
            op.weight_id
            for _, op in self.photonic_ops
            if op.weight_id is not None
        )

    def summary(self) -> dict:
        trace_summary = self.reader.summary()
        trace_summary["photonic_ops"] = len(self.photonic_ops)
        trace_summary["electronic_ops"] = len(self.electronic_ops)
        return trace_summary
