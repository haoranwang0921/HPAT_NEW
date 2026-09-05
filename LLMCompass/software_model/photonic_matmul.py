"""
Photonic candidate wrapper for fixed-weight Linear layers (Phase F1).

A standalone operator that stores GEMM dimensions and, when the weight
is static, queries the SimPhony backend for photonic execution cost.
Does not depend on the electronic matmul.py (which requires scalesim).
"""

import sys
import os
from typing import Optional

# Add joint_sim to path for backend import
_JOINT_SIM = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "joint_sim")
)
if _JOINT_SIM not in sys.path:
    sys.path.insert(0, _JOINT_SIM)


class PhotonicMatmul:
    """A GEMM operator eligible for photonic MRR weight bank execution.

    Independent of the electronic Matmul class so it works without
    scalesim. Stores M/K/N dimensions and queries the SimPhony backend
    for photonic cost when photonic_candidate is True.

    Parameters
    ----------
    M : int
        Batch dimension (number of input vectors).
    K : int
        Inner dimension (input features).
    N : int
        Output features.
    op_id : str, optional
        Operator identifier from the trace.
    op_role : str
        Semantic role (Q, K, V, O, FFN_up, FFN_down, c_proj, ...).
    weight_static : bool
        True for fixed pre-trained weights.
    weight_id : str, optional
        Stable hash of (module_path, weight_shape).
    photonic_candidate : bool
        If True and weight_static, use photonic backend.
    input_bits : int
    weight_bits : int
    output_bits : int
    """

    def __init__(
        self,
        M: int,
        K: int,
        N: int,
        op_id: Optional[str] = None,
        op_role: str = "unknown",
        weight_static: bool = False,
        weight_id: Optional[str] = None,
        photonic_candidate: bool = False,
        input_bits: int = 8,
        weight_bits: int = 8,
        output_bits: int = 8,
    ):
        self.M = M
        self.K = K
        self.N = N
        self.op_id = op_id
        self.op_role = op_role
        self.weight_static = weight_static
        self.weight_id = weight_id
        self.photonic_candidate = photonic_candidate and weight_static
        self.input_bits = input_bits
        self.weight_bits = weight_bits
        self.output_bits = output_bits
        self.photonic_cost = None

    # ------------------------------------------------------------------
    # Cost estimation (Plan F3)
    # ------------------------------------------------------------------

    def estimate_cost(self, backend=None) -> dict:
        """Return structured ExecutionCost for this operator.

        If photonic_candidate and a backend is available, queries SimPhony.
        """
        if self.photonic_candidate and backend is not None:
            return self._photonic_cost(backend)
        return self._electronic_cost()

    def _photonic_cost(self, backend) -> dict:
        kernel = backend.kernel_cost(
            M=self.M, K=self.K, N=self.N,
            input_bits=self.input_bits,
            weight_bits=self.weight_bits,
            output_bits=self.output_bits,
        )
        prog = backend.programming_cost()
        self.photonic_cost = {
            "latency_s": (
                kernel["compute_latency_s"]
                + kernel["operand_encoding_latency_s"]
                + kernel["conversion_latency_s"]
                + kernel["programming_latency_s"]
            ),
            "dynamic_energy_j": kernel["dynamic_energy_j"],
            "memory_energy_j": 0.0,
            "programming_energy_j": prog["programming_energy_j"],
            "resource": "photonic",
            "breakdown": {
                "compute_latency_s": kernel["compute_latency_s"],
                "operand_encoding_latency_s": kernel["operand_encoding_latency_s"],
                "conversion_latency_s": kernel["conversion_latency_s"],
                "programming_latency_s": kernel["programming_latency_s"],
                "dac_energy_j": kernel["dac_energy_j"],
                "adc_energy_j": kernel["adc_energy_j"],
                "laser_energy_j": kernel["laser_energy_j"],
                "mrr_tuning_energy_j": kernel["mrr_tuning_energy_j"],
                "mrr_hold_energy_j": kernel["mrr_hold_energy_j"],
                "utilization": kernel["utilization"],
            },
        }
        return self.photonic_cost

    def _electronic_cost(self) -> dict:
        flop_count = 2 * self.M * self.K * self.N
        return {
            "latency_s": 0.0,
            "dynamic_energy_j": 0.0,
            "memory_energy_j": 0.0,
            "programming_energy_j": 0.0,
            "resource": "electronic",
            "breakdown": {"flop_count": flop_count},
        }
