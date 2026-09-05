"""
Electronic energy model for LLMCompass (Phase G3).

Provides energy-per-operation estimates for electronic compute,
SRAM/HBM/NoC data movement, and static power. Parameters are
calibrated from public data, CACTI, or empirical measurements.

Used by the hybrid cost model to produce energy breakdowns that
are comparable with SimPhony's photonic energy estimates.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ElectronicEnergyModel:
    """Energy parameters for electronic compute and memory hierarchy.

    All energy values are in joules per operation or per byte.
    Power values are in watts.

    Parameters
    ----------
    matmul_energy_per_mac_j : float
        Energy per multiply-accumulate (fused) in the systolic array.
    vector_energy_per_op_j : float
        Energy per vector ALU operation (add, mul, etc.).
    sram_energy_per_byte_j : float
        Energy to read one byte from L1 SRAM.
    hbm_energy_per_byte_j : float
        Energy to read one byte from HBM.
    noc_energy_per_byte_j : float
        Energy to move one byte across the network-on-chip / crossbar.
    static_power_w : float
        Total static/leakage power of the electronic chip.
    """

    # Dynamic energy (Joules)
    matmul_energy_per_mac_j: float = 0.4e-12     # 0.4 pJ/MAC (7nm systolic)
    vector_energy_per_op_j: float = 0.9e-12      # 0.9 pJ/op (7nm vector ALU)
    sram_energy_per_byte_j: float = 3.7e-12      # 3.7 pJ/byte (L1 SRAM read)
    hbm_energy_per_byte_j: float = 300e-12        # 300 pJ/byte (HBM2e)
    noc_energy_per_byte_j: float = 2.0e-12        # 2.0 pJ/byte (on-chip NoC)

    # Static power (Watts)
    static_power_w: float = 15.0                  # chip-level leakage

    # Process node info
    process_node: str = "7nm"
    voltage_v: float = 0.7

    # ------------------------------------------------------------------
    # Per-operation energy queries
    # ------------------------------------------------------------------

    def matmul_energy(self, mac_count: int) -> float:
        """Energy for `mac_count` fused multiply-accumulates."""
        return mac_count * self.matmul_energy_per_mac_j

    def vector_energy(self, op_count: int) -> float:
        """Energy for `op_count` vector ALU operations."""
        return op_count * self.vector_energy_per_op_j

    def sram_read_energy(self, bytes_count: int) -> float:
        return bytes_count * self.sram_energy_per_byte_j

    def hbm_read_energy(self, bytes_count: int) -> float:
        return bytes_count * self.hbm_energy_per_byte_j

    def noc_energy(self, bytes_count: int) -> float:
        return bytes_count * self.noc_energy_per_byte_j

    def static_energy(self, duration_s: float) -> float:
        """Static energy over `duration_s` seconds."""
        return self.static_power_w * duration_s

    # ------------------------------------------------------------------
    # Pre-configured models
    # ------------------------------------------------------------------

    @classmethod
    def low(cls) -> "ElectronicEnergyModel":
        """Optimistic (low-power) estimates."""
        return cls(
            matmul_energy_per_mac_j=0.2e-12,
            vector_energy_per_op_j=0.5e-12,
            sram_energy_per_byte_j=2.0e-12,
            hbm_energy_per_byte_j=200e-12,
            noc_energy_per_byte_j=1.0e-12,
            static_power_w=10.0,
        )

    @classmethod
    def nominal(cls) -> "ElectronicEnergyModel":
        """Nominal estimates."""
        return cls()

    @classmethod
    def high(cls) -> "ElectronicEnergyModel":
        """Conservative (high-power) estimates."""
        return cls(
            matmul_energy_per_mac_j=0.6e-12,
            vector_energy_per_op_j=1.5e-12,
            sram_energy_per_byte_j=5.0e-12,
            hbm_energy_per_byte_j=400e-12,
            noc_energy_per_byte_j=3.0e-12,
            static_power_w=25.0,
        )
