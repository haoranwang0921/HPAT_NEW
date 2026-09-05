"""
Typed cost schemas for the direct GEMM API and joint-simulation interface.

All fields use explicit SI units as required by the implementation plan (Section 4).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PhotonicCostRequest:
    """A single GEMM operation to be costed on a photonic core.

    M: number of input vectors (batch dimension)
    K: inner dimension (input features / dot-product size)
    N: number of output neurons (output features)

    Internally SimPhony treats:  weight matrix = [N, K]
                                 input matrix  = [K, M]
                                 D = K  (the dimension mapped to wavelength)
    """
    M: int
    K: int
    N: int
    input_bits: int = 8
    weight_bits: int = 8
    output_bits: int = 8
    dataflow: str = "weight_stationary"
    architecture_id: str = ""
    architecture_hash: str = ""


@dataclass(frozen=True)
class PhotonicKernelCost:
    """Cost of executing one GEMM on the photonic accelerator (compute only)."""
    compute_latency_s: float          # optical pass-through time
    operand_encoding_latency_s: float # DAC + modulator encoding time
    conversion_latency_s: float       # TIA + ADC conversion time
    programming_latency_s: float      # MRR weight tuning time (one-shot per tile)
    dynamic_energy_j: float           # switching energy
    dac_energy_j: float               # DAC energy component
    adc_energy_j: float               # ADC energy component
    laser_energy_j: float             # laser wall-plug energy
    mrr_tuning_energy_j: float        # MRR program energy (one-shot per weight load)
    mrr_hold_energy_j: float          # MRR static hold energy over compute duration
    input_bytes: int = 0
    output_bytes: int = 0
    iter_M: int = 1
    iter_K: int = 1
    iter_N: int = 1
    switching_cycles: int = 1
    max_cycles: int = 1
    utilization: float = 1.0


@dataclass(frozen=True)
class WeightProgrammingCost:
    """One-time cost to program a set of MRR weights."""
    tile_count: int = 0
    programmed_mrr_count: int = 0
    programming_latency_s: float = 0.0
    programming_energy_j: float = 0.0
    hold_power_w: float = 0.0


@dataclass(frozen=True)
class ArchitectureCost:
    """One-time architecture-level costs (reported once, not per layer)."""
    pic_area_um2: float = 0.0
    rf_eic_area_um2: float = 0.0
    total_area_um2: float = 0.0
    core_insertion_loss_db: float = 0.0
    laser_wall_plug_power_w: float = 0.0
    mrr_count: int = 0
    pd_count: int = 0
    dac_count: int = 0
    adc_count: int = 0


# Schema version — bump when schemas change incompatibly
SCHEMA_VERSION = "1.0.0"
