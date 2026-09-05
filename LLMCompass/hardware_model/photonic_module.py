"""
Photonic accelerator module for LLMCompass hardware model (Phase G1).

Models an HPAT photonic chiplet with MRR weight banks, DAC/ADC,
and laser source. Plugs into Device as an optional photonic_module.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhotonicModule:
    """A photonic accelerator chiplet attached to an electronic Device.

    Parameters
    ----------
    architecture_id : str
        Identifier matching the SimPhony architecture config.
    core_count : int
        Number of photonic cores (tiles * cores_per_tile).
    bank_count : int
        Number of independent MRR weight banks.
    bank_capacity_rings : int
        Total MRR rings available for weight storage across all banks.
    mrr_count : int
        Total number of MRR devices in the photonic chip.
    pd_count : int
        Total number of photodetectors.
    dac_count : int
        Number of input DAC channels.
    adc_count : int
        Number of output ADC channels.
    core_height : int
        Number of dot-product rows per core (H).
    core_width : int
        Number of dot-product columns per core (W).
    num_wavelength : int
        Number of WDM wavelengths (N).
    work_freq_ghz : float
        Clock frequency in GHz.
    programming_parallelism : int
        How many MRRs can be programmed simultaneously.
    pic_area_um2 : float
        PIC die area (from SimPhony).
    rf_eic_area_um2 : float
        RF electronic IC area.
    core_insertion_loss_db : float
        Optical insertion loss per core.
    laser_wall_plug_power_w : float
        Laser wall-plug power in watts.
    hold_power_per_ring_w : float
        Static hold power per MRR ring.
    program_energy_per_ring_j : float
        Energy to program one MRR ring.
    program_time_per_ring_s : float
        Time to program one MRR ring.
    """

    architecture_id: str = "HPAT_v1"
    core_count: int = 4          # tiles * cores_per_tile = 2 * 2
    bank_count: int = 4          # one bank per core
    bank_capacity_rings: int = 16384
    mrr_count: int = 1024   # H*R*C*N = 64 nodes × 16 MRRs
    pd_count: int = 64       # H*R*C = 64 nodes
    dac_count: int = 32
    adc_count: int = 16
    core_height: int = 16
    core_width: int = 16
    num_wavelength: int = 16
    work_freq_ghz: float = 5.0
    programming_parallelism: int = 16   # one DAC per column can program in parallel
    pic_area_um2: float = 0.0
    rf_eic_area_um2: float = 0.0
    core_insertion_loss_db: float = 0.0
    laser_wall_plug_power_w: float = 0.0
    hold_power_per_ring_w: float = 0.1e-3   # 0.1 mW
    program_energy_per_ring_j: float = 0.0
    program_time_per_ring_s: float = 1000e-9  # 1000 ns

    @classmethod
    def from_simphony_backend(cls, backend, architecture_id: str = "HPAT_v1"):
        """Construct PhotonicModule from a SimPhonyBackend query.

        Parameters
        ----------
        backend : SimPhonyBackend
            A connected SimPhony backend instance.
        architecture_id : str
            Architecture identifier string.
        """
        arch = backend.architecture_cost()
        prog = backend.programming_cost()
        return cls(
            architecture_id=architecture_id,
            core_count=prog.get("tile_count", 4),
            bank_count=prog.get("tile_count", 4),
            bank_capacity_rings=prog.get("programmed_ring_count", 16384),
            mrr_count=arch.get("mrr_count", 16384),
            pd_count=arch.get("pd_count", 1024),
            dac_count=arch.get("dac_count", 32),
            adc_count=arch.get("adc_count", 16),
            core_height=16,
            core_width=16,
            num_wavelength=16,
            work_freq_ghz=5.0,
            pic_area_um2=arch.get("pic_area_um2", 0.0),
            rf_eic_area_um2=arch.get("rf_eic_area_um2", 0.0),
            core_insertion_loss_db=arch.get("core_insertion_loss_db", 0.0),
            laser_wall_plug_power_w=arch.get("laser_wall_plug_power_w", 0.0),
            hold_power_per_ring_w=prog.get("hold_power_w", 0.0) / max(arch.get("mrr_count", 1), 1),
            program_energy_per_ring_j=(
                prog.get("programming_energy_j", 0.0)
                / max(prog.get("programmed_ring_count", 1), 1)
            ),
            program_time_per_ring_s=prog.get("programming_latency_s", 1000e-9),
        )

    @property
    def total_hold_power_w(self) -> float:
        return self.hold_power_per_ring_w * self.mrr_count

    @property
    def total_program_energy_j(self) -> float:
        return self.program_energy_per_ring_j * self.mrr_count

    @property
    def total_program_latency_s(self) -> float:
        return self.program_time_per_ring_s * (
            self.mrr_count / self.programming_parallelism
        )

    @property
    def clock_period_s(self) -> float:
        return 1.0 / (self.work_freq_ghz * 1e9)

    def summary(self) -> dict:
        return {
            "architecture_id": self.architecture_id,
            "core_count": self.core_count,
            "bank_count": self.bank_count,
            "mrr_count": self.mrr_count,
            "pd_count": self.pd_count,
            "dac_count": self.dac_count,
            "adc_count": self.adc_count,
            "core_dims": f"{self.core_height}x{self.core_width}",
            "num_wavelength": self.num_wavelength,
            "work_freq_ghz": self.work_freq_ghz,
            "pic_area_um2": self.pic_area_um2,
            "rf_eic_area_um2": self.rf_eic_area_um2,
            "core_il_db": self.core_insertion_loss_db,
            "laser_power_w": self.laser_wall_plug_power_w,
            "total_hold_power_w": self.total_hold_power_w,
            "total_program_energy_j": self.total_program_energy_j,
            "total_program_latency_s": self.total_program_latency_s,
        }
