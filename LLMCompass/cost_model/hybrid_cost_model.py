"""
Hybrid electronic-photonic cost model (Phase G4).

Combines LLMCompass electronic area estimates (calc_compute_chiplet_area_mm2,
calc_io_die_area_mm2) with SimPhony photonic area (PIC + RF EIC) to produce
a unified, per-die area breakdown.

Area rule (Plan Section 11 G4):
  A_digital = A_compute_chiplet - A_replaced_systolic + A_io_die
  A_hybrid  = A_digital + A_pic + A_rf_eic

The systolic array is replaced because photonic MRR banks perform the
dot-product operations in the optical domain.
"""

import json
import os
from typing import Optional

from hardware_model.device import Device
from hardware_model.photonic_module import PhotonicModule
from hardware_model.energy_model import ElectronicEnergyModel
from cost_model.cost_model import (
    calc_compute_chiplet_area_mm2,
    calc_io_die_area_mm2,
    calc_systolic_array_area_mm2,
)


class HybridCostModel:
    """Merges electronic and photonic costs into a unified, per-die breakdown.

    Parameters
    ----------
    electronic_config_path : str
        Path to the LLMCompass device JSON config (e.g. configs/GA100.json).
    photonic_module : PhotonicModule
        The photonic accelerator module (populated from SimPhony).
    energy_model : ElectronicEnergyModel, optional
    """

    def __init__(
        self,
        electronic_config_path: str,
        photonic_module: PhotonicModule,
        energy_model: Optional[ElectronicEnergyModel] = None,
    ):
        self._config_path = electronic_config_path
        self.photonic_module = photonic_module
        self.energy_model = energy_model or ElectronicEnergyModel.nominal()

        with open(electronic_config_path, "r") as f:
            self._device_config = json.load(f)

        # Cache
        self._compute_area_mm2: Optional[float] = None
        self._io_area_mm2: Optional[float] = None
        self._core_breakdown: Optional[dict] = None
        self._io_breakdown: Optional[dict] = None
        self._compute_breakdown: Optional[dict] = None

    # ------------------------------------------------------------------
    # Area computation
    # ------------------------------------------------------------------

    def compute_electronic_areas(self):
        """Run LLMCompass cost_model once and cache all results."""
        if self._compute_area_mm2 is not None:
            return

        config = self._device_config

        # Compute chiplet (digital logic)
        compute_area, core_bd, compute_bd = calc_compute_chiplet_area_mm2(
            config, verbose=True
        )
        self._compute_area_mm2 = compute_area
        self._core_breakdown = core_bd
        self._compute_breakdown = compute_bd

        # IO die (HBM PHY, global buffer, NVLink)
        io_area, io_bd = calc_io_die_area_mm2(config, verbose=True)
        self._io_area_mm2 = io_area
        self._io_breakdown = io_bd

    # ------------------------------------------------------------------
    # Hybrid area breakdown
    # ------------------------------------------------------------------

    def hybrid_area_breakdown(self) -> dict:
        """Return per-die area breakdown with photonic substitution.

        A_digital = A_cores - A_systolic_array + A_io_die
        A_hybrid  = A_digital + A_pic + A_rf_eic
        """
        self.compute_electronic_areas()

        # Systolic array area to subtract (replaced by photonic)
        sa_area_mm2 = self._core_breakdown.get("sa_area", 0.0)
        cores_area_mm2 = self._core_breakdown.get("total_core_area", 0.0)
        control_area_mm2 = self._core_breakdown.get("control_area", 0.0)
        alu_area_mm2 = self._core_breakdown.get("alu_area", 0.0)
        regfile_area_mm2 = self._core_breakdown.get("regfile_area", 0.0)
        local_buffer_mm2 = self._core_breakdown.get("local_buffer_area", 0.0)
        crossbar_area_mm2 = self._compute_breakdown.get("crossbar_area", 0.0)

        # Compute die without systolic arrays
        cores_without_sa = cores_area_mm2 - sa_area_mm2
        io_die_mm2 = self._io_area_mm2

        digital_mm2 = cores_without_sa + crossbar_area_mm2 + io_die_mm2

        # Photonic
        pic_mm2 = self.photonic_module.pic_area_um2 * 1e-6
        rf_eic_mm2 = self.photonic_module.rf_eic_area_um2 * 1e-6
        photonic_mm2 = pic_mm2 + rf_eic_mm2

        total_mm2 = digital_mm2 + photonic_mm2

        # Electronic baseline (for comparison)
        electronic_baseline_mm2 = (
            self._compute_area_mm2 + self._io_area_mm2
        )

        return {
            # Photonic dies
            "pic_area_mm2": pic_mm2,
            "rf_eic_area_mm2": rf_eic_mm2,
            "photonic_total_mm2": photonic_mm2,
            # Digital dies (electronic minus SA)
            "cores_without_sa_mm2": cores_without_sa,
            "crossbar_mm2": crossbar_area_mm2,
            "io_die_mm2": io_die_mm2,
            "digital_total_mm2": digital_mm2,
            # Totals
            "hybrid_total_mm2": total_mm2,
            "electronic_baseline_mm2": electronic_baseline_mm2,
            # Removed area
            "replaced_systolic_array_mm2": sa_area_mm2,
            # Per-component electronic breakdown
            "electronic_breakdown": {
                "control_area_mm2": control_area_mm2,
                "alu_area_mm2": alu_area_mm2,
                "sa_area_mm2": sa_area_mm2,
                "regfile_area_mm2": regfile_area_mm2,
                "local_buffer_mm2": local_buffer_mm2,
                "crossbar_mm2": crossbar_area_mm2,
                "io_die_mm2": io_die_mm2,
            },
            # IO die breakdown
            "io_breakdown": self._io_breakdown,
        }

    # ------------------------------------------------------------------
    # Energy / Power
    # ------------------------------------------------------------------

    def electronic_energy_for_matmul(
        self, M: int, K: int, N: int, latency_s: float
    ) -> dict:
        mac_count = M * K * N
        compute_j = self.energy_model.matmul_energy(mac_count)
        io_bytes = (M * K + K * N + M * N) * 2
        memory_j = (
            self.energy_model.sram_read_energy(io_bytes // 2)
            + self.energy_model.hbm_read_energy(io_bytes // 2)
        )
        static_j = self.energy_model.static_energy(latency_s)
        return {
            "compute_energy_j": compute_j,
            "memory_energy_j": memory_j,
            "static_energy_j": static_j,
            "total_energy_j": compute_j + memory_j + static_j,
        }

    def total_static_power_w(self) -> float:
        e_static = self.energy_model.static_power_w
        p_laser = self.photonic_module.laser_wall_plug_power_w
        p_hold = self.photonic_module.total_hold_power_w
        return e_static + p_laser + p_hold

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "photonic": self.photonic_module.summary(),
            "electronic": {
                "matmul_energy_per_mac_pJ": self.energy_model.matmul_energy_per_mac_j * 1e12,
                "sram_energy_per_byte_pJ": self.energy_model.sram_energy_per_byte_j * 1e12,
                "hbm_energy_per_byte_pJ": self.energy_model.hbm_energy_per_byte_j * 1e12,
                "static_power_w": self.energy_model.static_power_w,
                "process_node": self.energy_model.process_node,
            },
            "total_static_power_w": self.total_static_power_w(),
        }
