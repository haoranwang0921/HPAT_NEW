"""
Direct GEMM cost simulator for photonic accelerators.

Exposes a clean API that accepts (M, K, N) dimensions and returns
structured PhotonicKernelCost and ArchitectureCost without requiring
a full PyTorch nn.Module or TorchONN dependency.

This is the SimPhony-side backend for the joint LLMCompass-SimPhony
simulation described in the implementation plan (Section 7 / Phase C).

Usage:
    sim = GemmCostSimulator(
        arch_cfg="configs/design/architectures/HPAT_hetero.yml",
        device_root="configs/devices",
        arch_version="v1",
    )
    cost = sim.simulate_gemm(M=257, K=512, N=512)
    arch = sim.get_architecture_cost()
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from onnarchsim.database.device_db import DeviceLib
from onnarchsim.database.hetero_arch_db import HeteroArchitectureLib
from onnarchsim.database.utils import break_path
from onnarchsim.workflow.area import area_calculator, chip_area_calculator
from onnarchsim.workflow.dataflow import cycles, loading_factor_calculator
from onnarchsim.workflow.energy import energy_calculator, chip_energy_calculator
from onnarchsim.workflow.insertion_loss import architecture_insertion_loss
from onnarchsim.workflow.utils import load_required_devices
from onnarchsim.schema import (
    PhotonicCostRequest,
    PhotonicKernelCost,
    ArchitectureCost,
    WeightProgrammingCost,
    SCHEMA_VERSION,
)

__all__ = ["GemmCostSimulator"]


class GemmCostSimulator:
    """Direct GEMM-to-photonic-cost simulator.

    Initialises the architecture and device databases once, then allows
    repeated gemm queries with cached architecture-level costs.
    """

    def __init__(
        self,
        arch_cfg: str = "configs/design/architectures/HPAT_hetero.yml",
        device_root: str = "configs/devices",
        device_cfg_globs: List[str] | str = ["*/*.yml"],
        arch_version: str = "v1",
        sub_arch_name: str = "HPAT",
    ):
        root, config_file = break_path(arch_cfg)
        self.arch_db = HeteroArchitectureLib(
            root=root, config_file=config_file, version=arch_version
        )
        self.device_db = DeviceLib(
            root=device_root,
            config_files=(
                device_cfg_globs
                if isinstance(device_cfg_globs, list)
                else [device_cfg_globs]
            ),
        )
        self.arch_version = arch_version

        # Resolve the named sub-architecture
        hetero = self.arch_db.dict().get("hetero_arch", {})
        self.sub_arch = None
        for _key, sa in hetero.get("sub_archs", {}).items():
            if sa.get("name") == sub_arch_name:
                self.sub_arch = sa
                break
        if self.sub_arch is None:
            raise ValueError(
                f"Sub-architecture '{sub_arch_name}' not found in {arch_cfg}"
            )
        self.sub_arch_name = sub_arch_name

        # Default dim_map for Linear layers
        self._dim_map = {"M": "width", "N": "height", "D": "wavelength"}

        # Cached architecture-level costs
        self._arch_cost: Optional[ArchitectureCost] = None
        self._insertion_loss_cache: Optional[Dict] = None
        self._components_cache: Optional[Dict] = None
        self._area_cache: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # Architecture cost (one-time, cached)
    # ------------------------------------------------------------------
    def get_architecture_cost(self) -> ArchitectureCost:
        if self._arch_cost is not None:
            return self._arch_cost

        il = self._get_insertion_loss()
        area_raw, _ = self._get_area()

        # Count devices
        components = self._get_components()
        mrr_count = 0
        pd_count = 0
        dac_count = 0
        adc_count = 0
        for dev_name, dev_info in components.items():
            if isinstance(dev_info, dict):
                cnt = dev_info.get("count", 0)
                dn = dev_name.lower()
                if "mrr" in dn and "rerouter" not in dn:
                    mrr_count += cnt
                elif "photodetector" in dn:
                    pd_count += cnt
                elif "dac" in dn:
                    dac_count += cnt
                elif "adc" in dn:
                    adc_count += cnt

        # Compute laser wall-plug power directly (static_power is only set
        # inside energy_calculator, not available in raw component configs).
        laser_wall_plug = self._compute_laser_power(components, il.get("core_insertion_loss", 0.0))

        self._arch_cost = ArchitectureCost(
            pic_area_um2=_safe_get(area_raw, "PIC_1", 0.0),
            rf_eic_area_um2=_safe_get(area_raw, "RF_EIC", 0.0)
            + _safe_get(area_raw, "PIC_2", 0.0),
            total_area_um2=_safe_get(area_raw, "total", 0.0),
            core_insertion_loss_db=il.get("core_insertion_loss", 0.0),
            laser_wall_plug_power_w=laser_wall_plug,
            mrr_count=mrr_count,
            pd_count=pd_count,
            dac_count=dac_count,
            adc_count=adc_count,
        )
        return self._arch_cost

    def get_weight_programming_cost(self) -> WeightProgrammingCost:
        """Return the one-time cost to program all MRR weights."""
        arch = self.get_architecture_cost()
        core_cfg = self.sub_arch["core"]
        work_freq = core_cfg.get("work_freq", 5.0)
        response_time_ns = 1000.0  # default from HPAT_MRR

        # Try to read from device config
        try:
            mrr_dev = self.device_db._db.get("mrr_weight", {}).get("HPAT_MRR", {})
            cfgs = mrr_dev.get("cfgs", {})
            response_time_ns = cfgs.get("response_time", response_time_ns)
            program_power_mw = cfgs.get("program_power", cfgs.get("dynamic_power", 0.093))
            hold_power_mw = cfgs.get("static_power", 0.1)
        except Exception:
            program_power_mw = 0.093
            hold_power_mw = 0.1

        tuning_cycles = response_time_ns * work_freq
        programming_latency_s = tuning_cycles / work_freq * 1e-9
        # E = P_program * T_program * mrr_count
        programming_energy_j = (
            program_power_mw * 1e-3  # mW -> W
            * response_time_ns * 1e-9  # ns -> s
            * arch.mrr_count
        )
        hold_power_w = arch.mrr_count * float(hold_power_mw) * 1e-3

        return WeightProgrammingCost(
            tile_count=(
                self.sub_arch.get("tiles", 1)
                * self.sub_arch.get("cores_per_tile", 1)
            ),
            programmed_mrr_count=arch.mrr_count,
            programming_latency_s=programming_latency_s,
            programming_energy_j=programming_energy_j,
            hold_power_w=hold_power_w,
        )

    # ------------------------------------------------------------------
    # Per-GEMM cost
    # ------------------------------------------------------------------
    def simulate_gemm(
        self,
        M: int,
        K: int,
        N: int,
        input_bits: int = 8,
        weight_bits: int = 8,
        output_bits: int = 8,
        dataflow: str = "weight_stationary",
    ) -> PhotonicKernelCost:
        """Simulate a single GEMM: input[M,K] @ weight[N,K]^T.

        Args:
            M: batch size / number of input vectors.
            K: inner dimension (dot-product size).
            N: number of output features.
            input_bits: bit width of input activations.
            weight_bits: bit width of weights.
            output_bits: bit width of accumulated outputs.
            dataflow: one of weight_stationary, input_stationary, output_stationary.

        Returns:
            PhotonicKernelCost with latency, energy, and tiling breakdown.
        """
        core_cfg = self.sub_arch["core"]
        work_freq = core_cfg.get("work_freq", 5)
        num_wavelength = core_cfg.get("num_wavelength", 1)
        forward_type = core_cfg.get("forward", "direct")
        weight_rep = core_cfg.get("range", {}).get("weight", "full")
        input_rep = core_cfg.get("range", {}).get("input", "full")
        output_rep = core_cfg.get("range", {}).get("output", "full")
        op_type = self.sub_arch.get("op_type", "MVM")

        # Architecture dimensions for tiling
        tiles = self.sub_arch.get("tiles", 1)
        cores_per_tile = self.sub_arch.get("cores_per_tile", 1)
        core_height = core_cfg.get("height", 16)
        core_width = core_cfg.get("width", 16)
        miniblock = [tiles, cores_per_tile, core_height, core_width]

        # Call cycles() — the core tiling function
        matrices = ((K, M), (N, K))
        (
            iter_N, iter_D, iter_M,
            _N, _D, _M,
            fw_w, fw_x, arch_dims,
        ) = cycles(
            miniblock=miniblock,
            matrices=matrices,
            layer_sizes={},
            dataflow=dataflow,
            op_type=op_type,
            dim_map=self._dim_map,
            multi_wavelength=num_wavelength,
            work_freq=work_freq,
            forward_type=forward_type,
            weight_representation=weight_rep,
            input_representation=input_rep,
            output_representation=output_rep,
        )

        switching_cycles = iter_N * iter_D * iter_M

        # Call energy_calculator for device-level costs
        il = self._get_insertion_loss()
        core_il = il.get("core_insertion_loss", 0.0)

        # Build a single-entry cycles dict
        cycles_dict = {
            "gemm": (
                iter_N, iter_D, iter_M,
                N, K, M,
                fw_w, fw_x,
            )
        }
        dim_map_dict = {"gemm": self._dim_map}
        # We need a sub_arch dict that energy_calculator can consume.
        energy_devices, comp_latency = energy_calculator(
            sub_arch=self.sub_arch,
            device_db=self.device_db._db,
            dim_map_cfgs=dim_map_dict,
            cycles=cycles_dict,
            insertion_loss=core_il,
        )

        # Extract per-component energies
        dac_energy_pJ = 0.0
        adc_energy_pJ = 0.0
        mrr_tune_energy_pJ = 0.0
        mrr_hold_energy_pJ = 0.0
        laser_energy_pJ = 0.0
        total_dynamic_pJ = 0.0
        total_static_pJ = 0.0

        for dev_name, dev_info in energy_devices.items():
            if not isinstance(dev_info, dict):
                continue
            dn = dev_name.lower()
            de = dev_info.get("total_dynamic_energy", 0.0)
            se = dev_info.get("total_static_energy", 0.0)
            total_dynamic_pJ += de
            total_static_pJ += se
            if "dac" in dn:
                dac_energy_pJ += de + se
            elif "adc" in dn:
                adc_energy_pJ += de + se
            elif "mrr" in dn and "rerouter" not in dn:
                mrr_tune_energy_pJ += de
                mrr_hold_energy_pJ += se
            elif "laser" in dn:
                laser_energy_pJ += se

        # Compute utilisation: actual MACs / max MACs in the allocated cycles
        total_macs = M * K * N
        max_macs_per_cycle = (
            arch_dims.get("width", 1)
            * arch_dims.get("height", 1)
            * arch_dims.get("wavelength", 1)
        )
        max_possible_macs = 0
        max_cycles_val = 0
        if isinstance(comp_latency, dict):
            # Legacy fields are in seconds after B4; convert back to cycles
            tuning_s = float(comp_latency.get("total_tuning_latency", 0.0))
            static_s = float(comp_latency.get("total_static_latency", 0.0))
            max_cycles_val = int(round(max(tuning_s, static_s) * work_freq * 1e9))
        max_possible_macs = switching_cycles * max_macs_per_cycle
        utilization = total_macs / max_possible_macs if max_possible_macs > 0 else 0.0

        # Latency breakdown from energy_calculator
        compute_lat = comp_latency.get("compute_latency_s", 0.0)
        encode_lat = comp_latency.get("operand_encoding_latency_s", 0.0)
        conv_lat = comp_latency.get("conversion_latency_s", 0.0)
        prog_lat = comp_latency.get("programming_latency_s", 0.0)

        # Fallback: if new fields are zero, derive from legacy fields
        if compute_lat == 0.0:
            tuning_lat = float(comp_latency.get("total_tuning_latency", 0.0))
            static_lat = float(comp_latency.get("total_static_latency", 0.0))
            compute_lat = tuning_lat
            conv_lat = static_lat

        return PhotonicKernelCost(
            compute_latency_s=compute_lat,
            operand_encoding_latency_s=encode_lat,
            conversion_latency_s=conv_lat,
            programming_latency_s=prog_lat,
            dynamic_energy_j=(total_dynamic_pJ * 1e-12),
            dac_energy_j=(dac_energy_pJ * 1e-12),
            adc_energy_j=(adc_energy_pJ * 1e-12),
            laser_energy_j=(laser_energy_pJ * 1e-12),
            mrr_tuning_energy_j=(mrr_tune_energy_pJ * 1e-12),
            mrr_hold_energy_j=(mrr_hold_energy_pJ * 1e-12),
            input_bytes=M * K * input_bits // 8,
            output_bytes=M * N * output_bits // 8,
            iter_M=iter_M,
            iter_K=iter_D,
            iter_N=iter_N,
            switching_cycles=switching_cycles,
            max_cycles=max_cycles_val,
            utilization=utilization,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_laser_power(self, components: Dict, core_il_db: float) -> float:
        """Compute laser wall-plug power in watts from component configs.

        Mirrors the logic in energy.py:initalize_all_devices_power /
        cal_laser_power, but operates on raw component dicts before
        power initialisation.
        """
        from onnarchsim.workflow.energy import cal_laser_power
        from onnarchsim.workflow.utils import evaluate_factors, parse_device_name

        core_cfg = self.sub_arch["core"]
        R = self.sub_arch.get("tiles", 1)
        C = self.sub_arch.get("cores_per_tile", 1)
        H = core_cfg.get("height", 1)
        W = core_cfg.get("width", 1)
        N = core_cfg.get("num_wavelength", 1)
        in_bit = core_cfg.get("precision", {}).get("in_bit", 8)

        # Distribution factor
        dist_factor = evaluate_factors(
            core_cfg.get("netlist", {}).get(
                "laser_power_distribution",
                core_cfg.get("netlist", {}).get("laser_power_distribution_factor", 1),
            ),
            R, C, H, W, N,
        )

        laser_dev = None
        sensitivity = -27.0  # dBm — default photodetector sensitivity
        extinction_ratio = 0.0

        for dev_name, dev_cfg in components.items():
            if not isinstance(dev_cfg, dict):
                continue
            # parse_device_name may fail for "node" or other special entries
            try:
                part, dev_type, idx, instance, name = parse_device_name(dev_name)
            except (ValueError, IndexError):
                dev_type = dev_name
            # Laser — match on_chip/off_chip but NOT laser_splitter
            is_laser = (
                "on_chip_laser" in dev_type
                or "off_chip_laser" in dev_type
                or "on_chip_laser" in dev_name.lower()
                or "off_chip_laser" in dev_name.lower()
            )
            if is_laser and "splitter" not in dev_type.lower() and "splitter" not in dev_name.lower():
                laser_dev = dev_cfg
            # Photodetector sensitivity
            if "photodetector" in dev_type and "cfgs" in dev_cfg:
                sensitivity = max(sensitivity,
                                  dev_cfg["cfgs"].get("sensitivity", sensitivity))
            # Extinction ratio from MZM
            if "mzm" in dev_type and "cfgs" in dev_cfg:
                er = dev_cfg["cfgs"].get("extinction_ratio", 0.0) or 0.0
                extinction_ratio = max(extinction_ratio, er)

        if laser_dev is None:
            return 0.0

        # cal_laser_power returns mW; convert to W
        laser_power_mw = cal_laser_power(
            laser_config=laser_dev,
            in_bit=in_bit,
            insertion_loss=core_il_db,
            photo_detector_sensitivity=sensitivity,
            modulation_extinction_ratio=extinction_ratio,
            distribution_factor=int(dist_factor),
        )
        return laser_power_mw / 1000.0  # mW -> W

    def _get_components(self) -> Dict:
        if self._components_cache is not None:
            return self._components_cache
        components, _, _, _ = load_required_devices(
            self.sub_arch, self.device_db._db
        )
        self._components_cache = components
        return components

    def _get_insertion_loss(self) -> Dict:
        if self._insertion_loss_cache is not None:
            return self._insertion_loss_cache
        _, node_il, _, core_il = architecture_insertion_loss(
            self.sub_arch, self.device_db._db
        )
        self._insertion_loss_cache = {
            "node_insertion_loss": node_il,
            "core_insertion_loss": core_il,
        }
        return self._insertion_loss_cache

    def _get_area(self) -> Tuple:
        if self._area_cache is not None:
            return self._area_cache
        area_breakdown = area_calculator(
            config=self.sub_arch, device_db=self.device_db._db
        )
        total_area_raw = sum(
            v.get("total_area", 0)
            for v in area_breakdown.values()
            if isinstance(v, dict) and "node" not in v
        )
        node_area = area_breakdown.get("node", {}).get("total_area", 0)
        total_area = total_area_raw + node_area
        # Classify by chip type
        chip_areas = {}
        for _dev_name, dev_info in area_breakdown.items():
            if isinstance(dev_info, dict):
                ct = dev_info.get("chip_type", "off_chip")
                a = dev_info.get("total_area", 0)
                chip_areas[ct] = chip_areas.get(ct, 0) + a
        chip_areas["total"] = total_area
        self._area_cache = (chip_areas, area_breakdown)
        return self._area_cache

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    @property
    def commit_hash(self) -> str:
        return "0bdc10d"  # TODO: read from git at build time


def _safe_get(d: Dict, key: str, default: Any = 0.0) -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return default
