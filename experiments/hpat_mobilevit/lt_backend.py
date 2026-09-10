"""Controlled independent-core DPTC comparison, not the complete LT accelerator.

Uses local SimPhony LT devices. Inter-core input/ADC sharing and analog temporal
integration are disabled so the explicit joint scheduler owns all transfers and
K accumulation. Full-range phase encoding needs no HPAT sign decomposition.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from .backends import simphony_context
from joint_sim.simphony_backend import SimPhonyBackend
from joint_sim.electronic_backend import ElectronicBackend


class LTCoreCosts:
    def __init__(self, config):
        self.config, self.cache = config, {}
        self.backend = SimPhonyBackend(
            arch_cfg="configs/design/architectures/LT_hetero.yml", sub_arch_name="LT")
        u, a = config["user_confirmed"], config["model_assumptions"]
        self.cores = u["tiles"] * u["cores_per_tile"]
        with simphony_context():
            sim = self.backend._get_simulator()
            sim.sub_arch.update(tiles=1, cores_per_tile=1, dataflow="output_stationary", op_type="MM")
            core = sim.sub_arch["core"]
            core.update(width=u["physical_array"][0], height=u["physical_array"][1],
                        num_wavelength=u["physical_array"][0], work_freq=a["core_frequency_ghz"])
            core["precision"] = {k: a["bits"] for k in ("in_bit", "w_bit", "out_bit")}
            core["netlist"]["temporal_accum_factor"]["duration"] = 1
            self.resolved_single_core = copy.deepcopy(sim.sub_arch)
            arch = self.backend.architecture_cost()
            from onnarchsim.workflow.energy import initalize_all_devices_power
            raw = copy.deepcopy(sim._get_components())
            raw.pop("node", None)
            self.devices = initalize_all_devices_power(sim.sub_arch, raw, arch["core_insertion_loss_db"])
        self.architecture = {k: (v if k == "core_insertion_loss_db" else v*self.cores)
                             for k, v in arch.items()}
        self.static_power = {"laser_static": 0.0, "photonic_periphery_static": 0.0}
        for name, d in self.devices.items():
            key = "laser_static" if ("on_chip_laser" in name or "off_chip_laser" in name) else "photonic_periphery_static"
            self.static_power[key] += d["static_power"]*d["count"]*1e-3*self.cores
        # Match HPAT's architecture-cost link-budget convention and the official
        # LT core formula: H*W already budgets the complete core fanout. Do not
        # multiply that budget again by the physical source-device count.
        self.laser_initializer_reference_w = self.static_power["laser_static"]
        self.static_power["laser_static"] = self.architecture["laser_wall_plug_power_w"]
        self.programming = dict(programmed_mrr_count=0, programming_energy_j=0.0,
                                programming_latency_s=0.0, hold_power_w=0.0, tile_count=self.cores)
        self.electronic = ElectronicBackend()
        self.electronic_energy = self.electronic.energy_model

    def kernel(self, M, K, N):
        key = M, K, N
        if key in self.cache:
            return self.cache[key]
        u, a = self.config["user_confirmed"], self.config["model_assumptions"]
        if not (0 < M <= u["physical_array"][0] and 0 < K <= u["physical_array"][0]
                and 0 < N <= u["physical_array"][1]):
            raise ValueError("LT query must fit one physical DPTC")
        with simphony_context():
            # Direct API supports output_stationary; the generic joint wrapper
            # deliberately remains HPAT-only, avoiding a global semantic change.
            sim = self.backend._get_simulator()
            c = sim.simulate_gemm(M=M, K=K, N=N, input_bits=a["bits"],
                                 weight_bits=a["bits"], output_bits=a["bits"], dataflow="output_stationary")
        if c.switching_cycles != 1:
            raise AssertionError("One DPTC block must be one full-range optical operation")
        dynamic = {name: d["dynamic_energy"]*d["count"]*1e-12 for name, d in self.devices.items()}
        result = dict(M=M, K=K, N=N, switching_cycles=1, input_sign_passes=1,
                      # Conservative serial encoding of both operands; the API
                      # calls the second operand stage 'programming' even for MZM.
                      encode_s=c.operand_encoding_latency_s+c.programming_latency_s,
                      compute_s=c.compute_latency_s, convert_s=c.conversion_latency_s,
                      dynamic_energy_j=sum(dynamic.values()), dynamic_components_j=dynamic)
        self.cache[key] = result
        return result

    def summary(self):
        return dict(architecture=self.architecture, static_power_w=self.static_power,
                    laser_initializer_reference_w=self.laser_initializer_reference_w,
                    laser_accounting="Per-core architecture H*W link budget times independent core count, not source count",
                    resolved_single_core=self.resolved_single_core, physical_cores=self.cores,
                    electronic_energy=asdict(self.electronic_energy), programming=self.programming,
                    scope="Independent LT DPTC cores; no inter-core optical sharing or analog temporal accumulation; not area matched")
