"""HPAT-specific parameter adapter around the cloned SimPhony/LLMCompass.

One kernel query represents ONE physical core. System footprint and continuous
power are queried separately for the complete four-core architecture. This
prevents multiplying an already aggregate kernel by the physical core count.
"""
from __future__ import annotations

import contextlib
import copy
import io
import math
import os
from dataclasses import asdict
from pathlib import Path

from joint_sim.simphony_backend import SimPhonyBackend, _SIMPHONY_ROOT
from joint_sim.electronic_backend import ElectronicBackend
from .energy_ledger import split_components


@contextlib.contextmanager
def simphony_context():
    cwd = Path.cwd()
    try:
        os.chdir(_SIMPHONY_ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        os.chdir(cwd)


class PhysicalCoreCosts:
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.backend = self._make_backend(single_core=True)
        self.system_backend = self._make_backend(single_core=False)
        with simphony_context():
            system = self.system_backend._get_simulator()
            self.architecture = self.system_backend.architecture_cost()
            self.resolved_architecture = copy.deepcopy(system.sub_arch)
            self.programming = self.system_backend.programming_cost()
            from onnarchsim.workflow.energy import initalize_all_devices_power
            raw_components = copy.deepcopy(system._get_components())
            raw_components.pop("node", None)
            components = initalize_all_devices_power(
                system.sub_arch, raw_components,
                system._get_insertion_loss()["core_insertion_loss"])
        self.static_power = {}
        for name, dev in components.items():
            if not isinstance(dev, dict):
                continue
            group = ("laser_static" if "laser" in name and "splitter" not in name else
                     "mrr_hold" if "mrr_weight" in name else "photonic_periphery_static")
            self.static_power[group] = self.static_power.get(group, 0.0) + (
                float(dev.get("static_power", 0)) * float(dev.get("count", 0)) * 1e-3)
        self.laser_initializer_reference_w = self.static_power["laser_static"]
        self.static_power["laser_static"] = self.architecture["laser_wall_plug_power_w"]
        u = config["user_confirmed"]
        expected = math.prod(u["physical_array"]) * u["tiles"] * u["cores_per_tile"]
        if self.architecture["mrr_count"] != expected:
            raise AssertionError(("MRR count", self.architecture["mrr_count"], expected))
        self.electronic = ElectronicBackend()
        if self.electronic.energy_model is None:
            raise RuntimeError("Formal results require the LLMCompass energy model")
        self.electronic_energy = self.electronic.energy_model

    def _make_backend(self, single_core):
        backend = SimPhonyBackend()
        u, a = self.config["user_confirmed"], self.config["model_assumptions"]
        with simphony_context():
            sim = backend._get_simulator()
        sim.sub_arch["tiles"] = 1 if single_core else u["tiles"]
        sim.sub_arch["cores_per_tile"] = 1 if single_core else u["cores_per_tile"]
        # A broadcast MVM streams one input vector per cycle. Physical width
        # is NOT a second independent axis for concurrent input vectors.
        sim._dim_map = {"M": None, "N": "height", "D": "wavelength"}
        core = sim.sub_arch["core"]
        core.update(width=u["physical_array"][0], height=u["physical_array"][1],
                    num_wavelength=u["physical_array"][0], work_freq=a["core_frequency_ghz"])
        core["range"] = {k: "positive" for k in ["weight", "input", "output"]}
        core["precision"] = {"in_bit": a["bits"], "w_bit": a["bits"], "out_bit": a["bits"]}
        # Every physical output row receives its own optical power budget.
        # Upstream H-only distribution omits additional cores for a shared laser.
        core["netlist"]["laser_power_distribution"] = "H*R*C"
        return backend

    def kernel(self, M, K, logical_N):
        """Two positive-input passes through paired +/- weight rows.

Digital subtraction/reduction is charged by the scheduler, not embedded here.
Static component energies and weight-writing components are excluded from the
dynamic ledger; their power/time or event costs are accounted exactly once.
        """
        key = (M, K, logical_N)
        if key in self.cache:
            return self.cache[key]
        physical_N = 2*logical_N
        u, a = self.config["user_confirmed"], self.config["model_assumptions"]
        if K > u["physical_array"][0] or physical_N > u["physical_array"][1]:
            raise ValueError("A core query cannot exceed the physical array")
        bits = a["bits"]
        with simphony_context():
            sim = self.backend._get_simulator()
            base = self.backend.kernel_cost(M, K, physical_N, bits, bits, bits)
            from onnarchsim.workflow.energy import energy_calculator
            devices, latency = energy_calculator(
                sub_arch=sim.sub_arch, device_db=sim.device_db._db,
                dim_map_cfgs={"gemm": sim._dim_map},
                cycles={"gemm": (base["iter_N"], base["iter_K"], base["iter_M"],
                                 physical_N, K, M, 1, 1)},
                insertion_loss=sim._get_insertion_loss()["core_insertion_loss"])
        dynamic = {}
        bias_sample_energy = 0.0
        for name, dev in devices.items():
            if "mrr_weight" in name:
                continue  # Scheduled write events + full-duration hold power.
            if "dac" in name and "_i8-" in name:
                bias_sample_energy = float(dev["dynamic_energy"])*1e-12
                continue  # Bias samples are charged at physical weight writes.
            # In weight-stationary execution inputs change every vector. The
            # upstream reuse factor mistakenly discounts input DAC/EOM energy
            # by M. Use its initialized per-device energy and explicit activity.
            dynamic[name] = 2*M*float(dev["count"])*float(dev["dynamic_energy"])*1e-12
        stage_components = split_components(dynamic)
        result = {
            "M": M, "K": K, "logical_N": logical_N, "physical_N": physical_N,
            "input_sign_passes": 2, "kernel_scope": "one_physical_core",
            "encode_s": 2*latency["operand_encoding_latency_s"],
            "compute_s": 2*latency["compute_latency_s"],
            "convert_s": 2*latency["conversion_latency_s"],
            "dynamic_components_j": dynamic, "dynamic_energy_j": sum(dynamic.values()),
            "stage_components_j": stage_components,
            "stage_energy_j": {s: sum(c.values()) for s, c in stage_components.items()},
            "initialized_devices": devices,
            "bias_dac_sample_j": bias_sample_energy,
            "raw_simphony_reference": base,
        }
        if not all(math.isfinite(result[k]) and result[k] > 0
                   for k in ["encode_s", "compute_s", "convert_s"]):
            raise ValueError("Missing or invalid SimPhony stage latency")
        self.cache[key] = result
        return result

    def summary(self):
        return {"architecture": self.architecture, "static_power_w": self.static_power,
                "laser_initializer_reference_w": self.laser_initializer_reference_w,
                "laser_accounting_source": "architecture_cost API (MZM extinction, total row fanout)",
                "programming": self.programming,
                "resolved_architecture": self.resolved_architecture,
                "electronic_energy": asdict(self.electronic_energy),
                "kernel_scope": "one_physical_core", "physical_cores":
                self.config["user_confirmed"]["tiles"]*self.config["user_confirmed"]["cores_per_tile"]}
