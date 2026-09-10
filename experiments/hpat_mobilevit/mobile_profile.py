"""Explicit integer-throughput and energy adapter; historical code is unchanged."""
from __future__ import annotations

import copy
import math
from dataclasses import replace


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def resolve_profile(base, specification, tops, kind):
    positive(tops, "dense_int8_tops")
    utilization = positive(specification["compute_utilization"], "compute_utilization")
    if utilization > 1 or specification["operations_per_mac"] != 2:
        raise ValueError("Require utilization <= 1 and two operations per MAC")
    if kind not in specification["profiles"]:
        raise ValueError(f"Unknown profile: {kind}")
    cfg = copy.deepcopy(base)
    a, u = cfg["model_assumptions"], cfg["user_confirmed"]
    if a["bits"] != 8 or a["accumulator_bits"] != 32:
        raise ValueError("This reference profile requires INT8/INT32 cost representation")
    a["electronic_peak_ops_per_s"] = tops*1e12
    a["electronic_compute_utilization"] = utilization
    a["electronic_effective_ops_per_s"] = tops*1e12*utilization
    # Existing scheduler divides 2*MAC by this field. It is a numeric alias,
    # NOT a claim of floating-point throughput in the new integer profile.
    a["electronic_peak_flops"] = a["electronic_effective_ops_per_s"]
    a["electronic_operations_per_mac"] = 2
    a["electronic_matrix_precision"] = specification["matrix_precision"]
    u["electronic_profile"] = f"{kind}_{tops:g}tops"
    cfg["sweeps"] = {}
    overrides = {}
    if kind == "mobile_reference":
        m = specification["mobile_assumptions"]
        for key in ["external_memory_bandwidth_bytes_per_s", "scratchpad_bandwidth_bytes_per_s",
                    "electronic_static_power_w", "external_memory_energy_per_byte_j", "vector_peak_fraction"]:
            positive(m[key], key)
        if m["vector_peak_fraction"] > 1:
            raise ValueError("vector_peak_fraction must be <= 1")
        a["hbm_bandwidth_bytes_per_s"] = m["external_memory_bandwidth_bytes_per_s"]
        a["sram_bandwidth_bytes_per_s"] = m["scratchpad_bandwidth_bytes_per_s"]
        a["vector_peak_fraction"] = m["vector_peak_fraction"]
        a["external_memory_type"] = m["external_memory_type"]
        overrides = {"static_power_w": m["electronic_static_power_w"],
                     "hbm_energy_per_byte_j": m["external_memory_energy_per_byte_j"]}
    elif kind != "compute_only_control":
        raise ValueError(f"Unsupported profile: {kind}")
    a["electronic_energy_overrides"] = overrides
    # Batch-2 accounting switches (default off = historical behaviour):
    #   electronic_io_compute_overlap: input DRAM prefetch runs concurrently
    #     with electronic compute (latency = max(compute, fetch) per operator
    #     instead of the serial sum) — the bandwidth-floor semantics.
    #   activity_gated_static: electronic static power is charged on the
    #     electronic-busy union of events, idle windows at a fraction; the
    #     laser is charged only during photonic-busy windows (gating) with an
    #     optional standby fraction.
    gates = m if kind == "mobile_reference" else {}
    a["electronic_io_compute_overlap"] = bool(gates.get("electronic_io_compute_overlap", False))
    a["activity_gated_static"] = bool(gates.get("activity_gated_static", False))
    a["reduce_compute_overlap"] = bool(gates.get("reduce_compute_overlap", False))
    a["accumulator_residency"] = bool(gates.get("accumulator_residency", False))
    a["dma_batch_fetch"] = bool(gates.get("dma_batch_fetch", False))
    # Stage-level timing switch (default off = historical serialized behaviour):
    #   photonic_stage_pipeline: model DAC encoding, optical pass-through and
    #     ADC conversion as three separate hardware pools (the architecture
    #     reports distinct dac/adc/mrr device counts) that stream different
    #     input vectors concurrently, instead of chaining the three stage
    #     latencies serially on one core timeline. Energy per stage is
    #     unchanged; only the achievable overlap of the three stages changes.
    a["photonic_stage_pipeline"] = bool(gates.get("photonic_stage_pipeline", False))
    a["electronic_idle_power_fraction"] = float(gates.get("electronic_idle_power_fraction", 0.1))
    a["laser_standby_power_fraction"] = float(gates.get("laser_standby_power_fraction", 0.0))
    for fname in ["electronic_idle_power_fraction", "laser_standby_power_fraction"]:
        if not 0.0 <= a[fname] <= 1.0:
            raise ValueError(f"{fname} must be within [0, 1]")
    cfg["mobile_reference"] = dict(profile_kind=kind, dense_int8_tops=tops,
        source_specification=copy.deepcopy(specification),
        compatibility_aliases={"electronic_peak_flops": "effective integer operations/s",
                               "hbm": "external-DRAM accounting resource, not HBM hardware"})
    cfg["scope"] = ("Uncalibrated mobile-class architecture reference, not a named phone or measured NPU. "
                    "Dense INT8, 1 MAC=2 operations; no sparsity or utilization gain. "
                    "Same profile for D0 and hybrid residuals. 10 ps whole-array writes at unchanged event energy. "
                    + ("Compute-only control retains old memory and energy parameters; it is not a phone profile."
                       if kind == "compute_only_control" else "Memory, vector and power assumptions require calibration."))
    return cfg


def profile_costs(base_costs, cfg):
    result = copy.copy(base_costs)
    result.config = cfg
    allowed = {"static_power_w", "hbm_energy_per_byte_j", "matmul_energy_per_mac_j"}
    overrides = cfg["model_assumptions"]["electronic_energy_overrides"]
    if not set(overrides) <= allowed:
        raise ValueError("Unsupported electronic energy override")
    for key, value in overrides.items():
        positive(value, key)
    result.electronic_energy = replace(base_costs.electronic_energy, **overrides)
    result.electronic = copy.copy(base_costs.electronic)
    result.electronic.energy_model = result.electronic_energy
    return result
