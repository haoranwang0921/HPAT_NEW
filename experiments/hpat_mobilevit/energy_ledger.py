"""Event attribution only: all public energies are joules, durations seconds."""
import math

STAGES = ("dac_encode", "optical_compute", "adc_convert")


def split_components(components):
    groups = {stage: {} for stage in STAGES}
    for name, energy in components.items():
        if not math.isfinite(energy) or energy < 0:
            raise ValueError(f"Invalid component energy: {name}")
        if "mrr_weight" in name or ("dac" in name and "_i8-" in name):
            raise ValueError("Programming/hold components must not enter kernel dynamics")
        if any(part in name for part in ("_dac_", "_mzm_")):
            stage = "dac_encode"
        elif any(part in name for part in ("_adc_", "_tia_", "_photodetector_")):
            stage = "adc_convert"
        elif any(part in name for part in ("_on_chip_laser_", "_micro_comb_", "_laser_splitter_")):
            stage = "optical_compute"
        else:
            raise ValueError(f"Unclassified dynamic component: {name}")
        groups[stage][name] = energy
    return groups


def stage_energies(kernel):
    if "stage_energy_j" in kernel:
        energies = kernel["stage_energy_j"]
        if set(energies) != set(STAGES) or any(
                not math.isfinite(v) or v < 0 for v in energies.values()):
            raise ValueError("Invalid stage energy ledger")
        if not math.isclose(sum(energies.values()), kernel["dynamic_energy_j"],
                            rel_tol=1e-12, abs_tol=1e-24):
            raise ValueError("Stage energy ledger does not close")
        return energies
    # Legacy synthetic test backends expose only an aggregate. Real HPAT
    # kernels always supply the explicit ledger; never infer a device split.
    if "dynamic_components_j" in kernel:
        return {s: sum(c.values()) for s, c in split_components(kernel["dynamic_components_j"]).items()}
    return dict(zip(STAGES, (0.0, kernel["dynamic_energy_j"], 0.0)))
