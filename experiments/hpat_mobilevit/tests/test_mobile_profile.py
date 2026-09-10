import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from joint_sim.electronic_backend import ElectronicBackend
from experiments.hpat_mobilevit.mobile_profile import resolve_profile, profile_costs
from experiments.hpat_mobilevit.streaming import TileStreamSimulation
from experiments.hpat_mobilevit.tests.test_streaming import FakeCosts, rec

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return (json.loads((ROOT/"config_10ps_fixed_write_energy.json").read_text()),
            json.loads((ROOT/"mobile_reference.json").read_text()))


@pytest.mark.parametrize("tops", [10, 20, 35])
def test_dense_int8_throughput_units_and_execution(tops):
    base, spec = inputs()
    cfg = resolve_profile(base, spec, tops, "mobile_reference")
    assert cfg["model_assumptions"]["electronic_peak_ops_per_s"] == tops*1e12
    sim = TileStreamSimulation(cfg, FakeCosts(), "digital", keep_events=True)
    sim.run([rec()], 1)
    event = next(e for e in sim.events if e["event_type"] == "electronic_dynamic")
    assert event["duration_s"] == pytest.approx(2*rec()["macs"]/(tops*1e12), rel=1e-12, abs=1e-25)
    assert cfg["user_confirmed"]["program_response_time_s"] == 1e-11
    assert cfg["model_assumptions"]["program_energy_multiplier"] == 100000


def test_mobile_energy_overrides_do_not_mutate_original():
    base, spec = inputs()
    original = copy.deepcopy(base)
    electronic = ElectronicBackend()
    reference = SimpleNamespace(config=base, electronic=electronic, electronic_energy=electronic.energy_model)
    cfg = resolve_profile(base, spec, 10, "mobile_reference")
    actual = profile_costs(reference, cfg)
    assert base == original
    assert reference.electronic_energy.static_power_w == 15
    assert actual.electronic_energy.static_power_w == 0.3
    assert actual.electronic_energy.hbm_energy_per_byte_j == 1e-10
    assert actual.electronic_energy.matmul_energy_per_mac_j == reference.electronic_energy.matmul_energy_per_mac_j
    assert actual.electronic.energy_model is actual.electronic_energy


def test_compute_only_control_retains_memory_and_energy():
    base, spec = inputs()
    cfg = resolve_profile(base, spec, 10, "compute_only_control")
    for key in base["model_assumptions"]:
        if key != "electronic_peak_flops":
            assert cfg["model_assumptions"][key] == base["model_assumptions"][key]
    assert cfg["model_assumptions"]["electronic_energy_overrides"] == {}


def test_d0_user_utilization_and_mac_energy():
    base, spec = inputs()
    spec['compute_utilization'] = 0.8
    cfg = resolve_profile(base, spec, 10, 'mobile_reference')
    cfg['model_assumptions']['electronic_energy_overrides']['matmul_energy_per_mac_j'] = 0.9e-12
    electronic = ElectronicBackend()
    original = electronic.energy_model.matmul_energy_per_mac_j
    costs = profile_costs(SimpleNamespace(electronic=electronic,
        electronic_energy=electronic.energy_model), cfg)
    assert costs.electronic_energy.matmul_energy(1000) == pytest.approx(9e-10)
    assert electronic.energy_model.matmul_energy_per_mac_j == original
    sim = TileStreamSimulation(cfg, FakeCosts(), 'digital', keep_events=True)
    sim.run([rec()], 1)
    event = next(e for e in sim.events if e['event_type'] == 'electronic_dynamic')
    assert event['duration_s'] == pytest.approx(2*rec()['macs']/8e12, abs=1e-25)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_peak_rejected(value):
    base, spec = inputs()
    with pytest.raises(ValueError):
        resolve_profile(base, spec, value, "mobile_reference")


def test_dense_peak_cannot_silently_receive_sparsity_multiplier():
    base, spec = inputs()
    spec["compute_utilization"] = 2
    with pytest.raises(ValueError):
        resolve_profile(base, spec, 10, "mobile_reference")


def test_vector_and_digital_reduction_share_same_mobile_throughput():
    base, spec = inputs()
    cfg = resolve_profile(base, spec, 10, "mobile_reference")
    sim = TileStreamSimulation(cfg, FakeCosts(), "linear", keep_events=True)
    sim.run([rec(M=4, K=16, N=8)], 1)
    event = next(e for e in sim.events if e["event_type"] == "signed_reduce")
    assert event["duration_s"] == pytest.approx(4*8*3/1e12, rel=1e-12, abs=1e-25)
