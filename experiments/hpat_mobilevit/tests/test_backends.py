import json
from pathlib import Path

import pytest

from experiments.hpat_mobilevit.backends import PhysicalCoreCosts


@pytest.fixture(scope="module")
def backend():
    root = Path(__file__).resolve().parents[1]
    return PhysicalCoreCosts(json.loads((root/"config.json").read_text()))


def test_small_physical_architecture(backend):
    assert backend.architecture["mrr_count"] == 1024
    assert backend.architecture["pd_count"] == 64
    assert backend.architecture["adc_count"] == 64
    assert backend.architecture["dac_count"] == 128
    assert backend.backend._get_simulator().get_architecture_cost().mrr_count == 256


def test_input_vector_count_scales_latency_and_dac_energy(backend):
    a = backend.kernel(1, 16, 8)
    b = backend.kernel(10, 16, 8)
    assert b["compute_s"] == pytest.approx(10*a["compute_s"])
    assert b["compute_s"] == pytest.approx(20/5e9)
    assert b["dynamic_energy_j"] == pytest.approx(10*a["dynamic_energy_j"])
    dac = next(k for k in a["dynamic_components_j"] if "dac" in k)
    assert b["dynamic_components_j"][dac] == pytest.approx(10*a["dynamic_components_j"][dac])
    assert all("_i8-" not in k for k in a["dynamic_components_j"])


def test_single_core_query_bounds_and_energy_closure(backend):
    with pytest.raises(ValueError, match="physical array"):
        backend.kernel(1, 17, 8)
    with pytest.raises(ValueError, match="physical array"):
        backend.kernel(1, 16, 9)
    c = backend.kernel(1, 16, 8)
    assert c["dynamic_energy_j"] == sum(c["dynamic_components_j"].values())
    assert c["bias_dac_sample_j"] > 0
