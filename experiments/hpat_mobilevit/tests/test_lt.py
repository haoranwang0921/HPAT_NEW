import json
from pathlib import Path
import pytest
from experiments.hpat_mobilevit.lt_backend import LTCoreCosts
from experiments.hpat_mobilevit.lt_streaming import LTStreamSimulation
from experiments.hpat_mobilevit.tests.test_streaming import config, rec


@pytest.fixture(scope="module")
def costs():
    return LTCoreCosts(config())


def test_lt_topology_and_full_range(costs):
    core = costs.resolved_single_core["core"]
    assert core["range"] == dict(input="full", weight="full", output="full")
    assert core["width"] == core["height"] == core["num_wavelength"] == 16
    assert costs.cores == 4 and costs.architecture["mrr_count"] == 0
    assert costs.programming["programming_energy_j"] == 0
    assert costs.static_power["laser_static"] == costs.architecture["laser_wall_plug_power_w"]
    assert core["netlist"]["temporal_accum_factor"]["duration"] == 1


def test_lt_kernel_and_bounds(costs):
    r = costs.kernel(16,16,16)
    assert r["switching_cycles"] == r["input_sign_passes"] == 1
    assert r["dynamic_energy_j"] == sum(r["dynamic_components_j"].values())
    assert r["encode_s"] > 0 and r["convert_s"] > 0
    with pytest.raises(ValueError): costs.kernel(17,16,16)


@pytest.mark.parametrize("dynamic", [False, True])
def test_lt_tail_macs_no_programming(costs, dynamic):
    r = rec(M=19,K=33,N=17)
    if dynamic:
        r.update(op_type="MatMul",op_role="attention_score",weight_static=False,weight_id=None,batch_repetitions=2,macs=19*33*17*2)
    s=LTStreamSimulation(config(),costs,"linear_pointwise_attention",keep_events=True)
    result=s.run([r],2)
    assert not any(e["event_type"]=="programming" for e in s.events)
    for f in range(2):
        assert sum(e.get("macs",0) for e in s.events if e["frame"]==f)==r["macs"]
    assert all(result["checks"].values())
