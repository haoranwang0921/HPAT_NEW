import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.hpat_mobilevit.streaming import TileStreamSimulation, weight_blocks, eligible, vector_work

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/"config.json").read_text())


class FakeEnergy:
    def matmul_energy(self, n): return n*0.4e-12
    def vector_energy(self, n): return n*0.9e-12
    def hbm_read_energy(self, n): return n*300e-12
    def sram_read_energy(self, n): return n*3.7e-12
    def noc_energy(self, n): return n*2e-12
    def static_energy(self, t): return t*15


class FakeCosts:
    electronic_energy = FakeEnergy()
    programming = {"programming_energy_j": 93e-12*1024, "programmed_mrr_count": 1024}
    static_power = {"laser_static": 0.9, "mrr_hold": 0.1024}
    def kernel(self, M, K, N):
        return dict(encode_s=M*0.4e-9, compute_s=M*0.4e-9, convert_s=M*0.4e-9,
                    dynamic_energy_j=M*16*1e-12, bias_dac_sample_j=3e-12)


def rec(M=4, K=33, N=17, **changes):
    result = dict(op_id="op", order=0, op_type="Linear", op_role="ffn_linear", aten_op="aten.addmm.default",
                  module_path="fc", weight_static=True, weight_id="weight", M=M, K=K, N=N,
                  input_bits=8, output_bits=8, weight_bits=8, batch_repetitions=1,
                  macs=M*K*N, dependencies=[], output_elements=M*N)
    result.update(changes)
    return result


def test_tail_blocks_cover_matrix_once():
    blocks = list(weight_blocks(rec(), 16, 8))
    assert len(blocks) == 9
    assert sum(b.K*b.N for b in blocks) == 33*17
    assert max(b.K for b in blocks) <= 16
    assert max(b.N for b in blocks)*2 <= 16


def test_signed_four_product_reconstruction():
    rng = np.random.default_rng(9)
    x, w = rng.normal(size=(7, 33)), rng.normal(size=(33, 17))
    result = np.zeros((7, 17))
    for b in weight_blocks(rec(7), 16, 8):
        k, n = b.k_index*16, b.n_index*8
        xx, ww = x[:, k:k+b.K], w[k:k+b.K, n:n+b.N]
        xp, xn, wp, wn = np.maximum(xx, 0), np.maximum(-xx, 0), np.maximum(ww, 0), np.maximum(-ww, 0)
        result[:, n:n+b.N] += (xp@wp-xp@wn)-(xn@wp-xn@wn)
    np.testing.assert_allclose(result, x@w, rtol=1e-12, atol=1e-12)


def test_streamed_weights_are_not_overwritten_before_use():
    sim = TileStreamSimulation(config(), FakeCosts(), "linear", keep_events=True)
    result = sim.run([rec()], 2)
    previous_end = {}
    state = {}
    for event in sim.events:
        if event["event_type"] == "programming":
            core = event["core"]
            assert event["start_s"] >= previous_end.get(core, 0)
            state[core] = event["weight_block"]
        if event["event_type"] == "optical_compute":
            assert state[event["core"]] == event["weight_block"]
        if event["event_type"] == "adc_convert":
            previous_end[event["core"]] = event["end_s"]
    assert result["frames"][0]["event_counts"]["programming"] == 9
    assert result["frames"][1]["sram_hit_rate"] == 1.0
    assert all(result["checks"].values())


def test_program_parallelism_changes_latency():
    cfg = config()
    four = TileStreamSimulation(cfg, FakeCosts(), "linear").run([rec()], 1)
    cfg["model_assumptions"]["program_parallelism"] = 1
    one = TileStreamSimulation(cfg, FakeCosts(), "linear").run([rec()], 1)
    assert one["frames"][0]["latency_s"] > four["frames"][0]["latency_s"]


def test_dynamic_context_is_fresh_each_image_and_batch():
    r = rec(op_type="MatMul", op_role="attention_score", weight_static=False, weight_id=None, batch_repetitions=2)
    blocks0 = {b.key for b in weight_blocks(r, 16, 8, 0)}
    blocks1 = {b.key for b in weight_blocks(r, 16, 8, 1)}
    assert len(blocks0) == 18
    assert blocks0.isdisjoint(blocks1)
    assert not eligible(r, "linear_pointwise")
    assert eligible(r, "linear_pointwise_attention")


def test_unsupported_operator_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        vector_work(dict(op_type="unknown", aten_op="aten.unknown", module_path="unknown"))


def test_energy_ledger_closes_and_digital_has_no_photonic_static():
    for mode in ["digital", "linear"]:
        result = TileStreamSimulation(config(), FakeCosts(), mode).run([rec()], 1)
        frame = result["frames"][0]
        assert frame["energy_j"] == sum(frame["energy_breakdown_j"].values())
        assert ("laser_static" in frame["energy_breakdown_j"]) == (mode != "digital")


def test_scratch_capacity_not_silently_expanded():
    cfg = config()
    cfg["model_assumptions"]["activation_scratchpad_bytes"] = 1
    with pytest.raises(ValueError, match="scratchpad bytes"):
        TileStreamSimulation(cfg, FakeCosts(), "linear").run([rec()], 1)
