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


class StageCosts(FakeCosts):
    """Kernel fake that also reports M, as the real PhysicalCoreCosts does."""
    def kernel(self, M, K, N):
        return dict(M=M, **super().kernel(M, K, N))


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


def _hbm_events(sim):
    return [e for e in sim.events if e["event_type"] == "hbm"]


def test_dma_batch_default_off_matches_legacy_per_tile_fixed():
    """dma_batch_fetch defaults off: every weight tile pays one fixed latency."""
    cfg = config()
    sim = TileStreamSimulation(cfg, FakeCosts(), "linear", keep_events=True)
    sim.run([rec(M=4, K=33, N=17)], 1)  # 9 weight tiles
    fixed = cfg["model_assumptions"]["dma_fixed_latency_s"]
    weight_fetches = [e for e in _hbm_events(sim) if e["purpose"] == "static_weight"]
    assert len(weight_fetches) == 9
    assert not any(e["purpose"] == "dma_burst_overhead" for e in _hbm_events(sim))
    # each per-tile fetch bills the full fixed latency
    flow = 128/cfg["model_assumptions"]["hbm_bandwidth_bytes_per_s"]
    for e in weight_fetches:
        assert abs(e["duration_s"] - (flow + fixed)) < 1e-15


def test_dma_batch_amortizes_not_deletes_fixed_overhead():
    """Batch mode keeps one fixed overhead per burst; it must NOT be zero."""
    base_cfg = config()
    fixed = base_cfg["model_assumptions"]["dma_fixed_latency_s"]
    cfg = copy.deepcopy(base_cfg)
    cfg["model_assumptions"]["dma_batch_fetch"] = True
    cfg["model_assumptions"]["dma_burst_bytes"] = 1024.0  # 8 tiles per burst
    sim = TileStreamSimulation(cfg, FakeCosts(), "linear", keep_events=True)
    sim.run([rec(M=4, K=33, N=17)], 1)  # 9 tiles -> 2 bursts (ceil(9/8))

    overheads = [e for e in _hbm_events(sim) if e["purpose"] == "dma_burst_overhead"]
    assert overheads, "batch mode must retain a per-burst fixed-overhead event"
    assert abs(overheads[0]["duration_s"] - 2*fixed) < 1e-15, \
        "fixed overhead must be billed once per burst (2 bursts for 9 tiles)"
    weight_fetches = [e for e in _hbm_events(sim) if e["purpose"] == "static_weight"]
    assert all(abs(e["duration_s"] - 128/base_cfg["model_assumptions"]["hbm_bandwidth_bytes_per_s"])
               < 1e-15 for e in weight_fetches), \
        "per-tile events carry flow time only; fixed cost is deferred to the burst event"


SWITCHES = ["electronic_io_compute_overlap", "activity_gated_static",
            "reduce_compute_overlap", "accumulator_residency", "dma_batch_fetch",
            "photonic_stage_pipeline"]


def test_switches_default_off_reproduce_legacy_ledger():
    """Explicitly forcing every batch-2/3 switch off reproduces the fully
    default (legacy) ledger bit-for-bit."""
    cfg_off = copy.deepcopy(config())
    for key in SWITCHES:
        cfg_off["model_assumptions"][key] = False
    a = TileStreamSimulation(config(), FakeCosts(), "linear").run([rec()], 2)["frames"][1]
    b = TileStreamSimulation(cfg_off, FakeCosts(), "linear").run([rec()], 2)["frames"][1]
    assert a["latency_s"] == b["latency_s"]
    assert a["energy_j"] == b["energy_j"]
    assert a["energy_breakdown_j"] == b["energy_breakdown_j"]


def test_digital_mode_photonic_switches_are_noops():
    """In digital mode, photonic-path switches must not alter the ledger:
    gating has no laser to gate, M1/M2 touch the photonic path only."""
    rec9 = rec(M=8, K=64, N=16)
    base = TileStreamSimulation(config(), FakeCosts(), "digital").run([rec9], 1)["frames"][0]
    for key in ["activity_gated_static", "reduce_compute_overlap", "accumulator_residency"]:
        cfg = copy.deepcopy(config())
        cfg["model_assumptions"][key] = True
        f = TileStreamSimulation(cfg, FakeCosts(), "digital").run([rec9], 1)["frames"][0]
        assert f["latency_s"] == base["latency_s"] and f["energy_j"] == base["energy_j"], key


def test_io_overlap_is_latency_lower_bound():
    """io_overlap is an idealized lower bound: latency must not increase."""
    rec9 = rec(M=8, K=64, N=16)
    cfg = copy.deepcopy(config())
    cfg["model_assumptions"]["electronic_io_compute_overlap"] = True
    base = TileStreamSimulation(config(), FakeCosts(), "digital").run([rec9], 1)["frames"][0]
    ov = TileStreamSimulation(cfg, FakeCosts(), "digital").run([rec9], 1)["frames"][0]
    assert ov["latency_s"] <= base["latency_s"]


def test_gated_static_cuts_laser_on_hybrid_only():
    """activity_gated_static shrinks laser_static in hybrid modes."""
    cfg = copy.deepcopy(config())
    cfg["model_assumptions"]["activity_gated_static"] = True
    base = TileStreamSimulation(config(), FakeCosts(), "linear").run([rec()], 1)["frames"][0]
    gated = TileStreamSimulation(cfg, FakeCosts(), "linear").run([rec()], 1)["frames"][0]
    assert gated["energy_breakdown_j"]["laser_static"] < base["energy_breakdown_j"]["laser_static"]


def test_dma_batch_is_behavioural_change():
    """dma_batch_fetch enabled must actually change HBM timing (shorter)."""
    rec9 = rec(M=4, K=33, N=17)
    base = TileStreamSimulation(config(), FakeCosts(), "linear", keep_events=True)
    base.run([rec9], 1)
    cfg = copy.deepcopy(config())
    cfg["model_assumptions"]["dma_batch_fetch"] = True
    bat = TileStreamSimulation(cfg, FakeCosts(), "linear", keep_events=True)
    bat.run([rec9], 1)

    def hbm_bus_end(sim):
        return max(e["end_s"] for e in _hbm_events(sim))

    assert hbm_bus_end(bat) < hbm_bus_end(base), \
        "amortizing fixed overhead must shorten the HBM channel critical path"


def test_stage_pipeline_default_off_is_serial_sum():
    """With the switch off (default) the three stages stay chained on one core
    timeline: latency equals the serialized sum, unchanged from legacy."""
    a = TileStreamSimulation(config(), StageCosts(), "linear").run([rec(M=8, K=64, N=16)], 1)["frames"][0]
    cfg = copy.deepcopy(config())
    cfg["model_assumptions"]["photonic_stage_pipeline"] = False
    b = TileStreamSimulation(cfg, StageCosts(), "linear").run([rec(M=8, K=64, N=16)], 1)["frames"][0]
    assert a["latency_s"] == b["latency_s"]
    assert a["energy_breakdown_j"] == b["energy_breakdown_j"]


def test_stage_pipeline_overlaps_but_conserves_dynamic_energy():
    """Enabling stage overlap must shorten latency without changing any dynamic
    energy component; only duration-proportional static terms may shrink."""
    records = [rec(M=8, K=64, N=16)]
    base = TileStreamSimulation(config(), StageCosts(), "linear").run(records, 1)["frames"][0]
    cfg = copy.deepcopy(config())
    cfg["model_assumptions"]["photonic_stage_pipeline"] = True
    pipe = TileStreamSimulation(cfg, StageCosts(), "linear").run(records, 1)["frames"][0]

    assert pipe["latency_s"] < base["latency_s"]
    # Three equal stages overlap to the bottleneck stage; the per-invocation
    # floor is one stage's total work, so latency cannot fall below 1/3.
    assert pipe["latency_s"] > base["latency_s"]/3
    dynamic = ["hbm", "sram_noc", "programming", "dac_encode", "optical_compute",
               "adc_convert", "signed_reduce"]
    for key in dynamic:
        assert pipe["energy_breakdown_j"][key] == pytest.approx(
            base["energy_breakdown_j"][key], rel=1e-12), key
    for key in ["electronic_static", "laser_static", "mrr_hold"]:
        assert pipe["energy_breakdown_j"][key] <= base["energy_breakdown_j"][key]


def test_stage_pipeline_is_noop_in_digital_mode():
    """D0 has no photonic stage chain, so the switch must not touch it."""
    records = [rec(M=8, K=64, N=16)]
    base = TileStreamSimulation(config(), FakeCosts(), "digital").run(records, 1)["frames"][0]
    cfg = copy.deepcopy(config())
    cfg["model_assumptions"]["photonic_stage_pipeline"] = True
    f = TileStreamSimulation(cfg, FakeCosts(), "digital").run(records, 1)["frames"][0]
    assert f["latency_s"] == base["latency_s"]
    assert f["energy_breakdown_j"] == base["energy_breakdown_j"]
