"""Unit and accounting regressions for the user-confirmed 10 ps scenario."""
import json
import math
from pathlib import Path

import pytest

from experiments.hpat_mobilevit.streaming import TileStreamSimulation
from experiments.hpat_mobilevit.tests.test_streaming import FakeCosts, config, rec


def fast_config():
    path = Path(__file__).resolve().parents[1]/"config_10ps_fixed_write_energy.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_10ps_units_and_compensating_energy_factor():
    cfg = fast_config()
    duration = cfg["user_confirmed"]["program_response_time_s"]
    assert duration == 10e-12
    assert math.isclose(duration/1e-6*cfg["model_assumptions"]["program_energy_multiplier"], 1.0)


def test_10ps_preserves_energy_of_each_programming_event():
    simulations = [TileStreamSimulation(cfg, FakeCosts(), "linear", keep_events=True)
                   for cfg in [config(), fast_config()]]
    results = [s.run([rec()], 2) for s in simulations]
    before, after = [[e for e in s.events if e["event_type"] == "programming"]
                     for s in simulations]
    assert len(before) == len(after) > 0
    for old, new in zip(before, after):
        assert old["weight_block"] == new["weight_block"]
        assert new["duration_s"] == 10e-12
        assert new["end_s"] > new["start_s"]
        assert new["energy_j"] == pytest.approx(old["energy_j"], rel=1e-12, abs=1e-20)
    assert results[1]["frames"][0]["latency_s"] < results[0]["frames"][0]["latency_s"]
    assert all(results[1]["checks"].values())


def test_10ps_does_not_change_digital_baseline():
    before, after = [TileStreamSimulation(cfg, FakeCosts(), "digital").run([rec()], 2)
                     for cfg in [config(), fast_config()]]
    assert before["frames"] == after["frames"]
