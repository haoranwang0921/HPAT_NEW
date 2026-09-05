import hashlib
import json

import pytest

from experiments.hpat_mobilevit.figures.gen_fig_evaluation import read_rows, unique


def inputs(tmp_path, rows):
    data = "run_id,frame,latency_ms,energy_mj,sram_hit_rate\n" + rows
    (tmp_path/"aggregate.csv").write_text(data)
    (tmp_path/"verification.json").write_text(json.dumps({"valid": True}))
    digest = hashlib.sha256((tmp_path/"aggregate.csv").read_bytes()).hexdigest()
    (tmp_path/"completion.json").write_text(json.dumps({"aggregate_sha256": digest}))


def test_plot_rejects_duplicate_measurement(tmp_path):
    inputs(tmp_path, "r,0,1,2,0\nr,0,1,2,0\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_rows(tmp_path)


def test_plot_rejects_changed_data(tmp_path):
    inputs(tmp_path, "r,0,1,2,0\n")
    (tmp_path/"aggregate.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        read_rows(tmp_path)


def test_plot_rejects_incomplete_comparison():
    with pytest.raises(ValueError, match="Expected one row"):
        unique([], model="MobileViT-S", mode="digital")
