import hashlib
import json

import pytest

from experiments.hpat_mobilevit.figures.gen_fig_mobile import read_mobile_rows


def _write_aggregate(tmp_path, rows):
    data = "profile,run_id,frame,latency_ms,energy_mj,dense_int8_tops\n"+rows
    path = tmp_path/"aggregate.csv"
    path.write_text(data)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/"completion.json").write_text(json.dumps(dict(aggregate_sha256=h,frames=len(rows.splitlines()))))
    (tmp_path/"verification.json").write_text(json.dumps(dict(valid=True,aggregate_sha256=h)))


def test_mobile_plot_allows_same_run_id_in_different_profiles(tmp_path):
    _write_aggregate(tmp_path,"p1,r,0,1,2,10\np2,r,0,1,2,20\n")
    assert len(read_mobile_rows(tmp_path)) == 2


def test_mobile_plot_rejects_duplicate_profile_frame(tmp_path):
    _write_aggregate(tmp_path,"p1,r,0,1,2,10\np1,r,0,1,2,10\n")
    with pytest.raises(ValueError,match="Duplicate"):
        read_mobile_rows(tmp_path)


def test_mobile_plot_rejects_data_modified_after_verification(tmp_path):
    _write_aggregate(tmp_path,"p1,r,0,1,2,10\n")
    (tmp_path/"aggregate.csv").write_text("changed")
    with pytest.raises(ValueError,match="changed"):
        read_mobile_rows(tmp_path)
