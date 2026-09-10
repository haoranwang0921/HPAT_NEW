import json
import pytest
from experiments.hpat_mobilevit.plan_sweep import configure, P0, union

def test_fixed_write_energy_and_independent_lanes():
    base=json.loads((P0/'config.json').read_text())
    for ps in [10,100,1000,10000]:
        c=configure(base,{'program_ps':ps})
        assert c['user_confirmed']['program_response_time_s']*c['model_assumptions']['program_energy_multiplier']==pytest.approx(1e-6)
    c=configure(base,{'organization':[4,2]})
    assert c['model_assumptions']['program_parallelism']==4
    assert base['user_confirmed']['tiles']==2

def test_union_is_not_service_sum():
    assert union([(0,2),(1,3),(4,5)])==4
    assert union([])==0

def test_reject_unknown_sweep():
    with pytest.raises(ValueError): configure(json.loads((P0/'config.json').read_text()),{'made_up':1})
