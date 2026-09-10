import copy
import json
import pytest
from experiments.hpat_mobilevit.device_profiles import configure_profile,ProfileSimulation
from experiments.hpat_mobilevit.plan_sweep import P0
from experiments.hpat_mobilevit.streaming import TileStreamSimulation
from experiments.hpat_mobilevit.pipeline_operator import verify_pipeline
from experiments.hpat_mobilevit.tests.test_streaming import config,FakeCosts,rec


class Costs(FakeCosts):
    def kernel(self,M,K,N):
        return dict(M=M,**super().kernel(M,K,N))


def test_profiles_keep_electronics_and_write_energy():
    base=json.loads((P0/'config.json').read_text()); old=copy.deepcopy(base)
    for name in ['P0','P1','P2']:
        c=configure_profile(base,name);a=c['model_assumptions']
        assert a['electronic_peak_flops']==base['model_assumptions']['electronic_peak_flops']
        assert a['electronic_energy_overrides']==base['model_assumptions']['electronic_energy_overrides']
        assert a['program_energy_multiplier']*c['user_confirmed']['program_response_time_s']==pytest.approx(1e-6)
    assert base==old


@pytest.mark.parametrize('mode',['digital','linear'])
def test_causal_burst_tail_and_small_cache(mode):
    c=config(); c['model_assumptions'].update(profile_burst_bytes=512,profile_pipeline_tokens=0,sram_capacity_bytes=512)
    sim=ProfileSimulation(c,Costs(),mode,keep_events=True);sim.run([rec(K=65,N=25)],2)
    for e in sim.events:
        if e.get('purpose')=='static_weight':
            assert e['bytes']<=512
            assert e['duration_s']>=c['model_assumptions']['dma_fixed_latency_s']
    assert sim.sram.occupied<=sim.sram.capacity_tiles


def test_pipeline_preserves_dynamic_energy_and_dependencies():
    c=config();c['model_assumptions'].update(profile_burst_bytes=128,profile_pipeline_tokens=3)
    sims=[cls(copy.deepcopy(c),Costs(),'linear',keep_events=True) for cls in [TileStreamSimulation,ProfileSimulation]]
    results=[s.run([rec(M=7)],2) for s in sims]
    verify_pipeline(sims[1])
    for a,b in zip(results[0]['frames'],results[1]['frames']):
        for k in ['optical_compute','programming','signed_reduce','sram_noc','hbm']:
            assert a['energy_breakdown_j'][k]==pytest.approx(b['energy_breakdown_j'][k])
