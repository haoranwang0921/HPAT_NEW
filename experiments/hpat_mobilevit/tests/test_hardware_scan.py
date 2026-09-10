from experiments.hpat_mobilevit.tests.test_streaming import config, FakeCosts, rec
from experiments.hpat_mobilevit.streaming import TileStreamSimulation


def test_fixed_cache_decouples_digital_from_optical_shape():
    frames=[]
    for shape in ([16,16],[16,32],[32,16],[32,32]):
        c=config();c['user_confirmed']['physical_array']=shape
        c['model_assumptions']['weight_cache_block_shape']=[16,8]
        sim=TileStreamSimulation(c,FakeCosts(),'digital')
        frames.append(sim.run([rec(M=16,K=64,N=64)],2)['frames'])
    assert all(f==frames[0] for f in frames)


def test_larger_photonic_blocks_fetch_same_cold_weight_bytes():
    counts=[]
    for shape in ([16,16],[32,32]):
        c=config();c['user_confirmed']['physical_array']=shape
        c['model_assumptions']['weight_cache_block_shape']=[16,8]
        s=TileStreamSimulation(c,FakeCosts(),'linear',keep_events=True)
        s.run([rec(M=16,K=64,N=64)],1)
        counts.append(sum(e.get('bytes',0) for e in s.events if e.get('purpose')=='static_weight'))
    assert counts==[4096,4096]
