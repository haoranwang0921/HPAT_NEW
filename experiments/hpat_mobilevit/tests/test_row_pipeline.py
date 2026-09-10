import copy
import pytest
from experiments.hpat_mobilevit.row_pipeline import RowPipeline,dispatch_dag,verify_nodes
from experiments.hpat_mobilevit.tests.test_streaming import config,FakeCosts,rec
from experiments.hpat_mobilevit.streaming import TileStreamSimulation


def test_fifo_bus_serves_ready_return_before_later_input():
    nodes={'input':dict(resource='bus',deps=[]),
           'compute':dict(resource='core',deps=['input']),
           'return':dict(resource='bus',deps=['compute']),
           'next':dict(resource='bus',deps=['return'])}
    seen=[]
    def emit(k,n,t):seen.append((k,t));return t+1
    assert dispatch_dag(nodes,0,emit)==4
    assert seen==[('input',0),('compute',1),('return',2),('next',3)]


@pytest.mark.parametrize('chunk',[1,4,64])
def test_tail_rows_and_k_dependencies_conserve_cost(chunk):
    c=config();c['model_assumptions']['pipeline_row_chunk']=chunk
    for k in ['dma_batch_fetch','electronic_io_compute_overlap','reduce_compute_overlap','accumulator_residency','activity_gated_static']:c['model_assumptions'][k]=False
    r=rec(M=7,K=33,N=17)
    sims=[cls(copy.deepcopy(c),FakeCosts(),'linear',keep_events=True) for cls in [TileStreamSimulation,RowPipeline]]
    res=[s.run([r],2) for s in sims]
    verify_nodes(sims[1].node_records)
    for frame in [0,1]:
        for purpose in ['two_sign_input_passes','partial_sum_read','partial_sum_write']:
            counts=[sum(e.get('bytes',0) for e in s.events if e['frame']==frame and e.get('purpose')==purpose) for s in sims]
            assert counts[0]==counts[1]
        for k in ['optical_compute','signed_reduce','programming','sram_noc']:
            assert res[0]['frames'][frame]['energy_breakdown_j'][k]==pytest.approx(res[1]['frames'][frame]['energy_breakdown_j'][k])


def test_insufficient_row_buffer_rejected():
    c=config();c['model_assumptions'].update(pipeline_row_chunk=16,activation_scratchpad_bytes=1)
    with pytest.raises(ValueError,match='capacity'):
        RowPipeline(c,FakeCosts(),'linear').run([rec()],1)
