"""Versioned complete-QKV comparison with row-chunk scheduling."""
import copy
import gzip
import json
import math
from collections import defaultdict
from .row_pipeline import RowPipeline,verify_nodes
from .pipeline_operator import PipelineOperator,verify_pipeline
from .streaming import TileStreamSimulation
from .backends import PhysicalCoreCosts
from .mobile_profile import profile_costs
from .run import REPO,write_json,write_csv
from joint_sim.trace_io import load_manifest_jsonl,sha256_file


def main():
    out=REPO/'results/hpat_mobilevit/row_pipeline_qkv_v2'
    out.mkdir(parents=True,exist_ok=False)
    cfgpath=REPO/'results/hpat_mobilevit/hardware_combo_32_10_25_v1/array32x32_10ghz_sram25/config.json'
    cfg=json.loads(cfgpath.read_text());costs=profile_costs(PhysicalCoreCosts(cfg),cfg)
    trace=REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl'
    _,records=load_manifest_jsonl(trace)
    r=copy.deepcopy(next(x for x in records if x['op_type']=='Linear' and x['op_role']=='qkv_projection'))
    r['dependencies']=[]
    rows=[];checks=[];base=None
    variants=[('serial',TileStreamSimulation,576),('block_pipeline',PipelineOperator,576)]
    variants += [(f'rows{n}',RowPipeline,n) for n in [1,4,16,64,576]]
    variants += [('rows16_no_core_overlap',RowPipeline,16),('rows576_no_core_overlap',RowPipeline,576)]
    for name,cls,chunk in variants:
        c=copy.deepcopy(cfg);c['model_assumptions']['pipeline_row_chunk']=chunk
        c['model_assumptions']['row_core_pipeline']='no_core_overlap' not in name
        sim=cls(c,costs,'linear',keep_events=True);res=sim.run([r],2)
        traffic=defaultdict(int)
        for e in sim.events:traffic[e['frame'],e.get('purpose','')]+=e.get('bytes',0)
        if base is None:base=(traffic,res)
        else:
            assert traffic==base[0],name
            for f,b in zip(res['frames'],base[1]['frames']):
                for k,v in b['energy_breakdown_j'].items():
                    if k not in {'electronic_static',*costs.static_power}:
                        assert math.isclose(f['energy_breakdown_j'][k],v,rel_tol=1e-8,abs_tol=1e-15),(name,k)
        check=dict(mode=name,traffic_equal=True,dynamic_energy_equal=True)
        if name!='serial':check.update(verify_pipeline(sim))
        if isinstance(sim,RowPipeline):
            check['verified_dag_nodes']=verify_nodes(sim.node_records)
            write_json(out/f'{name}_nodes.json',sim.node_records)
        checks.append(check)
        write_json(out/f'{name}_summary.json',res)
        write_json(out/f'{name}_config.json',c)
        with gzip.open(out/f'{name}_events.jsonl.gz','wt',encoding='utf8') as f:
            for e in sim.events:f.write(json.dumps(e)+'\n')
        for f in res['frames']:
            rows.append(dict(mode=name,state=f['state'],latency_us=f['latency_s']*1e6,
                energy_uj=f['energy_j']*1e6,buffer_bytes=getattr(sim,'pipeline_buffer_bytes',0),
                programming_events=f['event_counts']['programming']))
        print(name,rows[-1],flush=True)
    write_csv(out/'comparison.csv',rows);write_json(out/'verification.json',checks)
    write_json(out/'provenance.json',dict(config=cfg,operator=r,trace_sha256=sha256_file(trace),
        sources={p:sha256_file(REPO/p) for p in ['experiments/hpat_mobilevit/row_pipeline.py',
            'experiments/hpat_mobilevit/run_row_pipeline.py','experiments/hpat_mobilevit/streaming.py',
            'experiments/hpat_mobilevit/pipeline_smoke.py']},
        scope='Full isolated QKV; row-buffered input/read/reduce/write DAG, FIFO shared bus, '
        'one in-flight chunk/core until output written, conservative wave barriers. '
        'Same bytes/dynamic compute and memory energy; new arbitration/buffer control costs uncalibrated. '
        'Bias and requantization omissions inherited; no end-to-end claim.'))


if __name__=='__main__':main()
