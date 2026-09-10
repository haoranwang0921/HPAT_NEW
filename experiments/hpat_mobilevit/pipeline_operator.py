"""Conservative complete-operator experiment: block-buffered supply and reduction."""
import copy
import gzip
import json
import math
from collections import defaultdict
from .streaming import TileStreamSimulation, io_sizes
from .pipeline_smoke import schedule, checks
from .backends import PhysicalCoreCosts
from .energy_ledger import stage_energies
from .mobile_profile import profile_costs
from .run import REPO, run_case, write_json, write_csv
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


class PipelineOperator(TileStreamSimulation):
    def photonic_compute(self,r,ready):
        if any([self.io_overlap,self.reduce_overlap,self.dma_batch,self.accum_resident,self.gated_static]):
            raise ValueError('This controlled experiment requires original conservative switches')
        # Reserve finite per-core block input and digitized-output buffers until
        # the whole wave is reduced. Count them in the existing scratch capacity.
        ib,ob=io_sizes(r)
        acc=r['M']*min(r['N']*r['batch_repetitions'],self.ln*self.cores)*self.a['accumulator_bits']//8
        extra=self.cores*r['M']*(2*self.pk+4*self.ln)*self.a['bits']//8
        required=ib+ob+acc+extra
        if required>self.a['activation_scratchpad_bytes']:
            raise ValueError('Pipeline block buffers exceed explicit scratch capacity')
        self.scratch_peak=max(self.scratch_peak,required)
        self.pipeline_buffer_bytes=max(getattr(self,'pipeline_buffer_bytes',0),extra)
        return super().photonic_compute(r,ready)

    def execute_core_block(self,op,begin,kernel,core,block):
        M=kernel['M'];current=begin
        ds=[kernel[k]/(2*M) for k in ['encode_s','compute_s','convert_s']]
        for sign in ['positive','negative']:
            elapsed,events=schedule(M,ds,ready=current)
            for e in events:
                kind=['dac_encode','optical_compute','adc_convert'][e['stage']]
                self.event(kind,op,e['start_s'],e['end_s']-e['start_s'],
                    f'core_{core}_stage_{e["stage"]}',
                    stage_energies(kernel)[kind]/(2*M),
                    core=core,weight_block=block.key,sign=sign,token=e['token'],
                    input_ready_s=begin,sign_ready_s=current)
            current+=elapsed
        return current


def verify_pipeline(sim):
    previous={};last=defaultdict(float);tokens={};reductions=defaultdict(float)
    for e in sim.events:
        c=e.get('core');k=e['event_type'];f=e['frame']
        if k=='programming':
            assert e['start_s']+1e-14>=last[c]
            previous[c]=e['weight_block'];last[c]=e['end_s']
        if k in ['dac_encode','optical_compute','adc_convert']:
            assert previous[c]==e['weight_block']
            assert e['start_s']+1e-14>=e['input_ready_s']
            key=(f,e['op_id'],c,e['weight_block'],e['sign'],e['token'])
            stage=['dac_encode','optical_compute','adc_convert'].index(k)
            if stage: assert tokens[key][0]==stage-1 and e['start_s']+1e-14>=tokens[key][1]
            else: assert key not in tokens
            tokens[key]=(stage,e['end_s'])
            last[c]=max(last[c],e['end_s'])
        if k=='signed_reduce': assert e['start_s']+1e-14>=last[c]
        if e.get('purpose') in ['partial_sum_read','partial_sum_write']:
            key=(f,e['op_id'],e['accumulator'])
            if e['purpose']=='partial_sum_read': assert e['start_s']+1e-14>=reductions[key]
            else: reductions[key]=e['end_s']
    assert all(s==2 for s,_ in tokens.values())
    return dict(tokens=len(tokens),stage_chains=True,input_readiness=True,weight_exclusion=True,
                reduction_after_conversion=True,partial_sum_dependencies=True)


def main():
    out=REPO/'results/hpat_mobilevit/pipeline_full_operator_v2'
    out.mkdir(parents=True,exist_ok=False)
    config_path=REPO/'results/hpat_mobilevit/hardware_combo_32_10_25_v1/array32x32_10ghz_sram25/config.json'
    cfg=json.loads(config_path.read_text())
    costs=profile_costs(PhysicalCoreCosts(cfg),cfg)
    trace=REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl'
    _,records=load_manifest_jsonl(trace)
    r=copy.deepcopy(next(x for x in records if x['op_type']=='Linear' and x['op_role']=='qkv_projection'))
    r['dependencies']=[] # isolated operator; input supplied by explicit DRAM event
    sims={};results={}
    for name,cls in [('serial',TileStreamSimulation),('pipeline',PipelineOperator)]:
        sim=cls(cfg,costs,'linear',keep_events=True)
        result=sim.run([r],2)
        sims[name]=sim;results[name]=result
        write_json(out/f'{name}_summary.json',result)
        with gzip.open(out/f'{name}_events.jsonl.gz','wt',encoding='utf8') as f:
            for e in sim.events:f.write(json.dumps(e)+'\n')
    check=verify_pipeline(sims['pipeline'])
    for frame in [0,1]:
        for purpose in ['static_weight','weight_cache_fill','weight_cache_read','two_sign_input_passes',
                        'partial_sum_read','partial_sum_write','photonic_input','photonic_output']:
            sums=[sum(e.get('bytes',0) for e in s.events if e['frame']==frame and e.get('purpose')==purpose) for s in sims.values()]
            assert sums[0]==sums[1],(purpose,sums)
        b,p=(results[x]['frames'][frame] for x in ['serial','pipeline'])
        for k,v in b['energy_breakdown_j'].items():
            if k not in {'electronic_static',*costs.static_power}:
                assert math.isclose(v,p['energy_breakdown_j'][k],rel_tol=1e-9,abs_tol=1e-15),(k,v)
    rows=[dict(mode=name,state=f['state'],latency_us=f['latency_s']*1e6,energy_uj=f['energy_j']*1e6,
               programming_events=f['event_counts']['programming']) for name,res in results.items() for f in res['frames']]
    write_csv(out/'comparison.csv',rows)
    write_json(out/'verification.json',dict(checks() | check,traffic_equal=True,dynamic_energy_equal=True))
    write_json(out/'provenance.json',dict(config=cfg,operator=r,trace_sha256=sha256_file(trace),
        source_sha256={p:sha256_file(REPO/p) for p in ['experiments/hpat_mobilevit/streaming.py',
            'experiments/hpat_mobilevit/pipeline_operator.py','experiments/hpat_mobilevit/pipeline_smoke.py']},
        scope='Full isolated QKV GEMM, all weight blocks and four cores; unchanged block supply/reduction. '
        'Finite whole-block buffers reserved in scratch; no per-row supply/reduce overlap. '
        'Additional buffer access/control energy not calibrated. Bias/requantization omissions inherited. '
        'No end-to-end model speedup claim.'))
    print(json.dumps(rows,indent=2));print('buffers',sims['pipeline'].pipeline_buffer_bytes)


if __name__=='__main__':main()
