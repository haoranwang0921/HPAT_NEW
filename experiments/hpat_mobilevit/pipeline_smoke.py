"""Isolated single-core pipeline; no changes to production simulator."""
import json
import math
from .run import REPO, write_json, write_csv
from .backends import PhysicalCoreCosts
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


def schedule(count, durations, pipelined=True, ready=0.0):
    """One reserved output-buffer slot per producer; credit released on consume.

    Each stage is non-pipelined (II equals service latency). The producer must
    reserve its output slot before starting. This bounds queued/in-flight
    outputs and propagates downstream backpressure conservatively.
    """
    assert count > 0 and all(d > 0 for d in durations)
    starts=[]; ends=[]; events=[]
    for j in range(count):
        s=[]; e=[]
        for stage,d in enumerate(durations):
            t=max(ready, e[stage-1] if stage else ready,
                  ends[j-1][stage] if j else ready)
            if j and stage<2:
                t=max(t,starts[j-1][stage+1])  # preceding output consumed
            if not pipelined and j:
                t=max(t,ends[j-1][2])
            s.append(t);e.append(t+d)
            events.append(dict(token=j,stage=stage,start_s=t,end_s=t+d))
        starts.append(s);ends.append(e)
    for j in range(count):
        for k in range(3):
            if k: assert starts[j][k]>=ends[j][k-1]
            if j: assert starts[j][k]>=ends[j-1][k]
            if j and k<2: assert starts[j][k]>=starts[j-1][k+1]
    return ends[-1][-1]-ready,events


def checks():
    assert schedule(1,[1,1,1])[0]==3
    assert schedule(16,[1,1,1])[0]==18
    assert schedule(16,[1,1,1],False)[0]==48
    slow,events=schedule(16,[1,1,4])
    assert slow==66
    es=[e['start_s'] for e in events if e['stage']==0]
    assert any(b-a>1 for a,b in zip(es,es[1:]))
    assert schedule(1,[1,1,1],ready=100)[1][0]['start_s']==100
    return dict(single_token_no_speedup=True,equal_stage_closed_form=True,
                slow_adc_backpressure=True,input_readiness=True,
                capacity_one_credit_constraints=True)


def main():
    out=REPO/'results/hpat_mobilevit/pipeline_smoke_v2'
    out.mkdir(parents=True,exist_ok=False)
    cfg_path=REPO/'results/hpat_mobilevit/hardware_combo_32_10_25_v1/array32x32_10ghz_sram25/config.json'
    cfg=json.loads(cfg_path.read_text());costs=PhysicalCoreCosts(cfg)
    trace=REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl'
    _,records=load_manifest_jsonl(trace)
    selected=[next(r for r in records if r['op_role']==role and r['op_type'] in {'Linear','Conv2d','MatMul'}) for role in
              ['qkv_projection','ffn_linear','pointwise_conv','attention_score','attention_value']]
    rows=[];all_events=[]
    for r in selected:
        M=r['M'];k=costs.kernel(M,min(r['K'],32),min(r['N'],16))
        durations=[k[x]/(2*M) for x in ['encode_s','compute_s','convert_s']]
        for factor in [1,4]:
            ds=durations[:];ds[2]*=factor
            for mode in ['serial','pipeline']:
                end=1e-11 # weight settled before any encoding
                for sign in ['positive','negative']:
                    elapsed,events=schedule(M,ds,mode=='pipeline',end)
                    all_events.extend(dict(op_id=r['op_id'],adc_factor=factor,mode=mode,sign=sign,**e) for e in events)
                    end+=elapsed  # conservative sign-phase drain barrier
                rows.append(dict(op_id=r['op_id'],role=r['op_role'],M=M,adc_factor=factor,
                    mode=mode,block_latency_ns=end*1e9,token_count=2*M,
                    nominal_dynamic_energy_nj=k['dynamic_energy_j']*1e9))
        assert math.isclose(next(x['block_latency_ns'] for x in rows if x['op_id']==r['op_id'] and x['mode']=='serial' and x['adc_factor']==1)*1e-9,
                            sum(k[x] for x in ['encode_s','compute_s','convert_s'])+1e-11,rel_tol=1e-9)
    write_csv(out/'blocks.csv',rows);write_json(out/'events.json',all_events)
    write_json(out/'verification.json',checks())
    write_json(out/'provenance.json',dict(trace_sha256=sha256_file(trace),config_sha256=sha256_file(cfg_path),
        script_sha256=sha256_file(__file__),scope='Single resident 32x32 core weight block; two sign phases with drain. '
        'Per-token service times derived by dividing existing aggregate costs by 2M, uncalibrated. '
        'No DRAM/SRAM contention or digital reduction; one-slot credit backpressure, unlimited final sink. '
        'Dynamic activity count unchanged; no buffer/control energy. Slow ADC factor is timing-only stress test, '
        'not a physical energy model. No E2E speedup or power claim.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
