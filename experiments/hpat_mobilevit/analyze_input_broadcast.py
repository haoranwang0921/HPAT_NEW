"""Exact input sharing counts and optimistic traffic-only bounds, not implementation."""
import json
import math
from collections import defaultdict
from .run import REPO, ROOT, write_json, write_csv, run_case
from .streaming import eligible, weight_blocks
from .backends import PhysicalCoreCosts
from .mobile_profile import profile_costs
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


def analyze(r, pk, ln, cores, bits):
    blocks=list(weight_blocks(r,pk,ln))
    per=lambda b: 2*r['M']*b.K*bits//8
    original=sum(map(per,blocks))
    current=0; duplicates=0
    for i in range(0,len(blocks),cores):
        seen=set()
        for b in blocks[i:i+cores]:
            key=(b.batch_index,b.k_index)
            if key in seen: duplicates+=1
            else: current+=per(b);seen.add(key)
    # Reorder: each wave serves up to four N blocks with the same batch/K.
    groups=defaultdict(list)
    for b in blocks: groups[b.batch_index,b.k_index].append(b)
    reordered=sum(math.ceil(len(bs)/cores)*per(bs[0]) for bs in groups.values())
    temporal=sum(per(bs[0]) for bs in groups.values())
    assert 0 <= temporal <= reordered <= original and current <= original
    return dict(op_id=r['op_id'],module_path=r['module_path'],op_type=r['op_type'],
        role=r['op_role'],M=r['M'],K=r['K'],N=r['N'],batch=r['batch_repetitions'],
        n_blocks=math.ceil(r['N']/ln),k_blocks=math.ceil(r['K']/pk),
        original_bytes=original,current_wave_saved_bytes=original-current,
        reordered_fourway_saved_bytes=original-reordered,
        all_n_temporal_saved_bytes=original-temporal,current_duplicate_reads=duplicates,
        max_input_buffer_bytes=2*r['M']*min(r['K'],pk)*bits//8,
        potential_fourway=r['N']>ln)


def main():
    out=REPO/'results/hpat_mobilevit/input_broadcast_analysis_v1'
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((REPO/'results/hpat_mobilevit/hardware_combo_32_10_25_v1/array32x32_10ghz_sram25/config.json').read_text())
    costs=profile_costs(PhysicalCoreCosts(cfg),cfg)
    a=cfg['model_assumptions'];u=cfg['user_confirmed']
    all_rows=[]; totals=[]; estimates=[]
    for variant in ['xxs','xs','s']:
        path=REPO/f'results/hpat_mobilevit/trace_v1/{variant}/operator_trace.jsonl'
        manifest,records=load_manifest_jsonl(path)
        for mode in ['linear','linear_pointwise_attention']:
            rows=[dict(variant=variant,mode=mode,**analyze(r,u['physical_array'][0],u['physical_array'][1]//2,4,a['bits'])) for r in records if eligible(r,mode)]
            all_rows+=rows
            for role in sorted(set(r['role'] for r in rows)):
                group=[r for r in rows if r['role']==role]
                totals.append(dict(variant=variant,mode=mode,role=role,ops=len(group),
                    shareable_ops=sum(r['potential_fourway'] for r in group),
                    **{k:sum(r[k] for r in group) for k in ['original_bytes','current_wave_saved_bytes','reordered_fourway_saved_bytes','all_n_temporal_saved_bytes']}))
            if variant=='xxs':
                base=run_case(cfg,costs,mode,manifest,records,out/mode,mode)[-1]
                for scenario,key in [('current_wave','current_wave_saved_bytes'),('reordered_fourway','reordered_fourway_saved_bytes')]:
                    saved=sum(r[key] for r in rows)
                    dt=saved/a['sram_bandwidth_bytes_per_s']
                    de=saved*(costs.electronic_energy.sram_energy_per_byte_j+costs.electronic_energy.noc_energy_per_byte_j)
                    static=sum(costs.static_power.values())+costs.electronic_energy.static_power_w
                    estimates.append(dict(mode=mode,scenario=scenario,base_ms=base['latency_ms'],base_mj=base['energy_mj'],
                        saved_bytes=saved,max_service_time_saved_us=dt*1e6,
                        optimistic_latency_ms=base['latency_ms']-dt*1e3,
                        optimistic_latency_reduction_percent=dt*1e3/base['latency_ms']*100,
                        idealized_ledger_saving_mj=de*1e3,
                        optimistic_energy_mj=base['energy_mj']-(de+static*dt)*1e3,
                        optimistic_energy_reduction_percent=(de+static*dt)*1e3/base['energy_mj']*100))
        print(variant,'done',flush=True)
    write_csv(out/'operators.csv',all_rows);write_csv(out/'role_totals.csv',totals)
    write_csv(out/'xxs_bounds.csv',estimates)
    write_json(out/'scope.json',dict(config=cfg,script_sha256=sha256_file(__file__),
        scope='Exact logical operand matches only within each op and batch, no cross-op sharing. '
        '4-way assumes inter-tile multicast; current model has no tile topology. '
        'Bounds delete saved source-read SRAM+NoC service and energy at zero distribution/buffer cost; '
        'all deleted time assumed critical. Reordering effects on partial sums/writes/parallelism NOT simulated. '
        'DAC/EOM costs unchanged. Temporal counts assume retaining input across all N groups, not 4-way broadcast. '
        'No changes to default simulator. HPAT residual profile remains original, not updated D0 profile.'))
    print(json.dumps(estimates,indent=2))


if __name__=='__main__':main()
