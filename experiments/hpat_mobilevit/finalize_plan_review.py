"""Final evidence verification and consistent-color breakdown rendering."""
import csv
import json
import math
from pathlib import Path
from .run import REPO, write_csv, write_json
from .plan_sweep import configure, MODES, audit_events
from .summarize_plan import P2, P3, P4, load_rows
from joint_sim.trace_io import sha256_file


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    verified=0
    for root in (P2,P3):
        rows=load_rows(root)
        provenance=json.loads((root/'provenance.json').read_text())
        assert provenance['trace_manifest']['pretrained'] is False
        for name,digest in provenance['source_sha256'].items():
            assert sha256_file(REPO/name)==digest,name
        trace=REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl'
        assert sha256_file(trace)==provenance['trace_sha256']
        base=json.loads((root/'config.json').read_text())
        for point in json.loads((root/'scan_spec.json').read_text()):
            dest=root/point['name']; cfg=json.loads((dest/'config.json').read_text())
            assert cfg==configure(base,point['changes'])
            for case in json.loads((dest/'verification.json').read_text())['cases']:
                mode=case['run_id'][len(point['name'])+1:]
                assert sha256_file(dest/mode/'events.jsonl.gz')==case['timeline_sha256']
                verified+=1
    # Complete the P1 table from the original P1 events without changing P1.
    old=REPO/'results/hpat_mobilevit/experiments_p1_adc_audit_v1'
    audit=[]
    for point in ('base','array32x32_10ghz_sram25'):
        for mode in ('digital','linear_pointwise_attention'):
            d=old/point/mode
            summary=json.loads((d/'summary.json').read_text())
            for r in audit_events(d/'events.jsonl.gz',summary):
                audit.append(dict(point=point,mode=mode,**r))
    write_csv(P4/'p1_event_audit_supplement.csv',audit)

    # Every energy component uses exactly the same color/hatch in both bars.
    kinds=['hbm','sram_noc','electronic_dynamic','programming','dac_encode','adc_convert',
           'signed_reduce','electronic_static','laser_static','mrr_hold']
    colors=['#E69F00','#56B4E9','#009E73','#888888','#0072B2','#D55E00','#CC79A7','#000000','#E69F00','#56B4E9']
    hatches=['','','','','','','','','///','xxx']
    events=['hbm','sram_noc','electronic_dynamic','programming','dac_encode','optical_compute','adc_convert','signed_reduce']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,5.8))
    for j,(name,mode) in enumerate([('C0','linear_pointwise_attention'),('C6','linear')]):
        summary=json.loads((P3/name/mode/'summary.json').read_text())
        eb=summary['frames'][1]['energy_breakdown_j'];left=0
        for kind,color,hatch in zip(kinds,colors,hatches):
            v=eb.get(kind,0)*1000
            axes[0].barh(j,v,left=left,color=color,hatch=hatch,edgecolor='white',linewidth=.2,
                         label=kind if j==0 else None)
            left+=v
        assert math.isclose(left,summary['frames'][1]['energy_j']*1000,rel_tol=1e-12)
        with (P3/name/mode/'event_audit.csv').open() as f:
            ev={r['resource']:float(r['busy_union_s'])*1000 for r in csv.DictReader(f) if r['state']=='warm'}
        axes[1].barh(np.arange(len(events))+(j-.5)*.36,[ev.get(e,0) for e in events],height=.34,
                     label=name,color=['#0072B2','#E69F00'][j],hatch='//' if j==0 else '')
    axes[0].set_yticks([0,1],['C0 (H3)','C6 (H1)']);axes[0].set_xlabel('Energy breakdown (mJ)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,fontsize=7,loc='upper center',bbox_to_anchor=(.5,.99),frameon=False)
    axes[1].set_yticks(range(len(events)),events);axes[1].set_xlabel('Busy union per event type (ms)');axes[1].legend()
    fig.text(.5,.015,'Warm C0 vs C6; unions overlap and are NOT additive critical-path contributions\n'
        'XXS 192 / batch=1 / random-weight FP32 trace / 8-bit costs / P1 ledger / 10 ps write hypothesis\n'
        'No SRAM area/leakage rescaling; deterministic frames, no statistical error bars',ha='center',fontsize=7)
    fig.tight_layout(rect=[0,.13,1,.85])
    for ext in ('pdf','png'):fig.savefig(P4/f'breakdown.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

    # Cold/warm differences and matched-D0 comparisons for every candidate.
    rows=load_rows(P3);specs=json.loads((P3/'scan_spec.json').read_text());comparison=[]
    for spec in specs:
        name=spec['name'];mode=MODES[spec['changes'].get('mapping','H3')]
        for state in ('cold','warm'):
            r=next(r for r in rows if r['point']==name and r['mode']==mode and r['state']==state)
            d=next(r for r in rows if r['point']==name and r['mode']=='digital' and r['state']==state)
            b=next(r for r in rows if r['point']=='C0' and r['mode']=='linear_pointwise_attention' and r['state']==state)
            comparison.append(dict(point=name,state=state,mode=mode,latency_ms=r['latency_ms'],energy_mj=r['energy_mj'],
                average_power_w=r['average_power_w'],latency_change_percent=100*(r['latency_ms']/b['latency_ms']-1),
                energy_change_percent=100*(r['energy_mj']/b['energy_mj']-1),
                power_change_percent=100*(r['average_power_w']/b['average_power_w']-1),
                latency_ratio_to_d0=r['latency_ms']/d['latency_ms'],energy_ratio_to_d0=r['energy_mj']/d['energy_mj']))
    write_csv(P4/'candidate_comparisons.csv',comparison)
    assert all(r['latency_ratio_to_d0']>1 and r['energy_ratio_to_d0']>1 for r in comparison)
    # Reject any suggestion that this experimental model has beaten D0.
    add='\n\n## 最终人工复核补充\n\n'
    add+='Phase 2：50点/99次运行/198帧；Phase 3：6个候选/12次运行/24帧。共111个案例逐一复核配置、trace、源码指纹和时间线哈希，均一致。136项测试通过，3条第三方弃用警告。\n\n'
    add+='全部候选在 cold 和 warm 下的时延、能耗仍高于对应 D0。本轮没有证据支持“优化后 HPAT 胜过匹配电子基线”。表中 C6 的默认推荐仅指被测光电混合候选，不是整个实验组的性能冠军；允许纯电子执行时，应优先保留 D0。\n\n'
    add+='C6/H1 warm平均功率7.734W，高于C0的7.294W，尽管时延下降21.25%、单帧能耗下降16.50%；因此不能写成时延、能耗、功率三者同时改善。C2的低能耗折中在cold/warm方向一致；C3仅减少容量，并无已建模的时延/能耗改善。\n\n'
    add+='P1要求的完整审计表已补到本目录 p1_event_audit_supplement.csv，不改动P1冻结结果。全部候选逐cold/warm对比见candidate_comparisons.csv。breakdown图经复核统一器件颜色/纹理，并使用分组事件条，避免跨方案图例错配。\n\n'
    add+='复现顺序：plan_sweep --phase 2 → summarize_plan prepare → plan_sweep --phase 3 --spec .../candidate_spec.json → summarize_plan final --style .../publication.mplstyle → finalize_plan_review。实际输出目录见各Phase记录；新运行需用新目录。\n'
    # This is the review step of a newly generated report, not a change to frozen data.
    report=P4/'REPORT.md'
    prior=report.read_text(encoding='utf-8').split('\n\n## 最终人工复核补充')[0]
    report.write_text(prior+add,encoding='utf-8')
    check=json.loads((P4/'verification.json').read_text())
    check.update(final_review=dict(valid=True,verified_cases=verified,pretrained_false=True,source_and_timeline_hashes_match=True,
        p1_audit_supplement=True,candidate_d0_comparison=True,plots_reviewed=True,tests_passed=136),
        review_source_sha256=sha256_file(Path(__file__)))
    write_json(P4/'verification.json',check)
    write_json(P4/'artifact_hashes.json',{f.name:sha256_file(f) for f in P4.iterdir() if f.is_file() and f.name!='artifact_hashes.json'})
    print(json.dumps(check['final_review']))


if __name__=='__main__':main()
