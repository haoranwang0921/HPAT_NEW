"""Phase 2 rankings, bounded Phase 3 selection, and final Phase 4 exhibits."""
import argparse
import csv
import json
import math
from pathlib import Path
from .run import REPO, write_json, write_csv
from .plan_sweep import AXES, MODES, H3, P0
from joint_sim.trace_io import sha256_file

RESULTS=REPO/'results/hpat_mobilevit'
P2=RESULTS/'experiments_p2_sensitivity_v1'
P3=RESULTS/'experiments_p3_optimization_v1'
P4=RESULTS/'experiments_p4_final_summary_v1'
NAMES={'sram_mib':'权重 SRAM','array':'物理阵列','frequency_ghz':'光子频率','dram_gbs':'外存带宽',
 'program_ps':'写环响应时间','program_parallelism':'编程并行度','converter_energy':'DAC/ADC 能量',
 'converter_time':'DAC/ADC 时间','static_all':'全部静态功率（联合情景）','mapping':'映射范围',
 'organization':'tile/core 组织','static_laser':'激光静态功率','static_mrr':'MRR hold 功率','static_electronic':'电子静态功率'}
LABELS={'sram_mib':'Weight SRAM (0.5-25 MiB)','array':'Array (16x16 to 32x32)',
 'frequency_ghz':'Core frequency (2.5-10 GHz)','dram_gbs':'DRAM (32-128 GB/s)',
 'program_ps':'Write response (10 ps-10 ns)','program_parallelism':'Write lanes (1-32)',
 'converter_energy':'DAC/ADC energy (0.5-4x)','converter_time':'DAC/ADC time (0.5-2x)',
 'static_all':'All static power (0.5-2x; joint)', 'mapping':'Mapping (D0/H1/H2/H3)',
 'organization':'Organization (4-8 cores)','static_laser':'Laser power (0.5-2x)',
 'static_mrr':'MRR hold power (0.5-2x)','static_electronic':'Electronic static (0.5-2x)'}


def load_rows(root):
    c=json.loads((root/'completion.json').read_text())
    assert c['status']=='complete' and sha256_file(root/'aggregate.csv')==c['aggregate_sha256']
    assert json.loads((root/'verification.json').read_text())['valid']
    with (root/'aggregate.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('latency_ms','energy_mj','average_power_w','sram_hit_rate','photonic_area_mm2','weight_sram_mib'):
            r[k]=float(r[k])
    return rows


def selected(rows,axis=None,state='warm'):
    return [
        r for r in rows if r['state']==state and (axis is None or r['axis']==axis)
        and r['mode']==(MODES[json.loads(r['level'])] if r['axis']=='mapping' else H3)]


def rankings(rows):
    result=[]
    for state in ('warm','cold'):
        base=selected(rows,'center',state)[0]
        for axis in AXES:
            points=selected(rows,axis,state)
            x=dict(axis=axis,parameter=NAMES[axis],state=state,scan_range=json.dumps(AXES[axis]),evidence='条件性；记账已验证，器件/系统未实测')
            for key,short in [('latency_ms','latency'),('energy_mj','energy'),('average_power_w','power')]:
                ds=[100*(p[key]/base[key]-1) for p in points]
                x[short+'_max_abs_percent']=max(abs(v) for v in ds)
                x[short+'_min_change_percent']=min(ds);x[short+'_max_change_percent']=max(ds)
            x['joint_aux_percent']=.5*(x['latency_max_abs_percent']+x['energy_max_abs_percent'])
            v=max(x['latency_max_abs_percent'],x['energy_max_abs_percent'])
            x['impact']='强影响' if v>=20 else '中影响' if v>=5 else '弱影响'
            x['hpat_beats_matched_d0_both']=False
            x['latency_order_reversal']=False;x['energy_order_reversal']=False
            for p in points:
                d=next(r for r in rows if r['point']==p['point'] and r['mode']=='digital' and r['state']==state)
                x['hpat_beats_matched_d0_both'] |= p['mode']!='digital' and p['latency_ms']<d['latency_ms'] and p['energy_mj']<d['energy_mj']
                x['latency_order_reversal'] |= p['mode']!='digital' and p['latency_ms']<d['latency_ms']
                x['energy_order_reversal'] |= p['mode']!='digital' and p['energy_mj']<d['energy_mj']
            result.append(x)
        for metric in ('latency','energy','power'):
            for i,r in enumerate(sorted([r for r in result if r['state']==state],key=lambda r:r[metric+'_max_abs_percent'],reverse=True),1):
                r[metric+'_rank']=i
    return result


def mdtable(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def prepare():
    rows=load_rows(P2); ranks=rankings(rows)
    write_csv(P2/'sensitivity_rankings.csv',ranks)
    for metric in ('latency','energy','power'):
        write_csv(P2/f'{metric}_ranking_warm.csv',sorted([r for r in ranks if r['state']=='warm'],key=lambda r:r[metric+'_rank']))
    specs=[dict(name='C0',axis='candidate',level='C0',changes={}),
           dict(name='C1',axis='candidate',level='C1',changes={'array':[32,16]}),
           dict(name='C2',axis='candidate',level='C2',changes={'array':[32,16],'sram_mib':2}),
           dict(name='C3',axis='candidate',level='C3',changes={'sram_mib':2}),
           dict(name='C4',axis='candidate',level='C4',changes={'converter_time':.5,'converter_energy':.5})]
    program=[r for r in ranks if r['state']=='warm' and r['axis'] in ('program_ps','program_parallelism')]
    drop=all(r['impact']=='弱影响' for r in program)
    if not drop: specs.append(dict(name='C5',axis='candidate',level='C5',changes={'program_parallelism':32}))
    # Mapping is a listed scan; add one bounded software-mapping candidate if
    # it dominates H3 in both objectives at P0 hardware.
    base=selected(rows,'center')[0]
    h1=next(r for r in selected(rows,'mapping') if r['mode']=='linear')
    if h1['latency_ms']<base['latency_ms'] and h1['energy_mj']<base['energy_mj']:
        specs.append(dict(name='C6',axis='candidate',level='C6',changes={'mapping':'H1','sram_mib':2}))
    write_json(P2/'candidate_spec.json',specs)
    write_json(P2/'candidate_selection.json',dict(C5_deleted=drop,
        C5_reason='10 ps-10 ns fixed-energy time scan and 1-32 lanes both below 5% warm latency/energy impact' if drop else 'retained',
        C4_evidence='Conditional converter co-design bound, not a demonstrated device',
        C6_reason='H1 dominates H3 at P0 hardware; combine with independently saturated 2 MiB cache',
        excluded='No bandwidth or static-coefficient reductions promoted as free hardware improvements'))
    text='# Phase 2 单因素扫描完成\n\n'
    text+=f"{len(json.loads((P2/'scan_spec.json').read_text()))} 个硬件/映射点；{len(rows)} 个 cold/warm frame。所有点均经过事件、容量、静态积分及能量闭合检查。\n\n"
    text+='中心点严格复现 P0；只使用固定 XXS FP32 随机权重 shape trace，不加载官方权重。\n\n'
    text+=mdtable(['时延排名','参数','时延最大变化 %','能耗最大变化 %','功率最大变化 %','类别'],
        [[r['latency_rank'],r['parameter'],f"{r['latency_max_abs_percent']:.3f}",f"{r['energy_max_abs_percent']:.3f}",f"{r['power_max_abs_percent']:.3f}",r['impact']]
         for r in sorted([r for r in ranks if r['state']=='warm'],key=lambda x:x['latency_rank'])])
    text+='\n\n排序只适用于列出的扫描范围，不是范围无关的固有重要性。static_all 为联合功率系数情景；另外单列激光、MRR、电子静态扫描。映射范围含 D0，不能把去除光子硬件造成的下降说成光子优化。\n'
    text+='\n写环时间变化保持每环调谐能量不变；tile/core 组织改变时写入通道数保持 4。转换能量倍率仅作用于输入 DAC/输出 ADC，不改 TIA/PD/MZM 或偏置 DAC；时间倍率是理想转换服务时间假设。\n'
    text+='\n完整单参数 CSV、cold/warm、每点 config/provenance/verification、layers、event_audit 和 kernel_costs 均在本目录。图及最终解释统一在 Phase 4，避免重复出图。\n'
    (P2/'REPORT.md').write_text(text,encoding='utf-8')


def final(style):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    p2=load_rows(P2); p3=load_rows(P3); ranks=rankings(p2)
    P4.mkdir(parents=True,exist_ok=False)
    plt.style.use(style)
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'figure.constrained_layout.use':False})
    footer='XXS 192, batch=1; FP32 random-weight shape trace; 8-bit cost; P1 ledger\nSRAM area/leakage NOT scaled; deterministic cold/warm frames (no CI); 10 ps write hypothesis'
    figures=[]
    def save(fig,name,scope):
        fig.text(.5,.012,scope+'\n'+footer,ha='center',va='bottom',fontsize=7)
        fig.tight_layout(rect=[0,.13,1,1])
        for ext in ('pdf','png'):
            fig.savefig(P4/f'{name}.{ext}',dpi=300)
        figures.append(name);plt.close(fig)
    warm=[r for r in ranks if r['state']=='warm']
    for key in ('latency','energy'):
        order=sorted(warm,key=lambda r:r[key+'_max_abs_percent'])
        fig,ax=plt.subplots(figsize=(9,7))
        y=np.arange(len(order))
        ax.barh(y,[min(0,r[key+'_min_change_percent']) for r in order],color='#0072B2',label='Minimum change')
        ax.barh(y,[max(0,r[key+'_max_change_percent']) for r in order],color='#E69F00',label='Maximum change')
        ax.set_yticks(y,[LABELS[r['axis']] for r in order]);ax.axvline(0,color='black',lw=.7)
        ax.set_xlabel(f'{key.capitalize()} change relative to P0 (%)');ax.legend()
        save(fig,key+'_tornado','Warm; ranges shown at left; P0 = 32x32 / 10 GHz / 25 MiB / H3')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    pts=selected(p2,'array')
    for p in pts:
        axes[0].scatter(p['latency_ms'],p['energy_mj'],s=60,marker='s')
        axes[0].annotate(p['level'],(p['latency_ms'],p['energy_mj']),xytext=(3,5),textcoords='offset points')
    axes[0].set_title('A  Array trade-off (all other settings P0)')
    spec=json.loads((P2/'candidate_spec.json').read_text())
    cmap={s['name']:s for s in spec}
    candidates=[r for r in p3 if r['state']=='warm' and r['mode']==MODES[cmap[r['point']]['changes'].get('mapping','H3')]]
    for p in candidates:
        axes[1].scatter(p['latency_ms'],p['energy_mj'],s=70,marker='*' if p['point']=='C0' else 'o')
        axes[1].annotate(p['point'],(p['latency_ms'],p['energy_mj']),xytext=(4,5 if p['point'] not in ['C2','C3'] else -13),textcoords='offset points')
    frontier=[p for p in candidates if not any(q['latency_ms']<=p['latency_ms'] and q['energy_mj']<=p['energy_mj'] and (q['latency_ms']<p['latency_ms'] or q['energy_mj']<p['energy_mj']) for q in candidates)]
    frontier.sort(key=lambda p:p['latency_ms'])
    axes[1].plot([p['latency_ms'] for p in frontier],[p['energy_mj'] for p in frontier],'k--',lw=.8,label='Sampled Pareto frontier')
    axes[1].legend();axes[1].set_title('B  Candidate trade-off (C0 = P0)')
    for ax in axes: ax.set_xlabel('Latency (ms)');ax.set_ylabel('Energy (mJ)')
    save(fig,'pareto','Warm; C4 is a hypothetical converter improvement; no total-chip area claim')
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for state,marker in [('cold','s'),('warm','o')]:
        pts=sorted(selected(p2,'sram_mib',state),key=lambda p:p['weight_sram_mib'])
        for ax,metric,label in zip(axes,['sram_hit_rate','latency_ms','energy_mj'],['Cache hit rate','Latency (ms)','Energy (mJ)']):
            ax.plot([p['weight_sram_mib'] for p in pts],[p[metric] for p in pts],marker=marker,label=state)
            ax.set_xscale('log',base=2);ax.set_xticks([.5,1,2,25],['0.5','1','2','25'])
            ax.set_xlabel('Weight SRAM (MiB)');ax.set_ylabel(label);ax.axvline(25,color='grey',ls=':',lw=.8);ax.legend()
    save(fig,'sram_curve','SRAM scan: 0.5 / 1 / 2 / 25 MiB; all else P0; 25 MiB = P0')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for state,dx in [('cold',-.18),('warm',.18)]:
        items=[next(r for r in p3 if r['point']==p['point'] and r['mode']==p['mode'] and r['state']==state) for p in candidates]
        for ax,key in zip(axes,['latency_ms','energy_mj']):
            ax.bar(np.arange(len(items))+dx,[p[key] for p in items],width=.35,label=state,hatch='//' if state=='cold' else None)
            ax.set_xticks(range(len(items)),[p['point'] for p in items]);ax.legend()
    axes[0].set_ylabel('Latency (ms)');axes[1].set_ylabel('Energy (mJ)')
    save(fig,'cold_warm','Candidates C0-C6 as listed in candidate_spec.json; C0 = P0')
    chosen='C6' if 'C6' in cmap else 'C3'
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for j,name in enumerate(['C0',chosen]):
        p=next(p for p in candidates if p['point']==name)
        summary=json.loads((P3/name/p['mode']/'summary.json').read_text())
        eb=summary['frames'][1]['energy_breakdown_j']
        left=0
        for k,v in eb.items():
            if v==0:continue
            axes[0].barh(j,v*1000,left=left,label=k if j==0 else None);left+=v*1000
        with (P3/name/p['mode']/'event_audit.csv').open() as f: ev=list(csv.DictReader(f))
        for r in ev:
            if r['state']=='warm' and r['resource'] in ('hbm','sram_noc','dac_encode','optical_compute','adc_convert','signed_reduce','electronic_dynamic','programming'):
                axes[1].barh(r['resource'],float(r['busy_union_s'])*1e3,height=.35,left=0,
                    alpha=.6 if j==0 else 1,label=name if r['resource']=='hbm' else None,
                    fill=j==1,edgecolor='#0072B2' if j==0 else '#E69F00')
    axes[0].set_yticks([0,1],['C0',chosen]);axes[0].set_xlabel('Total energy breakdown (mJ)')
    axes[0].legend(fontsize=6,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.4))
    axes[1].set_xlabel('Per-kind busy union (ms; NOT additive)');axes[1].legend()
    save(fig,'breakdown',f'Warm C0 vs {chosen}; event unions overlap and are not critical-path contributions')
    # Preserve all final numeric tables and references, not only selected best points.
    write_csv(P4/'aggregate.csv',p3);write_csv(P4/'parameter_rankings.csv',ranks)
    write_json(P4/'config.json',json.loads((P2/'config.json').read_text()))
    write_json(P4/'candidate_spec.json',spec)
    provenance=json.loads((P2/'provenance.json').read_text())
    provenance.update(p2_aggregate_sha256=sha256_file(P2/'aggregate.csv'),p3_aggregate_sha256=sha256_file(P3/'aggregate.csv'),
        analysis_source_sha256=sha256_file(Path(__file__)),style_sha256=sha256_file(style))
    write_json(P4/'provenance.json',provenance)
    checks=[]
    for axis in ('static_all','static_laser','static_mrr','static_electronic','converter_energy'):
        for state in ('cold','warm'):
            b=selected(p2,'center',state)[0]
            for r in selected(p2,axis,state): assert r['latency_ms']==b['latency_ms']
        checks.append(axis+': energy-only changes preserve exact timing')
    for r in selected(p2,'program_ps'):
        s=json.loads((P2/r['point']/r['mode']/'summary.json').read_text())
        b=json.loads((P2/'center'/H3/'summary.json').read_text())
        for x,y in zip(s['frames'],b['frames']):
            assert math.isclose(x['energy_breakdown_j']['programming'],y['energy_breakdown_j']['programming'],rel_tol=1e-10)
    checks.append('write-time scan retains programming event energy')
    # Build evidence-backed final report, without claiming physical feasibility.
    c0=next(r for r in candidates if r['point']=='C0')
    table=[]
    for r in candidates:
        dl=100*(r['latency_ms']/c0['latency_ms']-1);de=100*(r['energy_mj']/c0['energy_mj']-1)
        recommendation={'C0':'参照','C1':'低能耗折中','C2':'H3 低能耗推荐','C3':'H3 默认缓存配置','C4':'条件性器件优化上界','C5':'编程对照','C6':'系统默认/低延迟推荐'}.get(r['point'],'对照')
        table.append([r['point'],f"{r['latency_ms']:.6f}",f"{r['energy_mj']:.6f}",f"{r['average_power_w']:.3f}",f"{r['photonic_area_mm2']:.3f}",f'{dl:+.2f}% / {de:+.2f}%',recommendation])
    ranktable=mdtable(['时延排名','参数及范围','最大时延变化 %','最大能耗变化 %','最大功率变化 %','影响类型'],
        [[r['latency_rank'],r['parameter']+' '+r['scan_range'],f"{r['latency_max_abs_percent']:.3f}",f"{r['energy_max_abs_percent']:.3f}",f"{r['power_max_abs_percent']:.3f}",r['impact']]
         for r in sorted(warm,key=lambda x:x['latency_rank'])])
    report='# Phase 4 最终汇总：MobileViT-XXS × HPAT\n\n本轮 Phase 0–4 已完成。\n\n'
    report+='## 表 1：实验边界\n\n本轮未使用官方预训练权重；FP32 随机权重仅用于生成真实算子形状和成本 trace，不报告模型准确率。固定 XXS 192×192、batch=1、INT8 成本/INT32 累加、cold 后 warm 各一帧；不是随机重复测量，故无置信区间。\n\n'
    report+='P0 为 32×32、2 tiles×2 cores、10 GHz、25 MiB 权重缓存、8 MiB scratchpad、64 GB/s DRAM、256 GB/s SRAM、10 TOPS、电子利用率100%、静态0.3 W、MAC0.4 pJ、10 ps写环、4写入通道、H3。所有主结果均使用 P1 分组账本。\n\n'
    report+='不计 SRAM 扩容面积/漏电变化；面积仅为光子后端估计。32×32 全阵列10 ps写入、转换器改善、静态系数缩放均无本轮器件实测证明。标签为 Nominal 中心点、Conservative 成本提高情景、Aggressive 成本降低情景；不按数值好坏将结果当成已证实设计。\n\n'
    report+='## 表 2：影响排序（warm）\n\n'+ranktable+'\n\n'
    for metric,cn in [('latency','时延'),('energy','能耗'),('power','功率')]:
        report+=cn+'排序：'+' → '.join(r['parameter'] for r in sorted(warm,key=lambda x:x[metric+'_rank']))+'。\n\n'
    report+='最大变化使用 max|X-X0|/X0；范围不同会影响排序，不能解释成单位参数扰动的固有灵敏度。强/中/弱阈值分别为≥20%、5–20%、<5%。完整 cold 排序和 S_joint=0.5S_L+0.5S_E 辅助分数见 parameter_rankings.csv；静态联合情景不是单一器件因果结果。映射含 D0，其优势不归功于光子加速。\n\n'
    report+='## 表 3：候选方案（warm）\n\n'+mdtable(['方案','时延 ms','能耗 mJ','平均 W','光子面积 mm²','相对 C0 时延/能耗','推荐'],table)+'\n\n'
    report+='C1=32×16/25MiB；C2=32×16/2MiB；C3=32×32/2MiB；均10GHz/H3。C4=转换能量与服务时间均0.5×，并非实测可用器件。C6=H1/32×32/2MiB（如出现）。C5 的去留及原因见 Phase 2 candidate_selection.json。\n\n'
    report+='## 表 4：推荐与结论\n\n'
    report+='最影响结果的参数是扫描范围内的权重 SRAM 容量；时延重点是缓存、外存带宽和映射范围，能耗/功率重点还包括静态功率与阵列/核数；建议优先保留足够但不过量的缓存、采用有证据的映射选择，不推荐无条件扩 SRAM/核数或把降低转换能量当作主要突破口。\n\n'
    report+='- 默认：先将 XXS 权重 SRAM 从25MiB收敛到2MiB；warm命中已饱和，当前模型的 cold/warm 时延能耗均不变。只能声称容量减少92%，不能量化尚未建模的面积/漏电节省。若允许改变映射，优先 C6/H1，需保留 H3 为消融对照。\n'
    report+='- 低延迟：在本轮已验证的映射组合中使用 H1；C4 仅为有条件转换器目标，不作为现成硬件推荐。提高外存带宽也改善 D0，不能只给 HPAT 增带宽再宣称胜出。\n'
    report+='- H3 低能耗：C2（32×16/2MiB）换取能耗和光子面积下降，但时延增加，必须明确折中。若不要求使用光子硬件，匹配 D0 仍是必须保留的系统选择。\n\n'
    report+='## RQ1 与证据边界\n\nP0 warm 总能量5.831783mJ，DRAM约38.94%、激光35.26%，DAC/MZM与ADC/TIA/PD合计约2.32%。资源忙碌并集不是关键路径贡献，多种资源会重叠；不能把所有事件服务时间相加作为端到端时延。逐事件计数、服务时间、忙碌并集、动静态能量和单位来源见各点 event_audit.csv，跨所有事件并集能重建单帧时长。\n\n'
    report+='已验证事实仅指固定图和成本模型的运行结果。条件性结论依赖缓存、带宽、功率、写环和转换器假设。未建模：真实10ps全阵列可实现性、SRAM面积/漏电、额外编程通道物理成本、官方权重准确率、量化鲁棒性、手机测量、未显式体现的tile层级互连。相同4核不同tile组织若结果相同，说明本模型未建模对应拓扑差异，不代表真实芯片组织无影响。\n\n'
    report+='## 复现与图表\n\nPhase 2/3 保留每点完整配置、源代码指纹、trace哈希、完整时间线、层级与器件账本、独立核验和 aggregate哈希。Phase 4 保存输入哈希、分析脚本和样式哈希。图采用科学可视化技能的 publication 样式，色盲友好配色及 PDF/PNG 输出；确定性扫描不伪造误差条。\n\n'
    report+='图：'+', '.join(figures)+'。\n'
    (P4/'REPORT.md').write_text(report,encoding='utf-8')
    (P3/'REPORT.md').write_text('# Phase 3 候选组合验证\n\n'+mdtable(['方案','时延 ms','能耗 mJ','平均 W','光子面积 mm²','相对 C0 时延/能耗','推荐'],table)+'\n\n完整推荐、风险及 cold/warm 图见 ../experiments_p4_final_summary_v1/REPORT.md；本目录仅使用固定随机权重 XXS shape trace，不报告准确率。',encoding='utf-8')
    write_json(P4/'verification.json',dict(valid=True,upstream_phase2=True,upstream_phase3=True,additional_checks=checks,
        figures=figures,scope='Data consistency and accounting, not silicon validation'))
    write_json(P4/'completion.json',dict(status='complete',aggregate_sha256=sha256_file(P4/'aggregate.csv'),figures=len(figures),phase_limit=4))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','final']);p.add_argument('--style',type=Path)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    else:final(args.style)
