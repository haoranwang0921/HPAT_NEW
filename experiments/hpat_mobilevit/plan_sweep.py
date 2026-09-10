"""Bounded Phase 2/3 sweeps from the frozen P0, with explicit independent knobs."""
import argparse
import copy
import gzip
import json
import math
import platform
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from .backends import PhysicalCoreCosts
from .energy_ledger import split_components
from .mobile_profile import profile_costs
from .run import ROOT, REPO, run_case, write_json, write_csv
from .verify import verify_timeline
from joint_sim.trace_io import load_manifest_jsonl, sha256_file

H3 = 'linear_pointwise_attention'
MODES = {'D0':'digital', 'H1':'linear', 'H2':'linear_pointwise', 'H3':H3}
P0 = REPO/'results/hpat_mobilevit/hardware_combo_32_10_25_v1/array32x32_10ghz_sram25'
AXES = {
    'sram_mib': [.5,1,2,25], 'array': [[16,16],[32,16],[16,32],[32,32]],
    'frequency_ghz': [2.5,5,10], 'dram_gbs': [32,64,128],
    'program_ps': [10,100,1000,10000], 'program_parallelism': [1,4,16,32],
    'converter_energy': [.5,1,2,4], 'converter_time': [.5,1,2],
    'static_all': [.5,1,2], 'mapping': ['D0','H1','H2','H3'],
    'organization': [[1,4],[2,2],[4,1],[4,2]],
    # Disaggregate the joint static scenario to avoid attributing it to one device.
    'static_laser': [.5,1,2], 'static_mrr': [.5,1,2], 'static_electronic': [.5,1,2],
}


def configure(base, changes):
    c = copy.deepcopy(base); a=c['model_assumptions']; u=c['user_confirmed']
    for key, value in changes.items():
        if key == 'sram_mib': a['sram_capacity_bytes']=int(value*1024**2)
        elif key == 'array': u['physical_array']=value
        elif key == 'frequency_ghz': a['core_frequency_ghz']=value
        elif key == 'dram_gbs': a['hbm_bandwidth_bytes_per_s']=value*1e9
        elif key == 'program_ps':
            u['program_response_time_s']=value*1e-12
            a['program_energy_multiplier']=1e-6/u['program_response_time_s']
        elif key == 'program_parallelism': a[key]=value
        elif key == 'organization': u['tiles'],u['cores_per_tile']=value
        elif key not in {'converter_energy','converter_time','static_all','static_laser','static_mrr','static_electronic','mapping'}:
            raise ValueError(key)
    u['program_response_scope']='entire_current_physical_array_written_and_settled_hypothesis'
    c['plan_overrides']=changes
    c['experiment_evidence']={'pretrained':False, 'parameter_tag':'Nominal' if not changes else 'Conditional sensitivity',
        'feasibility_tag':'Aggressive whole-array write assumption', 'scope':'XXS shape/cost trace only; no accuracy or device measurement',
        'program_energy_policy':'fixed per-ring tuning energy across response-time scan',
        'organization_policy':'program lanes fixed at P0 unless explicitly scanned',
        'converter_policy':'DAC/ADC only; excludes MZM/TIA/PD and programming-bias DAC',
        'static_policy':'power coefficients only, no gating or timing changes'}
    return c


class SweepCosts:
    def __init__(self, c):
        self.base=profile_costs(PhysicalCoreCosts(c),c)
        self.config=c; self.cache={}; self.changes=c.get('plan_overrides',{})
        self.architecture=self.base.architecture
        self.programming=self.base.programming
        self.static_power=copy.deepcopy(self.base.static_power)
        d=self.changes; all_factor=d.get('static_all',1)
        for key in self.static_power:
            self.static_power[key] *= all_factor*d.get({'laser_static':'static_laser','mrr_hold':'static_mrr'}.get(key,''),1)
        self.electronic_energy=replace(self.base.electronic_energy,
            static_power_w=self.base.electronic_energy.static_power_w*all_factor*d.get('static_electronic',1))

    def kernel(self,M,K,N):
        key=(M,K,N)
        if key not in self.cache:
            k=copy.deepcopy(self.base.kernel(M,K,N))
            for name in k['dynamic_components_j']:
                if '_dac_' in name or '_adc_' in name:
                    k['dynamic_components_j'][name] *= self.changes.get('converter_energy',1)
            k['dynamic_energy_j']=sum(k['dynamic_components_j'].values())
            k['stage_components_j']=split_components(k['dynamic_components_j'])
            k['stage_energy_j']={s:sum(v.values()) for s,v in k['stage_components_j'].items()}
            for name in ('encode_s','convert_s'):
                k[name] *= self.changes.get('converter_time',1)
            k['sensitivity_overrides']=self.changes
            self.cache[key]=k
        return self.cache[key]

    def summary(self):
        from dataclasses import asdict
        s=self.base.summary()
        s.update(static_power_w=self.static_power,electronic_energy=asdict(self.electronic_energy),
                 sensitivity_overrides=self.changes, reference_device_values_are_before_overrides=True)
        return s


def union(intervals):
    total=0.; end=-math.inf
    for s,e in sorted(intervals):
        total += max(0.,e-max(end,s)); end=max(end,e)
    return total


def audit_events(path, summary):
    groups=defaultdict(list); energies=defaultdict(float)
    all_intervals=defaultdict(list)
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            e=json.loads(line); key=(e['frame'],e['event_type'])
            groups[key].append((e['start_s'],e['end_s']))
            all_intervals[e['frame']].append((e['start_s'],e['end_s']))
            energies[key]+=e['energy_j']
    rows=[]
    for frame in summary['frames']:
        i=frame['frame']; eb=frame['energy_breakdown_j']
        for (fi,kind),intervals in groups.items():
            if fi != i: continue
            assert math.isclose(energies[(i,kind)],eb[kind],rel_tol=1e-9,abs_tol=1e-14)
            rows.append(dict(frame=i,state=frame['state'],resource=kind,event_count=len(intervals),
                service_sum_s=sum(e-s for s,e in intervals),busy_union_s=union(intervals),
                dynamic_energy_j=energies[(i,kind)],static_energy_j=0.,unit_source='event seconds/joules; SimPhony pJ*1e-12 or electronic model J'))
        for kind in ('laser_static','mrr_hold','photonic_periphery_static','electronic_static'):
            if kind in eb:
                rows.append(dict(frame=i,state=frame['state'],resource=kind,event_count=0,
                    service_sum_s=frame['latency_s'],busy_union_s=frame['latency_s'],
                    dynamic_energy_j=0.,static_energy_j=eb[kind],unit_source='W * inference seconds; integrated, not discrete event'))
        total=sum(r['dynamic_energy_j']+r['static_energy_j'] for r in rows if r['frame']==i)
        assert math.isclose(total,frame['energy_j'],rel_tol=1e-10)
        # Union across ALL events reconstructs wall time; resource service sums do not.
        assert math.isclose(union(all_intervals[i]),frame['latency_s'],rel_tol=1e-9,abs_tol=1e-13)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['2','3'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spec',type=Path)
    args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    base=json.loads((P0/'config.json').read_text())
    trace=REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl'
    manifest,records=load_manifest_jsonl(trace)
    if args.phase=='2':
        points=[dict(name='center',axis='center',level='P0',changes={})]
        points += [dict(name=f'{axis}_{i}',axis=axis,level=json.dumps(v),changes={axis:v})
                   for axis,values in AXES.items() for i,v in enumerate(values)]
    else:
        points=json.loads(args.spec.read_text())
    write_json(out/'scan_spec.json',points); write_json(out/'config.json',base)
    files=list(ROOT.rglob('*.py'))+list((REPO/'joint_sim').rglob('*.py'))+list((REPO/'SimPhony/onnarchsim').rglob('*.py'))+list((REPO/'SimPhony/configs').rglob('*.yml'))+[REPO/'LLMCompass/hardware_model/energy_model.py']
    hashes={str(f.relative_to(REPO)):sha256_file(f) for f in sorted(files)}
    provenance=dict(source_sha256=hashes,trace_sha256=sha256_file(trace),trace_manifest=manifest,
        python=platform.python_version(),parameter_tag='Nominal center / Conservative cost-increase / Aggressive cost-reduction scenarios',
        pretrained=False,plan='EXPERIMENT_PLAN_REVISED_20260908.md',sram_area_leakage_rescaling=False)
    write_json(out/'provenance.json',provenance)
    rows=[]; checks=[]; center={}
    for point in points:
        c=configure(base,point['changes']); costs=SweepCosts(c)
        dest=out/point['name'];dest.mkdir()
        write_json(dest/'config.json',c);write_json(dest/'backend.json',costs.summary());write_json(dest/'provenance.json',provenance)
        mapped=MODES[point['changes'].get('mapping','H3')]
        case_checks=[]
        for mode in dict.fromkeys(['digital',mapped]):
            rid=f"{point['name']}_{mode}"
            part=run_case(c,costs,mode,manifest,records,dest/mode,rid,axis=point['axis'],value=point['level'],timeline=True)
            summary=json.loads((dest/mode/'summary.json').read_text())
            nevents=verify_timeline(dest/mode/'events.jsonl.gz',summary)
            write_csv(dest/mode/'event_audit.csv',audit_events(dest/mode/'events.jsonl.gz',summary))
            assert summary['scratchpad_peak_bytes']<=c['model_assumptions']['activation_scratchpad_bytes']
            assert summary['sram']['occupied']<=summary['sram']['capacity_tiles']
            for f in summary['frames']:
                assert math.isclose(sum(f['energy_breakdown_j'].values()),f['energy_j'],rel_tol=1e-12)
                assert math.isclose(f['energy_breakdown_j']['electronic_static'],costs.electronic_energy.static_power_w*f['latency_s'],rel_tol=1e-12)
                if mode!='digital':
                    for kind,power in costs.static_power.items():
                        assert math.isclose(f['energy_breakdown_j'][kind],power*f['latency_s'],rel_tol=1e-12,abs_tol=1e-24)
            if point['name']=='center':
                reference=json.loads((P0/mode/'summary.json').read_text())
                for f,b in zip(summary['frames'],reference['frames']):
                    assert f['latency_s']==b['latency_s']
                    assert math.isclose(f['energy_j'],b['energy_j'],rel_tol=1e-10)
                center[mode]=part
            elif args.phase=='2' and mode=='digital' and point['axis'] not in ('sram_mib','dram_gbs','static_all','static_electronic'):
                for f,b in zip(part,center['digital']):
                    assert f['latency_ms']==b['latency_ms'] and f['energy_mj']==b['energy_mj'], 'D0 optical coupling'
            for r in part:
                r.update(point=point['name'],level=point['level'],average_power_w=r['energy_mj']/r['latency_ms'],
                    photonic_area_mm2=costs.architecture['total_area_um2']/1e6 if mode!='digital' else 0.,
                    weight_sram_mib=c['model_assumptions']['sram_capacity_bytes']/1024**2)
            rows.extend(part)
            case_checks.append(dict(run_id=rid,valid=True,events=nevents,timeline_sha256=sha256_file(dest/mode/'events.jsonl.gz')))
            print(json.dumps(part[-1]),flush=True)
        write_json(dest/'kernel_costs.json',list(costs.cache.values()))
        write_json(dest/'verification.json',dict(valid=True,cases=case_checks))
        checks.extend(case_checks)
    write_csv(out/'aggregate.csv',rows)
    for axis in dict.fromkeys(p['axis'] for p in points):
        write_csv(out/f'{axis}.csv',[r for r in rows if r['axis']==axis])
    for f,digest in hashes.items(): assert sha256_file(REPO/f)==digest,f
    write_json(out/'verification.json',dict(valid=True,cases=checks,source_unchanged=True,scope='accounting, resources, existing dependencies, capacity, static integration; not physical validation'))
    write_json(out/'completion.json',dict(status='complete',points=len(points),runs=len(checks),frames=len(rows),aggregate_sha256=sha256_file(out/'aggregate.csv')))


if __name__=='__main__':main()
