"""Run independent supported-subset profiles, retaining full config/provenance."""
import argparse
import copy
import json
import math
import platform
from pathlib import Path
from .device_profiles import SPECS, configure_profile, ProfileSimulation
from .backends import PhysicalCoreCosts
from .mobile_profile import profile_costs
from .run import REPO, ROOT, run_case, write_json, write_csv
from .plan_sweep import P0, audit_events
from .freeze_h123 import verify
from .pipeline_smoke import checks
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=['qkv','network'],default='qkv'); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    frozen=REPO/'results/hpat_mobilevit/h123_frozen_20260910_v1'
    before=verify(frozen); checks()
    base=json.loads((P0/'config.json').read_text())
    trace=REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl'
    manifest,records=load_manifest_jsonl(trace)
    if args.scope=='qkv':
        r=copy.deepcopy(next(r for r in records if r['op_type']=='Linear' and r['op_role']=='qkv_projection'))
        r['dependencies']=[]; records=[r]
    files=list(ROOT.rglob('*.py'))+list((REPO/'joint_sim').rglob('*.py'))+list((REPO/'SimPhony/configs').rglob('*.yml'))+list((REPO/'SimPhony/onnarchsim').rglob('*.py'))+[REPO/'LLMCompass/hardware_model/energy_model.py']
    hashes={str(f.relative_to(REPO)):sha256_file(f) for f in files}
    write_json(out/'provenance.json',dict(source_sha256=hashes,trace_sha256=sha256_file(trace),
        python=platform.python_version(),scope=args.scope,trace_manifest=manifest,
        frozen_before=before,parameter_source='DEVICE_PROFILES_20260910.md'))
    rows=[]; validations=[]
    modes={'D0':'digital','H1':'linear'}
    if args.scope=='network': modes.update(H2='linear_pointwise',H3='linear_pointwise_attention')
    for name in SPECS:
        cfg=configure_profile(base,name); costs=profile_costs(PhysicalCoreCosts(cfg),cfg)
        dest=out/name;dest.mkdir();write_json(dest/'config.json',cfg);write_json(dest/'backend.json',costs.summary())
        for label,mode in modes.items():
            part=run_case(cfg,costs,mode,manifest,records,dest/label,f'{name}_{label}',
                timeline=True,simulation_class=ProfileSimulation)
            summary=json.loads((dest/label/'summary.json').read_text())
            audit=audit_events(dest/label/'events.jsonl.gz',summary)
            write_csv(dest/label/'event_audit.csv',audit)
            assert summary['scratchpad_peak_bytes']<=cfg['model_assumptions']['activation_scratchpad_bytes']
            for f in summary['frames']:
                assert math.isclose(sum(f['energy_breakdown_j'].values()),f['energy_j'],rel_tol=1e-10)
                powers={'electronic_static':costs.electronic_energy.static_power_w}
                if mode!='digital': powers.update(costs.static_power)
                for k,v in powers.items(): assert math.isclose(f['energy_breakdown_j'][k],v*f['latency_s'],rel_tol=1e-10,abs_tol=1e-20)
            for row in part:
                row.update(profile=name,mapping=label,average_power_w=row['energy_mj']/row['latency_ms'])
            rows.extend(part);write_csv(out/'aggregate.csv',rows)
            validations.append(dict(profile=name,mapping=label,energy_closed=True,capacity_checked=True,
                static_full_duration=True,resource_exclusion='checked by event()',
                stage_dependencies='checked by schedule() per chunk',
                timeline_sha256=sha256_file(dest/label/'events.jsonl.gz')))
            print(json.dumps(part[-1]),flush=True)
        write_json(dest/'kernel_costs.json',list(costs.cache.values()))
    for f,d in hashes.items(): assert sha256_file(REPO/f)==d
    write_json(out/'verification.json',dict(cases=validations,frozen_after=verify(frozen),source_unchanged=True,
        physical_validation=False,full_requested_profile_implemented=False))
    write_json(out/'completion.json',dict(status='complete_supported_subset',scope=args.scope,
        runs=len(validations),frames=len(rows),aggregate_sha256=sha256_file(out/'aggregate.csv')))


if __name__=='__main__': main()
