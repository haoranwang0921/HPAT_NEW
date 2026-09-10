"""Reproduce immutable P0 configs with stage-attributed energy; no retuning."""
import argparse
import json
import math
import platform
from pathlib import Path
from .run import REPO, ROOT, run_case, write_json, write_csv
from .backends import PhysicalCoreCosts
from .energy_ledger import STAGES
from .mobile_profile import profile_costs
from .verify import verify_timeline
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    old = REPO/'results/hpat_mobilevit/hardware_combo_32_10_25_v1'
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    trace = REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl'
    manifest, records = load_manifest_jsonl(trace)
    files = list(ROOT.rglob('*.py')) + list((REPO/'joint_sim').rglob('*.py')) + list((REPO/'SimPhony/onnarchsim').rglob('*.py')) + list((REPO/'SimPhony/configs').rglob('*.yml')) + [REPO/'LLMCompass/hardware_model/energy_model.py']
    hashes = {str(f.relative_to(REPO)): sha256_file(f) for f in sorted(files)}
    frozen_hash = sha256_file(old/'aggregate.csv')
    assert frozen_hash == json.loads((old/'completion.json').read_text())['aggregate_sha256']
    write_json(out/'provenance.json', dict(source_sha256=hashes, trace_sha256=sha256_file(trace),
        python=platform.python_version(), trace_manifest=manifest, frozen_aggregate_sha256=frozen_hash,
        parameter_tag='Nominal', feasibility_tag='Aggressive hypothetical 10 ps whole-array write; not device validated'))
    rows = []; checks = []
    for point in ('base', 'array32x32_10ghz_sram25'):
        cfg = json.loads((old/point/'config.json').read_text())
        costs = profile_costs(PhysicalCoreCosts(cfg), cfg)
        target = out/point
        target.mkdir()
        write_json(target/'config.json', cfg)
        write_json(target/'backend.json', costs.summary())
        write_json(target/'override_audit.json', dict(
            device_default_response_ns=1000, device_reference_response_s=1e-6,
            scheduler_effective_response_s=cfg['user_confirmed']['program_response_time_s'],
            override_field='user_confirmed.program_response_time_s (scheduler only; device DB unchanged)',
            declared_scope=cfg['user_confirmed']['program_response_scope'],
            effective_physical_array=cfg['user_confirmed']['physical_array'],
            energy_multiplier=cfg['model_assumptions']['program_energy_multiplier'],
            tuning_energy_rule='reference per-ring joules * physical ring count * (response_s/1e-6) * multiplier; bias DAC added separately',
            units=dict(time='s', event_energy='J', static_power='W', initialized_device_energy='pJ', initialized_device_power='mW')))
        for mode in ('digital', 'linear_pointwise_attention'):
            run_id = f'{point}_{mode}'
            part = run_case(cfg, costs, mode, manifest, records, target/mode, run_id, timeline=True)
            summary = json.loads((target/mode/'summary.json').read_text())
            reference = json.loads((old/point/mode/'summary.json').read_text())
            verify = verify_timeline(target/mode/'events.jsonl.gz', summary)
            comparison = []
            for new, prior in zip(summary['frames'], reference['frames']):
                eb = new['energy_breakdown_j']; pb = prior['energy_breakdown_j']
                assert new['latency_s'] == prior['latency_s']
                assert new['event_counts'] == prior['event_counts']
                assert math.isclose(new['energy_j'], prior['energy_j'], rel_tol=1e-10)
                assert math.isclose(sum(eb.values()), new['energy_j'], rel_tol=1e-12)
                assert math.isclose(sum(eb.get(s, 0) for s in STAGES), sum(pb.get(s, 0) for s in STAGES), rel_tol=1e-10, abs_tol=1e-24)
                for name in set(eb) | set(pb):
                    if name not in STAGES:
                        assert eb.get(name, 0) == pb.get(name, 0), name
                if mode != 'digital':
                    assert eb['dac_encode'] > 0 and eb['adc_convert'] > 0
                comparison.append(dict(frame=new['frame'], latency_delta_s=new['latency_s']-prior['latency_s'],
                    energy_delta_j=new['energy_j']-prior['energy_j'], closure_error_j=sum(eb.values())-new['energy_j']))
            rows.extend(part)
            checks.append(dict(run_id=run_id, timeline=verify, reproduction=comparison))
            print(json.dumps(part[-1]), flush=True)
        write_json(target/'kernel_costs.json', list(costs.cache.values()))
    write_csv(out/'aggregate.csv', rows)
    for f, digest in hashes.items():
        assert sha256_file(REPO/f) == digest, f
    assert sha256_file(old/'aggregate.csv') == frozen_hash
    write_json(out/'verification.json', dict(valid=True, cases=checks, frozen_aggregate_unchanged=True,
        tolerance='energy relative 1e-10; exact latency/event counts/non-stage energy; numerical closure 1e-12'))
    write_json(out/'completion.json', dict(status='complete', runs=len(checks), frames=len(rows), aggregate_sha256=sha256_file(out/'aggregate.csv')))


if __name__ == '__main__':
    main()
