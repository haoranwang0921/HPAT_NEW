"""D0-only user-selected sensitivity; does not alter hybrid defaults."""
import argparse
import json
import math
from .run import ROOT, REPO, run_case, write_json, write_csv
from .backends import PhysicalCoreCosts
from .mobile_profile import profile_costs
from .verify import verify_timeline
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--static-power-w', type=float, default=0.5)
    args = p.parse_args()
    out = REPO / args.output
    out.mkdir(parents=True, exist_ok=False)
    source = REPO/'results/hpat_mobilevit/hardware_combo_32_10_25_v1/base/config.json'
    cfg = json.loads(source.read_text())
    a = cfg['model_assumptions']
    a.update(electronic_compute_utilization=0.8,
             electronic_effective_ops_per_s=8e12, electronic_peak_flops=8e12,
             sram_capacity_bytes=2*1024*1024)
    a['electronic_energy_overrides']['matmul_energy_per_mac_j'] = 0.9e-12
    a['electronic_energy_overrides']['static_power_w'] = args.static_power_w
    cfg['mobile_reference']['source_specification']['compute_utilization'] = 0.8
    cfg['mobile_reference']['source_specification']['mobile_assumptions']['electronic_static_power_w'] = args.static_power_w
    cfg['scope'] = ('D0-only user-selected sensitivity: 10 TOPS peak, 80% compute utilization, '
                    '2 MiB weight SRAM, 0.9 pJ/MAC. Vector effective throughput also scales '
                    'with utilization under existing scheduler semantics. HPAT unchanged. '
                    f'Static power {args.static_power_w:g} W. '
                    'Not device-calibrated; SRAM area/leakage not rescaled.')
    costs = profile_costs(PhysicalCoreCosts(cfg), cfg)
    assert costs.electronic_energy.matmul_energy_per_mac_j == 0.9e-12
    assert costs.electronic_energy.static_power_w == args.static_power_w
    assert a['electronic_peak_ops_per_s'] == 10e12
    write_json(out/'config.json', cfg)
    write_json(out/'backend.json', costs.summary())
    trace = REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl'
    files = list(ROOT.rglob('*.py')) + list((REPO/'joint_sim').rglob('*.py'))
    hashes = {str(f.relative_to(REPO)): sha256_file(f) for f in files}
    write_json(out/'provenance.json', dict(source_sha256=hashes,
        input_config_sha256=sha256_file(source), trace_sha256=sha256_file(trace)))
    manifest, records = load_manifest_jsonl(trace)
    rows = run_case(cfg, costs, 'digital', manifest, records, out/'digital', 'd0_u80_sram2_mac0p9', timeline=True)
    summary = json.loads((out/'digital/summary.json').read_text())
    events = verify_timeline(out/'digital/events.jsonl.gz', summary)
    for f in summary['frames']:
        assert math.isclose(sum(f['energy_breakdown_j'].values()), f['energy_j'], rel_tol=1e-10)
    for row in rows:
        row['average_power_w'] = row['energy_mj']/row['latency_ms']
    write_csv(out/'aggregate.csv', rows)
    for f, digest in hashes.items():
        assert sha256_file(REPO/f) == digest
    write_json(out/'completion.json', dict(status='complete', events=events, frames=len(rows)))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
