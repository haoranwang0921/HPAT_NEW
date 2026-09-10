"""XXS one-factor hardware sensitivity; fixed electronic cache layout, fresh costs."""
import argparse
import copy
import json
import math
import platform
from pathlib import Path
from .run import ROOT, REPO, run_case, write_json, write_csv
from .backends import PhysicalCoreCosts
from .mobile_profile import resolve_profile, profile_costs
from .verify import verify_timeline
from joint_sim.trace_io import load_manifest_jsonl, sha256_file

POINTS = {
    "base": {},
    "cores2_1x2": {"tiles":1},
    "cores8_4x2": {"tiles":4},
    "cores4_1x4": {"tiles":1,"cores_per_tile":4},
    "cores4_4x1": {"tiles":4,"cores_per_tile":1},
    "array16x32": {"physical_array":[16,32]},
    "array32x16": {"physical_array":[32,16]},
    "array32x32": {"physical_array":[32,32]},
    "freq2p5": {"core_frequency_ghz":2.5},
    "dram32": {"hbm_bandwidth_bytes_per_s":32e9},
    "sram0p5": {"sram_capacity_bytes":524288},
    "sram2": {"sram_capacity_bytes":2097152},
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--combo-32-10-25", action="store_true",
                   help="Run baseline and 32x32/10GHz/25MiB combined point")
    args=p.parse_args()
    points = ({"base": {}, "array32x32_10ghz_sram25": {
        "physical_array": [32,32], "core_frequency_ghz": 10,
        "sram_capacity_bytes": 25*1024*1024}} if args.combo_32_10_25 else POINTS)
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    cfg=resolve_profile(json.loads((ROOT/"config_10ps_fixed_write_energy.json").read_text()),
                        json.loads((ROOT/"mobile_reference.json").read_text()),10,"mobile_reference")
    cfg["model_assumptions"].update(weight_cache_block_shape=[16,8],inferences_per_run=2)
    for k in ["dma_batch_fetch","electronic_io_compute_overlap","activity_gated_static","reduce_compute_overlap","accumulator_residency"]:
        cfg["model_assumptions"][k]=False
    trace=REPO/"results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl"
    manifest,records=load_manifest_jsonl(trace)
    files=list(ROOT.rglob("*.py"))+list((REPO/"joint_sim").rglob("*.py"))+list((REPO/"SimPhony/onnarchsim").rglob("*.py"))+list((REPO/"SimPhony/configs").rglob("*.yml"))+[REPO/"LLMCompass/hardware_model/energy_model.py"]
    hashes={str(f.relative_to(REPO)):sha256_file(f) for f in sorted(files)}
    write_json(out/"provenance.json",dict(source_sha256=hashes,trace_sha256=sha256_file(trace),python=platform.python_version(),
       scope="Conditional architecture sensitivity; no SRAM area/leakage rescaling, no accuracy/physical feasibility claim"))
    write_json(out/"scan_spec.json",dict(base=cfg,points=points))
    rows=[]; checks=[]; warm=[]
    for name,changes in points.items():
        c=copy.deepcopy(cfg)
        for k,v in changes.items():
            c["user_confirmed" if k in ("tiles","cores_per_tile","physical_array") else "model_assumptions"][k]=v
        # One writing lane per core, explicitly part of the scaled hardware.
        u=c["user_confirmed"]; a=c["model_assumptions"]
        a["program_parallelism"]=u["tiles"]*u["cores_per_tile"]
        costs=profile_costs(PhysicalCoreCosts(c),c)
        d=out/name; d.mkdir()
        write_json(d/"config.json",c);write_json(d/"backend.json",costs.summary())
        for mode in ("digital","linear_pointwise_attention"):
            rid=f"{name}_{mode}"
            part=run_case(c,costs,mode,manifest,records,d/mode,rid,timeline=True)
            summary=json.loads((d/mode/"summary.json").read_text())
            events=verify_timeline(d/mode/"events.jsonl.gz",summary)
            for f in summary["frames"]:
                assert math.isclose(sum(f["energy_breakdown_j"].values()),f["energy_j"],rel_tol=1e-10)
            for r in part:
                r.update(point=name,average_power_w=r["energy_mj"]/r["latency_ms"],
                         photonic_area_mm2=costs.architecture["total_area_um2"]/1e6 if mode!="digital" else 0)
            rows.extend(part);warm.append(part[-1])
            checks.append(dict(run_id=rid,events=events,timeline_sha256=sha256_file(d/mode/"events.jsonl.gz")))
            print(json.dumps(part[-1]),flush=True)
        write_json(d/"kernel_costs.json",list(costs.cache.values()))
    base_d=next(r for r in warm if r["point"]=="base" and r["mode"]=="digital")
    base_h=next(r for r in warm if r["point"]=="base" and r["mode"]!="digital")
    for r in warm:
        b=base_d if r["mode"]=="digital" else base_h
        r["latency_change_percent"]=(r["latency_ms"]/b["latency_ms"]-1)*100
        r["energy_change_percent"]=(r["energy_mj"]/b["energy_mj"]-1)*100
        if r["mode"]=="digital" and not any(k in points[r["point"]] for k in (
                "hbm_bandwidth_bytes_per_s", "sram_capacity_bytes")):
            assert r["latency_ms"]==base_d["latency_ms"] and r["energy_mj"]==base_d["energy_mj"], "D0 optical coupling"
    # warm dicts also occur in rows: write homogeneous CSVs separately.
    write_csv(out/"aggregate.csv",[{k:v for k,v in r.items() if not k.endswith("change_percent")} for r in rows])
    write_csv(out/"warm_comparison.csv",warm)
    for f,digest in hashes.items(): assert sha256_file(REPO/f)==digest, f
    write_json(out/"verification.json",dict(valid=True,cases=checks,d0_invariant_to_optical_parameters=True))
    write_json(out/"completion.json",dict(status="complete",runs=len(checks),frames=len(rows),aggregate_sha256=sha256_file(out/"aggregate.csv")))


if __name__=="__main__":main()
