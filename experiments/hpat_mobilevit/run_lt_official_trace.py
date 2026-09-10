"""Feed actual MobileViT Linear shapes to the unmodified official LT predictor.

This is a Linear-only reference smoke test, NOT end-to-end MobileViT. It uses
the official DOTA-B 8-bit configuration and native memory/energy accounting,
so its totals must not be compared directly to the unified-ledger HPAT totals.
Run in a fresh process to isolate the upstream `utils` package.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variant",choices=["xxs","xs","s"],default="xxs")
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    vendor=repo/"external/Lightening-Transformer/hardware_simulator"
    trace=repo/"results/hpat_mobilevit/trace_v1"/args.variant/"operator_trace.jsonl"
    manifest,records=load_manifest_jsonl(trace)
    sys.path.insert(0,str(vendor))
    os.chdir(vendor)
    from utils.config import configs
    from simulator_FFN import FFNPrediction
    configs.load("params/device_params/Dota_B_8bit.yaml",recursive=True)
    configs.arch.disable_crossbar_topology=0
    configs.arch.adc_share_flag=1
    configs.arch.time_accum_factor=3
    configs.arch.input_mod_sharing_flag=1
    rows=[]
    for r in records:
        if r["op_type"]!="Linear":
            continue
        op=dict(in_features=r["K"],out_features=r["N"],bs=r["M"]*r["batch_repetitions"])
        assert op["in_features"]*op["out_features"]*op["bs"]==r["macs"]
        predictor=FFNPrediction(op,configs)
        predictor.run(print_msg=False)
        e=predictor.energy_dict["linear"]
        rows.append(dict(op_id=r["op_id"],module_path=r["module_path"],M=op["bs"],K=r["K"],N=r["N"],
                         macs=r["macs"],energy_mj=e["comp"]["total"][0]+e["datamovement"]["total"][0],
                         latency_ms=predictor.latency_dict["linear"]["total"][1]))
    with (output/"linear_layers.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=dict(scope="Official LT DOTA-B 8bit, Linear-only trace import, NOT end-to-end or matched HPAT comparison",
                 model=manifest["model"],operators=len(rows),macs=sum(r["macs"] for r in rows),
                 energy_mj=sum(r["energy_mj"] for r in rows),latency_ms=sum(r["latency_ms"] for r in rows),
                 trace_sha256=sha256_file(trace),csv_sha256=sha256_file(output/"linear_layers.csv"),
                 official_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=vendor,text=True).strip(),
                 hardware=dict(tiles=4,cores_per_tile=2,width=12,height=12,wavelengths=12,bits=8),
                 config_sha256=sha256_file(vendor/"params/device_params/Dota_B_8bit.yaml"))
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
