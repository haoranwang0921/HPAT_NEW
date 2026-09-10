"""Paper-latency alignment experiment isolated from frozen HPAT paths."""
import argparse
import json
import platform
from dataclasses import asdict,replace
from pathlib import Path
from .run import REPO,write_json,write_csv
from .soc_reference import priors,calibrate,simulate
from .freeze_h123 import verify
from joint_sim.trace_io import load_manifest_jsonl,sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);args=p.parse_args()
    before=verify(args.freeze);out=args.output;out.mkdir(parents=True,exist_ok=False)
    manifest,records=load_manifest_jsonl(args.trace)
    source=dict(url='https://arxiv.org/html/2110.02178v2',location='Table 11; Section 4.3',accessed='2026-09-10',
        model='MobileViT-XS (~2.3M parameters)',input_shape=[1,3,256,256],patch_sizes=[2,2,2],
        paper_weights='pretrained full-precision models converted with CoreMLTools',
        timing='mean over 100 iterations',cpu_ms=17.86,neural_engine_ms=7.28,
        unknown=['internal execution precision','operator placement/fallback','CoreML fusion graph','warmup details','power/energy'])
    write_json(out/'paper_reference.json',source)
    rows=[];calibrations={}
    for backend,target in [('cpu',17.86),('neural_engine',7.28)]:
        fitted,fit=calibrate(records,manifest,target,priors(backend));calibrations[backend]=fit
        write_json(out/f'{backend}_profile.json',asdict(fitted))
        s,ops=simulate(records,fitted)
        write_csv(out/f'{backend}_xs256_operators.csv',ops)
        rows.append(dict(backend=backend,variant='xs',resolution=256,status='calibration_training_point',paper_ms=target,
                         latency_ms=s['latency_s']*1000,energy_mj=None,average_power_w=None))
        # Existing traces are extrapolations, never held-out device measurements.
        for variant in ('xxs','xs','s'):
            path=REPO/f'results/hpat_mobilevit/trace_v1/{variant}/operator_trace.jsonl'
            if not path.exists():continue
            m,rs=load_manifest_jsonl(path);s,ops=simulate(rs,fitted)
            write_csv(out/f'{backend}_{variant}{m["input_shape"][-1]}_operators.csv',ops)
            rows.append(dict(backend=backend,variant=variant,resolution=m['input_shape'][-1],status='unvalidated_extrapolation',
                paper_ms=None,latency_ms=s['latency_s']*1000,energy_mj=None,average_power_w=None))
    # Different priors fit the same single anchor. Do not mistake this for identification.
    ident=[]
    for scale in (.5,1,2):
        prior=replace(priors('neural_engine'),dispatch_s=2e-6*scale,
                      logical_bandwidth_bytes_per_s=32e9*scale)
        fit,meta=calibrate(records,manifest,7.28,prior)
        m,rs=load_manifest_jsonl(REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl')
        estimate,_=simulate(rs,fit)
        ident.append(dict(prior_scale=scale,fitted_runtime_scale=fit.runtime_scale,
            training_xs256_ms=simulate(records,fit)[0]['latency_s']*1000,extrapolated_xxs192_ms=estimate['latency_s']*1000))
    write_csv(out/'identifiability.csv',ident);write_csv(out/'aggregate.csv',rows)
    write_json(out/'calibration.json',calibrations)
    write_json(out/'provenance.json',dict(python=platform.python_version(),trace_sha256=sha256_file(args.trace),
        trace_manifest=manifest,source_sha256={str(f.relative_to(REPO)):sha256_file(f) for f in
            [Path(__file__),Path(__file__).with_name('soc_reference.py'),Path(__file__).with_name('freeze_h123.py')]},
        h123_freeze=before,pretrained_loaded=False,precision_policy='32-bit logical tensor accounting; not proven CoreML internal precision'))
    after=verify(args.freeze);assert before==after
    write_json(out/'verification.json',dict(valid=True,h123_unchanged=after,
        scope='code accounting, protocol checks, in-sample fit; not independent hardware validation',energy_unavailable=True))
    write_json(out/'completion.json',dict(status='complete',aggregate_sha256=sha256_file(out/'aggregate.csv')))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
