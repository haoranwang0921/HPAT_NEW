"""Local FP32 CPU shape microbenchmarks, NOT mobile INT8 D0 calibration."""
import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer
from .run import write_json, write_csv
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


def signature(r):
    keys = ['op_type','M','K','N','batch_repetitions','input_shapes','output_shape',
            'groups','kernel_size','stride','padding','dilation','bias_elements']
    return {k:r[k] for k in keys if k in r}


def build(r):
    if r['op_type'] == 'Conv2d':
        x = torch.randn(r['input_shapes'][0])
        w = torch.randn(r['input_shapes'][1])
        bias = torch.randn(r['output_shape'][1]) if r.get('bias_elements',0) else None
        return lambda: F.conv2d(x,w,bias,stride=r['stride'],padding=r['padding'],
                               dilation=r['dilation'],groups=r['groups'])
    m,k,n,b = (r[t] for t in ['M','K','N','batch_repetitions'])
    if r['op_type'] == 'Linear':
        x,w = torch.randn(m,k),torch.randn(n,k)
        bias = torch.randn(n) if r.get('bias_elements',0) else None
        assert b == 1
        return lambda: F.linear(x,w,bias)
    x,w = torch.randn(b,m,k),torch.randn(b,k,n)
    return lambda: torch.bmm(x,w)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    assert args.threads > 0
    args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20260906)
    torch.set_num_threads(args.threads)
    manifest,records=load_manifest_jsonl(args.trace)
    groups={}
    for r in records:
        if r['op_type'] not in {'Conv2d','Linear','MatMul'}: continue
        key=json.dumps(signature(r),sort_keys=True)
        groups.setdefault(key,[]).append(r)
    rows=[]; raw=[]
    with torch.inference_mode():
        for key,rs in groups.items():
            r=rs[0]; fn=build(r); y=fn()
            assert list(y.shape)==r['output_shape'] and torch.isfinite(y).all()
            measurements=[Timer('fn()',globals={'fn':fn},num_threads=args.threads)
                          .blocked_autorange(min_run_time=0.1) for _ in range(3)]
            med=[x.median for x in measurements]
            sid=hashlib.sha256(key.encode()).hexdigest()[:16]
            row=dict(shape_id=sid,op_type=r['op_type'],groups=r.get('groups',1),
                M=r['M'],K=r['K'],N=r['N'],batch=r['batch_repetitions'],
                occurrences=len(rs),macs=r['macs'],median_us=statistics.median(med)*1e6,
                min_repeat_us=min(med)*1e6,max_repeat_us=max(med)*1e6,
                effective_fp32_gops=2*r['macs']/statistics.median(med)/1e9,
                analytical_d0_int8_us=2*r['macs']/8e12*1e6)
            rows.append(row)
            raw.append(dict(shape_id=sid,signature=json.loads(key),op_ids=[x['op_id'] for x in rs],
                repeats=[dict(number_per_run=x.number_per_run,raw_times_s=x.raw_times) for x in measurements]))
            print(f'{len(rows)}/{len(groups)} {r["op_type"]}: {row["median_us"]:.3f} us',flush=True)
    write_csv(args.output/'shape_timings.csv',rows)
    write_json(args.output/'raw_measurements.json',raw)
    write_json(args.output/'manifest.json',dict(torch=torch.__version__,platform=platform.platform(),
        processor=platform.processor(),threads=args.threads,trace_sha256=sha256_file(args.trace),
        script_sha256=sha256_file(Path(__file__)),trace_manifest=manifest,
        scope='FP32 CPU eager, synthetic contiguous operands, hot repeated operators, allocation included; '
              'not original-stride replay, not fused graph latency, not INT8 NPU, no energy calibration; '
              'do not add measured times to existing DRAM ledger (memory costs can overlap).',
        shapes=len(rows),covered_calls=sum(x['occurrences'] for x in rows),
        covered_macs=sum(x['macs']*x['occurrences'] for x in rows),
        isolated_weighted_sum_ms=sum(x['median_us']*x['occurrences'] for x in rows)/1000,
        d0_matrix_formula_sum_ms=sum(x['analytical_d0_int8_us']*x['occurrences'] for x in rows)/1000))


if __name__=='__main__': main()
