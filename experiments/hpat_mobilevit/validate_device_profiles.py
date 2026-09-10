"""Streaming independent stage/weight/resource check for profile timelines."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from .run import write_json


def validate(path):
    resources=defaultdict(float); busy=defaultdict(float); weights={}; settled={}
    tokens={}; count=0
    with gzip.open(path,'rt',encoding='utf8') as f:
        for line in f:
            e=json.loads(line);count+=1;s=e['start_s'];end=e['end_s'];kind=e['event_type'];c=e.get('core')
            assert s+1e-14>=resources[e['resource']]
            resources[e['resource']]=max(resources[e['resource']],end)
            if kind=='programming':
                assert s+1e-14>=busy[c]
                weights[c]=e['weight_block'];settled[c]=end;busy[c]=end
            elif kind in ('dac_encode','optical_compute','adc_convert'):
                assert weights[c]==e['weight_block'] and s+1e-14>=settled[c]
                if 'token' in e:
                    key=(e['frame'],e['op_id'],c,e['weight_block'],e['sign'],e['token'])
                    if kind=='dac_encode':
                        assert key not in tokens and s+1e-14>=e['input_ready_s']
                        tokens[key]=(1,end)
                    elif kind=='optical_compute':
                        stage,prior=tokens[key];assert stage==1 and s+1e-14>=prior
                        tokens[key]=(2,end)
                    else:
                        stage,prior=tokens.pop(key);assert stage==2 and s+1e-14>=prior
                else: assert s+1e-14>=busy[c]
                busy[c]=max(busy[c],end)
            elif kind=='signed_reduce': assert s+1e-14>=busy[c]
    assert not tokens
    return dict(events=count,stage_dependencies=True,resource_exclusion=True,weight_exclusion=True,reduction_readiness=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    rows={str(f.relative_to(args.directory)):validate(f) for f in args.directory.glob('*/*/events.jsonl.gz')}
    write_json(args.directory/'independent_stage_verification.json',rows)
    print(json.dumps(rows))
