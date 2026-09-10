"""Immutable H1/H2/H3 snapshot; no mutation of source or historical results."""
import json
import zipfile
from pathlib import Path
from .run import ROOT, REPO, write_json
from joint_sim.trace_io import sha256_file


def freeze(out):
    out.mkdir(parents=True,exist_ok=False)
    result=REPO/'results/hpat_mobilevit/experiments_p2_sensitivity_v1'
    sources=list(ROOT.rglob('*.py'))+list(ROOT.rglob('*.json'))+list((REPO/'joint_sim').rglob('*.py'))
    sources+=list((REPO/'SimPhony/onnarchsim').rglob('*.py'))+list((REPO/'SimPhony/configs').rglob('*.yml'))
    sources+=[REPO/'LLMCompass/hardware_model/energy_model.py']
    files=sources+[p for i in (1,2,3) for p in (result/f'mapping_{i}').rglob('*') if p.is_file()]
    files += [REPO/'results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl']
    files=sorted(set(files)); hashes={str(f.relative_to(REPO)):sha256_file(f) for f in files}
    with zipfile.ZipFile(out/'snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for f in files: archive.write(f,str(f.relative_to(REPO)))
    write_json(out/'manifest.json',dict(scope='XXS192 H1/H2/H3 at 32x32/10GHz/25MiB/64GBs; not all XS/S',
        files_sha256=hashes,archive_sha256=sha256_file(out/'snapshot.zip'),
        modes={'H1':'mapping_1/linear','H2':'mapping_2/linear_pointwise','H3':'mapping_3/linear_pointwise_attention'},
        policy='New SoC backend is separate; no replacement of frozen hybrid electronic residual costs'))
    return verify(out)


def verify(out):
    m=json.loads((out/'manifest.json').read_text())
    assert sha256_file(out/'snapshot.zip')==m['archive_sha256']
    for name,digest in m['files_sha256'].items(): assert sha256_file(REPO/name)==digest,name
    return dict(valid=True,files=len(m['files_sha256']),archive_sha256=m['archive_sha256'])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(freeze(args.output))
