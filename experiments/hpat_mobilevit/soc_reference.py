"""Separate, explicitly empirical SoC runtime reference; never imported by H1-H3.

Not A14 microarchitecture reconstruction. Published graph latency can fit one
scale per execution backend, not identify throughput/bandwidth/fusion/fallback.
Logical tensor traffic is a proxy, not measured DRAM bytes. Energy stays unknown.
"""
import math
from dataclasses import dataclass, asdict, replace

VIEWS={'view','_unsafe_view','transpose','permute','t','unbind','expand','detach','alias','slice','select','as_strided','empty'}
COPY={'clone','cat','copy_','_to_copy','upsample_bilinear2d'}
VECTOR={'native_batch_norm':6,'native_layer_norm':6,'_softmax':5,'silu':8,'silu_':8,
        'add':1,'mul':1,'mean':1,'gelu':8,'relu':1,'relu_':1}


@dataclass(frozen=True)
class SoCProfile:
    backend: str
    dense_ops_per_s: float
    logical_bandwidth_bytes_per_s: float
    depthwise_efficiency: float = .1
    attention_efficiency: float = .25
    vector_efficiency: float = .1
    dispatch_s: float = 2e-6
    runtime_scale: float = 1.
    logical_storage_bits: int = 32

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name!='backend' and (not math.isfinite(value) or value<=0):raise ValueError(name)
        if self.logical_storage_bits not in (16,32):raise ValueError('No claimed INT8 equivalence to the full-precision paper')


def priors(backend):
    # Explicit uncalibrated shape-model priors, NOT Apple hardware specifications.
    return SoCProfile(backend,100e9 if backend=='cpu' else 1e12,20e9 if backend=='cpu' else 32e9)


def operator_features(r,bits):
    kind=r['op_type']
    if kind in VIEWS:return dict(category='metadata',operations=0,logical_bytes=0)
    logical_bytes=(sum(math.prod(s) for s in r['input_shapes'])+r['output_elements'])*bits/8
    if r.get('macs',0):
        cat='depthwise' if kind=='Conv2d' and r.get('groups',1)>1 else 'attention' if kind=='MatMul' else 'dense'
        ops=2*r['macs']
    elif kind in COPY:cat='layout';ops=0
    elif kind in VECTOR:
        cat='vector';ops=VECTOR[kind]*(math.prod(r['input_shapes'][0]) if kind=='mean' else r['output_elements'])
    else:raise ValueError(f'Unsupported operator {kind}')
    return dict(category=cat,operations=ops,logical_bytes=logical_bytes)


def simulate(records,profile):
    rows=[];time=0.;done={}
    for r in records:
        if any(dep not in done for dep in r['dependencies']):raise ValueError('Trace not in dependency order')
        f=operator_features(r,profile.logical_storage_bits);cat=f['category']
        efficiency={'depthwise':profile.depthwise_efficiency,'attention':profile.attention_efficiency,
                    'vector':profile.vector_efficiency}.get(cat,1.)
        compute=f['operations']/(profile.dense_ops_per_s*efficiency)
        memory=f['logical_bytes']/profile.logical_bandwidth_bytes_per_s
        dispatch=0 if cat=='metadata' else profile.dispatch_s
        # Serialized launch + intra-op roofline bound. Not a causal DMA timeline.
        raw=dispatch+max(compute,memory)
        elapsed=raw*profile.runtime_scale
        start=max([time,*[done[d] for d in r['dependencies']]])
        time=start+elapsed;done[r['op_id']]=time
        rows.append(dict(op_id=r['op_id'],op_type=r['op_type'],module_path=r['module_path'],**f,
            compute_proxy_s=compute,logical_memory_proxy_s=memory,dispatch_prior_s=dispatch,
            raw_duration_s=raw,start_s=start,end_s=time,duration_s=elapsed,
            resource=f'{profile.backend}_runtime_proxy',energy_j=None))
    assert math.isclose(sum(r['duration_s'] for r in rows),time,rel_tol=1e-12)
    return dict(latency_s=time,energy_j=None,average_power_w=None,operators=len(rows),
        profile=asdict(profile),model_level='empirical calibrated graph/shape model; not cycle-accurate SoC',
        operator_placement='unidentified; backend denotes graph execution target, not proven per-op ANE residency'),rows


def calibrate(records,manifest,target_ms,profile):
    if manifest['timm_model']!='mobilevit_xs' or manifest['input_shape']!=[1,3,256,256]:
        raise ValueError('Table 11 anchor requires XS256 batch1, not old XS224 or XXS192')
    if not math.isfinite(target_ms) or target_ms<=0:raise ValueError('Invalid target')
    raw,_=simulate(records,replace(profile,runtime_scale=1.))
    fitted=replace(profile,runtime_scale=target_ms*1e-3/raw['latency_s'])
    return fitted,dict(target_ms=target_ms,raw_latency_ms=raw['latency_s']*1e3,
        fitted_runtime_scale=fitted.runtime_scale,free_parameters=1,training_points=1,
        heldout_validation_points=0,status='in-sample alignment only; no prediction-accuracy claim',
        non_identifiability='One graph target cannot identify operator efficiency, fusion, bandwidth, launch or CPU fallback')
