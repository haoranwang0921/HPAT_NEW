import pytest
from experiments.hpat_mobilevit.soc_reference import priors,simulate,calibrate,SoCProfile

def record():
    return dict(op_id='a',dependencies=[],op_type='Linear',module_path='x',macs=128,
                input_shapes=[[1,8],[8,16]],output_elements=16)

def test_energy_unknown_and_time_closure():
    result,events=simulate([record()],priors('cpu'))
    assert result['energy_j'] is None and result['average_power_w'] is None
    assert events[0]['end_s']==result['latency_s']>0

def test_protocol_guard():
    with pytest.raises(ValueError):calibrate([record()],dict(timm_model='mobilevit_xs',input_shape=[1,3,224,224]),7.28,priors('neural_engine'))

def test_fit_is_one_training_point():
    p,m=calibrate([record()],dict(timm_model='mobilevit_xs',input_shape=[1,3,256,256]),7.28,priors('neural_engine'))
    assert simulate([record()],p)[0]['latency_s']==pytest.approx(.00728)
    assert m['heldout_validation_points']==0

def test_unknown_ops_and_dependencies_rejected():
    r=record();r['dependencies']=['missing']
    with pytest.raises(ValueError):simulate([r],priors('cpu'))
    r=record();r.update(op_type='unknown',macs=0)
    with pytest.raises(ValueError):simulate([r],priors('cpu'))

def test_invalid_hardware():
    with pytest.raises(ValueError):SoCProfile('cpu',-1,1)
