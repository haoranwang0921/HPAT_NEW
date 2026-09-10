import pytest
from experiments.hpat_mobilevit.energy_ledger import split_components, stage_energies


def test_component_split_and_closure():
    c = {'core_dac_0_i1-DAC_2': 3.57e-12, 'core_mzm_0_i0-LT_MZM': .45e-12,
         'core_adc_0_i2-ADC_SAR_1': 1.48e-12, 'core_tia_0_i7-TIA_1': .3e-12,
         'node_photodetector_0_i1-LT_PD': .11e-12, 'core_on_chip_laser_0_i3-laser': 0.0}
    groups = split_components(c)
    k = dict(dynamic_energy_j=sum(c.values()), stage_energy_j={s: sum(v.values()) for s,v in groups.items()})
    e = stage_energies(k)
    assert e['dac_encode'] == pytest.approx(4.02e-12, abs=1e-24)
    assert e['adc_convert'] == pytest.approx(1.89e-12, abs=1e-24)
    assert e['optical_compute'] == 0
    assert sum(e.values()) == pytest.approx(sum(c.values()), abs=1e-24)


@pytest.mark.parametrize('name', ['unknown', 'core_mrr_weight_0_i6-HPAT_MRR', 'core_dac_0_i8-DAC_2'])
def test_reject_unmapped_and_double_counted_devices(name):
    with pytest.raises(ValueError):
        split_components({name: 1e-12})


def test_reject_unclosed_energy():
    with pytest.raises(ValueError):
        stage_energies(dict(dynamic_energy_j=2e-12, stage_energy_j=dict(dac_encode=1e-12, optical_compute=0, adc_convert=0)))
