"""Grouped diagnostics preserve panel order and repeated owner labels."""
import math
import pytest
from kika.processing.doppler import core, prepare_curve, DopplerContext, SupportPolicy

@pytest.mark.parametrize('labels',[
    ['region']*12,
    ['a','b','a','c']*3,
    ['región','λ','región','λ']*3,
])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('terms',[8,32,80,81])
def test_component_runs_match_individual_panels(labels,packed,terms):
    native=core._native
    panels=tuple((float(i),float(i+1),float((-1)**i*(i+1)),float(i+2),
        'linlin',label.encode().decode(),1.) for i,label in enumerate(labels))
    expected={}
    for i,panel in enumerate(panels):
        result=native.integrate(panels,i,i+1,4.,.3,-8.,8.,terms,10000)
        for owner,value in result[2].items():expected[owner]=expected.get(owner,0.)+value
    source=native.prepare_panels(panels)[0] if packed else panels
    result=native.integrate(source,0,len(panels),4.,.3,-8.,8.,terms,10000)
    assert result[2]==expected
    assert math.isfinite(result[0]) and math.isfinite(result[1])

def test_prepared_panels_share_region_label_identity():
    p=prepare_curve((([.1,.2,.4,.8],[1.,2.,3.,4.]),),context=DopplerContext(56.),
        source_temperature=0.,support=SupportPolicy('zero','zero'))
    assert len({id(panel[5]) for panel in p._panels})==1
    assert all(panel[5]=='region-0' for panel in p._panels)

def test_unicode_subclass_keeps_dictionary_key_semantics():
    class Owner(str):
        __hash__=object.__hash__
    labels=[Owner('same') for _ in range(4)]
    source=tuple((float(i),float(i+1),1.,1.,'constant',label,0.) for i,label in enumerate(labels))
    expected={}
    native=core._native
    for i in range(4):
        expected.update(native.integrate(source,i,i+1,2.,.3,-8.,8.,80,10000)[2])
    result=native.integrate(source,0,4,2.,.3,-8.,8.,80,10000)
    assert len(expected)==4 and result[2]==expected
