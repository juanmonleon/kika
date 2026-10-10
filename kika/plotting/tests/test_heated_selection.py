"""Heated selection and provenance follow the style, including custom labels."""
import numpy as np
import pytest
from kika.nuclear_data.model import ReactionSuite, Reaction, ReactionId, CrossSection, XYs1d, Evaluated, Heated, PhysicalQuantity, crossSectionAxes
from kika.plotting.quantities import plottable


@pytest.mark.parametrize('label', ['hot', 'warm600', 'custom'])
def test_arbitrary_heated_label_selects_heated_curve(label):
    s=ReactionSuite('consumer contract','n','Fe56')
    s.styles.add(Evaluated('eval'))
    s.styles.add(Heated(label,derivedFrom='eval',temperature=PhysicalQuantity(600.,'K')))
    s.reactions.append(Reaction(ReactionId('capture',ENDF_MT=102),CrossSection({
        'eval':XYs1d([.01,1.],[1.,2.],axes=crossSectionAxes(),label='eval'),
        label:XYs1d([.01,1.],[3.,4.],axes=crossSectionAxes(),label=label)})))
    hot=plottable(s,'cross_section',mt=102,form=label)
    cold=plottable(s,'cross_section',mt=102,form='eval')
    np.testing.assert_array_equal(hot.data.y,[3.,4.])
    np.testing.assert_array_equal(cold.data.y,[1.,2.])
    assert hot.data.provenance.state=='heated'
    assert cold.data.provenance.state=='evaluated'
