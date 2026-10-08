"""Repeated copies of the same ENDF threshold carry no intermediate state."""
import numpy as np
import pytest
from kika.algebra import evaluate,split_at_discontinuities,join_pieces


@pytest.mark.parametrize('ordinates',[[1.,2.,2.,2.,8.],[1.,2.,3.,3.,8.],[1.,2.,2.,3.,8.]])
def test_identical_copies_preserve_values_laws_and_limits(ordinates):
    x=[1.,2.,2.,2.,4.];laws=[2,1,5,5]
    parts=split_at_discontinuities(x,ordinates,laws)
    assert len(parts)==2
    queries=[1.,1.5,np.nextafter(2.,1.),2.,np.nextafter(2.,4.),3.,4.]
    joined=join_pieces(parts)
    np.testing.assert_allclose(evaluate(*joined,queries),evaluate(x,ordinates,laws,queries),rtol=1e-15)
    np.testing.assert_array_equal(parts[0][2],[2]);np.testing.assert_array_equal(parts[1][2],[5])


def test_distinct_intermediate_values_are_not_guessed_away():
    with pytest.raises(ValueError,match='isolated'):split_at_discontinuities([1.,2.,2.,2.,4.],[1.,2.,3.,4.,8.],2)


def test_two_point_identical_boundary_keeps_existing_region_structure():
    assert len(split_at_discontinuities([1.,2.,2.,4.],[1.,2.,2.,8.],2))==2
