import numpy as np
import pytest
from kika.algebra.prepared import prepare_evaluator
from kika.algebra import evaluate, left_limit, right_limit


@pytest.mark.parametrize('law', [1, 2, 3, 4, 5])
def test_snapshot_matches_values_limits_steps_and_outside(law):
    x=np.array([1., 2., 2., 4.]);y=np.array([3., 5., 7., 9.])
    q=np.array([.5, 1., 1.5, np.nextafter(2., 0.), 2., 3., 4., 5.])
    table=prepare_evaluator(x,y,law)
    for outside in ('zero','hold'):
        for method,reference in ((table,evaluate),(table.left_limit,left_limit),
                                 (table.right_limit,right_limit)):
            np.testing.assert_array_equal(method(q,outside),reference(x,y,law,q,outside))
    expected=table(q);x[:]=100.;y[:]=-1.
    np.testing.assert_array_equal(table(q),expected)
    with pytest.raises(ValueError):table._x.setflags(write=True)
    with pytest.raises(ValueError):table(0.,outside='raise')
    with pytest.raises(ValueError):table(2.,outside='invalid')


def test_invalid_tables_still_raise_at_preparation():
    with pytest.raises(ValueError):prepare_evaluator([2.,1.],[1.,2.],2)
    with pytest.raises(ValueError):prepare_evaluator([1.,2.],[-1.,2.],4)
    with pytest.raises(ValueError):prepare_evaluator([1.,2.],[1.,2.],6)
    with pytest.raises(ValueError):prepare_evaluator([1.,2.],[[1.],[2.]],2)


def test_empty_and_scalar_snapshots():
    assert prepare_evaluator([],[],2)(1.)==0.
    assert prepare_evaluator([2.],[3.],2)(2.)==3.
