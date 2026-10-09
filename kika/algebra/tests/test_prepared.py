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


@pytest.mark.parametrize('law',[1,2,3,4,5])
def test_constant_span_is_proven_from_source_panels(law):
    table=prepare_evaluator([1.,2.,3.,4.],[2.,2.,2.,5.],law)
    assert table.constant_on(1.2,2.8)==2.
    assert table.constant_on(2.8,3.2) is None
    assert table.constant_on(.5,2.) is None
    q=np.linspace(1.2,2.8,101)
    np.testing.assert_array_equal(table(q),np.full(len(q),2.))


def test_constant_proof_keeps_steps_bumps_and_numeric_limits():
    bump=prepare_evaluator([1.,2.,3.],[0.,1.,0.],2)
    assert bump.constant_on(1.,3.) is None  # Equal endpoint samples do not suffice.
    step=prepare_evaluator([1.,2.,2.,3.],[0.,0.,1.,1.],2)
    assert step.constant_on(1.,np.nextafter(2.,1.))==0.
    assert step.constant_on(1.,2.) is None
    assert step.constant_on(2.,3.)==1.
    assert prepare_evaluator([1.,2.],[-0.,-0.],2).constant_on(1.,2.) is None
    assert prepare_evaluator([1e-300,1e300],[2.,2.],3).constant_on(1e-200,1e200) is None
    with pytest.raises(ValueError):step.constant_on(2.,2.)


def test_mixed_source_linear_span_uses_exact_panel_arithmetic():
    x=np.array([1.,2.,2.,4.,8.]);y=np.array([3.,5.,7.,9.,11.])
    table=prepare_evaluator(x,y,[5,2,2,4])
    q=np.array([3.,2.,np.nextafter(4.,2.),4.,2.5])
    expected=(9.-7.)/(4.-2.)*(q-2.)+7.
    np.testing.assert_array_equal(table(q),expected)
    np.testing.assert_array_equal(table.right_limit(q),expected)
    # At the repeated boundary the left limit belongs to the log-log panel.
    expected[1]=5.
    np.testing.assert_array_equal(table.left_limit(q),expected)


def test_shared_linear_reader_preserves_columns_nodes_and_shapes():
    from kika.algebra.prepared import _prepare_shared_linear_evaluator
    x=np.array([1.,2.,2.,4.,8.]);y=np.array([3.,5.,7.,9.,11.])
    ys=np.column_stack((y,y*2,y*.125));laws=[5,2,2,2]
    batch=_prepare_shared_linear_evaluator(x,ys,laws)
    for q in (3.,np.array([8.,2.,3.,4.]),np.array([[2.,3.],[4.,8.]]),np.array([])):
        expected=np.stack([evaluate(x,col,laws,q) for col in ys.T],axis=-1)
        for cells in (1,5,65536):np.testing.assert_array_equal(batch(q,max_cells=cells),expected)
    for q in ([1.5,3.],[.5,3.],[3.,9.],[np.nan]):assert batch(q) is None
    expected=batch([2.,3.]);ys[:]=100.
    np.testing.assert_array_equal(batch([2.,3.]),expected)
    with pytest.raises(ValueError):batch._y.setflags(write=True)
    with pytest.raises(ValueError):_prepare_shared_linear_evaluator([1.,2.],[1.,2.],2)
    with pytest.raises(ValueError):batch([3.],max_cells=0)
