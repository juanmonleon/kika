import numpy as np
import pytest
from kika.algebra import compress_flat, evaluate, left_limit, right_limit


@pytest.mark.parametrize('laws',[1,2,[2,2,1,1,2,2]])
def test_only_exact_constant_spans_are_removed(laws):
    x=np.arange(1.,8.);y=np.array([0.,0.,0.,4.,4.,4.,4.])
    cx,cy,cl=compress_flat(x,y,laws)
    q=np.linspace(0.,8.,401)
    for read in (evaluate,left_limit,right_limit):
        np.testing.assert_array_equal(read(cx,cy,cl,q),read(x,y,laws,q))
    assert len(cx)<len(x)
    assert not np.shares_memory(cx,x) and not np.shares_memory(cy,y)


def test_steps_law_boundaries_and_signed_zero_survive():
    x=np.array([1.,2.,2.,3.,4.,5.,6.]);y=np.array([0.,0.,1.,1.,1.,-0.,0.])
    cx,cy,cl=compress_flat(x,y,[2,2,2,1,2,2])
    np.testing.assert_array_equal(cx,x)
    np.testing.assert_array_equal(np.signbit(cy),np.signbit(y))


def test_log_laws_and_nonconstant_nodes_are_not_thinned():
    x=np.arange(1.,5.);y=np.ones(4)
    for law in (3,4,5):
        np.testing.assert_array_equal(compress_flat(x,y,law)[0],x)
    y[2]=np.nextafter(1.,2.)
    np.testing.assert_array_equal(compress_flat(x,y,2)[0],x)
