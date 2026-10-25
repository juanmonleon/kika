"""Compiled Clenshaw preserves the NumPy reference's finite-domain arithmetic.

The kernel replays ``numpy.polynomial.legendre.legval`` operation for operation,
in the order NumPy has used since 2.3 (numpy a7e6113 forms each recurrence
ratio first: ``c1*((nd-1)/nd)``). NumPy <= 2.2 multiplied first and divided
after, ``(c1*(nd-1))/nd``, which differs in the last bit, and kika does not
bound NumPy (CI resolves it from poetry.lock, the app pins its own). So the
bit-for-bit gate is against :func:`_legval_numpy23`, a transcription of that
recurrence, on every NumPy; and where the installed NumPy is >= 2.3 a second
test holds the transcription to ``legval`` itself, so a future change of order
upstream fails here instead of passing unnoticed.
"""
import numpy as np
import pytest

native=pytest.importorskip("kika.algebra._polynomial_native")

_NUMPY_23=tuple(int(p) for p in np.__version__.split(".")[:2])>=(2,3)


def _legval_numpy23(x,c):
    """``legval(x, c)`` for 1-D float64 ``c``, as NumPy >= 2.3 computes it."""
    if len(c)==1:return c[0]+0*x
    if len(c)==2:return c[0]+c[1]*x
    nd=len(c);c0=c[-2];c1=c[-1]
    for i in range(3,len(c)+1):
        tmp=c0;nd=nd-1
        c0=c[-i]-c1*((nd-1)/nd)
        c1=tmp+c1*x*((2*nd-1)/nd)
    return c0+c1*x


def _case(degree,shape):
    rng=np.random.default_rng(1703+degree)
    c=np.ascontiguousarray(rng.normal(size=degree+1))
    x=np.ascontiguousarray(rng.uniform(-1,1,size=shape))
    return c,x


@pytest.mark.parametrize("degree",[0,1,2,7,16,33,64])
@pytest.mark.parametrize("shape",[(0,),(1,),(17,),(3,7)])
def test_legendre_values_are_bit_identical_to_numpy(degree,shape):
    c,x=_case(degree,shape);out=np.empty_like(x)
    assert native.legendre_values(c,x,out)
    expected=np.asarray(_legval_numpy23(x,c),dtype=float)
    np.testing.assert_array_equal(out.view(np.uint64),expected.view(np.uint64))
    if not _NUMPY_23:
        # The older order: the same values to rounding, not to the bit.
        np.testing.assert_allclose(out,np.polynomial.legendre.legval(x,c),
                                   rtol=1e-12,atol=1e-12*np.abs(c).sum())


@pytest.mark.skipif(not _NUMPY_23,reason="NumPy < 2.3 evaluates legval in the older order")
@pytest.mark.parametrize("degree",[0,1,2,7,16,33,64])
def test_the_transcribed_reference_is_numpy_legval_bit_for_bit(degree):
    c,x=_case(degree,(3,7))
    expected=np.polynomial.legendre.legval(x,c)
    np.testing.assert_array_equal(np.asarray(_legval_numpy23(x,c)).view(np.uint64),
                                  expected.view(np.uint64))


@pytest.mark.parametrize("point",[1.1,float("nan"),float("inf")])
def test_exceptional_domain_requests_reference_without_modifying_output(point):
    out=np.full(1,7.)
    assert not native.legendre_values(np.array([1.,.2]),np.array([point]),out)
    np.testing.assert_array_equal(out,[7.])


def test_buffer_contract_refuses_alias_mismatch_and_wrong_dtype():
    c=np.array([1.,.2]);x=np.linspace(-1,1,5)
    with pytest.raises(ValueError,match="overlap"):
        native.legendre_values(c,x,x)
    with pytest.raises(ValueError,match="matching"):
        native.legendre_values(c,x,np.empty(2))
    with pytest.raises(ValueError,match="float64"):
        native.legendre_values(c.astype(np.float32),x,np.empty_like(x))
    with pytest.raises((ValueError,BufferError)):
        native.legendre_values(c,x[::2],np.empty(3))
    readonly=np.empty_like(x);readonly.setflags(write=False)
    with pytest.raises((ValueError,BufferError)):
        native.legendre_values(c,x,readonly)


def test_unaligned_buffers_request_reference_instead_of_unsafe_access():
    x=np.ndarray((5,),dtype=np.float64,buffer=bytearray(41),offset=1);x[:]=np.linspace(-1,1,5)
    assert not x.flags.aligned
    out=np.full(5,7.)
    assert not native.legendre_values(np.array([1.,.2]),x,out)
    np.testing.assert_array_equal(out,7.)
