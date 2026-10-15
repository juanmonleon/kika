"""Compiled Clenshaw preserves the NumPy reference's finite-domain arithmetic."""
import numpy as np
import pytest

native=pytest.importorskip("kika.algebra._polynomial_native")


@pytest.mark.parametrize("degree",[0,1,2,7,16,33,64])
@pytest.mark.parametrize("shape",[(0,),(1,),(17,),(3,7)])
def test_legendre_values_are_bit_identical_to_numpy(degree,shape):
    rng=np.random.default_rng(1703+degree)
    c=np.ascontiguousarray(rng.normal(size=degree+1))
    x=np.ascontiguousarray(rng.uniform(-1,1,size=shape));out=np.empty_like(x)
    assert native.legendre_values(c,x,out)
    expected=np.polynomial.legendre.legval(x,c)
    np.testing.assert_array_equal(out.view(np.uint64),expected.view(np.uint64))


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
