"""Optional contraction versus the independent direct-level solver."""
import numpy as np
import pytest
from kika.processing.resonances import _rm_acceleration as backend
from kika.processing.resonances.reich_moore import RMGroup,RMLevel,solve_collision


def problem(c=3):
    rng=np.random.default_rng(926)
    er=np.linspace(1.,70.,73)
    gamma=rng.uniform(.02,.2,len(er))
    a=rng.normal(0,.1,(len(er),c))
    e=np.r_[np.geomspace(.1,100.,151),er]
    f=rng.uniform(.1,1.3,(len(e),c))
    return e,er,gamma,a,f


@pytest.mark.parametrize('amplitude',[.1,1000.])
def test_envelope_guard_against_independent_sum_outside_all_poles(amplitude):
    if backend._native is None:pytest.skip('optional extension unavailable')
    from kika.processing.resonances.reich_moore import level_matrix
    n=5000;c=3;er=np.linspace(1.,70.,n);gamma=np.full(n,.1)
    a=np.full((n,c),amplitude);e=np.array([100.,1e12]);f=np.ones((len(e),c))
    r=np.empty((len(e),c,c),complex);unsafe=np.zeros(len(e),np.uint8)
    backend._native.matrices(len(e),n,c,e,er,gamma,a.ravel(),f.ravel(),r.view(float).ravel(),unsafe)
    reciprocal=1/(er[None,:]-e[:,None]-.5j*gamma[None,:])
    expected=level_matrix(reciprocal,a)
    np.testing.assert_allclose(r,expected,rtol=4e-13,atol=0.)
    envelope=np.sum((abs(reciprocal.real)+abs(reciprocal.imag))@ (a*a),axis=1)
    np.testing.assert_array_equal(unsafe,envelope>1e8)
    assert unsafe[1]==0
    assert unsafe[0]==(amplitude==1000.)


def test_prepared_coefficients_are_immutable_and_cached():
    from copy import deepcopy
    import pickle
    levels=(RMLevel(1.,.5,.1,.2,0.,neutron_amplitude=.3,fission_amplitudes=(0.,.4)),
            RMLevel(2.,.5,.2,.3,0.,neutron_amplitude=.4,fission_amplitudes=(0.,-.5)))
    group=RMGroup(0,None,None,levels)
    arrays=group.kernel_data
    assert arrays is group.kernel_data
    np.testing.assert_array_equal(arrays[2],[[.3,.4],[.4,-.5]])
    for array in arrays:
        with pytest.raises(ValueError):array.flags.writeable=True
    for copied in (deepcopy(group),pickle.loads(pickle.dumps(group))):
        for actual,expected in zip(copied.kernel_data,arrays):
            np.testing.assert_array_equal(actual,expected)
            with pytest.raises(ValueError):actual.flags.writeable=True
    assert RMGroup(0,None,None,()).kernel_data[2].shape==(0,1)


@pytest.mark.parametrize('c',[1,2,3])
@pytest.mark.parametrize('entrance',[0,1,2])
@pytest.mark.parametrize('width',[None,1e-30,0.])
def test_native_matches_direct(c,entrance,width):
    if entrance>=c:pytest.skip('entrance absent')
    if backend._native is None:pytest.skip('optional extension unavailable')
    e,er,gamma,a,f=problem(c)
    if width is not None:gamma[:]=width
    kwargs=dict(entrance=entrance,reduced=a,channel_factors=f,return_absorption=True)
    expected=solve_collision(e,er,gamma,None,**kwargs)
    diagnostics={}
    actual=solve_collision(e,er,gamma,None,diagnostics,absorption_rtol=1e-8,**kwargs)
    assert diagnostics['rm_native_energies']==len(e)
    np.testing.assert_allclose(actual[0],expected[0],rtol=3e-11,atol=1e-12)
    np.testing.assert_allclose(actual[1],expected[1],rtol=3e-11,atol=0.)


@pytest.mark.parametrize('case',['undamped','coincident','tiny','large','weak','missing','strict'])
def test_reference_limits_retained(case,monkeypatch):
    e,er,gamma,a,f=problem()
    if case=='undamped':gamma[:]=0.
    if case=='coincident':
        gamma[:2]=0.;er[1]=er[0];e=np.r_[e,er[0]];f=np.vstack([f,f[0]])
    if case=='tiny':gamma[:]=1e-200
    if case=='large':a*=1e25
    if case=='weak':f*=1e-200
    if case=='missing':monkeypatch.setattr(backend,'_native',None)
    kwargs=dict(reduced=a,channel_factors=f,return_absorption=True)
    with np.errstate(all='ignore'):
        try:expected=solve_collision(e,er,gamma,None,**kwargs)
        except (FloatingPointError,np.linalg.LinAlgError) as error:
            with pytest.raises(type(error)):
                solve_collision(e,er,gamma,None,absorption_rtol=1e-8,**kwargs)
            return
        actual=solve_collision(e,er,gamma,None,absorption_rtol=1e-10 if case=='strict' else 1e-8,**kwargs)
    for x,y in zip(actual,expected):np.testing.assert_allclose(x,y,rtol=3e-11,atol=1e-12)


def test_independent_reference_never_accelerates(monkeypatch):
    e,er,gamma,a,f=problem()
    def forbidden(*args,**kwargs):raise AssertionError('witness used acceleration')
    monkeypatch.setattr(backend,'solve',forbidden)
    solve_collision(e,er,gamma,None,reduced=a,channel_factors=f,return_absorption=True)


@pytest.mark.parametrize('reason',['width','missing','rejected'])
def test_streaming_fallback_limits_reference_arrays(reason,monkeypatch):
    import kika.processing.resonances.reich_moore as rm
    e,er,gamma,a,f=problem()
    if reason=='width':gamma[:]=1e-200
    expected=solve_collision(e,er,gamma,None,reduced=a,channel_factors=f,return_absorption=True)
    if reason=='missing':monkeypatch.setattr(backend,'_native',None)
    if reason=='rejected':monkeypatch.setattr(backend,'solve',lambda *args,**kwargs:None)
    sizes=[];original=rm.level_matrix
    def observed(reciprocal,reduced):
        sizes.append(len(reciprocal));return original(reciprocal,reduced)
    monkeypatch.setattr(rm,'level_matrix',observed)
    work=256*1024
    actual=solve_collision(e,er,gamma,None,reduced=a,channel_factors=f,
        work_bytes=work,return_absorption=True,absorption_rtol=1e-8)
    assert sizes and max(sizes)<=backend.reference_block_size(len(er),a.shape[1],work)
    np.testing.assert_allclose(actual[0],expected[0],rtol=3e-11,atol=1e-12)
    np.testing.assert_allclose(actual[1],expected[1],rtol=3e-11,atol=0.)


def test_native_streaming_workspace_plan_retains_reference_limit(monkeypatch):
    from types import SimpleNamespace
    from kika.processing.resonances.prepare import energy_block_size
    from kika.processing.resonances.grid import ReconstructionConvergenceError
    _,er,gamma,a,_=problem()
    levels=tuple(RMLevel(float(e),.5,.1,float(g),0.,neutron_amplitude=float(row[0]),
        fission_amplitudes=tuple(row[1:])) for e,g,row in zip(er,gamma,a))
    region=SimpleNamespace(approximation='ReichMoore',groups=(RMGroup(0,None,None,levels),))
    reference=energy_block_size(region,work_bytes=1024*1024)
    if backend._native is not None:
        assert energy_block_size(region,work_bytes=1024*1024,absorption_rtol=1e-8)>reference
    monkeypatch.setattr(backend,'_native',None)
    assert energy_block_size(region,work_bytes=1024*1024,absorption_rtol=1e-8)==reference
    with pytest.raises(ReconstructionConvergenceError):
        energy_block_size(region,work_bytes=1,absorption_rtol=1e-8)


@pytest.mark.parametrize('bad',['short','float32','strided','readonly','unaligned','dimensions'])
def test_native_rejects_invalid_buffers(bad):
    if backend._native is None:pytest.skip('optional extension unavailable')
    e,er,gamma,a,f=problem();n=len(er);c=a.shape[1]
    out=np.empty((len(e),c,c),complex).view(float).ravel();flags=np.zeros(len(e),np.uint8)
    args=[len(e),n,c,e,er,gamma,a.ravel(),f.ravel(),out,flags]
    if bad=='short':args[3]=e[:-1]
    if bad=='float32':args[3]=e.astype('f')
    if bad=='strided':args[3]=np.tile(e,2)[::2]
    if bad=='readonly':out.flags.writeable=False
    if bad=='unaligned':args[3]=np.ndarray(e.shape,dtype='d',buffer=bytearray(e.nbytes+1),offset=1)
    if bad=='dimensions':args[0]=2**62
    with pytest.raises((ValueError,BufferError)):backend._native.matrices(*args)


def test_direct_positive_capture_and_empty_buffers():
    if backend._native is None:pytest.skip('optional extension unavailable')
    e,er,gamma,a,f=problem();c=a.shape[1]
    r=np.empty((len(e),c,c),complex);flags=np.zeros(len(e),np.uint8)
    backend._native.matrices(len(e),len(er),c,e,er,gamma,a.ravel(),f.ravel(),r.view(float).ravel(),flags)
    assert not flags.any()
    matrix=np.eye(c)[None,:,:]-1j*r
    rhs=np.zeros((len(e),c,1),complex);rhs[:,0]=1.
    y=np.ascontiguousarray(np.linalg.solve(matrix,rhs)[:,:,0]);cap=np.empty(len(e))
    backend._native.absorption(len(e),len(er),c,e,er,gamma,a.ravel(),f.ravel(),y.view(float).ravel(),cap)
    expected=solve_collision(e,er,gamma,None,reduced=a,channel_factors=f,return_absorption=True)[1]
    np.testing.assert_allclose(cap,expected,rtol=3e-11,atol=1e-12)
    empty=np.empty(0)
    backend._native.matrices(0,0,1,empty,empty,empty,empty,empty,empty,np.empty(0,np.uint8))


@pytest.mark.parametrize('c',[1,2,3])
@pytest.mark.parametrize('entrance',[0,1,2])
def test_pivoted_native_channel_solve(c,entrance):
    if entrance>=c:pytest.skip('entrance absent')
    if backend._native is None or not hasattr(backend._native,'channel_solve'):
        pytest.skip('optional channel solver unavailable')
    rng=np.random.default_rng(415)
    matrix=rng.normal(size=(211,c,c))+1j*rng.normal(size=(211,c,c))
    if c>1:matrix[0]=np.eye(c)[::-1]
    matrix*=np.geomspace(1e-30,1e7,len(matrix))[:,None,None]
    rhs=np.zeros((len(matrix),c,1),complex);rhs[:,entrance]=1.
    output=np.empty((len(matrix),c),complex)
    assert backend._native.channel_solve(len(matrix),c,entrance,
        matrix.view(float).ravel(),output.view(float).ravel())
    expected=np.linalg.solve(matrix,rhs)[:,:,0]
    np.testing.assert_allclose(output,expected,rtol=4e-13,atol=0.)
    residual=matrix@output[:,:,None]-rhs
    ratio=abs(residual)/(np.linalg.norm(matrix,axis=(1,2))[:,None,None]*np.linalg.norm(output,axis=1)[:,None,None]+1.)
    assert np.max(ratio)<1e-14


@pytest.mark.parametrize('kind',['singular','nonfinite','large'])
def test_native_channel_solve_refuses_unsafe_matrix(kind):
    if backend._native is None or not hasattr(backend._native,'channel_solve'):
        pytest.skip('optional channel solver unavailable')
    matrix=np.eye(3,dtype=complex)[None].copy()
    matrix[0,0,0]={'singular':0.,'nonfinite':np.nan,'large':1e100}[kind]
    output=np.empty((1,3),complex)
    assert not backend._native.channel_solve(1,3,0,matrix.view(float).ravel(),output.view(float).ravel())


def test_older_binary_without_channel_solver(monkeypatch):
    from types import SimpleNamespace
    if backend._native is None:pytest.skip('optional extension unavailable')
    e,er,gamma,a,f=problem(3)
    expected=backend.solve(e,er,gamma,None,reduced=a,channel_factors=f,
        return_absorption=True,absorption_rtol=1e-8)
    monkeypatch.setattr(backend,'_native',SimpleNamespace(
        matrices=backend._native.matrices,absorption=backend._native.absorption))
    actual=backend.solve(e,er,gamma,None,reduced=a,channel_factors=f,
        return_absorption=True,absorption_rtol=1e-8)
    for x,y in zip(actual,expected):np.testing.assert_allclose(x,y,rtol=3e-11,atol=1e-12)


@pytest.mark.parametrize('kind',['short','float32','strided','readonly','unaligned','dimensions','entrance'])
def test_native_channel_solver_checks_buffers(kind):
    if backend._native is None or not hasattr(backend._native,'channel_solve'):
        pytest.skip('optional channel solver unavailable')
    matrix=np.eye(3,dtype=complex)[None].view(float).ravel()
    output=np.empty(6);args=[1,3,0,matrix,output]
    if kind=='short':args[3]=matrix[:-1]
    if kind=='float32':args[3]=matrix.astype('f')
    if kind=='strided':args[3]=np.tile(matrix,2)[::2]
    if kind=='readonly':output.flags.writeable=False
    if kind=='unaligned':args[3]=np.ndarray(matrix.shape,dtype='d',buffer=bytearray(matrix.nbytes+1),offset=1)
    if kind=='dimensions':args[0]=2**62
    if kind=='entrance':args[2]=3
    with pytest.raises((ValueError,BufferError)):backend._native.channel_solve(*args)
