"""Owned, validated table snapshots for repeated evaluation.

No global cache and no reference to mutable source arrays. Values and limits
use the same evaluator as the ordinary algebra entry points.
"""
from dataclasses import dataclass
import numpy as np
from .laws import validate
from .evaluate import _read_checked, OUTSIDE


def _owned_readonly(array):
    # A bytes-backed view cannot be made writable with setflags(write=True).
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class _Evaluator:
    _x: object
    _y: object
    _laws: object

    def _read(self, q, side, outside):
        if outside not in OUTSIDE:
            raise ValueError(f'outside must be one of {OUTSIDE}, got {outside!r}')
        return _read_checked(self._x, self._y, self._laws, q, side, outside)

    def __call__(self, q, outside='zero'):
        return self._read(q, 'point', outside)

    def left_limit(self, q, outside='zero'):
        return self._read(q, 'left', outside)

    def right_limit(self, q, outside='zero'):
        return self._read(q, 'right', outside)

    def constant_on(self, low, high):
        """Return an exact constant on a closed span, or None if unproved.

        Inspect the source panels, including repeated nodes and both ends;
        equal sampled values alone cannot establish constancy. Conservative
        refusal is intentional, including signed zero and extreme log spans.
        """
        x,y,laws=self._x,self._y,self._laws
        if not np.isfinite(low) or not np.isfinite(high) or low>=high:
            raise ValueError('constant span requires finite low < high')
        if len(x)<2 or low<x[0] or high>x[-1]:return None
        panels=(x[:-1]<high)&(x[1:]>low)
        # At the left boundary the point value is the last repeated node;
        # its discarded left limit does not belong to this closed span.
        nodes=(x>low)&(x<=high)
        values=np.r_[y[:-1][panels],y[1:][panels],y[nodes]]
        if not len(values):return None
        value=float(values[0])
        if not np.isfinite(value) or not np.all(values==value):return None
        if value==0. and np.any(np.signbit(values)):return None
        logarithmic=panels&np.isin(laws,(3,5))
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            ratio=x[1:][logarithmic]/x[:-1][logarithmic]
            if np.any(~np.isfinite(ratio)) or np.any(ratio<=1.):return None
        if not np.all(self(np.array([low,high]))==value):return None
        return value


def prepare_evaluator(x, y, laws):
    """Validate and own a 1-d table once; return a callable snapshot.

    Later edits to the source do not affect the snapshot. Domain rules,
    repeated nodes, endpoint values, and one-sided limits match ``evaluate``,
    ``left_limit`` and ``right_limit``. All interpolation remains in algebra.
    """
    x, y, laws = validate(x, y, laws)
    if y.ndim != 1:
        raise ValueError('prepared evaluation requires one-dimensional ordinates')
    return _Evaluator(*(_owned_readonly(a) for a in (x, y, laws)))


@dataclass(frozen=True)
class _SharedLinearEvaluator:
    """One interval search for columns on the exact same source grid/laws.

    Only queries entirely in positive-width lin-lin panels are handled. None
    asks the caller to retain its ordinary evaluator for any other regime.
    Arithmetic and node overrides match panel_value, without regrouping sums.
    """
    _x: object
    _y: object
    _laws: object

    def __call__(self,q,*,max_cells=65536):
        if not isinstance(max_cells,int) or isinstance(max_cells,bool) or max_cells<1:
            raise ValueError('max_cells must be a positive integer')
        q=np.asarray(q,dtype=float);flat=q.reshape(-1)
        if not len(flat):return np.empty(q.shape+(self._y.shape[1],))
        if (np.any(~np.isfinite(flat)) or np.any(flat<self._x[0])
                or np.any(flat>self._x[-1])):return None
        k=np.minimum(np.searchsorted(self._x,flat,side='right')-1,len(self._x)-2)
        x1,x2=self._x[k],self._x[k+1]
        if np.any(self._laws[k]!=2) or np.any(x2<=x1):return None
        # The returned columns are retained output; bound the additional
        # ordinate matrices, especially on large source-diagnostic queries.
        value=np.empty((len(flat),self._y.shape[1]))
        rows=max(1,max_cells//self._y.shape[1])
        for begin in range(0,len(flat),rows):
            sl=slice(begin,begin+rows)
            a,b=self._y[k[sl]],self._y[k[sl]+1]
            with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
                v=(b-a)/(x2[sl]-x1[sl])[:,None]*(flat[sl]-x1[sl])[:,None]+a
            start=flat[sl]==x1[sl];end=flat[sl]==x2[sl]
            v[start]=a[start];v[end]=b[end]
            value[sl]=v
        return value.reshape(q.shape+(self._y.shape[1],))


def _prepare_shared_linear_evaluator(x,y,laws):
    """Own shared-grid columns for the optional linear-panel batch reader."""
    x,y,laws=validate(x,y,laws)
    if len(x)<2 or y.ndim!=2 or not y.shape[1]:
        raise ValueError('shared reader requires two nodes and one or more columns')
    return _SharedLinearEvaluator(*(_owned_readonly(v) for v in (x,y,laws)))
