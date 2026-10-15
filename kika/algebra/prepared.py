"""Owned, validated table snapshots for repeated evaluation.

No global cache and no reference to mutable source arrays. Values and limits
use the same evaluator as the ordinary algebra entry points.
"""
from dataclasses import dataclass, field
import math

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
    _linlin: bool = field(init=False, repr=False)

    def __post_init__(self):
        # This is an owned immutable snapshot. Discover its law once rather
        # than scan a potentially large source table on every small query.
        object.__setattr__(self, '_linlin', bool(np.all(self._laws == 2)))

    def __reduce__(self):
        # NumPy restores writable arrays when copied/pickled. Rebuild the
        # owned snapshot and its law certificate together.
        return prepare_evaluator,(self._x,self._y,self._laws)

    def _read(self, q, side, outside):
        if outside not in OUTSIDE:
            raise ValueError(f'outside must be one of {OUTSIDE}, got {outside!r}')
        return _read_checked(self._x, self._y, self._laws, q, side, outside,
                             linlin=self._linlin)

    def __call__(self, q, outside='zero'):
        return self._read(q, 'point', outside)

    def left_limit(self, q, outside='zero'):
        return self._read(q, 'left', outside)

    def right_limit(self, q, outside='zero'):
        return self._read(q, 'right', outside)

    def constant_on_many(self, lows, highs):
        """:meth:`constant_on` for many spans at once: the constant, or NaN
        where it returns None. Spans over a log-law panel are passed to
        :meth:`constant_on` itself; the rest are decided by prefix counts of
        value changes and signed zeros over each span's nodes, which is what
        constant_on checks for lin-lin and histogram panels."""
        lows=np.asarray(lows,dtype=float);highs=np.asarray(highs,dtype=float)
        x,y=self._x,self._y
        out=np.full(len(lows),np.nan)
        if len(x)<2:return out
        laws=np.broadcast_to(self._laws,(len(x)-1,))
        if np.any(~np.isfinite(lows)) or np.any(~np.isfinite(highs)) or np.any(lows>=highs):
            raise ValueError('constant span requires finite low < high')
        inside=(lows>=x[0])&(highs<=x[-1])
        j0=np.maximum(np.searchsorted(x,lows,'right')-1,0);j1=np.searchsorted(x,highs,'left')
        n0=np.searchsorted(x,lows,'right');n1=np.searchsorted(x,highs,'right')
        # constant_on reads y over panels [j0, j1) (both ends) and nodes [n0, n1):
        # one contiguous run of indices, from first to last.
        has_panels=j1>j0;has_nodes=n1>n0
        first=np.where(has_panels,j0,n0);last=np.maximum(np.where(has_panels,j1,j0),np.where(has_nodes,n1-1,0))
        empty=~(has_panels|has_nodes)
        changes=np.r_[0,np.cumsum(y[1:]!=y[:-1])]
        negative=np.r_[0,np.cumsum((y==0)&np.signbit(y))]
        other=np.r_[0,np.cumsum(~np.isin(laws,(1,2)))]
        safe=np.clip(first,0,len(y)-1);top=np.clip(last,0,len(y)-1)
        value=y[safe]
        equal=changes[top]==changes[safe]
        signed=(value==0)&(negative[top+1]-negative[safe]>0)
        linear=other[np.maximum(j1,j0)]==other[j0]
        decided=inside&~empty&linear
        good=decided&equal&np.isfinite(value)&~signed
        out[good]=value[good]
        # Spans over log-law panels: constant_on's further checks, all
        # spans at once -- every log-x panel inside widens (finite ratio
        # above 1), and the table reads the constant at both ends.
        logs=np.flatnonzero(inside&~empty&~linear&equal&np.isfinite(value)&~signed)
        if len(logs):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                ratio=x[1:]/x[:-1]
            bad=np.r_[0,np.cumsum(np.isin(laws,(3,5))&~(np.isfinite(ratio)&(ratio>1.)))]
            logs=logs[bad[np.maximum(j1,j0)[logs]]==bad[j0[logs]]]
            if len(logs):
                at=value[logs]
                logs=logs[(self(lows[logs])==at)&(self(highs[logs])==at)]
                out[logs]=value[logs]
        return out

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
        # The panels x[j] < high, x[j+1] > low and the nodes low < x <= high,
        # found by bisection: a table is read once per span, and a material
        # cut into thousands of spans asked thousands of times.
        j0=max(int(np.searchsorted(x,low,'right'))-1,0);j1=int(np.searchsorted(x,high,'left'))
        n0=int(np.searchsorted(x,low,'right'));n1=int(np.searchsorted(x,high,'right'))
        panels=slice(j0,max(j0,j1))
        if j1-j0<=8 and n1-n0<=8:
            # A narrow span: the same checks on Python floats.
            law=np.broadcast_to(laws,(len(x)-1,))
            values=[float(v) for v in y[j0:max(j0,j1)]]+[float(v) for v in y[j0+1:max(j0,j1)+1]]+[float(v) for v in y[n0:n1]]
            if not values:return None
            value=values[0]
            if not math.isfinite(value) or any(v!=value for v in values):return None
            if value==0. and any(math.copysign(1.,v)<0 for v in values):return None
            used=[int(v) for v in law[panels]]
            if all(v in (1,2) for v in used):
                # Equal ends under lin-lin or histogram: every point inside is
                # that value exactly (y1+0*t is y1), so no evaluation is needed.
                return value
            for j,v in zip(range(j0,max(j0,j1)),used):
                if v in (3,5):
                    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                        ratio=x[j+1]/x[j]
                    if not np.isfinite(ratio) or ratio<=1.:return None
            if not np.all(self(np.array([low,high]))==value):return None
            return value
        # At the left boundary the point value is the last repeated node;
        # its discarded left limit does not belong to this closed span.
        values=np.r_[y[:-1][panels],y[1:][panels],y[n0:n1]]
        if not len(values):return None
        value=float(values[0])
        if not np.isfinite(value) or not np.all(values==value):return None
        if value==0. and np.any(np.signbit(values)):return None
        logarithmic=np.isin(np.broadcast_to(laws,(len(x)-1,))[panels],(3,5))
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            ratio=x[1:][panels][logarithmic]/x[:-1][panels][logarithmic]
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
