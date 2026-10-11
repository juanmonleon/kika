"""Analytic panel integration with bounded thermal support and local moments.

Gaussian Taylor coefficients are integrated term by term against a degree <=4
polynomial. Narrow panels use their local coordinate, avoiding subtraction of
nearby global energies. Low-z moments evolve sum/difference together. Error
estimates are conservative diagnostics, not interval-arithmetic certificates.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
import math
import numbers

import numpy as np
from scipy.special import erfc
from kika._constants import BOLTZMANN_CONSTANT

KB = BOLTZMANN_CONSTANT * 1e6
ENGINE_VERSION = "doppler-analytic-3"
EPS = np.finfo(float).eps
try:
    from importlib import import_module
    _native = import_module(__package__ + '._doppler_native')
except ModuleNotFoundError as exc:
    if exc.name != __package__ + '._doppler_native':
        raise
    _native = None


class DopplerInputError(ValueError):
    def __init__(self, category, message, *, energy=None, budget=None):
        super().__init__(message)
        self.category, self.energy, self.budget = category, energy, budget


class UnsupportedDopplerError(DopplerInputError):
    pass


class DopplerConvergenceError(DopplerInputError):
    pass


def _quantity(v, name, zero=False):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise DopplerInputError("invalid-quantity", name)
    v = float(v)
    if not math.isfinite(v) or (v < 0 if zero else v <= 0):
        raise DopplerInputError("invalid-quantity", name)
    return v


def _integer(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral) or v <= 0:
        raise DopplerInputError("invalid-options", name)


def _immutable(v):
    a = np.ascontiguousarray(v, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


# Default five moment reductions. Immutable, shared 3200-byte workspace.
_MOMENT_WEIGHTS = _immutable(1. / (np.arange(5)[:, None] + np.arange(80)[None, :] + 1.))


@dataclass(frozen=True)
class DopplerContext:
    atomic_weight_ratio: float
    mass_origin: str = "explicit"

    def __post_init__(self):
        object.__setattr__(self,'atomic_weight_ratio',_quantity(self.atomic_weight_ratio, "atomic_weight_ratio"))
        if not isinstance(self.mass_origin, str) or not self.mass_origin:
            raise DopplerInputError("missing-mass-origin", "mass_origin")


@dataclass(frozen=True)
class SupportPolicy:
    lower: str
    upper: str

    def __post_init__(self):
        if self.lower not in ("zero", "constant", "one_over_v") or self.upper not in ("zero", "constant"):
            raise UnsupportedDopplerError("invalid-support", "explicit support required")


@dataclass(frozen=True)
class DopplerOptions:
    rtol: float = 1e-3
    atol: float = 1e-8
    kernel_share: float = .15
    tail_share: float = .10
    table_share: float = .45
    serialization_share: float = .20
    max_points: int = 200000
    max_iterations: int = 40
    block_size: int = 2048
    max_work_bytes: int = 64 * 1024 * 1024
    max_panels: int = 200000
    max_series_terms: int = 80
    max_radius: int = 26
    threshold_policy: str = "convolution"
    threshold: float | None = None
    backend: str = "auto"

    def __post_init__(self):
        if not isinstance(self.backend, str) or self.backend not in ("auto", "python", "native"):
            raise DopplerInputError("invalid-options", "backend must be auto, python or native")
        for n in ("rtol", "atol", "kernel_share", "tail_share", "table_share", "serialization_share"):
            object.__setattr__(self,n,_quantity(getattr(self, n), n))
        if self.rtol >= 1 or self.kernel_share + self.tail_share + self.table_share + self.serialization_share > 1:
            raise DopplerInputError("invalid-options", "rtol/shares")
        for n in ("block_size", "max_work_bytes", "max_panels", "max_series_terms", "max_radius", "max_points", "max_iterations"):
            _integer(getattr(self, n), n)
            object.__setattr__(self,n,int(getattr(self,n)))
        if self.max_series_terms < 8 or self.max_radius > 26:
            raise DopplerInputError("invalid-options", "series/radius limits")
        if self.max_points < 2:
            raise DopplerInputError("invalid-options", "max_points >=2 required")
        if self.threshold_policy not in ("convolution", "zero_below"):
            raise UnsupportedDopplerError("invalid-threshold-policy", self.threshold_policy)
        if self.threshold_policy == "zero_below" and self.threshold is None:
            raise DopplerInputError("missing-threshold", "explicit threshold required")
        if self.threshold is not None:
            object.__setattr__(self,'threshold',_quantity(self.threshold, "threshold", True))


@dataclass(frozen=True)
class PreparedDopplerCurve:
    regions: tuple
    context: DopplerContext
    support: SupportPolicy
    source_temperature: float = 0.
    negative_policy: str = "reject"
    source_temperature_origin: str = "explicit"

    def __post_init__(self):
        if not isinstance(self.context, DopplerContext) or not isinstance(self.support, SupportPolicy):
            raise DopplerInputError("invalid-context", "context/support types")
        object.__setattr__(self,'source_temperature',_quantity(self.source_temperature, "source_temperature", True))
        if self.negative_policy not in ("reject", "preserve_signed"):
            raise DopplerInputError("invalid-sign-policy", self.negative_policy)
        if not isinstance(self.source_temperature_origin, str) or not self.source_temperature_origin:
            raise DopplerInputError("missing-temperature-origin", "source_temperature_origin")
        result = []
        for xs, ys in self.regions:
            try:
                x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
            except (TypeError, ValueError) as exc:
                raise DopplerInputError("invalid-shape", str(exc)) from exc
            if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
                raise DopplerInputError("invalid-shape", "two paired points per region")
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise DopplerInputError("nonfinite-input", "finite arrays required")
            if x[0] < 0 or np.any(np.diff(x) <= 0):
                raise DopplerInputError("invalid-grid", "strictly increasing nonnegative grid")
            if self.negative_policy == "reject" and np.any(y < 0):
                raise DopplerInputError("negative-input", "explicit preserve_signed required")
            if result and result[-1][0][-1] != x[0]:
                raise UnsupportedDopplerError("noncontiguous-regions", "gaps/overlaps")
            result.append((_immutable(x), _immutable(y)))
        if not result:
            raise DopplerInputError("empty-input", "no regions")
        object.__setattr__(self, "regions", tuple(result))
        # Sorted intervals preserve both sides of discontinuities. Build once.
        panels = []
        x, y = result[0]
        if x[0] > 0 and self.support.lower != "zero":
            panels.append((0., float(x[0]), float(y[0]), float(y[0]),
                           self.support.lower, "lower", float(x[0])))
        for n, (x, y) in enumerate(result):
            panels.extend((float(x[i]), float(x[i+1]), float(y[i]), float(y[i+1]),
                           "linlin", f"region-{n}", 0.) for i in range(len(x)-1))
        x, y = result[-1]
        if self.support.upper == "constant":
            panels.append((float(x[-1]), math.inf, float(y[-1]), float(y[-1]), "constant", "upper", 0.))
        object.__setattr__(self, "_panels", tuple(panels))
        object.__setattr__(self, "_starts", _immutable([p[0] for p in panels]))
        object.__setattr__(self, "_ends", _immutable([p[1] for p in panels]))
        m = max(float(np.max(np.abs(y))) for _, y in result)
        x, y = result[0]
        k = abs(float(y[0])) * math.sqrt(float(x[0])) if self.support.lower == "one_over_v" else 0.
        object.__setattr__(self, "_envelope", (m, k))

    @property
    def source_hash(self):
        h = sha256(repr((self.context, self.support, self.source_temperature,
                         self.negative_policy, self.source_temperature_origin)).encode())
        for x, y in self.regions:
            h.update(x.tobytes()); h.update(y.tobytes())
        return h.hexdigest()

    def evaluate_source(self, energy):
        x, y = self.regions[0]
        if energy < x[0]:
            if self.support.lower == "zero": return 0.
            if self.support.lower == "constant": return float(y[0])
            return float(y[0]) * math.sqrt(float(x[0]) / energy)
        x, y = self.regions[-1]
        if energy > x[-1]:
            return float(y[-1]) if self.support.upper == "constant" else 0.
        for x, y in reversed(self.regions):
            if x[0] <= energy <= x[-1]:
                return float(np.interp(energy, x, y))
        raise DopplerInputError("invalid-domain", "uncovered energy")


def prepare_curve(form, *, context, source_temperature, support,
                  negative_policy="reject", source_temperature_origin="explicit"):
    """Snapshot linlin model forms or tuple of (eV-array, barn-array) regions.

    Array input is explicitly canonical eV/b. Model forms require axes, converted
    independently of codecs. PoPs/style/URR and graph checks belong to the suite
    preparer, not this pointwise D2 entry.
    """
    if isinstance(form, tuple):
        regions = form
    else:
        from kika.nuclear_data.model import XYs1d, Regions1d
        from kika.nuclear_data.model.units import conversion_factor
        children = form.function1ds if isinstance(form, Regions1d) else [form]
        regions = []
        for child in children:
            if not isinstance(child, XYs1d) or getattr(child.interpolation, 'value', None) != "lin-lin":
                raise UnsupportedDopplerError("unsupported-form", "linlin XYs1d/Regions1d only")
            axes = child.axes or getattr(form, "axes", None)
            if axes is None:
                raise DopplerInputError("missing-units", "axes required")
            try:
                fx = conversion_factor(axes.byIndex(1).unit, "eV")
                fy = conversion_factor(axes.byIndex(0).unit, "b")
            except (ValueError, KeyError, IndexError) as exc:
                raise DopplerInputError("invalid-units", str(exc)) from exc
            regions.append((child.xs * fx, child.ys * fy))
    return PreparedDopplerCurve(tuple(regions), context, support, source_temperature,
                                negative_policy, source_temperature_origin)


@dataclass(frozen=True)
class DopplerEvaluation:
    energies: np.ndarray
    values: np.ndarray
    kernel_error_estimates: np.ndarray
    tail_bounds: np.ndarray
    threshold_removed: np.ndarray
    report: dict

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=True) if copy else np.asarray(self.values, dtype=dtype)


def _tail(z, h, m, k, radius):
    e = math.exp(-radius * radius)
    i0 = math.sqrt(math.pi) * float(erfc(radius)) / 2
    i1 = e / 2
    i2 = radius * e / 2 + i0 / 2
    # Algebra scaled to avoid z² overflow and improve large-z behaviour.
    value = 2 / math.sqrt(math.pi) * (m * (i2/z/z + 2*i1/z + i0) + (k/h/z)*(i0+i1/z))
    if not math.isfinite(value):
        raise DopplerConvergenceError("unresolvable-scale", "tail envelope overflow")
    return max(np.nextafter(0., 1.), value * (1+64*EPS))


def _moments(s, d, z, terms):
    """Integral u^j [Gaussian difference] du, j<=4, u in [0,1].

    Analytic recurrence (Taylor ODE), not sampled numerical quadrature. At low
    z evolve sum H and difference D without subtracting two near-equal moments.
    Local thermal interval <=1/4 ensures rapidly decreasing coefficient tails.
    """
    plus = math.exp(-s*s)
    t = z+s
    minus = plus * math.exp(-4*z*t)
    if z < .125:
        dif = [plus * (-math.expm1(-4*z*t))]
        sums = [plus + minus]
        for n in range(1, terms):
            a = -2*t*d*dif[-1] + 2*z*d*sums[-1]
            b = -2*t*d*sums[-1] + 2*z*d*dif[-1]
            if n > 1:
                a -= 2*d*d*dif[-2]; b -= 2*d*d*sums[-2]
            dif.append(a/n); sums.append(b/n)
    else:
        gp, gm = [plus], [minus]
        for n in range(1, terms):
            a, b = -2*s*d*gp[-1], -2*(s+2*z)*d*gm[-1]
            if n > 1:
                a -= 2*d*d*gp[-2]; b -= 2*d*d*gm[-2]
            gp.append(a/n); gm.append(b/n)
        dif = [a-b for a, b in zip(gp, gm)]
    if terms == 80:
        moments = (_MOMENT_WEIGHTS @ np.asarray(dif)).tolist()
    else:
        moments = [math.fsum(v/(n+j+1) for n, v in enumerate(dif)) for j in range(5)]
    # Analytic coefficient-tail stopping diagnostic plus floating accumulation.
    remainder = math.fsum(abs(v) for v in dif[-4:]) * 4
    norm = math.fsum(abs(v) for v in dif)
    if terms == 80:
        # Charge cached reciprocal rounding and the dot-product accumulation
        # in addition to the existing recurrence/roundoff diagnostic. Weights
        # lie in (0,1]; gamma_80 + two rounded operations is < 83*EPS.
        remainder += 83*EPS*norm
    return moments, remainder, norm


def _point(curve, energy, h, options):
    x = math.sqrt(energy); z = x/h
    if not math.isfinite(z) or z <= 0 or not math.isfinite(z*z) or z*z == 0:
        raise DopplerConvergenceError("unresolvable-scale", "scaled energy", energy=energy)
    m, k = curve._envelope
    radius = min(8, options.max_radius)
    while _tail(z, h, m, k, radius) > options.tail_share * options.atol:
        radius += 1
        if radius > options.max_radius:
            raise DopplerConvergenceError("tail-budget-exhausted", "thermal radius limit", energy=energy)
    lower_s = max(-z, -float(radius)); upper_s = float(radius)
    # searchsorted on prepared panel edges, no output-energy × input-grid matrix.
    elow = max(0., x+h*lower_s)**2; ehigh = (x+h*upper_s)**2
    first = int(np.searchsorted(curve._ends, elow, side="left"))
    last = int(np.searchsorted(curve._starts, ehigh, side="right"))
    if options.backend != "python" and _native is not None:
        try:
            value, error, components, count = _native.integrate(
                curve._panels, first, last, energy, h, lower_s, upper_s,
                options.max_series_terms, options.max_panels)
        except OverflowError as exc:
            raise DopplerConvergenceError("resources", str(exc), energy=energy) from exc
        except ArithmeticError as exc:
            raise DopplerConvergenceError("unresolvable-scale", str(exc), energy=energy) from exc
        budget = options.kernel_share*(options.atol+options.rtol*abs(value))
        if error > budget:
            raise DopplerConvergenceError("kernel-budget-exhausted", "analytic error estimate exceeds budget", energy=energy, budget=budget)
        if curve.negative_policy == "reject" and value < 0:
            raise DopplerConvergenceError("negative-output", "recalculate with stable higher precision", energy=energy)
        return value, error, _tail(z, h, m, k, radius), components, count, radius
    values, errors, components = [], [], {}
    count = 0
    for panel_index in range(first, last):
        lo, hi, v0, v1, kind, owner, anchor = curve._panels[panel_index]
        # Exact zero polynomial: no moments, uncertainty or thermal work. This
        # is a source proof, not a cutoff based on a small sampled value.
        if v0 == 0. and v1 == 0.:
            components.setdefault(owner,0.)
            continue
        ylo = math.sqrt(lo)
        slo = (lo-energy)/(h*(ylo+x))
        shi = math.inf if math.isinf(hi) else (hi-energy)/(h*(math.sqrt(hi)+x))
        left, right = max(lower_s, slo), min(upper_s, shi)
        if left >= right: continue
        ds = math.inf if math.isinf(hi) else (hi-lo)/(h*(math.sqrt(hi)+ylo))
        # Preserve narrow source-panel widths when subtraction of global s loses ulps.
        narrow = ds < 1
        qleft, qright = (max(0., (lower_s-slo)/ds), min(1., (upper_s-slo)/ds)) if narrow else (0., 1.)
        width = (qright-qleft)*ds if narrow else right-left
        subdivisions = max(1, math.ceil(width/.25))
        if count + subdivisions > options.max_panels:
            raise DopplerConvergenceError("resources", "analytic subpanel limit", energy=energy)
        for i in range(subdivisions):
            d = width/subdivisions
            s = slo + (qleft+i*(qright-qleft)/subdivisions)*ds if narrow else left+i*d
            yr, dr = 1+s/z, d/z
            if kind == "one_over_v":
                poly = [v0*math.sqrt(anchor)/x*yr, v0*math.sqrt(anchor)/x*dr]
            else:
                if kind == "linlin":
                    if narrow:
                        q = qleft+i*(qright-qleft)/subdivisions
                        dy = ds*h
                        de0 = q*dy*(2*ylo+q*dy)
                        sig0 = v0+(v1-v0)*(de0/(hi-lo))
                    else:
                        de0 = (energy-lo)+h*s*(2*x+h*s)
                        sig0 = v0+(v1-v0)*(de0/(hi-lo))
                    slope = (v1-v0)/(hi-lo)
                    sig = [sig0, slope*(2*(x+h*s)*h*d), slope*(h*d)**2]
                else:
                    sig = [v0]
                poly = np.convolve([yr*yr, 2*yr*dr, dr*dr], sig).tolist()
            moments, remainder, norm = _moments(s, d, z, options.max_series_terms)
            factor = d/math.sqrt(math.pi)
            value = factor*math.fsum(p*v for p, v in zip(poly, moments))
            err = factor*math.fsum(abs(p) for p in poly)*(remainder+256*EPS*norm)
            if not math.isfinite(value) or not math.isfinite(err):
                raise DopplerConvergenceError("unresolvable-scale", "panel overflow", energy=energy)
            values.append(value); errors.append(err)
            components[owner] = components.get(owner, 0.) + value
            count += 1
    value = math.fsum(values)
    error = math.fsum(errors) + 8*EPS*math.fsum(abs(v) for v in values)
    budget = options.kernel_share*(options.atol+options.rtol*abs(value))
    if error > budget:
        raise DopplerConvergenceError("kernel-budget-exhausted", "analytic error estimate exceeds budget", energy=energy, budget=budget)
    if curve.negative_policy == "reject" and value < 0:
        raise DopplerConvergenceError("negative-output", "recalculate with stable higher precision", energy=energy)
    return value, error, _tail(z, h, m, k, radius), components, count, radius


def _selected_backend(options):
    if options.backend == "native" and _native is None:
        raise UnsupportedDopplerError("native-unavailable", "build the native Doppler extension or select backend='python'")
    return "native" if options.backend != "python" and _native is not None else "python"


def evaluate_broadened(prepared, energies, *, temperature, options=None):
    """Evaluate complete supported input on requested energies, no tabulation.

    Returns immutable arrays and diagnostics. The resource limit controls working
    scratch, not retained inputs/results or process RSS. Fail without partial result.
    ``options.backend`` selects auto (compiled when available), native (required),
    or the Python reference. The report records the backend actually used.
    Native integration uses the same Taylor recurrence with a coefficient-tail
    stopping diagnostic charged to the kernel budget; no fast-math or thinning.
    """
    if not isinstance(prepared, PreparedDopplerCurve):
        raise DopplerInputError("invalid-prepared", "prepare_curve required")
    options = options or DopplerOptions()
    if not isinstance(options, DopplerOptions):
        raise DopplerInputError("invalid-options", "DopplerOptions required")
    backend = _selected_backend(options)
    temperature = _quantity(temperature, "temperature", True)
    dt = temperature-prepared.source_temperature
    if dt < 0: raise DopplerInputError("cooling", "T1<T0")
    raw = np.asarray(energies)
    if raw.dtype.kind not in "fiu" or (isinstance(energies, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in energies)):
        raise DopplerInputError("invalid-energy", "real numeric energies required, no booleans")
    try: energies = np.asarray(energies, dtype=float)
    except (ValueError, TypeError) as exc: raise DopplerInputError("invalid-energy", str(exc)) from exc
    if energies.ndim != 1 or not len(energies) or np.any(~np.isfinite(energies)) or np.any(energies <= 0):
        raise DopplerInputError("invalid-energy", "nonempty positive finite energy vector")
    scratch = 4096 + options.max_series_terms*128 + options.block_size*8 + options.max_panels*128
    if scratch > options.max_work_bytes:
        raise DopplerConvergenceError("resources", "max_work_bytes below scratch requirement")
    a = KB*dt/prepared.context.atomic_weight_ratio
    if dt > 0 and (a <= 0 or not math.isfinite(a)):
        raise DopplerConvergenceError("unresolvable-increment", "thermal scale")
    h = math.sqrt(a)
    values, errors, tails, removed = [], [], [], []
    totals, count, maxradius = {}, 0, 0
    for start in range(0, len(energies), options.block_size):
        for energy in energies[start:start+options.block_size]:
            if dt == 0:
                v, err, tail, comp, n, radius = prepared.evaluate_source(float(energy)), 0., 0., {"identity": prepared.evaluate_source(float(energy))}, 0, 0
            else:
                v, err, tail, comp, n, radius = _point(prepared, float(energy), h, options)
            drop = v if options.threshold_policy == "zero_below" and energy < options.threshold else 0.
            values.append(0. if options.threshold_policy == "zero_below" and energy < options.threshold else v)
            errors.append(err); tails.append(tail); removed.append(drop)
            count += n; maxradius = max(maxradius, radius)
            for owner, contribution in comp.items(): totals[owner] = totals.get(owner, 0.)+contribution
    report = dict(engine=ENGINE_VERSION, backend=backend, source_hash=prepared.source_hash,
                  temperature_source=prepared.source_temperature, temperature_final=temperature,
                  delta_temperature=dt, atomic_weight_ratio=prepared.context.atomic_weight_ratio,
                  mass_origin=prepared.context.mass_origin, kB_eV_per_K=KB,
                  source_temperature_origin=prepared.source_temperature_origin,
                  support=dict(lower=prepared.support.lower, upper=prepared.support.upper),
                  accuracy_scope="input_table", empirical_verification=False, global_error_bound=False,
                  error_estimates_certified=False, options=asdict(options),
                  units=dict(energy="eV", cross_section="b", temperature="K"),
                  threshold_policy=options.threshold_policy, threshold=options.threshold,
                  model="free-gas-plus-threshold-convention" if options.threshold_policy=="zero_below" else "free-gas",
                  negative_policy=prepared.negative_policy, analytic_subpanels=count,
                  max_thermal_radius=maxradius, scratch_estimate_bytes=scratch,
                  retained_result_bytes=len(energies)*5*8, components_sum_over_probes=totals,
                  source_minimum=min(float(np.min(y)) for _, y in prepared.regions),
                  result_minimum=min(values),
                  distribution_temperature_treatment="not-processed")
    return DopplerEvaluation(*(_immutable(a) for a in (energies, values, errors, tails, removed)), report)
