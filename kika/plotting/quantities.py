"""
Plottable quantities: one verb for every source.

::

    from kika.plotting import plottable, PlotBuilder

    items = [
        plottable(endf, 'cross_section', mt=2),              # ENDF tape
        plottable('Fe56.pendf', 'cross_section', mt=2),      # a path works too
        plottable(ace, 'cross_section', mt=2),               # ACE, heated
        plottable(experiment, 'cross_section'),              # EXFOR, with error bars
    ]
    builder = PlotBuilder(style='signature')
    for item in items:
        builder.add_data(item)
    fig = builder.set_units(x='MeV').build()

Whatever the source, :func:`plottable` returns a :class:`~kika.plotting.PlotItem`:
a curve in **canonical units** (eV, barn, b/sr) with its quantity, units and
:class:`~kika.plotting.Provenance` filled in, plus its uncertainty as a single
:class:`~kika.plotting.UncertaintyBand` (or ``None``). ``PlotBuilder`` converts to
display units and writes the axis labels from that metadata.

**Sources.** ENDF and PENDF tapes (``read_endf``), ACE files (``read_ace``),
``ReactionSuite`` (``kika.read``), EXFOR experiments, the ``CrossSection`` and
``AngularDistribution`` façades, the covariance classes, and file paths (the
format is sniffed from content). New ones are added with :func:`register_adapter`.

**Quantities** are listed in :data:`kika.plotting.units.QUANTITIES`:
``cross_section``, ``legendre_coefficient``, ``angular_distribution``,
``differential_cross_section``, ``relative_uncertainty``.

**What is derived, and how honestly it says so.** Some pairs are not stored in
the file and are computed (the :class:`~kika.plotting.Provenance` ``state`` and
the legend say which):

- ENDF cross section of a tape with resonances in MF2: MF3 alone is only the
  background. ``reconstructed='auto'`` (default) uses ``endf.pendf`` when the
  caller has set it and the background otherwise, labelled as such;
  ``reconstructed=True`` raises :class:`ReconstructionRequired` instead of
  falling back.
- ACE Legendre coefficients: projected from the tabulated f(mu) by quadrature.
- ENDF f(mu | E): summed from the Legendre series (exact).
- dsigma/dOmega = f(mu | E) sigma(E) / 2 pi: inherits sigma's state.

Nothing here reads a file twice or runs NJOY: reconstruction is the caller's
decision (``kika.processing.njoy_reconstruct``), and so is experimental resolution
(``resolution=`` on the angular quantities folds in energy explicitly).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .plot_data import PlotData, PlotItem, Provenance, UncertaintyBand
from .units import QUANTITIES, get_quantity

__all__ = [
    'plottable',
    'register_adapter',
    'supported_quantities',
    'NotPlottable',
    'ReconstructionRequired',
    'fold_in_energy',
]

BOLTZMANN_MEV_PER_K = 8.617333262e-11
TWO_PI = 2.0 * np.pi


class NotPlottable(ValueError):
    """The source cannot supply that quantity (or that reaction)."""


class ReconstructionRequired(NotPlottable):
    """The quantity needs resonance reconstruction the caller has not provided.

    Run NJOY (``kika.processing.njoy_reconstruct``), set ``endf.pendf`` to the
    result, and ask again. Or pass ``reconstructed=False`` to plot the MF3
    background knowingly.
    """


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

Adapter = Callable[..., PlotItem]
_ADAPTERS: Dict[Tuple[str, str], Adapter] = {}


def register_adapter(type_name: str, quantity: str) -> Callable[[Adapter], Adapter]:
    """
    Register ``fn(source, **selector) -> PlotItem`` for sources of ``type_name``.

    ``type_name`` is the class's ``module.QualName``. Matching walks the source's
    MRO, so subclasses are covered, and nothing is imported to register: a source
    type costs nothing until somebody plots one.
    """
    get_quantity(quantity)

    def decorator(fn: Adapter) -> Adapter:
        _ADAPTERS[(type_name, quantity)] = fn
        return fn
    return decorator


def _type_names(obj: Any) -> List[str]:
    return [f'{c.__module__}.{c.__qualname__}' for c in type(obj).__mro__]


def supported_quantities(source: Any) -> List[str]:
    """Quantities :func:`plottable` can produce for ``source`` (not per reaction)."""
    names = _type_names(source)
    return [q for q in QUANTITIES if any((n, q) in _ADAPTERS for n in names)]


def plottable(
    source: Any,
    quantity: str,
    *,
    label: Optional[str] = None,
    evaluation: Optional[str] = None,
    **selector: Any,
) -> PlotItem:
    """
    One curve of ``quantity`` from ``source``, in canonical units.

    Parameters
    ----------
    source
        A parsed file (ENDF, ACE, ReactionSuite, EXFOR experiment, covariance
        object, façade) or a path to one.
    quantity : str
        ``'cross_section'``, ``'legendre_coefficient'``, ``'angular_distribution'``,
        ``'differential_cross_section'`` or ``'relative_uncertainty'``.
    label : str, optional
        Legend entry. Default: built from the provenance
        (``'JEFF-4.0 Fe56 (n,el) · reconstructed'``).
    evaluation : str, optional
        Library name to show, when the file does not say it reliably (a PENDF
        written by a processing code often does not).
    **selector
        What to take from the source. Common keys: ``mt`` (reaction), ``order``
        (Legendre order), ``energy`` (incident energy in **eV**), ``uncertainty``
        (attach the covariance band), ``sigma`` (band width in standard
        deviations). Source-specific keys are documented on each adapter.

    Returns
    -------
    PlotItem

    Raises
    ------
    NotPlottable
        The source cannot give that quantity or that reaction.
    ReconstructionRequired
        The quantity needs reconstructed cross sections that are not available.
    """
    q = get_quantity(quantity)
    if isinstance(source, (str, os.PathLike)):
        source = _open(source)

    names = _type_names(source)
    adapter = next((_ADAPTERS[(n, quantity)] for n in names if (n, quantity) in _ADAPTERS), None)
    if adapter is None:
        supported = supported_quantities(source)
        raise NotPlottable(
            f"{type(source).__name__} cannot supply {quantity!r}"
            + (f"; it can supply {', '.join(supported)}" if supported else '')
        )

    item = adapter(source, **selector)
    data = item.data
    data.quantity = quantity
    data.x_unit = data.x_unit or q.x_unit
    data.y_unit = data.y_unit or q.y_unit
    data.interpolation = data.interpolation or q.interpolation
    if evaluation is not None:
        data.provenance = _replace(data.provenance, evaluation=evaluation)
    if label is not None:
        data.label = label
    elif not data.label:
        data.label = auto_label(quantity, data.provenance, **selector)
    return item


# -----------------------------------------------------------------------------
# Labels and provenance helpers
# -----------------------------------------------------------------------------

def _replace(prov: Optional[Provenance], **changes: Any) -> Provenance:
    from dataclasses import replace
    return replace(prov or Provenance(), **changes)


def format_energy(energy_ev: float) -> str:
    """``2000000.0`` -> ``'2 MeV'``."""
    e = float(energy_ev)
    if e >= 1e6:
        return f'{e / 1e6:.4g} MeV'
    if e >= 1e3:
        return f'{e / 1e3:.4g} keV'
    return f'{e:.4g} eV'


def auto_label(quantity: str, prov: Optional[Provenance], *, order: Optional[int] = None,
               energy: Optional[float] = None, **_: Any) -> str:
    """kika's one label convention: ``'<evaluation> <nuclide> <reaction> [L=..] [@ E] · <state>'``."""
    from kika._constants import mt_to_reaction

    prov = prov or Provenance()
    head = [prov.evaluation, prov.nuclide]
    if prov.reaction is not None:
        head.append(mt_to_reaction(prov.reaction))
    text = ' '.join(str(p) for p in head if p)
    if quantity == 'legendre_coefficient' and order is not None:
        text += f' L={order}'
    if quantity == 'relative_uncertainty' and order is not None:
        text += f' L={order}'
    if quantity in ('angular_distribution', 'differential_cross_section') and energy is not None:
        text += f' @ {format_energy(energy)}'
    suffix = prov.describe()
    if suffix:
        text = f'{text} · {suffix}' if text else suffix
    return text or quantity.replace('_', ' ')


_LIBRARY_PATTERNS = (
    (re.compile(r'ENDF\s*/?\s*B[-\s]?(?:(\d+)|([IVX]+))(?:\.(\d+))?', re.I), 'endfb'),
    (re.compile(r'JEFF[-\s]?(\d+(?:\.\d+)*)', re.I), 'JEFF'),
    (re.compile(r'JENDL[-\s]?(\d+(?:\.\d+)*)', re.I), 'JENDL'),
    (re.compile(r'TENDL[-\s]?(\d{4})', re.I), 'TENDL'),
    (re.compile(r'CENDL[-\s]?(\d+(?:\.\d+)*)', re.I), 'CENDL'),
    (re.compile(r'BROND[-\s]?(\d+(?:\.\d+)*)', re.I), 'BROND'),
    (re.compile(r'FENDL[-\s]?(\d+(?:\.\d+)*)', re.I), 'FENDL'),
    (re.compile(r'INDEN|INDL', re.I), 'INDEN'),
)

_ROMAN = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X'}
_ROMAN_BACK = {v: k for k, v in _ROMAN.items()}


def library_from_text(text: Optional[str]) -> Optional[str]:
    """The evaluated library named in free text (a TPID line, an ACE comment)."""
    if not text:
        return None
    for pattern, kind in _LIBRARY_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if kind == 'endfb':
            major = int(m.group(1)) if m.group(1) else _ROMAN_BACK.get(m.group(2).upper())
            if major is None:
                continue
            minor = m.group(3)
            return f'ENDF/B-{_ROMAN.get(major, major)}' + (f'.{minor}' if minor is not None else '')
        if kind == 'INDEN':
            return 'INDEN'
        return f'{kind}-{m.group(1)}'
    return None


#: ENDF-6 manual, MF1/MT451 NLIB.
_NLIB = {0: 'ENDF/B', 1: 'ENDF/A', 2: 'JEFF', 3: 'EFF', 5: 'CENDL', 6: 'JENDL',
         17: 'TENDL', 18: 'ROSFOND', 31: 'INDL/V', 32: 'INDL/A', 33: 'FENDL', 34: 'IRDF',
         35: 'BROND'}


def _library_from_mt451(h: Any) -> Optional[str]:
    nlib = getattr(h, '_nlib', None)
    nver = getattr(h, '_nver', None)
    lrel = getattr(h, '_lrel', None)
    name = _NLIB.get(nlib)
    if name is None:
        return None
    if name == 'ENDF/B' and nver:
        return f'ENDF/B-{_ROMAN.get(nver, nver)}.{lrel or 0}'
    if name in ('JEFF', 'CENDL', 'BROND') and nver:
        return f'{name}-{nver}.{lrel or 0}'
    if name == 'JENDL' and nver:
        return f'JENDL-{nver}' + (f'.{lrel}' if lrel else '')
    return name


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

def _open(path: Union[str, os.PathLike]) -> Any:
    """Read ``path`` as the format object kika uses for it (not the model)."""
    from kika._read import sniff_format

    path = Path(path)
    fmt = sniff_format(path)
    if fmt == 'ace':
        from kika.ace.parsers.parse_ace import read_ace
        obj = read_ace(str(path))
    elif fmt == 'endf':
        from kika.endf.read_endf import read_endf
        obj = read_endf(str(path))
        with open(path, 'r', errors='replace') as fh:
            obj._kika_tpid = fh.readline()[:66]
    else:
        from kika._read import read
        obj = read(path)
    try:
        obj._kika_source = str(path)
    except AttributeError:
        pass
    return obj


# -----------------------------------------------------------------------------
# Numerics shared by adapters
# -----------------------------------------------------------------------------

def _step_lookup(edges: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-bin ``values`` on ``edges`` read at points ``x`` (0 outside the edges)."""
    edges = np.asarray(edges, dtype=float)
    values = np.asarray(values, dtype=float)[: len(edges) - 1]
    idx = np.searchsorted(edges, x, side='right') - 1
    inside = (idx >= 0) & (idx < len(values))
    out = np.zeros_like(np.asarray(x, dtype=float))
    out[inside] = values[idx[inside]]
    return out


def fold_in_energy(
    evaluate: Callable[[float], np.ndarray],
    energy: float,
    sigma_energy: float,
    *,
    bounds: Optional[Tuple[float, float]] = None,
    n_nodes: int = 21,
) -> np.ndarray:
    """
    Gaussian average of ``evaluate(E)`` over incident energy.

    :math:`\\int f(E)\\,N(E; E_0, \\sigma_E^2)\\,dE` by Gauss-Hermite quadrature (the
    nodes ``kika.utils.numerics`` shares with every other folding path). Nodes are
    clamped into ``bounds`` and the weights renormalised, which only matters
    within a few sigma of a table edge.

    This is experimental resolution made explicit: the plottable adapters call it
    only when the caller passes ``resolution=``.
    """
    from kika.utils.numerics import gauss_hermite_nodes

    nodes, weights = gauss_hermite_nodes(energy, sigma_energy, n_nodes=n_nodes)
    if bounds is not None:
        nodes = np.clip(nodes, bounds[0], bounds[1])
    weights = weights / weights.sum()
    total = None
    for e, w in zip(nodes, weights):
        value = np.asarray(evaluate(float(e)), dtype=float) * w
        total = value if total is None else total + value
    return total


def _tof_sigma_ev(energy_ev: float, resolution: Tuple[float, float]) -> float:
    from kika.utils.energy_folding import tof_energy_resolution

    flight_path_m, delta_t_ns = resolution
    return 1e6 * tof_energy_resolution(energy_ev / 1e6, flight_path_m=flight_path_m,
                                       delta_t_ns=delta_t_ns)


def _resolution_detail(resolution: Optional[Tuple[float, float]]) -> Optional[str]:
    if resolution is None:
        return None
    return f'folded, TOF {resolution[0]:g} m / {resolution[1]:g} ns'


def _cosines(cosines: Optional[Iterable[float]], num_points: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, num_points) if cosines is None else np.asarray(cosines, dtype=float)


# -----------------------------------------------------------------------------
# ENDF / PENDF tapes
# -----------------------------------------------------------------------------

_ENDF = 'kika.endf.classes.endf.ENDF'


def _mt451(endf: Any) -> Any:
    mf1 = endf.files.get(1) if hasattr(endf, 'files') else None
    return mf1.sections.get(451) if mf1 is not None and hasattr(mf1, 'sections') else None


def _endf_lrp(endf: Any) -> Optional[int]:
    h = _mt451(endf)
    return getattr(h, '_lrp', None) if h is not None else None


def _endf_is_pendf(endf: Any) -> bool:
    return _endf_lrp(endf) == 2


def _endf_provenance(endf: Any, mt: Optional[int], **changes: Any) -> Provenance:
    h = _mt451(endf)
    evaluation = library_from_text(getattr(endf, '_kika_tpid', None))
    if evaluation is None and h is not None and not _endf_is_pendf(endf):
        evaluation = _library_from_mt451(h)
    temperature = getattr(h, '_temp', None) if h is not None else None
    base = dict(
        format='pendf' if _endf_is_pendf(endf) else 'endf',
        evaluation=evaluation,
        nuclide=getattr(endf, 'isotope', None),
        reaction=mt,
        temperature=temperature or None,
        source=getattr(endf, '_kika_source', None),
    )
    base.update(changes)
    return Provenance(**base)


def _sigma_arrays(section: Any) -> Tuple[np.ndarray, np.ndarray]:
    """``(E, sigma)`` of an MF3MT or a CrossSection (they spell sigma differently)."""
    energies = np.asarray(section.energies, dtype=float)
    values = section.cross_sections if hasattr(section, 'cross_sections') else section.values
    return energies, np.asarray(values, dtype=float)


def _endf_sigma_section(endf: Any, mt: int, reconstructed: Union[bool, str]) -> Tuple[Any, str]:
    """The MF3 section to plot for ``mt`` and the state its numbers are in."""
    if reconstructed not in (True, False, 'auto'):
        raise ValueError("reconstructed must be True, False or 'auto'")
    mf3 = endf.files.get(3)
    section = mf3.mt.get(mt) if mf3 is not None else None
    pendf = getattr(endf, 'pendf', None) or {}

    if _endf_is_pendf(endf):
        if section is None:
            raise NotPlottable(f'MT{mt} is not in this PENDF (MF3 has {_available(mf3)})')
        return section, 'reconstructed'
    if reconstructed is not False and mt in pendf:
        return pendf[mt], 'reconstructed'

    has_resonances = _endf_lrp(endf) == 1
    if reconstructed is True and has_resonances:
        raise ReconstructionRequired(
            f'MT{mt}: this evaluation has resonance parameters (MF2), so MF3 is only the '
            f'background. Reconstruct it (kika.processing.njoy_reconstruct) and set '
            f'endf.pendf, or pass reconstructed=False to plot the background.'
        )
    if section is None:
        raise NotPlottable(f'MT{mt} is not in MF3 (available: {_available(mf3)})')
    return section, 'background' if has_resonances else 'evaluated'


def _available(mf: Any) -> str:
    if mf is None:
        return 'nothing: the file was read without it'
    keys = sorted(int(k) for k in mf.mt.keys())
    return ', '.join(str(k) for k in keys[:30]) + (' ...' if len(keys) > 30 else '')


def _mf33_relative(endf: Any, mt: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """MF33 relative standard deviation of ``mt``: ``(edges eV, fraction per group)``.

    ``None`` when the tape has no MF33 for ``mt``. Any other failure raises: a
    swallowed band error hid a broken MF33 path for eight months once (library
    gap D13), and a missing band is not the same as no covariance.
    """
    mf33 = endf.files.get(33)
    if mf33 is None or mt not in mf33.mt:
        return None
    siblings = {int(k): v for k, v in mf33.sections.items() if int(k) != mt} \
        if hasattr(mf33, 'sections') else None
    mf3_secs = None
    if 3 in endf.files and hasattr(endf.files[3], 'sections'):
        mf3_secs = {int(k): v for k, v in endf.files[3].sections.items()}
    if getattr(endf, 'pendf', None):
        mf3_secs = dict(mf3_secs or {})
        mf3_secs.update({int(k): v for k, v in endf.pendf.items()})
    covmat = mf33.mt[mt].to_xs_covmat(sibling_sections=siblings, mf3_sections=mf3_secs)
    nuclide = endf.zaid if endf.zaid is not None else int(mf33.mt[mt]._za)
    _, unc = covmat.to_plot_data(nuclide=nuclide, mt=mt, sigma=1.0)
    if unc is None:
        return None
    return np.asarray(unc.x, dtype=float), np.asarray(unc.y, dtype=float) / 100.0


def _mf34_relative(endf: Any, mt: int, order: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """MF34 relative standard deviation of a_L: ``(edges eV, fraction per group)``."""
    mf34 = endf.files.get(34)
    if mf34 is None or mt not in mf34.mt:
        return None
    covmat = mf34.mt[mt].to_ang_covmat(mf4_data=endf.files.get(4))
    nuclide = endf.zaid if endf.zaid is not None else int(mf34.mt[mt]._za)
    _, unc = covmat.to_plot_data(nuclide=nuclide, mt=mt, order=order, uncertainty_type='relative')
    if unc is None:
        return None
    return np.asarray(unc.x, dtype=float), np.asarray(unc.y, dtype=float) / 100.0


def _relative_band(x: np.ndarray, rel: Optional[Tuple[np.ndarray, np.ndarray]], sigma: float,
                   ) -> Optional[UncertaintyBand]:
    if rel is None:
        return None
    edges, fractions = rel
    return UncertaintyBand(x=x, relative_uncertainty=_step_lookup(edges, fractions, x), sigma=sigma)


@register_adapter(_ENDF, 'cross_section')
def _endf_cross_section(endf: Any, *, mt: int, reconstructed: Union[bool, str] = 'auto',
                        uncertainty: bool = False, sigma: float = 1.0) -> PlotItem:
    """ENDF/PENDF sigma(E).

    ``reconstructed``: ``'auto'`` (default) uses ``endf.pendf`` when set and the MF3
    background otherwise (labelled as such); ``True`` requires reconstructed data
    and raises :class:`ReconstructionRequired` without it; ``False`` always plots
    MF3 as written. ``uncertainty=True`` attaches the MF33 band.
    """
    section, state = _endf_sigma_section(endf, mt, reconstructed)
    energies, values = _sigma_arrays(section)
    data = PlotData(x=energies, y=values,
                    provenance=_endf_provenance(endf, mt, state=state),
                    interpolation='lin-lin' if state == 'reconstructed' else None)
    band = _relative_band(energies, _mf33_relative(endf, mt), sigma) if uncertainty else None
    return PlotItem(data, band)


@register_adapter(_ENDF, 'legendre_coefficient')
def _endf_legendre(endf: Any, *, mt: int = 2, order: int = 1, uncertainty: bool = False,
                   sigma: float = 1.0) -> PlotItem:
    """ENDF MF4 a_L(E). Tabulated MF4 is projected by quadrature (state ``'projected'``).
    ``uncertainty=True`` attaches the MF34 band of a_L."""
    mf4 = endf.files.get(4)
    if mf4 is None or mt not in mf4.mt:
        raise NotPlottable(f'MF4/MT{mt} is not in this tape (MF4 has {_available(mf4)})')
    section = mf4.mt[mt]
    native = section.to_plot_data(order=order)
    tabulated = type(section).__name__ == 'MF4MTTabulated'
    data = PlotData(x=np.asarray(native.x, dtype=float), y=np.asarray(native.y, dtype=float),
                    provenance=_endf_provenance(endf, mt, state='projected' if tabulated else 'evaluated',
                                                frame=_frame(section)))
    band = _relative_band(data.x, _mf34_relative(endf, mt, order), sigma) if uncertainty else None
    return PlotItem(data, band)


def _frame(section: Any) -> Optional[str]:
    frame = getattr(section, 'frame', None)
    if frame is None:
        return None
    text = str(frame).upper()
    return 'CM' if 'C' in text and 'M' in text else ('LAB' if 'LAB' in text else text)


def _endf_pdf(endf: Any, mt: int) -> Any:
    mf4 = endf.files.get(4)
    if mf4 is None or mt not in mf4.mt:
        raise NotPlottable(f'MF4/MT{mt} is not in this tape (MF4 has {_available(mf4)})')
    return mf4.mt[mt]


@register_adapter(_ENDF, 'angular_distribution')
def _endf_angular(endf: Any, *, mt: int = 2, energy: float, cosines=None, num_points: int = 201,
                  resolution: Optional[Tuple[float, float]] = None) -> PlotItem:
    """ENDF f(mu | E) summed from MF4, at incident ``energy`` (eV).

    ``resolution=(flight_path_m, delta_t_ns)`` folds it with a TOF resolution
    (Gaussian in energy, ``delta_t_ns`` read as a FWHM)."""
    section = _endf_pdf(endf, mt)
    mu = _cosines(cosines, num_points)

    def pdf(e: float) -> np.ndarray:
        return np.squeeze(np.asarray(section.evaluate_angular_pdf(mu, e), dtype=float))

    values = pdf(energy) if resolution is None else fold_in_energy(
        pdf, energy, _tof_sigma_ev(energy, resolution))
    prov = _endf_provenance(endf, mt, state='evaluated', frame=_frame(section),
                            detail=_resolution_detail(resolution))
    return PlotItem(PlotData(x=mu, y=values, provenance=prov))


@register_adapter(_ENDF, 'differential_cross_section')
def _endf_dsigma(endf: Any, *, mt: int = 2, energy: float, cosines=None, num_points: int = 201,
                 reconstructed: Union[bool, str] = 'auto',
                 resolution: Optional[Tuple[float, float]] = None) -> PlotItem:
    """ENDF dsigma/dOmega = f(mu | E) sigma(E) / 2 pi at incident ``energy`` (eV).

    sigma(E) follows the ``reconstructed`` rule of the cross section. On a tape with
    resonances and no reconstruction, ``'auto'`` raises here rather than scaling by
    the MF3 background: a differential cross section built on the background is
    not a physical quantity. Pass ``reconstructed=False`` to accept it anyway."""
    if reconstructed == 'auto':
        reconstructed = True
    xs_section, state = _endf_sigma_section(endf, mt, reconstructed)
    e_grid, s_grid = _sigma_arrays(xs_section)
    section = _endf_pdf(endf, mt)
    mu = _cosines(cosines, num_points)

    def dsigma(e: float) -> np.ndarray:
        f = np.squeeze(np.asarray(section.evaluate_angular_pdf(mu, e), dtype=float))
        return f * float(np.interp(e, e_grid, s_grid)) / TWO_PI

    values = dsigma(energy) if resolution is None else fold_in_energy(
        dsigma, energy, _tof_sigma_ev(energy, resolution), bounds=(e_grid[0], e_grid[-1]))
    prov = _endf_provenance(endf, mt, state=state, frame=_frame(section),
                            detail=_resolution_detail(resolution))
    return PlotItem(PlotData(x=mu, y=values, provenance=prov))


@register_adapter(_ENDF, 'relative_uncertainty')
def _endf_relative_uncertainty(endf: Any, *, mt: int, order: Optional[int] = None) -> PlotItem:
    """Relative standard deviation per group: MF33 of sigma, or MF34 of a_L if ``order`` is given."""
    rel = _mf33_relative(endf, mt) if order is None else _mf34_relative(endf, mt, order)
    if rel is None:
        mf = 33 if order is None else 34
        raise NotPlottable(f'MF{mf}/MT{mt} is not in this tape')
    return _step_item(rel, _endf_provenance(endf, mt, state='multigroup'))


def _step_item(rel: Tuple[np.ndarray, np.ndarray], prov: Provenance) -> PlotItem:
    edges, fractions = rel
    values = np.asarray(fractions, dtype=float)[: len(edges) - 1] * 100.0
    y = np.append(values, values[-1]) if len(values) else values
    return PlotItem(PlotData(x=edges, y=y, plot_type='step', provenance=prov))


# -----------------------------------------------------------------------------
# ACE
# -----------------------------------------------------------------------------

_ACE = 'kika.ace.classes.ace.Ace'


def _ace_provenance(ace: Any, mt: Optional[int], **changes: Any) -> Provenance:
    from kika._utils import zaid_to_symbol

    header = getattr(ace, 'header', None)
    kt = getattr(header, 'temperature', None)
    zaid = getattr(header, 'zaid', None) or getattr(ace, 'zaid', None)
    base = dict(
        format='ace',
        evaluation=library_from_text(getattr(header, 'comment', None)),
        nuclide=zaid_to_symbol(int(zaid)) if zaid else None,
        reaction=mt,
        temperature=(kt / BOLTZMANN_MEV_PER_K) if kt else None,
        state='heated',
        source=getattr(ace, '_kika_source', None) or getattr(ace, 'filename', None),
    )
    base.update(changes)
    return Provenance(**base)


def _ace_sigma(ace: Any, mt: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        native = ace.cross_section.to_plot_data(mt)
    except (KeyError, ValueError, AttributeError) as exc:
        raise NotPlottable(f'MT{mt} is not in this ACE file: {exc}') from exc
    if native is None:
        raise NotPlottable(f'MT{mt} is not in this ACE file')
    return np.asarray(native.x, dtype=float) * 1e6, np.asarray(native.y, dtype=float)


def _ace_angular(ace: Any, mt: int) -> Any:
    # The façade module is imported directly (not through kika.nuclear_data) so
    # plotting does not trip its deprecation warning. The GNDS model does not
    # decode ACE angular data yet; when it does, this is the line to change.
    from kika.nuclear_data.angular_distribution import AngularDistribution

    try:
        return AngularDistribution.from_ace(ace, mt)
    except (KeyError, ValueError) as exc:
        raise NotPlottable(f'No angular distribution for MT{mt} in this ACE file: {exc}') from exc


@register_adapter(_ACE, 'cross_section')
def _ace_cross_section(ace: Any, *, mt: int, uncertainty: bool = False, sigma: float = 1.0) -> PlotItem:
    """ACE sigma(E), Doppler broadened at the file's temperature. ACE carries no
    covariances, so ``uncertainty`` is accepted and ignored."""
    energies, values = _ace_sigma(ace, mt)
    return PlotItem(PlotData(x=energies, y=values, provenance=_ace_provenance(ace, mt),
                             interpolation='lin-lin'))


@register_adapter(_ACE, 'legendre_coefficient')
def _ace_legendre(ace: Any, *, mt: int = 2, order: int = 1, uncertainty: bool = False,
                  sigma: float = 1.0) -> PlotItem:
    """a_L(E) projected from ACE's tabulated f(mu) by Gauss-Legendre quadrature,
    ``a_L = int f(mu) P_L(mu) dmu`` (ENDF convention, ``a_0 = 1``)."""
    ad = _ace_angular(ace, mt)
    if order not in (ad.coefficients or {}):
        if ad.tabulated_data is not None:
            ad.project_to_legendre(max_order=max(order, 1))
    coefficients = ad.coefficients or {}
    if order not in coefficients:
        raise NotPlottable(f'MT{mt}: no Legendre order {order} (representation {ad.representation!r})')
    projected = ad.representation != 'legendre'
    prov = _ace_provenance(ace, mt, state='projected' if projected else 'heated', frame=_frame(ad))
    return PlotItem(PlotData(x=np.asarray(ad.energies, dtype=float),
                             y=np.asarray(coefficients[order], dtype=float), provenance=prov))


@register_adapter(_ACE, 'angular_distribution')
def _ace_angular_distribution(ace: Any, *, mt: int = 2, energy: float, cosines=None,
                              num_points: int = 201,
                              resolution: Optional[Tuple[float, float]] = None) -> PlotItem:
    """ACE f(mu | E) at incident ``energy`` (eV), interpolated between tabulated energies."""
    ad = _ace_angular(ace, mt)
    mu = _cosines(cosines, num_points)

    def pdf(e: float) -> np.ndarray:
        return np.asarray(ad.evaluate_pdf(e, mu)[1], dtype=float)

    values = pdf(energy) if resolution is None else fold_in_energy(
        pdf, energy, _tof_sigma_ev(energy, resolution))
    prov = _ace_provenance(ace, mt, frame=_frame(ad), detail=_resolution_detail(resolution))
    return PlotItem(PlotData(x=mu, y=values, provenance=prov))


@register_adapter(_ACE, 'differential_cross_section')
def _ace_dsigma(ace: Any, *, mt: int = 2, energy: float, cosines=None, num_points: int = 201,
                resolution: Optional[Tuple[float, float]] = None) -> PlotItem:
    """ACE dsigma/dOmega = f(mu | E) sigma(E) / 2 pi at incident ``energy`` (eV)."""
    ad = _ace_angular(ace, mt)
    e_grid, s_grid = _ace_sigma(ace, mt)
    mu = _cosines(cosines, num_points)

    def dsigma(e: float) -> np.ndarray:
        return np.asarray(ad.evaluate_pdf(e, mu)[1], dtype=float) * float(np.interp(e, e_grid, s_grid)) / TWO_PI

    values = dsigma(energy) if resolution is None else fold_in_energy(
        dsigma, energy, _tof_sigma_ev(energy, resolution), bounds=(e_grid[0], e_grid[-1]))
    prov = _ace_provenance(ace, mt, frame=_frame(ad), detail=_resolution_detail(resolution))
    return PlotItem(PlotData(x=mu, y=values, provenance=prov))


# -----------------------------------------------------------------------------
# The GNDS-shaped model
# -----------------------------------------------------------------------------

_SUITE = 'kika.nuclear_data.model.suite.ReactionSuite'


@register_adapter(_SUITE, 'cross_section')
def _suite_cross_section(suite: Any, *, mt: int, form: Optional[str] = None,
                         uncertainty: bool = False, sigma: float = 1.0) -> PlotItem:
    """sigma(E) from a ``ReactionSuite`` (``kika.read``). ``form`` picks the style
    label; by default the evaluated form, or the only one there is (ACE decodes
    to a heated, gridded form)."""
    try:
        reaction = suite.reactionByENDF_MT(mt)
    except Exception as exc:
        raise NotPlottable(f'MT{mt} is not in this ReactionSuite: {exc}') from exc
    labels = list(reaction.crossSection.keys()) if hasattr(reaction.crossSection, 'keys') else []
    if form is None:
        form = 'eval' if 'eval' in labels or not labels else labels[0]
    energies, values = suite.cross_section(mt, form=form)
    state = 'heated' if 'heat' in form.lower() or 'grid' in form.lower() else 'evaluated'
    prov = Provenance(format='gnds', evaluation=getattr(suite, 'evaluation', None),
                      nuclide=getattr(suite, 'target', None), reaction=mt, state=state)
    return PlotItem(PlotData(x=np.asarray(energies, dtype=float), y=np.asarray(values, dtype=float),
                             provenance=prov))


# -----------------------------------------------------------------------------
# Façades (kika.nuclear_data.CrossSection / AngularDistribution)
# -----------------------------------------------------------------------------

_XS_FACADE = 'kika.nuclear_data.cross_section.CrossSection'
_AD_FACADE = 'kika.nuclear_data.angular_distribution.AngularDistribution'


def _facade_nuclide(obj: Any) -> Optional[str]:
    from kika._utils import zaid_to_symbol
    zaid = getattr(obj, 'nuclide_id', None)
    return zaid_to_symbol(int(zaid)) if zaid else None


@register_adapter(_XS_FACADE, 'cross_section')
def _facade_cross_section(xs: Any, *, mt: Optional[int] = None, state: Optional[str] = None,
                          uncertainty: bool = False, sigma: float = 1.0) -> PlotItem:
    """sigma(E) of a ``CrossSection`` (what ``njoy_reconstruct`` returns per MT).
    Pass ``state='reconstructed'`` when that is what it holds."""
    temperature = getattr(xs, 'temperature', 0.0) or None
    prov = Provenance(nuclide=_facade_nuclide(xs), reaction=getattr(xs, 'reaction', mt),
                      temperature=temperature, state=state)
    return PlotItem(PlotData(x=np.asarray(xs.energies, dtype=float), y=np.asarray(xs.values, dtype=float),
                             provenance=prov))


@register_adapter(_AD_FACADE, 'legendre_coefficient')
def _facade_legendre(ad: Any, *, order: int = 1, mt: Optional[int] = None, **_: Any) -> PlotItem:
    if order not in (ad.coefficients or {}) and ad.tabulated_data is not None:
        ad.project_to_legendre(max_order=max(order, 1))
    if order not in (ad.coefficients or {}):
        raise NotPlottable(f'No Legendre order {order} in this angular distribution')
    prov = Provenance(nuclide=_facade_nuclide(ad), reaction=getattr(ad, 'reaction', mt),
                      frame=_frame(ad), state='projected' if ad.representation != 'legendre' else 'evaluated')
    return PlotItem(PlotData(x=np.asarray(ad.energies, dtype=float),
                             y=np.asarray(ad.coefficients[order], dtype=float), provenance=prov))


@register_adapter(_AD_FACADE, 'angular_distribution')
def _facade_angular(ad: Any, *, energy: float, cosines=None, num_points: int = 201,
                    mt: Optional[int] = None, **_: Any) -> PlotItem:
    mu, f = ad.evaluate_pdf(energy, _cosines(cosines, num_points))
    prov = Provenance(nuclide=_facade_nuclide(ad), reaction=getattr(ad, 'reaction', mt), frame=_frame(ad))
    return PlotItem(PlotData(x=np.asarray(mu, dtype=float), y=np.asarray(f, dtype=float), provenance=prov))


# -----------------------------------------------------------------------------
# EXFOR
# -----------------------------------------------------------------------------

_EXFOR_XS = 'kika.exfor.cross_section.ExforCrossSection'
_EXFOR_AD = 'kika.exfor.angular_distribution.ExforAngularDistribution'


def _exfor_provenance(exp: Any, mt: Optional[int], **changes: Any) -> Provenance:
    target = getattr(exp, 'target', None)
    if target in ('Unknown', ''):
        target = None
    base = dict(format='exfor', evaluation=None, nuclide=target, reaction=mt, state='measured',
                detail=f'EXFOR {getattr(exp, "entry", "")}{("." + exp.subentry) if getattr(exp, "subentry", None) else ""}'.strip())
    base.update(changes)
    return Provenance(**base)


def _exfor_label(exp: Any) -> str:
    label = getattr(exp, 'label', None) or 'EXFOR'
    if getattr(exp, 'is_natural_target', False):
        label += ' [nat]'
    return label


def _markers(data: PlotData) -> PlotData:
    data.linestyle = 'none'
    data.marker = data.marker or 'o'
    return data


@register_adapter(_EXFOR_XS, 'cross_section')
def _exfor_cross_section(exp: Any, *, energy: Optional[Tuple[float, float]] = None,
                         uncertainty: bool = True, sigma: float = 1.0, mt: Optional[int] = None) -> PlotItem:
    """An EXFOR cross-section set as points; its errors as error bars.
    ``energy`` is an optional ``(min, max)`` window in eV."""
    df = exp.to_dataframe(energy=energy, energy_unit='eV', cross_section_unit='b')
    if df is None or df.empty:
        raise NotPlottable(f'EXFOR {getattr(exp, "entry", "?")}: no points in the requested range')
    df = df.sort_values('energy')
    x = df['energy'].to_numpy(dtype=float)
    y = df['cross_section'].to_numpy(dtype=float)
    data = _markers(PlotData(x=x, y=y, label=_exfor_label(exp), provenance=_exfor_provenance(exp, mt)))
    band = None
    if uncertainty and 'error' in df.columns:
        err = np.nan_to_num(df['error'].to_numpy(dtype=float)) * sigma
        band = UncertaintyBand(x=x, y_lower=y - err, y_upper=y + err, style='errorbar', capsize=2)
    return PlotItem(data, band)


@register_adapter(_EXFOR_AD, 'differential_cross_section')
def _exfor_dsigma(exp: Any, *, energy: float, frame: Optional[str] = None, uncertainty: bool = True,
                  sigma: float = 1.0, mt: Optional[int] = None, **kwargs: Any) -> PlotItem:
    """An EXFOR angular distribution at incident ``energy`` (eV), in b/sr versus
    cosine, optionally converted to ``frame`` ('CM' or 'LAB')."""
    native = exp.to_plot_data(energy / 1e6, frame=frame, angle_unit='cos', cross_section_unit='b/sr',
                              uncertainty=True, **kwargs)
    if isinstance(native, tuple):
        native = native[0]
    if isinstance(native, list):
        raise NotPlottable('Several EXFOR energies match; pass a single energy with a tighter tolerance')
    if native is None:
        raise NotPlottable(f'EXFOR {getattr(exp, "entry", "?")}: no data at {format_energy(energy)}')
    x = np.asarray(native.x, dtype=float)
    y = np.asarray(native.y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    actual = native.metadata.get('actual_energy')
    detail = f'EXFOR {exp.entry}' + (f' @ {format_energy(actual * 1e6)}' if actual else '')
    data = _markers(PlotData(x=x, y=y, label=_exfor_label(exp),
                             provenance=_exfor_provenance(exp, mt, frame=frame or native.metadata.get('frame'),
                                                          detail=detail)))
    band = None
    yerr = native.metadata.get('yerr')
    if uncertainty and yerr is not None:
        err = np.asarray(yerr, dtype=float)[order] * sigma
        band = UncertaintyBand(x=x, y_lower=y - err, y_upper=y + err, style='errorbar', capsize=2)
    return PlotItem(data, band)


# -----------------------------------------------------------------------------
# Covariance objects
# -----------------------------------------------------------------------------

_XS_COV = 'kika.cov.cross_section_covariance.CrossSectionCovariance'
_LEG_COV = 'kika.cov.legendre_covariance.LegendreCovariance'


@register_adapter(_XS_COV, 'relative_uncertainty')
def _xs_cov_uncertainty(cov: Any, *, nuclide: Union[int, str], mt: int, **_: Any) -> PlotItem:
    """Relative standard deviation of sigma per group, from a covariance matrix."""
    _, unc = cov.to_plot_data(nuclide=nuclide, mt=mt, sigma=1.0)
    if unc is None:
        raise NotPlottable(f'No variance for {nuclide} MT{mt} in this covariance')
    return _step_item((np.asarray(unc.x, dtype=float), np.asarray(unc.y, dtype=float) / 100.0),
                      Provenance(format='covariance', nuclide=str(nuclide), reaction=mt, state='multigroup'))


@register_adapter(_LEG_COV, 'relative_uncertainty')
def _leg_cov_uncertainty(cov: Any, *, nuclide: Union[int, str], mt: int, order: int, **_: Any) -> PlotItem:
    """Relative standard deviation of a_L per group, from a Legendre covariance."""
    _, unc = cov.to_plot_data(nuclide=nuclide, mt=mt, order=order, uncertainty_type='relative')
    if unc is None:
        raise NotPlottable(f'No variance for {nuclide} MT{mt} L={order} in this covariance')
    return _step_item((np.asarray(unc.x, dtype=float), np.asarray(unc.y, dtype=float) / 100.0),
                      Provenance(format='covariance', nuclide=str(nuclide), reaction=mt, state='multigroup'))
