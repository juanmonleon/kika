"""A processed :class:`~kika.nuclear_data.model.suite.ReactionSuite` → an :class:`~kika.ace.classes.ace.Ace`.

**One serialisation path.** This module writes no word. It fills the
block objects the block encoders read (:mod:`kika.ace.encoders.blocks`), lets
:func:`kika.ace.encoders.layout.encode_xss` lay the XSS out and fill NXS and
JXS, and reads the result back with the parser, so what comes out is the same
``Ace`` a file read from disk is: the object the user, the app, the sampling
and ``patch_writer`` see, writable with ``write_ace``.

**What it takes.** The suite in the processed shape the HDF5 encoder takes
(:mod:`kika.hdf5.neutron_encode`) and :func:`~kika.ace.model_adapter.decode.decodeAce`
produces: one ``heated_<T>`` style per temperature, every cross section at
that temperature an ``XYs1d`` on one union grid (each a suffix of it, which is
what IE says), the distributions and multiplicities as ``MonteCarlo`` forms,
the URR tables under ``URR_<T>``. One ``Ace`` is one temperature; a suite with
several needs ``temperature=``.

=====================================  =====================================
model                                  ACE
=====================================  =====================================
union grid, MT 1, 101, 2, 301          ESZ (heating number = KERMA / σ_t)
reactions written as reactions          MTR, LQR (Q in MeV), TYR, LSIG/SIG
neutron product: multiplicity, frame   TY (constant, or > 100 with the table
                                       in DLW; negative in the CM frame;
                                       19 for fission)
``AngularTwoBody`` / ``Uncorrelated``  AND: LOCB 0 (isotropic), 32
angular                                equiprobable bins, or a tabulated
                                       record; LOCB -1 for laws 44/61/66/67
``DistributionMixture``                LNW chain, one law per ``Weighted``
                                       with its P(E)
``LevelEnergy``                        law 3 (or 33)
``Xs_pdf_cdf1d`` tables in E'          law 4 (or 1), discrete lines first
spectra                                laws 5, 7, 9, 11
``KalbachMann`` / ``EnergyAngularMC``  law 44 / law 61
``NBodyPhaseSpace``                    law 66
fission ν (prompt/total), MT 452/455   NU, DNU
``DelayedNeutron``                     BDD and DNED
photon products                        GPD, MTRP, SIGP (MFTYPE 12/13/16),
                                       ANDP, DLWP (laws 2 and 4), YP
``ProbabilityTables``                  UNR
=====================================  =====================================

**What the model cannot say and the ``AceProvenance`` of a decoded table
does.** MTR and YP in NJOY's order (MT 4 goes last when a photon yield needs
it), which law of two that share a form a chain used (3/33, 1/4), law 66's
xi table, a delayed group's BDD record, a photon's MTRP and MFTYPE, and the
P(E) of a one-law chain that the model keeps without a weight. When a suite
has none of it (it was processed by kika, not read from ACE), the rules
here are NJOY's (ACER's ``unionx``): secondary-neutron reactions first and
the rest ascending; of the sums MT 18 (or the fission chances, when MT 19
carries distributions), 103-107, MT 4 when a photon yield or the URR
competition (ILF = 4) needs it, last; MTRP = 1000·MT + i with the MFTYPE of
the file the photon was stated in (MF12 → 12, MF13 → 13, MF6 → 16), the
multipliers' MTs ascending; a reaction starts one point before its first
nonzero value; a P(E) of 1 over the law's domain; ESZ's heating number is
HEATR's ``local = 0`` KERMA, without the delayed betas OpenMC's MT301 adds.

**Not written:** the charged-particle production blocks (PTYPE, NTRO, IXS),
which the decoder does not convert either; reported.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace as NS
from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.nuclear_data.model import AceProvenance, ConversionReport

from .secondaries import DELAYED_NUBAR_LABEL, FISSION_MTS, TOTAL_NUBAR_LABEL

__all__ = ["encodeAce", "AceEncodeError"]

EV_PER_MEV = 1.0e6
PER_SHAKE_TO_PER_S = 1.0e8
#: Laws whose angle is in the law itself: their reaction's LOCB is -1.
CORRELATED_LAWS = (44, 61, 66, 67)
#: Redundant reactions NJOY's ACER keeps in MTR beside their components.
KEPT_SUMS = (18, 103, 104, 105, 106, 107)
#: ESZ's heating number comes from this production, never written to MTR.
HEATING_MT = 301
#: ACER's equiprobable cosine bins per incident energy (AND LC > 0, ANDP)
EQUIPROBABLE_BINS = 32
#: INTT of an outgoing-energy table of lines only, for a suite not read from ACE:
#: what NJOY 2016's ACER writes for an MF6 table of discrete lines (2), which
#: OpenMC reads; a table read from ACE keeps its own (NJOY 0 in older tables)
LINES_ONLY_INTT = 2
#: The sign of law 61's cosine-table locators (LC) for a suite not read from ACE:
#: positive, as NJOY 2016 and LANL's Lib81 write them. OpenMC reads a table only
#: behind a positive LC and takes any other as isotropic; MCNP takes |LC|. A
#: table read from ACE keeps its own (older NJOY tables are negative).
LAW61_LC_SIGN = 1
#: The light charged particles of an ACE table: model pid -> (PTYPE IPT, mass ratio to the
#: neutron, ENDF's AWP for them)
PARTICLE_TYPES = {"H1": (9, 0.99862347), "H2": (31, 1.99626), "H3": (32, 2.98960),
                  "He3": (33, 2.98903), "He4": (34, 3.96713)}
#: OpenMC's local-photon KERMA; an ACE table has only ESZ's heating number
LOCAL_HEATING_MT = 901


class AceEncodeError(ValueError):
    """The suite is not in the processed shape an ACE table needs."""


def _m():
    from kika.nuclear_data import model

    return model


# ----------------------------------------------------------------------
# Small conversions
# ----------------------------------------------------------------------

def _mev(values) -> np.ndarray:
    return np.asarray(values, dtype=float) / EV_PER_MEV


def _regions(function) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """``(x, y, NBT, INT)`` of a 1-d function; NR = 0 for one lin-lin region, as NJOY writes it."""
    x, y, pairs = function.toEndfRegions()
    pairs = [(int(b), int(c)) for b, c in pairs]
    if len(pairs) == 1 and pairs[0][1] == 2:
        pairs = []
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), [b for b, _ in pairs], [c for _, c in pairs]


def _tab2(form) -> Tuple[list, List[int], List[int]]:
    """``(functions, NBT, INT)`` of a 2-d function; NR = 0 for one lin-lin region."""
    from kika.nuclear_data.model.functions.higher import toEndfTab2

    functions, pairs = toEndfTab2(form)
    pairs = [(int(b), int(c)) for b, c in pairs]
    if len(pairs) == 1 and pairs[0][1] == 2:
        pairs = []
    return functions, [b for b, _ in pairs], [c for _, c in pairs]


def _intt(interpolation) -> int:
    m = _m()
    if interpolation == m.Interpolation.flat:
        return 1
    if interpolation == m.Interpolation.linlin:
        return 2
    raise AceEncodeError(f"an ACE table interpolates histogram or lin-lin, not {interpolation}")


def _attributes(node) -> dict:
    provenance = getattr(node, "provenance", None)
    return provenance.attributes if isinstance(provenance, AceProvenance) else {}


def _photonType(product) -> Optional[int]:
    """The MFTYPE a processed suite states for a photon (MF6 16, MF13 13, MF12 12)."""
    stated = _attributes(product).get("MFTYPE")
    if stated is None:
        stated = (getattr(getattr(product, "provenance", None), "attributes", None) or {}).get("MFTYPE")
    return None if stated is None else int(stated)


def _form(component, label: str, what: str):
    if component is None:
        return None
    forms = getattr(component, "forms", None)
    if forms is not None and label in forms:
        return forms[label]
    return None


def _monteCarlo(component, what: str):
    from kika.hdf5.names import MONTE_CARLO_LABEL

    form = _form(component, MONTE_CARLO_LABEL, what)
    if form is None and component is not None and len(getattr(component, "forms", {}) or {}) == 1:
        form = next(iter(component.forms.values()))
    if form is None:
        raise AceEncodeError(f"{what} has no {MONTE_CARLO_LABEL!r} form "
                             f"({sorted(getattr(component, 'forms', {}) or {})}); an ACE table holds "
                             f"tables with their CDF, which kika.processing.tabulation produces")
    return form


def _multiplicityForm(multiplicity):
    from kika.hdf5.names import MONTE_CARLO_LABEL

    if multiplicity is None:
        return None
    forms = getattr(multiplicity, "forms", None) or {}
    if MONTE_CARLO_LABEL in forms:
        return forms[MONTE_CARLO_LABEL]
    return multiplicity.form


def _isotropicTable(t) -> bool:
    """The two-point histogram OpenMC (and the decoder) write for an isotropic cosine table."""
    m = _m()
    return (len(t.xs) == 2 and t.xs[0] == -1.0 and t.xs[1] == 1.0 and t.interpolation == m.Interpolation.flat
            and t.pdf[0] == 0.5 and t.cdf[0] == 0.0 and t.cdf[1] == 1.0)


def _equiprobableTable(t) -> bool:
    """32 (or N) equiprobable bins: a histogram with CDF k/N and P = 1/(N·width)."""
    m = _m()
    n = len(t.xs) - 1
    if n < 2 or t.interpolation != m.Interpolation.flat:
        return False
    if not np.allclose(t.cdf, np.linspace(0.0, 1.0, n + 1), rtol=0.0, atol=1e-12):
        return False
    widths = np.diff(t.xs)
    with np.errstate(divide="ignore"):
        expected = np.where(widths > 0, 1.0 / (n * np.where(widths > 0, widths, 1.0)), 0.0)
    return bool(np.allclose(t.pdf[:n], expected, rtol=1e-12, atol=0.0))


# ----------------------------------------------------------------------
# Angular distributions (AND, ANDP)
# ----------------------------------------------------------------------

def _equiprobableEdges(t, bins: int = EQUIPROBABLE_BINS) -> np.ndarray:
    """The ``bins + 1`` cosines that cut a cosine table into bins of equal probability.

    The table's own CDF, inverted exactly: linear in a histogram panel,
    quadratic in a lin-lin one. ACER's form for a photon (ANDP holds nothing
    else), and what it makes of MF14.
    """
    m = _m()
    x, p = np.asarray(t.xs, dtype=float), np.asarray(t.pdf, dtype=float)
    dx = np.diff(x)
    flat = t.interpolation == m.Interpolation.flat
    mass = p[:-1] * dx if flat else 0.5 * (p[:-1] + p[1:]) * dx
    c = np.concatenate([[0.0], np.cumsum(mass)])
    p, c = p / c[-1], c / c[-1]
    edges = [x[0]]
    for q in np.arange(1, bins) / bins:
        i = int(np.clip(np.searchsorted(c, q, side="right") - 1, 0, len(dx) - 1))
        while mass[i] <= 0.0 and i + 1 < len(dx):
            i += 1
        dq = q - c[i]
        slope = 0.0 if flat else (p[i + 1] - p[i]) / dx[i]
        if abs(slope) * dx[i] <= 1e-12 * max(p[i], 1e-300):
            s = dq / p[i]
        else:
            s = (-p[i] + np.sqrt(max(p[i] * p[i] + 2.0 * slope * dq, 0.0))) / slope
        edges.append(x[i] + min(max(s, 0.0), dx[i]))
    edges.append(x[-1])
    return np.maximum.accumulate(np.asarray(edges))


def _asEquiprobable(t):
    """``t`` as the histogram of its equiprobable bins (the shape ``_equiprobableTable`` recognises)."""
    m = _m()
    from kika.nuclear_data.model import Xs_pdf_cdf1d

    edges = _equiprobableEdges(t)
    n = edges.size - 1
    pdf = np.zeros(n + 1)
    with np.errstate(divide="ignore"):
        pdf[:n] = np.where(np.diff(edges) > 0, 1.0 / (n * np.diff(edges)), 0.0)
    return Xs_pdf_cdf1d(xs=edges, pdf=pdf, cdf=np.linspace(0.0, 1.0, n + 1),
                        interpolation=m.Interpolation.flat, outerDomainValue=t.outerDomainValue)


def _angularRecord(angular, mt: int, photon: bool = False):
    """A 2-d cosine form (or ``Isotropic2d``/None) → the parser's angular object.

    ``photon``: ANDP states a photon's cosine as isotropic or as equiprobable
    bins only, so any other table is cut into 32 of them.
    """
    from kika.ace.classes.angular_distribution.distributions import (
        EquiprobableAngularDistribution, IsotropicAngularDistribution, TabulatedAngularDistribution)

    m = _m()
    if angular is None or isinstance(angular, m.Isotropic2d):
        return IsotropicAngularDistribution(mt=mt)
    tables = list(_tab2(angular)[0])
    energies = _mev([t.outerDomainValue for t in tables])
    forms = []
    for i, t in enumerate(tables):
        if _isotropicTable(t):
            forms.append("isotropic")
        elif _equiprobableTable(t):
            forms.append("equiprobable")
        elif photon:
            tables[i] = _asEquiprobable(t)
            forms.append("equiprobable")
        else:
            forms.append("tabulated")
    if photon and all(f == "isotropic" for f in forms):
        return IsotropicAngularDistribution(mt=mt)
    if all(f == "equiprobable" for f in forms):
        return EquiprobableAngularDistribution(mt=mt, _energies=energies,
                                               _cosine_bins=[np.asarray(t.xs, dtype=float) for t in tables],
                                               energy_forms=forms)
    return TabulatedAngularDistribution(
        mt=mt, _energies=energies, energy_forms=forms,
        interpolation=[_intt(t.interpolation) if f == "tabulated" else 0 for t, f in zip(tables, forms)],
        _cosine_grid=[np.asarray(t.xs, dtype=float) for t in tables],
        _pdf=[np.asarray(t.pdf, dtype=float) for t in tables],
        _cdf=[np.asarray(t.cdf, dtype=float) for t in tables])


def _correlatedRecord(mt: int):
    from kika.ace.classes.angular_distribution.distributions import KalbachMannAngularDistribution

    return KalbachMannAngularDistribution(mt=mt)


# ----------------------------------------------------------------------
# Energy laws (DLW, DLWP, DNED)
# ----------------------------------------------------------------------

def _applicability(weight, law: NS, stated: Optional[dict], fallback: Tuple[float, float]) -> None:
    """P(E) of one law: its ``Weighted.weight``, else the stated one, else 1 over its domain.

    A law with no incident-energy table of its own (2, 3, 33, 66) takes
    ``fallback``, the domain of the table (eV).
    """
    if weight is not None:
        x, y, nbt, interp = _regions(weight)
        law.applicability_energies, law.applicability_probabilities = _mev(x), y
        law.applicability_nbt, law.applicability_interp = nbt, interp
        return
    if stated is not None:
        law.applicability_energies = np.asarray(stated["energies"], dtype=float)
        law.applicability_probabilities = np.asarray(stated["probabilities"], dtype=float)
        law.applicability_nbt, law.applicability_interp = list(stated["nbt"]), list(stated["interp"])
        return
    lo, hi = law.domain if law.domain[1] > law.domain[0] else fallback
    law.applicability_energies = np.array([lo, hi]) / EV_PER_MEV
    law.applicability_probabilities = np.array([1.0, 1.0])
    law.applicability_nbt, law.applicability_interp = [], []


def _eoutTables(form, incidentWhat: str, linesOnlyINTT: int = 0):
    """A 2-d form of ``Xs_pdf_cdf1d`` in E' → (incident energies, NBT, INT, table dicts)."""
    tables, nbt, interp = _tab2(form)
    out = []
    for t in tables:
        nd = int(getattr(t, "numberOfDiscrete", 0) or 0)
        pdf = np.array(t.pdf, dtype=float)
        pdf[nd:] = pdf[nd:] * EV_PER_MEV
        # A table of lines only is a histogram in the model; NJOY writes it INTT = 0.
        intt = linesOnlyINTT if nd == len(pdf) else _intt(t.interpolation)
        out.append({"e_out": _mev(t.xs), "pdf": pdf, "cdf": np.array(t.cdf, dtype=float),
                    "intt": intt, "n_discrete": nd})
    return _mev([t.outerDomainValue for t in tables]), nbt, interp, out, tables


def _law(functional, lawNumber: Optional[int], attributes: dict, where: str) -> NS:
    """One model form → the attributes of the parser's law object."""
    m = _m()
    from kika._constants import NEUTRON_MASS_AMU

    if isinstance(functional, m.Uncorrelated):
        energy = functional.energy
        if isinstance(energy, m.DiscretePhotonEnergy):
            return NS(law=2, lp=int(energy.primaryFlag), eg=float(energy.energy) / EV_PER_MEV,
                      domain=(0.0, 0.0))
        if isinstance(energy, (m.DiscreteGamma, m.PrimaryGamma)):
            return NS(law=2, lp=2 if isinstance(energy, m.PrimaryGamma) else 0,
                      eg=float(energy.value) / EV_PER_MEV,
                      domain=(float(energy.domainMin), float(energy.domainMax)))
        if isinstance(energy, m.LevelEnergy):
            if lawNumber == 33:
                return NS(law=33, threshold=float(energy.threshold) / EV_PER_MEV,
                          energy_factor=float(energy.massRatio), domain=(0.0, 0.0))
            return NS(law=3, aplusoaabsq=float(energy.threshold) / EV_PER_MEV,
                      asquare=float(energy.massRatio), domain=(0.0, 0.0))
        if isinstance(energy, m.NBodyPhaseSpace):
            law = NS(law=66, npsx=int(energy.numberOfProducts),
                     ap=float(energy.mass.value) / NEUTRON_MASS_AMU, xi_grid=[], pdf=[], cdf=[], intt=0,
                     domain=(0.0, 0.0))
            xi = attributes.get("law66")
            if xi:
                law.intt, law.xi_grid, law.pdf, law.cdf = int(xi["intt"]), xi["xi"], xi["pdf"], xi["cdf"]
            return law
        if isinstance(energy, (m.SimpleMaxwellianFission, m.Evaporation, m.Watt, m.GeneralEvaporation)):
            return _spectrum(energy, where)
        if isinstance(energy, (m.XYs2d, m.Regions2d)):
            energies, nbt, interp, tables, raw = _eoutTables(energy, where,
                                                             int(attributes.get("linesOnlyINTT", LINES_ONLY_INTT)))
            if lawNumber == 1:
                return NS(law=1, nbt=nbt, interp=interp, incident_energies=energies,
                          distribution_data=[{"e_out": t["e_out"]} for t in tables],
                          domain=(raw[0].outerDomainValue, raw[-1].outerDomainValue))
            return NS(law=4, nbt=nbt, interp=interp, incident_energies=energies, distributions=tables,
                      domain=(raw[0].outerDomainValue, raw[-1].outerDomainValue))
        raise AceEncodeError(f"{where}: an uncorrelated energy form {type(energy).__name__} has no ACE law")
    if isinstance(functional, m.KalbachMann):
        energies, nbt, interp, tables, raw = _eoutTables(functional.f, where,
                                                         int(attributes.get("linesOnlyINTT", LINES_ONLY_INTT)))
        rs, _, _ = _tab2(functional.r)
        as_, _, _ = _tab2(functional.a)
        for t, r, a in zip(tables, rs, as_):
            t["r"], t["a"] = np.asarray(r.ys, dtype=float), np.asarray(a.ys, dtype=float)
        return NS(law=44, nbt=nbt, interp=interp, incident_energies=energies, distributions=tables,
                  domain=(raw[0].outerDomainValue, raw[-1].outerDomainValue))
    if isinstance(functional, m.EnergyAngularMC):
        return _law61(functional, where, int(attributes.get("law61LC", LAW61_LC_SIGN)),
                      int(attributes.get("linesOnlyINTT", LINES_ONLY_INTT)))
    raise AceEncodeError(f"{where}: {type(functional).__name__} has no ACE energy law")


def _law61(form, where: str, sign: int = LAW61_LC_SIGN, linesOnlyINTT: int = 0) -> NS:
    """Law 61: each E' table with its LC column, then a cosine table per E' point.

    One table per point, as NJOY writes it: in ENDF/B-VIII.1 Fe-56 MT5 not one
    LC of 2 518 is shared, though 70 tables repeat an earlier one's numbers, so
    merging equal tables would write a smaller table than NJOY's for nothing.
    """
    energies, nbt, interp, tables, raw = _eoutTables(form.energy, where, linesOnlyINTT)
    angular = []
    for t, inner in zip(tables, form.energyAngular.function2ds):
        lcs = []
        for mu in inner.function1ds:
            if _isotropicTable(mu):
                lcs.append(0)
                continue
            lc = sign * (len(angular) + 1)
            angular.append({"lc": lc, "jj": _intt(mu.interpolation), "cosines": np.asarray(mu.xs, dtype=float),
                            "pdf": np.asarray(mu.pdf, dtype=float), "cdf": np.asarray(mu.cdf, dtype=float)})
            lcs.append(lc)
        if len(lcs) != len(t["e_out"]):
            raise AceEncodeError(f"{where}: law 61 table has {len(t['e_out'])} E' points and "
                                 f"{len(lcs)} cosine tables")
        t["lc"] = lcs
    return NS(law=61, nbt=nbt, interp=interp, incident_energies=energies, distributions=tables,
              angular_tables=angular, domain=(raw[0].outerDomainValue, raw[-1].outerDomainValue))


def _spectrum(energy, where: str) -> NS:
    m = _m()
    if isinstance(energy, m.Watt):
        ax, ay, anbt, aint = _regions(energy.a)
        bx, by, bnbt, bint = _regions(energy.b)
        return NS(law=11, a_nbt=anbt, a_interp=aint, a_incident_energies=_mev(ax), a_values=ay / EV_PER_MEV,
                  b_nbt=bnbt, b_interp=bint, b_incident_energies=_mev(bx), b_values=by * EV_PER_MEV,
                  restriction_energy=float(energy.U.value) / EV_PER_MEV, domain=(ax[0], ax[-1]))
    x, y, nbt, interp = _regions(energy.theta)
    law = NS(temp_nbt=nbt, temp_interp=interp, incident_energies=_mev(x), temperatures=y / EV_PER_MEV,
             domain=(x[0], x[-1]))
    if isinstance(energy, m.GeneralEvaporation):
        law.law, law.equiprob_values = 5, np.asarray(energy.g.xs, dtype=float)
        return law
    law.law = 7 if isinstance(energy, m.SimpleMaxwellianFission) else 9
    law.restriction_energy = float(energy.U.value) / EV_PER_MEV
    return law


def _chain(distributionForm, attributes: dict, where: str,
           fallback: Tuple[float, float]) -> Tuple[List[NS], List[int]]:
    """A product's Monte Carlo form → its law chain (and the law numbers used)."""
    m = _m()
    stated = list(attributes.get("laws") or [])
    if isinstance(distributionForm, m.DistributionMixture):
        weighted = [(w.weight, w.functional) for w in distributionForm.weighted]
    else:
        weighted = [(None, distributionForm)]
    laws = []
    for k, (weight, functional) in enumerate(weighted):
        law = _law(functional, stated[k] if k < len(stated) else None, attributes, f"{where} law {k + 1}")
        _applicability(weight, law, attributes.get("applicability") if len(weighted) == 1 else None,
                       fallback)
        laws.append(law)
    return laws, [law.law for law in laws]


def _angularOf(distributionForm):
    """The cosine form a law chain shares: its ``Uncorrelated`` angular half, if any."""
    m = _m()
    functionals = ([w.functional for w in distributionForm.weighted]
                   if isinstance(distributionForm, m.DistributionMixture) else [distributionForm])
    for f in functionals:
        if isinstance(f, m.Uncorrelated):
            return f.angular
        if isinstance(f, m.AngularTwoBody):
            return f.angular
    return None


def _frameOf(distributionForm) -> bool:
    """True when the product is stated in the centre-of-mass frame."""
    m = _m()
    functionals = ([w.functional for w in distributionForm.weighted]
                   if isinstance(distributionForm, m.DistributionMixture) else [distributionForm])
    return any(getattr(f, "productFrame", None) == m.Frame.centerOfMass for f in functionals)


# ----------------------------------------------------------------------
# The suite
# ----------------------------------------------------------------------

class _Encoder:
    def __init__(self, suite, temperature, report, header=None):
        from kika.hdf5.names import temperatureName, temperatureOfLabel, urrLabel
        from kika.hdf5.neutron_encode import heatedTemperatures

        self.m = _m()
        self.suite, self.report = suite, report
        temps = heatedTemperatures(suite, None if temperature is None else [temperature])
        if len(temps) != 1:
            raise AceEncodeError(f"the suite has {len(temps)} temperatures {sorted(temps)}; an ACE table "
                                 f"holds one, so say which (temperature=)")
        (self.T, self.kT), = temps.items()
        self.label, self.urrLabel = f"heated_{self.T}", urrLabel(self.T)
        self.tableAttributes = _attributes(suite)
        self.overrides = header or {}
        self.domain = (0.0, 0.0)

    # --- reactions -------------------------------------------------------

    def nodes(self):
        """Every node with a cross section, by MT (a reaction wins over a sum of the same MT)."""
        s = self.suite
        out = {}
        for container in (s.sums, s.productions, s.fissionComponents, s.reactions):
            for node in container:
                out[int(node.ENDF_MT)] = (container, node)
        return out

    def xs(self, node, what):
        form = node.crossSection.forms.get(self.label) if node.crossSection is not None else None
        if form is None:
            raise AceEncodeError(f"{what} has no {self.label!r} cross section")
        if not isinstance(form, self.m.XYs1d):
            raise AceEncodeError(f"{what}: the {self.label!r} cross section is a {type(form).__name__}, "
                                 f"not an XYs1d on the union grid")
        return form

    def grid(self, nodes) -> np.ndarray:
        longest = None
        for mt, (_, node) in nodes.items():
            form = node.crossSection.forms.get(self.label) if node.crossSection is not None else None
            if isinstance(form, self.m.XYs1d) and (longest is None or len(form.xs) > len(longest)):
                longest = np.asarray(form.xs, dtype=float)
        if longest is None:
            raise AceEncodeError(f"no cross section under {self.label!r}")
        return longest

    def suffix(self, grid, form, what) -> int:
        """IE - 1: where ``form`` starts on the grid, checked to be a suffix of it."""
        x = np.asarray(form.xs, dtype=float)
        start = len(grid) - len(x)
        if start < 0 or not np.array_equal(grid[start:], x):
            raise AceEncodeError(f"{what}: its energies are not a suffix of the union grid "
                                 f"(ACE states a cross section by where it starts, IE)")
        return start

    def mtr(self, nodes, neutronMTs, photonHosts) -> List[int]:
        """MTR's members and order: the decoded table's, else NJOY's rule.

        The rule: every reaction, fission chance and production except the
        KERMA (it is ESZ's heating number), and of the sums only the ones ACER
        keeps (MT 18, 103-107) and the ones a photon yield is stated on.
        """
        s = self.suite
        members = []
        for container, node in nodes.values():
            mt = int(node.ENDF_MT)
            if mt == 2 or mt in (HEATING_MT, LOCAL_HEATING_MT) and container is s.productions:
                continue
            if container is s.sums and mt not in KEPT_SUMS and mt not in photonHosts:
                continue
            members.append(mt)
        stated = self.tableAttributes.get("MTR")
        if stated:
            # The decoded table's MTR decides which sums it carries; a reaction
            # that is not in it was added to the suite since, and goes after it.
            order = [mt for mt in stated if mt in nodes and mt != 2]
            extra = sorted(mt for mt in set(members) - set(order) if nodes[mt][0] is not s.sums)
            if extra:
                self.report.warn(f"MT {extra} are not in the decoded table's MTR; written after it")
            order += extra
        else:
            # ACER (unionx): MT18 when the fission chances carry no
            # distributions, else the chances (MT19 with MF4-6) and not MT18
            chances = [mt for mt in members if nodes[mt][0] is s.fissionComponents]
            if chances:
                if any(mt in neutronMTs for mt in chances):
                    members = [mt for mt in members if not (mt == 18 and nodes[mt][0] is s.sums)]
                else:
                    members = [mt for mt in members if mt not in chances]

            # ACER: the reactions with secondary neutrons, then the rest
            # ascending, then a sum written only to carry photon yields (MT4)
            def addedForPhotons(mt):
                return nodes[mt][0] is s.sums and mt not in KEPT_SUMS

            first = sorted(mt for mt in members if mt in neutronMTs)
            rest = sorted(mt for mt in members if mt not in neutronMTs and not addedForPhotons(mt))
            hosts = sorted(mt for mt in members if mt not in neutronMTs and addedForPhotons(mt))
            order = first + rest + hosts
        # NXS(5): the reactions with secondary neutrons first (contract)
        return [mt for mt in order if mt in neutronMTs] + [mt for mt in order if mt not in neutronMTs]

    # --- the table -------------------------------------------------------

    def encode(self):
        s, m = self.suite, self.m
        nodes = self.nodes()
        grid = self.grid(nodes)
        self.domain = (float(grid[0]), float(grid[-1]))

        # neutron products of every node (the elastic's is the AND elastic record)
        neutrons = {}
        for mt, (_, node) in nodes.items():
            if mt == 2 or node.outputChannel is None:
                continue
            product = next((p for p in node.outputChannel.products if p.pid == "n"), None)
            if product is not None:
                neutrons[mt] = product
        photons = self.photons(nodes)
        # A yield (MFTYPE 12/16) multiplies its MT's σ, which must be in MTR;
        # an MFTYPE 13 states σ_γ itself and needs nothing.
        photonHosts = {mt for mt, _, p in photons if (_photonType(p) or 16) in (12, 16)}
        unr = self.urr()
        if unr is not None and unr.inelastic_flag == 4:
            # ACER keeps MT4 when the URR tables compete with it (ILF = 4)
            photonHosts = photonHosts | {4}
        mtr = self.mtr(nodes, set(neutrons), photonHosts)
        secondary = [mt for mt in mtr if mt in neutrons]

        esz = self.esz(nodes, grid)
        self.totalOnGrid = esz.total_xs
        if LOCAL_HEATING_MT in nodes:
            self.report.lost("MT901, the KERMA with local photons: an ACE table has one heating "
                             "number, ESZ's (MT301, photons deposited elsewhere)")
        crossSection = NS(reaction={})
        q, ty = [], []
        angular = NS(elastic=None, incident_neutron={}, photon_production={}, particle_production=[])
        energy = NS(incident_neutron={}, neutron_yields={}, photon_production={}, delayed_neutron=[],
                    particle_production=[])
        for mt in mtr:
            container, node = nodes[mt]
            form = self.xs(node, f"MT{mt}")
            values = np.asarray(form.ys, dtype=float)
            if mt == 444:
                values = values / EV_PER_MEV
            start = self.suffix(grid, form, f"MT{mt}")
            if not isinstance(s.provenance, AceProvenance):
                # ACER starts a reaction one point before its first nonzero
                # value (a gas production is not written from MT5's threshold)
                first = np.flatnonzero(values)
                if first.size and first[0] > 1:
                    start, values = start + first[0] - 1, values[first[0] - 1:]
            crossSection.reaction[mt] = NS(energy_idx=start, _xs_entries=values)
            qv = node.outputChannel.Q.value if node.outputChannel is not None and \
                node.outputChannel.Q is not None else None
            if qv is None:
                qv = 0.0
                if container is not s.sums:
                    self.report.lost(f"MT{mt}: no Q value in the model; LQR states 0")
            q.append(float(qv) / EV_PER_MEV)
            ty.append(self.neutronBlock(mt, neutrons.get(mt), angular, energy))

        elastic = nodes.get(2)
        if elastic is None:
            raise AceEncodeError("no elastic reaction (MT2); ESZ needs one")
        product = next((p for p in elastic[1].outputChannel.products if p.pid == "n"), None) \
            if elastic[1].outputChannel is not None else None
        if product is None:
            angular.elastic = _angularRecord(None, 2)
        else:
            form = _monteCarlo(product.distribution, "MT2's neutron")
            angular.elastic = _angularRecord(_angularOf(form), 2)

        nubar, bdd, dned = self.fission(nodes)
        energy.delayed_neutron = dned
        gpd, mtrp, sigp, yp = self.photonBlocks(photons, nodes, grid, angular, energy)

        header = self.header()
        ids, mtrh, tyrh, hpd, sigh, yh = self.particleBlocks(nodes, grid, angular, energy,
                                                           float(header.atomic_weight_ratio))
        rmt = NS(incident_neutron=np.asarray(mtr, dtype=float),
                 secondary_neutron_mt=np.asarray(secondary, dtype=float),
                 photon_production=np.asarray(mtrp, dtype=float), particle_production=mtrh)
        return NS(header=header, esz_block=esz, nubar=nubar, reaction_mt_data=rmt,
                  q_values=NS(q_values=np.asarray(q, dtype=float)),
                  particle_release=NS(incident_neutron=np.asarray(ty, dtype=float), particle_production=tyrh),
                  cross_section=crossSection, angular_distributions=angular, energy_distributions=energy,
                  unresolved_resonance=unr, delayed_neutron_data=bdd,
                  photon_production_data=gpd, photon_production_xs=NS(cross_sections=sigp) if gpd else None,
                  photon_yield_multipliers=NS(multiplier_mts=yp) if gpd else None,
                  secondary_particle_types=NS(particle_ids=ids),
                  secondary_particle_cross_sections=NS(particle_data=hpd) if ids else None,
                  particle_production_xs=NS(particle_cross_sections=sigh) if ids else None,
                  particle_yield_multipliers=NS(particle_multipliers=yh) if ids else None)

    def esz(self, nodes, grid) -> NS:
        nes = len(grid)

        def full(mt, what):
            entry = nodes.get(mt)
            if entry is None:
                return None
            form = self.xs(entry[1], what)
            start = self.suffix(grid, form, what)
            out = np.zeros(nes)
            out[start:] = form.ys
            return out

        elastic = full(2, "MT2")
        total = full(1, "MT1")
        if total is None:
            total = elastic + sum((full(mt, f"MT{mt}") for mt in self.partials(nodes) if mt != 2),
                                  np.zeros(nes))
            self.report.approximated("ESZ's total is the sum of the partial reactions (the suite has no MT1)")
        absorption = full(101, "MT101")
        if absorption is None:
            from kika.hdf5.names import isDisappearance

            absorption = sum((full(mt, f"MT{mt}") for mt in self.partials(nodes) if isDisappearance(mt)),
                             np.zeros(nes))
        heating = np.zeros(nes)
        kerma = full(HEATING_MT, "MT301")
        if kerma is None:
            self.report.lost("the suite has no KERMA (MT301): ESZ's heating numbers are 0")
        else:
            kerma = kerma - self.delayedBetaKerma(nodes, grid, full)
            nonzero = total != 0.0
            heating[nonzero] = kerma[nonzero] / (total[nonzero] * EV_PER_MEV)
        return NS(energies=_mev(grid), total_xs=total, absorption_xs=absorption, elastic_xs=elastic,
                  heating_numbers=heating, has_data=True)

    def delayedBetaKerma(self, nodes, grid, full) -> np.ndarray:
        """EB·σ_f, the delayed betas OpenMC's MT301 counts and HEATR's does not.

        ESZ's heating number is ACER's, HEATR's ``local = 0`` KERMA / σ_t, whose
        fission part is the fragments' kinetic energy (EFR). A processed suite's
        MT301 is OpenMC's, which adds the delayed betas (EB, ~6.5 MeV for U-235,
        3.8 % of the fission KERMA): they are taken out again. A suite read from
        ACE carries ACER's own number, and is left as it is.
        """
        if isinstance(self.suite.provenance, AceProvenance):
            return np.zeros(len(grid))
        fission = nodes.get(18)
        release = None
        for mt in FISSION_MTS:
            node = nodes.get(mt)
            data = getattr(getattr(node[1], "outputChannel", None), "fissionFragmentData", None) if node else None
            if data is not None and data.fissionEnergyReleases:
                release = data.fissionEnergyReleases[0]
                break
        if release is None or getattr(release, "delayedBetaEnergy", None) is None:
            return np.zeros(len(grid))
        from kika.hdf5.fission_energy import evaluate

        sigmaF = full(18, "MT18") if fission is not None else sum(
            (full(mt, f"MT{mt}") for mt in (19, 20, 21, 38) if mt in nodes), np.zeros(len(grid)))
        self.report.warn("ESZ's heating number is HEATR's local = 0 KERMA: MT301 without the delayed "
                         "betas (EB) OpenMC's fission KERMA adds")
        return np.asarray(evaluate(release.delayedBetaEnergy, grid), dtype=float) * sigmaF

    def partials(self, nodes):
        s = self.suite
        hasFission = any(int(r.ENDF_MT) == 18 for r in s.reactions)
        out = [int(r.ENDF_MT) for r in s.reactions]
        if not hasFission:
            out += [int(r.ENDF_MT) for r in s.fissionComponents]
        return out

    # --- neutrons --------------------------------------------------------

    def neutronBlock(self, mt, product, angular, energy) -> int:
        """TY of one MTR reaction, filling AND and DLW when it has secondary neutrons."""
        m = self.m
        if product is None:
            return 0
        attributes = _attributes(product)
        form = _monteCarlo(product.distribution, f"MT{mt}'s neutron")
        laws, numbers = _chain(form, attributes, f"DLW MT{mt}", self.domain)
        energy.incident_neutron[mt] = laws
        if any(n in CORRELATED_LAWS for n in numbers):
            angular.incident_neutron[mt] = _correlatedRecord(mt)
        else:
            angular.incident_neutron[mt] = _angularRecord(_angularOf(form), mt)
        sign = -1 if _frameOf(form) else 1
        if mt in FISSION_MTS and self.isFissionNeutron(product):
            return 19 * sign
        multiplicity = _multiplicityForm(product.multiplicity)
        if isinstance(multiplicity, m.Polynomial1d) and len(multiplicity.coefficients) == 1:
            n = float(multiplicity.coefficients[0])
            if n != int(n):
                raise AceEncodeError(f"MT{mt}: a constant multiplicity {n} is not an integer TY")
            return sign * int(n)
        if isinstance(multiplicity, m.Constant1d):
            return sign * int(multiplicity.constant)
        x, y, nbt, interp = _regions(multiplicity)
        values = np.unique(np.asarray(y, dtype=float))
        if not isinstance(self.suite.provenance, AceProvenance) and values.size == 1 and                 values[0] == int(values[0]) and values[0] > 0:
            # a constant integer yield (Be-9's MT16: 2) is TY, as ACER writes it
            return sign * int(values[0])
        energy.neutron_yields[mt] = NS(nbt=nbt, interp=interp, energies=_mev(x), yields=y)
        return sign * 101

    def isFissionNeutron(self, product) -> bool:
        attributes = _attributes(product)
        if "TY" in attributes:
            return abs(int(attributes["TY"])) == 19
        return True

    def fission(self, nodes):
        """NU, DNU, BDD and the DNED chains, from the fission neutron and MT 452/455."""
        m, s = self.m, self.suite
        fission = next((nodes[mt][1] for mt in FISSION_MTS if mt in nodes
                        and any(p.pid == "n" for p in (nodes[mt][1].outputChannel.products or []))), None)
        if fission is None:
            return None, None, []
        neutron = next(p for p in fission.outputChannel.products if p.pid == "n")
        nu = _multiplicityForm(neutron.multiplicity)
        totalSum = s.sums.multiplicitySums.byLabel(TOTAL_NUBAR_LABEL) or s.sums.multiplicitySums.byENDF_MT(452)
        delayedSum = s.sums.multiplicitySums.byLabel(DELAYED_NUBAR_LABEL) or \
            s.sums.multiplicitySums.byENDF_MT(455)
        mode = _attributes(neutron).get("emission_mode")
        groups = list(fission.outputChannel.fissionFragmentData.delayedNeutrons) \
            if fission.outputChannel.fissionFragmentData is not None else []
        if mode is None:
            mode = "prompt" if (totalSum is not None or groups) else "total"
        prompt = total = None
        if mode == "prompt" and totalSum is not None:
            prompt, total = _nuRecord(nu), _nuRecord(_multiplicityForm(totalSum.multiplicity))
        else:
            total = _nuRecord(nu)
        delayed = None
        dnu = None
        if groups:
            if delayedSum is not None:
                dnu = _multiplicityForm(delayedSum.multiplicity)
            else:
                # An OpenMC file states each group's yield, not MF1/455: the
                # delayed nu is their sum
                dnu = _summedYields([_multiplicityForm(g.product.multiplicity) for g in groups])
                self.report.approximated("DNU is the sum of the delayed groups' yields "
                                         "(the suite states no delayed nu, MT455)")
            delayed = _nuRecord(dnu)
        nubar = NS(has_nubar=True, has_both_nu_types=prompt is not None, prompt=prompt, total=total,
                   has_delayed=delayed is not None, delayed=delayed)
        if not groups:
            return nubar, None, []
        precursors, chains = [], []
        for g, group in enumerate(groups):
            attributes = _attributes(group.product)
            if "energies" in attributes:
                energies = np.asarray(attributes["energies"], dtype=float)
                probabilities = np.asarray(attributes["probabilities"], dtype=float)
                regions = [tuple(r) for r in attributes.get("interpolation_regions", [])]
            else:
                form = _multiplicityForm(group.product.multiplicity)
                x, y, _, _ = _regions(form)
                d = np.asarray(dnu.evaluate(x), dtype=float)
                probabilities = np.divide(y, d, out=np.zeros_like(y), where=d != 0)
                energies, regions = _mev(x), []
                if probabilities.size > 2 and np.allclose(probabilities, probabilities[0], rtol=1e-12, atol=0):
                    # a constant fraction (MF1/455 LDG = 0): two points, as ACER writes it
                    energies = energies[[0, -1]]
                    probabilities = probabilities[[0, -1]]
            precursors.append(NS(decay_constant=float(group.rate.value) / PER_SHAKE_TO_PER_S,
                                 interpolation_regions=regions, energies=energies, probabilities=probabilities))
            form = _monteCarlo(group.product.distribution, f"delayed group {g + 1}")
            laws, _ = _chain(form, attributes, f"DNED group {g + 1}", self.domain)
            chains.append(laws)
        return nubar, NS(has_delayed_neutron_data=True, precursors=precursors), chains

    def urr(self):
        m = self.m
        for mt in (2, 102, 18):
            node = self.suite.findReactionByENDF_MT(mt)
            form = node.crossSection.forms.get(self.urrLabel) if node is not None and \
                node.crossSection is not None else None
            if form is None:
                continue
            t = form.table
            table = np.asarray(t.table, dtype=float)
            tables = [NS(energy=float(e) / EV_PER_MEV, cumulative_probabilities=row[0], total_xs=row[1],
                         elastic_xs=row[2], fission_xs=row[3], capture_xs=row[4],
                         heating_numbers=row[5] / EV_PER_MEV)
                      for e, row in zip(t.energies, table)]
            return NS(has_data=True, energies=_mev(t.energies), table_length=table.shape[2],
                      interpolation=int(t.interpolation), inelastic_flag=int(t.inelastic),
                      other_absorption_flag=int(t.absorption), factors_flag=1 if t.multiplySmooth else 0,
                      tables=tables)
        return None

    # --- photons ---------------------------------------------------------

    def photons(self, nodes):
        """``[(host MT, host node, product)]`` for every photon, orphans included."""
        out = []
        for mt, (_, node) in nodes.items():
            if node.outputChannel is None:
                continue
            for product in node.outputChannel.products:
                if product.pid == "photon":
                    out.append((mt, node, product))
        for orphan in self.suite.orphanProducts:
            for product in orphan.outputChannel.products:
                if product.pid == "photon":
                    out.append((int(orphan.ENDF_MT), orphan, product))
        return out

    def photonBlocks(self, photons, nodes, grid, angular, energy):
        if not photons:
            if getattr(self.suite, "photonProductionTotal", None) is not None:
                self.report.lost("the suite states a total photon production but no photon; no GPD written")
            return None, [], {}, []
        orphans = {id(o) for o in self.suite.orphanProducts}
        entries, counters = [], {}
        for mt, node, product in photons:
            attributes = _attributes(product)
            mtrp = attributes.get("MTRP")
            if mtrp is None:
                counters[mt] = counters.get(mt, 0) + 1
                mtrp = 1000 * mt + counters[mt]
            entries.append((int(mtrp), mt, node, product, attributes))
        stated = self.tableAttributes.get("MTRP")
        if stated:
            rank = {v: i for i, v in enumerate(stated)}
            entries.sort(key=lambda e: (rank.get(e[0], len(rank)), e[0]))
        sigp, mtrps, multipliers = {}, [], []
        for mtrp, mt, node, product, attributes in entries:
            mtrps.append(mtrp)
            mftype = _photonType(product) or (13 if id(node) in orphans else 16)
            multiplicity = _multiplicityForm(product.multiplicity)
            if mftype in (12, 16):
                x, y, nbt, interp = _regions(multiplicity)
                sigp[mtrp] = NS(mt=mtrp, mftype=mftype, mtmult=mt, interpolation_bounds=nbt,
                                interpolation_schemes=interp, energies=_mev(x), yields=y)
                multipliers.append(mt)
            else:
                host = self.xs(node, f"{'orphan ' if id(node) in orphans else ''}MT{mt}")
                if isinstance(multiplicity, self.m.Constant1d):
                    xs = np.asarray(host.xs, dtype=float)
                    sigma = float(multiplicity.constant) * np.asarray(host.ys, dtype=float)
                else:
                    xs = np.asarray(multiplicity.xs, dtype=float)
                    sigma = np.asarray(multiplicity.ys, dtype=float) * np.interp(xs, host.xs, host.ys)
                    for i, value in (attributes.get("sigmaGammaWhereSigmaIsZero") or {}).items():
                        sigma[int(i)] = value
                start = _position(grid, xs)
                if start is None:
                    # A processed suite's yield σ_γ/σ is on the evaluation's
                    # MF13 grid: σ_γ on the union grid from the reaction's
                    # threshold, as ACER puts MF13 on it
                    start = self.suffix(grid, host, f"MT{mt}")
                    xs = grid[start:]
                    y = np.interp(xs, multiplicity.xs, multiplicity.ys, left=0.0, right=0.0)
                    sigma = y * np.asarray(host.ys, dtype=float)
                sigp[mtrp] = NS(mt=mtrp, mftype=13, energy_grid_index=start + 1, cross_sections=sigma)
            mc = self.photonMonteCarlo(product, attributes, mtrp)
            laws, numbers = _chain(mc, attributes, f"DLWP MTRP {mtrp}", self.domain)
            energy.photon_production[mtrp] = laws
            angular.photon_production[mtrp] = _angularRecord(_angularOf(mc), mtrp, photon=True)

        yp = self.tableAttributes.get("YP")
        if yp is None:
            yp = sorted(set(multipliers))
        total = getattr(self.suite, "photonProductionTotal", None)
        if total is not None:
            x = np.asarray(total.xs, dtype=float)
            if not np.array_equal(x, grid):
                raise AceEncodeError("the total photon production is not on the union grid")
            gpdTotal = np.asarray(total.ys, dtype=float)
        else:
            gpdTotal = np.zeros(len(grid))
            for mtrp, entry in sigp.items():
                if entry.mftype == 13:
                    i = entry.energy_grid_index - 1
                    gpdTotal[i:i + len(entry.cross_sections)] += entry.cross_sections
                else:
                    host = self.xs(nodes[entry.mtmult][1], f"MT{entry.mtmult}")
                    y = np.interp(grid, entry.energies * EV_PER_MEV, entry.yields, left=0.0, right=0.0)
                    gpdTotal += y * np.interp(grid, host.xs, host.ys, left=0.0, right=0.0)
            self.report.approximated("GPD is the sum of the photon productions (the suite states no total)")
        return NS(total_xs=gpdTotal, outgoing_energies=None), mtrps, sigp, yp

    # --- light charged particles (PTYPE, NTRO, IXS) --------------------------

    def particleBlocks(self, nodes, grid, angular, energy, awr: float):
        """The charged-particle production blocks, as NJOY 2016's ACER writes them.

        Per particle (PTYPE 9, 31, 32, 33, 34 in that order): MTRH, the
        reactions producing it, ascending; TYRH +-1 by frame (-1 in the centre
        of mass); SIGH, each yield as MFTYPE 12 on its own reaction; ANDH and
        DLWH from its Monte Carlo form; HPD, the production cross section
        sum(y sigma) on the union grid from one point before its first nonzero
        value, with the heating number, the produced particles' energy per collision
        in MeV, sum(y sigma E) / sigma_t, normalised as ESZ's; YH, the
        reactions with a yield.
        """
        from kika.hdf5.names import MONTE_CARLO_LABEL
        from kika.processing.heating import _Kinematics, _meanEnergyOf

        s, m = self.suite, self.m
        ids, mtrh, tyrh, hpd, sigh, yh = [], [], [], {}, {}, {}
        angular.particle_production, energy.particle_production = [], []
        for pid, (ipt, awp) in PARTICLE_TYPES.items():
            entries = []
            for mt in sorted(nodes):
                container, node = nodes[mt]
                if container is s.sums or node.outputChannel is None or mt == HEATING_MT:
                    continue
                product = next((p for p in node.outputChannel.products if p.pid == pid), None)
                if product is not None and product.distribution is not None and \
                        MONTE_CARLO_LABEL in product.distribution.forms:
                    entries.append((mt, node, product))
            if not entries:
                continue
            i = len(ids)
            ids.append(ipt)
            mts, tys, records, angles, chains = [], [], {}, {}, {}
            production, heat = np.zeros(len(grid)), np.zeros(len(grid))
            for mt, node, product in entries:
                form = product.distribution.forms[MONTE_CARLO_LABEL]
                # a level's particle is law 33, the charged-particle level law
                level = isinstance(form, m.Uncorrelated) and isinstance(form.energy, m.LevelEnergy)
                laws, numbers = _chain(form, {"laws": [33]} if level else {}, f"DLWH {pid} MT{mt}",
                                       self.domain)
                chains[mt] = laws
                angles[mt] = (_correlatedRecord(mt) if any(n in CORRELATED_LAWS for n in numbers)
                              else _angularRecord(_angularOf(form), mt))
                mts.append(mt)
                tys.append(-1 if _frameOf(form) else 1)
                xs = self.xs(node, f"MT{mt}")
                x0 = np.asarray(xs.xs, dtype=float)
                multiplicity = _multiplicityForm(product.multiplicity)
                if isinstance(multiplicity, (m.Constant1d, m.Polynomial1d)):
                    c = float(multiplicity.constant if isinstance(multiplicity, m.Constant1d)
                              else multiplicity.coefficients[0])
                    x, y, nbt, interp = np.array([x0[0], x0[-1]]), np.array([c, c]), [], []
                else:
                    x, y, nbt, interp = _regions(multiplicity)
                records[mt] = NS(mt=mt, mftype=12, mtmult=mt, interpolation_bounds=nbt,
                                 interpolation_schemes=interp, energies=_mev(x),
                                 yields=np.asarray(y, dtype=float))
                sigma = np.zeros(len(grid))
                sigma[self.suffix(grid, xs, f"MT{mt}"):] = xs.ys
                contribution = np.interp(grid, x, y, left=0.0, right=0.0) * sigma
                production += contribution
                Q = node.outputChannel.Q.value if node.outputChannel.Q is not None else 0.0
                try:
                    mean = _meanEnergyOf(form, grid, _Kinematics(A=awr, Q=float(Q), awp=awp))
                except (TypeError, ValueError, NotImplementedError) as error:
                    self.report.approximated(f"HPD {pid}: MT{mt}'s mean energy not computed ({error}); "
                                             f"its heating counted as 0")
                    mean = np.zeros(len(grid))
                heat += contribution * mean
            first = np.flatnonzero(production)
            start = max(int(first[0]) - 1, 0) if first.size else 0
            total = np.asarray(self.totalOnGrid, dtype=float)
            heating = np.divide(heat, total, out=np.zeros(len(grid)), where=total > 0)
            hpd[i + 1] = NS(energy_grid_index=start + 1, xs_values=production[start:],
                            heating_numbers=heating[start:] / EV_PER_MEV)
            mtrh.append(np.asarray(mts, dtype=float))
            tyrh.append(np.asarray(tys, dtype=float))
            sigh[i + 1] = records
            yh[i + 1] = list(mts)
            angular.particle_production.append(angles)
            energy.particle_production.append(chains)
        return ids, mtrh, tyrh, hpd, sigh, yh

    def photonMonteCarlo(self, product, attributes, mtrp):
        """The photon's Monte Carlo form, else its heated ``Uncorrelated`` (lines and continua)."""
        from kika.hdf5.names import MONTE_CARLO_LABEL

        forms = product.distribution.forms if product.distribution is not None else {}
        if MONTE_CARLO_LABEL in forms:
            return forms[MONTE_CARLO_LABEL]
        if self.label in forms:
            return forms[self.label]
        raise AceEncodeError(f"MTRP {mtrp}: the photon has neither a {MONTE_CARLO_LABEL!r} nor a "
                             f"{self.label!r} distribution")

    # --- header ----------------------------------------------------------

    def header(self):
        from kika.ace.classes.header import Header

        s = self.suite
        p = s.provenance if isinstance(s.provenance, AceProvenance) else None
        target = s.PoPs[s.target] if s.target in s.PoPs else None
        known = target is not None and getattr(target, "Z", None) is not None
        zaid = self.overrides.get("zaid") or getattr(p, "zaid", None)
        if zaid is None and known:
            # An isomer as LANL writes it for MCNP, A + 300 + 100 m (Am-242m1
            # 95642), which MCNP and OpenMC read as that state
            from kika.hdf5.processing import metastableIndex

            meta = metastableIndex(s)
            zaid = 1000 * int(target.Z) + int(target.A) + (300 + 100 * meta if meta else 0)
        awr = self.overrides.get("awr") or getattr(p, "awr", None)
        if awr is None:
            # a suite kika processed states MF1/451's AWR, as ACER takes it
            awr = (getattr(s.provenance, "attributes", None) or {}).get("atomic_weight_ratio")
        if awr is None and known and getattr(target, "mass", None) is not None:
            from kika._constants import NEUTRON_MASS_AMU

            awr = float(target.mass.value) / NEUTRON_MASS_AMU
        if zaid is None or awr is None:
            raise AceEncodeError("the suite states no ZAID or AWR (its target has no Z, A or mass in "
                                 "PoPs); pass zaid= and awr=")
        h = Header()
        h.format_version = getattr(p, "formatVersion", None) or "legacy"
        h.zaid, h.extension = int(zaid), self.overrides.get("extension") or getattr(p, "extension", None)
        h.atomic_weight_ratio = float(awr)
        h.temperature = self.kT / EV_PER_MEV
        h.date, h.comment, h.matid = getattr(p, "date", None), getattr(p, "comment", None), getattr(p, "matid", None)
        if p is None:
            # A table kika processed: written today, titled by its target and
            # evaluation, and named by the tape's MAT, as ACER heads one
            import datetime

            from kika import __version__

            h.date = datetime.date.today().strftime("%m/%d/%y")
            h.comment = " ".join(x for x in (s.target, s.evaluation, f"(kika {__version__})") if x)[:70]
            h.matid = (getattr(s.provenance, "attributes", None) or {}).get("mat")
        h.izaw_array = [(0, 0.0)] * 16
        h.nxs_array = [0] * 17
        # NXS(2): the ground state's ZA, also for an isomer's ZAID
        za = self.tableAttributes.get("ZA")
        if za is None and target is not None and getattr(target, "Z", None) is not None:
            za = 1000 * int(target.Z) + int(target.A)
        h.nxs_array[2] = int(za if za is not None else zaid)
        # NXS(9..16): the decoded table's, else NJOY 2016's Z and A in NXS(10)/(11)
        stated = self.tableAttributes.get("NXS")
        h.nxs_array[9:17] = list(stated) if stated else [0, int(zaid) // 1000, int(zaid) % 1000, 0, 0, 0, 0, 0]
        h.jxs_array = [0] * 33
        return h


def _defaultExtension(kT: float) -> str:
    """The ZAID suffix of a table kika processed itself: its temperature's, ``.02c`` for 293.6 K.

    The suffix kika's NJOY runs give (:data:`kika._constants.K_TO_SUFFIX`), so
    a table from kika's processing takes the place and the name of the one
    NJOY made; the library goes in the file name and the xsdir line, not in
    the ZAID. A temperature with no suffix raises: pass ``extension=``.
    """
    from kika._utils import ace_temperature_suffix
    from kika.hdf5.names import K_BOLTZMANN

    try:
        return ace_temperature_suffix(kT / K_BOLTZMANN) + "c"
    except ValueError as e:
        raise AceEncodeError(str(e)) from None


def _position(grid: np.ndarray, xs: np.ndarray) -> Optional[int]:
    """Where ``xs`` sits on the grid as a contiguous slice; the grid may repeat an energy."""
    for start in np.flatnonzero(grid == xs[0]):
        if np.array_equal(grid[start:start + len(xs)], xs):
            return int(start)
    return None


def _summedYields(forms):
    """Σ of the groups' yields: polynomials coefficient by coefficient, else lin-lin on the union grid."""
    m = _m()
    if all(isinstance(f, m.Polynomial1d) for f in forms):
        n = max(len(f.coefficients) for f in forms)
        c = np.zeros(n)
        for f in forms:
            c[:len(f.coefficients)] += np.asarray(f.coefficients, dtype=float)
        return m.Polynomial1d(coefficients=c, domainMin_=forms[0].domainMin, domainMax_=forms[0].domainMax)
    x = np.unique(np.concatenate([np.asarray(f.xs, dtype=float) for f in forms if hasattr(f, "xs")]))
    y = sum(np.asarray(f.evaluate(x), dtype=float) for f in forms)
    return m.XYs1d(xs=x, ys=y)


def _nuRecord(form):
    """A ν form → the parser's ν record: LNU = 1 polynomial (coefficients per MeV^i) or 2 tabulated."""
    m = _m()
    if isinstance(form, m.Polynomial1d):
        c = np.array(form.coefficients, dtype=float)
        for i in range(c.size):
            c[i] *= EV_PER_MEV ** i
        return NS(format="polynomial", polynomial=NS(coefficients=c))
    x, y, nbt, interp = _regions(form)
    return NS(format="tabulated", tabulated=NS(interpolation_regions=list(zip(nbt, interp)), energies=_mev(x),
                                               nubar_values=y))


def encodeAce(suite, temperature=None, *, zaid: Optional[int] = None, awr: Optional[float] = None,
              extension: Optional[str] = None, report: Optional[ConversionReport] = None, strict: bool = True):
    """A processed suite → ``(Ace, report)`` at one temperature.

    ``temperature`` names the ``heated_<T>`` style to write (a name such as
    ``"294K"`` or kelvins) when the suite has more than one. ``zaid``,
    ``awr`` and ``extension`` override the header the suite's
    ``AceProvenance`` states. A suite kika processed itself has none: its ZAID
    is 1000·Z + A and its suffix its temperature's (``.02c`` for 293.6 K, the
    one kika's NJOY runs give; :func:`kika._utils.ace_temperature_suffix`),
    and a temperature with no suffix raises. The table is laid out by the block encoders and read
    back with the parser in ``strict`` mode, so a block that would not read
    raises here rather than in the code that opens the file.
    """
    from kika.ace.classes.ace import Ace
    from kika.ace.classes.read_report import ReadReport
    from kika.ace.classes.xss import pack_int_words
    from kika.ace.encoders.layout import encode_xss
    from kika.ace.parsers.parse_ace import parse_blocks

    report = report if report is not None else ConversionReport()
    encoder = _Encoder(suite, temperature, report,
                       header={"zaid": zaid, "awr": awr, "extension": extension})
    view = encoder.encode()
    if view.header.extension is None:
        view.header.extension = _defaultExtension(encoder.kT)
    enc = encode_xss(view)
    ace = Ace()
    ace.header = copy.deepcopy(view.header)
    ace.header.nxs_array = enc.nxs
    ace.header.jxs_array = enc.jxs
    ace.xss_data = enc.xss
    ace.xss_int_words = pack_int_words(enc.is_int)
    ace.read_report = ReadReport(strict=strict)
    ace = parse_blocks(ace)
    if not list(getattr(ace.secondary_particle_types, "particle_ids", None) or []):
        message = ("charged-particle production (PTYPE, NTRO, IXS) is not written: the suite carries no "
                   "light charged-particle products (a suite read from ACE or OpenMC has none)")
        if message not in report.unsupported:
            report.unsupportedNode(message)
    return ace, report


def writeAceSuite(suite, path, temperature=None, **options) -> ConversionReport:
    """``kika.write(suite, path, format="ace")``: the table at ``path``, and the report.

    The suite must be processed (``heated_<T>`` styles on a union grid with
    Monte Carlo forms); an evaluation raises :class:`AceEncodeError` rather than
    being processed here, since an ACE table is one temperature of a processing
    run (:meth:`kika.endf.EndfProcessing.write_ace`). The file is read back in
    strict mode before it replaces ``path``.
    """
    from pathlib import Path

    from kika.ace.writers.write_ace import write_ace_checked
    from kika.hdf5.writer import isProcessed

    if getattr(suite, "interaction", None) == "thermalNeutronScatteringLaw":
        from kika.nuclear_data.model import ConversionReport as Report
        from kika.tsl.ace_writer import write_thermal_suite

        write_thermal_suite(suite, path, temperature, **options)
        return Report()
    if not isProcessed(suite):
        raise AceEncodeError("ace-needs-processed-suite: an ACE table is a processed suite at one "
                             "temperature; process the evaluation first (kika.endf.process_endf(...)"
                             ".write_ace(directory), or kika.hdf5.processing.processSuite)")
    ace, report = encodeAce(suite, temperature, **options)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_ace_checked(ace, path)
    return report
