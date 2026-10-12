"""Thermal incoherent inelastic scattering tables from S(alpha, beta) (roadmap H9c).

kika's own THERMR + ACER for the ``incoherent_inelastic`` group of an OpenMC
thermal file: from the model's :class:`~kika.nuclear_data.model.IncoherentInelastic`
law at one temperature, the inelastic cross section on THERMR's fixed incident
grid and, at each incident energy, the outgoing-energy table with its pdf and
cdf and the 16 equiprobable cosines at each outgoing energy -- the "continuous"
representation (ACER ``iwt = 2``) the official libraries carry. The reading of
the Fortran is ``docs/library/thermal_inelastic_plan.md`` in kika-workspace.

- **THERMR** (``thermr.f90`` ``calcem``/``sigl``/``sig``/``terpq``) runs in C
  (``kika/processing/_thermal_native.c``), ported line for line: the
  quadratic interpolation of ln S, the liquid (diffusion) branch, the
  short-collision-time tail, the adaptive outgoing-energy and cosine grids
  with their ``sigfig`` roundings.
- **ACER** (``aceth.f90`` ``acesix``, ``iwt = 2``) is the merge of every
  interval whose accumulated area is below 1e-6 b into the next, and the
  normalisation to pdf and cdf; the cosines of the first and last outgoing
  points are copied from their neighbours.
- **OpenMC** (``thermal.py`` ``from_ace``): cosines sorted, and a 0 eV point with
  isotropic cosines when the cdf does not start at zero.

The cross section is THERMR's integral at each incident energy. THERMR also
writes it, Lagrange-interpolated, on the principal nuclide's PENDF grid, and
ACER reads it back from there; that round trip is not repeated here.

**Free gas** (THERMR ``iinc = 1``, :func:`freeGasTables`): adaptive calcem integration
and a consistent PDF/CDF on the closed-form free-gas S(alpha, beta) of a nucleus of mass ratio
``awr``, normalised to a free cross section of 1 b, over THERMR's fixed
45-point beta grid. As THERMR does, the cross section is not the kernel's
integral but the nuclide's elastic cross section at the temperature (BROADR's
MT2, which THERMR copies to the inelastic MT), read lin-lin at the incident
energies as ACER reads it. The reference that checks it is independent of
this code (``kika-workspace/kika_dev/processing/beyond_njoy/thermal_freegas``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["ThermalInelasticOptions", "EGRID", "FREE_GAS_BETA", "inelasticTables",
           "freeGasTables", "thermalInputs", "freeGasInputs"]

#: THERMR's incident-energy grid (``thermr.f90:1587-1608``), eV.
EGRID = np.array([
    1.e-5, 1.78e-5, 2.5e-5, 3.5e-5, 5.0e-5, 7.0e-5, 1.e-4, 1.26e-4, 1.6e-4, 2.0e-4, .000253,
    .000297, .000350, .00042, .000506, .000615, .00075, .00087, .001012, .00123, .0015, .0018,
    .00203, .002277, .0026, .003, .0035, .004048, .0045, .005, .0056, .006325, .0072, .0081,
    .009108, .01, .01063, .0115, .012397, .0133, .01417, .015, .016192, .0182, .0199, .020493,
    .0215, .0228, .0253, .028, .030613, .0338, .0365, .0395, .042757, .0465, .050, .056925,
    .0625, .069, .075, .081972, .09, .096, .1035, .111573, .120, .128, .1355, .145728, .160,
    .172, .184437, .20, .2277, .2510392, .2705304, .2907501, .3011332, .3206421, .3576813,
    .39, .4170351, .45, .5032575, .56, .625, .70, .78, .86, .95, 1.05, 1.16, 1.28, 1.42, 1.55,
    1.70, 1.855, 2.02, 2.18, 2.36, 2.59, 2.855, 3.12, 3.42, 3.75, 4.07, 4.46, 4.90, 5.35,
    5.85, 6.40, 7.00, 7.65, 8.40, 9.15, 9.85, 10.00])

#: THERMR's fixed beta grid for the free gas (``thermr.f90:1863-1907``): the
#: outgoing-energy panels start at E +- beta kT, then refine adaptively.
FREE_GAS_BETA = np.array([
    0.0, 0.1, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0,
    65.0, 70.0, 75.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 250.0, 300.0, 350.0,
    400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0, 2250.0,
    2500.0, 2750.0, 3000.0, 3500.0])

#: NJOY's Boltzmann constant (``phys.f90``) and THERMR's kT0.
BOLTZMANN_EV = 8.617333262e-5
KT0 = 0.0253
SABFLG = -225.0


@dataclass(frozen=True)
class ThermalInelasticOptions:
    """THERMR's ``tol``, ``nbin`` and ``iinc`` as OpenMC's ``make_ace_thermal`` passes them.

    ``iinc`` is THERMR's inelastic option: 2 the evaluation's S(alpha, beta),
    1 the free gas of the principal nuclide (:func:`freeGasTables`; there is
    no scattering law to read, so the input is the nuclide's neutron tape).
    """

    tol: float = 0.001
    nbin: int = 16
    iinc: int = 2
    #: THERMR's card-4 ``emax`` (eV) for the free gas; ``None`` is 10 eV, the
    #: top of THERMR's incident grid. The S(alpha, beta) path takes the
    #: evaluation's B(4) and does not accept an override yet.
    emax: Optional[float] = None
    #: Outgoing points one incident energy may have.
    capacity: int = 50000
    #: Threads for the incident energies (``calcem`` releases the GIL; each
    #: energy is independent, so the tables do not depend on it). ``None``:
    #: up to 8, bounded by the machine.
    workers: Optional[int] = None


@dataclass
class _Inputs:
    alpha: np.ndarray
    beta: np.ndarray
    sab: np.ndarray          # ln S, alpha fastest: sab[ib*na + ia]
    az: float
    sb: float
    sb2: float
    az2: float
    teff: float
    teff2: float
    lat: int
    lasym: int
    cliq: float
    emax: float
    tev: float
    iinc: int = 2


def _native():
    from . import _thermal_native

    return _thermal_native


def _valueAt(function, x: float) -> float:
    return float(np.asarray(function.evaluate([x]), dtype=float)[0])


def _massRatio(atom) -> float:
    from kika._constants import NEUTRON_MASS_AMU

    return float(atom.mass.convertedTo("amu").value) / NEUTRON_MASS_AMU


def thermalInputs(law, temperature: float, awr: Optional[float] = None) -> _Inputs:
    """THERMR's inputs at one tabulated temperature (``calcem``, ``thermr.f90:1655-1964``).

    ``awr`` overrides the principal atom's mass ratio (ENDF B(3), exactly as the
    tape states it; the model keeps it as a mass in amu).
    """
    from kika.nuclear_data import model as m

    atom = law.principal
    kernel = atom.selfScatteringKernel.kernel
    if not isinstance(kernel, m.Gridded3d):
        raise ValueError(f"the principal atom's kernel is a {type(kernel).__name__}, not a "
                         f"tabulated S(alpha, beta, T)")
    temperatures = np.asarray(kernel.grids[0].values, dtype=float)
    beta = np.asarray(kernel.grids[1].values, dtype=float)
    alpha = np.asarray(kernel.grids[2].values, dtype=float)
    match = np.flatnonzero(np.abs(temperatures - temperature) <= temperature / 500.0)
    if not match.size:
        raise ValueError(f"S(alpha, beta) is not tabulated at {temperature} K (THERMR does not "
                         f"interpolate in temperature); it is at {temperatures.tolist()}")
    values = np.asarray(kernel.values, dtype=float).reshape(len(temperatures), len(beta), len(alpha))
    S = values[match[0]]
    with np.errstate(divide="ignore"):
        lnS = np.where(S > np.exp(SABFLG), np.log(np.where(S > 0, S, 1.0)), SABFLG)
    sab = np.ascontiguousarray(lnS, dtype=float).ravel()       # [ib, ia], alpha fastest
    az = float(awr) if awr is not None else _massRatio(atom)
    sb = float(atom.boundAtomCrossSection.convertedTo("b").value)
    teff = _valueAt(atom.T_effective, temperature) * BOLTZMANN_EV if atom.T_effective is not None \
        else temperature * BOLTZMANN_EV
    sb2 = az2 = 0.0
    teff2 = 0.0
    for other in law.scatteringAtoms:
        if other is atom:
            continue
        if isinstance(other.selfScatteringKernel.kernel, m.SCTApproximation):
            if sb2:
                raise ValueError("THERMR allows two SCT atoms at most")
            az2 = _massRatio(other)
            # THERMR: sb2 = B(8)·((az2+1)/az2)² / B(12) / natom -- the bound cross
            # section of the secondary per principal atom, without M2.
            sb2 = float(other.boundAtomCrossSection.convertedTo("b").value) \
                / max(int(other.numberPerMolecule), 1) / max(int(atom.numberPerMolecule), 1) \
                * max(int(other.numberPerMolecule), 1)
            teff2 = (_valueAt(other.T_effective, temperature) * BOLTZMANN_EV
                     if other.T_effective is not None else temperature * BOLTZMANN_EV)
    lat = 1 if law.calculatedAtThermal else 0
    lasym = 0 if atom.selfScatteringKernel.symmetric in (None, True) else 1
    na = len(alpha)
    cliq = 0.0
    if lnS[0, 0] > lnS[0, 1]:
        cliq = (lnS[0, 0] - lnS[1, 0]) * alpha[0] / beta[1] ** 2
    return _Inputs(alpha=alpha, beta=beta, sab=sab, az=az, sb=sb, sb2=sb2, az2=az2, teff=teff,
                   teff2=teff2, lat=lat, lasym=lasym, cliq=cliq,
                   emax=float(atom.e_max.convertedTo("eV").value),
                   tev=float(temperature) * BOLTZMANN_EV)


def freeGasInputs(awr: float, temperature: float, emax: float) -> _Inputs:
    """THERMR's inputs for the free gas (``calcem`` label 200, ``thermr.f90:1858-1910``).

    ``awr`` is the nuclide's mass ratio as its tape states it (THERMR reads it
    from the PENDF); the bound cross section is ``((awr+1)/awr)**2``, a free
    one of 1 b; no alpha table, ``lat = lasym = 0``. Above 3000 K THERMR
    rescales the incident grid (``break``); that is not ported, so it is refused.
    """
    if not awr > 0:
        raise ValueError(f"the free gas needs a mass ratio > 0, got {awr}")
    if not 0 < temperature <= 3000.0:
        raise ValueError(f"free gas at {temperature} K: kelvin in (0, 3000]; above 3000 K THERMR "
                         f"rescales its incident grid, which kika does not port")
    if not emax > 0:
        raise ValueError(f"emax is an energy in eV > 0, got {emax}")
    az = float(awr)
    # calcem dimensions an alpha table it never reads for iinc=1; three nodes
    # satisfy the kernel's size check.
    alpha = np.array([1.0, 2.0, 3.0])
    beta = FREE_GAS_BETA.copy()
    sab = np.full(len(alpha) * len(beta), SABFLG)
    tev = float(temperature) * BOLTZMANN_EV
    return _Inputs(alpha=alpha, beta=beta, sab=sab, az=az, sb=((az + 1) / az) ** 2, sb2=0.0,
                   az2=0.0, teff=tev, teff2=0.0, lat=0, lasym=0, cliq=0.0, emax=float(emax),
                   tev=tev, iinc=1)


def _calcem(inputs: _Inputs, enow: float, options: ThermalInelasticOptions):
    nl = options.nbin + 1
    xs = np.zeros(options.capacity)
    ys = np.zeros(options.capacity * nl)
    n, xsi = _native().calcem(
        enow, inputs.tev, KT0, inputs.az, inputs.sb, inputs.sb2, inputs.az2, inputs.teff,
        inputs.teff2, inputs.cliq, inputs.lat, inputs.lasym, nl, options.tol,
        len(inputs.alpha), len(inputs.beta), options.capacity,
        inputs.alpha, inputs.beta, inputs.sab, xs, ys, inputs.iinc)
    return xs[:n].copy(), ys[:n * nl].reshape(n, nl).copy(), xsi


def _tpend(x, y, xsi):
    """What THERMR's ``tpend`` leaves on the tape ACER reads: ``(E', y)``.

    The density is σ(E→E')/σ(E), to 7 figures (6 below 1e-9) -- so ACER's 1e-6
    merge threshold is on a unit area, not in barns -- and every number then
    goes through NJOY's 11-column ENDF writer (``endf.f90`` ``a11``): 7
    significant figures in exponent form, 9 in the fixed form it uses for
    positive values in (0.1, 1e7). Two outgoing energies that differ beyond
    the 7th figure read back equal, and ACER skips the second.
    """
    native = _native()
    out = y.copy()
    rx = 1.0 / xsi
    out[:, 0] = [native.sigfig(v * rx, 7, 0) if v * rx >= 1e-9 else native.sigfig(v * rx, 6, 0)
                 for v in y[:, 0]]
    x = np.ascontiguousarray(x, dtype=float).copy()
    out = np.ascontiguousarray(out, dtype=float)
    native.endf11(x)
    native.endf11(out.reshape(-1))
    return x, out


def _acer(x, y, nang):
    """ACER's ``acesix`` with ``iwt = 2`` on one incident energy: ``(E', pdf, cdf, cosines)``.

    ``y`` is what THERMR's ``tpend`` leaves on the tape (:func:`_tpend`).
    """
    nep = len(x)
    cos = y[:, 1:1 + nang].copy()
    if nep >= 3:
        cos[0] = cos[1]
        cos[-1] = cos[-2]
    sig = y[:, 0]
    eps = 1e-5
    out_x, out_y, out_f, out_c = [], [], [], []
    xl, yl = x[0], sig[0]
    total = 0.0
    fract = 1.0
    i = 0
    while i < nep:
        xi, yi = x[i], sig[i]
        i += 1
        while True:
            add = (yi + yl) * (xi - xl) / 2
            if xi == xl:
                break
            if i == nep:
                xn = xi
                fract = total + add
            else:
                if total + add > eps / 10:
                    fract = total + add
                if not total + add >= fract - fract / 10000:
                    total += add
                    break
                if total + add < fract + fract / 10000:
                    xn = xi
                elif abs(yi - yl) > (yi + yl) / 100000:
                    f = (yi - yl) / (xi - xl)
                    disc = abs((yl / f) ** 2 + 2 * (fract - total) / f)
                    xn = xl - yl / f + np.sign(f) * np.sqrt(disc)
                else:
                    xn = min(xl + (fract - total) / yl, xi)
            yn = yl + (yi - yl) * (xn - xl) / (xi - xl)
            # The cosines at xn, found as ACER finds them: the first THERMR
            # point at or above xn (aceth.f90:432-443), then that point's own
            # cosines when xn is within eps of it (iwt=2: it always is).
            k = 1
            while k < nep - 1 and x[k] < xn:
                k += 1
            lo, hi = x[k - 1], x[k]
            if abs(xn - hi) < eps:
                c = cos[k]
            elif abs(xn - lo) < eps:
                c = cos[k - 1]
            else:
                c = cos[k - 1] + (cos[k] - cos[k - 1]) * (xn - lo) / (hi - lo)
            out_x.append(xn)
            out_y.append(yn)
            out_f.append(fract)
            out_c.append(np.clip(c, -1.0, 1.0))
            xl, yl = xn, yn
            fract = 1.0
            total = 0.0
            if xl < xi:
                continue
            break
        xl, yl = xi, yi
    area = np.cumsum(out_f)
    total = area[-1] if len(area) else 0.0
    return (np.asarray(out_x), np.asarray(out_y) / total, area / total, np.asarray(out_c))


def _linlin(table, x: float) -> float:
    E, sigma = table
    if not E[0] <= x <= E[-1]:
        raise ValueError(f"the elastic cross section does not reach {x} eV "
                         f"(it spans {E[0]}-{E[-1]} eV)")
    i = int(np.searchsorted(E, x, side="right")) - 1
    i = min(max(i, 0), len(E) - 2)
    if E[i + 1] == E[i]:
        return float(sigma[i + 1])
    return float(sigma[i] + (sigma[i + 1] - sigma[i]) * (x - E[i]) / (E[i + 1] - E[i]))


def _discrete(values, outer):
    from kika.nuclear_data import model as m

    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    return m.Xs_pdf_cdf1d(xs=values, pdf=np.full(n, 1.0 / n), cdf=np.arange(1, n + 1) / n,
                          interpolation=m.Interpolation.flat, numberOfDiscrete=n,
                          outerDomainValue=float(outer))


def inelasticTables(law, temperature: float, *, nmix: int = 1, awr: Optional[float] = None,
                    options: ThermalInelasticOptions = ThermalInelasticOptions(), report=None):
    """``(cross section, EnergyAngularMC, energy_max)`` at ``temperature`` (K).

    The cross section is per molecule as OpenMC stores it (THERMR's per
    principal atom, divided by ``nmix`` as ACER divides it).
    """
    if options.iinc != 2:
        raise ValueError(f"inelasticTables reads an S(alpha, beta) law (iinc=2); iinc="
                         f"{options.iinc} is freeGasTables")
    if options.emax is not None:
        raise ValueError("emax overrides only the free gas; S(alpha, beta) takes the "
                         "evaluation's B(4)")
    inputs = thermalInputs(law, temperature, awr)
    return _tables(inputs, temperature, nmix, options, report, None)


def freeGasTables(awr: float, temperature: float, elastic, *, emax: float = 10.0, nmix: int = 1,
                  options: ThermalInelasticOptions = ThermalInelasticOptions(iinc=1), report=None):
    """Free-gas law with singularity-free angular integration: ``(cross section, EnergyAngularMC, energy_max)`` at ``temperature``.

    ``awr`` is the nuclide's mass ratio; ``elastic`` its elastic cross section
    at ``temperature`` as ``(E, sigma)`` arrays in eV and b (BROADR's MT2,
    which THERMR writes as the inelastic cross section); ``emax`` is THERMR's
    card-4 ``emax`` in eV. The distributions do not depend on ``elastic``.
    """
    if options.iinc != 1:
        raise ValueError(f"freeGasTables is THERMR's iinc=1; the options say iinc={options.iinc}")
    E, sigma = (np.asarray(v, dtype=float) for v in elastic)
    if E.ndim != 1 or E.shape != sigma.shape or len(E) < 2 or np.any(np.diff(E) < 0):
        raise ValueError("elastic is (E, sigma): two 1-d arrays of equal length, E ascending")
    inputs = freeGasInputs(awr, temperature, emax)
    return _tables(inputs, temperature, nmix, options, report, (E, sigma))


def _tables(inputs: _Inputs, temperature, nmix, options, report, elastic):
    from kika.nuclear_data import model as m

    nang = options.nbin
    energies = []
    for e in EGRID:
        enow = _native().sigfig(float(e), 8, 0)
        if enow > inputs.emax or abs(enow - inputs.emax) < 1e-5:
            break
        energies.append(enow)
    # ACE stores the incident energies to 12 significant figures (1PE20.11):
    # sigfig's bias (1 + 1e-13) does not survive it, and 10 eV reads back 10.
    # ACE stores them in MeV to 12 significant figures (1PE20.11) and OpenMC
    # multiplies by 1e6 on read: 4.9 eV reads back 4.8999999999999995.
    stored = [float(f"{e * 1e-6:.11e}") * 1e6 for e in energies]
    workers = options.workers if options.workers is not None else min(8, os.cpu_count() or 1)
    if workers > 1 and len(energies) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            kernels = list(pool.map(lambda e: _calcem(inputs, e, options), energies))
    else:
        kernels = [_calcem(inputs, e, options) for e in energies]
    xs, spectra, angular = [], [], []
    for (x, y, xsi), eout, enow in zip(kernels, stored, energies):
        if inputs.iinc==1:
            # ACER skips small CDF increments and joins the last nonzero PDF to
            # a distant zero. That tail no longer integrates to its stored CDF
            # and can corrupt the second moment despite a small KS difference.
            # The corrected free-gas grid already controls density interpolation.
            ex=np.asarray(x,dtype=float)
            density=np.asarray(y[:,0],dtype=float)
            areas=np.diff(ex)*(density[:-1]+density[1:])/2
            cumulative=np.r_[0.,np.cumsum(areas)]
            norm=float(cumulative[-1])
            if not norm>0 or np.any(density<0):raise ValueError("invalid free-gas density")
            pdf=density/norm;cdf=cumulative/norm
            cos=np.clip(y[:,1:1+nang],-1.,1.)
        else:
            ex, pdf, cdf, cos = _acer(*_tpend(x, y, xsi), nang)
        tables = [_discrete(c, e) for c, e in zip(cos, ex)]
        if len(cdf) and cdf[0] > 0.0:
            ex, pdf, cdf = np.insert(ex, 0, 0.0), np.insert(pdf, 0, 0.0), np.insert(cdf, 0, 0.0)
            dmu = 2.0 / nang
            tables.insert(0, _discrete(np.linspace(-1 + dmu / 2, 1 - dmu / 2, nang), 0.0))
        spectra.append(m.Xs_pdf_cdf1d(xs=ex, pdf=pdf, cdf=cdf, interpolation=m.Interpolation.linlin,
                                      outerDomainValue=eout))
        angular.append(m.XYs2d(function1ds=tables, outerDomainValue=eout))
        if elastic is None:
            xs.append(xsi / nmix)
        else:
            # ACER's gety1 on THERMR's MT221, the PENDF elastic: lin-lin at the
            # incident energy as THERMR left it (enow, before the ACE rounding).
            xs.append(_linlin(elastic, enow) / nmix)
    form = m.EnergyAngularMC(energy=m.XYs2d(function1ds=spectra),
                             energyAngular=m.XYs3d(function2ds=angular))
    if report is not None:
        kind = "free gas" if inputs.iinc == 1 else "incoherent inelastic"
        method="singularity-free angular integration and consistent PDF/CDF" if inputs.iinc==1 else "THERMR calcem and ACER iwt=2"
        report.warn(f"{kind} at {temperature:g} K by kika ({method}, {len(energies)} incident energies)")
    return m.XYs1d(xs=np.asarray(stored), ys=np.asarray(xs)), form, float(stored[-1])
