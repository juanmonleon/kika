"""``fission_energy_release``: the nine functions OpenMC stores, as OpenMC builds them.

The model holds MF1/458 as the evaluation states it
(:class:`~kika.nuclear_data.model.FissionEnergyRelease`). OpenMC adds two
processing steps (``fission_energy.py:194-330``), reproduced here:

1. **Sher-Beck.** A component given as one constant (``NPLY = 0``) gets the
   energy dependence ENDF-102 §1.5.2.1 recommends: ``-0.075 E`` for the delayed
   photons and betas, ``-0.105 E`` for the neutrinos, and for the prompt
   neutrons ``EFR0 + 1.307 E - 8.07e6 (ν(E) - ν(0))``. The ENDF/B-VII.1 MeV/eV
   slip in second-order coefficients is corrected as OpenMC corrects it.
2. **The derived pair.** ``q_prompt = fragments + prompt_neutrons +
   prompt_photons - E`` and ``q_recoverable = fragments + prompt_neutrons +
   delayed_neutrons + prompt_photons + delayed_photons + betas - E``, summed on
   the union of the tabulated components' grids (``function.py:19-56``).

The C++ reads ``q_prompt``, ``q_recoverable``, ``fragments``, ``betas``,
``prompt_photons`` and ``delayed_photons`` (``nuclide.cpp:333-348``).
"""
from __future__ import annotations

from functools import reduce
from itertools import zip_longest
from typing import List, Tuple

import numpy as np

__all__ = ["derivedQ", "openmcTerms", "sumFunctions", "evaluate"]

#: MF1/458's order (``fission_energy.py:_NAMES``) and the model's terms.
_TERMS = ("promptProductKE", "promptNeutronKE", "delayedNeutronKE", "promptGammaEnergy",
          "delayedGammaEnergy", "delayedBetaEnergy", "neutrinoEnergy")


def _m():
    from kika.nuclear_data import model

    return model


def evaluate(function, x):
    """OpenMC's evaluation: a table lin-lin and held at its end values outside."""
    m = _m()
    x = np.asarray(x, dtype=float)
    if isinstance(function, m.Polynomial1d):
        return np.polynomial.polynomial.polyval(x, function.coefficients)
    if isinstance(function, m.XYs1d):
        return np.interp(x, function.xs, function.ys)
    if isinstance(function, m.Regions1d):
        xs, ys, _ = function.toEndfRegions()
        return np.interp(x, xs, ys)
    raise TypeError(f"cannot evaluate a {type(function).__name__}")


def sumFunctions(functions):
    """``function.py:19-56``: tables summed on the union grid, else coefficients added."""
    m = _m()
    functions = list(functions)
    grids = [f.xs for f in functions if isinstance(f, m.XYs1d)]
    grids += [f.toEndfRegions()[0] for f in functions if isinstance(f, m.Regions1d)]
    if grids:
        x = reduce(np.union1d, grids)
        y = sum(evaluate(f, x) for f in functions)
        return m.XYs1d(xs=x, ys=y)
    coefficients = [sum(c) for c in zip_longest(*(f.coefficients for f in functions), fillvalue=0.0)]
    return m.Polynomial1d(coefficients=np.asarray(coefficients), domainMin_=0.0,
                          domainMax_=float("inf"))


def _minusE():
    return _m().Polynomial1d(coefficients=np.array([0.0, -1.0]), domainMin_=0.0,
                             domainMax_=float("inf"))


def derivedQ(release) -> Tuple[object, object]:
    """``(q_prompt, q_recoverable)`` of a release whose seven components are set."""
    t = {name: getattr(release, name) for name in _TERMS}
    missing = [name for name in _TERMS[:6] if t[name] is None]
    if missing:
        raise ValueError(f"fission energy release lacks {missing}")
    prompt = sumFunctions([t["promptProductKE"], t["promptNeutronKE"], t["promptGammaEnergy"],
                           _minusE()])
    recoverable = sumFunctions([t["promptProductKE"], t["promptNeutronKE"], t["delayedNeutronKE"],
                                t["promptGammaEnergy"], t["delayedGammaEnergy"],
                                t["delayedBetaEnergy"], _minusE()])
    return prompt, recoverable


def openmcTerms(release, promptNu, report=None):
    """The seven components after OpenMC's Sher-Beck step, as a new release.

    ``promptNu`` is the fission neutron multiplicity (prompt, or total when
    only the total is given -- OpenMC takes whichever ``product_0`` holds).
    """
    m = _m()
    out = m.FissionEnergyRelease(label="MonteCarlo")
    provenance = getattr(release, "provenance", None)
    lfc = None
    stated = {}
    if provenance is not None:
        lfc = getattr(provenance, "lfc", None) or getattr(provenance, "attributes", {}).get("LFC")
    for name in _TERMS:
        function = getattr(release, name)
        if function is None:
            continue
        if isinstance(function, m.Polynomial1d):
            coefficients = np.array(function.coefficients, dtype=float)
            if len(coefficients) == 3 and abs(coefficients[2]) * (5e6) ** 2 > 1e8:
                coefficients[2] /= 1.0e6
                if report is not None:
                    report.approximated(f"MF1/458 {name}: second-order coefficient divided by "
                                        f"1e6 (the ENDF/B-VII.1 MeV/eV slip OpenMC corrects)")
            if len(coefficients) == 1:
                function = _sherBeck(name, float(coefficients[0]), promptNu)
            else:
                function = m.Polynomial1d(coefficients=coefficients, domainMin_=0.0,
                                          domainMax_=float("inf"))
        stated[name] = function
        setattr(out, name, function)
    return out


def _sherBeck(name, constant, nu):
    m = _m()
    poly = lambda c: m.Polynomial1d(coefficients=np.asarray(c, float), domainMin_=0.0,  # noqa: E731
                                    domainMax_=float("inf"))
    if name in ("delayedGammaEnergy", "delayedBetaEnergy"):
        return poly([constant, -0.075])
    if name == "neutrinoEnergy":
        return poly([constant, -0.105])
    if name == "promptNeutronKE":
        if nu is None:
            raise ValueError("Sher-Beck for the prompt neutrons needs the fission nu-bar")
        if isinstance(nu, m.XYs1d):
            return m.XYs1d(xs=nu.xs, ys=constant + 1.307 * nu.xs - 8.07e6 * (nu.ys - nu.ys[0]))
        if isinstance(nu, m.Polynomial1d):
            c = list(nu.coefficients)
            if len(c) == 1:
                return poly([constant, 1.307])
            return poly([constant, 1.307 - 8.07e6 * c[1]] + [-8.07e6 * x for x in c[2:]])
        if isinstance(nu, m.Regions1d):
            xs, ys, _ = nu.toEndfRegions()
            return m.XYs1d(xs=xs, ys=constant + 1.307 * xs - 8.07e6 * (ys - ys[0]))
        raise TypeError(f"nu-bar as a {type(nu).__name__}")
    return poly([constant])
