"""An elastic suite read from G4NDL, as the plain arrays a viewer draws.

The kika-app explores a G4NDL library the way it explores an ENDF tape, and
its angular views already read MF4 in one shape: the ``/api/endf/bulk-legendre``
response, coefficients per order on the Legendre energies and each table on
its own cosines above them. A G4NDL ``Elastic/FS`` is that same structure --
``repFlag`` 0/1/2/3 is MF4's LTT, with the same records -- so
:func:`angularBulk` hands it over in that shape and nothing downstream needs a
second reader. :func:`isotopeSummary` is the header a viewer shows above it.

Everything is read off the model (:func:`kika.g4ndl.decode.decodeElastic`), and
the two conventions this module has to choose are stated where they are made:

* **Tables are sent lin-lin in mu.** A segment on another law (C-12 has LOGLIN
  ones) is subdivided and evaluated on its own law, so a client interpolating
  linearly between the cosines it is given stays within
  :data:`LINEARISATION_POINTS` subdivisions of the file rather than reading a
  different function.
* **Coefficients at a table energy are projected**, ``a_l = int p P_l dmu``
  on each segment's own law (:func:`kika.g4ndl.physics._tableMoments`), so an
  a_l(E) curve spans the whole range. The table itself, not the projection, is
  what the angular views read there.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from kika.g4ndl.physics import (
    _blocks, _distribution, _segmentValues, _tableArrays, _tableMoments,
)

__all__ = ["angularBulk", "isotopeSummary", "LINEARISATION_POINTS",
           "PROJECTION_ORDER", "REPRESENTATIONS"]

#: Points each non-lin-lin segment of a table is cut into before it is sent.
LINEARISATION_POINTS = 16

#: Highest order a table is projected to when the isotope has no Legendre
#: block to take the order from (``repFlag=2``).
PROJECTION_ORDER = 20

#: ``repFlag`` → what it means; the numbers are MF4's LTT.
REPRESENTATIONS = {0: "isotropic", 1: "Legendre", 2: "tabulated",
                   3: "Legendre, then tabulated"}


def _frame(distribution) -> Optional[str]:
    frame = getattr(distribution, "productFrame", None)
    if frame is None:
        return None
    return "LAB" if "lab" in str(getattr(frame, "value", frame)).lower() else "CM"


def _crossSection(suite):
    form = suite.reactions[2].crossSection["recon"]
    return np.asarray(form.xs, dtype=float), np.asarray(form.ys, dtype=float)


def _linearTable(function):
    """``(mu, p)`` of one table, lin-lin in mu (see the module docstring)."""
    mu, p, nbt, codes = _tableArrays(function)
    if all(c == 2 for c in codes):
        return mu, p
    from kika.g4ndl.physics import _intervalCodes

    laws = _intervalCodes(nbt, codes, mu.size)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for j, law in enumerate(laws):
        x1, x2 = mu[j], mu[j + 1]
        if law == 2 or x2 == x1:
            x = np.array([x1])
            y = np.array([p[j]])
        else:
            x = np.linspace(x1, x2, LINEARISATION_POINTS, endpoint=False)
            y = _segmentValues(int(law), x1, x2, p[j], p[j + 1], x)
        xs.append(x)
        ys.append(y)
    xs.append(mu[-1:])
    ys.append(p[-1:])
    return np.concatenate(xs), np.concatenate(ys)


def _regionPairs(codes: np.ndarray) -> List[List[int]]:
    """Per-interval law codes → ENDF ``[NBT, INT]`` pairs (NBT 1-based)."""
    pairs: List[List[int]] = []
    for j, code in enumerate(codes):
        if pairs and pairs[-1][1] == int(code):
            pairs[-1][0] = j + 2
        else:
            pairs.append([j + 2, int(code)])
    return pairs


def angularBulk(suite, maxOrder: Optional[int] = None) -> Dict[str, Any]:
    """The elastic angular distribution in the shape of an MF4 bulk response.

    Returns a dict with ``energies`` (eV, the Legendre block then the tables
    above it), ``max_order``, ``coefficients_by_order`` (``str(l)`` → one value
    per energy, ``a_0 = 1`` included), ``energy_interpolation`` (the Legendre
    block's ``[NBT, INT]``), ``representation`` (``repFlag``), ``frame``
    (``"CM"``/``"LAB"``), ``pdf_mu_by_energy``/``pdf_by_energy`` (``None`` on a
    Legendre row), ``pdf_angular_linlin`` and the ``pdf_boundary*`` table a
    ``repFlag=3`` isotope stores at its transition energy.
    """
    distribution = _distribution(suite)
    repFlag = getattr(getattr(suite, "provenance", None), "repFlag", None)
    if not hasattr(distribution, "angular"):                  # isotropic
        energies, _ = _crossSection(suite)
        span = [float(energies[0]), float(energies[-1])]
        return dict(energies=span, max_order=0, coefficients_by_order={"0": [1.0, 1.0]},
                    energy_interpolation=None, representation=0 if repFlag is None else repFlag,
                    frame=_frame(distribution), pdf_mu_by_energy=None, pdf_by_energy=None,
                    pdf_angular_linlin=True, pdf_boundary_energy=None,
                    pdf_boundary_mu=None, pdf_boundary=None)

    blocks = _blocks(distribution.angular)
    legendre = next((b for b in blocks if b.kind == "Legendre"), None)
    table = next((b for b in blocks if b.kind == "table"), None)

    nativeOrder = 0
    if legendre is not None:
        nativeOrder = max(len(f.coefficients) - 1 for f in legendre.functions)
    order = maxOrder if maxOrder is not None else (nativeOrder or PROJECTION_ORDER)

    energies: List[float] = []
    rows: List[np.ndarray] = []
    pdfMu: List[Optional[List[float]]] = []
    pdf: List[Optional[List[float]]] = []
    if legendre is not None:
        for f in legendre.functions:
            a = np.zeros(order + 1)
            c = np.asarray(f.coefficients, dtype=float)[: order + 1]
            a[: c.size] = c
            energies.append(float(f.outerDomainValue))
            rows.append(a)
            pdfMu.append(None)
            pdf.append(None)

    boundary = None
    if table is not None:
        functions = list(table.functions)
        # The table at the transition energy is the left end of the first
        # tabulated interval; its row belongs to the Legendre side, where
        # Geant4 takes it too (physics._pickBlock).
        if energies and float(functions[0].outerDomainValue) == energies[-1]:
            mu, p = _linearTable(functions[0])
            boundary = (energies[-1], mu.tolist(), p.tolist())
            functions = functions[1:]
        for f in functions:
            mu, p, nbt, codes = _tableArrays(f)
            energies.append(float(f.outerDomainValue))
            rows.append(_tableMoments(mu, p, nbt, codes, order))
            lmu, lp = _linearTable(f)
            pdfMu.append(lmu.tolist())
            pdf.append(lp.tolist())

    matrix = np.vstack(rows) if rows else np.zeros((0, order + 1))
    return dict(
        energies=energies,
        max_order=order,
        coefficients_by_order={str(l): matrix[:, l].tolist() for l in range(order + 1)},
        energy_interpolation=_regionPairs(legendre.intervalCodes) if legendre is not None else None,
        representation=repFlag,
        frame=_frame(distribution),
        pdf_mu_by_energy=pdfMu if table is not None else None,
        pdf_by_energy=pdf if table is not None else None,
        pdf_angular_linlin=True,
        pdf_boundary_energy=boundary[0] if boundary else None,
        pdf_boundary_mu=boundary[1] if boundary else None,
        pdf_boundary=boundary[2] if boundary else None,
    )


def isotopeSummary(suite) -> Dict[str, Any]:
    """What a viewer shows above an isotope's plots: the file's header, the
    ranges of its two halves, and what the read lost or warned about."""
    prov = getattr(suite, "provenance", None)
    energies, sigma = _crossSection(suite)
    distribution = _distribution(suite)
    blocks = _blocks(distribution.angular) if hasattr(distribution, "angular") else []

    def block(kind):
        b = next((b for b in blocks if b.kind == kind), None)
        if b is None:
            return None
        e = b.energies
        out = dict(count=int(e.size), energy_min=float(e[0]), energy_max=float(e[-1]),
                   repeated_energies=int(np.sum(np.diff(e) == 0)),
                   energy_laws=sorted({int(c) for c in b.intervalCodes}))
        if kind == "Legendre":
            out["max_order"] = int(max(len(f.coefficients) - 1 for f in b.functions))
        else:
            out["max_points"] = int(max(len(_tableArrays(f)[0]) for f in b.functions))
        return out

    legendre, table = block("Legendre"), block("table")
    repFlag = getattr(prov, "repFlag", None)
    temperatures = list(getattr(prov, "legendreTemperatures", []) or []) + \
        list(getattr(prov, "tabulatedTemperatures", []) or [])
    tempdeps = list(getattr(prov, "legendreTempdeps", []) or []) + \
        list(getattr(prov, "tabulatedTempdeps", []) or [])
    report = getattr(suite, "report", None)
    step = np.diff(energies)
    return dict(
        target=suite.target,
        library=getattr(prov, "library", None),
        library_name=getattr(prov, "libraryName", None),
        cross_section_path=getattr(prov, "crossSectionPath", None),
        final_state_path=getattr(prov, "finalStatePath", None),
        compressed=bool(str(getattr(prov, "finalStatePath", "") or "").endswith(".z")),
        target_mass=getattr(prov, "targetMass", None),
        frame=_frame(distribution),
        rep_flag=repFlag,
        representation=REPRESENTATIONS.get(repFlag, "unknown"),
        transition_energy=legendre["energy_max"] if legendre and table else None,
        cross_section=dict(count=int(energies.size), energy_min=float(energies[0]),
                           energy_max=float(energies[-1]), sigma_min=float(sigma.min()),
                           sigma_max=float(sigma.max()),
                           repeated_energies=int(np.sum(step == 0))),
        legendre=legendre,
        tabulated=table,
        nonzero_temperatures=int(np.count_nonzero(temperatures)),
        nonzero_tempdeps=int(np.count_nonzero(tempdeps)),
        report=dict(warnings=list(report.warnings), losses=list(report.losses),
                    approximations=list(report.approximations),
                    unsupported=list(report.unsupported)) if report is not None else None,
    )
