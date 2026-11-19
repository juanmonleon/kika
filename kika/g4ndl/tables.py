"""A suite read from G4NDL, as the plain arrays a viewer draws.

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

**The inelastic.** A reaction whose outgoing neutron carries an angular
distribution of its own is an MF4 in the same sense: a two-body emission
(``angularTwoBody``, the discrete levels MT51-90 and MT2) or the angular half
of an ``uncorrelated`` pair (MF4 with MF5, which the continuum MT91, MT16, ...
often is). :func:`angularBulk` takes an ``mt`` and answers for any of them
(:func:`angularMTs` lists them), as the ENDF MF4 view lists every MT with an
MF4. A correlated energy-angle law (``energyAngular``, ``KalbachMann``: MF6)
is not served here. :func:`crossSections` gives every cross section the suite
holds, the sums MT4 and MT103-107
included, so a viewer lists them as it lists an ENDF tape's MF3.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from kika.g4ndl.physics import (
    _blocks, _distribution, _segmentValues, _tableArrays, _tableMoments,
)

__all__ = ["angularBulk", "angularMTs", "crossSections", "pointwiseCrossSection", "isotopeSummary",
           "LINEARISATION_POINTS", "PROJECTION_ORDER", "REPRESENTATIONS",
           "INELASTIC_TOTAL", "inelasticTotal", "captureSummary", "fissionSummary"]

#: Points each non-lin-lin segment of a table is cut into before it is sent.
LINEARISATION_POINTS = 16

#: Highest order a table is projected to when the isotope has no Legendre
#: block to take the order from (``repFlag=2``).
PROJECTION_ORDER = 20

#: ``repFlag`` → what it means; the numbers are MF4's LTT.
REPRESENTATIONS = {0: "isotropic", 1: "Legendre", 2: "tabulated",
                   3: "Legendre, then tabulated"}


#: Label of the G4NDL ``Inelastic/CrossSection`` sum in ``suite.sums``: the
#: process cross section, which has no ENDF MT (it is not MT3).
INELASTIC_TOTAL = "inelastic"


def _neutronDistribution(suite, mt: int):
    """The ``eval`` distribution of the outgoing neutron of reaction ``mt``."""
    reaction = suite.reactions.byENDF_MT(mt)
    products = reaction.outputChannel.products.byPid("n")
    if not products:
        raise ValueError(f"MT{mt} of {suite.target} emits no neutron")
    return products[0].distribution["eval"]


def _angular(distribution):
    """The angular part of a two-body or uncorrelated form, or ``None`` when
    it has none (a recoil, or a correlated energy-angle law)."""
    if type(distribution).__name__ in ("AngularTwoBody", "Uncorrelated"):
        return distribution.angular
    if type(distribution).__name__ == "Isotropic2d":
        return distribution
    return None


def _isIsotropic(angular) -> bool:
    return not (hasattr(angular, "function2ds") or hasattr(angular, "function1ds"))


def _repFlag(blocks) -> int:
    kinds = [b.kind for b in blocks]
    if not kinds:
        return 0
    if kinds == ["Legendre"]:
        return 1
    if kinds == ["table"]:
        return 2
    return 3


def angularMTs(suite) -> List[int]:
    """MTs whose outgoing neutron has an angular distribution of its own (see
    the module docstring): what :func:`angularBulk` serves.

    A reaction qualifies when its neutron product is a two-body emission, an
    ``uncorrelated`` pair or an isotropic distribution. Reactions without an
    ENDF MT, without a neutron product, or with a correlated energy-angle law
    (MF6) are left out.

    Parameters
    ----------
    suite
        A reaction suite, normally one read from G4NDL.

    Returns
    -------
    list of int
        The MTs in increasing order.
    """
    out = []
    for reaction in suite.reactions:
        mt = reaction.ENDF_MT
        if mt is None:
            continue
        try:
            angular = _angular(_neutronDistribution(suite, mt))
        except (ValueError, KeyError, TypeError):
            continue
        if angular is not None:
            out.append(int(mt))
    return sorted(out)


def pointwiseCrossSection(suite, mt: int):
    """``(E, sigma)`` of ``mt``: the ``recon`` form when the suite has one,
    else the evaluated form when it is already pointwise.

    A G4NDL suite labels every cross section ``recon``. A GNDS file translated
    by FUDGE carries ``recon`` only where the evaluation holds resonance
    parameters (``resonancesWithBackground``); every other reaction is an
    ``eval`` ``XYs1d``/``regions1d``, which is pointwise as it stands. Raises
    ``KeyError`` when neither exists, ``TypeError`` when the only form is not
    tabulated (resonances with no reconstruction beside them).
    """
    try:
        return suite.cross_section(int(mt), form="recon")
    except KeyError:
        return suite.cross_section(int(mt), form="eval")


def crossSections(suite) -> Dict[int, Any]:
    """MT → ``(E, sigma)`` (eV, b, 0 K, ``pointwiseCrossSection``) for every
    reaction and every sum with an ENDF MT; the process total is
    :func:`inelasticTotal`.

    The pair of an MT is its ``recon`` form when the suite has one, else its
    ``eval`` form when that is pointwise. An MT with neither is left out
    without a message.

    Parameters
    ----------
    suite
        A reaction suite, normally one read from G4NDL.

    Returns
    -------
    dict of int to tuple of numpy.ndarray
        Keys in increasing MT order. Each value is ``(energy, sigma)``, two
        arrays of equal length: incident energies in eV and cross-sections in
        barn, as the model holds them and in grid order.
    """
    out: Dict[int, Any] = {}
    for node in list(suite.reactions) + list(suite.sums):
        mt = node.ENDF_MT
        if mt is None:
            continue
        try:
            out[int(mt)] = pointwiseCrossSection(suite, int(mt))
        except (KeyError, TypeError):
            continue
    return dict(sorted(out.items()))


def inelasticTotal(suite):
    """``(E, sigma)`` of ``Inelastic/CrossSection``, or ``None``.

    Parameters
    ----------
    suite
        A reaction suite, normally one read from G4NDL.

    Returns
    -------
    tuple of numpy.ndarray or None
        ``(energy, sigma)`` of the ``recon`` form of the sum labelled
        ``"inelastic"``, as float arrays of equal length: energies in eV and
        cross-sections in barn, as G4NDL stores them (kika does not convert).
        ``None`` when the suite has no such sum.
    """
    for node in suite.sums:
        if node.label == INELASTIC_TOTAL:
            form = node.crossSection["recon"]
            return np.asarray(form.xs, dtype=float), np.asarray(form.ys, dtype=float)
    return None


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


def angularBulk(suite, maxOrder: Optional[int] = None, mt: int = 2) -> Dict[str, Any]:
    """Reaction ``mt``'s angular distribution (default the elastic) in the
    shape of an MF4 bulk response. ``mt`` is one of :func:`angularMTs`.

    Parameters
    ----------
    suite
        A reaction suite, normally one read from G4NDL.
    maxOrder
        Highest Legendre order of ``coefficients_by_order``. ``None`` takes the
        highest order of the Legendre block, or ``PROJECTION_ORDER`` (20) when
        there is none or its highest order is 0. Legendre rows longer than this
        are cut and shorter ones padded with zeros. It does nothing for an
        isotropic distribution.
    mt
        The reaction. For anything but 2 it must be one of :func:`angularMTs`.

    Returns
    -------
    dict
        The keys, all always present:

        * ``energies``: list of incident energies in eV: the Legendre
          energies in file order, then the energies of the tables above them
          (the table at the transition energy is not repeated here). For an
          isotropic distribution, two values: the first and last energy of the
          cross-section grid.
        * ``max_order``: the highest order in ``coefficients_by_order`` (0 for
          an isotropic distribution).
        * ``coefficients_by_order``: ``str(l)`` for ``l = 0 ... max_order``,
          each a list with one coefficient ``a_l`` per energy. A Legendre row
          holds the model's coefficients with ``a_0 = 1`` in front. A table row
          holds the projection ``a_l = int p(mu) P_l(mu) dmu`` of the table,
          from a 16-point Gauss rule on each segment under its own
          interpolation law (not exact for the logarithmic laws). An isotropic
          distribution gives ``{"0": [1.0, 1.0]}``.
        * ``energy_interpolation``: the Legendre block's interpolation in
          incident energy as ENDF ``[NBT, INT]`` pairs (``NBT`` 1-based);
          ``None`` without a Legendre block.
        * ``representation``: ``repFlag`` (0 isotropic, 1 Legendre, 2
          tabulated, 3 Legendre then tabulated). For ``mt=2`` of a suite read
          from G4NDL it is the file's own flag; otherwise it is read off the
          blocks.
        * ``frame``: ``"CM"`` or ``"LAB"``, or ``None`` when the distribution
          states no frame.
        * ``pdf_mu_by_energy``, ``pdf_by_energy``: one list per entry of
          ``energies``, holding ``mu`` and ``p(mu)`` of the table, lin-lin in
          ``mu`` (a segment on another law is cut into 16 points and evaluated
          on its own law); the entry of a Legendre energy is ``None``. Both are
          ``None`` when there is no tabulated block.
        * ``pdf_angular_linlin``: always ``True``.
        * ``pdf_boundary_energy``, ``pdf_boundary_mu``, ``pdf_boundary``: the
          table stored at the transition energy of a Legendre-then-tabulated
          distribution (energy in eV, then ``mu`` and ``p(mu)`` lin-lin); all
          ``None`` otherwise.

    Raises
    ------
    ValueError
        If ``mt`` is not 2 and its outgoing neutron has no angular distribution
        of its own (or the reaction emits no neutron).
    """
    if mt == 2:
        distribution = _distribution(suite)
        repFlag = getattr(getattr(suite, "provenance", None), "repFlag", None)
        angular = distribution.angular if hasattr(distribution, "angular") else None
        if angular is not None and _isIsotropic(angular):
            angular = None
        if repFlag is None:                     # not G4NDL: read it off the blocks
            repFlag = 0 if angular is None else _repFlag(_blocks(angular))
    else:
        distribution = _neutronDistribution(suite, mt)
        angular = _angular(distribution)
        if angular is None:
            raise ValueError(f"MT{mt} of {suite.target} has no angular distribution of its own; "
                             f"angular MTs: {angularMTs(suite)}")
        if _isIsotropic(angular):
            angular = None
        repFlag = 0 if angular is None else _repFlag(_blocks(angular))
    if angular is None:                                       # isotropic
        energies, _ = pointwiseCrossSection(suite, mt)
        span = [float(energies[0]), float(energies[-1])]
        return dict(energies=span, max_order=0, coefficients_by_order={"0": [1.0, 1.0]},
                    energy_interpolation=None, representation=0 if repFlag is None else repFlag,
                    frame=_frame(distribution), pdf_mu_by_energy=None, pdf_by_energy=None,
                    pdf_angular_linlin=True, pdf_boundary_energy=None,
                    pdf_boundary_mu=None, pdf_boundary=None)

    blocks = _blocks(angular)
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
    ranges of its two halves, and what the read lost or warned about.

    The elastic quantities (``cross_section``, ``legendre``, ``tabulated``,
    ``frame``, ``transition_energy``) are those of MT2.

    Parameters
    ----------
    suite
        A suite read from G4NDL. The file-level entries come from its
        provenance and are ``None`` when the suite has none.

    Returns
    -------
    dict
        Not a table: one dict, with these keys, all always present.

        * ``target``, ``library``, ``library_name``, ``target_mass`` (the mass
          ratio AWR), ``cross_section_path``, ``final_state_path``: from the
          suite and its provenance.
        * ``compressed``: whether the final-state path ends in ``.z``.
        * ``frame``: ``"CM"``, ``"LAB"`` or ``None``.
        * ``rep_flag`` (the file's ``repFlag``, or ``None``) and
          ``representation`` (its name, ``"unknown"`` for no flag).
        * ``transition_energy``: the last Legendre energy in eV when there is
          both a Legendre and a tabulated block, else ``None``.
        * ``cross_section``: dict with ``count`` (points), ``energy_min`` and
          ``energy_max`` (eV), ``sigma_min`` and ``sigma_max`` (barn) and
          ``repeated_energies`` (number of consecutive equal energies).
        * ``legendre``, ``tabulated``: ``None`` without that block, else a dict
          with ``count`` (incident energies), ``energy_min`` and ``energy_max``
          (eV), ``repeated_energies``, ``energy_laws`` (the sorted
          interpolation codes between the energies) and ``max_order`` (the
          highest Legendre order) or ``max_points`` (the most mu points of a
          table).
        * ``nonzero_temperatures``, ``nonzero_tempdeps``: how many of the
          file's temperatures and ``tempdep`` flags are not zero.
        * ``report``: dict of the lists ``warnings``, ``losses``,
          ``approximations`` and ``unsupported`` of the suite's conversion
          report, or ``None`` without one.
        * ``reactions``: a list with one row per MT that has a pointwise
          cross-section (the MTs of :func:`crossSections`), in MT order. A row
          is a dict: ``mt``; ``q_value`` (the reaction's QI as the model holds
          it, eV for a G4NDL suite; ``None`` for a sum or when no Q is
          stated); ``threshold`` (the first energy in eV where the
          cross-section is positive, ``None`` if it never is); ``count``,
          ``energy_min``, ``energy_max`` (eV) and ``sigma_max`` (barn) of the
          grid; ``sum`` (whether the MT is a sum such as MT4 or MT103-107);
          ``angular`` (whether :func:`angularBulk` serves it); and, only for an
          angular MT, ``rep_flag``, ``max_order`` (highest native Legendre
          order, 0 when isotropic or tabulated only) and
          ``transition_energy`` (eV or ``None``).
        * ``angular_mts``: the list :func:`angularMTs` returns.
        * ``inelastic_total``: ``None`` or a dict with ``count``,
          ``energy_min``, ``energy_max`` (eV) and ``sigma_max`` (barn) of
          ``Inelastic/CrossSection``.
        * ``capture``: the summary of MT102, a dict, or ``None`` without
          capture.
        * ``fission``: the result of :func:`fissionSummary`.
    """
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
        reactions=_reactionRows(suite),
        angular_mts=angularMTs(suite),
        inelastic_total=_range(inelasticTotal(suite)),
        capture=captureSummary(suite),
        fission=fissionSummary(suite),
    )


def captureSummary(suite) -> Optional[Dict[str, Any]]:
    """MT102 as a viewer shows it, or ``None`` without capture: its σ range,
    Q, which final-state file it was read from (``FSMF6``, ``FS``, ``None``),
    where its photons live (``"model"`` when the products reached the model,
    ``"verbatim"`` when they travel as G4NDL text, ``None`` without a final
    state, when Geant4 samples them from its photon evaporation) and the
    products' particles."""
    reaction = suite.findReactionByENDF_MT(102)
    if reaction is None:
        return None
    prov = getattr(reaction, "provenance", None)
    entry = getattr(prov, "finalStateEntry", None) or {}
    products = [p.label or p.pid for p in reaction.outputChannel.products]         if reaction.outputChannel is not None else []
    if products:
        photons = "model"
    elif entry.get("verbatim") is not None:
        photons = "verbatim"
    else:
        photons = None
    q = getattr(getattr(reaction.outputChannel, "Q", None), "value", None)
    try:
        sigma = _range(suite.cross_section(102, form="recon"))
    except KeyError:
        sigma = None
    return dict(cross_section=sigma, q_value=None if q is None else float(q),
                final_state=getattr(prov, "finalState", None), photons=photons,
                products=products,
                cross_section_path=getattr(prov, "crossSectionPath", None),
                final_state_path=getattr(prov, "finalStatePath", None))


#: Where ν̄ and the energy release are quoted: thermal incidence.
THERMAL_EV = 0.0253

#: A spectrum's model class → the ENDF law a viewer names it by.
_SPECTRUM_LAWS = {"XYs2d": "LF=1 table", "Regions2d": "LF=1 table",
                  "GeneralEvaporation": "LF=5 general evaporation",
                  "SimpleMaxwellianFission": "LF=7 Maxwellian", "Evaporation": "LF=9 evaporation",
                  "Watt": "LF=11 Watt", "MadlandNix": "LF=12 Madland-Nix",
                  "WeightedFunctionals": "weighted sum"}


def _atThermal(function) -> Optional[float]:
    if function is None:
        return None
    try:
        return float(np.asarray(function.evaluate(THERMAL_EV, outOfRange="hold")))
    except TypeError:
        return float(np.asarray(function.evaluate(THERMAL_EV)))


def _spectrumLaw(distribution) -> Optional[str]:
    energy = getattr(distribution, "energy", None)
    if energy is None:
        return None
    name = type(energy).__name__
    if name == "WeightedFunctionals":
        parts = [_SPECTRUM_LAWS.get(type(w.functional).__name__, type(w.functional).__name__)
                 for w in energy.weighted]
        return f"weighted sum of {len(parts)}: {', '.join(parts)}"
    return _SPECTRUM_LAWS.get(name, name)


def fissionSummary(suite) -> Optional[Dict[str, Any]]:
    """MT18 as a viewer shows it, or ``None`` without fission: its σ range;
    ν̄ total, prompt and delayed at thermal (``None`` for one the file does not
    state); the delayed families and their spectra's law; the prompt
    spectrum's law and where the neutron's angle is stated; the energy release
    at thermal; the chances (MT19, 20, 21, 38) with their σ range, Q and
    whether they carry a final state of their own; where the photons live
    (``"model"`` when they reached it; ``"verbatim"`` when any travels as G4NDL
    text; ``None`` without) and whether the fragment yields (``Fission/FF``)
    are kept.

    Thermal means an incident energy of 0.0253 eV.

    Parameters
    ----------
    suite
        A suite read from G4NDL.

    Returns
    -------
    dict or None
        ``None`` when the suite has no MT18. Otherwise one dict, not a table,
        with these keys, all always present:

        * ``cross_section``: ``None`` or a dict with ``count``, ``energy_min``,
          ``energy_max`` (eV) and ``sigma_max`` (barn) of MT18.
        * ``nubar_total``, ``nubar_prompt``, ``nubar_delayed``: ν̄ at thermal, or
          ``None`` when the file states none. With a total (MT452) in the
          suite, ``nubar_total`` is that and ``nubar_prompt`` is the neutron
          product's multiplicity; without one, the product's multiplicity is
          ``nubar_total`` and ``nubar_prompt`` is ``None``.
        * ``delayed_families``: number of delayed-neutron families.
        * ``delayed_spectra``: the ENDF law of their spectra as a string (the
          laws joined by ``", "`` when they differ), or ``None``.
        * ``prompt_spectrum``: the ENDF law of the prompt spectrum, or ``None``.
        * ``angular_frame``: ``"CM"``, ``"LAB"`` or ``None``.
        * ``energy_release``: ``None``, or a dict of ``totalEnergy``,
          ``nonNeutrinoEnergy``, ``promptProductKE``, ``promptNeutronKE``,
          ``promptGammaEnergy`` and ``neutrinoEnergy`` at thermal, as the model
          holds them (eV for a G4NDL suite).
        * ``chances``: a list with one dict per chance the suite has, in the
          order MT19, 20, 21, 38: ``mt``, ``directory`` (``"FC"``, ``"SC"``,
          ``"TC"``, ``"LC"``), ``cross_section`` (as above or ``None``),
          ``q_value`` (or ``None``) and ``final_state`` (whether the chance has
          a neutron distribution of its own).
        * ``photons``: ``"model"``, ``"verbatim"`` or ``None``.
        * ``verbatim_sections``: the sections below ``dataType`` 12 that travel
          as G4NDL text, as strings ``"(infoType, dataType)"``.
        * ``fragment_yields``: whether ``Fission/FF`` yields are kept.
        * ``cross_section_path``, ``final_state_path``,
          ``fragment_yields_path``: the files read, from the provenance.
    """
    reaction = suite.findReactionByENDF_MT(18)
    if reaction is None:
        return None
    prov = getattr(reaction, "provenance", None)
    channel = reaction.outputChannel
    neutron = next((p for p in channel.products if p.pid == "n"), None)
    sums = suite.sums.multiplicitySums
    total, delayed = sums.byENDF_MT(452), sums.byENDF_MT(455)
    primitive = getattr(getattr(neutron, "multiplicity", None), "form", None)
    if total is not None:
        nuTotal, nuPrompt = _atThermal(total.multiplicity.form), _atThermal(primitive)
    else:
        nuTotal, nuPrompt = _atThermal(primitive), None
    distribution = None
    if neutron is not None and neutron.distribution is not None and             "eval" in neutron.distribution.keys():
        distribution = neutron.distribution["eval"]
    data = channel.fissionFragmentData
    families = list(data.delayedNeutrons) if data is not None else []
    familyLaws = sorted({_spectrumLaw(f.product.distribution["eval"]) for f in families
                         if f.product is not None and f.product.distribution is not None
                         and "eval" in f.product.distribution.keys()})
    release = None
    releases = list(data.fissionEnergyReleases) if data is not None else []
    if releases:
        node = releases[0]
        release = {name: _atThermal(getattr(node, name)) for name in
                   ("totalEnergy", "nonNeutrinoEnergy", "promptProductKE",
                    "promptNeutronKE", "promptGammaEnergy", "neutrinoEnergy")}
    chances = []
    for mt, name in ((19, "FC"), (20, "SC"), (21, "TC"), (38, "LC")):
        r = suite.findReactionByENDF_MT(mt)
        if r is None:
            continue
        try:
            sigma = _range(suite.cross_section(mt, form="recon"))
        except KeyError:
            sigma = None
        q = getattr(getattr(r.outputChannel, "Q", None), "value", None)
        own = any(p.pid == "n" and p.distribution is not None for p in r.outputChannel.products)
        chances.append(dict(mt=mt, directory=name, cross_section=sigma,
                            q_value=None if q is None else float(q), final_state=own))
    sections = list(getattr(prov, "sections", None) or [])
    photonEntries = [e for e in sections if e.get("dataType", 0) >= 12]
    photons = ("verbatim" if any(e.get("verbatim") is not None for e in photonEntries)
               else "model" if photonEntries else None)
    kept = [f"({e['infoType']}, {e['dataType']})" for e in sections
            if e.get("verbatim") is not None and e.get("dataType", 0) < 12]
    try:
        sigma = _range(suite.cross_section(18, form="recon"))
    except KeyError:
        sigma = None
    return dict(cross_section=sigma,
                nubar_total=nuTotal, nubar_prompt=nuPrompt,
                nubar_delayed=_atThermal(delayed.multiplicity.form) if delayed else None,
                delayed_families=len(families),
                delayed_spectra=familyLaws[0] if len(familyLaws) == 1 else
                (", ".join(familyLaws) if familyLaws else None),
                prompt_spectrum=_spectrumLaw(distribution),
                angular_frame=_frame(distribution) if distribution is not None else None,
                energy_release=release,
                chances=chances,
                photons=photons,
                verbatim_sections=kept,
                fragment_yields=bool(getattr(prov, "fragmentYields", None)),
                cross_section_path=getattr(prov, "crossSectionPath", None),
                final_state_path=getattr(prov, "finalStatePath", None),
                fragment_yields_path=getattr(prov, "fragmentYieldsPath", None))


def _range(pair) -> Optional[Dict[str, Any]]:
    if pair is None:
        return None
    e, s = pair
    return dict(count=int(e.size), energy_min=float(e[0]), energy_max=float(e[-1]),
                sigma_max=float(s.max()))


def _reactionRows(suite) -> List[Dict[str, Any]]:
    """One row per MT with a cross section: Q, the σ grid's range, whether it
    is a sum (MT4, MT103-107) and whether :func:`angularBulk` serves it; for
    those, its ``rep_flag``, native Legendre order and transition energy, which
    is what a viewer says about an MF4 section before reading it."""
    angular = set(angularMTs(suite))
    sums = {int(n.ENDF_MT) for n in suite.sums if n.ENDF_MT is not None}
    rows = []
    for mt, (e, s) in crossSections(suite).items():
        q = None
        if mt not in sums:
            from kika.nuclear_data.model.endf_conversion import reactionQI
            q = reactionQI(suite.reactions.byENDF_MT(mt), suite)
        positive = np.nonzero(s > 0)[0]
        row = dict(mt=mt, q_value=None if q is None else float(q),
                   threshold=float(e[positive[0]]) if positive.size else None,
                   sum=mt in sums, angular=mt in angular, **_range((e, s)))
        if mt in angular:
            row.update(_angularShape(suite, mt))
        rows.append(row)
    return rows


def _angularShape(suite, mt: int) -> Dict[str, Any]:
    if mt == 2:
        distribution = _distribution(suite)
        angular = getattr(distribution, "angular", None)
    else:
        angular = _angular(_neutronDistribution(suite, mt))
    if angular is None or _isIsotropic(angular):
        return dict(rep_flag=0, max_order=0, transition_energy=None)
    blocks = _blocks(angular)
    legendre = next((b for b in blocks if b.kind == "Legendre"), None)
    table = next((b for b in blocks if b.kind == "table"), None)
    order = max(len(f.coefficients) - 1 for f in legendre.functions) if legendre else 0
    return dict(rep_flag=_repFlag(blocks), max_order=int(order),
                transition_energy=float(legendre.energies[-1]) if legendre and table else None)
