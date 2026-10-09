"""The radioactive decay sublibrary (ENDF NSUB=4, MF8/MT457) ↔ a standalone §12 PoPs.

A decay evaluation is a nuclide and how it decays, nothing else. GNDS writes it
as a ``PoPs`` file whose nuclide's ``nucleus`` carries ``halflife``, ``spin``,
``parity`` and ``decayData``; that is what FUDGE's ``ENDF_ITYPE_4`` produces
and what this module produces too, with FUDGE's names and its decay-path rules
(``addDecayMode``), so the two files describe the same nuclide the same way.

**What GNDS cannot hold, and where it goes.** MT457 states a few numbers no §12
node carries: each spectrum's normalisation (FD, FC and their uncertainties),
its average energy ERAV, the order in which ENDF lists lines that GNDS files
under different decay modes, and the exact text of fields (``-77.777`` for an
unknown spin, a TYPE that is not a beta transition's). FUDGE drops them and
recomputes FD = 1 and ERAV on the way back. kika keeps them in the PoPs'
``provenance`` under :data:`DECAY_KEY`:

- a **layout** that says, for each ENDF spectrum, its STYP, LCON, normalisation
  record and which model emission each of its lines and its continuum is;
- **field overrides**: after the section has been rebuilt from the model and
  the layout, every 11-column field that still differs from the tape is kept as
  ``(rebuilt text, tape text)`` and replaced on the way out **only if the
  rebuilt field is still the same text** -- so an override restores a digit the
  model cannot express and never undoes an edit to the model.

From a GNDS-read PoPs, which has neither, the section is built the way FUDGE's
``PoPs_toENDF6`` builds it, except that spectra of one radiation are merged
across decay modes (ENDF lists one spectrum per STYP) and NER counts lines only.

Measured on ENDF/B-VIII.1 (3 821 nuclides, roadmap E7b): ENDF → model → ENDF
byte for byte, and model → GNDS → model identical.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from kika.nuclear_data.model import (AVERAGE_ENERGY_LABELS, SPECTRUM_LABELS, Axes, Axis,
                                     TRANSITION_TYPES, Continuum, ConversionReport,
                                     Decay, DecayData, DecayMode, DecayModes, DecayPath,
                                     Discrete, MetaStable, Nuclide, Particle,
                                     PhysicalQuantity, PoPs, Product, Shell, Spectrum, XYs1d,
                                     pidFromZA, zaFromPid)
from kika.nuclear_data.model.enums import (ENDF_INT_TO_INTERPOLATION,
                                           INTERPOLATION_TO_ENDF_INT, Interpolation)
from kika.nuclear_data.model.quantities import Uncertainty

__all__ = ["decodeDecaySublibrary", "decodeFissionYieldSublibrary", "encodeSublibrary",
           "writeSublibraryTape", "encodeDecaySublibrary", "writeDecayTape",
           "DECAY_KEY", "TAPE_KEY", "DECAY_NSUB", "SFY_NSUB", "NFY_NSUB", "MF8_SUBLIBRARIES"]

DECAY_NSUB = 4
SFY_NSUB = 5
NFY_NSUB = 11
#: The ENDF sublibraries whose evaluation is a standalone PoPs.
MF8_SUBLIBRARIES = (DECAY_NSUB, SFY_NSUB, NFY_NSUB)
DECAY_KEY = "mf8_457"
#: The tape's own conventions (sequence numbers, control records), kept so the
#: copy is byte for byte.
TAPE_KEY = "endf_tape"
#: Sections a sublibrary tape carries that a PoPs has no node for, as read.
VERBATIM_KEY = "verbatim_sections"
NEUTRON_MASS_AMU = 1.00866491595

#: RTYP's digits → the decay mode's name, FUDGE's ``decayType``.
DECAY_TYPES = {0: "gamma", 1: "beta-", 2: "beta+ or e.c.", 3: "IT", 4: "alpha", 5: "n",
               6: "SF", 7: "p", 8: "e-", 9: "xray", 10: "unknown"}
_DECAY_DIGITS = {name: digit for digit, name in DECAY_TYPES.items()}

#: The particle a decay step (RTYP digit) emits, FUDGE's ``decayParticles``.
_EMITTED = {0: "photon", 1: "e-", 2: "e+", 3: None, 4: "He4", 5: "n", 6: None, 7: "H1",
            8: "e-", 9: "photon", 10: "nu_e-_anti", 11: "nu_e"}

#: ``spectrum/@pid`` by STYP. FUDGE has none for STYP=6 and refuses those
#: spectra; kika names the fragments ``SF`` and says so.
_SPECTRUM_PID = {0: "photon", 1: "e-", 2: "e+", 4: "He4", 5: "n", 6: "SF", 7: "H1",
                 8: "e-", 9: "photon", 10: "nu_e-_anti", 11: "nu_e"}
_STYP = {label: styp for styp, label in SPECTRUM_LABELS.items()}

_UNKNOWN_SPIN = -77.777


# ---------------------------------------------------------------------------
# Small conversions
# ---------------------------------------------------------------------------

def _quantity(value: float, sigma: float, unit: str = "", label: Optional[str] = None):
    return PhysicalQuantity(value=float(value), unit=unit, label=label,
                            uncertainty=Uncertainty(float(sigma)) if sigma else None)


def _sigma(quantity) -> float:
    uncertainty = getattr(quantity, "uncertainty", None)
    return 0.0 if uncertainty is None else float(uncertainty.value)


def _rtypKey(rtyp: float) -> str:
    """``1.5`` → ``"1.5"``: the digits FUDGE's ``getRTYP`` reads."""
    text = repr(float(rtyp))
    return text[:-2] if text.endswith(".0") else text


def _rtypDigits(rtyp: float) -> List[int]:
    initial, _, rest = _rtypKey(rtyp).partition(".")
    return [int(initial)] + [int(d) for d in rest]


def _modeName(rtyp: float) -> str:
    return ",".join(DECAY_TYPES.get(d, "unknown") for d in _rtypDigits(rtyp))


def _rtypFromMode(mode: str) -> float:
    digits = [_DECAY_DIGITS[name] for name in mode.split(",")]
    return float(f"{digits[0]}." + "".join(str(d) for d in digits[1:])) if len(digits) > 1 \
        else float(digits[0])


def _isotope(za: int) -> str:
    return pidFromZA(za)


# ---------------------------------------------------------------------------
# ENDF → model
# ---------------------------------------------------------------------------

def decodeDecaySublibrary(endf, report: Optional[ConversionReport] = None,
                          sourcePath=None) -> Tuple[PoPs, ConversionReport]:
    """A decay sublibrary tape (parsed by ``read_endf``) → a standalone PoPs.

    *sourcePath*, the tape the object was parsed from, lets the fields the parse
    normalised (a non-canonical ``-1.00000+0``) and the tape's control records
    (blank or zero SEND/FEND/MEND) be kept exactly; without it they are written
    the canonical way.
    """
    from .decode import TAPE_ID_KEY, decodeMF1MT451

    report = report if report is not None else ConversionReport()
    mt451 = endf.mf[1].mt[451]
    _, style, provenance, report = decodeMF1MT451(mt451, report)
    tapeId = getattr(endf, "tape_id", None)
    if tapeId is not None:
        provenance.headerFields[TAPE_ID_KEY] = tapeId
    pops = PoPs(name="protare_internal", version="1.0", styles=[style], provenance=provenance)

    mf8 = endf.mf.get(8)
    section = getattr(mf8, "mt", {}).get(457) if mf8 is not None else None
    if section is None:
        report.lost("this decay sublibrary tape has no MF8/MT457, so no nuclide is decoded")
        pops.report = report
        return pops, report
    for mt in sorted(getattr(mf8, "mt", {})):
        if mt not in (454, 457, 459):
            report.unsupportedNode(f"MF8/MT{mt} in a decay sublibrary tape is not decoded")

    za = int(round(section._za))
    lis, liso = int(section._lis), int(section._liso)
    pid = pidFromZA(za, lis)
    awr = float(section._awr)
    mass = PhysicalQuantity(value=awr * NEUTRON_MASS_AMU, unit="amu")
    if za < 1000:
        # The free neutron (dec-000_Nn_001): a §12 baryon, which carries its
        # halflife and decayData itself.
        nuclide = Particle(id=pid, mass=mass, charge=0)
    else:
        nuclide = Nuclide(id=pid, Z=za // 1000, A=za % 1000, charge=0, mass=mass,
                          nuclearLevel=lis, decayDataOnNucleus=True,
                          energy=PhysicalQuantity(value=float(getattr(mt451, "_elis", 0.0) or 0.0),
                                                  unit="eV"))
    if section.spin > -77.8 and not (-77.78 < section.spin < -77.77):
        nuclide.spin = PhysicalQuantity(value=float(section.spin), unit="hbar")
    if section.parity:
        nuclide.parity = int(section.parity)
    if lis:
        alias = MetaStable(id=f"{_isotope(za)}_m{liso}", pid=pid, metaStableIndex=liso)
        pops.aliases[alias.id] = alias
    if section.is_stable:
        nuclide.halflife = "stable"
    else:
        nuclide.halflife = _quantity(section.halflife, section.dhalflife, "s")

    decayData = DecayData()
    energies = section.average_energies
    if any(v != 0 for v in energies):
        for i, label in enumerate(AVERAGE_ENERGY_LABELS[:len(energies) // 2]):
            value, sigma = energies[2 * i], energies[2 * i + 1]
            if value >= 0:
                decayData.averageEnergies.append(_quantity(value, sigma, "eV", label))

    byKey: Dict[str, DecayMode] = {}
    if not section.is_stable:
        for index, mode in enumerate(section.decay_modes):
            decayMode = DecayMode(label=str(index), mode=_modeName(mode.rtyp),
                                  probability=float(mode.br),
                                  probabilityUncertainty=float(mode.dbr) if mode.dbr else None,
                                  probabilityLabel="BR",
                                  Q=_quantity(mode.q, mode.dq, "eV", "eval"))
            decayMode.decayPath = _decayPath(pops, za, mode.rtyp, int(mode.rfs))
            decayData.decayModes.decayModes.append(decayMode)
            if _rtypKey(mode.rtyp) in byKey:
                report.warn(f"MF8/MT457: two decay modes state RTYP={_rtypKey(mode.rtyp)} "
                            f"(they differ in RFS); a line names its mode by RTYP alone, "
                            f"so its spectra go to the last of them, as FUDGE files them")
            byKey[_rtypKey(mode.rtyp)] = decayMode
    nuclide.decayData = decayData
    pops.add(nuclide)

    layout = []
    for spectrum in section.spectra:
        layout.append(_decodeSpectrum(spectrum, decayData, byKey, report))

    book = {"layout": layout, "nc": len(energies) // 2, "mat": section._mat,
            "pad": (section.pad.pairs, section.pad.values, section.pad.interp)}
    rebuilt = str(_buildSection(pops, nuclide, book)).split("\n")[:-1]
    source = str(section).split("\n")[:-1]
    if sourcePath is not None:
        raw = _rawSection(sourcePath, 8, 457)
        if raw:
            source = raw
    book["overrides"] = _overrides(rebuilt, source, report)
    provenance.headerFields[DECAY_KEY] = book
    _keepTapeConventions(provenance, mt451, sourcePath)
    _attachYields(nuclide, mf8, False, provenance, sourcePath, report)
    _keepUnmodelledSections(provenance, endf, sourcePath, report)
    if not len(decayData.decayModes) and not decayData.averageEnergies:
        # A stable nuclide states nothing to decay: no decayData, as FUDGE
        # writes it (the empty one above was only for the rebuild).
        nuclide.decayData = None
        if isinstance(nuclide, Nuclide):
            nuclide.decayDataOnNucleus = False
    pops.report = report
    return pops, report


def _decayPath(pops: PoPs, parentZA: int, rtyp: float, rfs: int) -> DecayPath:
    """FUDGE's ``addDecayMode``, step by step through RTYP's digits."""
    path = DecayPath()
    digits = _rtypDigits(rtyp)
    za = parentZA
    for step, digit in enumerate(digits):
        last = step == len(digits) - 1
        emitted = _EMITTED.get(digit)
        decay = Decay(index=step, complete=False if digit in (2, 6) else None,
                      mode={3: "isomeric transition", 6: "spontaneous fission"}.get(digit))
        dz = da = 0
        if emitted is not None:
            if emitted != "e+":
                decay.products.append(Product(pid=emitted, label=emitted))
            if emitted == "e-":
                dz = -1
                decay.products.append(Product(pid="nu_e-_anti", label="nu_e-_anti"))
            elif emitted == "e+":
                dz = 1
            elif emitted == "n":
                da = 1
            elif emitted == "H1":
                dz, da = 1, 1
            elif emitted == "He4":
                dz, da = 2, 4
        za = za - (1000 * dz + da)
        if digit != 6:
            residual = "n" if za == 1 else pidFromZA(za)
            if last and rfs:
                level = f"{residual}_e{rfs}"
                aliasId = f"{residual}_m{rfs}"
                pops.aliases.setdefault(aliasId, MetaStable(id=aliasId, pid=level,
                                                            metaStableIndex=rfs))
                residual = aliasId
            labels = {p.label for p in decay.products}
            label = residual if residual not in labels else f"{residual}__a"
            decay.products.append(Product(pid=residual, label=label))
        path.decays.append(decay)
    return path


def _modeFor(byKey, rtyp, decayData, report, what) -> Optional[DecayMode]:
    mode = byKey.get(_rtypKey(rtyp))
    if mode is not None:
        return mode
    modes = list(decayData.decayModes)
    if not modes:
        return None
    initial = _rtypDigits(rtyp)[0]
    fallback = next((m for m in modes if _rtypDigits(_rtypFromMode(m.mode))[0] == initial),
                    modes[0])
    report.warn(f"MF8/MT457: a {what} states RTYP={_rtypKey(rtyp)}, which is no decay mode "
                f"of the nuclide; it is filed under mode {fallback.label} ({fallback.mode}) "
                f"and the ENDF route keeps its RTYP (FUDGE drops it)")
    return fallback


def _spectrumIn(mode: DecayMode, styp: int) -> Tuple[int, Spectrum]:
    label = SPECTRUM_LABELS.get(styp, f"STYP={styp}")
    for i, spectrum in enumerate(mode.spectra):
        if spectrum.label == label:
            return i, spectrum
    spectrum = Spectrum(label=label, pid=_SPECTRUM_PID.get(styp, "unknown"))
    mode.spectra.append(spectrum)
    return len(mode.spectra) - 1, spectrum


def _decodeSpectrum(endfSpectrum, decayData, byKey, report) -> dict:
    styp = int(endfSpectrum.styp)
    if styp == 6:
        report.warn("MF8/MT457: a spontaneous-fission spectrum (STYP=6) has no particle "
                    "in FUDGE's table and FUDGE refuses it; kika files it with pid 'SF'")
    modes = list(decayData.decayModes)
    entry = {"styp": endfSpectrum.styp, "lcon": endfSpectrum.lcon, "head": list(endfSpectrum.head),
             "c1": endfSpectrum.c1, "l2": endfSpectrum.l2, "n2": endfSpectrum.n2,
             "items": [], "covariance": None}
    for line in endfSpectrum.lines:
        mode = _modeFor(byKey, line.rtyp, decayData, report, "discrete line")
        if mode is None:
            report.lost("MF8/MT457: a discrete line of a nuclide with no decay mode is not "
                        "in the model")
            continue
        index, spectrum = _spectrumIn(mode, styp)
        spectrum.emissions.append(_discrete(line, styp))
        entry["items"].append(("d", modes.index(mode), index, len(spectrum.emissions) - 1))
    continuum = endfSpectrum.continuum
    if continuum is not None:
        mode = _modeFor(byKey, continuum.rtyp, decayData, report, "continuum")
        table = continuum.table
        if mode is None or len(table.interp) != 1:
            report.lost("MF8/MT457: a continuum the model cannot place (no decay mode, or "
                        "more than one interpolation region) is not in the model")
        else:
            index, spectrum = _spectrumIn(mode, styp)
            spectrum.emissions.append(Continuum(spectrum=XYs1d(
                xs=list(table.x), ys=list(table.y),
                interpolation=ENDF_INT_TO_INTERPOLATION[int(table.interp[0][1])],
                axes=Axes(axes=[Axis(index=1, label="energy_out", unit="eV"),
                                Axis(index=0, label="P(energy_out)", unit="1/eV")]))))
            entry["items"].append(("c", modes.index(mode), index, len(spectrum.emissions) - 1))
        if continuum.covariance is not None:
            entry["covariance"] = continuum.covariance
            report.lost("MF8/MT457: the continuum's covariance (LCOV != 0) has no §12 node; "
                        "only the ENDF route keeps it")
    return entry


def _discrete(line, styp: int) -> Discrete:
    v = list(line.values) + [0.0] * (12 - len(line.values))
    rtyp, kind, ri, dri, ris, dris = v[:6]
    discrete = Discrete(intensity=_quantity(ri, dri), energy=_quantity(line.er, line.der, "eV"),
                        type=TRANSITION_TYPES[int(kind) - 1] if kind in (1.0, 2.0, 3.0) else None)
    if ris:
        if styp == 2:
            discrete.positronEmissionIntensity = _quantity(ris, dris)
        else:
            discrete.internalPairFormationCoefficient = _quantity(ris, dris)
    nt = len(line.values)
    if nt > 6:
        discrete.internalConversionCoefficients.append(Shell("total", v[6], v[7] or None))
    if nt > 8:
        discrete.internalConversionCoefficients.append(Shell("K", v[8], v[9] or None))
        discrete.internalConversionCoefficients.append(Shell("L", v[10], v[11] or None))
    return discrete


# ---------------------------------------------------------------------------
# model → ENDF
# ---------------------------------------------------------------------------

def _decayNuclide(pops: PoPs) -> Particle:
    """The one particle a sublibrary evaluation is about: the one with decay data
    or fission yields, or the only particle there is (a stable nuclide)."""
    nuclides = [p for p in pops.particles.values()
                if p.decayData is not None or getattr(p, "fissionFragmentData", None) is not None]
    if not nuclides and len(pops.particles) == 1:
        nuclides = list(pops.particles.values())
    if len(nuclides) != 1:
        raise ValueError(f"a decay or fission-yield evaluation is about one nuclide; this "
                         f"PoPs has {len(nuclides)} candidates")
    return nuclides[0]


def _productYield(nuclide):
    data = getattr(nuclide, "fissionFragmentData", None)
    yields = list(getattr(data, "productYields", None) or [])
    return yields[0] if yields else None


def _nsub(pops: PoPs, nuclide) -> int:
    fields = getattr(pops.provenance, "headerFields", None) or {}
    if fields.get("nsub") in MF8_SUBLIBRARIES:
        return int(fields["nsub"])
    if nuclide.decayData is not None or nuclide.halflife == "stable":
        return DECAY_NSUB
    productYield = _productYield(nuclide)
    induced = productYield is not None and any(e.incidentEnergies for e in productYield.elapsedTimes)
    return NFY_NSUB if induced else SFY_NSUB


def _rfs(pops: PoPs, mode: DecayMode) -> int:
    for decay in mode.decayPath:
        for product in decay.products:
            alias = pops.aliases.get(product.pid)
            if alias is not None:
                return int(alias.metaStableIndex)
    return 0


def _lineValues(mode: DecayMode, discrete: Discrete) -> List[float]:
    kind = (TRANSITION_TYPES.index(discrete.type) + 1) if discrete.type in TRANSITION_TYPES else 0
    ris = discrete.positronEmissionIntensity or discrete.internalPairFormationCoefficient
    values = [_rtypFromMode(mode.mode), float(kind),
              discrete.intensity.value, _sigma(discrete.intensity),
              ris.value if ris is not None else 0.0, _sigma(ris)]
    for shell in discrete.internalConversionCoefficients:
        values += [shell.value, shell.uncertainty or 0.0]
    return values


def _defaultLayout(decayData: DecayData) -> List[dict]:
    """One ENDF spectrum per radiation, lines then the continuum, modes merged."""
    order: List[int] = []
    items: Dict[int, List[tuple]] = {}
    for m, mode in enumerate(decayData.decayModes):
        for s, spectrum in enumerate(mode.spectra):
            styp = _STYP.get(spectrum.label)
            if styp is None:
                raise ValueError(f"spectrum label {spectrum.label!r} has no ENDF STYP")
            if styp not in items:
                order.append(styp)
                items[styp] = []
            for e, emission in enumerate(spectrum.emissions):
                items[styp].append(("c" if isinstance(emission, Continuum) else "d", m, s, e))
    layout = []
    for styp in order:
        entries = sorted(items[styp], key=lambda item: item[0] == "c")
        discretes = [i for i in entries if i[0] == "d"]
        continua = [i for i in entries if i[0] == "c"]
        lcon = 2 if discretes and continua else (1 if continua else 0)
        layout.append({"styp": float(styp), "lcon": lcon, "head": None, "c1": 0.0, "l2": 0,
                       "n2": len(discretes), "items": entries, "covariance": None})
    return layout


def _emission(decayData, item):
    _, m, s, e = item
    mode = decayData.decayModes.decayModes[m]
    return mode, mode.spectra[s].emissions[e]


def _layoutFits(layout, decayData) -> bool:
    try:
        seen = set()
        for entry in layout:
            for item in entry["items"]:
                mode, emission = _emission(decayData, item)
                if (item[0] == "c") != isinstance(emission, Continuum):
                    return False
                seen.add(tuple(item[1:]))
        total = sum(len(sp.emissions) for mode in decayData.decayModes for sp in mode.spectra)
        return len(seen) == total
    except (IndexError, KeyError, TypeError):
        return False


def _buildSection(pops: PoPs, nuclide: Nuclide, book: Optional[dict],
                  report: Optional[ConversionReport] = None):
    from kika.endf.classes.mf12.base import PhotonTable
    from kika.endf.classes.mf8.decay import (ContinuousSpectrum, DecaySpectrum, DiscreteLine,
                                             MF8MT457)
    from kika.endf.utils import PadStyle

    book = book or {}
    decayData = nuclide.decayData if nuclide.decayData is not None else DecayData()
    za = getattr(nuclide, "ZA", None) or zaFromPid(nuclide.id)
    section = MF8MT457(number=457, _za=float(za), _lis=int(getattr(nuclide, "nuclearLevel", 0)),
                       _mat=book.get("mat"))
    mass = nuclide.mass
    section._awr = float(mass.value) / NEUTRON_MASS_AMU if mass is not None else 0.0
    section._liso = next((a.metaStableIndex for a in pops.aliases.values() if a.pid == nuclide.id), 0)
    stable = nuclide.halflife == "stable"
    section._nst = 1 if stable else 0
    if not stable and isinstance(nuclide.halflife, PhysicalQuantity):
        section.halflife = float(nuclide.halflife.convertedTo("s").value)
        section.dhalflife = _sigma(nuclide.halflife)
    if "pad" in book:
        section.pad = PadStyle(*book["pad"])

    nc = book.get("nc") or (17 if any(e.label in AVERAGE_ENERGY_LABELS[3:]
                                      for e in decayData.averageEnergies) else 3)
    energies = []
    for label in AVERAGE_ENERGY_LABELS[:nc]:
        quantity = decayData.averageEnergy(label)
        energies += [quantity.value, _sigma(quantity)] if quantity is not None else [0.0, 0.0]
    section.average_energies = energies

    section.spin = nuclide.spin.value if nuclide.spin is not None else _UNKNOWN_SPIN
    section.parity = float(nuclide.parity) if nuclide.parity is not None else 0.0
    modes = list(decayData.decayModes)
    if stable or not modes:
        section.modes_raw, section.modes_n2 = [0.0] * 6, 0
    else:
        raw = []
        for mode in modes:
            raw += [_rtypFromMode(mode.mode), float(_rfs(pops, mode)),
                    mode.Q.value if mode.Q is not None else 0.0, _sigma(mode.Q),
                    mode.probability, mode.probabilityUncertainty or 0.0]
        section.modes_raw, section.modes_n2 = raw, len(modes)

    layout = book.get("layout")
    if layout is None or not _layoutFits(layout, decayData):
        if layout is not None and report is not None:
            report.warn("MF8/MT457: the decay spectra no longer match the layout they were "
                        "read with, so they are written one per radiation, FUDGE's way")
        layout = _defaultLayout(decayData)
    for entry in layout:
        lines, continuum = [], None
        for item in entry["items"]:
            mode, emission = _emission(decayData, item)
            if item[0] == "d":
                lines.append(DiscreteLine(er=emission.energy.value, der=_sigma(emission.energy),
                                          values=_lineValues(mode, emission)))
            else:
                function = emission.spectrum
                interpolation = getattr(function, "interpolation", Interpolation.linlin)
                continuum = ContinuousSpectrum(table=PhotonTable(
                    c1=_rtypFromMode(mode.mode), c2=0.0, l1=0,
                    l2=1 if entry.get("covariance") else 0,
                    interp=[(len(function.xs), INTERPOLATION_TO_ENDF_INT[Interpolation(interpolation)])],
                    x=[float(x) for x in function.xs], y=[float(y) for y in function.ys]),
                    covariance=entry.get("covariance"))
        head = entry.get("head")
        if head is None:
            head = _derivedHead(lines, continuum)
            if report is not None and lines and int(entry["styp"]) in (1, 2):
                report.approximated(
                    f"MF8/MT457 STYP={int(entry['styp'])}: ERAV is derived as the sum of "
                    f"ER*RI over the lines (FUDGE's rule); for a beta spectrum ER is the "
                    f"end-point energy, so this overstates the mean")
        section.spectra.append(DecaySpectrum(styp=entry["styp"], lcon=entry["lcon"], head=list(head),
                                             c1=entry.get("c1", 0.0), l2=entry.get("l2", 0),
                                             n2=entry.get("n2", len(lines)), lines=lines,
                                             continuum=continuum))
    return section


def _derivedHead(lines, continuum) -> List[float]:
    """``(FD, dFD, ERAV, dERAV, FC, dFC)`` with no layout: FUDGE's
    ``PoPs_toENDF6`` (FD = 1, ERAV = sum ER*RI with its propagated uncertainty),
    FD = 0 when there is no line and FC = 1 when there is a continuum."""
    from math import sqrt

    erav = variance = 0.0
    for line in lines:
        er, der, ri, dri = line.er, line.der, line.values[2], line.values[3]
        erav += er * ri
        if er and ri:
            variance += (er * ri) ** 2 * ((der / er) ** 2 + (dri / ri) ** 2)
    return [1.0 if lines else 0.0, 0.0, erav, sqrt(variance),
            1.0 if continuum is not None else 0.0, 0.0]


def _fields(line: str) -> List[str]:
    text = line[:66].ljust(66)
    return [text[k:k + 11] for k in range(0, 66, 11)]


def _overrides(rebuilt: List[str], source: List[str], report, what: str = "MF8/MT457") -> dict:
    """Every field of *source* the rebuild did not reproduce: ``(rebuilt, tape)``."""
    if len(rebuilt) != len(source):
        report.warn(f"{what}: the section rebuilt from the model has {len(rebuilt)} records "
                    f"where the tape has {len(source)}; it is kept verbatim for the way back")
        return {"verbatim": source}
    out = {}
    for i, (a, b) in enumerate(zip(rebuilt, source)):
        if a[:66] == b[:66]:
            continue
        for k, (x, y) in enumerate(zip(_fields(a), _fields(b))):
            if x != y:
                out[f"{i}:{k}"] = (x, y)
    return {"fields": out, "records": len(source)}


def _applyOverrides(text: List[str], overrides: dict, report,
                    what: str = "MF8/MT457") -> List[str]:
    if not overrides:
        return text
    if "verbatim" in overrides:
        report.warn(f"{what} is written as it was read: the model could not rebuild it")
        return list(overrides["verbatim"])
    if overrides.get("records") != len(text):
        report.warn(f"{what}: the section changed length, so the fields kept from the tape "
                    f"are not applied")
        return text
    lines = list(text)
    stale = 0
    for key, (rebuilt, tape) in overrides["fields"].items():
        i, k = (int(part) for part in key.split(":"))
        fields = _fields(lines[i])
        if fields[k] != rebuilt:
            stale += 1
            continue
        fields[k] = tape
        lines[i] = "".join(fields) + lines[i][66:]
    if stale:
        report.warn(f"{what}: {stale} field(s) kept from the tape were not applied because "
                    f"the model changed them")
    return lines


class _TextSection:
    """A section whose text is already final (overrides applied)."""

    def __init__(self, lines: List[str], mat: int, mf: int = 8):
        from kika.endf.utils import format_endf_send_record
        self._text = "\n".join(lines + [format_endf_send_record(mat, mf)])

    def __str__(self) -> str:
        return self._text


def encodeSublibrary(pops: PoPs, mat: Optional[int] = None,
                     report: Optional[ConversionReport] = None):
    """A decay or fission-yield PoPs → ``[(MF, MT, section), …]``, the MAT, the report.

    MF1/451 always; MF8/457 for a decay evaluation; MF8/454 and /459 for the
    fission yields the nuclide carries.
    """
    from .encode import encodeMF1MT451
    from .fission_yields import FPY_KEY, encodeFissionYields

    report = report if report is not None else ConversionReport()
    nuclide = _decayNuclide(pops)
    provenance = pops.provenance if pops.provenance is not None else _synthesiseProvenance(pops, report)
    mat = int(mat if mat is not None else provenance.mat)
    fields = provenance.headerFields or {}
    mt451, report = encodeMF1MT451(SimpleNamespace(styles=pops.styles, provenance=provenance),
                                   mat=mat, report=report)
    sections = [(1, 451, mt451)]
    awr = float(nuclide.mass.value) / NEUTRON_MASS_AMU if nuclide.mass is not None else 0.0
    za = getattr(nuclide, "ZA", None) or zaFromPid(nuclide.id)
    productYield = _productYield(nuclide)
    if _nsub(pops, nuclide) == DECAY_NSUB:
        book = dict(fields.get(DECAY_KEY) or {})
        book["mat"] = mat
        section = _buildSection(pops, nuclide, book, report)
        lines = str(section).split("\n")[:-1]
        lines = _applyOverrides(lines, book.get("overrides"), report)
        yieldSections = (encodeFissionYields(productYield, za, awr, fields.get(FPY_KEY) or {},
                                             mat, report) if productYield is not None else [])
        # ENDF's order within MF8: MT454, MT457, MT459.
        sections += [s for s in yieldSections if s[1] == 454]
        sections.append((8, 457, _TextSection(lines, mat)))
        sections += [s for s in yieldSections if s[1] == 459]
    elif productYield is not None:
        sections += encodeFissionYields(productYield, za, awr, fields.get(FPY_KEY) or {},
                                        mat, report)
    else:
        report.lost("this fission-yield evaluation's nuclide carries no productYield, so the "
                    "tape has MF1 only")
    for mf, mt, lines in fields.get(VERBATIM_KEY) or []:
        sections.append((mf, mt, _TextSection([line[:66].ljust(66) + _idColumns(mat, mf, mt, n + 1)
                                               for n, line in enumerate(lines)], mat, mf)))
    sections.sort(key=lambda item: (item[0], item[1]))
    return sections, mat, report


def _idColumns(mat: int, mf: int, mt: int, ns: int) -> str:
    return f"{mat:>4}{mf:>2}{mt:>3}{ns:>5}"


def encodeDecaySublibrary(pops: PoPs, mat: Optional[int] = None,
                          report: Optional[ConversionReport] = None):
    """The name E7b gave :func:`encodeSublibrary`."""
    return encodeSublibrary(pops, mat, report)


def _synthesiseProvenance(pops: PoPs, report: ConversionReport):
    """MF1/451 for a PoPs read from GNDS: FUDGE's ``PoPs_toENDF6/database.py``."""
    import re

    from kika.nuclear_data.model import EndfProvenance
    from .mf1_header import _parseVersion, nlibFromLibrary

    nuclide = _decayNuclide(pops)
    style = next(iter(pops.styles), None)
    text = []
    documentation = getattr(style, "documentation", None)
    endfCompatible = getattr(documentation, "endfCompatible", None)
    if endfCompatible is not None and endfCompatible.text:
        text = endfCompatible.text.split("\n")
    match = re.search(r"MATERIAL\s+(\d+)", text[2].upper()) if len(text) > 2 else None
    if match is None:
        raise ValueError("a decay evaluation read from GNDS names its MAT only in its ENDF "
                         "documentation (third record, 'MATERIAL nnn'), and this one has "
                         "none; pass mat=")
    version = _parseVersion(getattr(style, "version", None)) or (0, 0, 0)
    modes = list(nuclide.decayData.decayModes) if nuclide.decayData is not None else []
    nsub = _nsub(pops, nuclide)
    productYield = _productYield(nuclide)
    emax = 0.0
    if nsub == NFY_NSUB and productYield is not None:
        emax = max((ie.energy.value for e in productYield.elapsedTimes for ie in e.incidentEnergies),
                   default=0.0)
    fields = {"lrp": -1, "lfi": int(any(m.mode == "SF" for m in modes)),
              "nlib": nlibFromLibrary(getattr(style, "library", None)) or 0,
              "nmod": version[2],
              "elis": float(nuclide.energy.value) if getattr(nuclide, "energy", None) else 0.0,
              "sta": 1 if modes else 0, "lis": int(getattr(nuclide, "nuclearLevel", 0)),
              "liso": next((a.metaStableIndex for a in pops.aliases.values()
                            if a.pid == nuclide.id), 0),
              "nfor": 6, "awi": 1.0 if nsub == NFY_NSUB else 0.0, "emax": emax,
              "lrel": version[1], "nsub": nsub, "nver": version[0], "ldrv": 0, "temp": 0.0}
    report.warn(f"MF1/451 of the NSUB={nsub} tape is derived from the PoPs, FUDGE's rule "
                f"(LRP=-1, MAT from the documentation's third record)")
    awr = float(nuclide.mass.value) / NEUTRON_MASS_AMU if nuclide.mass is not None else 0.0
    za = getattr(nuclide, "ZA", None) or zaFromPid(nuclide.id)
    return EndfProvenance(mat=int(match.group(1)), awr=awr, za=za, headerFields=fields,
                          descriptiveText=[line[:66] for line in text], directory=[])


def writeDecayTape(pops: PoPs, path, mat: Optional[int] = None, tapeId: Optional[str] = None,
                   report: Optional[ConversionReport] = None) -> ConversionReport:
    """The name E7b gave :func:`writeSublibraryTape`."""
    return writeSublibraryTape(pops, path, mat, tapeId, report)


def writeSublibraryTape(pops: PoPs, path, mat: Optional[int] = None,
                        tapeId: Optional[str] = None,
                        report: Optional[ConversionReport] = None) -> ConversionReport:
    """Write a decay or fission-yield PoPs out as its ENDF-6 sublibrary tape."""
    import os
    from pathlib import Path

    from kika.endf.writers.assemble import assembleTape
    from kika.endf.writers.update_directory import update_mf1_directory
    from .decode import TAPE_ID_KEY

    sections, mat, report = encodeSublibrary(pops, mat, report)
    kept = None
    if tapeId is None and pops.provenance is not None:
        kept = (pops.provenance.headerFields or {}).get(TAPE_ID_KEY)
    path = Path(os.fspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    text = assembleTape(sections, mat, tapeId, tapeRecord=kept)
    path.write_text(text, newline="\n")
    book = ((pops.provenance.headerFields or {}).get(TAPE_KEY) or {}) if pops.provenance else {}
    # The directory as read stays when no section changed length: nine decay
    # tapes of ENDF/B-VIII.1 state an NC one or two records off (Br-88 says
    # 535 for a 537-record MT457), and a copy should be a copy. Otherwise the
    # counts are rebuilt from what was written.
    if book.get("counts") is None or book["counts"] != _sectionCounts(text.split("\n")):
        if not update_mf1_directory(str(path), added_sections={(mf, mt) for mf, mt, _ in sections}):
            report.warn(f"the MF1/451 directory of {path.name} could not be rebuilt")
    if book.get("sequence") is False or book.get("control"):
        # The source tape's conventions, so the round trip is byte for byte:
        # no sequence numbers (columns 76-80), and each SEND/FEND/MEND/TEND
        # spelled as it was (NNDC writes some with zeros and some blank).
        text = path.read_text().split("\n")
        control = list(book.get("control") or [])
        indices = [i for i, line in enumerate(text) if i and _isControl(line)]
        if control and len(control) == len(indices):
            for i, kept in zip(indices, control):
                text[i] = kept + text[i][66:]
        if book.get("sequence") is False:
            text = [line[:75] for line in text]
        path.write_text("\n".join(text), newline="\n")
    return report


def _isControl(line: str) -> bool:
    return len(line) >= 75 and line[72:75] == "  0"


def _rawSection(path, mf: int, mt: int) -> List[str]:
    """The section's records as the tape spells them, ID columns kept, SEND out."""
    out = []
    with open(path, encoding="latin-1") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if len(line) < 75:
                continue
            try:
                lmf, lmt = int(line[70:72]), int(line[72:75])
            except ValueError:
                continue
            if lmf == mf and lmt == mt:
                out.append(line)
    return out


def _keepUnmodelledSections(provenance, endf, sourcePath, report) -> None:
    """Every section but MF1/451 and MF8/454/457/459, as read, and named.

    36 decay tapes of ENDF/B-VIII.1 carry spontaneous-fission neutron data
    (Cf-252: MF1/452/455/456, MF5/18/455, MF31, MF35). A PoPs has no node for
    a reaction's multiplicities or spectra -- FUDGE's ``ENDF_ITYPE_4`` drops
    them -- so they ride along in the provenance and only the ENDF route
    writes them back.
    """
    kept = []
    for mf in sorted(getattr(endf, "mf", {})):
        for mt in sorted(getattr(endf.mf[mf], "mt", {})):
            if (mf, mt) == (1, 451) or (mf == 8 and mt in (454, 457, 459)):
                continue
            lines = _rawSection(sourcePath, mf, mt) if sourcePath is not None else []
            if not lines:
                lines = str(endf.mf[mf].mt[mt]).split("\n")[:-1]
            kept.append((mf, mt, [line[:66] for line in lines]))
    if kept:
        provenance.headerFields[VERBATIM_KEY] = kept
        report.unsupportedNode(
            f"{len(kept)} section(s) ({', '.join(f'MF{mf}/MT{mt}' for mf, mt, _ in kept)}) "
            f"have no node in a decay or fission-yield PoPs (spontaneous-fission neutron "
            f"data); they are kept as read for the ENDF route and absent from GNDS")


def _keepTapeConventions(provenance, mt451, sourcePath) -> None:
    raw = (getattr(mt451, "_text_lines", None) or [""])[0].rstrip("\r\n")
    # NNDC's decay and yield tapes stop at column 75: no sequence numbers.
    book = {"sequence": len(raw) > 75}
    if sourcePath is not None:
        book["control"] = _controlRecords(sourcePath)
        with open(sourcePath, encoding="latin-1") as handle:
            book["counts"] = _sectionCounts([line.rstrip("\r\n") for line in handle])
    provenance.headerFields[TAPE_KEY] = book


def _sectionCounts(lines) -> dict:
    """``{"MF/MT": records}`` over the data records of a tape (SEND and the like out)."""
    counts: dict = {}
    for line in lines[1:]:
        if len(line) < 75 or _isControl(line):
            continue
        try:
            key = f"{int(line[70:72])}/{int(line[72:75])}"
        except ValueError:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _attachYields(nuclide, mf8, induced: bool, provenance, sourcePath, report) -> None:
    from kika.nuclear_data.model import FissionFragmentData
    from .fission_yields import FPY_KEY, decodeFissionYields

    if mf8 is None:
        return
    productYield, book = decodeFissionYields(mf8, induced, sourcePath, report)
    if productYield is None:
        return
    nuclide.fissionFragmentData = FissionFragmentData(productYields=[productYield])
    provenance.headerFields[FPY_KEY] = book


def decodeFissionYieldSublibrary(endf, report: Optional[ConversionReport] = None,
                                 sourcePath=None) -> Tuple[PoPs, ConversionReport]:
    """A fission yield tape (NSUB=5 spontaneous, 11 neutron-induced) → a PoPs whose
    target nuclide carries the yields, as FUDGE writes a spontaneous one."""
    from .decode import TAPE_ID_KEY, decodeMF1MT451

    report = report if report is not None else ConversionReport()
    mt451 = endf.mf[1].mt[451]
    basePops, style, provenance, report = decodeMF1MT451(mt451, report)
    tapeId = getattr(endf, "tape_id", None)
    if tapeId is not None:
        provenance.headerFields[TAPE_ID_KEY] = tapeId
    pops = PoPs(name="protare_internal", version="1.0", styles=[style], provenance=provenance)
    nuclides = [p for p in basePops.particles.values() if isinstance(p, Nuclide)]
    if len(nuclides) != 1:
        raise ValueError("MF1/451 of a fission yield tape names no target nuclide")
    nuclide = nuclides[0]
    lis = int(getattr(mt451, "_lis", 0) or 0)
    liso = int(getattr(mt451, "_liso", 0) or 0)
    if lis:
        nuclide.id = pidFromZA(nuclide.ZA, lis)
        alias = MetaStable(id=f"{_isotope(nuclide.ZA)}_m{liso}", pid=nuclide.id,
                           metaStableIndex=liso)
        pops.aliases[alias.id] = alias
    if nuclide.halflife is None:
        nuclide.halflife = "unstable"           # FUDGE's ENDF_ITYPE_5
    if nuclide.charge is None:
        nuclide.charge = 0                      # the atom, as FUDGE writes it
    pops.add(nuclide)
    mf8 = endf.mf.get(8)
    if mf8 is None:
        report.lost("this fission yield tape has no MF8")
    else:
        for mt in sorted(getattr(mf8, "mt", {})):
            if mt not in (454, 459):
                report.unsupportedNode(f"MF8/MT{mt} in a fission yield tape is not decoded")
    nsub = int(getattr(mt451, "_nsub", 0) or 0)
    _attachYields(nuclide, mf8, nsub == NFY_NSUB, provenance, sourcePath, report)
    _keepTapeConventions(provenance, mt451, sourcePath)
    _keepUnmodelledSections(provenance, endf, sourcePath, report)
    pops.report = report
    return pops, report


def _controlRecords(path) -> List[str]:
    """Columns 1-66 of every SEND/FEND/MEND/TEND record of the tape, in order."""
    with open(path, encoding="latin-1") as handle:
        lines = [line.rstrip("\r\n") for line in handle]
    return [line[:66].ljust(66) for line in lines[1:] if _isControl(line)]
