"""Phase 10, step 2: the inelastic final-state and level-scheme grammars.

The counterpart of :mod:`kika.g4ndl.parse` for ``Inelastic/``, with the same
contract: each parser consumes a :class:`~kika.g4ndl.tokens.TokenStream` to its
last token, keeps every token the consumer reads
(:mod:`kika.g4ndl.inelastic_records`), and refuses what a misread would
silently corrupt — a token read as an integer that is not one, a count that
does not fit, an interpolation table that leaves points uncovered, a grid that
decreases, a negative cross section, probability or spectrum, a μ outside
[-1, 1], a ``dataType`` or law the consumer does not know. **Physics is not
checked here**, as for the elastic.

``Inelastic/CrossSection`` needs no parser of its own: it is the elastic
cross-section grammar, :func:`kika.g4ndl.parse.parse_cross_section`.

The consumer's sources, Geant4 v11.4.3 ``particle_hp``:

* sections — ``G4ParticleHPInelasticCompFS::Init`` (``src/…CompFS.cc:130-228``)
  and ``G4ParticleHPInelasticBaseFS::Init`` (``src/…BaseFS.cc:84-170``);
* ``dataType=4`` — ``G4ParticleHPAngular::Init``;
* ``dataType=5`` — ``G4ParticleHPEnergyDistribution::Init`` and the six laws
  (``G4ParticleHPArbitaryTab``, ``…EvapSpectrum``, ``…FissionSpectrum``,
  ``…SimpleEvapSpectrum``, ``…WattSpectrum``, ``…MadlandNixSpectrum``);
* ``dataType=6`` — ``G4ParticleHPEnAngCorrelation::Init``,
  ``G4ParticleHPProduct::Init`` and the laws (``…ContEnergyAngular`` with
  ``…ContAngularPar``, ``…DiscreteTwoBody``, ``…NBodyPhaseSpace``,
  ``…LabAngularEnergy``);
* ``dataType=12-15`` — ``G4ParticleHPPhotonDist::InitMean``, ``InitPartials``,
  ``InitAngular``, ``InitEnergies``;
* ``Gammas/`` — ``G4ParticleHPDeExGammas::Init``.

Two places where the consumer's own reading depends on state rather than on
the tokens in front of it, reproduced exactly:

* ``dataType=15`` reads its spectra only if a discrete photon of the same
  reaction (``dataType=12`` or ``13``) has ``disType == 1``. Which reaction is
  "the same" is the slot ``it`` (``sfType % 50`` for ``sfType >= 600`` or
  ``50 <= sfType < 100``, else 50) in a composite file, and the one photon
  object in a base file.
* ``dataType=14`` needs the photon object of that slot to exist already; the
  consumer dereferences a null pointer otherwise, kika refuses.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from kika.g4ndl.inelastic_records import (
    COMPOSITE_CHANNELS, DT_ANGULAR, DT_CROSS_SECTION, DT_ENERGY, DT_ENERGY_ANGLE,
    DT_PHOTON_ANGULAR, DT_PHOTON_ENERGY, DT_PHOTON_MULTIPLICITY, DT_PHOTON_PARTIALS,
    AngularBody, ContinuumBody, ContinuumEnergy, CrossSectionBody, DiscreteTwoBodyBody,
    DiscreteTwoBodyEnergy, EnergyAngleBody, EnergyBody, EnergyLaw, GammasRecord,
    InelasticFSRecord, LabAngleEnergyAngle, LabAngleEnergyBody, LabAngleEnergyEnergy,
    NBodyBody, Pairs, PhotonAngularBody, PhotonAngularLine, PhotonCascadeBody,
    PhotonEnergyBody, PhotonLegendre, PhotonLine, PhotonMultiplicityBody, PhotonPartial,
    PhotonPartialsBody, PhotonSpectrum, PhotonSpectrumAt, PhotonTabulated,
    PhotonTransition, ProductRecord, Section, Tab1,
)
from kika.g4ndl.parse import (
    _block, _check_log_domain, _check_non_decreasing, parse_interpolation,
)
from kika.g4ndl.records import FRAME_LAB, REP_ISOTROPIC, REP_LEGENDRE, REP_TABULATED
from kika.g4ndl.tokens import TokenStream

__all__ = ["parse_inelastic_fs", "parse_gammas", "parseSectionBody", "photonSlot",
           "DATA_TYPES", "ENERGY_LAWS", "PRODUCT_LAWS"]

DATA_TYPES = frozenset({DT_CROSS_SECTION, DT_ANGULAR, DT_ENERGY, DT_ENERGY_ANGLE,
                        DT_PHOTON_MULTIPLICITY, DT_PHOTON_PARTIALS,
                        DT_PHOTON_ANGULAR, DT_PHOTON_ENERGY})

#: ``G4ParticleHPEnergyDistribution::Init``'s switch. Any other value the
#: consumer reads as law 1 (``default:``); kika refuses it instead, since a
#: file that means law 1 says 1.
ENERGY_LAWS = frozenset({1, 5, 7, 9, 11, 12})

#: ``G4ParticleHPProduct::Init``: 0 and 3 isotropic, 1 continuum, 2 discrete
#: two-body, 4 recoil (nothing read), 6 N-body, 7 lab angle-energy. 5 throws.
PRODUCT_LAWS = frozenset({0, 1, 2, 3, 4, 6, 7})

_FRAMES = (1, 2)
#: ``dataType=6`` adds ENDF's LCT=3, centre of mass for light products and
#: laboratory for the rest (``G4ParticleHPEnAngCorrelation.cc:127, 206``).
_FRAMES_MF6 = (1, 2, 3)


def _interpolation(stream, n, what):
    """Inelastic tables carry the C/U variants (``G4NDL_token_spec.md`` §10)."""
    return parse_interpolation(stream, n, what, qualified=True)


def photonSlot(sfType: Optional[int]) -> int:
    """The consumer's ``it`` (``G4ParticleHPInelasticCompFS.cc:177-178``)."""
    if sfType is None:
        return 50
    if sfType >= 600 or 50 <= sfType < 100:
        return sfType % 50
    return 50


# ------------------------------------------------------------------ primitives

def _count(stream: TokenStream, what: str, minimum: int = 0) -> int:
    pos = stream.position
    n = stream.int(what)
    if n < minimum:
        stream._fail(f"count {n} is below {minimum}", what, pos)
    return n


def _frame(stream: TokenStream, what: str, allowed=_FRAMES) -> int:
    pos = stream.position
    f = stream.int(what)
    if f not in allowed:
        stream._fail(f"frameFlag={f}; 1 is laboratory, 2 centre of mass"
                     + (", 3 centre of mass for A <= 4 and laboratory otherwise"
                        if 3 in allowed else ""), what, pos)
    return f


def _pairs(stream: TokenStream, n: int, what: str, *, nonnegative: bool = True,
           grid: str = "x") -> Pairs:
    start = stream.position
    x, y = stream.pairs(n, f"{what}: {n} pairs")
    _check_non_decreasing(stream, x, f"{what}: {grid}", start, 2, grid)
    if nonnegative:
        _nonnegative(stream, y, what, start)
    return Pairs(x, y)


def _nonnegative(stream, y, what, start, stride: int = 2, offset: int = 1):
    neg = np.flatnonzero(y < 0)
    if neg.size:
        k = int(neg[0])
        stream._fail(f"negative value {float(y[k])!r} ({neg.size} negative)",
                     what, start + stride * k + offset)


def _tab1(stream: TokenStream, what: str, *, nonnegative: bool = True,
          minimum: int = 1) -> Tab1:
    """``N``, ``NR (NBT INT)×NR``, ``N`` pairs (``G4ParticleHPVector::Init(stream)``)."""
    n = _count(stream, f"{what}: N", minimum)
    interp = _interpolation(stream, n, f"{what}: interpolation") if n else None
    if interp is None:
        # N = 0 still has an interpolation record in front of no points; the
        # consumer reads it (theManager.Init) whatever N is.
        interp = _emptyInterpolation(stream, what)
    start = stream.position
    pts = _pairs(stream, n, what, nonnegative=nonnegative)
    _check_log_domain(stream, interp, pts.x, pts.y, f"{what}: (x, y)", start)
    return Tab1(interp, pts.x, pts.y)


def _emptyInterpolation(stream, what):
    """An interpolation record governing zero points: ``NR`` and ``NR`` pairs, unchecked
    against ``N`` (there is nothing to cover)."""
    from kika.g4ndl.records import Interpolation
    nr = _count(stream, f"{what}: NR")
    nbt, codes = [], []
    for k in range(nr):
        nbt.append(stream.int(f"{what}: NBT[{k}]"))
        codes.append(stream.int(f"{what}: INT[{k}]"))
    return Interpolation(tuple(nbt), tuple(codes))


# ------------------------------------------------------------------ the file

def parse_inelastic_fs(stream: TokenStream, channel: str) -> InelasticFSRecord:
    """``Inelastic/<channel>/<name>`` — ``channel`` is the directory, ``"F01"`` … ``"F36"``.

    Whether the file is read section by section with ``sfType`` (composite) or
    with one ``Qvalue`` up front (base) is the channel's, not the file's: the
    consumer has no way to tell from the tokens.
    """
    composite = channel in COMPOSITE_CHANNELS
    sections: List[Section] = []
    photons: Dict[int, Optional[object]] = {}
    qvalue = qdummy = None
    while not stream.atEnd():
        k = len(sections) + 1
        tag = f"{channel} section {k}"
        info = stream.int(f"{tag}: infoType")
        pos = stream.position
        dt = stream.int(f"{tag}: dataType")
        if dt not in DATA_TYPES:
            stream._fail(f"dataType={dt}; Geant4 knows {sorted(DATA_TYPES)} and only "
                         f"warns on anything else, leaving the stream misaligned",
                         f"{tag}: dataType", pos)
        sf = dummy = None
        if composite:
            sf = stream.int(f"{tag}: sfType")
            dummy = stream.int(f"{tag}: dummy")
        elif not sections:
            qvalue = stream.float(f"{tag}: Qvalue")
            qdummy = stream.int(f"{tag}: dummy")
        tag = f"{tag} (dataType={dt}" + (f", sfType={sf})" if composite else ")")
        slot = photonSlot(sf) if composite else 50
        body = _body(stream, dt, composite, tag, photons, slot)
        sections.append(Section(info, dt, sf, dummy, body))
    if not sections:
        stream._fail("the file holds no section", f"{channel}: infoType", stream.position)
    return InelasticFSRecord(stream.path, stream.header, channel, composite,
                             tuple(sections), qvalue, qdummy)


def parseSectionBody(text: str, dataType: int, composite: bool,
                     photons: Dict[int, Optional[object]], slot: int, tag: str = "section"):
    """One section's body from its text (:func:`~kika.g4ndl.inelastic_format.formatSectionBody`).

    ``photons`` is the per-slot photon state a ``dataType`` 14 or 15 body
    depends on; a 12 or 13 body updates it. The text must be read to its end.
    """
    stream = TokenStream(text)
    body = _body(stream, dataType, composite, tag, photons, slot)
    stream.expectEnd(f"{tag}: end of body")
    return body


def _body(stream, dt, composite, tag, photons, slot):
    if dt == DT_CROSS_SECTION:
        qi = lr = None
        if composite:
            qi = stream.float(f"{tag}: QI")
            lr = stream.int(f"{tag}: LR")
        n = _count(stream, f"{tag}: N", 1)
        return CrossSectionBody(qi, lr, _pairs(stream, n, f"{tag}: (E, sigma)", grid="energy"))
    if dt == DT_ANGULAR:
        return _angular(stream, tag)
    if dt == DT_ENERGY:
        return _energy(stream, tag)
    if dt == DT_ENERGY_ANGLE:
        return _energyAngle(stream, tag)
    if dt == DT_PHOTON_MULTIPLICITY:
        body = _photonMean(stream, tag)
        photons[slot] = body
        return body
    if dt == DT_PHOTON_PARTIALS:
        body = _photonPartials(stream, tag)
        photons[slot] = body
        return body
    if photons.get(slot) is None:
        stream._fail(f"dataType={dt} before any dataType 12 or 13 of the same reaction; "
                     f"Geant4 dereferences a null photon distribution here",
                     tag, stream.position)
    if dt == DT_PHOTON_ANGULAR:
        return _photonAngular(stream, tag)
    assert dt == DT_PHOTON_ENERGY
    return _photonEnergies(stream, tag, photons[slot])


# ------------------------------------------------------------------ dataType 4

def _angular(stream, tag, *, zeroMassInLab: bool = False) -> AngularBody:
    """``zeroMassInLab`` admits ``targetMass = 0`` in a laboratory-frame body,
    where ``G4ParticleHPAngular`` never uses the mass: four prompt-fission
    MF4s write it so (``G4NDL_token_spec.md`` §12)."""
    pos = stream.position
    rep = stream.int(f"{tag}: repFlag")
    if rep not in (REP_ISOTROPIC, REP_LEGENDRE, REP_TABULATED):
        stream._fail(f"repFlag={rep}; G4ParticleHPAngular knows 0, 1 and 2 and throws "
                     f"on anything else", f"{tag}: repFlag", pos)
    pos = stream.position
    mass = stream.float(f"{tag}: targetMass")
    frame = _frame(stream, f"{tag}: frameFlag")
    if mass < 0 or (mass == 0 and not (zeroMassInLab and frame == FRAME_LAB)):
        stream._fail(f"targetMass={mass!r} is not positive", f"{tag}: targetMass", pos)
    leg = tab = None
    if rep == REP_LEGENDRE:
        leg = _block(stream, tabulated=False, tag=f"{tag} Legendre")
    elif rep == REP_TABULATED:
        tab = _block(stream, tabulated=True, tag=f"{tag} table")
    return AngularBody(rep, mass, frame, leg, tab)


# ------------------------------------------------------------------ dataType 5

def _energy(stream, tag) -> EnergyBody:
    dummy = stream.float(f"{tag}: dummy")
    n = _count(stream, f"{tag}: nPartials", 1)
    partials = []
    for i in range(n):
        what = f"{tag}, partial {i + 1} of {n}"
        pos = stream.position
        law = stream.int(f"{what}: law")
        if law not in ENERGY_LAWS:
            stream._fail(f"energy law {law}; Geant4 reads anything but "
                         f"{sorted(ENERGY_LAWS)} as law 1, kika refuses it",
                         f"{what}: law", pos)
        prob = _tab1(stream, f"{what}: p(E)")
        if law == 1:
            nd = _count(stream, f"{what}: nDistFunc")
            interp = _interpolation(stream, max(nd, 1), f"{what}: energy interpolation")
            spectra = []
            start = stream.position
            for j in range(max(nd, 1)):
                e = stream.float(f"{what}, energy {j + 1}: E")
                spectra.append((e, _tab1(stream, f"{what}, energy {j + 1}: g(E')")))
            _incidentOrder(stream, [e for e, _ in spectra], what, start)
            partials.append(EnergyLaw(law, prob, nDistFunc=nd, interpolation=interp,
                                      spectra=tuple(spectra)))
        elif law == 12:
            efl = stream.float(f"{what}: EFL")
            efh = stream.float(f"{what}: EFH")
            tm = _tab1(stream, f"{what}: T_M(E)")
            partials.append(EnergyLaw(law, prob, parameters=(tm,), scalars=(efl, efh)))
        else:
            names = {5: ("theta(E)", "g(x)"), 7: ("theta(E)",), 9: ("theta(E)",),
                     11: ("a(E)", "b(E)")}[law]
            params = tuple(_tab1(stream, f"{what}: {name}") for name in names)
            partials.append(EnergyLaw(law, prob, parameters=params))
    return EnergyBody(dummy, tuple(partials))


def _incidentOrder(stream, energies, what, start):
    e = np.asarray(energies, dtype=np.float64)
    bad = np.flatnonzero(np.diff(e) < 0)
    if bad.size:
        k = int(bad[0]) + 1
        stream._fail(f"incident energy decreases at record {k + 1}: {float(e[k - 1])!r} -> "
                     f"{float(e[k])!r}", f"{what}: E", start)


# ------------------------------------------------------------------ dataType 6

def _energyAngle(stream, tag) -> EnergyAngleBody:
    pos = stream.position
    mass = stream.float(f"{tag}: targetMass")
    if mass <= 0:
        stream._fail(f"targetMass={mass!r} is not positive", f"{tag}: targetMass", pos)
    frame = _frame(stream, f"{tag}: frameFlag", _FRAMES_MF6)
    n = _count(stream, f"{tag}: nProducts")
    products = tuple(_product(stream, f"{tag}, product {i + 1}") for i in range(max(n, 1)))
    return EnergyAngleBody(mass, frame, n, products)


def _product(stream, what) -> ProductRecord:
    zap = stream.float(f"{what}: massCode")
    awp = stream.float(f"{what}: mass")
    lip = stream.int(f"{what}: isomerFlag")
    pos = stream.position
    law = stream.int(f"{what}: distLaw")
    if law not in PRODUCT_LAWS:
        stream._fail(f"distLaw={law}; G4ParticleHPProduct knows {sorted(PRODUCT_LAWS)} "
                     f"and throws on anything else", f"{what}: distLaw", pos)
    q0 = stream.float(f"{what}: groundStateQ")
    q1 = stream.float(f"{what}: actualStateQ")
    yld = _tab1(stream, f"{what}: yield", nonnegative=False)
    tag = f"{what} (law {law})"
    body = None
    if law == 1:
        body = _continuum(stream, tag)
    elif law == 2:
        body = _discreteTwoBody(stream, tag)
    elif law == 6:
        body = NBodyBody(stream.float(f"{tag}: totalMass"),
                         _count(stream, f"{tag}: totalCount"))
    elif law == 7:
        body = _labAngleEnergy(stream, tag)
    return ProductRecord(zap, awp, lip, law, q0, q1, yld, body)


def _continuum(stream, tag) -> ContinuumBody:
    code = stream.float(f"{tag}: targetCode")
    lang = stream.int(f"{tag}: angularRep")
    lep = stream.int(f"{tag}: secondary interpolation")
    ne = _count(stream, f"{tag}: nEnergy")
    interp = _interpolation(stream, ne, f"{tag}: energy interpolation") if ne \
        else _emptyInterpolation(stream, f"{tag}: energy interpolation")
    start = stream.position
    energies = []
    for i in range(ne):
        what = f"{tag}, energy {i + 1} of {ne}"
        e = stream.float(f"{what}: E")
        nep = _count(stream, f"{what}: nEnergies")
        nd = _count(stream, f"{what}: nDiscrete")
        na = _count(stream, f"{what}: nAngularParameters")
        if nd > nep:
            stream._fail(f"{nd} discrete energies out of {nep}", f"{what}: nDiscrete",
                         stream.position - 2)
        rows = stream.floats(nep * (1 + na), f"{what}: {nep} rows").reshape(nep, 1 + na)
        energies.append(ContinuumEnergy(e, nd, na, rows))
    _incidentOrder(stream, [x.energy for x in energies], tag, start)
    return ContinuumBody(code, lang, lep, interp, tuple(energies))


def _discreteTwoBody(stream, tag) -> DiscreteTwoBodyBody:
    ne = _count(stream, f"{tag}: nEnergy")
    interp = _interpolation(stream, max(ne, 1), f"{tag}: energy interpolation")
    start = stream.position
    energies = []
    for i in range(max(ne, 1)):
        what = f"{tag}, energy {i + 1}"
        e = stream.float(f"{what}: E")
        rep = stream.int(f"{what}: representation")
        nc = _count(stream, f"{what}: nCoefficients")
        npts = nc * 2 if rep > 0 else nc
        vals = stream.floats(npts, f"{what}: {npts} values")
        if rep > 0:
            mu, p = vals[0::2], vals[1::2]
            if np.any((mu < -1) | (mu > 1)):
                stream._fail("mu outside [-1, 1]", f"{what}: mu", start)
            if np.any(p < 0):
                stream._fail("negative probability density", f"{what}: p", start)
        energies.append(DiscreteTwoBodyEnergy(e, rep, nc, vals))
    _incidentOrder(stream, [x.energy for x in energies], tag, start)
    return DiscreteTwoBodyBody(ne, interp, tuple(energies))


def _labAngleEnergy(stream, tag) -> LabAngleEnergyBody:
    ne = _count(stream, f"{tag}: nEnergies")
    interp = _interpolation(stream, ne, f"{tag}: energy interpolation") if ne \
        else _emptyInterpolation(stream, f"{tag}: energy interpolation")
    energies = []
    for i in range(ne):
        what = f"{tag}, energy {i + 1} of {ne}"
        e = stream.float(f"{what}: E")
        nmu = _count(stream, f"{what}: nCosTh")
        mi = _interpolation(stream, max(nmu, 1), f"{what}: mu interpolation")
        angles = []
        for j in range(max(nmu, 1)):
            pos = stream.position
            mu = stream.float(f"{what}, mu {j + 1}: mu")
            if not -1 <= mu <= 1:
                stream._fail(f"mu={mu!r} outside [-1, 1]", f"{what}, mu {j + 1}", pos)
            angles.append(LabAngleEnergyAngle(mu, _tab1(stream, f"{what}, mu {j + 1}: p(E')")))
        energies.append(LabAngleEnergyEnergy(e, nmu, mi, tuple(angles)))
    return LabAngleEnergyBody(interp, tuple(energies))


# ------------------------------------------------------------------ photons

def _photonMean(stream, tag, *, nonnegative: bool = True):
    """``nonnegative=False`` admits a negative multiplicity, which Geant4 reads as
    written: Pa-231's fission photons carry two of -7e-9 (spec §12)."""
    pos = stream.position
    rep = stream.int(f"{tag}: repFlag")
    mass = stream.float(f"{tag}: targetMass")
    if rep == 1:
        n = _count(stream, f"{tag}: nDiscrete")
        lines = []
        for i in range(max(n, 1)):
            what = f"{tag}, photon {i + 1}"
            dis = stream.int(f"{what}: disType")
            e = stream.float(f"{what}: energy")
            lines.append(PhotonLine(dis, e, _tab1(stream, f"{what}: multiplicity",
                                                  nonnegative=nonnegative)))
        return PhotonMultiplicityBody(rep, mass, n, tuple(lines))
    if rep == 2:
        ic = stream.int(f"{tag}: conversion flag")
        base = stream.float(f"{tag}: base energy")
        pos2 = stream.position
        ic2 = stream.int(f"{tag}: conversion flag (again)")
        if ic2 not in (1, 2):
            stream._fail(f"conversion flag {ic2}; Geant4 throws on anything but 1 or 2",
                         f"{tag}: conversion flag", pos2)
        n = _count(stream, f"{tag}: nGammaEnergies")
        trans = []
        for i in range(max(n, 1)):
            what = f"{tag}, transition {i + 1}"
            lev = stream.float(f"{what}: level energy")
            pr = stream.float(f"{what}: probability")
            fr = stream.float(f"{what}: photon fraction") if ic2 == 2 else None
            trans.append(PhotonTransition(lev, pr, fr))
        return PhotonCascadeBody(rep, mass, ic, base, ic2, n, tuple(trans))
    stream._fail(f"photon repFlag={rep}; Geant4 knows 1 and 2 and throws otherwise",
                 f"{tag}: repFlag", pos)


def _photonPartials(stream, tag) -> PhotonPartialsBody:
    n = _count(stream, f"{tag}: nDiscrete")
    mass = stream.float(f"{tag}: targetMass")
    total = _tab1(stream, f"{tag}: total sigma") if n != 1 else None
    partials = []
    for i in range(max(n, 1)):
        what = f"{tag}, photon {i + 1}"
        e = stream.float(f"{what}: energy")
        es = stream.float(f"{what}: shell")
        lp = stream.int(f"{what}: isPrimary")
        lf = stream.int(f"{what}: disType")
        partials.append(PhotonPartial(e, es, lp, lf, _tab1(stream, f"{what}: sigma")))
    return PhotonPartialsBody(n, mass, total, tuple(partials))


def _photonAngular(stream, tag) -> PhotonAngularBody:
    iso = stream.int(f"{tag}: isoFlag")
    if iso == 1:
        return PhotonAngularBody(iso)
    pos = stream.position
    ltt = stream.int(f"{tag}: tabulationType")
    if ltt not in (1, 2):
        stream._fail(f"tabulationType={ltt}; Geant4 knows 1 and 2 and throws otherwise",
                     f"{tag}: tabulationType", pos)
    nd2 = _count(stream, f"{tag}: nDiscrete2")
    niso = _count(stream, f"{tag}: nIso")
    if niso > nd2:
        stream._fail(f"nIso={niso} > nDiscrete2={nd2}", f"{tag}: nIso", stream.position - 1)
    iso_lines = tuple((stream.float(f"{tag}, isotropic {i + 1}: energy"),
                       stream.float(f"{tag}, isotropic {i + 1}: shell")) for i in range(niso))
    lines = []
    for i in range(niso, nd2):
        what = f"{tag}, photon {i + 1}"
        e = stream.float(f"{what}: energy")
        es = stream.float(f"{what}: shell")
        nn = _count(stream, f"{what}: nNeu")
        if ltt == 1:
            interp = _interpolation(stream, nn, f"{what}: energy interpolation") if nn \
                else _emptyInterpolation(stream, f"{what}: energy interpolation")
            recs = []
            for j in range(nn):
                w = f"{what}, energy {j + 1}"
                en = stream.float(f"{w}: E")
                npoly = _count(stream, f"{w}: nPoly")
                recs.append(PhotonLegendre(en, stream.floats(npoly, f"{w}: coefficients")))
        else:
            interp = None
            recs = []
            for j in range(nn):
                w = f"{what}, energy {j + 1}"
                en = stream.float(f"{w}: E")
                npr = _count(stream, f"{w}: nProb", 1)
                mi = _interpolation(stream, npr, f"{w}: mu interpolation")
                start = stream.position
                mu, p = stream.pairs(npr, f"{w}: (mu, p)")
                if np.any((mu < -1) | (mu > 1)):
                    stream._fail("mu outside [-1, 1]", f"{w}: mu", start)
                _check_non_decreasing(stream, mu, f"{w}: mu", start, 2, "mu")
                _nonnegative(stream, p, f"{w}: p", start)
                recs.append(PhotonTabulated(en, mi, mu, p))
        _incidentOrder(stream, [r.energy for r in recs], what, stream.position)
        lines.append(PhotonAngularLine(e, es, nn, interp, tuple(recs)))
    return PhotonAngularBody(iso, ltt, nd2, niso, iso_lines, tuple(lines))


def _energiesNeeded(photon) -> bool:
    """``InitEnergies``: ``disType[i] == 1`` for some ``i < nDiscrete``.

    ``nDiscrete`` is the slot's: what ``InitMean`` (repFlag=1) or
    ``InitPartials`` set. A cascade (``InitMean`` repFlag=2) leaves it 0.
    """
    if isinstance(photon, PhotonMultiplicityBody):
        return any(line.disType == 1 for line in photon.lines[:photon.nDiscrete])
    if isinstance(photon, PhotonPartialsBody):
        return any(p.disType == 1 for p in photon.partials[:photon.nDiscrete])
    return False


def _photonEnergies(stream, tag, photon) -> PhotonEnergyBody:
    if not _energiesNeeded(photon):
        return PhotonEnergyBody(False)
    n = _count(stream, f"{tag}: nPartials")
    spectra = []
    for i in range(n):
        what = f"{tag}, partial {i + 1} of {n}"
        dummy = stream.int(f"{what}: dummy")
        prob = _tab1(stream, f"{what}: p(E)")
        nen = _count(stream, f"{what}: nen", 1)
        interp = _interpolation(stream, nen, f"{what}: energy interpolation")
        start = stream.position
        at = []
        for j in range(nen):
            e = stream.float(f"{what}, energy {j + 1}: E")
            at.append(PhotonSpectrumAt(e, _tab1(stream, f"{what}, energy {j + 1}: g(E')")))
        _incidentOrder(stream, [a.energy for a in at], what, start)
        spectra.append(PhotonSpectrum(dummy, prob, interp, tuple(at)))
    return PhotonEnergyBody(True, tuple(spectra))


# ------------------------------------------------------------------ Gammas/

def parse_gammas(stream: TokenStream, Z: int, A: int) -> GammasRecord:
    """``Inelastic/Gammas/z<Z>.a<A>``: ``(E_level, E_gamma, probability)`` to the end.

    The consumer reads triples while ``>> elevel`` succeeds; a file whose token
    count is not a multiple of three would leave its last level half read, so
    kika refuses it. Nothing else is checked: real files list levels out of order
    (134 of G4NDL 4.7.1's 1 643) and the consumer copes, searching every level
    read so far for the one a gamma feeds.
    """
    if stream.header is not None:
        stream._fail("a Gammas file is read with std::ifstream: a 'G4NDL' header "
                     "would be read as data", "Gammas: header", 0)
    n, rest = divmod(stream.remaining, 3)
    if rest:
        stream._fail(f"{stream.remaining} tokens is not a whole number of "
                     f"(level, gamma, probability) triples", "Gammas", stream.position)
    flat = stream.floats(3 * n, f"Gammas: {n} triples")
    lev, gam, pr = flat[0::3].copy(), flat[1::3].copy(), flat[2::3].copy()
    return GammasRecord(stream.path, Z, A, lev, gam, pr)
