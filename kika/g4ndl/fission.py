"""G4NDL ``Fission/``: the records, their reader and their writer's text.

The fourth process after the elastic, the inelastic and the capture. What
Geant4 reads (``G4ParticleHPFission::BuildPhysicsTable`` registers one
``G4ParticleHPFissionFS`` per element on ``<G4NEUTRONHPDATA>/Fission``):

``Fission/CrossSection/<name>``
    σ of the whole process, MT18: the elastic cross-section grammar
    (``G4ParticleHPIsoData::Init``), bookkeeping ``0 0`` in both libraries.
``Fission/FS/<name>``
    ``G4ParticleHPFSFissionFS::Init`` (``src/G4ParticleHPFSFissionFS.cc``): a
    sequence of ``infoType dataType`` sections to the end of the stream.
    ``infoType`` says what the section is about, ``dataType`` is an ENDF file
    number, and the pair picks the reader:

    ====  ====  ==========================================  =================
    info  data  consumer                                    ENDF
    ====  ====  ==========================================  =================
    1     4     ``G4ParticleHPAngular::Init``               MF4/MT18
    1     5     ``G4ParticleHPEnergyDistribution::Init``    MF5/MT18 (prompt)
    1     12    ``G4ParticleHPPhotonDist::InitMean``        MF12/MT18
    1     14    ``…PhotonDist::InitAngular``                MF14/MT18
    1     15    ``…PhotonDist::InitEnergies``               MF15/MT18
    2     1     ``G4ParticleHPParticleYield::InitMean``     MF1/452
    3     1     ``…ParticleYield::InitDelayed``             MF1/455
    3     5     ``G4ParticleHPEnergyDistribution::Init``    MF5/MT455
    4     1     ``…ParticleYield::InitPrompt``              MF1/456
    5     1     ``G4ParticleHPFissionERelease::Init``       MF1/458
    ====  ====  ==========================================  =================

    Any other ``infoType`` throws in the consumer; any other ``dataType`` under
    a known one is read by nobody, and the stream goes on misaligned. kika
    refuses both.
``Fission/FC``, ``SC``, ``TC``, ``LC`` ``/<name>``
    first-, second-, third- and fourth-chance fission (MT19, 20, 21, 38),
    ``G4ParticleHPFissionBaseFS::Init``: ``b0 b1 N (E σ)×N`` — the
    cross-section grammar, ``b0`` the chance's Q in eV and ``b1`` 0 — and,
    optionally, ``i d`` + an MF4 body + ``i d`` + an MF5 body for its neutrons.
    The two pairs ``i d`` are read into a dummy; both libraries write ``1 4``
    and ``3 5``. A chance file with no final state makes Geant4 sample the
    chance's neutrons from ``FS``.
``Fission/FF/<name>``
    ``G4ParticleHPFFFissionFS::Init``: fission-product yields (ENDF MF8/454 and
    459), blocks of ``MT MF dummy imax`` and ``imax + 1`` energies, each ``E
    jmax ip`` and ``jmax`` triples ``ZA·1000+… isomer yield`` (the ``FSP``
    ``mFSP`` ``Y`` of the consumer). Only G4NDL 4.7.1 has it, and Geant4 reads
    it only with ``G4ParticleHPManager::SetProduceFissionFragments(true)``.

Every parser reads its file to the last token and keeps every token the
consumer reads; the ``G4NDL <source>`` header is kept where a file has one
(``FF`` does). The census of both libraries is in ``G4NDL_token_spec.md`` §12.

Like the rest of :mod:`kika.g4ndl`, nothing here imports the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from kika.g4ndl.inelastic_records import (
    DT_ANGULAR, DT_ENERGY, DT_PHOTON_ANGULAR, DT_PHOTON_ENERGY, DT_PHOTON_MULTIPLICITY,
    AngularBody, EnergyBody, PhotonAngularBody, PhotonCascadeBody, PhotonEnergyBody,
    PhotonMultiplicityBody, Tab1,
)
from kika.g4ndl.records import CrossSectionRecord
from kika.g4ndl.tokens import TokenStream

__all__ = [
    "FISSION_MT", "CHANCE_MT", "CHANCES", "CROSS_SECTION_DIR", "FS_DIR", "FF_DIR",
    "CHANCE_DIRS", "FISSION_DIRS", "SECTION_TYPES",
    "INFO_NEUTRONS", "INFO_NU_TOTAL", "INFO_DELAYED", "INFO_NU_PROMPT", "INFO_ENERGY_RELEASE",
    "DT_YIELD", "NuTotalBody", "NuPromptBody", "NuDelayedBody", "EnergyReleaseBody",
    "FissionSection", "FissionFSRecord", "ChanceFissionRecord", "FragmentYieldsAt",
    "FragmentYieldBlock", "FragmentYieldsRecord",
    "parse_fission_fs", "parse_chance_fission", "parse_fragment_yields",
    "parseFissionSectionBody", "formatFissionFS", "formatFissionSectionBody",
    "formatChanceFission", "formatFragmentYields", "fissionDifferences",
]

FISSION_MT = 18
#: The chance directories, in the order ``G4ParticleHPFissionFS::ApplyYourself``
#: stacks their σ, and the MT each is.
CHANCE_MT: Dict[str, int] = {"FC": 19, "SC": 20, "TC": 21, "LC": 38}
CHANCES = tuple(CHANCE_MT)
CROSS_SECTION_DIR = "Fission/CrossSection"
FS_DIR = "Fission/FS"
FF_DIR = "Fission/FF"
CHANCE_DIRS = tuple(f"Fission/{c}" for c in CHANCES)
FISSION_DIRS = (CROSS_SECTION_DIR, FS_DIR) + CHANCE_DIRS + (FF_DIR,)

INFO_NEUTRONS, INFO_NU_TOTAL, INFO_DELAYED, INFO_NU_PROMPT, INFO_ENERGY_RELEASE = 1, 2, 3, 4, 5
#: ``dataType`` 1 under infoType 2, 3 and 4 (MF1).
DT_YIELD = 1
#: ``infoType`` -> the ``dataType`` values its ``case`` reads.
SECTION_TYPES: Dict[int, Tuple[int, ...]] = {
    INFO_NEUTRONS: (DT_ANGULAR, DT_ENERGY, DT_PHOTON_MULTIPLICITY, DT_PHOTON_ANGULAR,
                    DT_PHOTON_ENERGY),
    INFO_NU_TOTAL: (DT_YIELD,),
    INFO_DELAYED: (DT_YIELD, DT_ENERGY),
    INFO_NU_PROMPT: (DT_YIELD,),
    INFO_ENERGY_RELEASE: (DT_YIELD,),
}

#: ``G4ParticleHPFissionERelease::Init`` reads a dummy and nine terms, in MF1/458's
#: order (ENDF's EFR, ENP, END, EGP, EGD, EB, ENU, ER, ET).
N_ENERGY_RELEASE = 10


# ------------------------------------------------------------------ records

@dataclass(frozen=True, eq=False)
class NuTotalBody:
    """infoType 2: ``targetMass iflag``, then ν̄_total (MF1/452).

    ``iflag == 1`` is a polynomial (``nPoly`` and its coefficients, ENDF LNU=1);
    anything else is a ``Tab1`` to the consumer, and kika accepts only 2 (LNU=2).
    ``targetMass`` is the mass ``G4ParticleHPFissionFS`` uses for the target.
    """

    targetMass: float
    iflag: int
    coefficients: Optional[np.ndarray] = None
    table: Optional[Tab1] = None


@dataclass(frozen=True, eq=False)
class NuPromptBody:
    """infoType 4: ``targetMass iflag``, then ν̄_prompt (MF1/456).

    ``iflag == 2`` is a ``Tab1``; anything else is **one** value read as a
    constant (``theSpontPrompt``), and kika accepts only 1 for that.
    """

    targetMass: float
    iflag: int
    value: Optional[float] = None
    table: Optional[Tab1] = None


@dataclass(frozen=True, eq=False)
class NuDelayedBody:
    """infoType 3, dataType 1: ``targetMass iflag``, the precursor decay constants
    (``N λ_1 … λ_N``, 1/s) and ν̄_delayed (MF1/455): a ``Tab1`` when ``iflag == 2``,
    one constant otherwise (kika accepts 1)."""

    targetMass: float
    iflag: int
    decayConstants: np.ndarray
    value: Optional[float] = None
    table: Optional[Tab1] = None


@dataclass(frozen=True, eq=False)
class EnergyReleaseBody:
    """infoType 5: a dummy and the nine MF1/458 terms, in eV (``values[1:]``).

    The consumer keeps one number per term: the constant of an LFC=0 NPLY=0
    MF1/458, with no uncertainty and no energy dependence.
    """

    values: np.ndarray


FissionBody = Union[NuTotalBody, NuPromptBody, NuDelayedBody, EnergyReleaseBody, AngularBody,
                    EnergyBody, PhotonMultiplicityBody, PhotonCascadeBody, PhotonAngularBody,
                    PhotonEnergyBody]


@dataclass(frozen=True, eq=False)
class FissionSection:
    infoType: int
    dataType: int
    body: FissionBody


@dataclass(frozen=True, eq=False)
class FissionFSRecord:
    """``Fission/FS/<name>``: its sections in file order."""

    path: Optional[Path]
    header: Optional[Tuple[str, str]]
    sections: Tuple[FissionSection, ...]

    def section(self, infoType: int, dataType: int) -> Optional[FissionSection]:
        """The last section of this type: what the consumer keeps when one repeats."""
        found = [s for s in self.sections if (s.infoType, s.dataType) == (infoType, dataType)]
        return found[-1] if found else None


@dataclass(frozen=True, eq=False)
class ChanceFissionRecord:
    """``Fission/FC`` … ``LC``: one chance's σ and, when present, its neutrons.

    ``crossSection`` is the cross-section grammar (``bookkeeping[0]`` the
    chance's Q in eV); ``angularHead``/``energyHead`` are the two ``i d``
    pairs the consumer reads into a dummy.
    """

    path: Optional[Path]
    chance: str
    crossSection: CrossSectionRecord
    angularHead: Optional[Tuple[int, int]] = None
    angular: Optional[AngularBody] = None
    energyHead: Optional[Tuple[int, int]] = None
    energy: Optional[EnergyBody] = None

    @property
    def header(self) -> Optional[Tuple[str, str]]:
        return self.crossSection.header

    @property
    def hasFinalState(self) -> bool:
        return self.angular is not None


@dataclass(frozen=True, eq=False)
class FragmentYieldsAt:
    """One incident energy of an ``FF`` block: ``E jmax ip`` and ``jmax`` triples.

    ``fsp`` is ``Z*1000 + A`` (the consumer's ``FSP``, ENDF's ZAFP), ``isomer``
    the ``mFSP`` (ENDF's FPS), ``yields`` Y; ``interpolation`` is ``ip`` (ENDF's
    I of the LIST, the interpolation to the next energy).
    """

    energy: float
    interpolation: int
    fsp: np.ndarray
    isomer: np.ndarray
    yields: np.ndarray

    def __len__(self) -> int:
        return len(self.fsp)


@dataclass(frozen=True, eq=False)
class FragmentYieldBlock:
    """``MT MF dummy imax`` and ``imax + 1`` energies (MF8/454 or 459)."""

    MT: int
    MF: int
    dummy: float
    imax: int
    energies: Tuple[FragmentYieldsAt, ...]


@dataclass(frozen=True, eq=False)
class FragmentYieldsRecord:
    """``Fission/FF/<name>``: its blocks in file order."""

    path: Optional[Path]
    header: Optional[Tuple[str, str]]
    blocks: Tuple[FragmentYieldBlock, ...]


# ------------------------------------------------------------------ parse

def _yieldHead(stream, tag) -> Tuple[float, int]:
    pos = stream.position
    mass = stream.float(f"{tag}: targetMass")
    if mass <= 0:
        stream._fail(f"targetMass={mass!r} is not positive", f"{tag}: targetMass", pos)
    return mass, stream.int(f"{tag}: iflag")


def _flag(stream, flag, allowed, tag, meaning):
    if flag not in allowed:
        stream._fail(f"iflag={flag}; {meaning}", f"{tag}: iflag", stream.position - 1)


def _nuTotal(stream, tag) -> NuTotalBody:
    from kika.g4ndl.inelastic_parse import _count, _tab1

    mass, flag = _yieldHead(stream, tag)
    _flag(stream, flag, (1, 2), tag, "1 is a polynomial and 2 a table (ENDF LNU); Geant4 "
          "reads anything but 1 as a table, kika refuses it")
    if flag == 1:
        n = _count(stream, f"{tag}: nPoly", 1)
        return NuTotalBody(mass, flag, coefficients=stream.floats(n, f"{tag}: coefficients"))
    return NuTotalBody(mass, flag, table=_tab1(stream, f"{tag}: nu(E)"))


def _constantOrTable(stream, tag, flag):
    from kika.g4ndl.inelastic_parse import _tab1

    _flag(stream, flag, (1, 2), tag, "2 is a table and anything else one constant to "
          "Geant4; kika accepts 1 for the constant")
    if flag == 2:
        return None, _tab1(stream, f"{tag}: nu(E)")
    pos = stream.position
    value = stream.float(f"{tag}: nu")
    if value < 0:
        stream._fail(f"negative multiplicity {value!r}", f"{tag}: nu", pos)
    return value, None


def _nuPrompt(stream, tag) -> NuPromptBody:
    mass, flag = _yieldHead(stream, tag)
    value, table = _constantOrTable(stream, tag, flag)
    return NuPromptBody(mass, flag, value, table)


def _nuDelayed(stream, tag) -> NuDelayedBody:
    from kika.g4ndl.inelastic_parse import _count

    mass, flag = _yieldHead(stream, tag)
    n = _count(stream, f"{tag}: number of decay constants", 1)
    start = stream.position
    rates = stream.floats(n, f"{tag}: decay constants")
    # Zero is admitted: Cm-240 and Bk-247 state one family with λ = 0, and the
    # consumer then emits its delayed neutrons at t = 0 with a warning.
    bad = np.flatnonzero(rates < 0)
    if bad.size:
        stream._fail(f"decay constant {float(rates[bad[0]])!r} is negative",
                     f"{tag}: decay constants", start + int(bad[0]))
    value, table = _constantOrTable(stream, tag, flag)
    return NuDelayedBody(mass, flag, rates, value, table)


def _energyRelease(stream, tag) -> EnergyReleaseBody:
    return EnergyReleaseBody(stream.floats(N_ENERGY_RELEASE, f"{tag}: dummy and nine terms"))


def _fsBody(stream, info, dt, tag, photons):
    from kika.g4ndl.inelastic_parse import (
        _angular, _energy, _photonAngular, _photonEnergies, _photonMean,
    )

    if info == INFO_NEUTRONS:
        if dt == DT_ANGULAR:
            return _angular(stream, tag, zeroMassInLab=True)
        if dt == DT_ENERGY:
            return _energy(stream, tag)
        if dt == DT_PHOTON_MULTIPLICITY:
            photons["mean"] = _photonMean(stream, tag, nonnegative=False)
            return photons["mean"]
        if photons.get("mean") is None:
            stream._fail(f"dataType={dt} before the dataType 12 it depends on; "
                         f"InitEnergies reads the disType of the photons InitMean read",
                         tag, stream.position)
        if dt == DT_PHOTON_ANGULAR:
            return _photonAngular(stream, tag)
        return _photonEnergies(stream, tag, photons["mean"])
    if info == INFO_DELAYED and dt == DT_ENERGY:
        return _energy(stream, tag)
    return {INFO_NU_TOTAL: _nuTotal, INFO_DELAYED: _nuDelayed, INFO_NU_PROMPT: _nuPrompt,
            INFO_ENERGY_RELEASE: _energyRelease}[info](stream, tag)


def _sectionType(stream, tag) -> Tuple[int, int]:
    pos = stream.position
    info = stream.int(f"{tag}: infoType")
    if info not in SECTION_TYPES:
        stream._fail(f"infoType={info}; G4ParticleHPFSFissionFS knows "
                     f"{sorted(SECTION_TYPES)} and throws on anything else",
                     f"{tag}: infoType", pos)
    pos = stream.position
    dt = stream.int(f"{tag}: dataType")
    if dt not in SECTION_TYPES[info]:
        stream._fail(f"dataType={dt} under infoType={info}; the consumer reads only "
                     f"{list(SECTION_TYPES[info])} there and would go on misaligned",
                     f"{tag}: dataType", pos)
    return info, dt


def parse_fission_fs(stream: TokenStream) -> FissionFSRecord:
    """``Fission/FS/<name>``, read to its last token.

    An empty file is refused: the consumer would have no ν̄ and no spectrum,
    and fission would emit no neutron at all.
    """
    sections: List[FissionSection] = []
    photons: Dict[str, object] = {}
    while not stream.atEnd():
        tag = f"Fission/FS section {len(sections) + 1}"
        info, dt = _sectionType(stream, tag)
        tag = f"{tag} (infoType={info}, dataType={dt})"
        sections.append(FissionSection(info, dt, _fsBody(stream, info, dt, tag, photons)))
    if not sections:
        stream._fail("the file holds no section", "Fission/FS: infoType", stream.position)
    return FissionFSRecord(stream.path, stream.header, tuple(sections))


def parseFissionSectionBody(text: str, infoType: int, dataType: int,
                            photons: Optional[Dict[str, object]] = None,
                            tag: str = "Fission/FS section"):
    """One ``Fission/FS`` section body from its text; ``photons`` is the 12 a 14 or 15
    depends on (``{"mean": body}``), and a 12 updates it."""
    stream = TokenStream(text)
    body = _fsBody(stream, infoType, dataType, tag, photons if photons is not None else {})
    stream.expectEnd(f"{tag}: end of body")
    return body


def parse_chance_fission(stream: TokenStream, chance: str) -> ChanceFissionRecord:
    """``Fission/<chance>/<name>``, ``chance`` one of ``FC SC TC LC``, read to its last token."""
    from kika.g4ndl.inelastic_parse import _angular, _energy
    from kika.g4ndl.parse import parse_cross_section

    if chance not in CHANCE_MT:
        raise ValueError(f"chance {chance!r} is not one of {CHANCES}")
    cs = parse_cross_section(stream, toEnd=False, nonnegative=False)
    if stream.atEnd():
        return ChanceFissionRecord(stream.path, chance, cs)
    tag = f"Fission/{chance}"
    ahead = (stream.int(f"{tag}: dummy"), stream.int(f"{tag}: dummy"))
    angular = _angular(stream, f"{tag} (MF4)", zeroMassInLab=True)
    ehead = (stream.int(f"{tag}: dummy"), stream.int(f"{tag}: dummy"))
    energy = _energy(stream, f"{tag} (MF5)")
    stream.expectEnd(f"{tag}: end of file")
    return ChanceFissionRecord(stream.path, chance, cs, ahead, angular, ehead, energy)


def parse_fragment_yields(stream: TokenStream) -> FragmentYieldsRecord:
    """``Fission/FF/<name>``, read to its last token.

    The consumer loops ``while (theData.good())``, so whitespace after the last
    block starts one more read that fails; kika stops at the last token.
    """
    from kika.g4ndl.inelastic_parse import _count

    blocks: List[FragmentYieldBlock] = []
    while not stream.atEnd():
        tag = f"Fission/FF block {len(blocks) + 1}"
        mt = stream.int(f"{tag}: MT")
        mf = stream.int(f"{tag}: MF")
        dummy = stream.float(f"{tag}: dummy")
        imax = _count(stream, f"{tag}: imax")
        energies = []
        for i in range(imax + 1):
            what = f"{tag}, energy {i + 1}"
            e = stream.float(f"{what}: E")
            n = _count(stream, f"{what}: jmax")
            ip = stream.int(f"{what}: interpolation")
            start = stream.position
            flat = stream.floats(3 * n, f"{what}: (FSP, mFSP, Y) triples")
            # FSP and mFSP are read with >> G4int: a token like 23066.0 would
            # leave ".0" in the stream, so the tokens themselves are checked.
            for k in range(n):
                for c in (0, 1):
                    tok = stream._tokens[start + 3 * k + c]
                    if not tok.lstrip("+-").isdigit():
                        stream._fail(f"expected an integer, found {tok!r}",
                                     f"{what}: triple {k + 1}", start + 3 * k + c)
            y = flat[2::3].copy()
            if np.any(y < 0):
                k = int(np.flatnonzero(y < 0)[0])
                stream._fail(f"negative yield {float(y[k])!r}", f"{what}: triple {k + 1}",
                             start + 3 * k + 2)
            energies.append(FragmentYieldsAt(e, ip, flat[0::3].astype(np.int64),
                                             flat[1::3].astype(np.int64), y))
        blocks.append(FragmentYieldBlock(mt, mf, dummy, imax, tuple(energies)))
    if not blocks:
        stream._fail("the file holds no block", "Fission/FF: MT", stream.position)
    return FragmentYieldsRecord(stream.path, stream.header, tuple(blocks))


# ------------------------------------------------------------------ format

def _headerLines(header) -> List[str]:
    return [f"{header[0]} {header[1]}"] if header is not None else []


def _yieldLines(body) -> List[str]:
    from kika.g4ndl.inelastic_format import _f, _i, _tab1, _values

    out = [f"{_f(body.targetMass)} {_i(body.iflag)}"]
    if isinstance(body, NuTotalBody):
        if body.iflag == 1:
            return out + [_i(len(body.coefficients))] + _values(body.coefficients)
        return out + _tab1(body.table)
    if isinstance(body, NuDelayedBody):
        out += [_i(len(body.decayConstants))] + _values(body.decayConstants)
    return out + (_tab1(body.table) if body.table is not None else [_f(body.value)])


def _bodyLines(info: int, dt: int, body) -> List[str]:
    from kika.g4ndl.inelastic_format import (
        _angular, _energy, _photonAngular, _photonEnergies, _photonMean, _values,
    )

    if isinstance(body, EnergyReleaseBody):
        return _values(body.values)
    if isinstance(body, (NuTotalBody, NuPromptBody, NuDelayedBody)):
        return _yieldLines(body)
    return {DT_ANGULAR: _angular, DT_ENERGY: _energy, DT_PHOTON_MULTIPLICITY: _photonMean,
            DT_PHOTON_ANGULAR: _photonAngular, DT_PHOTON_ENERGY: _photonEnergies}[dt](body)


def formatFissionSectionBody(section: FissionSection) -> str:
    """One section's body, its ``infoType dataType`` excluded: what a verbatim section keeps."""
    return "\n".join(_bodyLines(section.infoType, section.dataType, section.body))


def formatFissionFS(record: FissionFSRecord) -> str:
    """``Fission/FS`` text, section by section, in the consumer's reading order."""
    out = _headerLines(record.header)
    for s in record.sections:
        out.append(f"{int(s.infoType)} {int(s.dataType)}")
        out += _bodyLines(s.infoType, s.dataType, s.body)
    return "\n".join(out) + "\n"


def formatChanceFission(record: ChanceFissionRecord) -> str:
    """``Fission/FC`` … ``LC`` text: the σ and, when there is one, the final state."""
    from kika.g4ndl.encode import formatCrossSection
    from kika.g4ndl.inelastic_format import _angular, _energy

    text = formatCrossSection(record.crossSection)
    if not record.hasFinalState:
        return text
    out = [f"{int(record.angularHead[0])} {int(record.angularHead[1])}"]
    out += _angular(record.angular)
    out.append(f"{int(record.energyHead[0])} {int(record.energyHead[1])}")
    out += _energy(record.energy)
    return text + "\n".join(out) + "\n"


def formatFragmentYields(record: FragmentYieldsRecord) -> str:
    """``Fission/FF`` text: one value per line in the headers, a triple per line after."""
    from kika.g4ndl.inelastic_format import _f, _i

    out = _headerLines(record.header)
    for b in record.blocks:
        out += [_i(b.MT), _i(b.MF), _f(b.dummy), _i(b.imax)]
        for at in b.energies:
            out += [_f(at.energy), _i(len(at)), _i(at.interpolation)]
            out += [f"{int(z)} {int(m)} {_f(y)}"
                    for z, m, y in zip(at.fsp.tolist(), at.isomer.tolist(), at.yields.tolist())]
    return "\n".join(out) + "\n"


def fissionDifferences(a, b) -> List[str]:
    """Every field where two fission records differ (at most ~50); ``path`` ignored."""
    from kika.g4ndl.inelastic_format import inelasticDifferences

    return inelasticDifferences(a, b)
