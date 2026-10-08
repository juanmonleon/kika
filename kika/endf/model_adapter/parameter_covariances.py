"""MF32 → §25.3 ``parameterCovariances``.

Separate from :mod:`kika.endf.model_adapter.covariances` for the reason §25.3 is
a separate subsection of the standard: the rows of these matrices are *model
parameters*, not bins of a grid. Nothing about row 47 is recoverable from an
energy axis — it means "the neutron width of the twelfth resonance" or it means
nothing — so the container, the links and the decode all differ, and mixing them
into the cross-section path would eventually collapse a resonance index into an
energy without anything noticing.

**What this module had to measure rather than read.** ENDF-102 §32 leaves three
things ambiguous enough that implementing from the text alone gives a decoder
that is wrong on real tapes, and each was settled against the evaluations on
this machine (the commands are in ``docs/library/mf32-notes.md``):

1. **The vector is resonance-major.** Every parameter of resonance 1, then every
   parameter of resonance 2. Read parameter-major, Mn-55's block gives resonance
   50 a 67 % uncertainty on GG and resonance 5 a 2.5e-8 one on GN — absurd both
   ways, which is what makes this checkable rather than a matter of taste.
2. **LCOMP=2's matrix does not cover every parameter.** It covers those columns
   carrying a non-zero uncertainty *somewhere in the section*, which is ER, GN
   and GG on both Na-23 and Th-232 — three per resonance, not six. Na-23 makes
   the distinction visible: 69 rows for 65 non-zero uncertainties, so the rule
   is per-column and not per-entry, and a per-entry reading is off by four.
3. **LRF=7 is the exception to (2)** — its NNN counts ``(NCH+1)`` per resonance
   whatever the uncertainties are, zeros included. Cl-35, Cu-63 and W-186 agree.

Every one of those is cross-checked at decode time against NNN, and a section
whose arithmetic disagrees is reported and skipped rather than reshaped into
something that fits.

**What is deliberately not decoded.** LCOMP=1's long-range blocks (§32.2.2.5,
NLRS>0) and LCOMP=1 for LRF=7 (§32.2.2.4). No evaluation on this machine has
either, so a decoder for them would be code no test could reach; they are
reported as unsupported and the rest of the section still converts.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from kika.nuclear_data.model import (
    ConversionReport,
    DataLink,
    ParameterCovariance,
    ParameterCovarianceMatrix,
    ParameterLink,
)

__all__ = ["decodeMF32MT", "encodeMF32MT", "MF32MT151_KEY",
           "resonanceParametersHref", "unresolvedParametersHref"]


# ---------------------------------------------------------------------------
# Parameter naming
# ---------------------------------------------------------------------------
#
# The six slots of a File 2 resonance record are positional, and which quantity
# sits in each depends on LRF. `_SLOTS` names them; `_COVERED` gives the order in
# which §32 admits them into a covariance, which is *not* the record order — AJ
# never carries an uncertainty and GT is redundant with GN+GG+GF, so both are
# skipped. MPAR counts down `_COVERED`, so MPAR=3 means ER, GN, GG.

_SLOTS = {
    1: ("ER", "AJ", "GT", "GN", "GG", "GF"),
    2: ("ER", "AJ", "GT", "GN", "GG", "GF"),
    3: ("ER", "AJ", "GN", "GG", "GFA", "GFB"),
}

_COVERED = {
    1: ("ER", "GN", "GG", "GF"),
    2: ("ER", "GN", "GG", "GF"),
    3: ("ER", "GN", "GG", "GFA", "GFB"),
}

#: §32.2.4's average unresolved parameters, in record order and in the order
#: MPAR admits them. AJ is skipped for the same reason as above.
_UNRESOLVED_SLOTS = ("D", "AJ", "GNO", "GG", "GF", "GX")
_UNRESOLVED_COVERED = ("D", "GNO", "GG", "GF", "GX")

#: The four parameters an LCOMP=0 block carries covariances for, and where its
#: twelve variance terms sit relative to them. §32.2.1 orders the terms
#: DE2, DN2, DNDG, DG2, DNDF, DGDF, DF2 and then four null spin terms; the four
#: are null *by construction* and §32.3 procedure 2 says a non-zero one is to be
#: treated as null anyway, so they are dropped rather than carried as zeros.
_LCOMP0_PARAMETERS = ("ER", "GN", "GG", "GF")


def _formalism(lrf: int) -> str:
    """Which model node an LRF's parameters live on.

    §19 splits the resolved region by formalism rather than by record position,
    so LRF is what decides the node — and **LRF=3 is an R-matrix node, not a
    Breit-Wigner one**. Reich-Moore is an R-matrix approximation and
    :func:`kika.endf.model_adapter.resonances._decodeReichMoore` builds an
    ``RMatrix`` for it; a href that said ``BreitWigner`` would point at a node
    that never exists for exactly the evaluations MF32 is most often written
    for (Th-232, Mn-55, Ta-181, Pu-239 are all LRF=3).
    """
    return "BreitWigner" if lrf in (1, 2) else "RMatrix"


def resonanceParametersHref(rangeIndex: int, formalism: str = "BreitWigner") -> str:
    """xPath to the resolved resonance table an MF32 covariance is about.

    Follows the shape of :func:`kika.endf.model_adapter.covariances.reactionHref`
    and carries the same caveat: the path matches what kika's own decoder builds
    (``Resonances.resolved`` is a list here), not a GNDS-mandated spelling.
    """
    return (
        f"/reactionSuite/resonances/resolved[{rangeIndex}]"
        f"/{formalism}/resonanceParameters/table"
    )


def unresolvedParametersHref() -> str:
    """xPath to the unresolved average-parameter table."""
    return "/reactionSuite/resonances/unresolved/tabulatedWidths"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triangleToFull(values: Sequence[float], order: int) -> np.ndarray:
    """Expand a row-major upper triangle, diagonal included, to a full matrix."""
    matrix = np.zeros((order, order), dtype=float)
    index = 0
    for row in range(order):
        for column in range(row, order):
            matrix[row, column] = matrix[column, row] = values[index]
            index += 1
    return matrix


def _intgOutsideMatrix(correlations, report: ConversionReport, what: str) -> bool:
    """True, and a loss reported, if an INTG record names a row outside NNN.

    JEFF-4.0 K-41 writes row 201 under NNN = 93. Unpacking it would index past
    the matrix; dropping the record silently would hand on a matrix without
    correlations the file states. Neither is a decode, so the section is skipped.
    """
    if correlations is None:
        return False
    nnn = int(correlations.nnn)
    bad = [(ii, jj) for ii, jj, _ in correlations.entries
           if not 1 <= ii <= nnn or not 1 <= jj <= nnn]
    if bad:
        report.lost(
            f"{what}: {len(bad)} INTG record(s) name a row or column outside the "
            f"NNN={nnn} matrix (first II={bad[0][0]}, JJ={bad[0][1]}); not decoded"
        )
    return bool(bad)


def _links(labels: Sequence[str], names: Sequence[str], href: str) -> List[ParameterLink]:
    """One link per resonance, each covering the same ``names`` in order."""
    width = len(names)
    return [
        ParameterLink(label=label, href=href, nParameters=width,
                      matrixStartIndex=index * width,
                      parameterNames=list(names))
        for index, label in enumerate(labels)
    ]


def _covered(lrf: int, report: ConversionReport) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    slots = _SLOTS.get(lrf)
    if slots is None:
        report.unsupportedNode(
            f"MF32: LRF={lrf} has no named parameter slots in this decoder "
            f"(§32 covers LRF=1, 2, 3 through these bodies and LRF=7 through "
            f"its own); the section's matrix is not decoded"
        )
        return None
    return slots, _COVERED[lrf]


# ---------------------------------------------------------------------------
# The five bodies
# ---------------------------------------------------------------------------

def _decodeLCOMP0(body, lrf: int, href: str,
                  report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.1 — 18 numbers a resonance, block-diagonal by resonance.

    Each resonance carries its own 4x4 over (ER, GN, GG, GF) and nothing
    correlates two resonances, which is the whole content of the ENDF/B-V
    format: the matrix this returns is block-diagonal by construction rather
    than by approximation.
    """
    names = _LCOMP0_PARAMETERS
    width = len(names)
    labels: List[str] = []
    blocks: List[np.ndarray] = []
    values: List[float] = []

    for block in body.l_blocks:
        raw = block.values
        count = int(block.n2)
        if len(raw) < 18 * count:
            report.lost(
                f"MF32 LCOMP=0: an L={block.l2} block declares NRS={count} "
                f"but carries {len(raw)} numbers, not {18 * count}"
            )
            return None
        table = np.asarray(raw[:18 * count], dtype=float).reshape(count, 18)
        for row in table:
            er, _aj, _gt, gn, gg, gf = row[:6]
            de2, dn2, dndg, dg2, dndf, dgdf, df2 = row[6:13]
            cell = np.array([
                [de2, 0.0,  0.0,  0.0],
                [0.0, dn2,  dndg, dndf],
                [0.0, dndg, dg2,  dgdf],
                [0.0, dndf, dgdf, df2],
            ], dtype=float)
            blocks.append(cell)
            labels.append(f"L{int(block.l2)}/resonance{len(labels)}")
            values.extend((er, gn, gg, gf))

    if not blocks:
        return None

    order = width * len(blocks)
    matrix = np.zeros((order, order), dtype=float)
    for index, cell in enumerate(blocks):
        start = index * width
        matrix[start:start + width, start:start + width] = cell

    return ParameterCovarianceMatrix(
        matrix=matrix, parameters=_links(labels, names, href),
        isRelative=False, parameterValues=np.asarray(values, dtype=float),
    )


def _decodeLCOMP1Block(block, lrf: int, href: str, blockIndex: int,
                       report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.2.1 — one short-range block: NRB resonances, then the triangle."""
    covered = _covered(lrf, report)
    if covered is None:
        return None
    slots, order_names = covered

    mpar = int(block.l1)
    count = int(block.n2)
    if not 1 <= mpar <= len(order_names):
        report.lost(
            f"MF32 LCOMP=1 block {blockIndex}: MPAR={mpar} is outside 1.."
            f"{len(order_names)} for LRF={lrf}"
        )
        return None

    names = order_names[:mpar]
    width = mpar
    npar = width * count
    raw = block.values
    expected = 6 * count + npar * (npar + 1) // 2
    if len(raw) < expected:
        report.lost(
            f"MF32 LCOMP=1 block {blockIndex}: MPAR={mpar} and NRB={count} "
            f"need {expected} numbers, the block carries {len(raw)}"
        )
        return None

    table = np.asarray(raw[:6 * count], dtype=float).reshape(count, 6)
    matrix = _triangleToFull(raw[6 * count:expected], npar)

    columns = [slots.index(name) for name in names]
    values = table[:, columns].reshape(-1)
    labels = [f"block{blockIndex}/resonance{i}" for i in range(count)]

    return ParameterCovarianceMatrix(
        matrix=matrix, parameters=_links(labels, names, href),
        isRelative=False, parameterValues=values,
    )


def _decodeLCOMP2(body, lrf: int, href: str,
                  report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.3.1-2 — parameters and uncertainties interleaved, then INTG.

    The covariance is rebuilt as ``D R D``: the file stores the correlations as
    packed integers and the standard deviations beside the parameters, and
    neither half is a covariance on its own.
    """
    covered = _covered(lrf, report)
    if covered is None:
        return None
    slots, order_names = covered

    record = body.parameters
    if record is None:
        return None
    count = int(record.n2)
    raw = record.values
    if count <= 0 or len(raw) < 12 * count:
        report.lost(
            f"MF32 LCOMP=2: NRSA={count} needs {12 * count} numbers, "
            f"the record carries {len(raw)}"
        )
        return None

    table = np.asarray(raw[:12 * count], dtype=float).reshape(count, 12)
    parameters, uncertainties = table[:, :6], table[:, 6:]

    # Which columns the matrix covers: those carrying an uncertainty anywhere in
    # the section. Per column and not per entry -- Na-23 has 69 rows against 65
    # non-zero uncertainties, so a per-entry rule is off by four and every row
    # after the first zero would be misassigned.
    candidates = [slots.index(name) for name in order_names]
    columns = [c for c in candidates if np.any(uncertainties[:, c] != 0.0)]
    if not columns:
        report.lost("MF32 LCOMP=2: no parameter carries an uncertainty")
        return None

    names = tuple(slots[c] for c in columns)
    width = len(columns)
    order = width * count

    if body.correlations is not None and int(body.correlations.nnn) != order:
        report.lost(
            f"MF32 LCOMP=2: the INTG record declares NNN="
            f"{int(body.correlations.nnn)} but {width} parameter(s) "
            f"{'/'.join(names)} over {count} resonances give {order} rows; "
            f"the section is not decoded rather than reshaped to fit"
        )
        return None

    if _intgOutsideMatrix(body.correlations, report, "MF32 LCOMP=2"):
        return None
    sigma = uncertainties[:, columns].reshape(-1)
    correlation = (body.correlations.correlation_matrix()
                   if body.correlations is not None else np.eye(order))
    matrix = correlation * np.outer(sigma, sigma)

    labels = [f"resonance{i}" for i in range(count)]
    return ParameterCovarianceMatrix(
        matrix=matrix, parameters=_links(labels, names, href),
        isRelative=False,
        parameterValues=parameters[:, columns].reshape(-1),
    )


def _decodeLCOMP2RML(body, href: str,
                     report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.3.3 — R-Matrix Limited: ER plus one width per channel.

    Unlike §32.2.3.2 the matrix covers **every** parameter of every resonance,
    zero uncertainties included: NNN is ``sum (NCH+1) * NRSA`` on Cl-35, Cu-63
    and W-186 alike, and dropping the zero-uncertainty rows here would make the
    order disagree with the file by exactly the number of unassigned channels.

    Each resonance occupies two rows of the LIST padded to a full record, the
    values then the uncertainties, so the stride is a multiple of six rather
    than ``NCH+1``.
    """
    labels: List[str] = []
    names: List[str] = []
    sigma: List[float] = []
    values: List[float] = []
    widths: List[int] = []

    for groupIndex, group in enumerate(body.spin_groups):
        channels = int(group.nch)
        count = int(group.nrsa)
        width = channels + 1
        stride = 6 * ((width + 5) // 6)
        raw = group.resonances.values
        # Equality, not "at least": every tape on this machine has NCH <= 3, so
        # the padding term has never been exercised above one line. If it is
        # wrong for a wider group, NPL is what says so — and a >= test would
        # read the first row of each resonance and silently mean something else.
        if int(group.resonances.n1) != 2 * stride * count or len(raw) < 2 * stride * count:
            report.lost(
                f"MF32 LCOMP=2 LRF=7: spin group {groupIndex} declares "
                f"NCH={channels} NRSA={count}, which needs NPL="
                f"{2 * stride * count} at a stride of {stride}; the record "
                f"declares NPL={int(group.resonances.n1)} and carries "
                f"{len(raw)} numbers"
            )
            return None
        table = np.asarray(raw[:2 * stride * count], dtype=float)
        table = table.reshape(count, 2, stride)
        groupNames = ["ER"] + [f"GAM{c + 1}" for c in range(channels)]
        for resonance in range(count):
            values.extend(table[resonance, 0, :width])
            sigma.extend(table[resonance, 1, :width])
            labels.append(f"spinGroup{groupIndex}/resonance{resonance}")
            names.extend(groupNames)
            widths.append(width)

    if not labels:
        return None

    order = len(sigma)
    if body.correlations is not None and int(body.correlations.nnn) != order:
        report.lost(
            f"MF32 LCOMP=2 LRF=7: the INTG record declares NNN="
            f"{int(body.correlations.nnn)} against sum (NCH+1)*NRSA = {order}"
        )
        return None

    if _intgOutsideMatrix(body.correlations, report, "MF32 LCOMP=2 LRF=7"):
        return None
    sigmaArray = np.asarray(sigma, dtype=float)
    correlation = (body.correlations.correlation_matrix()
                   if body.correlations is not None else np.eye(order))
    matrix = correlation * np.outer(sigmaArray, sigmaArray)

    links: List[ParameterLink] = []
    start = 0
    for label, width in zip(labels, widths):
        links.append(ParameterLink(
            label=label, href=href, nParameters=width, matrixStartIndex=start,
            parameterNames=names[start:start + width],
        ))
        start += width

    return ParameterCovarianceMatrix(
        matrix=matrix, parameters=links, isRelative=False,
        parameterValues=np.asarray(values, dtype=float),
    )


def _decodeUnresolved(body, href: str,
                      report: ConversionReport) -> Optional[ParameterCovarianceMatrix]:
    """§32.2.4 — average parameters per (L, J), and one **relative** triangle.

    ``NPAR`` is computed rather than read: Th-232 writes 0 in the field §32.2.4
    draws it in, and the triangle it then carries — 120 numbers for MPAR=3 over
    five (L, J) combinations — only makes sense against the computed 15.
    """
    matrixRecord = body.matrix
    if matrixRecord is None:
        report.lost("MF32 LRU=2: no covariance record after the L blocks")
        return None

    mpar = int(matrixRecord.l1)
    if not 1 <= mpar <= len(_UNRESOLVED_COVERED):
        report.lost(
            f"MF32 LRU=2: MPAR={mpar} is outside 1..{len(_UNRESOLVED_COVERED)}"
        )
        return None
    names = _UNRESOLVED_COVERED[:mpar]
    columns = [_UNRESOLVED_SLOTS.index(name) for name in names]

    labels: List[str] = []
    values: List[float] = []
    for block in body.l_blocks:
        raw = block.values
        states = int(block.n2)
        if len(raw) < 6 * states:
            report.lost(
                f"MF32 LRU=2: an L={block.l1} block declares NJS={states} "
                f"but carries {len(raw)} numbers, not {6 * states}"
            )
            return None
        table = np.asarray(raw[:6 * states], dtype=float).reshape(states, 6)
        for state in range(states):
            labels.append(f"L{int(block.l1)}/J{state}")
            values.extend(table[state, columns])

    order = mpar * len(labels)
    triangle = matrixRecord.values
    expected = order * (order + 1) // 2
    if len(triangle) < expected:
        report.lost(
            f"MF32 LRU=2: MPAR={mpar} over {len(labels)} (L, J) states needs "
            f"a triangle of {expected} numbers, the record carries "
            f"{len(triangle)}"
        )
        return None

    return ParameterCovarianceMatrix(
        matrix=_triangleToFull(triangle[:expected], order),
        parameters=_links(labels, names, href),
        isRelative=True,
        parameterValues=np.asarray(values, dtype=float),
    )


# ---------------------------------------------------------------------------
# The descent
# ---------------------------------------------------------------------------

def _rangeLink(href: str, energyRange) -> DataLink:
    """The row link: the parameter table, restricted to this energy range."""
    return DataLink.forIncidentEnergyBand(
        href, float(energyRange.el), float(energyRange.eh),
        ENDF_MFMT="32/151", dimension=1,
    )


def decodeMF32MT(mf32mt, report: Optional[ConversionReport] = None):
    """One MF32/MT151 section → a list of :class:`ParameterCovariance`.

    One per covariance body, which is one per energy range except for LCOMP=1:
    its short-range blocks are independent covariances over disjoint sets of
    resonances, so each becomes its own node rather than being padded into a
    common block-diagonal matrix that would claim the zeros are a statement.
    """
    report = report if report is not None else ConversionReport()
    from .covariances import _sectionProvenance

    provenance = _sectionProvenance(mf32mt)
    covariances: List[ParameterCovariance] = []

    for isotopeIndex, isotope in enumerate(getattr(mf32mt, "isotopes", [])):
        for rangeIndex, energyRange in enumerate(isotope.energy_ranges):
            body = energyRange.body
            if body is None:
                continue
            kind = type(body).__name__
            stem = f"MF32-iso{isotopeIndex}-range{rangeIndex}"

            if energyRange.nro:
                report.unsupportedNode(
                    f"{stem}: NRO={energyRange.nro} gives the scattering radius "
                    f"its own File 33-style covariance (§32.2), which is not "
                    f"decoded; the resonance parameter matrix below is"
                )

            href = (unresolvedParametersHref() if kind == "UnresolvedBody"
                    else resonanceParametersHref(rangeIndex,
                                                 _formalism(energyRange.lrf)))

            if kind == "UnresolvedBody":
                forms = [_decodeUnresolved(body, href, report)]
            elif kind == "LCOMP0Body":
                forms = [_decodeLCOMP0(body, energyRange.lrf, href, report)]
            elif kind == "LCOMP2Body":
                forms = [_decodeLCOMP2(body, energyRange.lrf, href, report)]
            elif kind == "LCOMP2RMLBody":
                forms = [_decodeLCOMP2RML(body, href, report)]
            elif kind == "LCOMP1Body":
                if body.long_range:
                    report.unsupportedNode(
                        f"{stem}: {len(body.long_range)} long-range "
                        f"subsection(s) (§32.2.2.5, NLRS>0) are parsed but not "
                        f"decoded; the short-range blocks are"
                    )
                forms = [
                    _decodeLCOMP1Block(block, energyRange.lrf, href, index, report)
                    for index, block in enumerate(body.short_range)
                ]
            elif kind == "LCOMP1RMLBody":
                report.unsupportedNode(
                    f"{stem}: LCOMP=1 with LRF=7 (§32.2.2.4) is parsed but not "
                    f"decoded — no evaluation on this machine has one, so a "
                    f"decoder for it would be untestable"
                )
                forms = []
            else:
                report.unsupportedNode(f"{stem}: no decoder for a {kind}")
                forms = []

            link = _rangeLink(href, energyRange)
            for formIndex, form in enumerate(forms):
                if form is None:
                    continue
                suffix = f"-block{formIndex}" if len(forms) > 1 else ""
                covariances.append(ParameterCovariance(
                    label=f"{stem}{suffix}",
                    rowData=link,
                    columnData=None,
                    form=form,
                    provenance=provenance,
                ))

    if not covariances:
        report.lost("MF32/MT151: no parameter covariance decoded")
    return covariances, report


# ---------------------------------------------------------------------------
# The way back: model → MF32 (roadmap E1)
# ---------------------------------------------------------------------------
#
# **Why the encoder writes into the section it read, and not from File 2.** An
# MF32 section restates File 2's outer shape — isotopes, ranges, LRF, LCOMP,
# NRO, NAPS, the AP/DAP records, LRF=7's particle pairs and channels — and the
# model holds none of that beside the matrix: §25.3 points at the resonance
# table by ``href`` and stops. Rebuilding those records from the model's
# ``resonances`` would mean a second place that decides what an MF32 header
# looks like, and it would "correct" the departures from §32 the read gate
# exists to preserve (Th-232's INTG control record, Cl-35's NJS against NJSX).
# So the decoder of the suite keeps the section's text in the provenance, the
# way MF1/460 travels (``fission_energy.MF1MT460_KEY``), and this encoder
# parses it back and changes only what the model owns: the parameter values,
# the uncertainties and the matrix. Same rule as MF6, whose provenance decides
# whether a section exists at all: **a suite that never read an MF32 writes
# none**, and says so.
#
# **Unchanged means byte-identical.** Each covariance is compared with what the
# kept text decodes to. Equal, and its records are not touched, so a plain
# ENDF → model → ENDF trip gives the section back character for character.
# Different, and its LIST bodies are rewritten from the model, and an LCOMP=2
# correlation matrix is re-packed into INTG records at the section's own NDIGIT
# — which **quantizes** it, and is declared as an approximation every time.
#
# What the decoder could not decode (NRO's radius covariance, LCOMP=1
# long-range blocks, LCOMP=1 for LRF=7, a section whose arithmetic disagreed)
# never reached the model, so the model cannot have changed it: it goes back
# out as the bytes it came in as. The decode already reported each of them.

#: Where the decoder of the suite keeps an MF32 section's text, on the
#: provenance every :class:`ParameterCovariance` of that section shares.
MF32MT151_KEY = "mf32mt151"

_LABEL = re.compile(r"^MF32-iso(\d+)-range(\d+)(?:-block(\d+))?$")


def _templateKey(covariance) -> Optional[Tuple[int, int, int]]:
    """``(isotope, range, block)`` from the label :func:`decodeMF32MT` gave."""
    match = _LABEL.match(getattr(covariance, "label", None) or "")
    if match is None:
        return None
    iso, rng, block = match.groups()
    return int(iso), int(rng), int(block or 0)


def _names(form) -> List[Tuple[str, ...]]:
    return [tuple(link.parameterNames) for link in form.parameters]


def _sameForm(a, b) -> bool:
    """Exactly the matrix, values, flag and parameter names the file states."""
    if bool(a.isRelative) != bool(b.isRelative):
        return False
    if a.matrix.shape != b.matrix.shape or not np.array_equal(a.matrix, b.matrix):
        return False
    va = None if a.parameterValues is None else np.asarray(a.parameterValues)
    vb = None if b.parameterValues is None else np.asarray(b.parameterValues)
    if (va is None) != (vb is None):
        return False
    if va is not None and (va.shape != vb.shape or not np.array_equal(va, vb)):
        return False
    return _names(a) == _names(b)


def _assertShape(form, template, what: str) -> None:
    """A rewrite keeps every record length, so the shape must not move."""
    if form.matrix.shape != template.matrix.shape:
        raise ValueError(
            f"{what}: the model's matrix is {form.matrix.shape} and the section "
            f"states {template.matrix.shape}. MF32 restates File 2's resonance "
            f"list, so a covariance over a different set of parameters needs a "
            f"different File 2 first; it cannot be written into this one"
        )
    if _names(form) != _names(template):
        raise ValueError(
            f"{what}: the model's parameter list differs from the section's "
            f"(names or order); the rows would be written under the wrong "
            f"parameters"
        )
    if bool(form.isRelative) != bool(template.isRelative):
        raise ValueError(
            f"{what}: the model says isRelative={bool(form.isRelative)} and "
            f"this MF32 body can only state "
            f"isRelative={bool(template.isRelative)}"
        )
    if form.parameterValues is None or (
            np.asarray(form.parameterValues).shape
            != np.asarray(template.parameterValues).shape):
        raise ValueError(
            f"{what}: MF32 restates every parameter value beside its "
            f"uncertainty, and the model's parameterValues are missing or of "
            f"another length"
        )


def _sigmaAndCorrelation(matrix: np.ndarray, what: str,
                         report: ConversionReport):
    diagonal = np.diag(matrix).copy()
    if np.any(diagonal < 0.0):
        report.lost(
            f"{what}: {int(np.sum(diagonal < 0))} negative variance(s) cannot "
            f"be written as an uncertainty; written as 0"
        )
        diagonal = np.clip(diagonal, 0.0, None)
    sigma = np.sqrt(diagonal)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = matrix / np.outer(sigma, sigma)
    correlation[~np.isfinite(correlation)] = 0.0
    return sigma, correlation


def _packIntg(correlation: np.ndarray,
              ndigit: int) -> List[Tuple[int, int, List[int]]]:
    """§32.2.3's INTG records for the strict lower triangle of *correlation*.

    The inverse of :meth:`IntgMatrix.correlation_matrix`, which reads ``K`` as
    the centre of its range, ``(|K| + 0.5) / 10**NDIGIT``. So ``K`` is the
    floor of ``|C| * 10**NDIGIT``, signed and capped at ``10**NDIGIT - 1``: a
    coefficient read from a tape packs back to the integer it was read from,
    and one below ``10**-NDIGIT`` is dropped, which is what §32.2.3 means by
    "rounds to zero".
    """
    from kika.endf.utils import intg_row_length

    factor = 10 ** ndigit
    nrow = intg_row_length(ndigit)
    order = correlation.shape[0]
    entries: List[Tuple[int, int, List[int]]] = []
    for row in range(1, order):
        coefficients = correlation[row, :row]
        magnitude = np.minimum(np.floor(np.abs(coefficients) * factor + 1e-9),
                               factor - 1)
        packed = (np.sign(coefficients) * magnitude).astype(int)
        column = 0
        while column < row:
            if packed[column] == 0:
                column += 1
                continue
            stop = min(column + nrow, row)
            entries.append((row + 1, column + 1,
                            [int(k) for k in packed[column:stop]]))
            column = stop
    return entries


def _rewriteIntg(old, nnn: int, correlation: np.ndarray, what: str,
                 report: ConversionReport):
    """A new :class:`IntgMatrix` for *correlation*, in *old*'s conventions."""
    from kika.endf.classes.mf32.mf32mt151 import IntgMatrix
    from kika.endf.classes.mf32.records import DATA_WIDTH
    from kika.endf.utils import (ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                                 format_endf_data_line, format_intg)

    offDiagonal = bool(np.any(np.tril(correlation, -1) != 0.0))
    if old is None and not offDiagonal:
        return None
    ndigit = int(old.ndigit) if old is not None else 2
    if old is None:
        report.approximated(
            f"{what}: the section stated no INTG records (a diagonal matrix) "
            f"and the model has correlations; an INTG block is added at NDIGIT=2"
        )
    entries = _packIntg(correlation, ndigit)
    lines, _ = format_intg(entries, ndigit, 0, 0, 0, 1)
    nm = len(entries)
    # Th-232 writes NM in the sixth field too, where §32.2.3 draws a 0. The
    # section's own convention is kept, not the manual's.
    statesNm = old is not None and old.nm and old.n2 == old.nm
    n2 = nm if statesNm else (int(old.n2) if old is not None else 0)
    if old is not None and (old.ndigit, old.nnn, old.nm, old.n2) == (ndigit, nnn, nm, n2):
        control = old.control_raw
    else:
        control = format_endf_data_line(
            [0.0, 0.0, ndigit, nnn, nm, n2], 0, 0, 0, 0,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT] + [ENDF_FORMAT_INT] * 4,
        )[:DATA_WIDTH]
    report.approximated(
        f"{what}: the correlations were re-packed into INTG records at "
        f"NDIGIT={ndigit}, which keeps {ndigit} digit(s) of each coefficient"
    )
    return IntgMatrix(ndigit=ndigit, nnn=nnn, nm=nm, n2=n2,
                      control_raw=control,
                      lines=[line[:DATA_WIDTH] for line in lines])


def _rewriteLCOMP0(body, form, what: str, report: ConversionReport) -> None:
    """§32.2.1: each resonance's 4x4, back into its eighteen numbers."""
    values = np.asarray(form.parameterValues, dtype=float)
    matrix = form.matrix
    width = len(_LCOMP0_PARAMETERS)
    stated = np.zeros(matrix.shape, dtype=bool)
    resonance = 0
    for block in body.l_blocks:
        raw = list(block.body.values)
        for k in range(int(block.n2)):
            start = width * resonance
            cell = matrix[start:start + width, start:start + width]
            stated[start, start] = True
            stated[start + 1:start + width, start + 1:start + width] = True
            row = 18 * k
            er, gn, gg, gf = values[start:start + width]
            raw[row + 0], raw[row + 3], raw[row + 4], raw[row + 5] = er, gn, gg, gf
            raw[row + 6:row + 13] = [cell[0, 0], cell[1, 1], cell[1, 2],
                                     cell[2, 2], cell[1, 3], cell[2, 3],
                                     cell[3, 3]]
            resonance += 1
        block.body.set_values(raw)
    dropped = np.abs(matrix[~stated])
    if dropped.size and dropped.max() > 0.0:
        report.lost(
            f"{what}: LCOMP=0 states no covariance between resonances nor "
            f"between ER and the widths; {int(np.count_nonzero(dropped))} "
            f"non-zero term(s) of the model's matrix (largest "
            f"{dropped.max():.3e}) are not written"
        )


def _rewriteLCOMP1Block(block, lrf: int, form) -> None:
    """§32.2.2.1: the NRB resonances' parameters, then the upper triangle."""
    slots = _SLOTS[lrf]
    count = int(block.n2)
    columns = [slots.index(name) for name in form.parameters[0].parameterNames]
    raw = list(block.body.values)
    table = np.asarray(raw[:6 * count], dtype=float).reshape(count, 6)
    table[:, columns] = np.asarray(form.parameterValues,
                                   dtype=float).reshape(count, -1)
    upper = form.matrix[np.triu_indices(form.matrix.shape[0])]
    block.body.set_values(list(table.reshape(-1)) + list(upper))


def _rewriteLCOMP2(body, lrf: int, form, what: str,
                   report: ConversionReport) -> None:
    """§32.2.3.1-2: values and uncertainties interleaved, then INTG."""
    slots = _SLOTS[lrf]
    record = body.parameters
    count = int(record.n2)
    columns = [slots.index(name) for name in form.parameters[0].parameterNames]
    raw = list(record.body.values)
    table = np.asarray(raw[:12 * count], dtype=float).reshape(count, 12)
    sigma, correlation = _sigmaAndCorrelation(form.matrix, what, report)
    table[:, columns] = np.asarray(form.parameterValues,
                                   dtype=float).reshape(count, -1)
    table[:, [6 + c for c in columns]] = sigma.reshape(count, -1)
    record.body.set_values(list(table.reshape(-1)) + raw[12 * count:])
    body.correlations = _rewriteIntg(body.correlations, form.matrix.shape[0],
                                     correlation, what, report)


def _rewriteLCOMP2RML(body, form, what: str, report: ConversionReport) -> None:
    """§32.2.3.3: per spin group, a values row and an uncertainties row."""
    sigma, correlation = _sigmaAndCorrelation(form.matrix, what, report)
    values = np.asarray(form.parameterValues, dtype=float)
    start = 0
    for group in body.spin_groups:
        width = int(group.nch) + 1
        count = int(group.nrsa)
        stride = 6 * ((width + 5) // 6)
        raw = list(group.resonances.body.values)
        table = np.asarray(raw[:2 * stride * count],
                           dtype=float).reshape(count, 2, stride)
        for resonance in range(count):
            table[resonance, 0, :width] = values[start:start + width]
            table[resonance, 1, :width] = sigma[start:start + width]
            start += width
        group.resonances.body.set_values(list(table.reshape(-1))
                                         + raw[2 * stride * count:])
    body.correlations = _rewriteIntg(body.correlations, form.matrix.shape[0],
                                     correlation, what, report)


def _rewriteUnresolved(body, form) -> None:
    """§32.2.4: the average parameters per (L, J), then the relative triangle."""
    mpar = int(body.matrix.l1)
    columns = [_UNRESOLVED_SLOTS.index(name)
               for name in _UNRESOLVED_COVERED[:mpar]]
    values = np.asarray(form.parameterValues, dtype=float).reshape(-1, mpar)
    state = 0
    for block in body.l_blocks:
        states = int(block.n2)
        raw = list(block.body.values)
        table = np.asarray(raw[:6 * states], dtype=float).reshape(states, 6)
        table[:, columns] = values[state:state + states]
        state += states
        block.body.set_values(list(table.reshape(-1)) + raw[6 * states:])
    upper = list(form.matrix[np.triu_indices(form.matrix.shape[0])])
    tail = list(body.matrix.body.values)[len(upper):]
    body.matrix.body.set_values(upper + tail)


def encodeMF32MT(source, mat: Optional[int] = None,
                 report: Optional[ConversionReport] = None):
    """A :class:`CovarianceSuite`'s ``parameterCovariances`` → an ``MF32MT151``.

    Returns ``(section, report)``. ``section`` is ``None`` when there is
    nothing to write or nowhere to write it into: no parameter covariances, or
    covariances with no MF32 text in their provenance (built in memory, or
    read from GNDS). The second case is declared, not guessed; the notes above
    say why the File 2 layout is not rebuilt from the model.
    """
    from kika.endf.parsers.parse_mf32 import parse_mf32_mt151

    report = report if report is not None else ConversionReport()
    covariances = list(getattr(source, "parameterCovariances", None) or ())
    if not covariances:
        return None, report

    texts = {}
    orphans = []
    for covariance in covariances:
        header = getattr(getattr(covariance, "provenance", None),
                         "headerFields", None) or {}
        lines = header.get(MF32MT151_KEY)
        if lines is None or _templateKey(covariance) is None:
            orphans.append(covariance.label)
        else:
            texts.setdefault(id(lines), lines)

    if orphans:
        report.unsupportedNode(
            f"{len(orphans)} §25.3 parameter covariance(s) carry no MF32 "
            f"section to write into (first {orphans[0]!r}); MF32 restates File "
            f"2's layout, which the model does not hold beside the matrix, so "
            f"no MF32 is written for them"
        )
        return None, report
    if len(texts) != 1:
        report.lost(
            f"the parameter covariances come from {len(texts)} different MF32 "
            f"sections; one material has one MF32/MT151, so none is written"
        )
        return None, report

    section = parse_mf32_mt151(list(next(iter(texts.values()))), 151)
    if mat is not None:
        section._mat = int(mat)

    template, _ = decodeMF32MT(section, ConversionReport())
    templateForms = {_templateKey(c): c.form for c in template}
    modelForms = {_templateKey(c): c.form for c in covariances}

    missing = sorted(set(templateForms) - set(modelForms))
    extra = sorted(set(modelForms) - set(templateForms))
    if missing or extra:
        report.lost(
            f"MF32: the model's parameter covariances are not the section's "
            f"({len(missing)} the section states and the model lacks, "
            f"{len(extra)} the model has and the section has no body for); "
            f"writing it would restate or invent a covariance, so no MF32 is "
            f"written"
        )
        return None, report

    rewritten = 0
    for key in sorted(modelForms):
        form, old = modelForms[key], templateForms[key]
        if _sameForm(form, old):
            continue
        iso, rng, blockIndex = key
        energyRange = section.isotopes[iso].energy_ranges[rng]
        body = energyRange.body
        kind = type(body).__name__
        what = f"MF32-iso{iso}-range{rng}" + (
            f"-block{blockIndex}" if kind == "LCOMP1Body" else "")
        _assertShape(form, old, what)
        if kind == "LCOMP0Body":
            _rewriteLCOMP0(body, form, what, report)
        elif kind == "LCOMP1Body":
            _rewriteLCOMP1Block(body.short_range[blockIndex], energyRange.lrf,
                                form)
        elif kind == "LCOMP2Body":
            _rewriteLCOMP2(body, energyRange.lrf, form, what, report)
        elif kind == "LCOMP2RMLBody":
            _rewriteLCOMP2RML(body, form, what, report)
        elif kind == "UnresolvedBody":
            _rewriteUnresolved(body, form)
        else:  # pragma: no cover - the decoder builds no form for any other
            raise ValueError(f"{what}: no rewrite for a {kind}")
        rewritten += 1

    if rewritten:
        report.approximated(
            f"MF32/MT151: {rewritten} covariance body(ies) changed in the model "
            f"and were rewritten into the section's records; the File 2 values "
            f"MF32 restates are the model's parameterValues, not MF2's"
        )
    return section, report
