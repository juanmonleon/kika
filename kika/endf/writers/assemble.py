"""Model → a whole ENDF-6 tape. The direction that did not exist.

**What this closes.** ``kika/endf/model_adapter`` has held per-section encoders
for a while — ``encodeMF1MT451``, ``encodeMF2MT151``, ``encodeMF3MT``,
``encodeMF4MT``, ``encodeMF1MT452/455/456``, ``encodeMF31MT``, ``encodeMF33MT``,
``encodeMF34MT``, ``encodeMF35MT`` — each gated byte-exact against the file it
came from. What was missing was never the rendering; it was the **assembly**:
the order sections go in, the SEND/FEND/MEND/TEND bookkeeping, the tape
identification record and the MF1/451 directory. That is what this module is.
``docs/library/gnds_endf_conflicts.md`` §2.8, and §10 for why the sampling pipeline
cares.

**The gate is a fixed point inside the model, not byte identity against the
source tape** (decided 2026-08-13, owner Juan)::

    read(tape) → suite → writeEndfTape(suite, tape') → read(tape') → suite'
    assert suite' == suite

Deliberately weaker than byte identity, and deliberately so: where a field is
padded and whether ``1e-5`` is written ``1.0-5`` carry no information, and
chasing them is a different job from being correct. What the fixed point does
catch is the only thing that matters here — **a quantity that does not survive
the trip**. Its blind spot is stated in §2.8 and is real: anything the model
does not carry is absent from both sides and the comparison passes. So the
fixed point is necessary and not sufficient, and it leans on
:class:`~kika.nuclear_data.model.conversion.ConversionReport` being honest about
what did not come through.

**Sections come only from what the model has**, or from what the decoder kept
verbatim beside it and said so. MF12-15 come back from the photon products
(roadmap E5b) and, where the model does not carry them yet (LO=2, MF13, photons
with no reaction), from the text the decoder kept. MF5 comes back whole — its analytic
spectra have had nodes since roadmap E2 — and MF32 is written into the section
the decoder kept (E1). So does **all of MF6**: what the model does
not carry there — LAW=5, and any subsection whose LAW is negative — is kept
verbatim in the reaction's provenance, so the section is re-emitted whole even
where the distribution never reached a node.

**Two limits that no amount of code removes**, both format facts rather than
gaps here: a GNDS reaction may have no MT at all (§2.4), and MF13 is reached
from MF3 by arithmetic rather than by re-encoding (§2.6). Both are reported per
occurrence.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..utils import (format_endf_fend_record, format_endf_mend_record,
                     format_endf_tend_record)

__all__ = ["MF_WRITE_ORDER", "TAPE_ID_MAT", "DEFAULT_TAPE_ID",
           "encodeTapeSections", "assembleTape", "writeEndfTape",
           "writeEndfTapes", "writeReconstructedEndfTape"]

#: The MF numbers an encoder exists for, in the order ENDF-6 puts them on the
#: tape. Ascending, which is also §0.3.2's rule, so the constant is a statement
#: of *coverage* rather than of order: an MF absent here is one nothing can
#: write, not one that sorts late.
MF_WRITE_ORDER = (1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 31, 32, 33, 34, 35)

#: The MAT column of a tape identification record. ENDF-6 §0.6.2 fixes it at 1
#: regardless of the material that follows.
TAPE_ID_MAT = 1

#: What goes in a TPID's 66 text columns when the caller names nothing and the
#: suite kept none. ``read_endf`` keeps the tape's first line
#: (``ENDF.tape_id``) and the decoder hangs it on the suite's provenance, so a
#: round trip writes the source's record back verbatim; this label is for a
#: suite that never came from a tape, and is reported when used.
DEFAULT_TAPE_ID = "TAPE WRITTEN BY KIKA FROM A REACTION SUITE"


def _idColumns(mat: int, mf: int, mt: int, ns: int) -> str:
    """Columns 67-80 — MAT, MF, MT, NS — in their ENDF-6 field widths."""
    return f"{mat:>4}{mf:>2}{mt:>3}{ns:>5}"


def _tapeIdRecord(label: str) -> str:
    """The TPID record (§0.6.2): 66 text columns, MAT=1, MF=MT=NS=0."""
    return f"{label[:66]:<66}" + _idColumns(TAPE_ID_MAT, 0, 0, 0)


def _mat(suite, mat: Optional[int]) -> int:
    """The material number to write in every ID field."""
    if mat is not None:
        return int(mat)
    provenance = getattr(suite, "provenance", None)
    recorded = getattr(provenance, "mat", None)
    if recorded is None:
        raise ValueError(
            "this reactionSuite carries no MAT number -- its provenance has "
            "none, which means it did not come from an ENDF tape -- and every "
            "record of an ENDF file is stamped with one. Pass mat= explicitly."
        )
    return int(recorded)


def _mf1Sections(suite, mat, report, label=None):
    """MF1: the 451 header first, then whichever nu-bars the suite carries.

    *label* selects which §9.1 form of each nu-bar is written, with the same
    fall-back and the same report line MF3 has: a realisation that perturbed the
    prompt nu-bar and not the delayed one still writes a whole tape, and the
    report says which members carried the label and which fell back. That is
    what a ``multiplicity`` becoming a
    :class:`~kika.nuclear_data.model.component.Component` bought -- before it,
    a realisation had to *replace* the evaluated form and there was no label for
    this function to be told about.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    from ..model_adapter import (encodeMF1MT451, encodeMF1MT452,
                                 encodeMF1MT455, encodeMF1MT456)
    from ..model_adapter.multiplicity import nubarNode

    label = EVAL_LABEL if label is None else label

    sections = []
    header, report = encodeMF1MT451(suite, mat, report)
    from kika.nuclear_data.model import CrossSectionReconstructed
    try:
        selected_style = suite.styles[label]
    except KeyError:
        selected_style = None
    if isinstance(selected_style, CrossSectionReconstructed):
        missing = [r.label for r in _mf3Bearing(suite) if label not in r.crossSection]
        if missing:
            raise ValueError(f"reconstructed ENDF export cannot mix evaluated cross sections: {missing}")
        # ENDF-102 §1.1: keep MF2, but MF3 already contains its contribution.
        # Change the encoded header only; the evaluated model remains intact.
        header._lrp = 2
        header._ldrv = 1
    sections.append((1, 451, header))

    # 452, 455, 456 -- ascending, which is the order they sit in on the tape.
    # `nubarNode` rather than a try/except around each encoder: the encoders
    # raise on absence, and absence is the common case (nothing but a fissile
    # material has any of these).
    written, fellBack = [], []
    for mt, encode in ((452, encodeMF1MT452), (455, encodeMF1MT455),
                       (456, encodeMF1MT456)):
        node = nubarNode(suite, mt)
        if node is None:
            continue
        formLabel = label if label in node else EVAL_LABEL
        (written if formLabel == label else fellBack).append(mt)
        section, report = encode(suite, mat, report, label=formLabel)
        sections.append((1, mt, section))

    if label != EVAL_LABEL and (written or fellBack):
        report.warn(
            f"nu-bar written with the {label!r} form where there is one: "
            f"MT {written or 'none'} carry it, MT {fellBack or 'none'} fell back "
            f"to {EVAL_LABEL!r} because they have no {label!r} form"
        )

    # 458 and 460 after the nu-bars, ascending. 458 from the model, falling
    # back to the evaluated node like the nu-bars do; 460 from the records the
    # decoder kept, since it has no model node.
    from ..model_adapter.fission_energy import (encodeMF1MT458, encodeMF1MT460,
                                                fissionEnergyReleaseNode)

    if fissionEnergyReleaseNode(suite, label) is not None:
        section, report = encodeMF1MT458(suite, mat, report, label=label)
        sections.append((1, 458, section))
    section, report = encodeMF1MT460(suite, mat, report)
    if section is not None:
        sections.append((1, 460, section))
    return sections, report


def _mf2Sections(suite, mat, report):
    """MF2/151, when the suite has resonances **and** the provenance to write them."""
    from ..model_adapter import encodeMF2MT151

    resonances = getattr(suite, "resonances", None)
    if resonances is None:
        return [], report

    provenance = getattr(resonances, "provenance", None)
    if provenance is None:
        report.lost(
            "the suite carries resonances but no ENDF provenance for them, so "
            "MF2/151 cannot be written: QX, LRX, LAD and the particle-pair "
            "columns have no model node and would have to be invented"
        )
        return [], report

    section = encodeMF2MT151(resonances, provenance, report)
    # `encodeMF2MT151` returns the section alone; the report it was handed is
    # the one it wrote into.
    return [(2, 151, section)], report



def _mf3Bearing(suite):
    """Every reaction that owns an MF3 section, in tape order.

    ``reactions`` and ``sums`` -- §21.1's ``crossSectionSums`` are reactions in
    every respect the ENDF writer cares about, and sorting by MT rather than
    concatenating the two lists is what keeps the sections in the order a tape
    states them, which is the order the round-trip gate compares.
    """
    # MT851-870 are lumped covariance reactions (ENDF-6 §33.2): GNDS states
    # them as crossSectionSums (NNDC's U-235 has `lump0`, `lump1`), and ENDF
    # has no MF3 section for them -- they exist in MF33 only.
    both = list(suite.reactions) + [s for s in suite.sums
                                    if not (s.ENDF_MT is not None and 851 <= int(s.ENDF_MT) <= 870)]
    return sorted(both, key=lambda r: (r.ENDF_MT is None, r.ENDF_MT or 0))


def _mf3And4And5Sections(suite, mat, report, label=None):
    """MF3 for every reaction with an MT, MF4 and MF5 for what states them.

    **The provenance decides which sections are written, not the model's
    shape.** An `uncorrelated` whose angular half kika *inferred* — the tape
    said MF5 and no MF4 — has no `ltt` in its provenance, and writing an MF4 for
    it would put a section on the tape the source never carried. The same rule
    the MF4 encoder already lives by, applied one level up.
    """
    from kika.nuclear_data.model import EVAL_LABEL, TNSL_INTERACTION

    from ..model_adapter import encodeMF3MT, encodeMF4MT, encodeMF5MT
    from ..model_adapter.angular import mf4Ejectile

    # A thermal-scattering suite states no MF3, MF4 or MF5: its reactions'
    # cross sections and distributions are links to the law, written as MF7.
    if getattr(suite, "interaction", None) == TNSL_INTERACTION:
        return [], report

    label = EVAL_LABEL if label is None else label

    mf3, mf4, mf5 = [], [], []
    written, fellBack = [], []
    # `reactions` and `sums` both, because a tape's MF3 does not distinguish
    # them: §21.1 is a statement about what MT1 *means*, not about whether it is
    # in the file. Reading an evaluation and writing it back has to give the
    # sections back, wherever the model chose to keep them.
    for reaction in _mf3Bearing(suite):
        mt = reaction.ENDF_MT
        if mt is None:
            # §2.4, and it is irreducible: GNDS labels reactions and does not
            # require an MT, so a suite that did not come from ENDF may hold a
            # reaction there is no section number for.
            report.lost(
                f"reaction {reaction.label!r} has no ENDF MT, so it has no MF3 "
                f"section to be written into (gnds_endf_conflicts.md §2.4)"
            )
            continue

        # A reaction the caller did not perturb has no form under the
        # realization's label. Falling back to `eval` keeps the tape complete;
        # refusing would mean a partially perturbed suite could not be written
        # at all, and writing only the perturbed sections would be a tape with
        # holes in it.
        formLabel = label if label in reaction.crossSection else EVAL_LABEL
        (written if formLabel == label else fellBack).append(mt)
        from kika.nuclear_data.model import CrossSectionReconstructed
        try:
            reconstructed = isinstance(suite.styles[formLabel], CrossSectionReconstructed)
        except KeyError:
            reconstructed = False
        section, report = encodeMF3MT(reaction, mat, report, label=formLabel,
            precision="best" if reconstructed else "legacy")
        mf3.append((3, mt, section))

        # MF4 is about the particle the MT names (the p, d, t, He-3 or alpha
        # for MT600-849); MF5 is always about a neutron. For every other MT the
        # two are the same product and this is one lookup.
        for pid in dict.fromkeys((mf4Ejectile(mt), "n")):
            product = _product(reaction, pid)
            form = _evaluatedForm(product, label)
            if form is None and label != EVAL_LABEL:
                form = _evaluatedForm(product, EVAL_LABEL)
            provenance = getattr(product, "provenance", None)
            header = getattr(provenance, "headerFields", None) or {}

            angular = _mf4Form(form)
            if pid == mf4Ejectile(mt) and angular is not None and "ltt" in header:
                section, report = encodeMF4MT(angular, provenance, mt, report)
                mf4.append((4, mt, section))

            if pid == "n" and "mf5" in header:
                section, report = encodeMF5MT(_mf5Form(form), provenance, mt, report)
                mf5.append((5, mt, section))

    if label != EVAL_LABEL:
        # **A mixed tape has to say it is mixed.** The fallback above is right --
        # a partially perturbed suite must still produce a whole tape -- but the
        # file it produces cannot be told apart from a fully perturbed one by
        # reading it, and for an ensemble that distinction is the traceability.
        # So the report states both halves, and states them even when the
        # fallback did not fire: "0 fell back" is a claim, and its absence is not.
        report.warn(
            f"written with the {label!r} form where there is one: "
            f"MT {written or 'none'} carry it, MT {fellBack or 'none'} fell back "
            f"to {EVAL_LABEL!r} because they have no {label!r} form"
        )

    section, report = _mf5DelayedSection(suite, report)
    if section is not None:
        mf5.append((5, 455, section))

    return mf3 + mf4 + mf5, report


def _mf5DelayedSection(suite, report):
    """MF5/MT455, from the bytes kept on ``delayedNeutrons`` -- or ``None``.

    It has no reaction of its own, so the loop above never reaches it: the
    section's home is §18.4's precursor families on the fission channel
    (``fission_energy.attachDelayedSpectra``). The families' spectra go in as
    the edit check, so a spectrum changed in the model is refused rather than
    overwritten by the original.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    from ..model_adapter import encodeMF5MT

    reaction = suite.findReactionByENDF_MT(18)
    data = getattr(getattr(reaction, "outputChannel", None), "fissionFragmentData", None)
    families = getattr(data, "delayedNeutrons", None)
    provenance = getattr(families, "provenance", None)
    if provenance is None:
        return None, report
    forms = []
    for family in families:
        form = _evaluatedForm(getattr(family, "product", None), EVAL_LABEL)
        forms.append(getattr(form, "energy", None))
    energyForm = forms if forms and all(f is not None for f in forms) else None
    return encodeMF5MT(energyForm, provenance, 455, report)


def _product(reaction, pid):
    """The *pid* product of a reaction's output channel, or ``None``."""
    channel = getattr(reaction, "outputChannel", None)
    for product in getattr(channel, "products", None) or ():
        if getattr(product, "pid", None) == pid:
            return product
    return None


def _evaluatedForm(product, label):
    """The evaluated distribution of *product*, when it has one."""
    distribution = getattr(product, "distribution", None)
    if distribution is None:
        return None
    try:
        return distribution[label]
    except (KeyError, TypeError):
        return None


def _mf4Form(form):
    """The part of *form* MF4 states, in the shape ``encodeMF4MT`` takes.

    An `uncorrelated` is one GNDS node built from two ENDF files, so it is
    taken apart here rather than in the encoder — `encodeMF4MT` takes what MF4
    itself carries and would have to learn §18.3 to take anything else.

    **The angular half goes back inside an `angularTwoBody`**, which is not
    ceremony: §18.3 stores the `XYs2d` directly under `<angular>` while §18.2
    wraps it, and the encoder dispatches on the wrapper. An `isotropic2d` needs
    no wrapper — it is a distribution form in its own right.
    """
    from kika.nuclear_data.model import AngularTwoBody, Isotropic2d, Uncorrelated

    if isinstance(form, Uncorrelated):
        if form.angular is None or isinstance(form.angular, Isotropic2d):
            return form.angular
        return AngularTwoBody(angular=form.angular,
                              productFrame=form.productFrame)
    if isinstance(form, (AngularTwoBody, Isotropic2d)):
        return form
    return None


def _mf5Form(form):
    """The part of *form* MF5 states: the energy distribution, or ``None``.

    ``None`` is a real answer and not an absence: a section whose every law
    kika does not model round-trips out of the provenance alone, and
    :func:`~kika.endf.model_adapter.energy.encodeMF5MT` requires being told so.
    """
    from kika.nuclear_data.model import Uncorrelated

    return form.energy if isinstance(form, Uncorrelated) else None



def _mf6Sections(suite, mat, report, label=None):
    """MF6 for every reaction whose provenance carries one.

    *label* selects the §9.1 form of each product, falling back to ``eval``
    product by product, as :func:`_mf3And4And5Sections` does. It used to read
    ``eval`` unconditionally, so a realisation that perturbed a distribution
    stated in File 6 -- JENDL-5's actinide PFNS is LAW=1 there -- came out of
    the whole-tape emitter unperturbed, with nothing said.

    **The provenance decides, and it has to.** An MF6 section is a list of
    products in the evaluator's order with the evaluator's ``ZAP``/``AWP``/
    ``LIP``, its ``JP`` and its ``LCT``, and none of that is recoverable from a
    channel's products: the model holds the physics and the file holds the
    bookkeeping. So a suite that never read an MF6 writes none, and one that
    did writes exactly the section it read.

    Nothing here overlaps :func:`_mf3And4And5Sections`. That one keys on the
    *product's* provenance (``ltt`` for MF4, ``mf5`` for MF5) and this one on
    the *reaction's*, and an MT stated in File 6 does not restate its
    distributions in Files 4 and 5 — except through a negative LAW, which is a
    pointer rather than a duplicate, and which this adapter deliberately leaves
    to those two passes.
    """
    from kika.nuclear_data.model import EVAL_LABEL

    from ..model_adapter import encodeMF6MT

    label = EVAL_LABEL if label is None else label
    sections, carried = [], []
    for reaction in _mf3Bearing(suite):
        provenance = getattr(reaction, "provenance", None)
        header = getattr(provenance, "headerFields", None) or {}
        if "mf6" not in header:
            continue

        mt = reaction.ENDF_MT
        if mt is None:
            report.lost(
                f"reaction {reaction.label!r} carries an MF6 provenance and no "
                f"ENDF MT, so there is no section number to write it under "
                f"(gnds_endf_conflicts.md §2.4)"
            )
            continue

        # Only the products MF6 itself lists. A reaction can carry others --
        # the photons of MF12/MF13 beside an MF6 that does not state them
        # (ENDF/B-VIII.1 S-36 MT22), an excited residual -- and those are
        # written by their own files.
        listed = {record["label"] for record in header["mf6"].get("products", ())
                  if record.get("label") is not None}
        forms = {}
        # The reaction's products, then the products of their decays: MF6 can
        # list a residual's de-excitation photon (FUDGE does), and the model
        # keeps that photon where GNDS puts it, in the residual's decay.
        decays = [d for p in reaction.outputChannel.products
                  for d in (getattr(getattr(p, "outputChannel", None), "products", None) or ())]
        for product in list(reaction.outputChannel.products) + decays:
            if (product.label or product.pid) not in listed or (product.label or product.pid) in forms:
                continue
            form = _evaluatedForm(product, label)
            if form is not None and label != EVAL_LABEL:
                carried.append(mt)
            if form is None and label != EVAL_LABEL:
                form = _evaluatedForm(product, EVAL_LABEL)
            if form is not None:
                forms[product.label or product.pid] = form

        section, report = encodeMF6MT(forms, provenance, mt, report)
        sections.append((6, mt, section))

    if label != EVAL_LABEL and sections:
        report.warn(
            f"MF6 written with the {label!r} form where a product has one: "
            f"MT {sorted(set(carried)) or 'none'} carry it, the rest fell back "
            f"to {EVAL_LABEL!r}"
        )
    return sections, report


def _photonSections(suite, mat, report):
    """MF12-15: from the model where the decoder modelled them, else as kept (roadmap E5b)."""
    from ..model_adapter.photons import encodePhotonSections

    return encodePhotonSections(suite, mat, report)


def _mf7Sections(suite, mat, report):
    """MF7 for a thermal-scattering suite (roadmap E4); nothing for any other."""
    from ..model_adapter.thermal_scattering import encodeMF7Sections

    return encodeMF7Sections(suite, mat, report)


def _covarianceSections(suite, mat, report):
    """MF31/33/34/35, one section per (MF, row MT) the covariance suite carries."""
    from ..model_adapter import (encodeMF31MT, encodeMF32MT, encodeMF33MT,
                                 encodeMF34MT, encodeMF35MT)

    covarianceSuite = getattr(suite, "covarianceSuite", None)
    if covarianceSuite is None:
        return [], report

    encoders = {31: encodeMF31MT, 33: encodeMF33MT,
                34: encodeMF34MT, 35: encodeMF35MT}

    # One pass over the sections to learn which (MF, MT) pairs exist, because
    # the encoders take an MT and there is no listing of them anywhere else.
    # MF32 is absent from `encoders` because it is not a `covarianceSection`;
    # it is written below, from its own container.
    present = set()
    for section in getattr(covarianceSuite, "covarianceSections", ()):
        row = getattr(section, "rowData", None)
        if row is None:
            continue
        mf, mt = getattr(row, "ENDF_MF", None), getattr(row, "ENDF_MT", None)
        if mf is not None and mt is not None:
            present.add((int(mf), int(mt)))

    sections = []
    for mf, mt in sorted(present):
        encode = encoders.get(mf)
        if encode is None:
            report.unsupportedNode(
                f"MF{mf}/MT{mt} is in the covarianceSuite and has no encoder, "
                f"so it is absent from the written tape"
            )
            continue
        section, report = encode(covarianceSuite, mt, mat, report)
        sections.append((mf, mt, section))

    # Lumped covariance components (ENDF-6 §33.2): a reaction that is part of
    # lump MT851-870 has an MF33 section of its own holding only its HEAD, with
    # MTL naming the lump. GNDS states the same membership as the summands of
    # the lump's crossSectionSum (NNDC's U-235: `lump0`, `lump1`), so a suite
    # read from GNDS gets those sections from them.
    sections.extend(_lumpedComponents(suite, mat, present, report))

    # §25.3 lives in its own container, and it is **not** reachable through the
    # loop above -- `parameterCovariances` are not `covarianceSections`, so a
    # tape whose only covariance is MF32 produced an empty `present` and, until
    # the encoder existed, said nothing at all. `encodeMF32MT` writes into the
    # section the decoder kept, and declares the suites it cannot write.
    if getattr(covarianceSuite, "parameterCovariances", None):
        section, report = encodeMF32MT(covarianceSuite, mat, report)
        if section is not None:
            sections.append((32, 151, section))
    return sections, report


def _lumpedComponents(suite, mat, present, report):
    """The MTL-only MF33 sections of the reactions a lump (MT851-870) sums."""
    import re

    from kika.endf.classes.mf33.mf33 import MF33MT

    provenance = getattr(suite, "provenance", None)
    za, awr = getattr(provenance, "za", None), getattr(provenance, "awr", None)
    byLabel = {r.label: r for r in list(suite.reactions) + list(suite.sums)}
    out = []
    for lump in suite.sums:
        if lump.ENDF_MT is None or not 851 <= int(lump.ENDF_MT) <= 870:
            continue
        for summand in getattr(lump, "summands", None) or []:
            href = getattr(summand, "href", "")
            match = re.search(r"\[@label='([^']*)'\]", href)
            member = byLabel.get(match.group(1)) if match else None
            if member is None or member.ENDF_MT is None or (33, int(member.ENDF_MT)) in present:
                continue
            if za is None or awr is None:
                report.lost(f"MF33/MT{member.ENDF_MT}: a lumped component with no ZA/AWR "
                            f"to write its header with")
                continue
            out.append((33, int(member.ENDF_MT),
                        MF33MT(number=int(member.ENDF_MT), _za=float(za), _awr=float(awr),
                               _mtl=int(lump.ENDF_MT), _nl=0,
                               _mat=int(mat if mat is not None else getattr(provenance, "mat", 0) or 0))))
    return out


def encodeTapeSections(suite, mat: Optional[int] = None, report=None, *,
                       label: Optional[str] = None
                       ) -> Tuple[List[Tuple[int, int, object]], object, int]:
    """A :class:`ReactionSuite` → ``[(MF, MT, section), …]`` in tape order.

    Returns the sections, the report, and the MAT they were stamped with. The
    report is the part to read: a model missing a file comes back as a tape
    missing that file, and only the report says so.
    """
    from kika.nuclear_data.model import ConversionReport

    report = report if report is not None else ConversionReport()

    # A suite with no ENDF header to write back -- read from GNDS, or built by
    # hand -- gets its ENDF bookkeeping derived from the model, on a copy so the
    # caller's suite is not changed (`model_adapter/derive`). One read from ENDF
    # is never derived, and writes back what it read.
    from ..model_adapter.derive import deriveEndfProvenance, needsDerivation
    if needsDerivation(suite):
        import copy

        suite = copy.deepcopy(suite)
        report = deriveEndfProvenance(suite, report, mat=mat)
    mat = _mat(suite, mat)

    sections: List[Tuple[int, int, object]] = []
    for build in (_mf1Sections, _mf2Sections, _mf3And4And5Sections,
                  _mf6Sections, _mf7Sections, _photonSections, _covarianceSections):
        if build in (_mf1Sections, _mf3And4And5Sections, _mf6Sections):
            built, report = build(suite, mat, report, label)
        else:
            built, report = build(suite, mat, report)
        sections.extend(built)

    written = {mf for mf, _, _ in sections}
    for mf in sorted(written - set(MF_WRITE_ORDER)):  # pragma: no cover
        raise AssertionError(f"MF{mf} was built and MF_WRITE_ORDER omits it")

    # Ascending (MF, MT) is the tape order, and `sorted` is stable, so the
    # per-file builders above do not have to be.
    sections.sort(key=lambda entry: (entry[0], entry[1]))
    return sections, report, mat


def assembleTape(sections: Sequence[Tuple[int, int, object]], mat: int,
                 tapeId: Optional[str] = None, *,
                 tapeRecord: Optional[str] = None) -> str:
    """``[(MF, MT, section), …]`` → the text of a one-material ENDF tape.

    Each section renders itself, ID columns and trailing SEND included — that is
    what the per-section byte-exact gates test, so this function must not
    re-render anything. What it adds is the record bookkeeping the sections
    cannot know about, because none of them knows what follows it: the FEND
    after the last section of each file, the MEND after the material, and the
    TEND that ends the tape.

    Sequence numbers are **per section** and each section already restarts them
    at 1, which is §0.6.3's rule, so there is nothing to renumber here.
    """
    # ``tapeRecord`` is a whole first line kept from a source tape, written as
    # it was read: the libraries disagree on its ID columns (see
    # ``scan_tape_id``), so rebuilding it would change two of the three.
    lines = [tapeRecord if tapeRecord is not None
             else _tapeIdRecord(tapeId if tapeId is not None else DEFAULT_TAPE_ID)]

    previousMf = None
    for mf, _mt, section in sections:
        if previousMf is not None and mf != previousMf:
            lines.append(format_endf_fend_record(mat))
        lines.append(str(section).rstrip("\n"))
        previousMf = mf

    if previousMf is not None:
        lines.append(format_endf_fend_record(mat))
    lines.append(format_endf_mend_record())
    lines.append(format_endf_tend_record())
    return "\n".join(lines) + "\n"


def writeEndfTape(suite, path, mat: Optional[int] = None,
                  tapeId: Optional[str] = None, report=None, *,
                  label: Optional[str] = None):
    """Write *suite* out as an ENDF-6 tape. Returns the :class:`ConversionReport`.

    The directory is rebuilt **after** the file is on disk, by
    :func:`~kika.endf.writers.update_directory.update_mf1_directory`, and that
    ordering is not incidental: ``NXC`` entries carry ``NC``, a line count, so
    the only place the true counts exist is the written file. ``encodeMF1MT451``
    writes back the directory it read, which is right for a tape whose sections
    have not changed length and wrong the moment one has.

    ``label`` selects which §9.1 form of each cross section and distribution is
    written; the default is ``'eval'``. A ``realization`` (§9.3) drawn beside
    the evaluation is written by naming its label here. Reactions with no form
    under that label fall back to ``'eval'``, so a tape written from a partially
    perturbed suite carries the perturbed sections and the original ones rather
    than only the first.
    """
    from .update_directory import update_mf1_directory

    sections, report, mat = encodeTapeSections(suite, mat, report, label=label)
    if not sections:
        raise ValueError(
            "this reactionSuite produced no ENDF sections at all, so there is "
            "no tape to write; read the ConversionReport for what was refused"
        )

    path = Path(os.fspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    from ..model_adapter.decode import TAPE_ID_KEY

    kept = None
    if tapeId is None:
        header = getattr(getattr(suite, "provenance", None), "headerFields", None) or {}
        kept = header.get(TAPE_ID_KEY)
    path.write_text(assembleTape(sections, mat, tapeId, tapeRecord=kept),
                    newline="\n")

    # Every section written is declared, so the rebuild does not drop one that
    # the source tape's directory happened not to list.
    if not update_mf1_directory(str(path),
                                added_sections={(mf, mt) for mf, mt, _ in sections}):
        report.warn(
            f"the MF1/451 directory of {path.name} could not be rebuilt, so its "
            f"NC line counts are the ones the model was read with and are only "
            f"true if no section changed length"
        )

    if tapeId is None and kept is None:
        report.lost(
            "the tape identification record was written with kika's own label: "
            "this suite kept no first line from a source tape. Pass tapeId= to "
            "set it."
        )
    return report


def writeEndfTapes(suites, path, mats: Optional[Sequence[Optional[int]]] = None,
                   tapeId: Optional[str] = None, *,
                   label: Optional[str] = None) -> list:
    """Several suites → one ENDF-6 tape, one material each (roadmap T2).

    Returns one :class:`ConversionReport` per suite, in the order given.

    **Each material is written by** :func:`writeEndfTape` **and then spliced.**
    The MF1/451 directory can only be rebuilt from a written file (its NC
    entries are line counts), and the rebuild reads one material; so each suite
    is written to its own temporary tape, which gets a correct directory, and
    the tape here is those materials' records between one TPID and one TEND,
    each still closed by its own MEND (§0.6.3). Nothing is re-rendered.

    The materials keep the order given: §0 does not require ascending MAT and
    a caller may have a reason. Two suites with the same MAT are refused,
    because every record of a material is found by that number.

    The TPID is ``tapeId`` if given, else the first suite's kept first line,
    else kika's label (reported on the first suite's report).
    """
    import tempfile
    from ..model_adapter.decode import TAPE_ID_KEY
    from kika.nuclear_data.model import ConversionReport

    suites = list(suites)
    if not suites:
        raise ValueError("no suites given, so there is no tape to write")
    mats = list(mats) if mats is not None else [None] * len(suites)
    if len(mats) != len(suites):
        raise ValueError(f"{len(suites)} suites and {len(mats)} MAT numbers")
    resolved = [_mat(suite, mat) for suite, mat in zip(suites, mats)]
    repeated = sorted({m for m in resolved if resolved.count(m) > 1})
    if repeated:
        raise ValueError(
            f"MAT {repeated} appears more than once; every record of a material "
            f"is found by its MAT, so two materials cannot share one"
        )

    reports = []
    bodies: List[str] = []
    with tempfile.TemporaryDirectory(prefix="kika-tapes-") as scratch:
        for index, (suite, mat) in enumerate(zip(suites, resolved)):
            one = Path(scratch) / f"material{index}.endf"
            # The per-material TPID is thrown away below, so it is named here
            # to keep each report from claiming a label was invented.
            reports.append(writeEndfTape(suite, one, mat=mat,
                                         tapeId=DEFAULT_TAPE_ID,
                                         report=ConversionReport(), label=label))
            lines = one.read_text().rstrip("\n").split("\n")
            # Drop the TPID and the TEND; keep everything through the MEND.
            bodies.extend(lines[1:-1])

    record = None
    if tapeId is None:
        header = getattr(getattr(suites[0], "provenance", None),
                         "headerFields", None) or {}
        record = header.get(TAPE_ID_KEY)
        if record is None:
            reports[0].lost(
                "the tape identification record was written with kika's own "
                "label: the first suite kept no first line from a source tape. "
                "Pass tapeId= to set it."
            )
    first = record if record is not None else _tapeIdRecord(
        tapeId if tapeId is not None else DEFAULT_TAPE_ID)

    path = Path(os.fspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([first, *bodies, format_endf_tend_record()]) + "\n",
                    newline="\n")
    return reports


def writeReconstructedEndfTape(suite, result, path, mat=None, tapeId=None):
    """Write, reload and verify before replacing the requested destination.

    The result must already be attached to the suite. A new derived tape uses
    KIKA's identification label by default. Conversion losses and numerical
    failures abort publication and leave an existing destination intact.
    Only the model's represented ENDF sections can be preserved; inspect the
    source conversion report before preparing a reconstruction.
    """
    import tempfile
    from ..read_endf import read_endf
    from ..model_adapter import decodeReactionSuite

    result.verify_source(suite)
    result.verify_suite(suite)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.'+target.name+'.',
        suffix='.endf', dir=target.parent)
    os.close(descriptor)
    provisional = Path(name)
    try:
        report = writeEndfTape(suite, provisional, mat=mat,
            tapeId=DEFAULT_TAPE_ID if tapeId is None else tapeId, label=result.label)
        if not report.isClean:
            raise ValueError(f'reconstructed ENDF conversion is incomplete: {vars(report)}')
        reloaded, conversion = decodeReactionSuite(read_endf(str(provisional)))
        if not conversion.isCleanFor('cross-sections'):
            raise ValueError(f'reconstructed ENDF reload is incomplete: {vars(conversion)}')
        result.verify_suite(reloaded, label='eval')
        provisional.replace(target)
        return report
    finally:
        provisional.unlink(missing_ok=True)
