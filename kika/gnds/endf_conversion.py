"""FUDGE's ``applicationData/institution[@label='LLNL']/ENDFconversionFlags``, read and written.

The node is FUDGE's, not kika's: kika reads it so a GNDS file FUDGE wrote can be
turned back into ENDF with what FUDGE knew, and writes it back so a GNDS file
that came in with it goes out with it. An LLNL institution holding anything
else is left to the caller's loss report, untouched.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from kika.nuclear_data.model.endf_conversion import (ENDF_CONVERSION_INSTITUTION,
                                                     EndfConversionFlags)

__all__ = ["readConversionFlags", "writeConversionFlags", "applyResonanceFlags",
           "rMatrixHref"]

_TAG = "ENDFconversionFlags"
_SUITE_HREF = "/reactionSuite"
_NLSC = "LvaluesNeededForConvergence"


def readConversionFlags(applicationData: ET.Element, suite, report):
    """Pull the flags out of *applicationData* into the suite's model.

    Returns the ``applicationData`` the other readers should see: the same
    element when there were no flags, otherwise a copy without the institution
    that held them.
    """
    keep = []
    found = None
    for institution in applicationData:
        children = list(institution)
        if (institution.get("label") == ENDF_CONVERSION_INSTITUTION and children
                and all(child.tag == _TAG for child in children)):
            for child in children:
                found = found or EndfConversionFlags()
                for conversion in child:
                    if conversion.tag != "conversion":
                        report.lost(f"{_TAG}/{conversion.tag} is not modelled")
                        continue
                    found.conversions.append((conversion.get("href", ""),
                                              conversion.get("flags", "")))
            continue
        keep.append(institution)
    if found is None:
        return applicationData
    suite.applicationData.entries.append(found)
    rest = ET.Element(applicationData.tag, applicationData.attrib)
    rest.extend(keep)
    return rest


def _principalZA(suite) -> Optional[int]:
    """The ZA FUDGE's note names the scatterer by, when the evaluation states one.

    FUDGE takes it from a table keyed by the *file name* (``ENDF_ITYPE_2.py``:
    the principal isotope, or the most abundant one -- W184, Pb208). kika has
    no file name to go by, so it reads what the evaluation says, in order:

    1. the principal scattering atom of MF7/MT4, when it is a nuclide;
    2. the most abundant nuclide of the first element of ``targetInfo``
       (MF7/MT451 lists the principal element first);
    3. the tape's own header ZA, when it is a real one and not the MAT + 100
       pseudo-ZA (JEFF-4.0 writes Be metal as 4000, the element).

    FUDGE's ``toENDF6`` only looks a mass up by this number and then replaces
    it with the target's, so a real ZA is all it needs. ``None`` when nothing
    names one: kika does not guess a nuclide from an atom's mass.
    """
    from kika.nuclear_data.model import (EVAL_LABEL, Evaluated, IncoherentInelastic,
                                         Nuclide)
    from kika.nuclear_data.model.pops import zaFromPid

    for reaction in suite.reactions:
        ddcs = reaction.doubleDifferentialCrossSection
        form = ddcs.get(EVAL_LABEL) if ddcs is not None else None
        if isinstance(form, IncoherentInelastic) and form.principal is not None:
            particle = suite.PoPs.particles.get(form.principal.pid)
            if isinstance(particle, Nuclide):
                return zaFromPid(particle.id)
    style = next((s for s in suite.styles if isinstance(s, Evaluated)), None)
    info = getattr(style, "targetInfo", None)
    if info is not None and info.chemicalElements and info.chemicalElements[0].nuclides:
        nuclides = info.chemicalElements[0].nuclides
        return zaFromPid(max(nuclides, key=lambda n: n.atomFraction or 0.0).pid)
    provenance = getattr(suite, "provenance", None)
    za, mat = getattr(provenance, "za", None), getattr(provenance, "mat", None)
    if za is not None and mat is not None and int(za) >= 1000 and int(za) != int(mat) + 100:
        return int(za)
    return None


def _tslFlags(suite) -> Optional[EndfConversionFlags]:
    """FUDGE's ``MAT=…,ZA=…`` note on ``/reactionSuite``, for a TSL suite read from ENDF.

    A thermal-scattering target is an ``unorthodox`` particle (``tnsl-…``), so
    nothing in GNDS spells its MAT and no table has one. FUDGE writes this note
    on every TSL evaluation it converts (``ENDF_ITYPE_2.py``) and cannot write
    one back to ENDF without it. **Its ZA is the principal scatterer's** (1001
    for s-CH4, 4009 for Be metal), which FUDGE looks a mass up by -- not the
    MAT + 100 pseudo-ZA of the tape's headers. kika writes it when the evaluation
    names one (:func:`_principalZA`), and ``MAT=`` alone otherwise.
    """
    from kika.nuclear_data.model.thermal_scattering import TNSL_INTERACTION

    if getattr(suite, "interaction", None) != TNSL_INTERACTION:
        return None
    mat = getattr(getattr(suite, "provenance", None), "mat", None)
    if mat is None:
        return None
    za = _principalZA(suite)
    text = f"MAT={int(mat)}" + (f",ZA={za}" if za is not None else "")
    return EndfConversionFlags([(_SUITE_HREF, text)])


def rMatrixHref(suite, index: int) -> str:
    """The xPath of the *index*-th resolved region's ``RMatrix``, as FUDGE names
    a single one; kika writes several ranges as sibling ``resolved`` nodes."""
    regions = suite.resonances.resolved
    label = regions[index].formalism.label or "eval"
    where = "resolved" if len(regions) == 1 else f"resolved[{index + 1}]"
    return f"/reactionSuite/resonances/{where}/RMatrix[@label='{label}']"


def _reichMoore(suite):
    from kika.nuclear_data.model.resonances import RMatrix

    resonances = getattr(suite, "resonances", None)
    for index, region in enumerate(getattr(resonances, "resolved", None) or ()):
        formalism = region.formalism
        if isinstance(formalism, RMatrix) and formalism.approximation == "ReichMoore":
            yield rMatrixHref(suite, index), formalism


def _withResonanceFlags(suite, conversions):
    """FUDGE's note on a Reich-Moore range: ``LRF3`` for one read from ENDF LRF=3
    and NLSC as ``LvaluesNeededForConvergence=`` (the model's ``angularLCount``)."""
    from kika.nuclear_data.model.endf_conversion import parseFlags

    conversions = list(conversions)
    for href, formalism in _reichMoore(suite):
        nlsc = formalism.angularLCount or None
        positions = [k for k, (target, _) in enumerate(conversions) if target == href]
        if not positions:
            items = ["LRF3"] if any(group.spins for group in formalism.spinGroups) else []
            if nlsc:
                items.append(f"{_NLSC}={nlsc}")
            if items:
                conversions.append((href, ",".join(items)))
            continue
        for k in positions:
            flags = parseFlags(conversions[k][1])
            if _NLSC in flags or (nlsc and k == positions[0]):
                flags[_NLSC] = str(nlsc) if nlsc else None
                items = [key if value is None else f"{key}={value}"
                         for key, value in flags.items()
                         if not (key == _NLSC and value is None)]
                conversions[k] = (href, ",".join(items))
    return _withUnresolvedFlags(suite, [(href, text) for href, text in conversions if text])


def _withUnresolvedFlags(suite, conversions):
    """FUDGE's note on an unresolved range read from ENDF: ``LRF1`` for case A or B,
    and ``LFW<n>`` where the tape's LFW is not what fission in the suite implies."""
    resonances = getattr(suite, "resonances", None)
    widths = getattr(getattr(resonances, "unresolved", None), "tabulatedWidths", None)
    header = getattr(getattr(resonances, "provenance", None), "headerFields", None) or {}
    region = next((r for r in header.get("regions") or () if r.get("kind") == "unresolved"), None)
    if widths is None or region is None:
        return conversions
    href = f"/reactionSuite/resonances/unresolved/tabulatedWidths[@label='{widths.label or 'eval'}']"
    if any(target == href for target, _ in conversions):
        return conversions
    fission = (any(getattr(r, "ENDF_MT", None) in (18, 19, 20, 21, 38) for r in suite.reactions)
               or bool(len(getattr(suite, "fissionComponents", ()) or ())))
    lfw = header["isotopes"][region["isotope_index"]].get("lfw")
    items = ["LRF1"] if region.get("lrf") == 1 else []
    if lfw is not None and lfw != int(fission):
        items.append(f"LFW{lfw}")
    return conversions + ([(href, ",".join(items))] if items else [])


def applyResonanceFlags(suite) -> None:
    """NLSC back onto each Reich-Moore range whose ``RMatrix`` states none."""
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    flags = EndfConversionFlags.of(suite)
    if not flags:
        return
    for href, formalism in _reichMoore(suite):
        value = flags.flagsFor(href).get(_NLSC)
        if value is not None and formalism.angularLCount is None:
            formalism.angularLCount = int(value)


def writeConversionFlags(root: ET.Element, suite, written=()) -> None:
    """Append the suite's flags, then *written* (the writer's own), to *root*'s ``applicationData``.

    A TSL suite read from ENDF has no flags of its own; it gets FUDGE's
    ``MAT=…,ZA=…`` note (:func:`_tslFlags`), unless its flags already state a MAT.
    """
    flags = EndfConversionFlags.of(suite)
    if not flags or "MAT" not in flags.flagsFor(_SUITE_HREF):
        synthesised = _tslFlags(suite)
        if synthesised is not None:
            flags = EndfConversionFlags(synthesised.conversions
                                        + (list(flags.conversions) if flags else []))
    flags = EndfConversionFlags(_withResonanceFlags(suite, flags.conversions if flags else []))
    if written:
        flags = EndfConversionFlags(list(flags.conversions) + list(written))
    if not flags:
        return
    application = root.find("applicationData")
    if application is None:
        application = ET.SubElement(root, "applicationData")
    institution = ET.SubElement(application, "institution",
                                label=ENDF_CONVERSION_INSTITUTION)
    node = ET.SubElement(institution, _TAG)
    for href, text in flags.conversions:
        ET.SubElement(node, "conversion", flags=text, href=href)
