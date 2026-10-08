"""Run **under FUDGE's interpreter**, not kika's: ENDF tape → JSON of what FUDGE reads.

The other half of ``test_fudge_in_the_loop.py`` (ENDF-coverage roadmap T5).
It imports nothing from kika and kika imports nothing from it: FUDGE stays out
of kika's environment entirely, which is the rule (``gnds_roadmap.md``: FUDGE
generates fixtures and is never a dependency at run time). The test sends this
file's source and the tape's text on stdin, separated by ``TAPE_MARKER``, and
reads one JSON line back, prefixed by ``JSON_MARKER`` because FUDGE prints its
own progress to stdout.

What it reports is what kika's ENDF decoder also builds, in plain lists:

* ``crossSections`` — per MT, the evaluated pointwise table; a
  ``resonancesWithBackground`` is given as its background regions joined in
  domain order (FUDGE's translation splits MF3 at the resolved-region top);
* ``legendre`` — per MT, the outgoing neutron's Legendre coefficients per
  incident energy, ``a0`` included, across every Legendre region;
* ``resonances`` — every resolved resonance energy, sorted.
"""
import json
import os
import sys
import tempfile

# Spelt in two halves so this file's own source never contains the marker the
# caller splits stdin on.
TAPE_MARKER = "#---KIKA-" + "TAPE---"
JSON_MARKER = "KIKA-ORACLE-JSON:"


def _pairs(xys):
    return [[float(x) for x, _ in xys], [float(y) for _, y in xys]]


def _crossSection(evaluated):
    background = getattr(evaluated, "background", None)
    if background is not None:
        xs, ys = [], []
        for name in ("resolvedRegion", "unresolvedRegion", "fastRegion"):
            region = getattr(background, name, None)
            if region is None or getattr(region, "data", None) is None:
                continue
            x, y = _pairs(region.data)
            xs.extend(x)
            ys.extend(y)
        return [xs, ys]
    if hasattr(evaluated, "toPointwise_withLinearXYs") and hasattr(evaluated, "__iter__"):
        try:
            return _pairs(evaluated)
        except (TypeError, ValueError):
            return None
    return None


def _legendre(reaction):
    for product in reaction.outputChannel.products:
        if product.pid != "n":
            continue
        evaluated = product.distribution.evaluated
        angular = getattr(evaluated, "angularSubform", None)
        if angular is None:
            return None
        regions = list(angular) if type(angular).__name__ == "Regions2d" else [angular]
        rows = []
        for region in regions:
            for function in region:
                if type(function).__name__ != "Legendre":
                    break
                rows.append([float(function.outerDomainValue),
                             [float(c) for c in function.coefficients]])
        return rows
    return None


def _resonanceEnergies(reactionSuite):
    resonances = getattr(reactionSuite, "resonances", None)
    resolved = getattr(resonances, "resolved", None) if resonances else None
    if resolved is None:
        return None
    energies = []
    formalism = resolved.evaluated
    regions = getattr(resolved, "regions", None) or [formalism]
    for region in regions:
        form = getattr(region, "evaluated", region)
        groups = getattr(form, "spinGroups", None)
        if groups is not None:
            for group in groups:
                energies.extend(float(e) for e in group.resonanceParameters.table.getColumn("energy"))
        else:
            table = form.resonanceParameters.table
            energies.extend(float(e) for e in table.getColumn("energy"))
    return sorted(energies)


def main(tapeText):
    from brownies.legacy.converting import endfFileToGNDS

    with tempfile.NamedTemporaryFile("w", suffix=".endf", delete=False) as handle:
        handle.write(tapeText)
        path = handle.name
    try:
        translated = endfFileToGNDS.endfFileToGNDS(
            path, toStdOut=False, skipBadData=True, doCovariances=False,
            verboseWarnings=False, printBadNK14=False, ignoreBadDate=True,
            reconstructResonances=False)
    finally:
        os.unlink(path)
    reactionSuite = translated["reactionSuite"]

    out = {"crossSections": {}, "legendre": {}, "resonances": None}
    for reaction in reactionSuite.reactions:
        mt = reaction.ENDF_MT
        if mt is None:
            continue
        pair = _crossSection(reaction.crossSection.evaluated)
        if pair is not None:
            out["crossSections"][str(mt)] = pair
        rows = _legendre(reaction)
        if rows:
            out["legendre"][str(mt)] = rows
    out["resonances"] = _resonanceEnergies(reactionSuite)
    sys.stdout.write("\n" + JSON_MARKER + json.dumps(out) + "\n")


if __name__ == "oracle":
    main(TAPE)  # noqa: F821 - injected by the caller's exec
