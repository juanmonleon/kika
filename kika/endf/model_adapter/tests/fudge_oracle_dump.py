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


def _array(gridded):
    return [float(v) for v in gridded.array.constructArray().ravel()]


def _grid(gridded, index):
    return [float(v) for v in gridded.axes[index].values]


def _tsl(reactionSuite):
    """Each TSL reaction's form, in plain lists (kika's E4 decode builds the same)."""
    out = []
    for reaction in reactionSuite.reactions:
        form = reaction.doubleDifferentialCrossSection.evaluated
        kind = type(form).__module__.rsplit(".", 1)[-1]
        entry = {"kind": kind, "mt": reaction.ENDF_MT}
        if kind == "coherentElastic":
            table = form.S_table.gridded2d
            entry.update(temperatures=_grid(table, 2), energies=_grid(table, 1),
                         values=_array(table))
        elif kind == "incoherentElastic":
            dw = form.DebyeWallerIntegral.function1d
            entry.update(bound=float(form.boundAtomCrossSection.value),
                         temperatures=[float(x) for x, _ in dw],
                         values=[float(y) for _, y in dw])
        else:
            atoms = []
            for atom in form.scatteringAtoms:
                kernel = atom.selfScatteringKernel.kernel
                atoms.append({
                    "numberPerMolecule": int(atom.numberPerMolecule),
                    "mass": float(atom.mass.value),
                    "bound": float(atom.boundAtomCrossSection.value),
                    "e_max": float(atom.e_max.value),
                    "kernel": type(kernel).__name__,
                    "primary": bool(atom.primaryScatterer),
                })
            kernel = form.scatteringAtoms[0].selfScatteringKernel.kernel
            entry.update(atoms=atoms, temperatures=_grid(kernel, 3), betas=_grid(kernel, 2),
                         alphas=_grid(kernel, 1), values=_array(kernel),
                         calculatedAtThermal=bool(form.calculatedAtThermal))
        out.append(entry)
    return out


def _readGnds(text):
    """A GNDS ``reactionSuite`` *text* read by FUDGE (``fudge.reactionSuite.read``)."""
    from fudge import reactionSuite as reactionSuiteModule

    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "suite.xml")
    with open(path, "w") as handle:
        handle.write(text)
    try:
        return reactionSuiteModule.read(path, lazyParsing=False)
    finally:
        os.unlink(path)
        os.rmdir(folder)


def main(tapeText, name):
    from brownies.legacy.converting import endfFileToGNDS

    if name.endswith(".xml"):
        # The other direction (roadmap E4b): a GNDS file kika wrote, read by
        # FUDGE and reported exactly as an ENDF translation would be.
        reactionSuite = _readGnds(tapeText)
        out = {"tsl": _tsl(reactionSuite)}
        if str(reactionSuite.interaction) == "thermalNeutronScatteringLaw":
            # And on to ENDF, which FUDGE can only do with the MAT=…,ZA=… note
            # kika writes on a TSL suite (roadmap E4, GNDS → ENDF).
            import brownies.legacy.toENDF6.toENDF6  # noqa: F401 - attaches toENDF6
            try:
                out["endf"] = reactionSuite.toENDF6("eval", {"verbosity": 0})
            except Exception as error:  # reported, so the GNDS tests still run
                import traceback

                out["endfError"] = f"{type(error).__name__}: {error}"
                out["endfTraceback"] = traceback.format_exc()
        sys.stdout.write("\n" + JSON_MARKER + json.dumps(out) + "\n")
        return

    # FUDGE's TSL converter names the scatterer from the *file name*
    # (ENDF_ITYPE_2.py), so the tape is written under the name it expects.
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, name)
    with open(path, "w") as handle:
        handle.write(tapeText)
    try:
        translated = endfFileToGNDS.endfFileToGNDS(
            path, toStdOut=False, skipBadData=True, doCovariances=False,
            verboseWarnings=False, printBadNK14=False, ignoreBadDate=True,
            reconstructResonances=False)
    finally:
        os.unlink(path)
        os.rmdir(folder)
    reactionSuite = translated["reactionSuite"]
    if str(reactionSuite.interaction) == "thermalNeutronScatteringLaw":
        # FUDGE's own GNDS of the tape goes back too, for kika to read (E4b).
        gnds = "\n".join(reactionSuite.toXML_strList())
        sys.stdout.write("\n" + JSON_MARKER + json.dumps({"tsl": _tsl(reactionSuite),
                                                          "gnds": gnds}) + "\n")
        return

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
    main(TAPE, globals().get("NAME", "tape.endf"))  # noqa: F821 - injected by the caller
