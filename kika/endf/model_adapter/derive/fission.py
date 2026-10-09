"""G4a: MF1/452, 455, 456 and 458 bookkeeping from the model.

The nu-bars sit where :func:`~kika.endf.model_adapter.multiplicity.attachNubar`
puts them and where NNDC's GNDS has them: the fission neutron's multiplicity
(prompt when a total is summed, total otherwise) and the ``multiplicitySum``
nodes labelled with ``ENDF_MT`` 452 and 455. A nu-bar's MF1 header is

- **LNU** 1 for a ``polynomial1d``, 2 for a table;
- **LDG** 0 for MT455: §18.4's ``rate`` is a scalar, so a delayed nu-bar read
  from GNDS has energy-independent decay constants by construction;
- the TAB1's interpolation regions, the form's own.

The fission energy release (MF1/458) needs no header beyond MAT/ZA/AWR: LFC and
NPLY follow from its terms, which the encoder already reads off the model; LDRV
of a tabulated term, which the model does not carry, is written as the
format's default and reported by the encoder.
"""
from __future__ import annotations

from . import DERIVED, DerivationContext, register

__all__ = ["deriveMultiplicity", "deriveFissionEnergyRelease"]

#: The nu-bar MTs, in the order they sit on a tape.
_NUBAR = (452, 455, 456)


def _nubars(context: DerivationContext) -> dict:
    """``{id(multiplicity): MT}`` for the suite's nu-bars. Built once per pass."""
    found = getattr(context, "_nubarNodes", None)
    if found is not None:
        return found
    from kika.nuclear_data.model import EVAL_LABEL  # noqa: F401 - model import first

    found = {}
    suite = context.suite
    summed = {int(s.ENDF_MT): s.multiplicity for s in suite.sums.multiplicitySums
              if s.ENDF_MT is not None and int(s.ENDF_MT) in _NUBAR
              and s.multiplicity is not None}
    for mt, multiplicity in summed.items():
        found[id(multiplicity)] = mt
    fission = suite.findReactionByENDF_MT(18)
    if fission is not None:
        neutron = next((p for p in fission.outputChannel.products
                        if p.pid == "n" and p.multiplicity is not None
                        and p.multiplicity.isEvaluable), None)
        if neutron is not None:
            found[id(neutron.multiplicity)] = 456 if 452 in summed else 452
    context._nubarNodes = found
    return found


def deriveMultiplicity(multiplicity, path, context: DerivationContext, report):
    """MF1's provenance for a nu-bar, or ``None`` for any other multiplicity."""
    from kika.nuclear_data.model import EndfProvenance, Polynomial1d

    mt = _nubars(context).get(id(multiplicity))
    if mt is None:
        return None
    form = multiplicity.form
    header = {"lnu": 1 if isinstance(form, Polynomial1d) else 2}
    regions = []
    if header["lnu"] == 2:
        if not hasattr(form, "toEndfRegions"):
            report.lost(f"MF1/MT{mt}: the nu-bar is a {type(form).__name__}, which "
                        f"is neither a table nor a polynomial; MF1/{mt} is not written")
            return None
        _, _, regions = form.toEndfRegions()
    if mt == 455:
        header["ldg"] = 0
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, headerFields=header,
                          interpolationRegions=[tuple(pair) for pair in regions])


def deriveFissionEnergyRelease(node, path, context: DerivationContext, report):
    """LFC, NPLY and, for a tabulated term, its IFC order and TAB1 regions.

    LFC is 1 when any term is a table; NPLY is the polynomials' degree (0 with
    LFC=1, whose LIST is the constant shape). The tables go in the nine-term
    order of ``FissionEnergyRelease.TERMS`` -- IFC ascending, which is how the
    three LFC=1 evaluations of ENDF/B-VIII.1 write them. LDRV follows FUDGE's
    rule (2 for promptProductKE, 1 otherwise). What the model does not carry is
    left out and the encoder reports it: the thermal value-and-σ the LIST
    repeats beside a tabulated term.
    """
    from kika.nuclear_data.model import EndfProvenance, FissionEnergyRelease, Polynomial1d

    terms = {name: getattr(node, name) for name in FissionEnergyRelease.TERMS}
    tabulated = [name for name in FissionEnergyRelease.TERMS
                 if terms[name] is not None and not isinstance(terms[name], Polynomial1d)]
    orders = max([len(f.coefficients) for f in terms.values()
                  if isinstance(f, Polynomial1d)] or [1])
    header = {"lfc": 1 if tabulated else 0, "nply": 0 if tabulated else orders - 1}
    if tabulated:
        regions = {}
        for name in tabulated:
            _, _, pairs = terms[name].toEndfRegions()
            regions[name] = [list(pair) for pair in pairs]
        # LDRV is not in the model. FUDGE writes 2 for the fragments' kinetic
        # energy and 1 for the rest (its own FIXME, fissionEnergyReleased.py:52),
        # and that is what all three LFC=1 evaluations of ENDF/B-VIII.1
        # (U-235, U-238, Pu-239) state, measured 2026-10-09.
        ldrv = {name: 2 if name == "promptProductKE" else 1 for name in tabulated}
        header.update(order=tabulated, regions=regions, unplaced=[], ldrv=ldrv)
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, headerFields=header)


def _isMultiplicity(node) -> bool:
    from kika.nuclear_data.model.output_channel import Multiplicity

    return isinstance(node, Multiplicity)


def _isFissionEnergyRelease(node) -> bool:
    from kika.nuclear_data.model import FissionEnergyRelease

    return isinstance(node, FissionEnergyRelease)


register("multiplicities", _isMultiplicity, deriveMultiplicity)
register("fissionEnergyRelease", _isFissionEnergyRelease, deriveFissionEnergyRelease)
