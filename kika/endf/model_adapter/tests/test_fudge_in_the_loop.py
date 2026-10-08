"""kika's ENDF decode against FUDGE's, with FUDGE actually run (roadmap T5).

The three ``kika/gnds/tests/test_*_oracle.py`` compare kika with kika on a file
FUDGE translated offline, so they cannot be cited as "validated against FUDGE".
This one can: it runs FUDGE's ``endfFileToGNDS`` on the same committed
micro-tape kika decodes and compares what each reads.

**FUDGE never enters kika's environment.** ``KIKA_FUDGE_PYTHON`` names an
interpreter that has FUDGE (on Juan's machines, the course venv in WSL)::

    KIKA_FUDGE_PYTHON="wsl.exe -d Ubuntu -e /home/<user>/FUDGE_class/.venv/bin/python"
    pytest -m fudge kika/endf/model_adapter/tests/test_fudge_in_the_loop.py

The value is split on whitespace, so the interpreter path must not contain
spaces. ``fudge_oracle_dump.py`` is sent to it on stdin with the tape's text,
so no path has to be translated between Windows and WSL. Without the variable
every test here is skipped, and with the default markers it is deselected.

What is compared, and to what tolerance. Both sides read the same decimal text
into IEEE doubles, so the target is the last bit (``RTOL``), not a physics
tolerance:

* every MT's evaluated cross-section table, a resonance background joined over
  its regions as in ``test_cross_section_oracle``;
* the outgoing neutron's Legendre coefficients per incident energy;
* every resolved resonance energy.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import Legendre, Regions1d, Regions2d, XYs1d

pytestmark = pytest.mark.fudge

RTOL = 1e-14
_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parents[1] / "tests" / "data"
_TAPES = ("micro_fe56_xs_and_angular.endf", "micro_fe56_structural.endf")

TAPE_MARKER = "#---KIKA-" + "TAPE---"
JSON_MARKER = "KIKA-ORACLE-JSON:"
_RUNNER = ("import sys;s=sys.stdin.read();src,_,tape=s.partition('"
           + TAPE_MARKER + "');exec(src,{'TAPE':tape,'__name__':'oracle'})")


@pytest.fixture(scope="module")
def fudgePython():
    command = os.environ.get("KIKA_FUDGE_PYTHON", "").split()
    if not command:
        pytest.skip("KIKA_FUDGE_PYTHON is not set: no interpreter with FUDGE")
    return command


def _runFudge(command, tape: Path, name: str = "tape.endf") -> dict:
    """FUDGE's reading of *tape*, written for it under the file *name*.

    The name matters for a TSL tape only: FUDGE's converter names the
    scatterer from it (``ENDF_ITYPE_2.py``) and refuses a name it does not know.
    """
    source = (_HERE / "fudge_oracle_dump.py").read_text()
    stdin = f"NAME = {name!r}\n" + source + "\n" + TAPE_MARKER + tape.read_text()
    environment = dict(os.environ, MSYS_NO_PATHCONV="1")
    done = subprocess.run(command + ["-c", _RUNNER], input=stdin,
                          capture_output=True, text=True, timeout=900,
                          env=environment)
    lines = [l for l in done.stdout.replace("\0", "").splitlines()
             if l.startswith(JSON_MARKER)]
    if done.returncode != 0 or not lines:
        pytest.fail(f"FUDGE did not answer (exit {done.returncode}):\n"
                    f"{done.stderr[-2000:]}\n{done.stdout[-2000:]}")
    return json.loads(lines[-1][len(JSON_MARKER):])


@pytest.fixture(scope="module", params=_TAPES)
def pair(request, fudgePython):
    tape = _DATA / request.param
    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    return suite, _runFudge(fudgePython, tape)


def _collapse(xs, ys):
    """Drop a point that repeats the previous one in both coordinates (see the GNDS oracle)."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    keep = np.ones(xs.size, dtype=bool)
    keep[1:] = ~((xs[1:] == xs[:-1]) & (ys[1:] == ys[:-1]))
    return xs[keep], ys[keep]


def _kikaPointwise(form):
    if isinstance(form, XYs1d):
        return form.xs, form.ys
    if isinstance(form, Regions1d):
        xs, ys, _ = form.toEndfRegions()
        return xs, ys
    return None


def _kikaLegendre(reaction):
    for product in reaction.outputChannel.products:
        if (product.label or product.pid) != "n":
            continue
        try:
            form = product.distribution["eval"]
        except (KeyError, TypeError):
            return None
        angular = getattr(form, "angular", None)
        if angular is None:
            return None
        regions = list(angular) if isinstance(angular, Regions2d) else [angular]
        rows = []
        for region in regions:
            for function in region:
                if not isinstance(function, Legendre):
                    break
                rows.append((float(function.outerDomainValue),
                             np.asarray(function.coefficients, dtype=float)))
        return rows
    return None


def test_every_cross_section_reads_the_same(pair):
    suite, fudge = pair
    compared = 0
    for mtText, (fx, fy) in sorted(fudge["crossSections"].items()):
        reaction = suite.findReactionByENDF_MT(int(mtText))
        if reaction is None:
            continue
        kika = _kikaPointwise(reaction.crossSection.forms.get("eval"))
        assert kika is not None, f"MT{mtText}: kika has no pointwise form"
        kx, ky = _collapse(*kika)
        fx, fy = _collapse(fx, fy)
        assert kx.shape == fx.shape, f"MT{mtText}: {kx.size} points against {fx.size}"
        np.testing.assert_allclose(kx, fx, rtol=RTOL, atol=0, err_msg=f"MT{mtText} E")
        np.testing.assert_allclose(ky, fy, rtol=RTOL, atol=0, err_msg=f"MT{mtText} sigma")
        compared += 1
    assert compared, "no shared MT was compared"


def _collapseLegendre(rows):
    """Drop an incident energy that repeats the previous one with the same coefficients.

    The Legendre twin of :func:`_collapse`, and strict in the same way: a
    repeated energy with *different* coefficients is a discontinuity and stays.
    """
    kept = []
    for energy, coefficients in rows:
        if kept and kept[-1][0] == energy and np.array_equal(kept[-1][1], coefficients):
            continue
        kept.append((energy, coefficients))
    return kept


def test_the_legendre_coefficients_read_the_same(pair):
    suite, fudge = pair
    compared = 0
    for mtText, rows in sorted(fudge["legendre"].items()):
        reaction = suite.findReactionByENDF_MT(int(mtText))
        kika = _kikaLegendre(reaction) if reaction is not None else None
        assert kika is not None, f"MT{mtText}: FUDGE reads Legendre, kika does not"
        kika = _collapseLegendre(kika)
        assert len(kika) == len(rows), f"MT{mtText}: {len(kika)} energies against {len(rows)}"
        for (energy, coefficients), (fEnergy, fCoefficients) in zip(kika, rows):
            assert energy == pytest.approx(fEnergy, rel=RTOL)
            np.testing.assert_allclose(coefficients, fCoefficients, rtol=RTOL,
                                       atol=0, err_msg=f"MT{mtText} at {energy} eV")
        compared += 1
    assert compared, "no Legendre distribution was compared"


def test_the_one_legendre_difference_is_a_redundant_repeat_kika_keeps(pair):
    """Name the difference rather than leave it inside a helper.

    JEFF-4.0 Fe-56's MF4/MT2 writes the incident energy 3.905 MeV twice with
    the same coefficients (records 7831 and 7833 of the micro-tape). FUDGE's
    translation drops the second, as it drops a redundant cross-section point;
    kika keeps it, because the reader reproduces the tape. Found by this test
    the first time FUDGE ran in the loop (2026-10-08). If either side changes,
    this fails and says which.
    """
    suite, fudge = pair
    kika = _kikaLegendre(suite.findReactionByENDF_MT(2))
    repeated = sorted({e for i, (e, _c) in enumerate(kika) if i and kika[i - 1][0] == e})
    assert repeated == [3905000.0]
    fudgeEnergies = [row[0] for row in fudge["legendre"]["2"]]
    assert len(fudgeEnergies) == len(set(fudgeEnergies))


def test_the_resonance_energies_read_the_same(pair):
    suite, fudge = pair
    if fudge["resonances"] is None:
        pytest.skip("FUDGE reads no resolved region on this tape")
    energies = []
    for region in suite.resonances.resolved:
        for group in region.formalism.spinGroups:
            energies.extend(float(e) for e in group.energies)
    np.testing.assert_allclose(sorted(energies), fudge["resonances"], rtol=RTOL, atol=0)


# ----------------------------------------------------------------------
# The thermal scattering law (roadmap E4)
# ----------------------------------------------------------------------

#: The committed TSL micro-tapes, and the evaluation file name FUDGE knows each by.
_TSL = {"sch4": "tsl-s-CH4.endf", "un_elastic": "tsl-NinUN.endf",
        "bemetal_elastic": "tsl-Be-metal.endf", "jeff_be_elastic": "tsl-Be-metal.endf"}

#: kika's NEUTRON_MASS_AMU is CODATA 2018 and FUDGE's is older; masses in amu
#: agree to ~2e-10, and that is the constant, not the reading.
MASS_RTOL = 1e-9


@pytest.fixture(scope="module", params=sorted(_TSL))
def tslPair(request, fudgePython):
    from kika.nuclear_data.model import EVAL_LABEL

    tape = _DATA / f"micro_tsl_{request.param}.endf"
    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    forms = [r.doubleDifferentialCrossSection[EVAL_LABEL] for r in suite.reactions]
    return forms, _runFudge(fudgePython, tape, _TSL[request.param])["tsl"]


def test_the_tsl_forms_are_the_same_forms(tslPair):
    forms, fudge = tslPair
    kinds = {"CoherentElastic": "coherentElastic", "IncoherentElastic": "incoherentElastic",
             "IncoherentInelastic": "incoherentInelastic"}
    assert [kinds[type(f).__name__] for f in forms] == [entry["kind"] for entry in fudge]


def test_the_tsl_tables_read_the_same(tslPair):
    from kika.nuclear_data.model import (CoherentElastic, IncoherentElastic,
                                         IncoherentInelastic)

    forms, fudge = tslPair
    for form, entry in zip(forms, fudge):
        if isinstance(form, CoherentElastic):
            temperature, energy = form.S_table.grids
            np.testing.assert_allclose(temperature.values, entry["temperatures"], rtol=RTOL)
            np.testing.assert_allclose(energy.values, entry["energies"], rtol=RTOL)
            np.testing.assert_allclose(form.S_table.values.ravel(), entry["values"], rtol=RTOL)
        elif isinstance(form, IncoherentElastic):
            assert form.boundAtomCrossSection.value == pytest.approx(entry["bound"], rel=RTOL)
            # A repeated identical point is kept by kika and dropped by FUDGE
            # (ENDF_ITYPE_2.py:357); see the test naming it below.
            xs, ys = _collapse(*form.DebyeWallerIntegral.toEndfRegions()[:2])
            np.testing.assert_allclose(xs, entry["temperatures"], rtol=RTOL)
            np.testing.assert_allclose(ys, entry["values"], rtol=RTOL)
        elif isinstance(form, IncoherentInelastic):
            kernel = form.principal.selfScatteringKernel.kernel
            temperature, beta, alpha = kernel.grids
            np.testing.assert_allclose(temperature.values, entry["temperatures"], rtol=RTOL)
            np.testing.assert_allclose(beta.values, entry["betas"], rtol=RTOL)
            np.testing.assert_allclose(alpha.values, entry["alphas"], rtol=RTOL)
            np.testing.assert_allclose(kernel.values.ravel(), entry["values"], rtol=RTOL)
            assert form.calculatedAtThermal == entry["calculatedAtThermal"]


def test_the_scattering_atoms_read_the_same(tslPair):
    from kika.nuclear_data.model import IncoherentInelastic

    forms, fudge = tslPair
    for form, entry in zip(forms, fudge):
        if not isinstance(form, IncoherentInelastic):
            continue
        assert len(form.scatteringAtoms) == len(entry["atoms"])
        for atom, theirs in zip(form.scatteringAtoms, entry["atoms"]):
            assert atom.numberPerMolecule == theirs["numberPerMolecule"]
            assert atom.primaryScatterer == theirs["primary"]
            assert atom.boundAtomCrossSection.value == pytest.approx(theirs["bound"], rel=RTOL)
            assert atom.mass.value == pytest.approx(theirs["mass"], rel=MASS_RTOL)
            assert atom.e_max.value == pytest.approx(theirs["e_max"], rel=RTOL, abs=0.0)
            assert type(atom.selfScatteringKernel.kernel).__name__ == theirs["kernel"]


def test_the_one_tsl_difference_is_a_repeated_point_kika_keeps(fudgePython):
    """Name the TSL difference rather than leave it inside a helper.

    ENDF/B-VIII.1 s-CH4's MF7/MT2 tabulates W'(T) as two identical points at
    22 K (a one-temperature evaluation written as a two-point TAB1). FUDGE
    drops the second (ENDF_ITYPE_2.py:357); kika keeps it, because the reader
    reproduces the tape. Found the first time the TSL oracle ran (2026-10-08).
    """
    from kika.nuclear_data.model import EVAL_LABEL, IncoherentElastic

    tape = _DATA / "micro_tsl_sch4.endf"
    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    form = next(r.doubleDifferentialCrossSection[EVAL_LABEL] for r in suite.reactions
                if isinstance(r.doubleDifferentialCrossSection[EVAL_LABEL], IncoherentElastic))
    xs, _ys, _ = form.DebyeWallerIntegral.toEndfRegions()
    assert list(xs) == [22.0, 22.0]
    fudge = _runFudge(fudgePython, tape, _TSL["sch4"])["tsl"]
    assert next(e for e in fudge if e["kind"] == "incoherentElastic")["temperatures"] == [22.0]


# ----------------------------------------------------------------------
# The thermal scattering law through GNDS (roadmap E4b), both ways
# ----------------------------------------------------------------------

@pytest.fixture(scope="module", params=sorted(_TSL))
def tslGnds(request, fudgePython, tmp_path_factory):
    """kika's ENDF decode, FUDGE's reading of the tape, FUDGE's reading of the
    GNDS kika wrote from it, and kika's reading of the GNDS FUDGE wrote."""
    import kika

    key = request.param
    tape = _DATA / f"micro_tsl_{key}.endf"
    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    fromEndf = _runFudge(fudgePython, tape, _TSL[key])
    folder = tmp_path_factory.mktemp(f"tsl_{key}")
    ours = folder / "kika.xml"
    kika.write(suite, ours)
    fromKika = _runFudge(fudgePython, ours, "suite.xml")
    theirs = folder / "fudge.xml"
    theirs.write_text(fromEndf["gnds"])
    return suite, fromEndf["tsl"], fromKika["tsl"], kika.read(theirs, covariances=False)


def _assertSameDump(ours, theirs, path="tsl"):
    if isinstance(ours, dict):
        assert set(ours) == set(theirs), path
        for key in ours:
            _assertSameDump(ours[key], theirs[key], f"{path}.{key}")
    elif isinstance(ours, list) and ours and isinstance(ours[0], dict):
        assert len(ours) == len(theirs), path
        for index, (a, b) in enumerate(zip(ours, theirs)):
            _assertSameDump(a, b, f"{path}[{index}]")
    elif isinstance(ours, list):
        np.testing.assert_allclose(ours, theirs, rtol=RTOL, atol=0, err_msg=path)
    elif isinstance(ours, float):
        # Masses are in amu through each side's own neutron mass (see MASS_RTOL).
        rel = MASS_RTOL if path.endswith(".mass") else RTOL
        assert ours == pytest.approx(theirs, rel=rel), path
    else:
        assert ours == theirs, path


def test_fudge_reads_the_tsl_gnds_kika_writes(tslGnds):
    """FUDGE opens kika's file and finds what it finds in the ENDF tape."""
    _, fromEndf, fromKika, _ = tslGnds
    _assertSameDump(fromEndf, fromKika)


def test_kika_reads_the_tsl_gnds_fudge_writes(tslGnds):
    """kika opens FUDGE's file and finds what it decodes from the ENDF tape.

    The scatterers' pids and the target's id differ by design (FUDGE names
    them from the file name, kika from MF7/MT451 and ZSYMAM), so the
    comparison is of the physics: forms, grids, tables, scalars.
    """
    from kika.nuclear_data.model import (EVAL_LABEL, CoherentElastic, IncoherentElastic,
                                         IncoherentInelastic)

    suite, _, _, theirs = tslGnds
    assert str(theirs.interaction) == "thermalNeutronScatteringLaw"
    assert len(theirs.reactions) == len(suite.reactions)
    for mine, other in zip(suite.reactions, theirs.reactions):
        a = mine.doubleDifferentialCrossSection[EVAL_LABEL]
        b = other.doubleDifferentialCrossSection[EVAL_LABEL]
        assert type(a) is type(b)
        if isinstance(a, CoherentElastic):
            for ga, gb in zip(a.S_table.grids, b.S_table.grids):
                np.testing.assert_allclose(ga.values, gb.values, rtol=RTOL)
                assert str(ga.interpolation) == str(gb.interpolation)
            np.testing.assert_allclose(a.S_table.values, b.S_table.values, rtol=RTOL)
        elif isinstance(a, IncoherentElastic):
            assert a.boundAtomCrossSection.value == pytest.approx(
                b.boundAtomCrossSection.value, rel=RTOL)
            xa, ya = _collapse(*a.DebyeWallerIntegral.toEndfRegions()[:2])
            xb, yb, _ = b.DebyeWallerIntegral.toEndfRegions()
            np.testing.assert_allclose(xa, xb, rtol=RTOL)
            np.testing.assert_allclose(ya, yb, rtol=RTOL)
        elif isinstance(a, IncoherentInelastic):
            assert a.calculatedAtThermal == b.calculatedAtThermal
            assert len(a.scatteringAtoms) == len(b.scatteringAtoms)
            for x, y in zip(a.scatteringAtoms, b.scatteringAtoms):
                assert x.numberPerMolecule == y.numberPerMolecule
                assert x.primaryScatterer == y.primaryScatterer
                assert x.mass.value == pytest.approx(y.mass.value, rel=MASS_RTOL)
                assert x.boundAtomCrossSection.value == pytest.approx(
                    y.boundAtomCrossSection.value, rel=MASS_RTOL)
                assert x.e_max.value == pytest.approx(y.e_max.value, rel=RTOL)
                assert type(x.selfScatteringKernel.kernel) is type(y.selfScatteringKernel.kernel)
            ka = a.principal.selfScatteringKernel.kernel
            kb = b.principal.selfScatteringKernel.kernel
            for ga, gb in zip(ka.grids, kb.grids):
                np.testing.assert_allclose(ga.values, gb.values, rtol=RTOL)
            np.testing.assert_allclose(ka.values, kb.values, rtol=RTOL)
