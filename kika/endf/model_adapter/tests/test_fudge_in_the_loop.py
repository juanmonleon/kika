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


def _runFudge(command, tape: Path) -> dict:
    source = (_HERE / "fudge_oracle_dump.py").read_text()
    stdin = source + "\n" + TAPE_MARKER + tape.read_text()
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
