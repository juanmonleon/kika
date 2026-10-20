"""NJOY's thermal rule where kika applies it: every adaptive table is five times
tighter below 0.4999 eV, judged at the left node of each interval
(:mod:`kika.processing.tolerances`).

Each engine tabulates a 1/v cross-section, convex, so a lin-lin chord lies above
it everywhere between its nodes. Each test checks the intervals whose left node
is thermal against the tightened bound, and that the intervals above it use more
than that bound, so the factor is what keeps the thermal ones inside it.
"""
import numpy as np
import pytest

from kika.processing.tolerances import (DOPPLER_SHARE, SOURCE_SHARE, THERMAL_FACTOR,
                                        THERMAL_RANGE, heatedShares, thermalFactor)

RTOL = 1.0e-3


def one_over_v(energy):
    return 10.0 / np.sqrt(np.asarray(energy, dtype=float))


def chord_errors(x, y, exact, fractions=np.linspace(0.05, 0.95, 19)):
    """Largest relative chord error inside each interval of the table (x, y)."""
    left, width = x[:-1], np.diff(x)
    q = left[:, None] + width[:, None] * fractions
    chord = np.interp(q.ravel(), x, y).reshape(q.shape)
    truth = exact(q.ravel()).reshape(q.shape)
    return left, np.max(np.abs(chord - truth) / np.abs(truth), axis=1)


def split(left, error):
    thermal = left < THERMAL_RANGE
    assert thermal.any() and (~thermal).any()
    return error[thermal].max(), error[~thermal].max()


def test_the_factor_and_the_shares():
    assert thermalFactor([0.0253, THERMAL_RANGE - 1e-9, THERMAL_RANGE, 1.0]).tolist() == [5.0, 5.0, 1.0, 1.0]
    assert THERMAL_FACTOR == 5.0
    broadening, thinning = heatedShares(1.0e-3)
    assert broadening == pytest.approx(DOPPLER_SHARE * 1.0e-3)
    assert SOURCE_SHARE * 1.0e-3 + broadening + thinning == pytest.approx(1.0e-3, rel=1e-12)


def test_the_reconstruction_refinement_is_five_times_tighter_below_0_4999_eV():
    from kika.processing.resonances.grid import ReconstructionOptions, linearize

    options = ReconstructionOptions(rtol=RTOL, atol=1.0e-12)
    x, values, _ = linearize(lambda e: {102: one_over_v(e)}, [1.0e-5, 1.0, 20.0], options, 100_000)
    left, error = chord_errors(x, values[102], one_over_v)
    thermal, above = split(left, error)
    # The reconstruction tabulates at half its tolerance, a fifth of that below 0.4999 eV.
    assert thermal <= 1.02 * 0.5 * RTOL / THERMAL_FACTOR
    assert above <= 1.02 * 0.5 * RTOL
    assert above > 0.5 * RTOL / THERMAL_FACTOR


def test_the_doppler_table_is_five_times_tighter_below_0_4999_eV():
    from kika.processing.doppler import DopplerContext, DopplerOptions, SupportPolicy, broaden_curve, prepare_curve

    source = np.geomspace(1.0e-5, 20.0, 4001)
    prepared = prepare_curve(((source, one_over_v(source)),), context=DopplerContext(56.0), source_temperature=0.0,
                             support=SupportPolicy("zero", "zero"))
    options = DopplerOptions(rtol=RTOL)
    result = broaden_curve(prepared, temperature=293.6, options=options, output_domain=(1.0e-4, 10.0))
    x, y = np.asarray(result.form.xs), np.asarray(result.form.ys)
    left, error = chord_errors(x, y, lambda q: result.evaluate(q).values)
    thermal, above = split(left, error)
    share = options.table_share * RTOL
    assert thermal <= 1.05 * share / THERMAL_FACTOR
    assert above <= 1.05 * share
    assert above > share / THERMAL_FACTOR


def test_the_thinning_is_five_times_tighter_below_0_4999_eV():
    from kika.hdf5.processing import _thin

    grid = np.geomspace(1.0e-5, 20.0, 20_001)
    y = one_over_v(grid)
    kept, thinned = _thin(grid, {102: (y, 0)}, RTOL)
    assert kept[0] == grid[0] and kept[-1] == grid[-1]
    assert len(kept) < len(grid) // 10
    back = np.interp(grid, kept, thinned[102][0])
    error = np.abs(back - y) / y
    # The left node of the kept chord that covers each original point.
    left = kept[np.searchsorted(kept, grid, side="right") - 1]
    thermal, above = error[left < THERMAL_RANGE].max(), error[left >= THERMAL_RANGE].max()
    assert thermal <= RTOL / THERMAL_FACTOR
    assert above <= RTOL
    assert above > RTOL / THERMAL_FACTOR
