"""A table that opens below the kinematic threshold is raised as RECONR does.

Evaluations round ``-Q (A+1)/A`` to seven figures and sometimes round it down
(Au-197 ENDF/B-VIII.1 MT17, 51-63, 602-649). RECONR (``reconr.f90`` lunion)
moves the first energy to ``sigfig(thrx, 7, +1)`` and pushes up any point the
move overtakes; ACER flags a table it did not fix as a threshold error.
"""
import numpy as np
from kika.nuclear_data.model import Reaction, ReactionId, CrossSection, OutputChannel, Q, XYs1d, Regions1d
from kika.processing.resonances import NeutronContext, reconstruct_suite
from kika.processing.resonances.probability_tables import sigfig as _sigfig
from test_resonance_suite import suite_model
from test_resonance_tabulation import AXES

AWR = 56.0
THRESHOLD = 500.00037  # sigfig(., 7, +1) = 500.0005


def sigfig(x, digits, shade):
    """RECONR's value as written: seven figures, without sigfig's 1e-13 bias."""
    return float(f"{_sigfig(x, digits, shade):.7g}")


def _with(label, mt, xs, ys, threshold):
    suite = suite_model()
    channel = OutputChannel(Q=Q(value=-threshold * AWR / (AWR + 1), unit="eV"))
    form = XYs1d(xs, ys, axes=AXES, label="eval")
    suite.reactions.append(Reaction(ReactionId(label, ENDF_MT=mt), CrossSection({"eval": form}), outputChannel=channel))
    return suite


def _table(result, label):
    form = next(f for k, f in result.forms.items() if k.label == label)
    curves = form.function1ds if isinstance(form, Regions1d) else [form]
    return np.concatenate([c.xs for c in curves]), np.concatenate([c.ys for c in curves])


def test_a_table_below_the_kinematic_threshold_is_raised_to_reconr_s_value():
    result = reconstruct_suite(_with("inelastic", 51, [500.0, 1000.0], [0.0, 1.0], THRESHOLD), NeutronContext(AWR, 0.0))
    raised = sigfig(THRESHOLD, 7, +1)
    assert result.report["raised_thresholds"] == {"inelastic": (500.0, raised)}
    x, y = _table(result, "inelastic")
    assert y[x < raised].max() == 0.0
    # The value stays where the evaluation put it; only the abscissa moved.
    assert np.interp(750.0, x, y) == np.interp(750.0, [raised, 1000.0], [0.0, 1.0])


def test_points_the_move_overtakes_are_pushed_up_one_place_each():
    suite = _with("inelastic", 51, [499.9, 500.0001, 1000.0], [0.0, 0.25, 1.0], THRESHOLD)
    result = reconstruct_suite(suite, NeutronContext(AWR, 0.0))
    first = sigfig(THRESHOLD, 7, +1)
    second = sigfig(first, 7, +1)
    assert result.report["raised_thresholds"]["inelastic"] == (499.9, first)
    x, y = _table(result, "inelastic")
    i = np.searchsorted(x, second)
    assert x[i] == second and y[i] == 0.25


def test_a_table_already_at_reconr_s_threshold_is_left_alone():
    at = sigfig(THRESHOLD, 7, +1)
    result = reconstruct_suite(_with("inelastic", 51, [at, 1000.0], [0.0, 1.0], THRESHOLD), NeutronContext(AWR, 0.0))
    assert result.report["raised_thresholds"] == {}
    x, y = _table(result, "inelastic")
    assert x[np.flatnonzero(y > 0)[0] - 1] == at
