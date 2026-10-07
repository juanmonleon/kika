"""kika.algebra against independent oracles: scipy quadrature and hand values."""
import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from kika import algebra as A
from kika.algebra.laws import interval_laws, pairs_from_laws, validate


# ---------------------------------------------------------------------------
# Laws
# ---------------------------------------------------------------------------

def test_interval_laws_follow_cumulative_nbt():
    # Region 1 = points 1-3 (intervals 0, 1), region 2 = points 3-5.
    assert interval_laws(5, [(3, 5), (5, 2)]).tolist() == [5, 5, 2, 2]
    assert interval_laws(4, []).tolist() == [2, 2, 2]
    assert interval_laws(5, [(3, 4)]).tolist() == [4, 4, 4, 4]  # short NBT holds


def test_pairs_round_trip():
    laws = np.array([5, 5, 2, 2, 1])
    assert interval_laws(6, pairs_from_laws(laws)).tolist() == laws.tolist()


@pytest.mark.parametrize("code", [0, 6, 22])
def test_codes_outside_one_to_five_are_refused(code):
    with pytest.raises(ValueError, match="not 1-5"):
        interval_laws(3, [(3, code)])


@pytest.mark.parametrize("law,y", [(5, [1.0, 0.0]), (4, [-1.0, 2.0])])
def test_log_law_on_non_positive_values_is_refused(law, y):
    with pytest.raises(ValueError, match="ln y"):
        A.evaluate([1.0, 2.0], y, law, 1.5)


def test_log_x_law_on_non_positive_abscissa_is_refused():
    with pytest.raises(ValueError, match="ln x"):
        A.evaluate([0.0, 2.0], [1.0, 2.0], 3, 1.0)


def test_zero_width_interval_has_no_law_to_check():
    # A log-log table that steps down to zero at its last abscissa.
    validate([1.0, 2.0, 3.0, 3.0], [1.0, 2.0, 2.0, 0.0], 5 * np.ones(3, int))


# ---------------------------------------------------------------------------
# Evaluation and limits
# ---------------------------------------------------------------------------

def test_values_and_limits_at_a_step_and_at_the_edges():
    x, y = [1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 5.0, 6.0]
    assert A.evaluate(x, y, 2, [0.5, 1, 1.5, 2, 3, 4, 5]).tolist() == [0, 1, 1.5, 5, 5.5, 6, 0]
    assert A.left_limit(x, y, 2, [1, 2, 4]).tolist() == [0, 2, 6]
    assert A.right_limit(x, y, 2, [1, 2, 4]).tolist() == [1, 5, 0]
    assert A.left_limit(x, y, 2, 1.0, outside="hold") == 1.0
    with pytest.raises(ValueError):
        A.evaluate(x, y, 2, 5.0, outside="raise")


def test_histogram_left_limit_is_the_panel_value():
    x, y = [1.0, 2.0, 3.0], [4.0, 7.0, 9.0]
    assert A.left_limit(x, y, 1, 2.0) == 4.0
    assert A.evaluate(x, y, 1, [1.5, 2.0, 3.0]).tolist() == [4.0, 7.0, 9.0]


@pytest.mark.parametrize("law", [1, 2, 3, 4, 5])
def test_tabulated_points_read_back_bit_for_bit(law):
    x = np.geomspace(1.0, 1e6, 40)
    y = np.abs(np.sin(x)) + 0.1
    assert np.array_equal(A.evaluate(x, y, law, x), y)


@pytest.mark.parametrize("law,f", [
    (3, lambda t: 2.0 + 0.7 * np.log(t)),
    (4, lambda t: 3.0 * np.exp(-0.4 * t)),
    (5, lambda t: 5.0 * t ** -2.5),
])
def test_each_law_reproduces_its_own_function(law, f):
    x = np.array([1.5, 9.0])
    t = np.linspace(1.6, 8.9, 50)
    assert np.allclose(A.evaluate(x, f(x), law, t), f(t), rtol=1e-13, atol=0)


def test_evaluate_matches_the_legacy_linlin_bits():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 10, 500))
    y = rng.normal(size=500)
    q = rng.uniform(x[0], x[-1], 2000)
    assert np.array_equal(A.evaluate(x, y, 2, q), np.interp(q, x, y))


def test_sample_on_union_takes_both_limits_at_a_repeat():
    x, y = [2.0, 3.0], [1.0, 1.0]
    out = A.sample_on_union(x, y, 2, [1.0, 2.0, 2.0, 2.5, 3.0, 3.0, 4.0])
    assert out.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Integrals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", [-3.0, -2.0, -1.0, -0.999999, 0.0, 0.5, 2.0])
def test_loglog_integral_including_steep_decrease(b):
    # FUDGE's series branch is wrong for every b < -1 (+6 % at b = -2 on [1, 2]).
    for lo, hi in ((1.0, 2.0), (1.0, 10.0), (1.0, 1.0 + 1e-9)):
        x = np.array([lo, hi])
        exact = quad(lambda t: t ** b, lo, hi, epsabs=0, epsrel=1e-13)[0]
        assert A.integral(x, x ** b, 5) == pytest.approx(exact, rel=1e-12)
        exact_w = quad(lambda t: t ** (b - 1), lo, hi, epsabs=0, epsrel=1e-13)[0]
        assert A.integral(x, x ** b, 5, weight="1/x") == pytest.approx(exact_w, rel=1e-12)


@pytest.mark.parametrize("law", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("x", [[2.0, 7.0], [1.0, 1.0 + 1e-7]])
def test_every_law_integrates_in_closed_form(law, x):
    x, y = np.array(x), np.array([3.0, 0.5])
    f = lambda t: A.evaluate(x, y, law, t)
    exact = quad(f, x[0], x[1], epsabs=0, epsrel=1e-13)[0]
    assert A.integral(x, y, law) == pytest.approx(exact, rel=1e-11)
    if law != 4:
        exact_w = quad(lambda t: f(t) / t, x[0], x[1], epsabs=0, epsrel=1e-13)[0]
        assert A.integral(x, y, law, weight="1/x") == pytest.approx(exact_w, rel=1e-11)


def test_loglin_with_a_1_over_x_weight_raises_rather_than_approximates():
    with pytest.raises(ValueError, match="to_linlin"):
        A.integral([1.0, 2.0], [1.0, 3.0], 4, weight="1/x")


def test_one_over_x_on_a_coarse_table_is_exact():
    # The trapezoid in ln E that resonance_group_average used is -7 % here.
    x, y = np.array([1e5, 1e6, 2e7]), np.array([1.0, 10.0, 1.0])
    exact = sum(quad(lambda t: np.interp(t, x, y) / t, a, b, epsrel=1e-13)[0]
                for a, b in zip(x[:-1], x[1:]))
    assert A.integral(x, y, 2, weight="1/x") == pytest.approx(exact, rel=1e-13)


def test_group_integrals_cut_panels_and_ignore_the_outside():
    x = np.linspace(1.0, 10.0, 7)
    y = x ** -2.0
    edges = [0.0, 1.5, 3.3, 3.3, 9.0, 20.0]
    g = A.group_integrals(x, y, 5, edges)
    assert g[0] == pytest.approx(1 - 1 / 1.5, rel=1e-14)
    assert g[2] == 0.0
    assert g.sum() == pytest.approx(A.integral(x, y, 5), rel=1e-14)


def test_group_integrals_across_a_step():
    x, y = [0.0, 1.0, 1.0, 2.0], [1.0, 1.0, 3.0, 3.0]
    assert A.group_integrals(x, y, 2, [0.0, 1.0, 2.0]).tolist() == [1.0, 3.0]


# ---------------------------------------------------------------------------
# Refinement and to_linlin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", [3, 4, 5])
def test_to_linlin_meets_its_tolerance_and_keeps_every_point(law):
    x, y = np.array([1e5, 1e6, 2e7]), np.array([1.0, 10.0, 0.1])
    lx, ly = A.to_linlin(x, y, law, 1e-4)
    assert set(x) <= set(lx)
    assert np.array_equal(ly[np.isin(lx, x)], y)
    t = np.linspace(lx[0], lx[-1], 20001)
    assert np.max(np.abs(np.interp(t, lx, ly) / A.evaluate(x, y, law, t) - 1)) < 1e-4


def test_to_linlin_writes_a_histogram_as_steps():
    lx, ly = A.to_linlin([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1)
    assert lx.tolist() == [1, 2, 2, 3, 3] and ly.tolist() == [1, 1, 2, 2, 3]


def test_to_linlin_values_snapped_points_at_their_snapped_abscissa():
    snap = lambda q: np.round(q, 0)
    x, y = np.array([10.0, 1000.0]), np.array([1.0, 100.0])
    lx, ly = A.to_linlin(x, y, 5, 1e-4, snap=snap)
    assert np.array_equal(lx, np.round(lx, 0))
    assert np.allclose(ly, lx / 10.0, rtol=1e-12)  # y = x/10 exactly on this law


def test_refine_probes_only_new_panels():
    calls = []

    def f(q, owner):
        calls.append(q.size)
        return np.sin(q)

    x = np.linspace(0, 3, 4)
    res = A.refine(x, np.sin(x), f, lambda a, c: np.abs(a - c) / 1e-3, insert="worst")
    # A pass probes 3 fractions on each panel created by the previous pass.
    assert calls[0] == 3 * 3
    assert sum(calls) == res.evaluations
    # Each split replaces one panel by two, and only those two are probed next:
    # starting panels plus two per added point, three probes each.
    assert res.evaluations == 3 * ((x.size - 1) + 2 * (res.x.size - x.size))


def test_refine_budget_exhaustion_raises():
    with pytest.raises(A.RefinementError):
        A.refine(np.array([0.0, 1.0]), np.zeros(2), lambda q, o: np.sin(50 * q),
                 lambda a, c: np.abs(a - c) / 1e-12, max_points=50)


# ---------------------------------------------------------------------------
# Sums
# ---------------------------------------------------------------------------

def test_sum_steps_at_a_partial_edge():
    u, s = A.add([([1.0, 3.0], [1.0, 1.0], 2), ([2.0, 4.0], [2.0, 2.0], 2)])
    assert u.tolist() == [1, 2, 2, 3, 3, 4]
    assert s.tolist() == [1, 1, 3, 3, 2, 2]


def test_sum_of_loglog_partials_is_right_between_nodes():
    x = np.array([1e5, 1e6, 2e7])
    a, b = (x, 1e6 / x, 5), (x, (x / 1e5) ** 0.5, 5)
    u, s = A.add([a, b], tol=1e-5)
    t = np.geomspace(1e5, 2e7, 5001)
    exact = A.evaluate(*a, t) + A.evaluate(*b, t)
    assert np.max(np.abs(np.interp(t, u, s) / exact - 1)) < 1e-5


# ---------------------------------------------------------------------------
# Layering: the package imports nothing from kika
# ---------------------------------------------------------------------------

def test_algebra_imports_nothing_from_kika():
    root = Path(A.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level > 1 or mod.split(".")[0] == "kika":
                    offenders.append(f"{path.name}:{node.lineno} {mod}")
            elif isinstance(node, ast.Import):
                offenders += [f"{path.name}:{node.lineno} {a.name}" for a in node.names
                              if a.name.split(".")[0] == "kika"]
    assert not offenders, offenders
