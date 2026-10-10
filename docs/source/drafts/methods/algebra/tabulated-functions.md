# Evaluating, integrating and combining tabulated functions

```{admonition} At a glance
:class: note
**Computes:** the value of a tabulated function at any abscissa, its integral (with weight
1 or $1/x$) over any interval or set of groups, its Legendre moments, its lin-lin form to
a stated tolerance, the sum of several tables, and its convolution with a Gaussian.

**Applies to:** any table $(x_i, y_i)$ with one ENDF/GNDS interpolation law per interval
(codes 1-5).

**Entry points:** `kika.algebra.evaluate`, `kika.algebra.integral`,
`kika.algebra.group_averages`, `kika.algebra.legendre_moments`, `kika.algebra.to_linlin`,
`kika.algebra.add`, `kika.algebra.fold_tabulated`.

**Status:** stable.
```

## Background

Evaluated data are tables: a cross-section $\sigma(E)$, a multiplicity, a probability
$f(\mu)$. Between two tabulated points the function follows an interpolation law stated
by the file (ENDF-6 §0.5.2, GNDS-2.1 §3.4.4). The law is part of the data: the same points
under lin-lin and log-log describe different functions. Why kika keeps all of this in one
package, and its general rules, are in {doc}`../../architecture/algebra-layer`.

## What it computes

For a table with nodes $x_1 \le \dots \le x_n$, values $y_i$ and law $\ell_i$ on
$[x_i, x_{i+1}]$, the function $y(x)$ is, on each interval,

$$
y(x) =
\begin{cases}
y_i & \ell = 1 \text{ (histogram)},\\
y_i + (y_{i+1}-y_i)\,\dfrac{x - x_i}{x_{i+1}-x_i} & \ell = 2 \text{ (lin-lin)},\\
y_i + (y_{i+1}-y_i)\,\dfrac{\ln(x/x_i)}{\ln(x_{i+1}/x_i)} & \ell = 3 \text{ (lin-log)},\\
y_i\,\exp\!\Big(\ln\dfrac{y_{i+1}}{y_i}\,\dfrac{x - x_i}{x_{i+1}-x_i}\Big) & \ell = 4 \text{ (log-lin)},\\
y_i\,\Big(\dfrac{x}{x_i}\Big)^{\ln(y_{i+1}/y_i)/\ln(x_{i+1}/x_i)} & \ell = 5 \text{ (log-log)},
\end{cases}
$$ (eq-tabulated-functions-laws)

zero outside $[x_1, x_n]$, and right-continuous at a repeated abscissa (a step). Every
operation below acts on this function, not on the points.

## Method

**1. Integrals in closed form.** On a panel $[x_1, x_2]$ with $L = \ln(x_2/x_1)$,
$m = \ln(y_2/y_1)$ and $\Delta = x_2 - x_1$:

$$
\int_{x_1}^{x_2} y\,dx =
\begin{cases}
y_1 \Delta & \ell = 1,\\
\tfrac12 (y_1 + y_2)\,\Delta & \ell = 2,\\
\Delta\,[\,y_1 + (y_2 - y_1)\,h(L)\,], \quad h(L) = \dfrac{1}{1 - e^{-L}} - \dfrac1L & \ell = 3,\\
y_1 \Delta\, \dfrac{e^{m} - 1}{m} & \ell = 4,\\
y_1 x_1 L\, \dfrac{e^{s} - 1}{s}, \quad s = m + L & \ell = 5.
\end{cases}
$$ (eq-tabulated-functions-integrals)

$(e^s - 1)/s$ and $h(L)$ are evaluated in forms that stay accurate as their argument goes
to zero. With the weight $1/x$ every law but log-lin is also closed form; a log-lin panel
under $1/x$ is refused (it is an exponential integral) and has to be linearised first.

**2. Group integrals.** A group edge inside a panel cuts it into two panels of the same
law, the cut valued by that law. The group integral is then a sum of whole panels: nothing
is sampled, so it is the table's own integral. The group average divides by the integral
of the weight.

**3. Legendre moments** $\int y\,P_\ell\,dx$ of a lin-lin or histogram table are exact:
on a panel the integrand is a polynomial of degree $\ell+1$, integrated by Gauss-Legendre
with $\lfloor \ell/2 \rfloor + 2$ nodes per panel. A table under another law is linearised
first.

**4. Linearisation** (`to_linlin`). Every original point is kept with its value; points are
only added. Lin-lin panels are copied; a histogram panel gains the left limit of its right
end as a repeated abscissa (the exact lin-lin form of a step); a log-law panel is split
until the chord reproduces the law to a relative tolerance (default $10^{-4}$), tested at
$\{0.25, 0.5, 0.75\}$ of the panel in the law's own variable. Refinement runs pass by pass:
each pass tests only the panels created in the previous one.

**5. Sums.** Each table is linearised first, then all are sampled on the union of their
grids; a table's own domain edge with a non-zero value is carried as a repeated abscissa,
so the sum has the step there and not a ramp. The sum is exact up to the linearisation
tolerance. Summing log-log tables point by point and reading the result lin-lin would be
right at the nodes and wrong between them.

**6. Gaussian fold.** On a lin-lin panel $y = p + q\,t$, and with
$\alpha = (x_1 - x_0)/\sigma$, $\beta = (x_2 - x_0)/\sigma$,

$$
\int_{x_1}^{x_2} (p + q\,t)\,\mathcal{N}(t; x_0, \sigma^2)\,dt
= (p + q x_0)\,[\Phi(\beta) - \Phi(\alpha)] - q\,\sigma\,[\varphi(\beta) - \varphi(\alpha)],
$$ (eq-tabulated-functions-fold)

with $\Phi$ and $\varphi$ the standard normal distribution and density. Summed over panels,
the fold of the table's lin-lin interpolant is exact, with no quadrature. **Outside the table
the fold holds the end values** (so a threshold reaction that starts at zero folds to zero
below it); this is the one operation that does not take the table as zero outside its
domain. Where $\sigma \le 0$ the kernel is a delta and $y(x_0)$ is returned.

## Conventions and choices

- **One law per interval** inside the package, translated to and from ENDF's cumulative
  (NBT, INT) pairs on the way in and out.
- **Exact, or it raises**: no operation silently replaces a law by lin-lin. A log law on a
  non-positive value is an error, except a log-$y$ panel with one end exactly 0, read as the
  limit of its law (0 inside, the jump at the non-zero end), which is what NJOY gives and
  what evaluations write.
- **The default linearisation tolerance, $10^{-4}$**, sits an order of magnitude above the
  round-off of a seven-digit ENDF float.
- **Steps are repeated abscissae** and survive every operation.

## Limits and failure modes

- Codes other than 1-5 (INT=6, and the two-dimensional codes 11-15, 21-25) are refused.
- A log law on negative values, or a log-$x$ law on a non-positive abscissa, is refused.
- `to_linlin` and `refine` raise `RefinementError` when the point or pass budget runs out,
  or when a failing panel cannot be split in floating point (reason `"unresolvable"`).

## Validation

`kika/algebra/tests/`, 170 tests passed on 2026-10-11, among them: each law reproduces its
own function (`test_algebra.py::test_each_law_reproduces_its_own_function`); every law
integrates in closed form (`::test_every_law_integrates_in_closed_form`), the log-log case
including a steep decrease (`::test_loglog_integral_including_steep_decrease`); a $1/x$
weight on a coarse table is exact (`::test_one_over_x_on_a_coarse_table_is_exact`); group
integrals cut panels and cross steps (`::test_group_integrals_cut_panels_and_ignore_the_outside`,
`::test_group_integrals_across_a_step`); `to_linlin` meets its tolerance and keeps every
point (`::test_to_linlin_meets_its_tolerance_and_keeps_every_point`); a sum of log-log
partials is right between the nodes (`::test_sum_of_loglog_partials_is_right_between_nodes`);
Legendre moments are exact on a lin-lin table with a step
(`::test_legendre_moments_are_exact_on_a_linlin_table_with_a_step`); the fold of a flat
table is the identity and of a quadratic matches its analytic second moment
(`test_fold.py::test_fold_flat_is_identity`,
`::test_fold_quadratic_matches_analytic_second_moment`).

## Diagnostics

`refine` returns a `RefineResult` with the number of passes, of function evaluations and
of panels accepted unresolved.

## Implementation

`kika/algebra/`: `laws.py` (the codes, (NBT, INT) translation, validation), `evaluate.py`
({eq}`eq-tabulated-functions-laws`, one-sided limits), `grid.py` (unions that keep steps),
`integrate.py` ({eq}`eq-tabulated-functions-integrals`, groups, windows, Legendre moments),
`refine.py` (refinement, `to_linlin`), `arithmetic.py` (sums), `fold.py`
({eq}`eq-tabulated-functions-fold`), `compare.py` (differences of tables), `prepared.py`
(validated read-only snapshots for repeated evaluation).

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: §0.5.2.
- C. Mattoon et al., *GNDS-2.1 Specifications*, OECD Nuclear Energy Agency (2025): §3.4.4.
