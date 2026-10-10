# The algebra layer

```{admonition} At a glance
:class: note
**Decides:** every operation on a tabulated one-dimensional function (evaluating,
integrating, refining, summing, folding, comparing) is implemented once, in
`kika.algebra`, which takes plain arrays and imports nothing from kika.

**Packages:** `kika.algebra`.

**Enforced by:** `kika/algebra/tests/test_algebra.py` (no kika import, and a frozen list
of submodules).
```

## The structure

A tabulated function, to this package, is three arrays and nothing else:

- `x`, the abscissae, non-decreasing;
- `y`, the ordinates;
- `laws`, one interpolation law per **interval**: `laws[i]` joins points `i` and `i+1`.

The laws are the five codes ENDF-6 defines (ENDF-6 §0.5.2) and GNDS adopted (GNDS-2.1
§3.4.4):

| Code | Name | Between $(x_1, y_1)$ and $(x_2, y_2)$ |
|---|---|---|
| 1 | histogram | $y = y_1$ |
| 2 | lin-lin | $y$ linear in $x$ |
| 3 | lin-log | $y$ linear in $\ln x$ |
| 4 | log-lin | $\ln y$ linear in $x$ |
| 5 | log-log | $\ln y$ linear in $\ln x$ |

There are no units, reaction numbers, file formats or physics in the package. The
submodules:

| Module | Does |
|---|---|
| `laws` | The law codes, and the translation between one law per interval and ENDF's cumulative `(NBT, INT)` pairs. |
| `evaluate` | The value of a table, and its one-sided limits, at any abscissa. |
| `grid` | Unions of grids that keep steps; finding and splitting at discontinuities. |
| `integrate` | Closed-form integrals under each law; group and interval averages (weight 1 or $1/x$); averages in logarithmic windows; Legendre moments. |
| `refine` | Adaptive refinement, and the lin-lin form of any table to a stated tolerance (`to_linlin`). |
| `arithmetic` | Sums of tables, exact between the nodes and at every step; tables read on one common grid. |
| `fold` | Gaussian folds of a tabulated function, in closed form. |
| `compare` | Differences of two tables, and of a table against its own window or group averages. |
| `prepared` | A validated, read-only snapshot of a table for repeated evaluation. |

The model's function containers, the processing code (group averages, resonance
reconstruction), the ENDF and G4NDL code and the sampling code all call it.

## Design decisions

### Mathematics only

A function here takes and returns numbers, tables and laws: no energies or
cross-sections as such, no flux, no reaction, no physical constant. The law codes are a
numbering of interpolation laws, a convention of the mathematics, and not physics.
Anything that needs a physical model belongs in the calculation layer that builds on
this one: the ENDF fission-spectrum laws, for instance, are in
`kika.nuclear_data.spectrum_laws`, and resonance formalisms are in `kika.processing`.
Because the package holds no physics, it can import nothing from kika, and so the model,
the formats and the physics can all build on it without an import cycle.

### One implementation per operation

Evaluating, integrating or summing a table is done here or nowhere. Two implementations
of the same integral can disagree, and when they do it is never clear which result a
caller got. The model's `XYs1d` and `Regions1d` evaluate through the same code, so the
model and array-based code return the same numbers bit for bit.

### Exact, or it raises

Each law is evaluated and integrated in closed form. What has no closed form is first
rewritten as lin-lin to a tolerance the caller states (`to_linlin`). A logarithmic law on
a non-positive value is an error, never a silent switch to lin-lin. One case is read as
a limit instead of refused: a panel with a logarithmic $y$ and one end exactly 0 (and no
value below 0) is taken as the limit of its own law, 0 inside the panel with the jump at
the non-zero end. That is what NJOY gives for the same panel, and evaluations write such
panels.

### A discontinuity is a repeated abscissa

A step is written as two points at the same $x$, $(x, y_\text{left})$ and
$(x, y_\text{right})$, which is how ENDF and GNDS state one. Every operation keeps it as a
step; none turns it into a ramp. The value at the step is the right-hand limit;
`left_limit` and `right_limit` give both.

### Zero outside the domain

A table is zero outside $[x_1, x_n]$ unless the caller asks otherwise (`outside="hold"`
keeps the end values, `"raise"` refuses). A table that starts at a non-zero value
therefore starts with a step: its left limit at $x_1$ is the outside value.

### One law per interval, and not ENDF's regions

Inside the package each interval carries its own law, which is simpler to compute with
than ENDF's cumulative region boundaries. `laws.interval_laws` and `laws.pairs_from_laws`
translate in both directions, so a table read from ENDF and written back keeps its
regions.

### Two-dimensional codes are refused, not reduced

INT=6 (the charged-particle Coulomb law) is not implemented, and the codes 11-15 and
21-25 are refused: they are laws between incident energies of a two-dimensional table,
with a flag in the tens digit (unit-base or corresponding-point interpolation). Reading
such a code as the one-dimensional law of its units digit would drop the flag; the
decision belongs to the caller, who has the flag.

### Speed from numpy, pass by pass

Recursive refinement runs as a sequence of vectorised passes over every panel still
being refined, and not as recursion one panel at a time.

## What it rules out

- An import of any kika module from the package
  (`kika/algebra/tests/test_algebra.py::test_algebra_imports_nothing_from_kika`).
- A new submodule that has not been added to the frozen list in the test
  (`::test_algebra_holds_only_its_mathematical_submodules`), so that whether new code is
  mathematics is asked when it arrives.
- A second implementation of the same operation elsewhere in kika: a convention, held by
  review and not by a test.

## Known exceptions

None in the package itself. Elsewhere, some format-shaped classes (the flat MF 4 and
MF 5 classes of the ENDF parse layer, the ACE distribution classes) call the algebra
for their own evaluations; they do not reimplement it.

## Extending it

New code goes here only if it is mathematics on tables in the sense above. Add the
module to `MATHEMATICS_ONLY` in `kika/algebra/tests/test_algebra.py`, after asking that
question. Code that needs a unit, a reaction or a physical constant goes to the
calculation layer and calls the algebra. The mathematics of each operation (the closed
forms, the refinement criterion, the fold) is documented on its own methods page.

## Implementation

`kika/algebra/`: `laws.py`, `evaluate.py`, `grid.py`, `integrate.py`, `refine.py`,
`arithmetic.py`, `fold.py`, `compare.py`, `prepared.py`. The package exports the public
functions of all of them (`kika.algebra.evaluate`, `kika.algebra.integral`,
`kika.algebra.to_linlin`, and the rest).

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538. Cited as ENDF-6: §0.5.2 (interpolation laws).
- C. Mattoon et al., *GNDS-2.1 Specifications*, OECD Nuclear Energy Agency (2025): §3.4.4.
