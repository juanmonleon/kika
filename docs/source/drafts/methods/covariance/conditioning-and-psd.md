# Conditioning a covariance matrix for sampling

```{admonition} At a glance
:class: note
**Computes:** a pre-flight report on a set of covariance blocks (can each be sampled,
would a draw reproduce it, and what each available repair would measurably do to it), a
conditioning plan, the repaired blocks, and the decompositions a draw is made from.

**Applies to:** any square covariance block handed to a sampler: MF 31, 33, 34 and 35
covariances assembled by kika, MF 32 resonance-parameter covariances, and multigroup
covariances read from processed files.

**Entry points:** `kika.cov.conditioning.inspect_blocks`,
`kika.cov.conditioning.ConditioningReport.recommended_plan`,
`kika.cov.conditioning.apply_plan`; the repairs
{py:func}`kika.cov.decomposition.clip_projection`,
{py:func}`kika.cov.decomposition.nearest_psd_higham`,
{py:func}`kika.cov.decomposition.clip_and_rescale`,
{py:func}`kika.cov.decomposition.cap_variance_congruence`; the decompositions
{py:func}`kika.cov.decomposition.svd_decomposition`,
{py:func}`kika.cov.decomposition.eigen_decomposition`,
{py:func}`kika.cov.decomposition.cholesky_decomposition`.

**Status:** experimental — since 0.3.0.
```

## Background

A covariance matrix $C$ ($n \times n$) is symmetric and positive semi-definite (PSD):
every eigenvalue of $C = V \Lambda V^\top$ is $\lambda_i \ge 0$, so that every linear
combination of the quantities has a non-negative variance. A draw $y = F z$ with
$F F^\top = C$ and $z$ a vector of independent standard normals exists only for a PSD
$C$. The covariances of an evaluation (ENDF-6 Ch. 30-35) are PSD in exact arithmetic,
but the matrix a sampler receives is not that one. It is rebuilt from six-significant-digit
numbers, from correlation matrices packed as integers (the INTG records of the compact
resonance-parameter covariances, ENDF-6 Ch. 32), or by collapsing to a group structure; each step can leave eigenvalues slightly
below zero, correlations a few units of the last digit above 1, rows the processing never
filled (non-finite), or bins with an implausibly large variance.

Most of these matrices are also **rank-deficient by construction**: an evaluated
covariance is typically built from far fewer independent sources than it has rows. A
null direction ($\lambda_i = 0$) is part of what the evaluation states and is not a
defect. Some blocks carry a **sum rule**: the covariance of group probabilities of a
normalised spectrum (MF 35, ENDF-6 Ch. 35) satisfies $C\mathbf{1} = 0$, because the
probabilities always sum to one.

Repairing an indefinite matrix means replacing it with a PSD one near it. Which one is
"near" depends on what has to be kept: the eigenvectors, the stated variances (the
diagonal), or the sum rule. No repair keeps all three, and the choice moves the samples.

## What it computes

Input: a set of blocks, each a key and a matrix in the space it is to be sampled in
(relative for every pipeline in kika), optionally with a **family** label per row saying
which rows are comparable (one label per reaction, Legendre order or band).

Output, in three stages:

1. **Inspect.** A `ConditioningReport` with one `BlockReport` per block, each a list of
   findings. A finding has a check name, a severity and, where a repair exists, the
   repairs that apply with their measured effect on this matrix. Inspection changes no
   matrix.
2. **Decide.** A `ConditioningPlan`: an ordered list of steps (block, repair,
   parameters, reason). The recommended plan repairs definiteness and nothing else.
3. **Apply.** The repaired blocks, in the order they came in, and one record per step
   run.

A finding has one of three severities:

- `blocks`: a draw would fail, or would return something that is not a sample of
  anything;
- `distorts`: a draw succeeds, but not of the covariance that was handed in;
- `note`: true and worth knowing, and implies no action.

These severities belong to the assembled matrices. The checks of the covariance sections
as the file writes them, which run before any matrix is assembled, are a separate layer
with levels of their own ({doc}`covariance-checks`).

A block is **samplable** when it has no `blocks` finding, and **faithful** when it has
neither a `blocks` nor a `distorts` finding. The definiteness test that decides most
blocks uses the negativity ratio

$$
r = \frac{-\lambda_{\min}}{\lambda_{\max}}, \qquad
\text{not PSD} \iff r > n\,\varepsilon ,
$$ (eq-conditioning-and-psd-ratio)

with $\varepsilon = 2.2 \cdot 10^{-16}$ the float64 machine epsilon.

## Method

### Inspecting a block

A square block runs nine checks; the findings are sorted by severity (`blocks`, then
`distorts`, then `note`) and, within one severity, in the order below. The spectrum is
computed once, by a symmetric eigensolver, and shared by the checks that need it; it is
not computed when the block has non-finite entries.

| Check | Fires when | Severity |
|---|---|---|
| `finiteness` | any entry is NaN or infinite | `blocks` |
| `negative_variance` | a finite diagonal entry is $C_{ii} < 0$ | `blocks`, and no repair offered |
| `symmetry` | $a = \max\lvert C - C^\top\rvert / \max\lvert C\rvert > 10^{-12}$ | `distorts` if $a < 10^{-6}$, `blocks` otherwise |
| `correlation_bound` | an off-diagonal $\lvert\rho_{ij}\rvert - 1 > 10^{-10}$ | `note` if the worst excess is at most $10^{-4}$, or if the block is $\ln(1+\Sigma)$ and $\Sigma$ as stated has no excess above $10^{-4}$; `blocks` otherwise |
| `definiteness` | {eq}`eq-conditioning-and-psd-ratio` holds | `distorts` |
| `variance_outliers` | $C_{ii}$ exceeds 1000 times the median variance of its family | `distorts` |
| `inert_rows` | a row's variance is non-finite or $\lvert C_{ii}\rvert < 10^{-12}$ | `note` |
| `rank` | fewer than $n$ eigenvalues exceed $10^{-10}\lambda_{\max}$ | `note` |
| `sum_rule` | $s(C) \le 10^{-3}$, {eq}`eq-conditioning-and-psd-sumrule` | `note` |

The correlations are $\rho_{ij} = C_{ij}/\sqrt{C_{ii}C_{jj}}$, taken only over rows with a
finite, positive variance. The row-sum statistic is

$$
s(C) = \frac{\max_i \left\lvert \sum_j C_{ij} \right\rvert}{\max_{ij} \lvert C_{ij}\rvert},
$$ (eq-conditioning-and-psd-sumrule)

which is zero exactly when every row sums to zero. Two special cases of `definiteness`:
a block with no positive eigenvalue and $\max\lvert C\rvert \le 10^{-12}$ (a block that
states no uncertainty anywhere) is a `note` and draws no perturbation; any other block with
no positive eigenvalue is `blocks`. The outlier check uses only rows with a finite,
positive variance, and only families with at least four of them (`min_family_size`); with
no families given, the whole block is one family.

A rectangular block, or one declared `square_form=False`, is the off-diagonal partition of
a larger covariance (one Legendre order against another, one reaction against another).
Its diagonal is not a variance and its spectrum means nothing, so only `finiteness` runs,
plus a `cross_block` note saying that the block can only be conditioned inside the joint
matrix it belongs to.

### Where the negativity lives

When families are given, a `definiteness` finding also says which pair of families carries
the negative eigenvalues (`kika.cov.conditioning.negative_mass_by_family`). For each
eigenvector $v_k$ with $\lambda_k$ below $-n\varepsilon\lambda_{\max}$, let $v_{k,a}$ be
$v_k$ restricted to the rows of family $a$, and $C_{ab}$ the block of rows $a$ and columns
$b$. The quadratic form splits exactly,

$$
\lambda_k = v_k^\top C v_k = \sum_{a, b} v_{k,a}^\top C_{ab}\, v_{k,b},
\qquad
m_{ab} = (2 - \delta_{ab}) \sum_k v_{k,a}^\top C_{ab}\, v_{k,b}\quad (a \le b),
$$ (eq-conditioning-and-psd-negative-mass)

and the share of pair $(a, b)$ is $m_{ab} / \sum_k \lambda_k$. The shares sum to one, and a
pair that reduces the negativity has a negative share, so a share above one is possible.
The summary names the pair with the largest share ("91 % of the negative mass sits in
MT 4 × MT 16"). That is a statement about the evaluation, which a person can act on (by
leaving a reaction out of the request, or by correcting the file) and a projection cannot.

### Measuring the repairs

For a `definiteness` finding, each candidate PSD repair (by default `clip`,
`clip_rescale` and `higham`) is run on a copy of the matrix and measured
(`kika.cov.conditioning.predict_psd_repairs`); the copies are discarded. With $C$ the
block, $C'$ the repaired copy, and the **stated rows**
$S = \{ i : \lvert C_{ii}\rvert \ge 10^{-12} \max_j \lvert C_{jj}\rvert \}$, the effect of a
repair is:

$$
\begin{aligned}
\delta_{\text{diag}} &= \max_{i \in S} \frac{\lvert C'_{ii} - C_{ii}\rvert}{\lvert C_{ii}\rvert}, \qquad
\delta_F = \frac{\lVert C' - C\rVert_F}{\lVert C\rVert_F}, \\
\delta_{\text{inert}} &= \max_{i \notin S} \lvert C'_{ii} - C_{ii}\rvert, \qquad
r' = \frac{\lambda_{\min}(C')}{\lambda_{\max}(C')}, \qquad s(C'),
\end{aligned}
$$ (eq-conditioning-and-psd-effect)

with the number of non-stated rows that come out with a variance above the same floor,
and the time taken. The diagonal is split at the floor because a single relative change
over every row is unreadable on a spectrum whose lowest groups hold $10^{-17}$ of the
total: a negligible absolute change there is a huge relative one.

Higham's repair is iterative and costs one eigendecomposition per iteration, so when it is
only being measured it is capped at 200 iterations, and reports `converged=False` if it
hit the cap. Blocks larger than 1500 rows are not measured at all, and Higham is not
measured above 250 rows; the finding is still raised, and its repairs carry
`predicted=False` with the reason. The pipeline's own inspection runs with
`predict=False`, since the recommendation needs only the spectrum.

### The recommended plan

`ConditioningReport.recommended_plan` gives every block exactly one step:

- a block with a `distorts` definiteness finding gets the repair `psd_method="auto"`
  would choose ({eq}`eq-conditioning-and-psd-auto`): `clip` when $r < 10^{-2}$, `higham`
  otherwise;
- unless the block also carries a `sum_rule` note, in which case it gets `clip`, with the
  override written into the step's reason;
- every other block gets an explicit `none` step, so that the plan says which blocks
  were looked at and left alone.

Every other finding that offers a repair (outliers, inert rows, symmetry, a correlation
above 1, non-finite entries) is left unrepaired and written into the plan's `notes`, as is
every blocking finding with no repair. A plan serialises to JSON (`to_dict`, `from_dict`,
version 1); blocks are matched by the text of their key, with NumPy scalars rendered as
plain numbers, so a plan written on one machine matches on another.

### Applying a plan

`apply_plan` runs, for each block in input order, the steps that name it, in the order
the plan lists them (capping and then projecting is not the same matrix as projecting and
then capping). The repairs it can run are `none`, `clip`, `clip_rescale`, `higham`,
`cap`, `rescale_to_family_median` and `mask_inert`:

- `clip`: {eq}`eq-conditioning-and-psd-clip` with floor 0, no symmetrisation, and a block
  with no negative eigenvalue returned untouched, which is exactly what the SVD draw does
  with `psd_method="clip"`;
- `clip_rescale` and `higham` (diagonal kept unless the step says otherwise): as below;
- `cap`: {eq}`eq-conditioning-and-psd-congruence` with the step's `max_variance`;
- `rescale_to_family_median`: the one-sided congruence with the `indices` and `targets`
  the `variance_outliers` finding carries;
- `mask_inert`: the rows and columns the `inert_rows` test marks are set to zero (the
  order of the block does not change, so the indices of later steps stay valid).

Each record carries the block, the repair, its parameters and reason, $n$, whether the
matrix changed, $\delta_{\text{diag}}$ and $\delta_F$ against the matrix before that step,
the time, and the repair's own `info`. Running `inspect_blocks` again on the result is how
a plan is checked to have done what it was chosen for.

### The PSD repairs

**Eigenvalue clipping.** With $C = V\Lambda V^\top$ and a floor $f \ge 0$,

$$
C_{\text{clip}} = V \max(\Lambda, f)\, V^\top .
$$ (eq-conditioning-and-psd-clip)

With $f = 0$, $C_{\text{clip}}$ is the PSD matrix nearest to a symmetric $C$ in the
Frobenius norm (Higham 1988). It keeps the eigenvectors, so a null direction stays null
and a sum rule $C\mathbf{1} = 0$ survives, but it does not keep the diagonal: each clipped
eigenvalue $\lambda_k < 0$ adds $-\lambda_k v_{ik}^2$ to $C_{ii}$. A strictly positive
floor makes the result positive definite, which a Cholesky factorisation needs and an SVD
does not. The variant used for measurement symmetrises the product,
$(C_{\text{clip}} + C_{\text{clip}}^\top)/2$; the one used for drawing does not.

**Higham's diagonal-preserving projection.** The nearest PSD matrix with the stated
diagonal, in the Frobenius norm, found by alternating projections with Dykstra's
correction (Higham 2002; Dykstra 1983). The input is symmetrised and its non-finite entries are set to zero.
With $P_S$ the projection {eq}`eq-conditioning-and-psd-clip` (floor `eigval_floor`, no
symmetrisation) and $P_U$ the projection that puts back the stated diagonal $d$,

$$
\begin{aligned}
R_k &= Y_k - \Delta S_k, \qquad X_k = P_S(R_k), \qquad \Delta S_{k+1} = X_k - R_k,\\
Y_{k+1} &= \tfrac12\left(P_U(X_k) + P_U(X_k)^\top\right), \qquad Y_0 = C,\ \Delta S_0 = 0,
\end{aligned}
$$ (eq-conditioning-and-psd-higham)

until $\lVert Y_{k+1} - Y_k\rVert_F / \lVert Y_k\rVert_F < \text{tol}$. By default
$\text{tol} = 10^{-10}$ for $n \le 1000$ and $10^{-8}$ above, and at most 1000 iterations
for $n \le 2000$ and 200 above. The last iterate has the stated diagonal but may have
eigenvalues slightly below zero; if it does, it is clipped once more,
{eq}`eq-conditioning-and-psd-clip` with the same floor, without restoring the diagonal, so
the diagonal is kept to the size of those eigenvalues and not exactly. A matrix whose
smallest eigenvalue is at least $-\text{tol}\cdot\max_i\lvert\lambda_i\rvert$ is returned
symmetrised and otherwise as it came. The test is relative to the spectrum's scale, so a
matrix whose entries are all of order $10^{-12}$ (an absolute covariance in b², say) is
judged by the same standard as one of order one. With `preserve_diagonal=False` the function is a single clip.

**Clip and rescale.** Clip, then put the stated diagonal back by a congruence:

$$
C' = D\, C_{\text{clip}}\, D, \qquad
D_{ii} =
\begin{cases}
\sqrt{C_{ii} / (C_{\text{clip}})_{ii}} & C_{ii} > 0 \text{ and } (C_{\text{clip}})_{ii} > 0,\\
0 & \text{otherwise},
\end{cases}
$$ (eq-conditioning-and-psd-rescale)

symmetrised. A congruence of a PSD matrix is PSD, the stated diagonal comes back exactly,
and a row with no stated variance comes back identically zero. A sum rule does not
survive: $(D C D)\mathbf{1} = D C (D\mathbf{1})$ is zero only when $D$ is a multiple of
the identity.

**Variance cap and targeted rescales.** All three are the congruence

$$
C' = \operatorname{diag}(s)\, C\, \operatorname{diag}(s), \qquad
s_i =
\begin{cases}
\sqrt{t_i / C_{ii}} & C_{ii} > t_i,\\
1 & \text{otherwise},
\end{cases}
$$ (eq-conditioning-and-psd-congruence)

which keeps every correlation and keeps a PSD matrix PSD, and never raises a variance.
For the cap ({py:func}`kika.cov.decomposition.cap_variance_congruence`), $t_i$ is one
ceiling for every row. For the targeted rescale
({py:func}`kika.cov.decomposition.rescale_threshold_bins_congruence`), $t_i$ is given per
flagged row, and rows whose variance or target is not finite and positive are skipped. Two
functions choose the rows and targets for it, on a block laid out as reactions
$\times$ groups:

- {py:func}`kika.cov.decomposition.flag_threshold_bins`: for each reaction with a known
  threshold $E_{\text{th}}$, the group $g$ with $E_g \le E_{\text{th}} < E_{g+1}$, whose
  relative variance is inflated because its mean cross-section includes the region below
  threshold. The target is the median variance of the reaction's other groups (finite,
  positive), whatever their number if there is at least one, and otherwise the median of
  every finite, positive variance of the same nuclide;
- {py:func}`kika.cov.decomposition.flag_outlier_variance_bins`: in each reaction with at
  least four finite, positive variances, every group above 1000 times the reaction's
  median (the group itself included in the median). The target is the median of the
  other groups, which must also number at least four.

**The decomposition's own last resort.** Before a Cholesky factorisation, a matrix that is
still not positive definite gets a multiple of the identity,

$$
C' = C + \delta I, \qquad
\delta = \min\!\left(-\lambda_{\min} + 10^{-10}\lVert C\rVert_F,\ 10^{-3}\lVert C\rVert_F\right),
$$ (eq-conditioning-and-psd-jitter)

which repairs only a matrix whose negativity is below $10^{-3}\lVert C\rVert_F$.

### Choosing a repair inside a decomposition

The decompositions take a `psd_method`: `"auto"`, `"clip"`, `"clip_rescale"`,
`"higham"` or `"none"`; any other value is refused (`ValueError`). `"auto"` takes one
eigendecomposition, of the matrix with its non-finite entries set to zero, and resolves to

$$
\text{auto} =
\begin{cases}
\texttt{clip} & \max(-\lambda_{\min}, 0)/\lambda_{\max} < 10^{-2},\\
\texttt{higham} & \text{otherwise},
\end{cases}
$$ (eq-conditioning-and-psd-auto)

reusing that eigendecomposition on the clip path. The repair and the decomposition that
follow act on that same matrix, with its non-finite entries set to zero, so under `"auto"`
a NaN never reaches the SVD, the eigensolver or the Cholesky factorisation. (A Cholesky
factorisation does not fail on a NaN, it returns a NaN factor, so the cleaning comes before
its first attempt.) `"auto"` never resolves to `clip_rescale`.

- {py:func}`kika.cov.decomposition.svd_decomposition` repairs the matrix as asked (clip
  with floor 0, unsymmetrised, with an already-PSD matrix passed through untouched), then
  returns its SVD $U\Sigma V^\top$. This is the decomposition a draw uses by default. For a
  symmetric matrix the singular values are $\lvert\lambda_i\rvert$, and the sign of
  $\lambda_i$ is the sign of $u_i^\top v_i$; the draw uses it to tell a negative direction
  from a positive one (see *Limits*).
- {py:func}`kika.cov.decomposition.eigen_decomposition` returns $(\lambda, V)$. Under
  `clip` the eigenvectors are those of the input and the negative eigenvalues are set to
  zero; under `higham` and `clip_rescale` the repaired matrix is decomposed again.
- {py:func}`kika.cov.decomposition.cholesky_decomposition` returns $L$ with
  $C \approx LL^\top$. A positive definite matrix is factorised as it is. Otherwise
  `clip` floors the eigenvalues at $10^{-14}$; `higham` runs
  {eq}`eq-conditioning-and-psd-higham` with floor $10^{-14}$ and, if it did not converge
  and is more than 1 % away from the input in the relative Frobenius norm, discards the
  result and clips the input at $10^{-14}$ instead; `clip_rescale` applies
  {eq}`eq-conditioning-and-psd-rescale`; `none` adds
  {eq}`eq-conditioning-and-psd-jitter`. If the factorisation still fails after a clip,
  Higham or `clip_rescale` (a clip at floor 0 can leave the matrix singular), the jitter is
  added to the repaired matrix and the factorisation tried once more.

If the symmetric eigensolver fails to converge, every one of these falls back to an SVD of
the symmetric matrix, whose singular values are $\lvert\lambda_i\rvert$, and recovers the
signs as $\operatorname{sign}(v_i^\top C v_i)$.

## Conventions and choices

- **Inspection measures, it does not repair.** Whether a repair is admissible is a
  judgement about the evaluation, and the properties that decide it (must the diagonal be
  exact, must a sum rule survive) are properties of the data. So the report runs the
  candidates on a copy and reports numbers, and the choice is a plan a person can read and
  edit.
- **The recommended plan repairs definiteness and nothing else.** A negative eigenvalue is
  the one defect a draw cannot decline to act on: the decomposition projects the matrix
  somewhere whether anyone chose to or not. Capping a variance, rescaling an outlier or
  masking a row changes what the evaluation states, and is applied only when a plan says
  so.
- **Clip for shallow negativity, Higham for deep.** Below $r = 10^{-2}$ the eigenvalues
  being clipped are small, so the diagonal moves little and one eigendecomposition is
  enough; above it, clipping moves the stated variances appreciably and the diagonal is
  worth an iterative projection.
- **A sum-rule block is always clipped.** Clipping keeps the eigenvectors, so
  $C\mathbf{1} = 0$ survives; a congruence (`clip_rescale`) does not keep it, and neither,
  in general, does Higham's projection, which restores a non-uniform diagonal. A spectrum drawn from a matrix without the sum rule is no longer normalised.
- **`clip_rescale` is asked for by name only.** It gives a different numerical answer from
  `clip`, and is the right repair where the marginal uncertainties matter and no sum rule
  does (cross-sections, Legendre coefficients).
- **The definiteness test has a round-off tolerance**, $n\varepsilon$ on the negativity
  ratio. Without it, a matrix that has just been clipped (whose clipped eigenvalues are
  zero, and come back from a new eigendecomposition as $\pm 10^{-17}$) would still be
  reported as not PSD, and a run could never satisfy its own pre-flight. The tolerance
  grows with $n$ because the error of a symmetric eigensolver does.
- **A correlation above 1 by at most $10^{-4}$ is the file's rounding.** ENDF writes six
  significant digits, so a stated correlation of 1 can come back as $1 + O(10^{-6})$, and
  the ratio of two rounded variances moves it further. Such a matrix is only slightly
  indefinite, which the definiteness check handles. A correlation read back from an INTG
  matrix at the wrong number of digits is wrong by $10^{-3}$ or more, and is refused.
- **In log space the correlation bound is judged against the file.** A relative block
  drawn in log space is inspected as $\ln(1+\Sigma)$, the matrix the draw samples, and
  $\ln(1+x)$ is concave: two bins the file states fully correlated with different standard
  deviations $s_i \ne s_j$ have $\ln(1+s_i s_j) > \sqrt{\ln(1+s_i^2)\,\ln(1+s_j^2)}$, a
  correlation above 1. No pair of log-normal factors reproduces that pair exactly, but the
  file is not at fault, so when $\Sigma$ as stated (passed as `stated` to
  `kika.cov.conditioning.inspect_blocks`) has no correlation above
  $1 + 10^{-4}$, the finding is a note and the definiteness check takes over: the
  negative eigenvalue the excess implies is clipped, which brings every $\lvert\rho\rvert$
  back to at most 1 and only raises variances. A correlation above 1 the file states still
  blocks.
- **The sum-rule tolerance is $10^{-3}$** because a stated MF 35 band satisfies
  $C\mathbf{1} = 0$ only to the six digits the file holds, while a block without a sum
  rule has $s(C)$ of order one: a row sum of unrelated entries does not cancel.
- **Outliers are judged within their family.** Two reactions whose variances differ by
  decades are both normal; pooled, one would look like an outlier of the other.
- **Rank deficiency is a note.** Null directions are what an evaluated covariance is made
  of. Filling them in would claim an uncertainty the evaluator did not state.
- **A plan is applied exactly, and to exactly the blocks it names.** Steps run in the
  listed order, blocks come back in their input order (downstream, a block's position
  sets its seed), and a plan naming a block that is not present is refused: a plan that
  half-fits is a plan from another run.
- **Conditioning ahead of the draw gives the draw's own repair, bit for bit.** `clip` in a
  plan is the same arithmetic as `psd_method="clip"` in the SVD draw (floor 0, no
  symmetrisation, a clean matrix untouched), so a conditioned block drawn with
  `psd_method="none"` gives exactly the samples the unconditioned block gives with the
  repair inside the draw.

## Limits and failure modes

Refused, all with `ValueError`:

- a matrix that is not 2-D, or a rectangular one not declared a cross block;
- a family list whose length is not the block's order;
- an unknown candidate repair in `predict_psd_repairs`, and an unknown `psd_method` in a
  decomposition;
- in a plan: an unknown repair; `preserve` (the sum rule's constraint on the choice of
  repair, not an operation); `cap` without `max_variance`; `rescale_to_family_median`
  without `indices` and `targets`; `mask_inert` with any parameter but `floor`; a step
  naming a block that is not present; a plan of a version other than 1;
- in the congruence rescale, index and target lists of different lengths.

Reported instead of done: everything the recommended plan leaves alone, in its `notes`; a
repair not measured because the block is too large, with the reason; a Higham prediction
that hit its iteration cap (`converged=False`).

Limits of the methods:

- **No repair keeps eigenvectors, diagonal and sum rule together.** `clip` moves the
  stated variances, `clip_rescale` destroys the sum rule, and Higham moves the sum rule
  and keeps the diagonal only to the size of its final clip.
- **Higham may not converge** within its iteration limit, and reports so; the Cholesky path
  then falls back to clipping when the result is more than 1 % from the input.
- **The jitter of {eq}`eq-conditioning-and-psd-jitter` is capped** at $10^{-3}\lVert
  C\rVert_F$; a more negative matrix stays indefinite and the factorisation raises
  `numpy.linalg.LinAlgError`.
- **A block drawn as stated cannot keep a negative eigenvalue.** With `psd_method="none"`
  on an indefinite matrix (the pipeline's draw with `conditioningPlan=False`, or residue a
  repair leaves), {py:func}`kika.sampling.core.draw_samples` gives each direction with
  $\lambda_i < 0$ zero variance, under the SVD as under the eigendecomposition, and counts
  it in `n_null`. A negative variance cannot be realised, and drawing
  $\lvert\lambda_i\rvert$ would add variance in a direction the evaluation does not state.
  The realised covariance is then $V\max(\Lambda, 0)V^\top$, and **zeroing a direction
  lowers no component's variance**:

  $$
  \big(V\max(\Lambda, 0)V^\top\big)_{ii} = C_{ii} + \sum_{\lambda_k < 0} \lvert\lambda_k\rvert\, v_{ik}^2
  \;\ge\; C_{ii},
  $$ (eq-conditioning-and-psd-zeroed-diagonal)

  while drawing $V\lvert\Lambda\rvert V^\top$ adds twice that sum. A negative eigenvalue at
  the level of the file's rounding (a few $10^{-7}\lambda_{\max}$ on six-digit values)
  moves the variances by that much and no more; a clearly negative one raises the variances
  of the bins its eigenvector spans, which is conservative. The pre-flight is the place to
  choose a repair that keeps the diagonal exactly (`clip_rescale`, `higham`). The
  file-level samplers `perturb_ENDF_files` and `draw_relative_factors` (see
  {doc}`../../user_guide/legacy-samplers`) keep the SVD's $\lvert\lambda_i\rvert$
  (`negative_directions="fold"`), so that they reproduce their published draws.
  `min_over_max_eigenvalue` in the draw's diagnostics shows whether a block was indefinite.
- **Inert rows: two floors.** The `inert_rows` check, `mask_inert` and the multigroup draw
  use an absolute floor, $\lvert C_{ii}\rvert < 10^{-12}$: in a relative block, a variance
  that small is a row the file leaves empty. The stated rows $S$ in
  {eq}`eq-conditioning-and-psd-effect` and in the applied records use a relative one,
  $10^{-12}\max_j\lvert C_{jj}\rvert$, because the effect measurements also read absolute
  blocks (an MF 35 band, for instance), whose variances can all lie below the absolute
  floor. The two agree only when the largest variance is of order one.
- **The recommendation does not look at outliers.** A bin with a variance $10^9$ times its
  neighbours is drawn as stated unless a plan rescales it.
- **A cross block cannot be conditioned alone**; it is conditioned inside the joint matrix
  it belongs to.

## Validation

The tests below were run on 2026-10-11 and passed.

**The checks.** Each check fires on a matrix built to trigger it and no other:
`kika/cov/tests/test_conditioning.py::test_a_well_conditioned_matrix_reports_nothing`,
`::test_non_finite_entries_block`, `::test_a_negative_stated_variance_blocks_and_offers_nothing`,
`::test_a_correlation_above_one_blocks`,
`::test_roundoff_asymmetry_is_not_reported_and_real_asymmetry_blocks`,
`::test_a_wholly_inert_block_draws_nothing_and_a_partly_inert_one_is_noted`,
`::test_non_psd_is_distorting_not_blocking_and_names_what_auto_would_do` (a negativity of
$10^{-6}$ gives `clip`, of 0.5 gives `higham`),
`::test_outliers_are_judged_within_their_family`, `::test_rank_deficiency_is_a_note_not_a_problem`,
`::test_a_cross_block_runs_only_the_checks_that_mean_anything`. A correlation of
$1 + 2\cdot10^{-7}$ is a note and the plan clips the block; one of $1 + 10^{-3}$ blocks
(`kika/cov/tests/test_correlation_roundoff_is_a_note.py::test_roundoff_above_one_is_a_note_and_definiteness_takes_over`,
`::test_a_real_excess_still_blocks`). In log space, a correlation above 1 that only
$\ln(1+\Sigma)$ makes (two fully correlated bins with standard deviations 0.1 and 1) is a
note, the recommended clip leaves the block positive semi-definite without lowering any
diagonal entry, and a correlation of 1.2 the file states still blocks
(`kika/cov/tests/test_a_log_transform_correlation_is_not_a_refusal.py::test_the_log_transform_excess_is_a_note_and_clip_repairs_it`,
`::test_a_correlation_above_one_the_file_states_still_blocks`).

**The negative mass.** On a two-family block made indefinite through its cross block, the
masses sum to $\sum_k \lambda_k$ and the shares to 1, and the cross pair is named with a
share above 1
(`kika/cov/tests/test_negative_mass_by_family.py::test_the_split_is_exact`,
`::test_the_cross_block_is_named_as_the_culprit`).

**The repairs.**

- Higham: $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ (eigenvalues 3 and $-1$) goes to
  $\begin{pmatrix}1&1\\1&1\end{pmatrix}$ to $10^{-8}$
  (`kika/cov/tests/test_nearest_psd.py::TestNearestPsdHigham::test_2x2_known_solution`);
  on random indefinite matrices the diagonal is kept to $10^{-8}$ ($n = 50$,
  `::test_diagonal_preservation`), the result has $\lambda_{\min} \ge -10^{-10}$
  ($n = 100$, `::test_psd_guarantee`), and $n = 500$ converges (`::test_large_matrix`).
- `clip_rescale`: the diagonal comes back to a relative $10^{-12}$, the result has
  $\lambda_{\min} > -10^{-12}\lambda_{\max}$, a zero row stays exactly zero, and the result
  is exactly symmetric
  (`kika/cov/tests/test_clip_rescale.py::test_the_diagonal_comes_back_exactly`,
  `::test_the_result_is_psd`, `::test_a_component_with_no_stated_variance_gets_an_empty_row`,
  `::test_the_symmetric_result_is_exactly_symmetric`); `auto` gives the same eigenvalues as
  `clip` and not those of `clip_rescale` (`::test_it_is_not_reachable_from_auto`).
- The decompositions' edge cases. Cholesky with `clip_rescale` factorises
  $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ and keeps its diagonal
  (`kika/cov/tests/test_decomposition_edge_cases.py::test_cholesky_clip_rescale_is_its_own_branch`).
  Higham repairs that matrix scaled to $10^{-12}$, and leaves a rounding-level negative
  eigenvalue alone at either scale
  (`::test_higham_judges_definiteness_relative_to_the_matrix`). Under `auto` each
  decomposition of a block with a NaN gives a finite result
  (`::test_auto_hands_the_cleaned_matrix_on`).
- The draw gives a direction stated with negative variance zero variance under both
  decompositions, and `negative_directions="fold"` keeps its magnitude
  (`kika/sampling/tests/test_a_negative_direction_is_not_folded.py`). On the Fe-56 MF 34
  joint of the committed micro-tape, the folded draw has 87 null directions and the default
  draw 93
  (`kika/sampling/tests/test_mf34_migration_equivalence.py::test_the_draw_itself_is_bit_for_bit`).
- The single clip reproduces each former call site's arithmetic bit for bit, and returns a
  PSD matrix as the same object
  (`kika/cov/tests/test_conditioning_plan.py::test_clip_projection_reproduces_each_call_sites_old_arithmetic`,
  `::test_a_clean_matrix_passes_through_clip_untouched`).

**On a real spectrum covariance.** The four MF 35 bands of Cf-252 in the committed slice of
ENDF/B-VIII.1 (the `micro_pfns_tape` fixture) all carry a sum rule, with
$10^{-9} < s(C) < 10^{-4}$
(`kika/cov/tests/test_conditioning.py::test_the_pfns_bands_are_recognised_as_carrying_a_sum_rule`).
On every band, `clip` moves a stated variance by more than 1 % and leaves $s$ within twice
its stated value; `clip_rescale` keeps the diagonal to $10^{-12}$ (relative) and multiplies
$s$ by more than 100
(`::test_the_pfns_repair_trade_off_reproduces_the_hand_measurement`,
`kika/cov/tests/test_clip_rescale.py::test_the_sum_rule_does_not_survive_the_rescale`).
These are the measurements behind clipping every sum-rule block.

**Plans.** A plan round-trips through JSON with its order; blocks come back in input order;
steps run in plan order; each refusal of *Limits* raises; a sum-rule block is recommended
`clip`; applying the recommendation to a sum-rule block makes it faithful and keeps
$s < 10^{-9}$ (`kika/cov/tests/test_conditioning_plan.py`, 20 tests).

**Against the draw, bit for bit.** On the MF 34 joint of Fe-56 elastic scattering in the
committed micro-tape (orders 1-6, $252 \times 252$, indefinite), conditioning with a plan
and drawing with `psd_method="none"` gives the same samples, bit for bit, as drawing the
raw block with `psd_method` set to the same repair, for `none`, `clip`, `clip_rescale` and
`higham`
(`kika/sampling/tests/test_conditioning_matches_the_sampler.py::test_conditioning_first_draws_what_repairing_inside_the_draw_draws`).
The recommendation there is `clip`, and the conditioned block has no definiteness finding
left (`::test_the_recommendation_is_actionable_on_a_real_joint`). A plan holding only a
variance cap, derived for the MF 33 draw, gives the same factors as the MF 33 sampler
`kika.sampling.generators.generate_samples`
(`::test_a_derived_plan_closes_the_same_loop_as_a_psd_one`).

**Resonance-parameter covariances.** Every committed MF 32 sub-format decodes to blocks
with no blocking finding, and blocks above 1500 rows report their repairs as not measured
(`kika/cov/tests/test_conditioning.py::test_every_mf32_sub_format_decodes_to_something_drawable`).

## Diagnostics

`ConditioningReport`:

- `samplable` and `faithful`: the two questions; the report has no truth value of its own,
  so a caller has to choose one. `failing()` and `distorting()` list the blocks that fail
  each; `findings(check)` filters by check; `summary()` is one line; `to_markdown()` is a
  table.
- Per finding, `evidence`. For `definiteness`: `min_eigenvalue`, `max_eigenvalue`,
  `negativity_ratio` ($r$), `n_negative`, `auto_would_choose`, and
  `negative_mass_by_family` (up to 16 pairs, with `pair`, `mass` and `share`). **A large
  `negativity_ratio`** means any repair will move the matrix appreciably; read the
  remedies before accepting one. For `variance_outliers`: per flagged row, `index`,
  `family`, `variance`, `family_median` (the flagged row included), `ratio`. For `sum_rule`: `residual` ($s$). For
  `rank`: `rank`, `order`, `null_fraction`.
- Per measured repair, `effect`: `stated_diagonal_max_relative_change`
  ($\delta_{\text{diag}}$), `frobenius_relative_change` ($\delta_F$),
  `max_variance_added_to_an_inert_row` ($\delta_{\text{inert}}$),
  `inert_rows_gaining_variance`, `n_inert_rows`, `min_eigenvalue_ratio` ($r'$),
  `sum_rule_residual_before` and `_after`, `seconds`, and for Higham `converged` and
  `iterations`. **On a block with a sum rule, a `sum_rule_residual_after` far above
  `_before`** disqualifies the repair. **`converged=False`** means the Higham numbers are
  those of an unfinished projection.

`ConditioningPlan`: `steps` (each `block`, `remedy`, `parameters`, `reason`) and `notes`,
the findings left unrepaired. **Read the notes**: an outlier left in a block is drawn as
stated.

The records from `apply_plan`: `block`, `remedy`, `parameters`, `reason`, `n`, `changed`,
`stated_diagonal_max_relative_change`, `frobenius_relative_change`, `seconds`,
`applier_info`. The repairs' own `info`:

- `clip_projection`: `n_negative`, `min_eigenvalue`, `clipped` (false when a PSD matrix was
  passed through);
- `nearest_psd_higham`: `iterations`, `converged`, `frobenius_distance`,
  `relative_frobenius_error`, `max_diagonal_change`, `eigenvalue_range_before` and
  `_after`, `n_negative_eigenvalues_before`, `n_svd_fallback_iterations`;
- `clip_and_rescale`: `n_negative`, `min_eigenvalue`, `max_diagonal_error_before` and
  `_after`;
- `cap_variance_congruence`: `n_capped`, `max_original_variance`, `max_variance_cap`,
  `capped_entries`; the targeted rescale: `n_flagged`, `n_actually_rescaled`,
  `rescaled_entries`.

The `verify_*` functions of `kika.cov.decomposition` compare a matrix with its
reconstruction from a decomposition and return three percentages: the relative Frobenius
error, the largest relative diagonal error over rows with $\lvert C_{ii}\rvert \ge
10^{-12}$, and the largest off-diagonal error divided by $\lVert C\rVert_F$ (not by the
entry). {py:func}`kika.cov.decomposition.verify_psd_projection` returns the relative
Frobenius distance and largest diagonal change (both in per cent) and the largest change
of an off-diagonal correlation.

## Implementation

- `kika.cov.conditioning`: the checks and `inspect_matrix`, `inspect_blocks`;
  `negative_mass_by_family`; `predict_psd_repairs` and the effect measurements; the report,
  the plan and its JSON form; `recommended_plan`; `apply_plan` and its dispatch to the
  repairs.
- `kika.cov.decomposition`: the repairs (`clip_projection`, `nearest_psd_higham`,
  `clip_and_rescale`, `cap_variance_congruence`, `flag_threshold_bins`,
  `flag_outlier_variance_bins`, `rescale_threshold_bins_congruence`); the `auto` rule and
  its threshold; the eigensolver with its SVD fallback; the three decompositions and the
  `verify_*` checks.
- `kika.sampling.core`: the draw, which zeroes non-finite entries, decomposes with the
  chosen `psd_method`, gives the negative directions zero variance and truncates the null
  directions.
- `kika.sampling.model_perturbation`: the pipeline's use of the pre-flight (inspect,
  recommend, apply, draw with `psd_method="none"`); see
  {doc}`../sampling/perturbation-pipeline`.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: Ch. 30-35 (covariances), Ch. 32 (INTG records), Ch. 35 (MF 35).
- N. J. Higham, "Computing a nearest symmetric positive semidefinite matrix", *Linear
  Algebra and its Applications* 103, 103-118 (1988).
- N. J. Higham, "Computing the nearest correlation matrix: a problem from finance", *IMA
  Journal of Numerical Analysis* 22, 329-343 (2002).
- R. L. Dykstra, "An algorithm for restricted least squares regression", *Journal of the
  American Statistical Association* 78(384), 837-842 (1983): the correction to
  alternating projections.
