# Checking covariance sections as written in the file

```{admonition} At a glance
:class: note
**Computes:** a list of findings about the covariance sections of one ENDF tape, or of
every tape of a library, each with a level (note, warning, defect), a location in the
file and its evidence, without changing anything.

**Applies to:** MF 31, 32, 33, 34, 35 and 40 of ENDF-6 tapes, read with kika's ENDF
parser.

**Entry points:** `kika.endf.check_covariances`, `kika.endf.check_covariance_library`;
the reports are `kika.endf.checks.CovarianceCheckReport` and
`kika.endf.checks.CovarianceLibraryReport`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

An ENDF (Evaluated Nuclear Data File) evaluation states its uncertainties in the
covariance files: MF 31 for the multiplicities $\bar\nu$, MF 32 for the resonance
parameters of MF 2, MF 33 for cross-sections, MF 34 for the Legendre coefficients
$a_\ell$ of the angular distributions of MF 4, MF 35 for the energy distributions of
MF 5 and MF 40 for the production cross-sections of MF 10 (ENDF-6 Ch. 31-35 and 40).
MF 31, 33 and 40 share one layout (ENDF-6 §33.2): a section per reaction MT, a
subsection per partner reaction (MAT1, MT1), and in each subsection NC records that
derive a covariance from other reactions and NI records that state it. An NI record has
a form LB: LB=0 to 4 are tables of $(E_k, F_k)$ pairs whose products build the matrix,
LB=5 is a matrix on one grid (stored as the upper triangle when LS=1, in full when
LS=0), LB=6 a rectangular matrix on a row and a column grid, and LB=8 and 9 are
short-range variances. LB=0, 8 and 9 are absolute; LB=1 to 6 are relative. The covariance
of a block is the **sum** of its NI records, each a piecewise-constant matrix on its own
energy bins (ENDF-6 §33.2). MF 34 nests the same records one level deeper, per pair of
Legendre orders $(L, L_1)$ (ENDF-6 §34.2), and states for each its frame, LCT. MF 35
gives, per band of incident energies $[E_1, E_2)$, one LB=7 matrix: the absolute
covariance of the group probabilities
$P_i = \int_{E'_i}^{E'_{i+1}} \chi(E \to E')\, dE'$ of the outgoing spectrum
(ENDF-6 Ch. 35). MF 32 gives the covariance of the parameters of each resonance energy
range of MF 2, in the layouts LCOMP=0, 1 and 2 (ENDF-6 Ch. 32).

A covariance matrix $C$ is symmetric and positive semi-definite (PSD): every eigenvalue
$\lambda_k \ge 0$. With $C_{ii}$ the variances, the correlations

$$
\rho_{ij} = \frac{C_{ij}}{\sqrt{C_{ii}\, C_{jj}}}
$$ (eq-covariance-checks-rho)

satisfy $|\rho_{ij}| \le 1$. The format pins many counts and indices that a reader has to
trust (NL, NI, NC, NT, the number of $(L, L_1)$ pairs). A section can be well formed and
still not be a covariance, or state one that the file cannot apply (the covariance of a
reaction the tape does not give, for example).

## What it computes

Input: a tape read with `kika.endf.read_endf` or `kika.endf.open_endf`, the covariance
files to check (all six by default), and optionally the cross-sections $\sigma(E)$ by
MT, for instance from a PENDF. Output: a `CovarianceCheckReport`, a tuple of findings.
Each finding is

- a **check**, the name of the rule (`correlation_out_of_bounds`, `nt_mismatch`, …);
- a **level**: `note` (true and worth knowing, not a fault), `warn` (probably a fault,
  or one that matters only in some uses) or `defect` (the section is not what ENDF-6
  says it should be, or is not a covariance);
- a **location**: as many of MAT, MF, MT, MAT1, MT1, $L$, $L_1$, the NI and NC
  positions, LB, LS, the final states LFS and LFS1, the band and the MF 32 range as
  apply;
- a one-line **summary** and the **evidence**, a dictionary of the numbers the rule
  measured.

The checks look at the section **as written**: the records the parser kept, with their
counts and grids, and each LS=1 triangle before it is mirrored. Where a value check
needs a matrix, the block is summed on the union of its records' grids the way kika
sums it to build a model covariance, so that what the checks see is what kika would use.
No finding raises, blocks anything or changes the section, and no check proposes a
repair: what to do about a finding is decided later, by the pre-flight of the
covariances that are actually sampled (see *Conventions and choices*).

## Method

The checks of one tape run file by file. MF 31 and MF 33 share one path, MF 34 has its
own, and MF 32, MF 35 and MF 40 reuse the pieces of MF 33 with the differences stated
below. A section the parser could not read at all is a `parse_error` defect, with the
parser's message as evidence.

### Structure

For every record and section, without looking at a value:

- **Counts.** NL (MF 31, 33), NMT1 (MF 34), NI, NC, NCI (for LTY=0), NK (MF 35), NS and
  NL (MF 40), and for MF 34 the number of $(L, L_1)$ sub-subsections that NL, NL1 and
  LTT imply, against what was read: a difference is `count_mismatch`, a defect.
- **NT.** The length of each NI record against what its other counts imply,
  `nt_mismatch` (defect):

  $$
  \mathrm{NT} =
  \begin{cases}
  2\,\mathrm{NP} & \mathrm{LB} \in \{0, 1, 2, 3, 4, 8, 9\},\\
  \mathrm{NE} + \tfrac{1}{2}(\mathrm{NE}-1)\,\mathrm{NE} & \mathrm{LB}=5,\ \mathrm{LS}=1,\\
  \mathrm{NE} + (\mathrm{NE}-1)^2 & \mathrm{LB}=5,\ \mathrm{LS}=0,\\
  1 + \mathrm{NER}\cdot\mathrm{NEC} & \mathrm{LB}=6.
  \end{cases}
  $$ (eq-covariance-checks-nt)

- **Flags.** LB must be one the file defines (MF 31, 33 and 40: 0-6, 8, 9; MF 34: 0, 1,
  2, 5, 6), `lb_invalid`; LTY one of 0-3, `lty_invalid`; both defects. LCT outside 0-2
  is `lct_invalid`, a warning. An MF 34 LB=0-2 record with LT > 0 (two tables) is kept
  verbatim but not decoded, so its values are not checked (`lt_not_decoded`, a note).
- **Grids.** A grid with fewer than two points (`grid_too_short`) or a decreasing one
  (`grid_not_increasing`) is a defect and the record is not used further; a repeated
  point, a zero-width bin, is `grid_repeated_point`, a warning.
- **The last $F$.** NE energies bound NE-1 intervals, so $F_{\mathrm{NE}}$ of an LB=0-4,
  8 or 9 table closes none, and ENDF-6 §33.2.2.2 asks for it to be zero. A non-zero one
  (`trailing_value_not_zero`) is a warning for LB=0-4, because it usually means the
  table is shifted by one place, and a note for LB=8 and 9, where the major libraries
  write it non-zero as a habit. kika ignores the value.
- **Decoding.** A record that passes the above is decoded with the section's own
  decoder; one that cannot be is `decode_error` (defect).
- **Orders.** In MF 34, $(L, L_1)$ must lie in $[L_\text{min}, L_\text{min} + \mathrm{NL})
  \times [L_\text{min}, L_\text{min} + \mathrm{NL1})$, with $L_\text{min} = 0$ for LTT=2
  or 3 and 1 otherwise, and a self subsection (MT1 = MT) stores only $L \le L_1$:
  `l_out_of_range`, a defect. A block written twice is `duplicate_block` (defect).
- **LS=1 in a cross block.** An LB=5 LS=1 record stores a symmetric triangle. In a
  cross block (MT1 ≠ MT, or $L_1 \ne L$) the matrix is not symmetric, but kika (like any
  processor) mirrors the triangle into the lower half. The finding `ls1_in_cross_block`
  is a defect when the triangle has a non-zero entry and a note when it is all zeros.
  When both self blocks are available, the evidence also gives the largest $|\rho|$ of
  the stored triangle alone, so that the value check below can say whether the mirror
  creates a $|\rho| > 1$ or only worsens one.
- **Short-range variances between reactions.** LB=8 or 9 in a cross block is a defect
  (`short_range_in_cross_block`): ENDF-6 §33.3.3 allows them only in a self block.
  Within a self block, an LB=8 or 9 with a non-zero $F_k$ less than 1 MeV above a
  threshold that is itself above 0.1 MeV is a note (`short_range_near_threshold`), after
  the same rule of §33.3.3.

### Completeness

The blocks and references a section implies (ENDF-6 §33.3.2 for MF 31 and 33, §34.2 for
MF 34):

- a section with cross blocks but no self block (`missing_self_block`), and a cross block
  whose partner MT has no section, or is a component of a lumped reaction
  (`missing_partner`): defects, because $|\rho|$ of the block is undefined. In MF 34 a
  block that involves an order with no variance block is a defect, or a note when the
  block is all zeros. An order declared by NL that has no self block while other orders
  of the same reaction do is legal (ENDF-6 §34.2) and a note
  (`order_without_variance`);
- a cross block stored in the section of the higher MT is a note
  (`cross_block_below_diagonal`), since kika places it by transposition; a cross block
  stated in both sections is summed both ways on the union of their grids and compared,
  a note when they agree to $10^{-3}$ of the largest element and are both relative or
  both absolute (`symmetric_block_repeated`), a defect otherwise
  (`symmetric_block_conflict`), because the one placed last would win;
- MAT1 written as the file's own MAT where ENDF-6 §33.3.1 asks for 0 is read as the same
  material, and noted (`mat1_is_own_mat`); a block with another material cannot be
  evaluated from one file and is a note (`external_material`);
- NC records: LTY=0 or 1 outside the self block (`nc_misplaced`); an LTY=0 derivation
  from a reaction that is the derived one, has no section, is a lumped component, or has
  no self covariance (`unresolved_reference`); an LTY=1 standard that names this
  material, or has no subsection for it; two LTY=0 ranges that overlap
  (`nc_ranges_overlap`); a constituent itself derived over an overlapping range
  (`nc_chained`, defect, against ENDF-6 §33.3.1), or a chain that leads back to the MT
  through another range (`nc_chained`, note); several LTY=0 records with different sums,
  of which kika resolves only the first (`nc_only_first_resolved`, note); and any
  LTY=1-3 ratio to a standard of another file, which kika does not resolve
  (`ratio_to_standard`, note);
- lumped reactions (ENDF-6 §33.2.3): an MTL outside 851-870, or a component whose lump
  has no section (`unresolved_reference`, defect); a lump that no section names as its
  MTL (`lumped_without_components`, warning);
- the central values: an MF 33 section with no MF 3 for its MT, an MF 31 with no MF 1, an
  MF 34 with no MF 4 (`missing_central_values`), a defect, judged only when that file was
  read. A summation MT counts as present when its partials are (MT 452 when MT 455 and
  456 are); lumped MTs 851-870 have no MF 3 by construction; an MF 34 whose angular
  distribution is in MF 6 is a note;
- in MF 34, a non-null $(L, L_1) = (0, 0)$ block between two reactions, or a non-null
  $(0, 0)$ self block, `magnitude_covariance_in_mf34`: ENDF-6 §34.3 keeps the covariance
  of the integrated cross-section in MF 33 and asks for these to be null, so a processor
  that sums both files counts it twice. A self block equal to MF 33's (to $10^{-3}$) is a
  note; one that differs, or has no MF 33 to compare with, is a warning;
- in MF 34, the frame LCT against the frame of MF 4 for both reactions
  (`frame_differs_from_mf4`): allowed by ENDF-6 §34.1; a note for MT 2 and MT 51-90,
  which kika's samplers convert, a warning for any other reaction, which they refuse.

### Values of a self block

On each self block, summed on the union grid of its records:

- a negative variance (`negative_variance`, defect);
- rows that are exactly zero (`inert_rows`, a note): the test is
  $\max_j C_{ij} = \min_j C_{ij} = 0$, never a threshold on $\sigma$;
- a row with zero variance and a non-zero covariance (`covariance_without_variance`,
  defect: $|\rho|$ is infinite there);
- asymmetry, $a = \max|C - C^\top| / \max|C|$ (`asymmetric_self_block`): a note above
  $10^{-6}$, a defect above $10^{-3}$;
- correlations, {eq}`eq-covariance-checks-rho`, over the rows with positive variance of
  the symmetrised block (`correlation_out_of_bounds`): nothing up to $1 + 10^{-6}$ (an
  ENDF field keeps six or seven significant figures), a note up to $1 + 10^{-3}$, a
  defect above. The evidence counts the pairs above $1 + 10^{-3}$: a self block is
  symmetric, so each pair is counted once (its upper triangle), as in MF 32.

### Positive semi-definiteness

The eigenvalues of the symmetrised self block, without its exactly zero rows, are
computed (a row with covariances and no variance is kept, since it is what makes such a
block indefinite). If $\lambda_\text{min} \ge 0$ nothing is reported; if
$\lambda_\text{max} \le 0$ the block is a defect. Otherwise, with

$$
r = \frac{|\lambda_\text{min}|}{\lambda_\text{max}},
$$ (eq-covariance-checks-ratio)

the first grade of `not_positive_semidefinite` is a note for $r < 10^{-6}$, a warning
for $10^{-6} \le r \le 10^{-3}$ and a defect above $10^{-3}$. Three measurements then
move it:

**1. ENDF rounding.** Each non-zero element, written to six significant figures, carries
a rounding error of at most half a unit in the sixth figure. Their Frobenius norm,

$$
\varepsilon_\text{ENDF} = \Big[ \sum_{C_{ij} \ne 0} \big( \tfrac12\, 10^{\lfloor \log_{10}|C_{ij}| \rfloor - 5} \big)^2 \Big]^{1/2},
$$ (eq-covariance-checks-ulp)

bounds the spectral norm of the rounding, and by Weyl's inequality how far it can move an
eigenvalue. If $|\lambda_\text{min}| \le \varepsilon_\text{ENDF}$ the finding is a note,
whatever $r$.

**2. Quantised correlations.** Some evaluations write each LB=5 record as
$\sigma_i \sigma_j \rho_{ij}$ with $\rho$ rounded to a lattice $q$. For a warning, every
LB=5 record of the block with at least three rows of positive variance is tested (one
with fewer is skipped): its off-diagonal non-zero correlations (at least three of them)
sit on a lattice
$q \in \{10^{-2}, 10^{-3}, 10^{-4}\}$, tried coarsest first, when at least 98 % of them
are within $\min(5\cdot10^{-5}, q/20)$ of a multiple of $q$. A rounding error uniform in
$\pm q/2$ on an $n \times n$ symmetric matrix has a spectral norm of about
$2\,(q/\sqrt{12})\sqrt{n}$ (the edge of Wigner's semicircle, TODO(ref)), so the record
$D R D$ can reach $-1.5 \cdot 2\,(q/\sqrt{12})\sqrt{n}\, \max_i d_i$ with $d_i$ its
variances and 1.5 a slack. Projected onto the union grid, a record's negative
eigenvalue grows by at most $\max_j \sum_i T_{ij}^2$, with $T_{ij}$ the fraction of union
bin $i$ that lies in native bin $j$; by Weyl's inequality the bounds of the records add.
LB=0, 1, 2, 8, 9 records are PSD by construction and add nothing; an LB=3, 4 or 6 record,
or an LB=5 record not on a lattice, voids the bound. If $|\lambda_\text{min}|$ is within
the summed bound, the warning becomes a note.

**3. What clipping would do.** Setting the negative eigenvalues to zero (the nearest PSD
matrix in the Frobenius norm, Higham 1988) adds
$\sum_{k:\lambda_k<0} |\lambda_k|\, v_{ik}^2$ to each variance. The largest relative
change of $\sigma$,

$$
\delta_\sigma = \max_{i:\ \sigma_i \ge 0.01\,\sigma_\text{max}}
\left[ \Big( \frac{C_{ii} + \sum_{k:\lambda_k<0} |\lambda_k|\, v_{ik}^2}{C_{ii}} \Big)^{1/2} - 1 \right],
$$ (eq-covariance-checks-impact)

is computed over the rows whose $\sigma$ is at least 1 % of the block's largest (a tail
row with a tiny $\sigma$ would turn any absolute correction into an enormous relative
one). A warning left after steps 1 and 2 becomes a note if $\delta_\sigma < 2\,\%$ and a
defect if $\delta_\sigma \ge 25\,\%$. Only a warning is regraded: a note explained by
rounding stays a note, and a defect stays a defect, because clipping can keep the
variances while breaking the correlations, which $r$ does see.

A finding also lists the LB=5 records that are already indefinite on their own, with
$r > 10^{-3}$ (`records_indefinite_alone`).

### Values of a cross block

A cross block $(X, Y)$ is summed on the union of its own grid and the grids of the two
self blocks, and the variances of $X$ and $Y$ are summed on the same grid. A non-zero
entry where either variance is zero or negative is `covariance_without_variance`
(defect). On the rest, $\rho$ of {eq}`eq-covariance-checks-rho` with $C_{ii}$ and
$C_{jj}$ from the two self blocks is graded as for a self block. When the block holds an
LS=1 triangle, the summary says whether the stored triangle alone was within bounds, so
that the mirror is what created the violation, or not.

### The joint matrix

Self blocks that are PSD and correlations within $\pm1$ everywhere do not make the
joint matrix PSD: three reactions can be pairwise compatible and not jointly. In MF 31
and 33 every MT, and in MF 34 every pair (MT, $L$), is a node; nodes linked by a cross
block form connected components, and each component's joint matrix $J$ is assembled from
its self blocks and cross blocks, each on the node's own grid. In MF 34, an $a_0$ whose
own $(0, 0)$ block is absent or null takes MF 33's self block of that MT (ENDF-6 §34.3),
so a magnitude-shape cross block joins MF 33 and MF 34. A node whose self block mixes
relative and absolute parts is left out; when a cross block is relative and its nodes
absolute (or the reverse), the second of its two nodes is left out with all its blocks,
and the note `joint_node_left_out` names that node, so a joint check that passes is not
read as covering a reaction it never saw. Dropping only the cross block could make the
joint look indefinite, and dropping a whole node (a principal submatrix) cannot.

What the cross blocks add is graded, not what the self blocks already hold. With
$\lambda_\text{min}(S_m)$ the smallest eigenvalue of self block $m$,

$$
e = \min\big(\min_m \lambda_\text{min}(S_m),\, 0\big) - \min\big(\lambda_\text{min}(J),\, 0\big),
\qquad r_J = \frac{e}{\lambda_\text{max}(J)},
$$ (eq-covariance-checks-joint)

computed without $J$'s zero rows. Nothing is reported if $e \le \varepsilon_\text{ENDF}(J)$
or $r_J < 10^{-6}$; otherwise `joint_not_positive_semidefinite` is a warning up to
$10^{-3}$ and a defect above, a warning becomes a note when $J$'s correlations sit on a
lattice that can explain $e$ (step 2, applied to $J$ as one matrix), and the clipping
regrade of step 3 applies. The evidence names the cross blocks that are already
indefinite with their two self blocks alone (up to sixty are tried). A component with
more than 6000 rows is not decomposed (`psd_not_evaluated`, a note).

### Magnitude of the uncertainty

The size of a variance is judged against the central values, and only through bounds
that hold:

- **MF 33.** The central value of bin $i$ is the $1/E$-weighted average

  $$
  \bar\sigma_i = \frac{\int_{E_i}^{E_{i+1}} \sigma(E)\, \frac{dE}{E}}{\int_{E_i}^{E_{i+1}} \frac{dE}{E}},
  $$ (eq-covariance-checks-xs-average)

  integrated in closed form under the table's own interpolation laws (a flat average for
  a bin whose lower edge is not positive). $\sigma(E)$ comes from the cross-sections
  given (a PENDF), at every energy; without them, from MF 3 above the highest upper
  limit EH of the resolved and unresolved ranges of MF 2, where MF 3 is the
  cross-section and not a background. Without MF 2 it is not evaluated. A summation MT
  missing from MF 3 is the sum of its partials. With $\sigma^\text{rel}_i$ the relative
  standard deviation (or $\sqrt{C_{ii}}/\bar\sigma_i$ for an absolute block), a bin with
  $\sigma^\text{rel}_i > 1$ is a note (`relative_uncertainty_above_one`), with the
  probability that a normal draw there is negative, $\Phi(-1/\sigma^\text{rel}_i)$. A
  bin with $\sigma^\text{rel}_i > 10$ where $\bar\sigma_i \ge 0.01 \max_j \bar\sigma_j$
  (outside a threshold region, where $\bar\sigma \to 0$ makes any ratio legitimate) is a
  warning (`implausible_relative_uncertainty`). A cross-section has no upper bound, so no
  value of $\sigma^\text{rel}$ is a defect.
- **MF 31.** The same, with $\bar\nu$ of MF 1 averaged over each bin from five
  logarithmically spaced points.
- **MF 34.** For $\ell \ge 1$, $f(\mu) \ge 0$ bounds every normalised coefficient,
  $|a_\ell| \le 1$, and a quantity confined to an interval of width 2 has a standard
  deviation of at most 1 whatever its distribution (Popoviciu's inequality, TODO(ref)).
  A relative $\sigma$ is made absolute with the smallest magnitude the bin can mean,
  $\min(|\langle a_\ell \rangle_i|, \min_{E \in \text{bin}} |a_\ell(E)|)$, both exact from
  MF 4's table, so that a breach is certain and not an artefact of averaging. A bin with
  $\sigma(a_\ell) > 1$ is `variance_exceeds_physical_bound`, a defect. $a_0$ is not
  judged: it stands for the integrated cross-section.
- **Zero central values.** A variance over a bin where the central value is zero over the
  whole bin (`variance_where_central_value_is_zero`) is a note when it is relative (every
  sample stays at zero) and a warning when it is absolute (a normal draw there is
  negative half the time).
- **Coverage.** A covariance grid that starts above the first non-zero MF 3 energy (and
  above $10^{-5}$ eV) or stops below the last is a note (`grid_coverage`).

A block that mixes absolute (LB=0, 8, 9) and relative records needs $\bar\sigma_i
\bar\sigma_j$ to be summed. Without cross-sections, only its relative records are
summed, and a note says so (`mixed_needs_cross_sections`). In MF 34, where kika sums a
mixed block as relative, the same case is a warning (`mixed_absolute_relative`). A file
whose central values could not be found gets one note listing the MTs
(`central_values_unavailable`).

### MF 35

Each band is checked as a self block, then:

- **The sum rule.** $\sum_i P_i = 1$ for every spectrum, so $C\mathbf{1} = 0$. The
  residual $\max_i |\sum_j C_{ij}| / \max|C|$ above $10^{-2}$ is `sum_rule_violated`,
  a defect.
- **A bound that holds.** $0 \le P_i \le 1$, so $\operatorname{var}(P_i) \le P_i(1-P_i)$
  (the Bhatia-Davis inequality, TODO(ref)), never above $1/4$. With MF 5 a single
  tabulated (LF=1) spectrum, the bound of group $i$ is $\sqrt{\max_k P_{ik}(1 - P_{ik})}$
  over the incident energies $k$ of the band, the most lenient choice; otherwise
  $\sigma(P_i) \le 1/2$. A group above it is `variance_exceeds_physical_bound` (defect).
  A variance on a group MF 5 gives no probability at any incident energy of the band is
  a warning (`variance_where_distribution_is_zero`).
- **The bands.** Counted against NK; empty (a defect when $E_1 > E_2$, a warning when
  equal), in gaps (warning) or overlapping (defect), each with a relative tolerance of
  $10^{-6}$ on the edges; and covering MF 5's incident energies (`band_coverage`, note).
  An MF 35 section with no MF 5 is `missing_distribution` (defect).

Because a band is singular by construction, its $r$ alone misleads; the clipping regrade
of step 3 is what grades a light warning.

### MF 32

The rows are resonance parameters, not energy bins, and their scales differ by orders of
magnitude, so $r$ of the covariance measures the scales and not the matrix. PSD is
therefore judged on the **correlation matrix**, with the same three levels and the
clipping regrade. For LCOMP=2, which stores the correlations as NDIGIT-digit integers,
the rounding allowance is step 2 with $q = 10^{-\mathrm{NDIGIT}}$; otherwise it is
{eq}`eq-covariance-checks-ulp`. Before that:

- INTG records whose row or column lies outside the $\mathrm{NNN} \times \mathrm{NNN}$
  matrix (`correlation_index_out_of_range`, defect), and records the decoder skips
  (`decode_error`) or keeps without decoding (`not_decoded`, note);
- agreement with MF 2: a range of the same limits (or, failing that, of the same LRU and
  overlapping) must exist (`range_not_in_mf2`, warning), with the same limits to a
  relative $10^{-5}$ (`range_limits_differ_from_mf2`, note) and the same formalism
  (`formalism_mismatch`, defect; only LRU is compared in the unresolved range, which has
  one covariance format whatever MF 2's LRF, ENDF-6 §32.2.4);
- every resonance energy of the covariance must be in MF 2 to a relative $10^{-5}$
  (`parameters_not_in_mf2`, warning), and its widths must equal MF 2's to $10^{-5}$
  (`widths_differ_from_mf2`): a warning when some width differs by at least 0.1 of its
  own $\sigma$, a note otherwise (a copy with fewer digits). LRF=7 compares channel by
  channel within the spin group;
- an uncertainty on the spin $J$, which kika drops (`spin_uncertainty`, warning).

The variance checks are those of a self block. A width is unbounded above and, in LRF=3
and 7, signed, so $\sigma^\text{rel} > 1$ is only a note. A matrix with more than 6000
parameters with an uncertainty is not decomposed (`psd_not_evaluated`).

### MF 40

An MF 40 block is an MF 33 block named by two (MT, LFS) pairs, so every structural and
value check of MF 33 applies, with the final states in the location. kika does not read
MF 10, so no magnitude is judged, and a mixed block is checked on its relative part; one
note says so. A block with data of another file (XMF1 other than 0 or 10) is noted and
skipped.

### A library

`check_covariance_library` takes a directory (its files matching `*.endf`, `*.jeff`,
`*.txt`, `*.dat`, optionally recursively), a list of tapes, or a mapping from each tape
to its own covariance files, and walks the tapes in series. For each tape it reads the
MF/MT index, parses only the covariance files asked for that are present and the files
that give them central values (MF 1 for MF 31, MF 2 for MF 32, MF 2 and 3 for MF 33,
MF 4 and 33 for MF 34, MF 5 for MF 35, none for MF 40), runs `check_covariances`, and
releases the tape before the next. A tape without the requested covariances is a row
with `has_covariances=False` and nothing parsed. A tape that cannot be read, is not
ENDF, or whose check raises, is a row with its error (`read: …` or `check: …`), and the
walk goes on. No PENDF is attached, so mixed MF 33 blocks are checked on their relative
part. A caller can stop the walk between tapes (`should_stop`); the report then holds the
tapes done.

### Reports

A tape report and a library report are written by one renderer, in three forms:

- `to_dict(level)`: plain data, safe for JSON and msgpack (numpy values become Python
  numbers, NaN and infinities `None`), with `kind`, `schema`, the kika version, the
  tape or library identity, totals by level, counts by (MF, level, check) over **every**
  finding, the findings at `level` or worse (default `note`), and the legend of the
  checks found;
- `to_markdown(path, level)` and `to_html(path, level)`: self-contained pages (default
  `warn`), in the order verdict, identity, summary by check (and by tape), findings
  grouped by check, method and thresholds, and the legend of the checks shown. The
  level applies to the whole page: a level left out is neither listed nor counted
  there, except for one line saying how many findings were left out;
- for a library, `write(out_dir, level)`: three TSV tables, `summary`, `files` and
  `findings` (the last at `warn` or worse by default, evidence as JSON).

The method section and the legend are built from the constants the checks use and from
one table of descriptions, so they cannot drift from the code. The summaries are ASCII
in `str()`, `to_dataframe` and the TSV (a Windows console and a spreadsheet mangle
anything else) and carry Greek letters and math signs in the dictionary and the pages.

## Conventions and choices

- **Layer 1 only informs.** Nothing here raises on a bad section, changes it or proposes
  a remedy. The same file is used for different things (sampled, processed by NJOY,
  compared between libraries), and whether a finding matters depends on the use. Repairs
  belong to the pre-flight of the matrices actually drawn (`kika.cov.conditioning`),
  which runs on the assembled covariance. By then the LB and LS records a block came from
  are gone, so the pre-flight cannot say *why* a block is indefinite; the sampler traces
  each of its findings back to the layer-1 findings of the sections involved
  (`kika.sampling.section_checks`).
- **The level says how bad a finding is, not what to do.** A report has no truthiness:
  "is this file fine" depends on the levels the caller cares about (`report.defects`,
  `report.at_least("warn")`).
- **Values are checked as kika sums them.** Each block is assembled on its union grid by
  the same code that builds a model covariance, so a check cannot pass a block that kika
  would then read differently.
- **Only bounds that hold are defects.** A cross-section is unbounded above, so its
  $\sigma^\text{rel}$ is never a defect; $\sigma(a_\ell) \le 1$ and
  $\operatorname{var}(P) \le P(1-P)$ are mathematical, so breaking them is.
- **The warning edge for $\sigma^\text{rel}$ is absolute and outside threshold regions.**
  Large relative uncertainties near a threshold are legitimate, and an edge on
  $\sigma^\text{rel}$ alone flags them. The value 10 is set above what the major
  libraries state outside threshold regions. TODO(check): that survey is not
  reproducible from a test in the repository.
- **Rounding is measured, not assumed.** A negative eigenvalue is downgraded only when
  {eq}`eq-covariance-checks-ulp` or the quantised-correlation bound can produce it.
- **A light PSD warning is graded by consequence.** $r$ places a block between rounding
  and a defect, but in the light band it says nothing about what repairing it costs;
  $\delta_\sigma$ does.
- **MF 32 on the correlation matrix.** Its parameters mix units, and $r$ of the
  covariance would measure the units.
- **The joint grades only what the cross blocks add**, so a self-block defect is not
  reported twice.
- **MAT1 equal to the file's own MAT is the same material**, as kika reads it everywhere
  else; it is noted, not refused.
- **Against FUDGE** (6.14.0, from its source, not from runs). FUDGE checks covariances
  through the warning classes its `check()` methods raise (`fudge/warning.py`), with five
  levels. Every one of them has a counterpart here: a non-monotonic grid or a dimension
  mismatch (`grid_not_increasing`, `nt_mismatch`, `count_mismatch`), an empty matrix
  (`inert_rows`, by rows), a negative variance or a correlation out of range
  (`negative_variance`, `correlation_out_of_bounds`), negative eigenvalues
  (`not_positive_semidefinite`), a cyclic dependency or an unresolved link (`nc_chained`,
  `unresolved_reference`), a variance above 1000 % (`relative_uncertainty_above_one`,
  `implausible_relative_uncertainty`). Four checks here come from FUDGE's ENDF-6
  translation: MF 32 widths against MF 2 (`widths_differ_from_mf2`), an uncertainty on a
  spin (`spin_uncertainty`), a last value of LB 0-4, 8, 9 that is not zero
  (`trailing_value_not_zero`) and a covariance without its central values
  (`missing_central_values`). Where kika goes further: FUDGE tests $\lvert\rho\rvert \le 1$
  in MF 32 only; it does not test the matrix with its cross blocks (a `FIXME` in its
  source), which `joint_not_positive_semidefinite` does; and it judges positivity against
  an absolute eigenvalue of $-10^{-6}$, where kika normalises by the block and separates
  round-off from defects ({eq}`eq-covariance-checks-ulp`).
- **A library is checked one tape at a time**, so a library of hundreds of tapes runs in
  the memory of the largest one; failures are rows, because in a census they are
  findings too.

## Limits and failure modes

Refused: `check_covariance_library` raises `FileNotFoundError` for a path that is not a
directory, and `ValueError` when `mf` (or a tape's own entry of a mapping) names none of
31, 32, 33, 34, 35, 40, before the walk starts. `check_covariances` ignores files outside
that list. `CovarianceFinding` refuses a level other than `note`, `warn`, `defect`
(`ValueError`).

Reported instead of done: magnitudes without central values
(`central_values_unavailable`); mixed blocks checked on their relative part; MF 40
magnitudes (MF 10 is not read); blocks with another material; MF 34 LT > 0 records; MF 32
parts the decoder keeps verbatim; joint matrices and MF 32 matrices above 6000 rows; LTY=1-3
ratios to standards, which kika does not resolve.

Limits of the method: the checks see one file, so a covariance with another material is
never evaluated; MF 30 is not checked (it has no parser); NC derivations are checked for
their references but not assembled, so an MF 33 self block given only by NC LTY=0 is not
value-checked; the MF 33 magnitudes below EH need cross-sections from a PENDF; the
thresholds $10^{-6}$, $10^{-3}$, 2 %, 25 %, 10 and $10^{-2}$ (MF 35) are set from the
major libraries, and a new library may sit on a different side of them.

## Validation

The tests below passed on 2026-10-11 (114 passed, 2 skipped) unless marked.

**One rule at a time.** `kika/endf/tests/test_covariance_checks.py` builds synthetic
sections from the parser's record classes with one fault planted and nothing else wrong.
Among others: a clean block has nothing above a note
(`::test_a_clean_block_has_nothing_above_a_note`); a $\rho$ of 2 is a defect and one of
$1 + 10^{-5}$ a note (`::test_rho_above_one_in_a_self_block_and_rounding_below_the_line`);
$\sigma^\text{rel} = 2$ on a flat cross-section is a note with a probability of a negative
normal draw of 0.31 (`::test_sigma_rel_above_one_is_a_note_with_the_chance_of_a_negative_draw`);
$\sigma^\text{rel} = 20$ warns outside a threshold region and not inside one
(`::test_sigma_rel_above_ten_warns_only_outside_a_threshold_region`); the PSD levels at
$r = 10^{-4}$ (regraded to a note) and $10^{-2}$ (defect) (`::test_psd_levels`); a light
warning whose clipping changes $\sigma$ by about 10 % stays a warning and one of about
50 % becomes a defect, with $\delta_\sigma$ equal to
$\sqrt{1 + \text{depth}/(2a)} - 1$ to $10^{-6}$
(`::test_a_light_warning_is_graded_by_what_clipping_does_to_sigma`); an eigenvalue inside
{eq}`eq-covariance-checks-ulp` is a note (`::test_an_eigenvalue_inside_endf_rounding_is_a_note`);
correlations rounded to $10^{-3}$ explain a warning-tier eigenvalue
(`::test_correlations_rounded_to_a_lattice_explain_a_small_negative_eigenvalue`);
pairwise-compatible blocks make an indefinite joint
(`::test_pairwise_compatible_blocks_can_make_an_indefinite_joint`); a reaction whose self
block is absolute while its cross blocks are relative is left out of the joint and named
(`::test_a_node_left_out_of_the_joint_is_reported`); $\sigma(a_1) = 2$ is a
defect (`::test_mf34_sigma_wider_than_the_physical_range_of_a_l`). MF 35 and MF 40 are in
`kika/endf/tests/test_covariance_checks_mf35_mf40.py`.

**Real evaluations.** `kika/endf/tests/test_covariance_checks_on_real_cuts.py` checks
committed cuts of real tapes, each a closed verbatim extract:

| Tape and section | What the checks find | Test |
|---|---|---|
| JEFF-4.0 Ne-20 MF 34 MT 2 | 10 LS=1 cross blocks; 7 with $\lvert\rho\rvert > 1$, worst 4.6 where the stored triangle alone is 0.88; joint $r_J$ = 0.17 | `::test_ne20_mf34_the_mirror_of_an_ls1_triangle_is_what_breaks_rho` |
| JEFF-4.0 W-186 MF 34 MT 51 $L=1$ | $\lvert\rho\rvert$ up to 49 (112 pairs above $1 + 10^{-3}$), 10 negative variances, $r$ = 0.41, already in NI[0] | `::test_w186_mf34_rho_49_a_negative_variance_and_an_indefinite_block` |
| JEFF-4.0 Fe-57 MF 33 | 5 NC LTY=0 references to MTs with no section | `::test_fe57_mf33_sum_rules_that_name_reactions_with_no_section` |
| ENDF/B-VIII.1 Si-28 MF 33 | MT 1 derived from MT 4 over a range where MT 4 is itself derived | `::test_si28_mf33_a_constituent_derived_over_the_same_range` |
| JEFF-4.0 Hf-176 MF 33 MT 107 | $\sigma^\text{rel}$ = 11 at 9-10 MeV where $\bar\sigma$ is 1.3 % of its maximum (warning); 14 at 5-5.5 MeV, at the threshold, a note | `::test_hf176_mf33_a_repeated_point_and_an_implausible_sigma_rel` |
| JENDL-5 Fe-56 MF 33 and 34 | only notes: 50 indefinite blocks, all explained by rounding | `::test_fe56_jendl5_the_clean_case_has_only_notes` |
| JEFF-4.0 K-41 MF 32 | an INTG record at row 201 with NNN = 93 | `::test_k41_mf32_an_intg_record_outside_the_matrix` |
| ENDF/B-VIII.1 Th-232 MF 32 | correlation matrix with $\lambda_\text{min}$ = −0.66, past its NDIGIT rounding | `::test_th232_mf32_a_correlation_matrix_that_cannot_be_sampled` |
| JEFF-4.0 U-239 MF 35 | 21 bands break the sum rule (residual above 0.9), $\sigma(P)$ up to 2.9·10³ | `::test_u239_mf35_not_the_covariance_of_a_normalised_spectrum` |
| ENDF/B-VIII.1 Cf-252 MF 35 with MF 5 | nothing above a note; $r$ up to 1.4·10⁻⁴, $\delta_\sigma$ below 0.4 % | `::test_cf252_mf35_the_reference_spectrum_keeps_its_sum_rule_and_bound` |

On the full O-16 tape of ENDF/B-VIII.1 (MF 33 MT 2 a defect with $r \approx 1$, already
indefinite in its LB=5) and the full Fe-56 tape of JENDL-5 (no defect, every PSD finding
a note) the checks are pinned by
`kika/endf/tests/test_covariance_checks.py::test_o16_b81_mt2_is_indefinite_already_in_its_lb5`
and `::test_jendl5_fe56_has_no_defect_and_its_negative_eigenvalues_are_rounding`.
TODO(check): these need the full tapes and `--deep`; not re-run for this page.

**The library walk.** `kika/endf/tests/test_covariance_library_check.py`: every tape of a
directory is checked and gives the counts of a check of the tape alone
(`::test_every_tape_is_checked_and_the_report_matches_one_by_one`); a broken tape and a
tape without covariances are rows, not errors
(`::test_a_broken_tape_and_a_tape_without_covariances_are_rows_not_errors`); only the
present covariances and their support files are parsed
(`::test_only_present_covariances_and_their_support_are_parsed`,
`::test_no_covariances_do_not_invoke_any_mf_parser`).

**The reports.** `kika/endf/tests/test_covariance_reports_export.py`: every check the
code can emit has a description and every description is emitted
(`::test_every_check_is_described`); on the real cuts every finding's MF and level is one
its description lists (`::test_the_legend_agrees_with_what_the_cuts_give`); the
dictionaries are plain data (`::test_report_to_dict_is_plain_data`,
`::test_library_to_dict_is_plain_data`).

## Diagnostics

A `CovarianceCheckReport` iterates over its findings and offers `defects`, `warnings`,
`notes`, `at_least(level)`, `by_check(name)`, `counts()`, `summary()` (one row per MF,
level and check) and `to_dataframe()`. A location prints as, for example,
`MF33 MT2xMT102 NI[0] LB=5 LS=1`. The evidence keys that matter most:

- `max_abs_rho`, `n_above`, `at`: the worst $|\rho|$, how many pairs exceed
  $1 + 10^{-3}$ (each pair of a self block once, every entry of a cross block), and the
  two bins where the worst sits; `stored_triangle_max_abs_rho`
  when an LS=1 triangle is involved. **A value well above 1** means the block cannot be
  sampled as stated.
- `lambda_min`, `lambda_max`, `ratio`, `n_negative`, `rounding_bound`,
  `rho_quantum`, `quantised_rounding_bound`: the inputs of the PSD grade.
  `sigma_change_if_clipped` is $\delta_\sigma$ of {eq}`eq-covariance-checks-impact`:
  **above a few per cent**, any PSD repair changes the stated uncertainties, and the
  repair has to be chosen with care.
- `records_indefinite_alone` (self blocks) and `pairs_indefinite_alone`, `excess_ratio`
  (joint): where the indefiniteness comes from. `node` and `relative` (the relative flag
  of the cross block and of its two self blocks) for `joint_node_left_out`.
- `worst_rel`, `worst_bin`, `central_over_max`, `p_negative_if_normal` for magnitudes;
  `worst_sigma_abs`, `worst_sigma`, `worst_bound` for the physical bounds;
  `row_sum_residual` for MF 35.
- `mts` in `central_values_unavailable`: the reactions whose magnitudes were not judged.

A `CovarianceLibraryReport` adds `checked`, `failed`, `tapes_with(check, level, mf)`,
`report(name)`, and `summary()` with the number of findings and of tapes per row.
**Read `failed`**: a tape that could not be read was not checked at all.

## Implementation

- `kika.endf.checks.covariances`: `check_covariances`, the dispatch by file, and the
  checks of MF 31, 33 and 34 (structure, completeness, values, PSD, the joint matrix,
  magnitudes).
- `kika.endf.checks.mf32`, `kika.endf.checks.mf35`, `kika.endf.checks.mf40`: the
  checks specific to those files, reusing the pieces of MF 33; MF 32 decodes with the
  model's reader, `kika.endf.model_adapter.parameter_covariances.decodeMF32MT`.
- `kika.endf.checks.findings`: the levels, `CovarianceLocation`, `CovarianceFinding`,
  `CovarianceCheckReport`.
- `kika.endf.checks.library`: `check_covariance_library`, `TapeCheck`,
  `CovarianceLibraryReport`.
- `kika.endf.checks.descriptions`: the table of what each check means and when it gives
  each level; `kika.endf.checks.export`: the plain-data, Markdown and HTML renderings and
  the method section; `kika.endf.checks.symbols`: the ASCII to Greek conversion.
- The summing of a block on a union grid and the central-value averages are the
  section classes' own (`kika.endf.classes.mf33.mf33`, the MF 4 classes).
- `kika.sampling.section_checks`: the layer-1 findings of the sections a run perturbs,
  and the attribution of each pre-flight finding to them.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538. Cited as ENDF-6: Ch. 31-35 and 40, §33.2, §33.3, §34.1-§34.3,
  §32.2.4.
- N. J. Higham, "Computing a nearest symmetric positive semidefinite matrix", *Linear
  Algebra and its Applications* 103, 103-118 (1988).
- `TODO(ref)`: Popoviciu's and the Bhatia-Davis inequalities on the variance of a
  bounded variable; the spectral norm of a random symmetric matrix (Wigner's semicircle).
