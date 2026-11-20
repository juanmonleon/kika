"""Draw correlated realisations of covariance blocks.

``draw_samples`` decomposes each covariance block, draws uncorrelated standard
normals and imposes the correlation. It knows nothing about MT numbers, energy
groups, Legendre orders, ENDF, ACE or NJOY: a block is a ``(key, matrix)`` pair
and the key can be any hashable.

The format-level sampler ``kika.sampling.generators.generate_samples`` adds three
steps that are wrong for an absolute covariance such as a fission-spectrum one,
and this module has none of them:

* a mask that drops rows with no stated variance before the decomposition, which
  breaks ``C·1 = 0`` on the retained subspace and with it the normalisation of a
  linearly perturbed spectrum;
* a rescale of statistical outliers (any bin whose variance is far above the
  median), which fires on the genuine spread of a spectrum's variance;
* a cast to float32, which loses absolute deltas far below float32 epsilon once
  stored as ``1 + δ``. The output here is float64 unless *dtype* says otherwise.

Blocks are independent: the draw is block-diagonal and carries no covariance
between two blocks. Under Sobol all blocks of one call share a single sequence and
take disjoint columns of it. Under ``random`` and ``lhs`` block *i* is seeded with
``seed + i × BLOCK_SEED_STRIDE``. See :func:`realise_blocks`.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np

from ..cov.conditioning import as_blocks
from ..cov.decomposition import eigen_decomposition, svd_decomposition
# The QMC draw, shared rather than copied. It belongs here and will move when
# ``generate_samples`` is migrated onto this module; today the dependency runs
# core → generators and generators does not import core, so there is no cycle.
from .generators import _uncorrelated

__all__ = ["draw_samples", "draw_prepared_samples", "prepare_blocks",
           "realise_blocks", "BLOCK_SEED_STRIDE"]

#: Added to the base seed once per block index under ``random``/``lhs`` (Sobol
#: shares one sequence instead; see :func:`realise_blocks`). A prime, so that two callers
#: whose seeds differ by a small integer do not end up sharing draws on some
#: pair of blocks. Recorded in the diagnostics of every block, because a
#: reproduction that cannot say which seed produced which block is not one.
BLOCK_SEED_STRIDE = 1009

#: Singular values at or below ``null_tol × S.max()`` count as null directions.
DEFAULT_NULL_TOL = 1e-10


#: What a caller may hand in as *blocks*, resolved to ``(key, matrix)`` pairs.
#:
#: Defined in :mod:`kika.cov.conditioning` rather than here so that inspecting
#: a set of blocks and sampling it agree on what "a set of blocks" is by
#: construction. Two copies of this would drift the day one of them learned a
#: fourth input shape, and the failure would be that a pre-flight passed on a
#: different arrangement of matrices than the one that got sampled.
_as_blocks = as_blocks


def draw_samples(
    blocks,
    n_samples: int,
    *,
    space: str = "linear",
    returns: str = "deltas",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    psd_method: str = "auto",
    null_tol: Optional[float] = DEFAULT_NULL_TOL,
    negative_directions: str = "zero",
    dtype=np.float64,
    verbose: bool = True,
    logger=None,
) -> Tuple[Dict[Hashable, np.ndarray], Dict[Hashable, Dict[str, Any]]]:
    """Draw *n_samples* correlated realisations of each covariance block.

    Each block is decomposed into a factor ``L`` with one column per retained
    direction, ``Z`` of shape ``(n_samples, n_retained)`` standard normals are
    drawn, and the realisation is ``Y = Z L^T``. The blocks are independent of each other (see :func:`realise_blocks`).
    A matrix is a square two-dimensional array; entries that are not finite are
    set to 0 before the decomposition and counted in the diagnostics.

    Parameters
    ----------
    blocks
        ``{key: matrix}``, a sequence of ``(key, matrix)`` pairs, or a bare
        sequence of matrices (then the key is the index). The key is opaque —
        ``(isotope, mt)`` for a cross section, ``(isotope, mt, order)`` for a
        Legendre block, ``(isotope, mt, band)`` for a PFNS band. The order of the
        blocks sets the seed of each one (see *seed*).
    n_samples
        Number of realisations per block.
    space
        ``"linear"`` draws ``Y = Z L^T`` from the matrix as given.
        ``"log"`` takes the matrix as a **relative** covariance Σ and draws
        lognormal factors that reproduce it, up to the repair of ``psd_method``
        and the dropped null directions: ``ln f`` is normal with
        covariance ``log1p(Σ)`` and mean ``-diag(log1p Σ)/2``, so
        ``E[f] = 1`` and ``Cov(f) = Σ``. Only meaningful together with
        ``returns="factors"``; ``"log"`` with ``returns="deltas"`` is refused.
        A caller whose matrix is already ``log1p(Σ)``
        (it conditioned it in that space) calls :func:`draw_prepared_samples`.
    returns
        ``"deltas"`` returns ``Y`` — absolute departures, which is what an
        absolute covariance describes. ``"factors"`` returns ``Y + 1`` (linear)
        or ``exp(Y + m)`` (log), which is what a relative covariance describes
        and what the ENDF/ACE perturbation drivers multiply by.
    decomposition_method
        ``"svd"`` or ``"eigen"`` (any case). **Cholesky is refused**: these
        matrices are built from a handful of model parameters and are severely
        rank deficient by construction — two thirds of a typical MF35 band is
        null space — so a Cholesky factor either fails or is meaningless. With
        ``"svd"`` the factor is built from the singular vectors, with
        ``"eigen"`` from the eigenvectors sorted by decreasing eigenvalue.
    sampling_method
        ``"sobol"`` (scrambled Sobol sequence), ``"lhs"`` (Latin hypercube) or
        ``"random"`` (pseudo-random), in any case. The Sobol and Latin-hypercube
        uniforms become standard normals through the inverse normal CDF.
    seed
        Base seed; None draws a different ensemble at each call. Under Sobol
        every block uses it, on its own columns of one sequence (recorded as
        ``sobol_columns``); under ``random``/``lhs`` block *i* (its position in
        *blocks*) uses ``seed + i × BLOCK_SEED_STRIDE``.
    psd_method
        Passed through to the decomposition, which repairs the matrix before it
        is factorised: ``"auto"``, ``"higham"``, ``"clip"``, ``"clip_rescale"``
        or ``"none"``. ``"auto"`` routes the tiny
        negative eigenvalues these matrices carry to ``clip``, which rebuilds
        ``V·clip(Λ,0)·Vᵀ`` and so **preserves the eigenvectors** — including
        the near-null direction that carries a sum rule.
    null_tol
        Threshold for null directions: the singular values (the eigenvalues,
        for ``"eigen"``) at or below ``null_tol`` × the largest one, after the
        repair. They are counted and reported, never repaired: the rank
        deficiency is a property of the evaluation, and filling it in would
        invent uncertainty the evaluator did not claim. They are left out of the
        draw, so the number of normals drawn per block is the retained rank.

        ``None`` retains **every** direction, null ones included (a null
        direction then carries a vanishing amplitude), and is there for one
        purpose: reproducing a draw made without truncation. Truncation changes
        the dimension of the quasi-random sequence from *n* to the rank, so it
        moves every drawn column.

        It is an escape hatch for gates, not a modelling option. The diagnostics
        still count ``n_null`` and ``rank`` at :data:`~kika.cov.conditioning.DEFAULT_NULL_TOL`, so a
        run drawing in null directions says so.

        **The rank reported here and the rank a pre-flight reports can differ.**
        This one is taken on the singular
        values the draw actually decomposes — *after* ``psd_method`` has acted.
        With ``negative_directions="fold"`` an SVD folds a negative eigenvalue
        to its magnitude, so a direction the file states as negative-variance
        survives as a retained direction here while
        ``conditioning.inspect_blocks``, which reads the eigenvalues of the
        matrix as stated, counts it null.
        With the default ``"zero"`` those directions count null here too.
    negative_directions
        What the draw does with a direction the matrix it decomposes states
        with a **negative** eigenvalue — one ``psd_method`` left in place
        (``"none"``), or numerical residue after a repair. ``"zero"`` (the
        default) gives it zero variance, which is what the eigen path has
        always done and the closest a draw can come to "as stated": a negative
        variance cannot be realised. ``"fold"`` keeps what an SVD returns, the
        magnitude ``|λ|``, i.e. draws from ``V|Λ|Vᵀ`` and adds variance in a
        direction the evaluation does not state. It exists only so the frozen
        format-level samplers (``perturb_ENDF_files`` and
        ``kika.sampling.multigroup_draw.draw_relative_factors``) keep
        drawing exactly what produced the published results; nothing new
        should ask for it. A zeroed direction counts as null in ``n_null``.
        With ``decomposition_method="eigen"`` a negative eigenvalue always has
        zero variance and this argument changes nothing.
    dtype
        Output dtype. float64 by default and on purpose; see the module
        docstring.
    verbose, logger
        The decomposition and the repair named by *psd_method* report their
        progress when ``verbose`` is true: it is printed to the console and sent
        to ``logger`` (an object with an ``info`` method, such as a
        ``logging.Logger``; None for none).

    Returns
    -------
    samples : dict
        ``{key: array}``, each array of shape ``(n_samples, m)`` with ``m`` the
        size of the block and one row per realisation, of dtype *dtype*. With
        ``returns="deltas"`` the rows are ``Y``; with ``"factors"`` they are
        ``Y + 1`` (``space="linear"``) or ``exp(Y - diag/2)``
        (``space="log"``, ``diag`` the diagonal of ``log1p(Σ)``).
    diagnostics : dict
        ``{key: dict}`` with, for each block: ``n``, ``n_null``, ``rank``
        (``n - n_null``), ``null_fraction``, ``n_non_finite_zeroed``,
        ``psd_method``, ``decomposition_method``, ``space``, ``returns``,
        ``label`` (``str(key)``), ``min_over_max_eigenvalue`` (of the matrix
        before the repair), ``realised_covariance_error``, ``seed``,
        ``block_index`` and, under Sobol only, ``sobol_columns``. The matrix of
        these diagnostics is the one that is decomposed: the matrix as given for
        ``space="linear"``, ``log1p(Σ)`` for ``space="log"``, with non-finite
        entries set to 0. The error is
        ``||C_realised - Λ|| / ||Λ||`` in the Frobenius norm, taken on the
        eigenvectors of that matrix whose eigenvalue exceeds
        ``null_tol`` × the largest (``Λ`` their eigenvalues; with ``null_tol``
        None, ``DEFAULT_NULL_TOL``, 1e-10, is used), with the sample
        covariance of ``Y``; it is ``nan`` for a single sample or a matrix with
        no positive eigenvalue. ``seed`` is the base seed under Sobol and the
        seed of that block otherwise.

    Raises
    ------
    ValueError
        If *returns* is not ``"deltas"`` or ``"factors"``; *space* is not
        ``"linear"`` or ``"log"``; ``space="log"`` with ``returns="deltas"``;
        *decomposition_method* is ``"cholesky"`` or neither ``"svd"`` nor
        ``"eigen"``; *negative_directions* is neither ``"zero"`` nor
        ``"fold"``; *psd_method* is not one of the five above; a block is not a
        square two-dimensional matrix; or *sampling_method* is not one of the
        three above.
    """
    method = _validated_method(space, returns, decomposition_method)
    _validated_negative_directions(negative_directions)
    items = _as_blocks(blocks)
    if space == "log":
        # A relative covariance Σ is the covariance of f = exp(g) with
        # Cov(g) = log1p(Σ): Cov(f)_ij = exp(Cov(g)_ij) - 1. Drawing g from Σ
        # itself would give Cov(f) = expm1(Σ), 14 % too wide at 50 %.
        items = [(key, np.log1p(np.asarray(matrix, dtype=float)))
                 for key, matrix in items]
    return draw_prepared_samples(
        items, n_samples, space=space, returns=returns,
        decomposition_method=method, sampling_method=sampling_method,
        seed=seed, psd_method=psd_method, null_tol=null_tol,
        negative_directions=negative_directions, dtype=dtype,
        verbose=verbose, logger=logger)


def draw_prepared_samples(
    blocks,
    n_samples: int,
    *,
    space: str = "linear",
    returns: str = "deltas",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    psd_method: str = "auto",
    null_tol: Optional[float] = DEFAULT_NULL_TOL,
    negative_directions: str = "zero",
    dtype=np.float64,
    verbose: bool = True,
    logger=None,
) -> Tuple[Dict[Hashable, np.ndarray], Dict[Hashable, Dict[str, Any]]]:
    """:func:`draw_samples` for blocks already in the space the draw happens in.

    With ``space="log"`` each matrix is taken as ``log1p(Σ)`` -- the covariance
    of ``ln f`` -- and not transformed again. That is what a caller that
    conditions the matrix in log space holds
    (``kika.sampling.multigroup_draw.draw_relative_factors``). Everything
    else is :func:`draw_samples`'.

    Parameters
    ----------
    blocks, n_samples, space, returns, decomposition_method, sampling_method, seed, psd_method, null_tol, negative_directions, dtype, verbose, logger
        As in :func:`draw_samples`. With ``space="log"`` and ``returns="factors"``
        the realisation is ``exp(Y - diag/2)``, ``diag`` being the diagonal of the
        matrix handed in.

    Returns
    -------
    samples, diagnostics : tuple of dict
        As in :func:`draw_samples`: ``{key: (n_samples, m) array}`` and
        ``{key: diagnostics dict}``.

    Raises
    ------
    ValueError
        Under the conditions of :func:`draw_samples`.
    """
    prepared = prepare_blocks(
        blocks, space=space, returns=returns,
        decomposition_method=decomposition_method, psd_method=psd_method,
        null_tol=null_tol, negative_directions=negative_directions,
        verbose=verbose, logger=logger)
    return realise_blocks(prepared, n_samples,
                          sampling_method=sampling_method, seed=seed,
                          dtype=dtype)


def prepare_blocks(
    blocks,
    *,
    space: str = "linear",
    returns: str = "deltas",
    decomposition_method: str = "svd",
    psd_method: str = "auto",
    null_tol: Optional[float] = DEFAULT_NULL_TOL,
    negative_directions: str = "zero",
    verbose: bool = True,
    logger=None,
) -> list:
    """Decompose every block, draw nothing.

    The first half of :func:`draw_prepared_samples`, separated so that blocks
    asking for different spaces or conventions -- the relative MF33/MF34
    blocks and the absolute MF35 bands of one run -- can be decomposed under
    their own options and still be **realised together** by one
    :func:`realise_blocks` call. That is what keeps them independent under
    Sobol; see :func:`realise_blocks`.

    Parameters
    ----------
    blocks, space, returns, decomposition_method, psd_method, null_tol, negative_directions, verbose, logger
        As in :func:`draw_samples`. One call prepares all its blocks the same
        way; blocks that need another *space* or *returns* go through another
        call. With ``space="log"`` the matrices are taken as already
        ``log1p(Σ)``.

    Returns
    -------
    list
        One prepared block per input block, in the order of *blocks*, to be
        handed unchanged to :func:`realise_blocks`. Blocks prepared by several
        calls can be concatenated into one list.

    Raises
    ------
    ValueError
        If a block is not a square two-dimensional matrix, or under the
        conditions of :func:`draw_samples` on *returns*, *space*,
        *decomposition_method*, *negative_directions* and *psd_method*.
    """
    method = _validated_method(space, returns, decomposition_method)
    _validated_negative_directions(negative_directions)
    prepared = []
    for key, matrix in _as_blocks(blocks):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"block {key!r}: a covariance is a square 2-D matrix, got shape "
                f"{matrix.shape}"
            )
        prepared.append(_PreparedBlock(key, *_decompose_one_block(
            matrix,
            space=space, returns=returns, method=method,
            psd_method=psd_method, null_tol=null_tol,
            negative_directions=negative_directions,
            verbose=verbose, logger=logger, label=str(key),
        )))
    return prepared


def realise_blocks(
    prepared,
    n_samples: int,
    *,
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    dtype=np.float64,
) -> Tuple[Dict[Hashable, np.ndarray], Dict[Hashable, Dict[str, Any]]]:
    """Draw the uncorrelated normals for every prepared block and correlate them.

    **Under Sobol the blocks share one sequence.** Giving each block its own
    ``qmc.Sobol(d=rank_i, seed=seed + i*stride)`` and pairing the blocks by
    replica index would correlate them. Scrambling randomises the values but
    keeps the lattice of each dimension, so dimension *k* of two separately
    seeded Sobol sequences is the same function of the replica index up to a
    digital flip. Every block would look right on its own; the error is in the
    covariance *between* blocks, where a response that depends on two
    of them picks up a term the evaluation never stated, of either sign.

    So one ``qmc.Sobol(d=sum of ranks, seed=seed)`` is drawn and its columns
    are dealt to the blocks **in turn**: the leading direction of every block
    first, in block order, then every block's second, and so on, a block
    dropping out when its rank is used up. The leading Sobol dimensions are
    the best equidistributed, so every block's leading directions get good
    ones. A block's own directions keep their order: the columns of a
    single-block call are ``0..rank-1`` of ``Sobol(d=rank, seed=seed)``.

    Dealing by rank position and not by eigenvalue is deliberate: a block's
    columns then depend only on the ranks and the order of the blocks, and not
    on the spectra of the other blocks.

    ``"random"`` and ``"lhs"`` keep the per-block seed ladder
    (``seed + i * BLOCK_SEED_STRIDE``): independently seeded pseudo-random
    streams and independently permuted hypercubes are independent.

    Parameters
    ----------
    prepared
        The list that :func:`prepare_blocks` returns (several lists may be
        concatenated). Its blocks may differ in *space* and *returns*; each is
        realised as it was prepared.
    n_samples
        Number of realisations per block.
    sampling_method
        ``"sobol"``, ``"lhs"`` or ``"random"``, in any case.
    seed
        Base seed; None draws a different ensemble at each call. Under Sobol
        the sequence is seeded with it; under ``"random"`` and ``"lhs"`` block
        *i* (its position in *prepared*) is seeded with
        ``seed + i × BLOCK_SEED_STRIDE``.
    dtype
        Output dtype of every sample array.

    Returns
    -------
    samples : dict
        ``{key: array}`` of shape ``(n_samples, m)`` and dtype *dtype*, as in
        :func:`draw_samples`.
    diagnostics : dict
        ``{key: dict}``: what :func:`prepare_blocks` recorded for the block, plus
        ``realised_covariance_error``, ``seed`` (the base seed under Sobol, the
        block's seed otherwise), ``block_index`` (position in *prepared*) and,
        under Sobol, ``sobol_columns`` (the columns of the shared sequence, one
        per retained direction).

    Raises
    ------
    ValueError
        If *sampling_method* is not one of the three above.
    """
    prepared = list(prepared)
    samples: Dict[Hashable, np.ndarray] = {}
    diagnostics: Dict[Hashable, Dict[str, Any]] = {}
    normals = _block_normals(prepared, n_samples, sampling_method, seed)

    for index, (block, (z, block_seed, columns)) in enumerate(zip(prepared, normals)):
        y = z @ block.factor.T
        if block.returns == "deltas":
            drawn = y
        elif block.space == "linear":
            drawn = y + 1.0
        else:
            drawn = np.exp(y - 0.5 * block.diagonal)

        info = dict(block.info)
        info["realised_covariance_error"] = _realised_error(
            y, block.matrix, block.counting_tol)
        info["seed"] = block_seed
        info["block_index"] = index
        if columns is not None:
            info["sobol_columns"] = columns
        samples[block.key] = drawn.astype(dtype, copy=False)
        diagnostics[block.key] = info

    return samples, diagnostics


class _PreparedBlock:
    """One decomposed block: its factor and what finishing the draw needs."""

    __slots__ = ("key", "factor", "matrix", "diagonal", "space", "returns",
                 "counting_tol", "info")

    def __init__(self, key, factor, matrix, space, returns, counting_tol, info):
        self.key = key
        self.factor = factor
        self.matrix = matrix
        self.diagonal = np.diag(matrix)
        self.space = space
        self.returns = returns
        self.counting_tol = counting_tol
        self.info = info

    @property
    def rank(self) -> int:
        return int(self.factor.shape[1])


def _block_normals(prepared, n_samples, sampling_method, seed):
    """``[(z, seed_recorded, sobol_columns_or_None), ...]``, one per block."""
    if sampling_method.lower() != "sobol":
        out = []
        for index, block in enumerate(prepared):
            block_seed = None if seed is None else int(seed) + index * BLOCK_SEED_STRIDE
            z = _uncorrelated(dim=block.rank, n=n_samples,
                              method=sampling_method, seed=block_seed)
            out.append((z, block_seed, None))
        return out

    # Dealt in turn: direction j of block i goes before direction j of block
    # i+1 and before direction j+1 of any block.
    order = sorted((j, index) for index, block in enumerate(prepared)
                   for j in range(block.rank))
    columns = [np.empty(block.rank, dtype=np.int64) for block in prepared]
    for dimension, (j, index) in enumerate(order):
        columns[index][j] = dimension
    total = sum(block.rank for block in prepared)
    z = _uncorrelated(dim=total, n=n_samples, method="sobol", seed=seed)
    return [(z[:, cols], seed, cols) for cols in columns]


def _validated_negative_directions(negative_directions: str) -> None:
    if negative_directions not in ("zero", "fold"):
        raise ValueError(
            f"negative_directions must be 'zero' or 'fold', got "
            f"{negative_directions!r}"
        )


def _validated_method(space: str, returns: str, decomposition_method: str) -> str:
    """The checks both entry points make; returns the lower-cased method."""
    if returns not in ("deltas", "factors"):
        raise ValueError(f"returns must be 'deltas' or 'factors', got {returns!r}")
    if space not in ("linear", "log"):
        raise ValueError(f"space must be 'linear' or 'log', got {space!r}")
    if space == "log" and returns == "deltas":
        raise ValueError(
            "space='log' with returns='deltas' is not a thing: a log-space draw "
            "describes multiplicative factors, so ask for returns='factors'"
        )

    method = decomposition_method.lower()
    if method == "cholesky":
        raise ValueError(
            "Cholesky is refused here. These covariances are rank deficient by "
            "construction — an MF35 band typically has two thirds of its "
            "directions null — so a Cholesky factor is either a failure or a "
            "fiction. Use 'svd' (default) or 'eigen'."
        )
    if method not in ("svd", "eigen"):
        raise ValueError(
            f"decomposition_method must be 'svd' or 'eigen', got "
            f"{decomposition_method!r}"
        )

    return method


def _decompose_one_block(
    matrix: np.ndarray,
    *,
    space: str,
    returns: str,
    method: str,
    psd_method: str,
    null_tol: Optional[float],
    negative_directions: str,
    verbose: bool,
    logger,
    label: str,
) -> tuple:
    """Everything of one block's draw that does not touch the normals."""
    n = matrix.shape[0]

    finite = np.isfinite(matrix)
    n_non_finite = int((~finite).sum())
    if n_non_finite:
        matrix = np.where(finite, matrix, 0.0)

    if method == "svd":
        basis, spectrum, vt = svd_decomposition(
            space=space, psd_method=psd_method, verbose=verbose,
            logger=logger, matrix=matrix,
        )
        spectrum = np.asarray(spectrum, dtype=float)
        if negative_directions == "zero":
            # For a symmetric matrix the SVD returns |λ|; the sign of λ_i is
            # the sign of u_i · v_i (v_i = ±u_i). A negative one gets zero
            # variance, as the eigen branch below gives it.
            negative = np.einsum("ki,ik->i", basis, vt) < 0.0
            spectrum = np.where(negative, 0.0, spectrum)
    else:
        values, vectors = eigen_decomposition(
            space=space, psd_method=psd_method, verbose=verbose,
            logger=logger, matrix=matrix,
        )
        order = np.argsort(np.asarray(values, dtype=float))[::-1]
        spectrum = np.asarray(values, dtype=float)[order]
        basis = vectors[:, order]

    largest = float(spectrum.max()) if spectrum.size else 0.0
    counting_tol = DEFAULT_NULL_TOL if null_tol is None else null_tol
    null = (spectrum <= counting_tol * largest if largest > 0
            else np.ones(n, bool))
    n_null = int(null.sum())

    # ``null_tol=None`` draws in every direction, null ones included. The count
    # above is still taken at the default tolerance, so the diagnostics report
    # what is true of the matrix rather than what this draw chose to use.
    keep = np.ones(n, bool) if null_tol is None else ~null

    # **Truncate to the retained rank rather than drawing in all n directions.**
    #
    # Not an optimisation — a correctness fix, and the subtlest thing in this
    # module. A null direction has λ ≈ 1e-10·λmax, so its factor column carries
    # an amplitude ≈ 1e-5 of the leading one: numerical debris from the
    # decomposition, not uncertainty the evaluation claims. Keeping those
    # columns adds an absolute delta of ~1e-10 to *every* component, including
    # ones whose stated σ is exactly zero.
    #
    # In absolute terms that is nothing. But PFNS divides each delta by its
    # group probability, and a fission spectrum's lowest groups hold ~1e-17 of
    # the total — so 1e-10 of debris becomes a perturbation ratio of 1e+7, and
    # the written spectrum grows a spike in its low-energy tail that MF35 never
    # asked for. Measured on Cf-252 before this truncation: group 1 has
    # σ/P⁰ = 0.12 and was being perturbed by a factor of 2.6e+7.
    #
    # Truncating makes the sampled subspace exactly the one the covariance
    # spans, so a component with zero stated variance gets zero delta
    # identically — every retained eigenvector has a zero there. It also drops
    # the QMC dimension from n to the rank, which is where Sobol is better
    # behaved anyway.
    retained = np.clip(spectrum[keep], 0.0, None)
    factor = basis[:, keep] @ np.diag(np.sqrt(retained))

    raw_eigenvalues = np.linalg.eigvalsh(matrix)
    lam_max = float(raw_eigenvalues.max()) if raw_eigenvalues.size else 0.0

    info = {
        "n": n,
        "n_null": n_null,
        "rank": n - n_null,
        "null_fraction": (n_null / n) if n else 0.0,
        "n_non_finite_zeroed": n_non_finite,
        "psd_method": psd_method,
        "decomposition_method": method,
        "space": space,
        "returns": returns,
        "min_over_max_eigenvalue": (
            float(raw_eigenvalues.min() / lam_max) if lam_max > 0 else 0.0
        ),
        "label": label,
    }
    return factor, matrix, space, returns, counting_tol, info


def _realised_error(y: np.ndarray, matrix: np.ndarray, null_tol: float) -> float:
    """Relative error of the realised covariance, on the non-null subspace only.

    Comparing ``cov(Y)`` to ``C`` elementwise is meaningless when two thirds of
    ``C`` is null: the null directions contribute nothing to either, but they
    do contribute to any norm taken over the whole matrix, so the ratio comes
    out looking good for the wrong reason. Restricting to the retained
    eigenvectors measures the thing that was actually sampled.
    """
    if y.shape[0] < 2:
        return float("nan")

    values, vectors = np.linalg.eigh(matrix)
    largest = float(values.max()) if values.size else 0.0
    if largest <= 0:
        return float("nan")
    keep = values > null_tol * largest
    if not np.any(keep):
        return float("nan")

    basis = vectors[:, keep]
    projected = y @ basis
    realised = np.cov(projected, rowvar=False)
    target = np.diag(values[keep])
    return float(np.linalg.norm(realised - target) / np.linalg.norm(target))
