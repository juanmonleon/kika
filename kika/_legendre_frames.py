"""Legendre coefficients across the CM/LAB change of frame, for MF34's LCT.

ENDF-6 §34.1 lets MF34 state the covariance of the Legendre coefficients in a
frame other than the one MF4 tabulates them in: ``LCT=1`` is LAB, ``LCT=2`` is
CM, and ``LCT=0`` is "the same as MF4". The census of 8-oct-2026 found the
mismatch in U-235 and U-238 (ENDF/B-VIII.1), and in Th-232, Mn-55 and
W-180/182/183/184/186 in both ENDF/B-VIII.1 and JEFF-4.0. It is always the same
way round: the covariance is of LAB coefficients and MF4 is in CM, for MT2 and MT51.

A covariance of LAB coefficients describes LAB factors, so a sample has to be
applied **in the LAB frame and brought back**. Applying the LAB factors to the
CM coefficients would perturb a different quantity from the one the covariance
describes. This module does that round trip, node by node (option A of
``docs/library/cov_checks_roadmap.md``, N2):

1. ``b = T a``. These are the LAB coefficients of the evaluated CM vector ``a``.
   ``T`` is linear but couples every order.
2. ``Δb_l = (f_l - 1) b_l``. This applies the drawn factors where the covariance
   states them.
3. ``Δa = T⁻¹ Δb``, solved on orders ``0..N`` and added to ``a``. The output
   keeps MF4's frame.

**Only the perturbation is converted back, not the whole distribution.** With
all factors equal to 1, ``Δb = 0`` and ``Δa = 0``, so the evaluation comes back
bit for bit. A full conversion followed by its inverse would lose the identity to
truncation. Solving the square system on orders ``0..N`` puts exactly
``(1 + Δ) b_l`` in each of those LAB orders. The CM image of a finite LAB series
is an infinite series, so ``N`` is grown past the highest perturbed order until
what spills into higher LAB orders is negligible. :data:`PAD_TOLERANCE` sets the
cut-off and :func:`scaleInOtherFrame` documents it.

**The kinematics is two-body neutron scattering**, ``A(n,n')A*``: elastic
(MT2) and the discrete inelastic levels (MT51-90). The ratio of the
centre-of-mass speed to the neutron's speed in the CM frame is

    γ(E) = (1/A) · sqrt( E / (E + Q (A+1)/A) )

which is ``1/A`` for elastic and grows without bound at an inelastic threshold.
The conversion needs ``γ < 1``, where the CM/LAB cosine map is one-to-one. At or
just above a threshold, where ``γ ≥ 1``, every CM direction is thrown forward in
LAB, and no CM distribution has the LAB coefficients the factors ask for. Those
nodes are left unperturbed and reported. On a heavy target the window is a few
eV wide; MF4 usually puts one node there, at the threshold itself.

The two cosine maps live here, and :mod:`kika.endf.dcs` imports them from
here. They are plain numerics, and this module has to be importable from both
the model and the sampler, which may import neither each other nor ``kika.endf``
(``kika/tests/test_layering.py``). Their ``alpha`` is this ``γ``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional, Tuple

import numpy as np

__all__ = ["LAB", "CM", "normaliseFrame", "FrameConversion",
           "cos_lab_from_cos_cm", "cos_cm_from_cos_lab",
           "twoBodyGamma", "transferMatrix", "scaleInOtherFrame",
           "TWO_BODY_NEUTRON_MTS"]

LAB = "LAB"
CM = "CM"

#: The reactions whose kinematics :func:`twoBodyGamma` describes: elastic, and
#: inelastic scattering to a discrete level. MT91 (the continuum) is not
#: two-body, and neither is anything that emits a different particle.
TWO_BODY_NEUTRON_MTS = frozenset([2, *range(51, 91)])

#: The relative size, against the largest correction, below which the
#: correction's highest solved orders count as converged. The pad grows until
#: the last two orders of ``Δa`` are this small. Anything left in higher LAB
#: orders is then about this size times the largest ``|Δa|``. That is three
#: orders of magnitude below the 7 significant digits ENDF writes.
PAD_TOLERANCE = 1.0e-10

#: How far the pad may grow. A=12 elastic converges at 8 extra orders and A=1.5
#: at γ = 0.67 at about 30, so reaching the cap means γ is close to 1.
MAX_PAD = 40

#: Quadrature points, as in NJOY's 64-point CM→LAB. Raised with the orders.
QUADRATURE_POINTS = 64


def cos_lab_from_cos_cm(mu_cm, alpha: float):
    r""":math:`\mu_L = (\mu_C + \alpha)/\sqrt{1 + 2\alpha\mu_C + \alpha^2}`."""
    mu_cm = np.asarray(mu_cm, dtype=float)
    return (mu_cm + alpha) / np.sqrt(1.0 + 2.0 * alpha * mu_cm + alpha * alpha)


def cos_cm_from_cos_lab(mu_lab, alpha: float):
    r"""Inverse of :func:`cos_lab_from_cos_cm` (forward branch).

    .. math::
        \mu_C = -\alpha(1 - \mu_L^2) + \mu_L\sqrt{1 - \alpha^2(1 - \mu_L^2)}

    Single-valued only for :math:`\alpha < 1` (target heavier than the
    neutron), which holds for every nuclide but hydrogen.
    """
    mu_lab = np.asarray(mu_lab, dtype=float)
    s = 1.0 - mu_lab * mu_lab
    return -alpha * s + mu_lab * np.sqrt(np.maximum(1.0 - alpha * alpha * s, 0.0))


def normaliseFrame(value) -> Optional[str]:
    """``"LAB"``, ``"CM"`` or ``None``.

    ``None`` means "the same as the distribution", which is LCT=0 or no frame
    given. Accepts the strings kika uses (MF34's ``"same-as-MF4"``/``"LAB"``/
    ``"CM"``, the model's ``Frame.lab``/``Frame.centerOfMass``) and the LCT
    integers.
    """
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    key = text.lower()
    if key in ("lab", "1"):
        return LAB
    if key in ("cm", "centerofmass", "2"):
        return CM
    if key in ("same-as-mf4", "0", ""):
        return None
    raise ValueError(f"{value!r} is not a reference frame kika knows "
                     f"(LAB, CM, or same-as-MF4)")


def twoBodyGamma(energy: float, awr: float, q: float = 0.0) -> float:
    """``γ = V_cm / v'_cm`` for ``A(n,n')A*`` at incident *energy* (eV).

    *awr* is the target's mass in neutron masses (ENDF's AWR). *q* is the
    reaction's QI in eV: 0 for elastic, minus the level energy for MT51-90.
    Returns ``inf`` at or below the threshold.
    """
    awr = float(awr)
    if not awr > 0:
        raise ValueError(f"AWR={awr!r}: the frame change needs the target mass")
    available = float(energy) + float(q) * (awr + 1.0) / awr
    if available <= 0.0:
        return float("inf")
    return float(np.sqrt(float(energy) / available) / awr)


def transferMatrix(gamma: float, nOut: int, nIn: int, direction: str,
                   nQuad: Optional[int] = None) -> np.ndarray:
    """``M`` with ``b = M a``: the coefficients of *a*'s distribution in the
    other frame.

    *direction* is ``"CM->LAB"`` or ``"LAB->CM"``. With
    ``f(μ) = Σ (2k+1)/2 a_k P_k(μ)``, the integral

        b_l = ∫ f_from(μ) P_l(μ_to(μ)) dμ

    runs over the source cosine. The source is where *a* is a polynomial, and the
    change of variable absorbs the Jacobian. So
    ``M[l, k] = (2k+1)/2 ∫ P_k(μ) P_l(μ_to(μ)) dμ``, by Gauss-Legendre. Row 0 is
    ``e_0``: a frame change preserves normalisation.
    """
    from numpy.polynomial.legendre import leggauss, legvander

    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"γ={gamma!r}: the CM/LAB cosine map is one-to-one "
                         f"only for γ < 1")
    if direction == "CM->LAB":
        mapping = cos_lab_from_cos_cm
    elif direction == "LAB->CM":
        mapping = cos_cm_from_cos_lab
    else:
        raise ValueError(f"direction must be 'CM->LAB' or 'LAB->CM', got {direction!r}")
    n = nQuad or max(QUADRATURE_POINTS, nIn + nOut + 16)
    mu, weights = leggauss(n)
    pSource = legvander(mu, nIn - 1)                     # (n, nIn)
    pTarget = legvander(mapping(mu, gamma), nOut - 1)    # (n, nOut)
    return (pTarget * weights[:, None]).T @ pSource * (2 * np.arange(nIn) + 1) / 2.0


@lru_cache(maxsize=8192)
def _system(gamma: float, n: int, width: int, direction: str):
    """``transferMatrix`` with the inverse and condition of its square block.

    Cached because the conversion runs per node and per sample. For elastic
    scattering ``γ = 1/A`` at every node, so the cache holds one system. For an
    inelastic level each node has its own ``γ``, and every sample after the
    first reuses it. Without the cache the Fe-56 micro-tape took 17.6 s per
    sample; the arrays come back read-only.
    """
    matrix = transferMatrix(gamma, n, width, direction)
    square = matrix[:, :n]
    inverse = np.linalg.inv(square)
    condition = float(np.linalg.cond(square))
    matrix.setflags(write=False)
    inverse.setflags(write=False)
    return matrix, inverse, condition


@dataclass(frozen=True)
class FrameConversion:
    """The covariance's frame, the distribution's, and the kinematics between.

    Built once per reaction. :meth:`gamma` is evaluated per incident energy.
    """

    distributionFrame: str
    covarianceFrame: str
    awr: float
    q: float = 0.0
    mt: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("distributionFrame", "covarianceFrame"):
            if getattr(self, name) not in (LAB, CM):
                raise ValueError(f"{name}={getattr(self, name)!r}: LAB or CM")
        if self.distributionFrame == self.covarianceFrame:
            raise ValueError("a FrameConversion between a frame and itself is "
                             "the identity; pass None instead")
        if self.mt is not None and int(self.mt) not in TWO_BODY_NEUTRON_MTS:
            raise ValueError(
                f"MT{self.mt}: the covariance is in {self.covarianceFrame} and "
                f"the distribution in {self.distributionFrame}, and kika converts "
                f"between the two only for two-body neutron scattering (MT2, "
                f"MT51-90). Perturbing in the wrong frame would perturb a "
                f"different quantity from the one the covariance describes")

    @property
    def direction(self) -> str:
        return f"{self.distributionFrame}->{self.covarianceFrame}"

    def gamma(self, energy: float) -> float:
        return twoBodyGamma(energy, self.awr, self.q)

    @classmethod
    def between(cls, distributionFrame, covarianceFrame, *, awr, q=0.0,
                mt=None) -> Optional["FrameConversion"]:
        """The conversion, or ``None`` when the frames already agree."""
        cov = normaliseFrame(covarianceFrame)
        dist = normaliseFrame(distributionFrame)
        if cov is None or dist is None or cov == dist:
            return None
        if awr is None:
            raise ValueError(
                f"MT{mt}: the covariance is in {cov} and the distribution in "
                f"{dist}, and there is no AWR to convert with")
        return cls(dist, cov, float(awr), float(q or 0.0),
                   None if mt is None else int(mt))


def scaleInOtherFrame(coefficients, factors: Mapping[int, float],
                      gamma: float, direction: str
                      ) -> Tuple[Optional[np.ndarray], dict]:
    """Apply *factors* to the coefficients of the other frame, and come back.

    *coefficients* is the full vector in the distribution's frame, ``a_0``
    included. *factors* maps a Legendre order to the factor drawn for it in the
    covariance's frame. Returns ``(new vector, info)``, or ``(None, info)`` when
    ``γ ≥ 1`` and the node has to be left as it is.

    The correction is solved on orders ``0..N``, starting with ``N`` two above
    the highest factored order. ``N`` grows by two until the correction's
    highest two orders fall below :data:`PAD_TOLERANCE` times its largest entry.
    The pad orders take up what the CM image of a finite LAB perturbation spills
    upwards; without them that spill would show up as a perturbation of LAB
    orders the covariance says nothing about. Orders the evaluation did not
    carry are then kept only where the correction actually put something, so a
    vector does not grow by trailing zeros.

    ``info`` holds ``gamma``, ``n_solved`` (``N + 1``) and ``condition``, the
    condition number of the square system.
    """
    a = np.asarray(coefficients, dtype=float)
    info = {"gamma": float(gamma)}
    if not gamma < 1.0:
        return None, info
    orders = sorted(int(order) for order in factors)
    if not orders or min(orders) < 1:
        raise ValueError(f"factored orders {orders}: L>=1 only, a_0 is the "
                         f"normalisation")
    top = orders[-1]

    pad = 2
    while True:
        n = top + pad + 1
        width = max(a.size, n)
        matrix, inverse, condition = _system(float(gamma), n, width, direction)
        b = matrix[:, :a.size] @ a
        delta_b = np.zeros(n)
        for order in orders:
            delta_b[order] = (float(factors[order]) - 1.0) * b[order]
        delta_a = inverse @ delta_b
        scale = float(np.abs(delta_a).max())
        tail = float(np.abs(delta_a[-2:]).max())
        if scale == 0.0 or tail <= PAD_TOLERANCE * scale or pad >= MAX_PAD:
            break
        pad += 2

    info.update(n_solved=int(n), condition=condition,
                pad_capped=bool(pad >= MAX_PAD and scale and
                                tail > PAD_TOLERANCE * scale))
    out = np.zeros(width)
    out[:a.size] = a
    out[:n] += delta_a
    # Trim what the correction did not reach, beyond what the evaluation carried.
    keep = a.size
    threshold = PAD_TOLERANCE * scale
    for order in range(width - 1, a.size - 1, -1):
        if abs(out[order]) > threshold:
            keep = order + 1
            break
    return out[:keep], info
