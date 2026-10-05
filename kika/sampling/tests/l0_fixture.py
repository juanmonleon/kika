"""A covariance-only Fe-56 tape whose MF34 states L=0, built on demand.

Of the evaluated libraries to hand only U-235 and U-238 of ENDF/B-VIII.0 and
VIII.1 state L=0 in MF34 (scanned 2026-10-05 with
``kika-workspace/kika_dev/sampling/checks/scan_mf34_candidates.py``), and theirs
is a placeholder: L0xL0 one bin of zero variance, L0xLl noise at 2e-19. That
tape is committed as ``micro_u238_mf34_l0.endf`` and tests the plumbing; it
cannot test a cross term that is really there. This project's ``_a0cross``
tapes have one, in an MF34 of gigabytes. So the shape is fabricated here,
small enough to read in a test and with a covariance whose every number is
known.

The tape is MF3 (a header stub), MF33/MT2 and MF34/MT2 of ZA 26056, MAT 2631,
on one 3-bin grid. It is meant to be passed as ``covarianceSource`` beside
``micro_fe56_xs_and_angular.endf``, which carries the MF3 and MF4 to perturb.

The joint MF34 matrix is ``D (C_L kron C_E) D``: a Kronecker product of two
correlation matrices is a correlation matrix, so the whole thing is PSD by
construction, and every section is a slice of it:

* L=0 -- the magnitude, 3 % per bin;
* L=1 -- 10/8/12 %;  L=2 -- 15 %;
* orders correlated by ``C_L`` (L0-L1 0.4, L0-L2 0.2, L1-L2 0.3), energies by
  ``C_E`` (0.5 between neighbours, 0.25 two apart).

MF33/MT2 states **5/4/6 %** on the same grid with the same ``C_E`` -- different
from L0xL0 on purpose, so a test can tell which of the two set sigma's variance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ZA, AWR, MAT, MT = 26056, 5.536735e1, 2631, 2
GRID = np.array([8.5e5, 1.5e6, 2.5e6, 4.0e6])
C_E = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]])
C_L = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
SIGMA_BY_ORDER = {0: np.array([0.03, 0.03, 0.03]),
                  1: np.array([0.10, 0.08, 0.12]),
                  2: np.array([0.15, 0.15, 0.15])}
SIGMA_MF33 = np.array([0.05, 0.04, 0.06])


def block(l: int, l1: int) -> np.ndarray:
    """The (L, L1) relative covariance block of the MF34 matrix."""
    return C_L[l, l1] * np.outer(SIGMA_BY_ORDER[l], SIGMA_BY_ORDER[l1]) * C_E


def mf33Block() -> np.ndarray:
    return np.outer(SIGMA_MF33, SIGMA_MF33) * C_E


def _f(x: float) -> str:
    if x == 0.0:
        return " 0.000000+0"
    mantissa, exponent = f"{x:.6e}".split("e")
    return f"{mantissa}{int(exponent):+d}".rjust(11)


def _line(fields, mf: int, mt: int) -> str:
    body = "".join(f if isinstance(f, str) else
                   (_f(f) if isinstance(f, float) else f"{f:11d}") for f in fields)
    return f"{body:<66}{MAT:4d}{mf:2d}{mt:3d}     "


def _list(head, values, mf, mt):
    lines = [_line(head, mf, mt)]
    for start in range(0, len(values), 6):
        lines.append(_line([float(v) for v in values[start:start + 6]], mf, mt))
    return lines


def _lb5(matrix: np.ndarray, symmetric: bool, mf: int, mt: int):
    ne = GRID.size
    if symmetric:
        values = list(GRID) + [matrix[i, j] for i in range(ne - 1)
                               for j in range(i, ne - 1)]
    else:
        values = list(GRID) + list(matrix.ravel())
    return _list([0.0, 0.0, 1 if symmetric else 0, 5, len(values), ne],
                 values, mf, mt)


def _send(mf):
    return _line([0.0, 0.0, 0, 0, 0, 0], mf, 0).replace(f"{mf:2d}  0     ",
                                                        f"{mf:2d}  099999")


def write(path: Path, *, orders=(0, 1, 2)) -> Path:
    orders = list(orders)
    lines = [f"{'synthetic Fe-56 MF33+MF34 with L=0 (kika tests)':<66}"
             f"{1:4d}{0:2d}{0:3d}     "]
    # MF3 stub: the covariance file only has to name the reaction.
    lines += [_line([float(ZA), AWR, 0, 0, 0, 0], 3, MT),
              _line([0.0, 0.0, 0, 0, 0, 0], 3, MT).replace(" 3  2     ", " 3  299999"),
              _line([0.0, 0.0, 0, 0, 0, 0], 0, 0)]
    # MF33/MT2: one NI subsection, LB=5 symmetric.
    lines += [_line([float(ZA), AWR, 0, 0, 0, 1], 33, MT),
              _line([0.0, 0.0, 0, MT, 0, 1], 33, MT)]
    lines += _lb5(mf33Block(), True, 33, MT)
    lines += [_send(33), _line([0.0, 0.0, 0, 0, 0, 0], 0, 0)]
    # MF34/MT2: LTT=3 because a_0 is present (ENDF-6 §34.2: NL counts the
    # coefficients and the first is a_0), upper triangle of (L, L1).
    nl = len(orders)
    lines += [_line([float(ZA), AWR, 0, 3 if 0 in orders else 1, 0, 1], 34, MT),
              _line([0.0, 0.0, 0, MT, nl, nl], 34, MT)]
    for a, l in enumerate(orders):
        for l1 in orders[a:]:
            lines.append(_line([0.0, 0.0, l, l1, 0, 1], 34, MT))
            lines += _lb5(block(l, l1), l == l1, 34, MT)
    lines += [_send(34), _line([0.0, 0.0, 0, 0, 0, 0], 0, 0),
              _line([0.0, 0.0, 0, 0, 0, 0], 0, 0).replace(f"{MAT:4d}", "  -1")]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path
