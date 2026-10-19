"""The tolerances of the processing chain: NJOY's thermal rule, and how a heated
table's error is shared between the stages that make it.

**Thermal rule.** NJOY tightens RECONR's and BROADR's tolerances fivefold below
0.4999 eV (``trange``; ``reconr.f90`` l. 2150 and 2390-2392, ``broadr.f90``
l. 1408-1411), testing the left node of each interval. kika does the same in
every adaptive engine a cross section is tabulated or thinned with: the
resonance reconstruction (:func:`kika.algebra.refine_steps`' ``scale``), the
Doppler table (:mod:`kika.processing.doppler.table`) and the thinning
(:func:`kika.algebra.thin_common_grid`'s per-point ``rtol``). Without it the
thermal cross sections came out up to 1.3e-3 high (H-1 MT102 at 293.6 K, a
convex 1/v read lin-lin), against 2.7e-4 for NJOY (ACE roadmap A5, runbook
PXB-065).

**Heated budget.** A heated table is the 0 K source broadened onto an adaptive
table and then thinned, and the three errors add up. Of the declared
tolerance (``thinning``, BROADR's ``errthn``, 1e-3) half is the source's (the
reconstruction tabulates at half its rtol; MF3 under another law is linearised
to 1e-4), a twentieth the broadened table's and the rest, nine twentieths, the
thinning's, so the table delivered stays within the tolerance it states. The
thinning's share decides the points written; the Doppler table's only costs
time. Measured against a reference at 2e-5 (runbook PXB-066): U-235 79 069
points and 7.2e-4 at most (thermal 1.4e-4), against 126 313 with a quarter
each and 59 976 (1.4e-3, over the tolerance) with the whole of it each; a
tighter source (reconstruction at 2e-4) bought 62 438 points at 691 s against
74 s. NJOY spends its whole ``errthn`` on the broadening and again on the
thinning, on top of RECONR's ``err``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["THERMAL_RANGE", "THERMAL_FACTOR", "SOURCE_SHARE", "DOPPLER_SHARE", "thermalFactor",
           "heatedShares"]

#: eV: below it every tolerance is divided by :data:`THERMAL_FACTOR` (NJOY's ``trange``).
THERMAL_RANGE = 0.4999
THERMAL_FACTOR = 5.0
#: The shares of a heated table's tolerance: its 0 K source's, the Doppler table's;
#: the thinning takes the rest.
SOURCE_SHARE = 0.5
DOPPLER_SHARE = 0.05


def thermalFactor(energies):
    """The factor each interval's error is multiplied by, from its left node's energy (eV)."""
    return np.where(np.asarray(energies, dtype=float) < THERMAL_RANGE, THERMAL_FACTOR, 1.0)


def heatedShares(tolerance: float):
    """``(broadening, thinning)`` relative tolerances of a heated table stated to ``tolerance``."""
    tolerance = float(tolerance)
    return DOPPLER_SHARE * tolerance, (1.0 - SOURCE_SHARE - DOPPLER_SHARE) * tolerance
