"""Data models for parsed Serpent history output files (_his.m).

Serpent writes per-cycle statistics in MATLAB format::

    HIS_TIME = [
      1  1.50000E+01
      2  2.75000E+01
      ...
      % ----- Begin active cycles -----
      1  4.20000E+01
      2  5.10000E+01
      ...
    ];

    HIS_IMP_KEFF = [
      cycle  value  mean  rel_error
      ...
    ];

For multi-scored variables the pattern repeats as groups of 3
(value, mean, error) after the cycle number column.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HistoryVariable:
    """Per-cycle history data for a single scored quantity.

    A user gets it from ``HistoryFile.variables``. Each of the arrays has one row per
    data line of the ``HIS_<name>`` block.

    Attributes
    ----------
    name : str
        Variable name without the ``HIS_`` prefix.
    cycles : numpy.ndarray
        Shape ``(N,)``: the first column of the block.
    values, means, errors : numpy.ndarray
        Shape ``(N, n_scores)``. The parser slices the columns after the first in
        groups of three per score: the first of each group is ``values``, the second
        ``means`` and the third ``errors``. Serpent's convention is that these are
        the value of the cycle, the running mean and its relative statistical error.
    n_scores : int
        Number of groups of three columns after the first, ``(columns - 1) // 3``.
    """
    name: str
    cycles: np.ndarray       # (N,) — cycle numbers (combined inactive + active)
    values: np.ndarray       # (N, n_scores) — per-cycle instantaneous values
    means: np.ndarray        # (N, n_scores) — running cumulative means
    errors: np.ndarray       # (N, n_scores) — running relative errors
    n_scores: int            # number of independent scores

    @property
    def n_cycles(self) -> int:
        """Number of cycles, inactive and active."""
        return len(self.cycles)

    def __repr__(self) -> str:
        return (
            f"HistoryVariable('{self.name}', "
            f"cycles={self.n_cycles}, scores={self.n_scores})"
        )


@dataclass
class HistoryFile:
    """All per-cycle data from a Serpent ``_his.m`` file.

    A user gets one from :func:`read_history_file`.

    Attributes
    ----------
    time : numpy.ndarray
        Shape ``(N, 2)``: the two columns of ``HIS_TIME`` as written (Serpent's
        convention: cycle number and wall-clock time in seconds; kika does not
        convert). Empty when the file has no such block.
    variables : dict of str to HistoryVariable
        By variable name without the ``HIS_`` prefix.
    n_inactive : int
        Rows of ``HIS_TIME`` before the "Begin active cycles" comment line.
    n_total : int
        Rows of ``HIS_TIME``: inactive plus active cycles.
    """
    time: np.ndarray                         # (N, 2) → [cycle, wall_time_seconds]
    variables: Dict[str, HistoryVariable] = field(default_factory=dict)
    n_inactive: int = 0                       # number of inactive cycles
    n_total: int = 0                          # total cycles (inactive + active)

    @property
    def n_active(self) -> int:
        """Number of active cycles, ``n_total - n_inactive``."""
        return self.n_total - self.n_inactive

    @property
    def variable_names(self) -> List[str]:
        """Sorted list of variable names (without the ``HIS_`` prefix)."""
        return sorted(self.variables.keys())

    @property
    def imp_keff(self) -> Optional[HistoryVariable]:
        """The ``IMP_KEFF`` variable (implicit k-eff), or None if absent."""
        return self.variables.get("IMP_KEFF")

    @property
    def ana_keff(self) -> Optional[HistoryVariable]:
        """The ``ANA_KEFF`` variable (analog k-eff), or None if absent."""
        return self.variables.get("ANA_KEFF")

    @property
    def entropy(self) -> Optional[HistoryVariable]:
        """The ``ENTROPY`` variable, or None if absent."""
        return self.variables.get("ENTROPY")

    def inactive_mask(self, var_name: str) -> Optional[np.ndarray]:
        """Boolean mask for inactive cycles of a variable.

        Parameters
        ----------
        var_name : str
            Variable name without the ``HIS_`` prefix.

        Returns
        -------
        numpy.ndarray or None
            One entry per cycle of the variable, True for the first ``n_inactive``
            cycles; None if the variable is absent.
        """
        var = self.variables.get(var_name)
        if var is None:
            return None
        # Inactive cycles are the first n_inactive entries
        mask = np.zeros(var.n_cycles, dtype=bool)
        mask[:self.n_inactive] = True
        return mask

    def active_mask(self, var_name: str) -> Optional[np.ndarray]:
        """Boolean mask for active cycles of a variable.

        Parameters
        ----------
        var_name : str
            Variable name without the ``HIS_`` prefix.

        Returns
        -------
        numpy.ndarray or None
            One entry per cycle of the variable, False for the first ``n_inactive``
            cycles; None if the variable is absent.
        """
        var = self.variables.get(var_name)
        if var is None:
            return None
        mask = np.ones(var.n_cycles, dtype=bool)
        mask[:self.n_inactive] = False
        return mask

    def __repr__(self) -> str:
        return (
            f"HistoryFile({len(self.variables)} variables, "
            f"{self.n_inactive} inactive + {self.n_active} active cycles)"
        )
