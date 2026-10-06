"""Checks of ENDF sections as they are written in the file (layer 1).

:func:`check_covariances` reports what is wrong with the MF31/MF33/MF34 of a
tape -- structure, values, positive semi-definiteness -- without changing
anything; :func:`check_covariance_library` does it for every tape of a directory. Plan: kika-workspace ``docs/library/cov_checks_roadmap.md``.
"""
from .covariances import check_covariances
# Imported here so a frozen build (PyInstaller) bundles it: the reports reach
# it from inside their methods.
from . import export  # noqa: F401
from .library import (
    TAPE_PATTERNS,
    CovarianceLibraryReport,
    TapeCheck,
    check_covariance_library,
)
from .findings import (
    DEFECT,
    LEVELS,
    NOTE,
    WARN,
    CovarianceCheckReport,
    CovarianceFinding,
    CovarianceLocation,
)

__all__ = [
    "check_covariances",
    "check_covariance_library",
    "CovarianceLibraryReport",
    "TapeCheck",
    "TAPE_PATTERNS",
    "CovarianceCheckReport",
    "CovarianceFinding",
    "CovarianceLocation",
    "NOTE",
    "WARN",
    "DEFECT",
    "LEVELS",
]
