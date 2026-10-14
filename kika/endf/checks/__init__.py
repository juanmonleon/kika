"""Checks of ENDF sections as they are written in the file (layer 1).

:func:`check_covariances` reports what is wrong with the covariance files of a
tape (MF31, MF32, MF33, MF34, MF35 and MF40) -- structure, values, positive
semi-definiteness -- without changing anything; :func:`check_covariance_library`
does it for every tape of a directory.
"""
from .covariances import check_covariances
from .descriptions import CHECKS, CheckDescription
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
    "CHECKS",
    "CheckDescription",
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
