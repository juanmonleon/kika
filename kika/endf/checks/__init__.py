"""Checks of ENDF sections as they are written in the file (layer 1).

Two families share one vocabulary (findings, levels, reports):

* :func:`check_covariances` reports what is wrong with the covariance files of
  a tape (MF31, MF32, MF33, MF34, MF35 and MF40) -- structure, values,
  positive semi-definiteness;
* :func:`~kika.endf.checks.data.check_data` does the same for everything else
  (MF1-MF28): format, mathematics and physics.

:func:`check_evaluation` runs both on one tape, :func:`check_library` on every
tape of a directory (:func:`check_covariance_library` the covariance checks
alone), and :func:`kika.check` is the door in front of all of them. Nothing is
changed.
"""
from .covariances import check_covariances
from .descriptions import CHECKS, CheckDescription
# Imported here so a frozen build (PyInstaller) bundles it: the reports reach
# it from inside their methods.
from . import export  # noqa: F401
from . import suites  # noqa: F401
from .evaluation import check_evaluation
from .library import (
    TAPE_PATTERNS,
    CovarianceLibraryReport,
    LibraryReport,
    TapeCheck,
    check_covariance_library,
    check_library,
)
from .findings import (
    DEFECT,
    LEVELS,
    NOTE,
    WARN,
    CheckReport,
    CovarianceCheckReport,
    CovarianceFinding,
    CovarianceLocation,
    Finding,
    Location,
)

__all__ = [
    "CHECKS",
    "CheckDescription",
    "check_covariances",
    "check_covariance_library",
    "check_evaluation",
    "check_library",
    "CheckReport",
    "LibraryReport",
    "CovarianceLibraryReport",
    "TapeCheck",
    "TAPE_PATTERNS",
    "CovarianceCheckReport",
    "CovarianceFinding",
    "CovarianceLocation",
    "Finding",
    "Location",
    "NOTE",
    "WARN",
    "DEFECT",
    "LEVELS",
]
