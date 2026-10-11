"""Scalar free-gas Doppler evaluation, adaptive tables and model-suite heating.

Canonical eV/b/K arrays; analytic Gaussian moments, separate empirical checks,
and atomic Heated attachment. Distributions/covariances and export are not
processed; source reconstruction and support declarations remain explicit.
"""
from .core import (
    DopplerContext, DopplerOptions, SupportPolicy, PreparedDopplerCurve,
    DopplerEvaluation, DopplerInputError, UnsupportedDopplerError,
    DopplerConvergenceError, prepare_curve, evaluate_broadened, ENGINE_VERSION,
)
from .table import DopplerResult, broaden_curve
from .suite import (DopplerInputAccuracy, SuiteDopplerResult, ReactionKey,
                    broaden_suite, attach_broadening)

__all__ = ["DopplerContext", "DopplerOptions", "SupportPolicy",
           "PreparedDopplerCurve", "DopplerEvaluation", "DopplerInputError",
           "UnsupportedDopplerError", "DopplerConvergenceError", "prepare_curve",
           "evaluate_broadened", "ENGINE_VERSION", "DopplerResult", "broaden_curve",
           "DopplerInputAccuracy", "SuiteDopplerResult", "ReactionKey",
           "broaden_suite", "attach_broadening"]
