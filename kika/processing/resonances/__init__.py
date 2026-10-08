"""Experimental, model-based point evaluation of resonance cross sections.

The tabulator accepts explicit model backgrounds and sums. The suite facade
assembles supported resolved and dilute URR models across their evaluated domain and attaches a
separate reconstructed style after verification. Format round trips require
``verify_suite``; unsupported materials and precision failures are rejected.
These APIs do not replace the legacy ``reconstruct`` entry point.
Model imports are deferred until preparation.
"""
from .context import NeutronContext
from .prepare import PreparedResonances, UnsupportedResonanceError, prepare_resonances
from .grid import ReconstructionOptions, ReconstructionConvergenceError
from .tabulate import ReconstructionResult, tabulate_resonances
from .suite import ReactionKey, SuiteReconstructionResult, reconstruct_suite, attach_reconstruction

__all__ = ["NeutronContext", "PreparedResonances", "UnsupportedResonanceError",
           "prepare_resonances", "ReconstructionOptions", "ReconstructionConvergenceError",
           "ReconstructionResult", "tabulate_resonances", "ReactionKey", "SuiteReconstructionResult",
           "reconstruct_suite", "attach_reconstruction"]
