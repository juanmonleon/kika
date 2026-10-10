"""Native model-based reconstruction of resonance cross sections.

The tabulator accepts explicit model backgrounds and sums. The suite facade
assembles supported resolved and dilute URR models across their evaluated domain and attaches a
separate reconstructed style after verification. Format round trips require
``verify_suite``; unsupported materials and precision failures are rejected.
The end-to-end entry for an ENDF-6 tape is ``kika.endf.reconstruct_endf``.
Model imports are deferred until preparation.
"""
from .context import NeutronContext
from .prepare import PreparedResonances, UnsupportedResonanceError, prepare_resonances
from .grid import ReconstructionOptions, ReconstructionConvergenceError
from .tabulate import ReconstructionResult, tabulate_resonances
from .suite import ReactionKey, SuiteReconstructionResult, reconstruct_suite, attach_reconstruction, ENGINE_VERSION
from .capabilities import REJECTION_CATEGORIES, native_available

__all__ = ["NeutronContext", "PreparedResonances", "UnsupportedResonanceError",
           "prepare_resonances", "ReconstructionOptions", "ReconstructionConvergenceError",
           "ReconstructionResult", "tabulate_resonances", "ReactionKey", "SuiteReconstructionResult",
           "reconstruct_suite", "attach_reconstruction", "ENGINE_VERSION",
           "native_available", "REJECTION_CATEGORIES"]
