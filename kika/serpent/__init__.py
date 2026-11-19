# ──────────────────────────────────────────────────────────────────────────────
# File: serpent/__init__.py
# Public API for Serpent Monte Carlo code support.
# ──────────────────────────────────────────────────────────────────────────────

"""Readers for the Serpent input deck and its text outputs.

The package parses the cards of an input file (detectors, energy and time grids,
``sens`` cards) and the MATLAB-format outputs of a run: sensitivities (``.sens``),
detectors (``_det.m``), run results (``_res.m``), per-cycle history (``_his.m``) and
depletion (``_dep.m``). Each ``read_*`` function takes a path or the raw text. The
output readers return a container of numpy arrays holding the numbers as Serpent
writes them; the input reader returns a ``SerpentInput`` of dataclasses.
"""

# Sensitivity output (.sens files)
from .sens import (
    PertCategory,
    Material,
    Nuclide,
    Perturbation,
    Response,
    SensitivitySet,
    SensitivityFile,
)
from .parse_sens import read_sensitivity_file, parse_sensitivity_text

# Input file parsing
from .input import (
    DetectorDefinition,
    EnergyGrid,
    SensitivityConfig,
    SensPerturbation,
    SensResponse,
    SerpentInput,
    TimeGrid,
)
from .parse_input import read_serpent_input

# Detector output (_det.m files)
from .detector import DetectorFile, DetectorResult
from .parse_det import read_detector_file, parse_detector_text

# Results output (_res.m files)
from .results import ResultsFile
from .parse_res import read_results_file, parse_results_text

# History output (_his.m files)
from .history import HistoryVariable, HistoryFile
from .parse_his import read_history_file, parse_history_text

# Depletion output (_dep.m files)
from .depletion import DepletionMaterial, DepletionFile
from .parse_dep import read_depletion_file, parse_depletion_text

__all__ = [
    # Sensitivity
    "PertCategory",
    "Material",
    "Nuclide",
    "Perturbation",
    "Response",
    "SensitivitySet",
    "SensitivityFile",
    "read_sensitivity_file",
    "parse_sensitivity_text",
    # Input
    "DetectorDefinition",
    "EnergyGrid",
    "SensitivityConfig",
    "SensPerturbation",
    "SensResponse",
    "SerpentInput",
    "TimeGrid",
    "read_serpent_input",
    # Detector output
    "DetectorFile",
    "DetectorResult",
    "read_detector_file",
    "parse_detector_text",
    # Results output
    "ResultsFile",
    "read_results_file",
    "parse_results_text",
    # History output
    "HistoryVariable",
    "HistoryFile",
    "read_history_file",
    "parse_history_text",
    # Depletion output
    "DepletionMaterial",
    "DepletionFile",
    "read_depletion_file",
    "parse_depletion_text",
]
