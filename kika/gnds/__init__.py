"""GNDS support: the capability census and staged processing.

Reading and writing a GNDS file goes through :func:`kika.read` and :func:`kika.write`.
Importing this package loads no model module: the GNDS reader and writer are imported
inside those calls.

:func:`capabilities` answers, without opening a file, what kika does with each node of
``gnds.xsd`` and ``covariances.xsd`` (300 nodes) and with the twelve names kika uses that
those schemas do not declare. It says what the library can lose; ``suite.report`` says
what a given file lost. :func:`process_gnds` runs a GNDS neutron evaluation through the
same stages as :func:`kika.endf.process_endf`.

GNDS versions 2.0 and 2.1 are read through one code path; 1.9, 1.10 and 2.2 are
refused. The writer emits the version the suite came in as, and 2.0 for a suite that
came from ENDF. XML is the only meta-language implemented.
"""
from __future__ import annotations

from ._capabilities import (CAPABILITIES, Capabilities, Capability, Coverage,
                            NOT_IN_SCHEMA, capabilities)

#: The private module name is load-bearing, not a style choice. ``importlib``
#: binds a submodule onto its package as a side effect of importing it, so a
#: module named ``capabilities.py`` would overwrite the function above with
#: itself on first access — the first call returns the function and every later
#: one returns the module. ``kika/__init__.py:40-43`` documents the same trap.
__all__ = ["capabilities", "Capabilities", "Capability", "Coverage",
           "CAPABILITIES", "NOT_IN_SCHEMA", "process_gnds"]


def process_gnds(path, plan=None, progress=None):
    """Process a GNDS neutron evaluation stage by stage.

    The stages, the progress calls, the failure contract and the outputs are those of
    :func:`kika.endf.process_endf`. The input is a ``reactionSuite``. The 0 K
    cross-sections are always kika's, reconstructed from the evaluated resonance
    parameters; a ``crossSectionReconstructed`` style or any other processed style of
    the file is set aside, and the ``decode`` stage says so. A file with no resonance
    parameters takes its evaluated pointwise cross-sections as they are.

    Parameters
    ----------
    path
        Path of the GNDS file.
    plan
        What to compute (temperatures, broadening, heating, workers). None is a
        default :class:`~kika.endf.ProcessingPlan`.
    progress
        Called as ``progress(stage, fraction)``, with ``fraction`` None when a stage
        starts and a float in (0, 1] afterwards. None reports nothing.

    Returns
    -------
    kika.endf.EndfProcessing
        The stage reports and the results; the outputs are its ``write_*`` methods.

    Raises
    ------
    kika.processing.resonances.UnsupportedResonanceError
        If the file states no target mass or spin in PoPs and the reconstruction needs
        them (category ``context-missing``), or the resonance parameters are not
        supported.
    kika.processing.resonances.ReconstructionConvergenceError
        If the resonance reconstruction does not converge.
    RuntimeError
        If called while a worker process is importing the main script.
    """
    from kika.endf.process import process_gnds as run
    return run(path, plan, progress)
