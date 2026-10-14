"""ACE ↔ the GNDS-shaped canonical model.

Same rule as ``kika/endf/model_adapter``: **outside this package and its tests,
only the modules in** ``ALLOWED_IMPORTERS`` **of**
``kika/endf/model_adapter/tests/test_nothing_imports_the_adapter.py`` **may
import it, from inside a function.** It lives under ``kika/ace/`` because it imports both the
ACE classes and the model, and the dependency arrow runs format → calculation,
never the reverse. A module-level import from ``kika/ace/__init__.py`` would put
the model on the ``read_ace`` critical path and change what ``import kika``
costs for the cluster pipeline and the desktop app.
"""
from .decode import (EVALUATED_LABEL, MONTE_CARLO_LABEL, aceLabels, aceStyles, decodeAce,
                     decodeAceReaction, qValuesByMT)
from .encode import AceEncodeError, encodeAce
from .photons import attachAcePhotons, photonProductionTotal

__all__ = [
    "decodeAce", "decodeAceReaction", "aceStyles", "aceLabels", "qValuesByMT",
    "encodeAce", "AceEncodeError",
    "attachAcePhotons", "photonProductionTotal", "EVALUATED_LABEL", "MONTE_CARLO_LABEL",
]
