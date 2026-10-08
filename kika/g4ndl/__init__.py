"""G4NDL — the neutron data format Geant4's ParticleHP models read.

G4NDL is one more door into and out of the common model, like ENDF, ACE and
GNDS: a library directory (``Elastic/CrossSection``, ``Elastic/FS``, ...) of
per-isotope token streams, optionally zlib-compressed as ``<name>.z``. Geant4
itself is **not** a dependency of kika — not even an optional one. The contract
the reader follows is the consumer's source, Geant4 v11.4.3
``source/processes/hadronic/models/particle_hp``, written down with line
references in ``kika-workspace/docs/library/G4NDL_token_spec.md``; checking
against a real Geant4 build happens outside this repository.

**Scope today: elastic, inelastic, capture and fission**, read and written:
``Elastic/CrossSection`` and ``Elastic/FS`` (MT2); ``Inelastic/CrossSection``,
the 36 channel directories ``Inelastic/F01`` … ``F36`` and the residual level
schemes ``Inelastic/Gammas`` (:mod:`kika.g4ndl.inelastic_decode` says what of
them reaches the model); and ``Capture/CrossSection`` with its final state,
``Capture/FSMF6`` or ``Capture/FS`` (MT102, :mod:`kika.g4ndl.capture`); and
``Fission/CrossSection``, ``Fission/FS``, the chances ``Fission/FC`` … ``LC``
and the fragment yields ``Fission/FF`` (MT18 and MT19-21, 38: the grammar in
:mod:`kika.g4ndl.fission`, the model in :mod:`kika.g4ndl.fission_model`).
``kika.read(root, format="g4ndl", target=...)``,
``kika.write(suite, root, format="g4ndl")``, and :func:`patch_isotope` (or
:func:`patch_elastic`) to replace one isotope in a copy of a whole library. A
library also holding ``ThermalScattering`` is read *partially*,
and the conversion report says so.

Like :mod:`kika.gnds`, nothing here imports :mod:`kika.nuclear_data.model` at
module scope: ``import kika.g4ndl`` must not wake the model
(``kika/nuclear_data/model/tests/test_dormancy.py``).
"""

from kika.g4ndl.exceptions import (
    G4NDLError, G4NDLFormatError, G4NDLUnsupportedError, IsotopeNotFoundError,
)
from kika.g4ndl.library import G4NDLLibrary, open
from kika.g4ndl.names import IsotopeKey
from kika.g4ndl.patch import PatchResult, patch_elastic, patch_isotope
from kika.g4ndl.tables import (
    angularBulk, angularMTs, crossSections, fissionSummary, inelasticTotal, isotopeSummary,
)

__all__ = ["G4NDLError", "G4NDLFormatError", "G4NDLLibrary", "G4NDLUnsupportedError",
           "IsotopeKey", "IsotopeNotFoundError", "PatchResult", "angularBulk", "angularMTs",
           "crossSections", "fissionSummary", "inelasticTotal",
           "isotopeSummary", "open", "patch_elastic", "patch_isotope"]
