"""G4NDL — the neutron data format Geant4's ParticleHP models read.

G4NDL is one more door into and out of the common model, like ENDF, ACE and
GNDS: a library directory (``Elastic/CrossSection``, ``Elastic/FS``, ...) of
per-isotope token streams, optionally zlib-compressed as ``<name>.z``. Geant4
itself is **not** a dependency of kika — not even an optional one. The contract
the reader follows is the consumer's source, Geant4 v11.4.3
``source/processes/hadronic/models/particle_hp``, written down with line
references in ``kika-workspace/docs/library/G4NDL_token_spec.md``; checking
against a real Geant4 build happens outside this repository.

**Scope today: elastic scattering only** (``Elastic/CrossSection`` and
``Elastic/FS``, i.e. MT2). A library also holding ``Capture``, ``Inelastic`` or
``Fission`` is read *partially*, and the conversion report says so.

Like :mod:`kika.gnds`, nothing here imports :mod:`kika.nuclear_data.model` at
module scope: ``import kika.g4ndl`` must not wake the model
(``kika/nuclear_data/model/tests/test_dormancy.py``).
"""

from kika.g4ndl.exceptions import G4NDLError, G4NDLFormatError, IsotopeNotFoundError
from kika.g4ndl.library import G4NDLLibrary, open
from kika.g4ndl.names import IsotopeKey

__all__ = ["G4NDLError", "G4NDLFormatError", "G4NDLLibrary", "IsotopeKey",
           "IsotopeNotFoundError", "open"]
