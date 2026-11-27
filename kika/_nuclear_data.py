"""The evaluated-data tree, found through one variable, ``KIKA_NUCLEAR_DATA``, laid out
type/library: ``endf/endfb81``, ``decay/endfb81``, ``gnds/endfb81``, ``ace/…``,
``openmc/<lib>/cross_sections.xml``, ``g4ndl/…`` and ``_kika-tests/tapes``."""
import os
from pathlib import Path
from typing import Optional


def nuclear_data(*parts: str) -> Optional[Path]:
    root = os.environ.get("KIKA_NUCLEAR_DATA")
    return Path(root, *parts) if root else None
