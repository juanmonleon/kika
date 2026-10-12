"""
Auto-detection of MCNP file types (input deck, MCTAL or MESHTAL output).
"""

import re
from pathlib import Path
from typing import Optional

_NTAL = re.compile(r'^ntal\s+\d+', re.IGNORECASE)
_MESH_TALLY = re.compile(r'^\s*Mesh Tally Number\s+\d+\s*$')


def detect_mcnp_file_type(file_path: Optional[str] = None, content: Optional[str] = None) -> str:
    """Detect whether a file is an MCNP input deck, a MCTAL or a MESHTAL file.

    Uses content-based heuristics on the first lines: see :func:`_is_mctal`
    and :func:`_is_meshtal`. Anything else is taken as an input deck.

    :param file_path: Path to the file on disk (preferred — avoids loading full content).
    :param content: File content as a string (used if file_path is not given).
    :returns: ``'mcnp-input'``, ``'mcnp-mctal'`` or ``'mcnp-meshtal'``
    :raises ValueError: If neither file_path nor content is provided.
    """
    if file_path is not None:
        lines = _read_first_lines(file_path, n=6)
    elif content is not None:
        lines = _lines_from_content(content, n=6)
    else:
        raise ValueError("Either file_path or content must be provided")

    if _is_mctal(lines[:4]):
        return 'mcnp-mctal'
    if _is_meshtal(lines):
        return 'mcnp-meshtal'
    return 'mcnp-input'


def _read_first_lines(file_path: str, n: int) -> list:
    """Read the first *n* non-blank lines from a file on disk."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    result = []
    with open(p, 'r', errors='replace') as f:
        for raw in f:
            stripped = raw.rstrip('\n\r')
            if stripped.strip():
                result.append(stripped)
                if len(result) >= n:
                    break
    return result


def _lines_from_content(content: str, n: int) -> list:
    """Extract the first *n* non-blank lines from a content string."""
    result = []
    for raw in content.splitlines():
        stripped = raw.rstrip()
        if stripped.strip():
            result.append(stripped)
            if len(result) >= n:
                break
    return result


def _is_mctal(lines: list) -> bool:
    """Check whether the first non-blank lines are a MCTAL header.

    Line 1 ends with three integers (dump, nps, random numbers) and is
    otherwise the code name, version and problem id, which MCNP may leave
    blank or fill with another code's name (D1SUNED, MCNPX); line 2 is the
    title and line 3 ``ntal <n>``. A blank title drops out of *lines*, so
    ``ntal`` is accepted on line 2 as well.
    """
    if len(lines) < 2:
        return False
    tokens = lines[0].split()
    if len(tokens) < 3 or not all(t.isdigit() for t in tokens[-3:]):
        return False
    return any(_NTAL.match(line) for line in lines[1:3])


def _is_meshtal(lines: list) -> bool:
    """Check whether the first non-blank lines are a MESHTAL (FMESH) header.

    The file opens with an optional ``mcnp version ...`` line, the title and
    ``Number of histories used for normalizing tallies``, then the first
    ``Mesh Tally Number`` line.
    """
    head = lines[:6]
    return any(_MESH_TALLY.match(line) for line in head) and any(
        "Number of histories used for normalizing tallies" in line for line in head)
