"""Parse Serpent material definitions from input files.

Serpent material format::

    mat <name> <density> [rgb R G B] [tmp T] [moder lib ZA] ...
        <zaid>.<lib> <fraction>
        <zaid>.<lib> <fraction>
        ...

Density convention:
    - Negative value: mass density in g/cc
    - Positive value: atomic density in atoms/b-cm
    - ``sum``: density calculated from isotope fractions
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .._utils import symbol_to_zaid
from .material import Material, MaterialCollection, _symbol_of


_NUCLIDE_NAME = re.compile(r"^(\d+|[A-Za-z]{1,2}-\d+m?)(\.\w+)?$")


def _strip_comment(line: str) -> str:
    """Remove Serpent ``%`` comments from a line."""
    idx = line.find("%")
    if idx >= 0:
        return line[:idx]
    return line


def _tokens_from(lines: List[str], start_index: int):
    """``(token, line index)`` of every token from *start_index* on, comments removed."""
    for i in range(start_index, len(lines)):
        for token in _strip_comment(lines[i]).split():
            yield token, i


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _nuclide(name: str) -> Tuple[int, Optional[str]]:
    """ZAID and library suffix of a Serpent nuclide name, ``92235.09c`` or ``U-235.09c``."""
    base, _, lib = name.partition(".")
    if not base.isdigit():
        symbol, mass = base.split("-")
        base = str(symbol_to_zaid(symbol.capitalize() + mass))
    return int(base), lib or None


def _read_serpent_material(
    lines: List[str], start_index: int, mat_id: int
) -> Tuple[Material, int]:
    """Parse the Serpent ``mat`` card at *start_index*.

    The card runs, across lines, from ``mat`` to the first token that is neither one
    of its options nor a nuclide followed by its fraction.

    Returns
    -------
    tuple of (Material, int)
        Parsed Material and the index of the line after its last token.
    """
    stream = _tokens_from(lines, start_index)
    tokens: List[Tuple[str, int]] = []

    def peek(n: int) -> List[str]:
        while len(tokens) < n:
            nxt = next(stream, None)
            if nxt is None:
                break
            tokens.append(nxt)
        return [t for t, _ in tokens[:n]]

    head = peek(3)
    mat_name = head[1]
    material = Material(id=mat_id, name=mat_name)
    if head[2].lower() == "sum":
        material.metadata["serpent_density_sum"] = True
    else:
        density_val = float(head[2])
        if density_val < 0:
            material.set_density(abs(density_val), "g/cc")
        else:
            material.set_density(density_val, "atoms/b-cm")
    pos = 3
    last_line = tokens[2][1]

    def take(n: int) -> List[str]:
        nonlocal pos, last_line
        got = peek(pos + n)[pos:pos + n]
        if len(got) < n:
            raise ValueError(f"Serpent material '{mat_name}': the input ends inside its card")
        last_line = tokens[pos + n - 1][1]
        pos += n
        return got

    fraction_type: Optional[str] = None
    while True:
        ahead = peek(pos + 2)[pos:]
        if not ahead:
            break
        tok = ahead[0].lower()
        if tok in _OPTION_ARGS and fraction_type is None:
            args = take(1 + _OPTION_ARGS[tok])[1:]
            _set_option(material, tok, args)
            continue
        if len(ahead) == 2 and _NUCLIDE_NAME.match(ahead[0]) and _is_number(ahead[1]):
            name, raw = take(2)
            zaid, lib_suffix = _nuclide(name)
            raw_fraction = float(raw)
            current_type = "wo" if raw_fraction < 0.0 else "ao"
            if fraction_type is None:
                fraction_type = current_type
            elif fraction_type != current_type:
                raise ValueError(f"Material '{mat_name}' has mixed atomic/weight fractions")
            if _symbol_of(zaid) in material.nuclide:
                raise ValueError(f"Material '{mat_name}' lists {_symbol_of(zaid)} more than once")
            material.add_nuclide(nuclide=zaid, fraction=abs(raw_fraction),
                                 fraction_type=current_type, library=lib_suffix)
            continue
        break

    if fraction_type is None:
        raise ValueError(
            f"Serpent material '{mat_name}' has no nuclides: its card stops at "
            f"'{ahead[0] if ahead else 'the end of the input'}', which is neither an option kika "
            f"knows nor a nuclide and its fraction"
        )
    material.fraction_type = fraction_type
    return material, last_line + 1


# Number of arguments of each option of the mat card; moder takes one pair per use.
_OPTION_ARGS = {"tmp": 1, "tms": 1, "ettm": 1, "burn": 1, "vol": 1, "coeft": 1,
                "coefs": 1, "tft": 2, "fix": 2, "moder": 2, "rgb": 3}


def _set_option(material: Material, key: str, args: List[str]) -> None:
    if key == "tmp":
        material.set_temperature(float(args[0]))
    elif key in ("tms", "ettm"):
        material.metadata["tms"] = float(args[0])
    elif key == "burn":
        material.metadata["burn"] = int(float(args[0]))
    elif key in ("vol", "coeft"):
        material.metadata[key] = float(args[0])
    elif key == "coefs":
        material.metadata["coefs"] = args[0]
    elif key == "tft":
        material.metadata["tft"] = (float(args[0]), float(args[1]))
    elif key == "fix":
        material.metadata["fix"] = (args[0], float(args[1]))
    elif key == "moder":
        material.metadata.setdefault("moder", []).append((args[0], int(args[1])))
    elif key == "rgb":
        material.metadata["rgb"] = tuple(int(a) for a in args)
    if key in ("ettm", "coeft", "coefs"):
        material.report.lost(
            f"Serpent material '{material.name}': option '{key}' is read but "
            f"{'written back as tms' if key == 'ettm' else 'not written back'} by to_serpent"
        )


def _find_material_spans(
    lines: List[str],
) -> List[Tuple[str, int, int]]:
    """Find (name, start_line, end_line) spans of all materials in the file.

    Used by ``write_to_serpent`` for targeted replacement.
    """
    spans: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue
        tokens = stripped.split()
        if tokens[0].lower() == "mat" and len(tokens) >= 3:
            mat_name = tokens[1]
            start = i
            # Find where this material ends
            _, end = _read_serpent_material(lines, i, mat_id=0)
            spans.append((mat_name, start, end))
            i = end
        else:
            i += 1
    return spans


def read_serpent_materials(
    source: Union[str, Path, List[str]],
) -> MaterialCollection:
    """Parse all materials from a Serpent input file.

    Each ``mat`` card becomes a material named as in the card. A negative nuclide
    fraction is a weight fraction (``'wo'``) and a positive one an atomic fraction
    (``'ao'``); the stored fraction is the absolute value. A negative card density
    is stored as g/cc and a positive one as atoms/b-cm. A density of ``sum`` stores
    no density (``density`` stays None) and sets
    ``metadata["serpent_density_sum"] = True``.

    A card runs, across lines, from ``mat`` to the first token that is neither one
    of its options (``tmp``, ``tms``, ``ettm``, ``burn``, ``vol``, ``coeft``,
    ``coefs``, ``tft``, ``fix``, ``moder``, ``rgb``) nor a nuclide and its fraction.
    Nuclides are named ``92235.09c`` or ``U-235.09c``; a symbolic name is kept as
    its ZAID. Options that :meth:`~kika.materials.Material.to_serpent` does not
    write back are listed in the material's ``report``.

    Parameters
    ----------
    source : str, Path, or list of str
        File path or list of lines to parse.

    Returns
    -------
    ~kika.materials.MaterialCollection
        Materials keyed by IDs assigned 1, 2, ... in file order.

    Raises
    ------
    ValueError
        If a material mixes positive and negative nuclide fractions, lists a
        nuclide twice, or has no nuclides (its card stops at an option kika does
        not know).
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8", errors="surrogateescape") as f:
            lines = f.readlines()
    else:
        lines = list(source)

    collection = MaterialCollection()
    mat_id = 1
    i = 0

    while i < len(lines):
        stripped = _strip_comment(lines[i]).strip()
        if not stripped:
            i += 1
            continue

        tokens = stripped.split()
        if tokens[0].lower() == "mat" and len(tokens) >= 3:
            material, next_i = _read_serpent_material(lines, i, mat_id)
            collection.add_material(material)
            mat_id += 1
            i = next_i
        else:
            i += 1

    return collection
