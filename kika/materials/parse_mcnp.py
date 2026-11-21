import math
import re
from typing import List, Optional, Tuple

from .material import Material, _symbol_of


def _parse_nuclide_token(token: str) -> Tuple[int, Optional[str]]:
    """Return (zaid, library_suffix) parsed from token like '1001.80c' or '1001'."""
    if "." in token:
        nuclide_parts = token.split(".")
        return int(nuclide_parts[0]), nuclide_parts[1]
    return int(token), None


def _parse_kika_name_comment(line: str) -> Optional[str]:
    """Parse a KIKA material name comment line.

    Expected format: 'c KIKA_MAT_NAME: Material Name Here'

    Parameters
    ----------
    line : str
        Comment line to parse.

    Returns
    -------
    str or None
        Material name if found, None otherwise.
    """
    stripped = line.strip()
    if stripped.lower().startswith("c kika_mat_name:"):
        # Extract the name after the marker
        name_part = stripped[16:].strip()  # len("c KIKA_MAT_NAME:") = 16
        if name_part:
            return name_part
    return None


def _parse_kika_density_comment(line: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse a KIKA density comment from a material line.

    Expected format in inline comment: '$ KIKA_DENSITY: X.XXe+XX g/cc, Y.YYe+YY atoms/b-cm'
    or: '$ KIKA_DENSITY: X.XXe+XX g/cc'

    Parameters
    ----------
    line : str
        Line containing the density comment.

    Returns
    -------
    tuple of (float or None, float or None)
        (mass_density_gcc, atomic_density_abc) or (None, None) if not found.
    """
    if "KIKA_DENSITY:" not in line:
        return None, None

    # Find the KIKA_DENSITY part
    match = re.search(r'KIKA_DENSITY:\s*([\d.eE+-]+)\s*g/cc(?:,\s*([\d.eE+-]+)\s*atoms/b-cm)?', line)
    if match:
        mass_density = float(match.group(1))
        atomic_density = float(match.group(2)) if match.group(2) else None
        return mass_density, atomic_density

    return None, None


_LIB_KEYWORDS = {"nlib", "plib", "pnlib", "elib", "hlib", "alib", "slib", "tlib", "dlib", "ylib"}
_OTHER_KEYWORDS = {"gas", "estep", "hstep", "cond", "refi", "refc", "refs"}


def _is_comment(line: str) -> bool:
    low = line.strip().lower()
    return low == "c" or low.startswith("c ") or low.startswith("c\t")


def read_card(lines: List[str], start_index: int) -> Tuple[str, int]:
    """The text of the MCNP card that starts at *start_index*, with ``$`` comments, ``&``
    and comment lines inside it removed, and the index after its last line.

    A line continues the card when the previous one ends in ``&``, when it starts
    with five blanks or a tab, or when it starts with a number. A blank line ends it.
    """
    texts = [lines[start_index].split("$", 1)[0].strip()]
    continued = texts[0].endswith("&")
    end = i = start_index + 1
    while i < len(lines):
        raw = lines[i].rstrip("\r\n")
        if not raw.strip():
            break
        if _is_comment(raw):
            i += 1
            continue
        body = raw.split("$", 1)[0]
        if not (continued or raw.startswith("     ") or raw.startswith("\t")
                or re.match(r"\s*\d", body)):
            break
        texts.append(body.strip())
        continued = texts[-1].endswith("&")
        i += 1
        end = i
    text = " ".join(t[:-1] if t.endswith("&") else t for t in texts)
    return re.sub(r"\s*=\s*", "=", text), end


def _check_repeat(material: Material, zaid: int, library: Optional[str], fraction: float) -> None:
    """Refuse a second entry for a nuclide unless it names the same amount under another data class."""
    symbol = _symbol_of(zaid)
    if symbol not in material.nuclide:
        return
    nuclide = material.nuclide[symbol]
    lib_key = material._infer_lib_key_from_suffix(library) if library else None
    if lib_key is None or not nuclide.libs or lib_key in nuclide.libs:
        raise ValueError(f"Material {material.id} lists {symbol} more than once for the same data")
    if fraction != nuclide.fraction:
        raise ValueError(
            f"Material {material.id} gives {symbol} {nuclide.fraction} and {fraction} under different "
            f"data classes; kika keeps one amount per nuclide"
        )


def read_material(lines: List[str], start_index: int) -> Tuple[Optional[Material], int]:
    """Read and parse a material card from MCNP input lines.

    Handles KIKA-formatted comments for material name and density:
    - 'c KIKA_MAT_NAME: <name>' comment before material card
    - '$ KIKA_DENSITY: <mass> g/cc, <atomic> atoms/b-cm' inline comment
    - '$ <SYMBOL>' inline comments after nuclide fractions

    A negative ZAID fraction is a weight fraction (``'wo'``) and a positive one an
    atomic fraction (``'ao'``); the stored fraction is the absolute value, and the
    material's ``fraction_type`` is the one the card uses (``'ao'`` when it has no
    nuclides). The card continues on lines that start with five blanks, a tab or a
    number, or after a line ending in ``&``; comment lines inside it are skipped and
    a blank line ends it. The library keywords (``nlib``, ``plib``, ``pnlib``,
    ``elib``, ``hlib``, ``alib``, ``slib``, ``tlib``, ``dlib``) go to ``libs`` and
    the other keywords to ``keywords``, both written back by
    :meth:`~kika.materials.Material.to_mcnp`. A ZAID keeps the form the card gives
    it; isomers in the LANL (``95642``) or NEA (``95342``) form are recognised. The
    ``KIKA_DENSITY`` mass density is stored in g/cc; the atomic density in the
    comment is not used. The ``mt`` card is a separate card: see
    :func:`~kika.mcnp.parse_input.read_mcnp`.

    Parameters
    ----------
    lines
        Lines of the MCNP input.
    start_index
        Index in *lines* of the first line of the card.

    Returns
    -------
    tuple of (~kika.materials.Material or None, int)
        The material, or None if the line at *start_index* is not an ``m<n>`` card,
        and the index of the first line not read.

    Raises
    ------
    ValueError
        If the card mixes positive and negative fractions, lists a nuclide twice
        for the same data class (a ``.03c`` and a ``.04p`` entry with the same
        amount are one nuclide with two libraries), or holds a token that is
        neither a keyword nor a ZAID and its fraction.
    """
    first = lines[start_index].strip()
    match = re.match(r"^m(\d+)(?=\s|$)", first, re.IGNORECASE)
    if not match:
        return None, start_index + 1

    material_name: Optional[str] = None
    for back_offset in range(1, min(4, start_index + 1)):
        prev_line = lines[start_index - back_offset].strip()
        if prev_line and not _is_comment(prev_line):
            break
        name = _parse_kika_name_comment(prev_line)
        if name is not None:
            material_name = name
            break

    text, end = read_card(lines, start_index)
    material_id = int(match.group(1))
    material_obj = Material(id=material_id, name=material_name)
    mass_density, _ = _parse_kika_density_comment(first)
    if mass_density is not None:
        material_obj.set_density(mass_density, "g/cc")

    tokens = text.split()[1:]
    fraction_type: Optional[str] = None
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if "=" in tok:
            key, value = tok.split("=", 1)
            key = key.lower()
            if key in _LIB_KEYWORDS:
                material_obj.libs[key] = value
            elif key == "lib":
                lib_key = material_obj._infer_lib_key_from_suffix(value)
                if lib_key is None:
                    raise ValueError(f"Material {material_id}: 'lib={value}' names no library class")
                material_obj.libs[lib_key] = value
            else:
                if key not in _OTHER_KEYWORDS:
                    material_obj.report.warn(
                        f"m{material_id}: keyword '{key}' is not an MCNP m-card keyword kika "
                        f"knows; it is kept and written back as read"
                    )
                material_obj.keywords[key] = value
            idx += 1
            continue
        try:
            zaid, specific_lib = _parse_nuclide_token(tok)
            raw_fraction = float(tokens[idx + 1])
        except (ValueError, IndexError):
            raise ValueError(
                f"Material {material_id}: '{tok}' is neither a keyword nor a ZAID followed by its fraction"
            ) from None
        current_type = "wo" if math.copysign(1.0, raw_fraction) < 0 else "ao"  # -0.0 is a weight fraction
        if fraction_type is None:
            fraction_type = current_type
        elif fraction_type != current_type:
            raise ValueError(f"Material {material_id} has mixed atomic/weight fractions")
        _check_repeat(material_obj, zaid, specific_lib, abs(raw_fraction))
        material_obj.add_nuclide(nuclide=zaid, fraction=abs(raw_fraction),
                                 fraction_type=current_type, library=specific_lib)
        idx += 2

    material_obj.fraction_type = fraction_type or "ao"
    return material_obj, end
