"""Reader of the PERT cards and material cards of an MCNP input deck."""

from .input import Input
from .perturbations import Perturbation, Pert
from kika.materials import Material, MaterialCollection
from kika.materials.parse_mcnp import read_card, read_material
from pathlib import Path
from typing import List, Optional, Tuple
import re

_PERT_KEYWORDS = {"CELL", "MAT", "RHO", "METHOD", "RXN", "ERG"}
_READ_CARD = re.compile(r"\s{0,4}READ\b.*?\bFILE\s*=\s*(\S+)", re.IGNORECASE)

def _read_PERT(lines, start_index):
    """Internal helper function to read and parse a PERT card from MCNP input lines.

    :param lines: List of input lines from the MCNP input file
    :type lines: List[str]
    :param start_index: Starting index of the PERT card in the lines list
    :type start_index: int

    :returns: A tuple containing:
        - Pert object with parsed data, or None if parsing fails
        - The new index after processing the PERT card
    :rtype: tuple[Pert, int]
    """
    text, end = read_card(lines, start_index)
    header, _, body = text.partition(" ")
    header_match = re.fullmatch(r'PERT(\d+):([A-Za-z,]+)', header, re.IGNORECASE)
    if not header_match:
        return None, end

    values = {}
    key = None
    for token in body.replace(",", " ").replace("=", "= ").split():
        if token.endswith("="):
            key = token[:-1].upper()
            if key not in _PERT_KEYWORDS:
                break
            values[key] = []
        elif key is None:
            break
        else:
            values[key].append(token)

    pert_attrs = {}
    if "CELL" in values:
        pert_attrs['cell'] = [int(x) for x in values["CELL"]]
    if "MAT" in values:
        pert_attrs['material'] = int(values["MAT"][0])
    if "RHO" in values:
        pert_attrs['rho'] = float(values["RHO"][0])
    if "METHOD" in values:
        pert_attrs['method'] = int(values["METHOD"][0])
    if "RXN" in values:
        pert_attrs['reactions'] = tuple(int(x) for x in values["RXN"])
    if "ERG" in values:
        erg = values["ERG"]
        pert_attrs['energy'] = (float(erg[0]), float(erg[1])) if len(erg) >= 2 else None

    return Pert(id=int(header_match.group(1)), particle=header_match.group(2).lower(), **pert_attrs), end


def read_card_target(line: str) -> Optional[str]:
    """The file a ``READ FILE=`` card names, or None when *line* is not one."""
    match = _READ_CARD.match(line)
    return match.group(1) if match else None


def expand_read_cards(path, seen: Tuple[Path, ...] = (), read: Optional[List[Path]] = None) -> List[str]:
    """The lines of the deck at *path* with each ``READ FILE=`` card replaced by the lines of its file.

    A READ file is looked for beside the file whose card names it. Every file opened
    is appended to *read*, the deck first.

    Raises
    ------
    FileNotFoundError
        If *path* or a file named by a READ card does not exist.
    ValueError
        If READ cards read each other in a loop.
    """
    path = Path(path).resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ValueError(f"READ cards form a loop: {chain}")
    with open(path, 'r') as f:
        lines = f.readlines()
    if read is not None:
        read.append(path)
    expanded = []
    for line in lines:
        name = read_card_target(line)
        if name is None:
            expanded.append(line)
            continue
        target = path.parent / name
        if not target.is_file():
            raise FileNotFoundError(
                f"{path.name}: '{line.strip()}' names {target}, which does not exist "
                f"(a READ file is looked for beside the input that reads it)"
            )
        expanded.extend(expand_read_cards(target, (*seen, path), read))
    return expanded

def read_mcnp(file_path, pert_metadata=None):
    """Read the PERT cards and the material cards of an MCNP input deck.

    Every other card is skipped. A ``READ FILE=`` card is replaced by the lines of
    its file, looked for beside the input that reads it. A PERT card is a line that
    starts with ``PERT`` in any case; a material card is a line that starts with
    ``m`` followed by the material number. A PERT card takes the keywords ``CELL``,
    ``MAT``, ``RHO``, ``METHOD``, ``RXN`` and ``ERG``, each followed by its values
    separated by blanks or commas; the first other keyword ends the card. The cards written by
    :func:`~kika.mcnp.pert_generator.generate_PERTcards` are read back, with the ``zaid`` and
    ``original_material`` of the ``c kika:pert_zaid=<zaid> pert_mat=<material>``
    comment line that precedes them (it applies to every PERT card after it, until
    the next such comment).

    Parameters
    ----------
    file_path
        Path to the deck, opened with the default text encoding.
    pert_metadata : list of tuple, optional
        ``(start_pert, end_pert, zaid, material)`` tuples. Each sets ``zaid`` and
        ``original_material`` of the PERT cards numbered from *start_pert* to
        *end_pert*, both included, and replaces what the comment lines gave.

    Returns
    -------
    ~kika.mcnp.input.Input
        The PERT cards and the materials found, in ``perturbation.pert`` and
        ``materials.by_id``. Both are empty when the deck has none. A material's
        ``mt`` card is in its ``thermal_scattering``. ``source_files`` lists the
        deck and every file its READ cards read, as absolute paths.

    Raises
    ------
    ValueError
        If a material card mixes positive and negative fractions or lists a
        nuclide twice, or an ``mt`` card names a material the deck does not define.
    FileNotFoundError
        If *file_path*, or a file named by a ``READ`` card, does not exist.
    ValueError
        Also if ``READ`` cards read each other in a loop.
    """
    inst = Input()  # instance of the input class
    inst.perturbation = Perturbation()
    inst.materials = MaterialCollection()

    source_files = []
    lines = expand_read_cards(file_path, read=source_files)
    inst.source_files = [str(p) for p in source_files]

    i = 0
    current_zaid = None
    current_original_mat = None
    thermal = {}
    while i < len(lines):
        line = lines[i].strip()
        # Detect kika:pert_zaid comment (new format with pert_mat, or old format without)
        zaid_match = re.search(r'kika:pert_zaid=(\d+)(?:\s+pert_mat=(\d+))?', line)
        if zaid_match:
            current_zaid = int(zaid_match.group(1))
            current_original_mat = int(zaid_match.group(2)) if zaid_match.group(2) else None
            i += 1
        elif line[:4].upper() == "PERT":
            pert_obj, i = _read_PERT(lines, i)
            if pert_obj:
                if current_zaid is not None:
                    pert_obj.zaid = current_zaid
                if current_original_mat is not None:
                    pert_obj.original_material = current_original_mat
                inst.perturbation.pert[pert_obj.id] = pert_obj
        elif re.match(r"mt(\d+)(?=\s|$)", line, re.IGNORECASE):
            text, i = read_card(lines, i)
            tables = text.split()
            thermal[int(tables[0][2:])] = tables[1:]
        elif line[:1].lower() == "m":
            material_obj, i = read_material(lines, i)
            if material_obj:
                inst.materials.by_id[material_obj.id] = material_obj
        else:
            i += 1
    for mat_id, tables in thermal.items():
        if mat_id not in inst.materials.by_id:
            raise ValueError(f"mt{mat_id} names S(a,b) tables for a material the input does not define")
        inst.materials.by_id[mat_id].thermal_scattering = tables

    # Apply explicit pert_metadata if provided
    if pert_metadata is not None:
        for start, end, zaid, material in pert_metadata:
            for pert_id, pert_obj in inst.perturbation.pert.items():
                if start <= pert_id <= end:
                    pert_obj.zaid = zaid
                    pert_obj.original_material = material

    return inst
