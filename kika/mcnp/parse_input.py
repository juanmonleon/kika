"""Reader of the PERT cards and material cards of an MCNP input deck."""

from .input import Input
from .perturbations import Perturbation, Pert
from kika.materials import Material, MaterialCollection
from kika.materials.parse_mcnp import read_card, read_material
import re

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
    i = start_index
    line = lines[i].strip()
    card_lines = []
    
    # Accumulate lines that end with '&'
    while line.endswith("&"):
        card_lines.append(line.rstrip("&").strip())
        i += 1
        if i < len(lines):
            line = lines[i].strip()
        else:
            break
    card_lines.append(line)
    full_line = " ".join(card_lines)
    
    # Tokenize the full card line
    tokens = full_line.split()
    # Parse header: e.g. "PERT1:n"
    header_match = re.match(r'PERT(\d+):(\w)', tokens[0])
    if not header_match:
        return None, i + 1
        
    pert_num = int(header_match.group(1))
    particle = header_match.group(2)
    
    allowed_keywords = {"CELL", "MAT", "RHO", "METHOD", "RXN", "ERG"}
    pert_attrs = {}
    j = 1
    while j < len(tokens):
        token = tokens[j]
        if "=" in token:
            key, val = token.split("=", 1)
            key = key.upper()
            if key not in allowed_keywords:
                break
            if key == "CELL":
                cell_vals = val.replace(',', ' ').split()
                # Consume following bare tokens (space-separated cell IDs)
                j += 1
                while j < len(tokens) and "=" not in tokens[j]:
                    cell_vals.extend(tokens[j].replace(',', ' ').split())
                    j += 1
                pert_attrs['cell'] = [int(x) for x in cell_vals]
                continue
            elif key == "MAT":
                pert_attrs['material'] = int(val)
            elif key == "RHO":
                pert_attrs['rho'] = float(val)
            elif key == "METHOD":
                pert_attrs['method'] = int(val)
            elif key == "RXN":
                pert_attrs['reaction'] = int(val)
            elif key == "ERG":
                erg_numbers = [val]
                j += 1
                while j < len(tokens) and "=" not in tokens[j]:
                    erg_numbers.append(tokens[j])
                    j += 1
                if len(erg_numbers) >= 2:
                    pert_attrs['energy'] = (float(erg_numbers[0]), float(erg_numbers[1]))
                else:
                    pert_attrs['energy'] = None
                continue
            j += 1
        else:
            break
    
    return Pert(id=pert_num, particle=particle, **pert_attrs), i + 1

def read_mcnp(file_path, pert_metadata=None):
    """Read the PERT cards and the material cards of an MCNP input deck.

    Every other card is skipped. A PERT card is a line that starts with ``PERT``
    in capitals (continuation with a trailing ``&``); a material card is a line that
    starts with a lowercase ``m`` followed by the material number. A PERT card takes
    the keywords ``CELL``, ``MAT``, ``RHO``, ``METHOD``, ``RXN`` and ``ERG``; the
    first other keyword ends the card. The cards written by
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
        ``mt`` card is in its ``thermal_scattering``.

    Raises
    ------
    ValueError
        If a material card mixes positive and negative fractions or lists a
        nuclide twice, or an ``mt`` card names a material the deck does not define.
    FileNotFoundError
        If *file_path* does not exist.
    """
    inst = Input()  # instance of the input class
    inst.perturbation = Perturbation()
    inst.materials = MaterialCollection()

    with open(file_path, 'r') as f:
        lines = f.readlines()

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
        elif line.startswith("PERT"):
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
