"""
ENDF section operations (remove sections).
"""
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils import parse_endf_id
from .update_directory import update_mf1_directory
from ...utils import get_endf_logger

logger = get_endf_logger(__name__)


def remove_sections(
    content: str,
    sections: List[Tuple[int, Optional[int]]],
) -> Tuple[str, int]:
    """Remove MF/MT sections from ENDF content, bookkeeping included.

    The SEND record of a removed MT and the FEND record of an MF left with
    nothing go with it, and MF1/451's directory is rebuilt from what actually
    survived, so the result is a tape a parser can read rather than a tape with
    the right data in it.

    Parameters
    ----------
    content : str
        Raw ENDF file content.
    sections : list of (MF, MT) tuples
        Sections to remove. MT=None means remove entire MF.

    Returns
    -------
    (modified_content, sections_removed_count)
    """
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # Build set of specific (mf, mt) pairs to remove
    remove_specific: set = set()  # (mf, mt) pairs
    remove_whole_mf: set = set()  # mf values where MT=None

    for mf, mt in sections:
        if mt is None:
            remove_whole_mf.add(mf)
        else:
            remove_specific.add((mf, mt))

    # Protect MF1/MT451
    remove_specific.discard((1, 451))

    lines = content.splitlines(keepends=True)

    kept_lines = []
    sections_removed = set()

    # A SEND record closes the MT section immediately before it and a FEND the
    # MF block immediately before it, so both are decided **locally**: they go
    # when nothing was kept out of what they close. Deciding a SEND against the
    # first survivor anywhere in its MF instead left one behind for every MT
    # removed after the first survivor, and FEND records were never considered
    # at all, so a fully removed MF left its FEND stranded next to the previous
    # one. Neither tape is valid ENDF.
    kept_since_send = False   # a data line survived since the last SEND/FEND
    kept_since_fend = False   # ... since the last FEND
    saw_data_since_fend = False  # ... and whether there was any to survive

    for line in lines:
        if len(line.rstrip('\n')) < 75:
            kept_lines.append(line)
            continue

        mat, mf, mt = parse_endf_id(line)
        if mf is None or mt is None:
            kept_lines.append(line)
            continue

        if mf > 0 and mt > 0:
            # Data line. MF1/451 is protected: the directory is rebuilt below,
            # never cut, even when the whole of MF1 was named for removal.
            saw_data_since_fend = True
            if (mf, mt) != (1, 451) and (
                mf in remove_whole_mf or (mf, mt) in remove_specific
            ):
                sections_removed.add((mf, mt))
                continue
            kept_since_send = True
            kept_since_fend = True
            kept_lines.append(line)
            continue

        if mf > 0 and mt == 0:
            # SEND
            if kept_since_send:
                kept_lines.append(line)
            kept_since_send = False
            continue

        if mf == 0 and mt == 0:
            # FEND (MAT > 0) closes an MF block. MEND (MAT 0) and TEND (MAT -1)
            # close the material and the tape, and the TPID that opens a tape
            # wears the same id columns with no block behind it at all — which
            # is what ``saw_data_since_fend`` tells apart from an empty block.
            drop = (
                mat is not None and mat > 0
                and saw_data_since_fend and not kept_since_fend
            )
            if not drop:
                kept_lines.append(line)
            kept_since_send = False
            kept_since_fend = False
            saw_data_since_fend = False
            continue

        kept_lines.append(line)

    if not sections_removed:
        return content, 0

    modified_content = "".join(kept_lines)

    # Write to temp file and update MF1/MT451 directory
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".endf", delete=False, encoding="utf-8", newline=""
        ) as tmp:
            tmp.write(modified_content)
            tmp_path = tmp.name

        update_mf1_directory(tmp_path)

        with open(tmp_path, "r", encoding="utf-8", newline="") as f:
            modified_content = f.read()
    except Exception as e:
        logger.warning(f"Directory update after removal failed (non-fatal): {e}")
    finally:
        if tmp_path:
            p = Path(tmp_path)
            if p.exists():
                p.unlink()

    return modified_content, len(sections_removed)
