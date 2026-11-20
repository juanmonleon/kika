# ─────────────────────────────────────────────────────────────────────────────
# File: serpent_sens/parser.py
# Text parser for SERPENT .sens outputs → SensitivityFile
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np

from kika.serpent.sens import Response, SensitivityFile, SensitivitySet, Perturbation
from kika.serpent.utils import extract_numeric_list, extract_int_list, extract_string_list, parse_perturbation_label

# Regex patterns for named blocks
_SCALAR_INT_RE = re.compile(r"^\s*(SENS_N_\w+)\s*=\s*([+-]?\d+)\s*;\s*$", re.MULTILINE)
_BLOCK_RE = re.compile(r"^\s*(?P<name>SENS_\w+)\s*=\s*\[(?P<body>[\s\S]*?)\];", re.MULTILINE)

# Response variable blocks
_ED_RE = re.compile(
    r"ADJ_PERT_(?P<resp>[A-Za-z0-9_]+)_SENS\s*=\s*\[(?P<body>[\s\S]*?)\];",
    re.MULTILINE,
)
_INT_RE = re.compile(
    r"ADJ_PERT_(?P<resp>[A-Za-z0-9_]+)_SENS_E_INT\s*=\s*\[(?P<body>[\s\S]*?)\];",
    re.MULTILINE,
)


class ParseError(RuntimeError):
    """A Serpent sensitivity output is incomplete or inconsistent.

    Raised by :func:`~kika.serpent.read_sensitivity_file` and
    :func:`~kika.serpent.parse_sensitivity_text` when a required ``SENS_*``
    scalar or block is missing, a list or grid length disagrees with the counts in
    the header, or a response bin lacks its energy-dependent or integrated block.
    It is a ``RuntimeError``.
    """
    pass


def _read_text(path_or_text: str) -> str:
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8", errors="ignore")
    # Else treat as raw text
    return str(path_or_text)


def _get_scalar_int(name: str, text: str) -> Optional[int]:
    m = re.search(rf"^\s*{re.escape(name)}\s*=\s*([+-]?\d+)\s*;\s*$", text, re.MULTILINE)
    if m:
        return int(m.group(1))
    return None


def _get_block(name: str, text: str) -> Optional[str]:
    m = re.search(rf"^\s*{re.escape(name)}\s*=\s*\[(?P<body>[\s\S]*?)\];", text, re.MULTILINE)
    if not m:
        return None
    return m.group("body")


def _split_response_name(resp: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^(?P<base>.*)_BIN_(?P<bin>\d+)$", resp)
    if m:
        return m.group("base"), int(m.group("bin"))
    return resp, None


def parse_sensitivity_text(text: str) -> SensitivityFile:
    """Parse the text of a Serpent ``.sens`` file.

    Parameters
    ----------
    text : str
        Full content of the file.

    Returns
    -------
    SensitivityFile
        With the ``ADJ_PERT_<response>_SENS`` and ``_SENS_E_INT`` blocks grouped by
        base response (the name before ``_BIN_<n>``). Values, errors and energy edges
        are those of the file: nothing is reordered, normalised or converted. A base
        with one bin keeps the bare name unless the file wrote ``_BIN_0``.

    Raises
    ------
    ParseError
        (A ``RuntimeError`` of ``kika.serpent.parse_sens``.) If ``SENS_N_MAT``,
        ``SENS_N_ZAI``, ``SENS_N_PERT`` or ``SENS_N_ENE`` is missing; a list or block
        (``SENS_MAT_LIST``, ``SENS_ZAI_LIST``, ``SENS_PERT_LIST``, ``SENS_E``,
        ``SENS_LETHARGY_WIDTHS``) is missing or has a length that disagrees with the
        counts; no sensitivity block is found; a block has the wrong number of values;
        or a response bin lacks its energy-dependent or integrated block.
    """
    # ----------------------
    # Header scalars
    # ----------------------
    n_mat = _get_scalar_int("SENS_N_MAT", text)
    n_zai = _get_scalar_int("SENS_N_ZAI", text)
    n_pert = _get_scalar_int("SENS_N_PERT", text)
    n_ene = _get_scalar_int("SENS_N_ENE", text)
    n_mu = _get_scalar_int("SENS_N_MU", text)  # may be absent in docs

    missing = [name for name, val in [
        ("SENS_N_MAT", n_mat), ("SENS_N_ZAI", n_zai), ("SENS_N_PERT", n_pert), ("SENS_N_ENE", n_ene)
    ] if val is None]
    if missing:
        raise ParseError(f"Missing required scalars: {', '.join(missing)}")

    # ----------------------
    # Lists / arrays
    # ----------------------
    mats_block = _get_block("SENS_MAT_LIST", text)
    zais_block = _get_block("SENS_ZAI_LIST", text)
    perts_block = _get_block("SENS_PERT_LIST", text)
    e_block = _get_block("SENS_E", text)
    leth_block = _get_block("SENS_LETHARGY_WIDTHS", text)

    if any(x is None for x in [mats_block, zais_block, perts_block, e_block, leth_block]):
        raise ParseError("One or more required blocks (MAT_LIST, ZAI_LIST, PERT_LIST, E, LETHARGY_WIDTHS) are missing.")

    materials = extract_string_list(mats_block or "")
    zais = extract_int_list(zais_block or "")
    perts_raw = extract_string_list(perts_block or "") 
    energy_grid = np.array(extract_numeric_list(e_block or ""), dtype=float)
    lethargy_widths = np.array(extract_numeric_list(leth_block or ""), dtype=float)

    # Basic validation against header counts
    if len(materials) != n_mat:
        raise ParseError(f"SENS_MAT_LIST length {len(materials)} != SENS_N_MAT {n_mat}")
    if len(zais) != n_zai:
        raise ParseError(f"SENS_ZAI_LIST length {len(zais)} != SENS_N_ZAI {n_zai}")
    if len(perts_raw) != n_pert:
        raise ParseError(f"SENS_PERT_LIST length {len(perts_raw)} != SENS_N_PERT {n_pert}")
    
    if energy_grid.shape[0] != n_ene + 1:
        raise ParseError(f"SENS_E length {energy_grid.shape[0]} != SENS_N_ENE+1 {n_ene+1}")
    if lethargy_widths.shape[0] != n_ene:
        raise ParseError(f"SENS_LETHARGY_WIDTHS length {lethargy_widths.shape[0]} != SENS_N_ENE {n_ene}")

    # Normalize perturbations
    perts: List[Perturbation] = [
        parse_perturbation_label(raw, i) for i, raw in enumerate(perts_raw)
    ]

    # ----------------------
    # Responses
    # ----------------------
    ed_matches = list(_ED_RE.finditer(text))
    int_matches = list(_INT_RE.finditer(text))

    if not ed_matches or not int_matches:
        raise ParseError("No sensitivity data blocks found (ED or INT).")

    # Group by base response
    tmp: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    # structure: tmp[base]["ED"][bin] -> ndarray(M,Z,P,E,2)
    #            tmp[base]["INT"][bin] -> ndarray(M,Z,P,2)

    M, Z, P, E = n_mat, n_zai, n_pert, n_ene

    for m in ed_matches:
        resp = m.group("resp")
        base, b = _split_response_name(resp)
        nums = extract_numeric_list(m.group("body"))
        expected = M * Z * P * E * 2
        if len(nums) != expected:
            raise ParseError(
                f"ED block {resp}: expected {expected} numbers for shape (M,Z,P,E,2), got {len(nums)}"
            )
        arr = np.array(nums, dtype=float).reshape(M, Z, P, E, 2)
        tmp.setdefault(base, {}).setdefault("ED", {})[b or 0] = arr

    for m in int_matches:
        resp = m.group("resp")
        base, b = _split_response_name(resp)
        nums = extract_numeric_list(m.group("body"))
        expected = M * Z * P * 2
        if len(nums) != expected:
            raise ParseError(
                f"INT block {resp}: expected {expected} numbers for shape (M,Z,P,2), got {len(nums)}"
            )
        arr = np.array(nums, dtype=float).reshape(M, Z, P, 2)
        tmp.setdefault(base, {}).setdefault("INT", {})[b or 0] = arr

    # Assemble SensitivitySet per base
    data: Dict[str, SensitivitySet] = {}
    for base, parts in tmp.items():
        ed_bins = parts.get("ED", {})
        int_bins = parts.get("INT", {})
        all_bins = sorted(set(ed_bins.keys()) | set(int_bins.keys()))
        if not all_bins:
            continue
        # Ensure each bin has both ED and INT
        missing = [b for b in all_bins if b not in ed_bins or b not in int_bins]
        if missing:
            raise ParseError(f"Base '{base}' missing ED or INT for bins: {missing}")
        # Stack into (n_bins, ...)
        ed_stack = np.stack([ed_bins[b] for b in all_bins], axis=0)
        int_stack = np.stack([int_bins[b] for b in all_bins], axis=0)
        responses = [
            Response(base_name=base, bin_index=b, full_name=f"{base}_BIN_{b}")
            for b in all_bins
        ] if len(all_bins) > 1 else [
            Response(base_name=base, bin_index=0, full_name=base)
        ]
        # Edge case: if only one bin but input used _BIN_0, keep full_name with suffix
        if len(all_bins) == 1 and (0 in ed_bins) and (0 in int_bins):
            # If original text had explicit suffix, prefer that name
            had_suffix = re.search(rf"ADJ_PERT_{re.escape(base)}_BIN_0_SENS\s*=", text) is not None
            if had_suffix:
                responses = [Response(base_name=base, bin_index=0, full_name=f"{base}_BIN_0")]
        data[base] = SensitivitySet(responses=responses, energy_dependent=ed_stack, integrated=int_stack)

    # Build file object
    sf = SensitivityFile(
        n_materials=n_mat,
        n_nuclides=n_zai,
        n_perturbations=n_pert,
        n_energy_bins=n_ene,
        n_mu_bins=n_mu,
        materials=materials,
        nuclides=zais,
        perturbations_raw=perts_raw,
        perturbations=perts,
        energy_grid=energy_grid,
        lethargy_widths=lethargy_widths,
        data=data,
        meta={"notes": "Parsed by serpent_sens.parser"},
    )
    return sf


def read_sensitivity_file(path_or_text: str) -> SensitivityFile:
    """Load a Serpent ``.sens`` file from a path or from its text.

    Parameters
    ----------
    path_or_text : str
        Path of an existing file, which is read from disk. Any other string is taken
        as the content of the file.

    Returns
    -------
    SensitivityFile

    Raises
    ------
    ParseError
        As for :func:`parse_sensitivity_text`. A path that does not exist is parsed
        as text and fails here.
    """
    text = _read_text(path_or_text)
    return parse_sensitivity_text(text)