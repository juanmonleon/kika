"""The fixed-width record grammar the nuclear-data formats share.

ENDF-6 writes 11-character numeric fields, six to a line, followed by a
14-character identification field. **That convention is not ENDF's property.**
COVERX, COVFIL and BOXER use the same one, which is why
``kika/cov/parse_covmat.py`` used to import all of this from
``kika.endf.utils`` -- the single import in ``kika/cov`` that ran at *import*
time rather than at call time, and the one the layering ratchet called "the
genuine leak".

Moved here in phase 4's P4, on the precedent phase 2 set with
``interpolate_1d_endf`` -> ``kika.processing.interpolation``: the interpolation
laws are not ENDF's property either, and the fix for a real shared dependency
is to put the shared thing where both callers can reach it without one
importing the other.

``kika.endf.utils`` re-exports every name below, and that is **not** tidiness:
``kika/tests/test_library_export_surface.py`` pins
``("kika.endf.utils", "parse_endf_id")`` as public surface, and the MF1/MF2/MF4
classes, the parsers, the writers and ``scripts/build_group_cross.py`` all
import from there.

What deliberately stayed behind in ``kika.endf.utils``: the SEND/FEND/MEND/TEND
emitters and ``_TERMINATION_FORMATS``. A terminator record is an ENDF-6
structural statement, not a fixed-width convention -- COVERX has no SEND.

This module is a leaf. It imports nothing from kika.
"""
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _mantissa_precision(exponent: int) -> int:
    """Decimal places the mantissa gets, so the field stays 11 characters wide."""
    if abs(exponent) < 10:
        return 6
    if abs(exponent) < 100:
        return 5
    return 4


def format_endf_number(value: Union[int, float, None], width: int = 11) -> str:
    """
    Format a number according to ENDF specifications.

    The output is an 11-character field made up as follows:
      - The first character is '-' if the number is negative or a blank if positive.
      - The number is written in scientific notation without an 'E'.
      - When the exponent (after normalization) has only one digit (|exponent| < 10),
        the mantissa is printed with 6 decimal digits and the exponent with one digit.
      - When the exponent has two digits (|exponent| >= 10), the mantissa is printed with 5 decimal digits and the exponent with two digits.
      - When the exponent has three digits (|exponent| >= 100), the mantissa is
        printed with 4 decimal digits and the exponent with three digits.

    For example:
      - A number like -3.14159e-1 will be formatted as "-3.141590-1".
      - A number like 1.234567e+5 will be formatted as " 1.234567+5".
      - A number like 1.0e10 will be formatted as " 1.00000+10".
      - A number like 1.5963e-100 will be formatted as " 1.5963-100".

    The three-digit case is not hypothetical and used to be **written as zero**.
    ``tsl-ortho-H.endf`` tabulates S(α, β) down to 1e-100 and below — 1 403 of
    its MF7/MT4 records carry such a value — and every one of them came back
    from this function as ``" 0.000000+0"``. Silently: nothing warned, and the
    line stayed the right width, so the loss was invisible to a caller that did
    not diff against the source. Three digits is also the end of the ladder,
    since a finite double cannot exceed 1e308.

    Args:
        value: The number to be formatted. If None, returns a blank field.
        width: The total field width (default is 11 characters).

    Returns:
        A string representing the formatted number in ENDF style.
    """
    if value is None:
        return " " * width

    # Special handling for zero: use exponent 0 (one-digit) and 6 decimal places.
    if value == 0:
        return " 0.000000+0"

    if not math.isfinite(value):
        raise ValueError(f"Cannot format non-finite ENDF value: {value}")

    sign_char = "-" if value < 0 else " "
    abs_val = abs(value)

    # Python's own scientific formatting rounds the exact binary value once,
    # correctly. The mantissa used to be ``abs_val / 10**exponent`` rounded
    # afterwards -- a division that rounds first and a format that rounds
    # again, so 7.0760435e-4 (NNDC's Fe-56 MT103) came out 7.076043 where
    # FUDGE and any correct rounding write 7.076044. The reader had the same
    # defect and lost it on 2026-08-24 (memory `endf-float-parse-rounded-twice`).
    # The number of decimals keeps the field at 11 characters: one is given up
    # for each extra digit the exponent needs, and a carry that changes the
    # exponent's width (9.99999e9 -> 1.0e10) is formatted again.
    prec = _mantissa_precision(int(math.floor(math.log10(abs_val))))
    while True:
        mantissa_str, exponent_str = f"{abs_val:.{prec}e}".split("e")
        exponent = int(exponent_str)
        if _mantissa_precision(exponent) == prec:
            break
        prec = _mantissa_precision(exponent)

    exp_str = f"{abs(exponent):d}" if abs(exponent) < 10 else (
        f"{abs(exponent):02d}" if abs(exponent) < 100
        else f"{abs(exponent):03d}"
    )
    exp_sign = '+' if exponent >= 0 else '-'

    formatted = f"{sign_char}{mantissa_str}{exp_sign}{exp_str}"
    return formatted.rjust(width)


#: ``10.0 ** k`` for ``|k| <= 22``: the powers of ten a double holds exactly.
_EXACT_POWERS = {k: float(10 ** k) for k in range(23)}


def round_to_endf_field(values) -> "np.ndarray":
    """Each of *values* as :func:`format_endf_number` writes it and it reads back.

    The same doubles as ``parse_number(format_endf_number(v))``, bit for bit,
    without a string per value: the mantissa is rounded to its 7 (6, 5)
    significant digits as an integer ``N`` and the field's value is
    ``N * 10**k`` or ``N / 10**-k``, one correctly rounded operation on two
    exact operands while ``|k| <= 22`` -- which is what a decimal-to-double
    conversion of those digits gives. Values whose rounding is too close to a
    tie to decide in floating point, or whose ``k`` is out of that range, go
    through the strings.
    """
    import numpy as np

    v = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(v)):
        raise ValueError("Cannot format non-finite ENDF value")
    out = np.zeros(v.size)
    a = np.abs(v)
    live = np.flatnonzero(a > 0)
    if live.size == 0:
        return out
    a = a[live]
    with np.errstate(divide="ignore"):
        exponent = np.floor(np.log10(a)).astype(np.int64)
    # The mantissa as format_endf_number computes it, divisor and all.
    divisor = np.empty(a.size)
    for e in np.unique(exponent):
        divisor[exponent == e] = 10 ** int(e)
    mantissa = a / divisor
    places = np.where(np.abs(exponent) < 10, 6,
                      np.where(np.abs(exponent) < 100, 5, 4))
    scaled = mantissa * np.power(10.0, places)
    n = np.rint(scaled)
    k = exponent - places
    slow = ((np.abs(scaled - np.floor(scaled) - 0.5) < 1e-6)   # near a tie
            | (mantissa < 1.0) | (n >= 10.0 * np.power(10.0, places))  # carry
            | (np.abs(k) > 22))
    value = np.empty(a.size)
    up, down = (~slow) & (k >= 0), (~slow) & (k < 0)
    for e in np.unique(k[up | down]):
        at = (k == e) & ~slow
        if e >= 0:
            value[at] = n[at] * _EXACT_POWERS[int(e)]
        else:
            value[at] = n[at] / _EXACT_POWERS[int(-e)]
    for i in np.flatnonzero(slow):
        value[i] = float(parse_number(format_endf_number(float(a[i]))))
    out[live] = np.copysign(value, v[live])
    return out



def format_endf_number_precise(value, width=11):
    """Choose the closest legal ENDF decimal field (ENDF-102 2023, 0.6.2).

    Fixed notation can retain more digits than normalized exponent notation.
    The legacy formatter remains the default for unchanged evaluations.
    """
    legacy = format_endf_number(value, width)
    if value is None or value == 0:
        return legacy
    candidates = [legacy]
    digits = max(1, int(math.floor(math.log10(abs(value)))) + 1)
    sign = int(value < 0)
    if digits + sign <= width:
        # Among fixed decimals the finest fitting quantum cannot round worse
        # than a coarser one. A rounding carry may require one fewer place.
        places = max(0, width-digits-sign-1)
        while places >= 0:
            fixed = f"{value:.{places}f}"
            if len(fixed) <= width:
                # The fewest places that still name the same double. The finest
                # quantum writes 20000000.1 as ``20000000.10``, eleven columns
                # with no blank before it, where JEFF-4.0 B-10 writes
                # `` 20000000.1``: the same value, and only the latter is the
                # source's text.
                while places > 0:
                    shorter = f"{value:.{places - 1}f}"
                    if parse_number(shorter) != parse_number(fixed):
                        break
                    fixed, places = shorter, places - 1
                candidates.append(fixed.rjust(width))
                break
            places -= 1
    return min(candidates, key=lambda field: abs(parse_number(field)-value))


# Format constants for ENDF data types
ENDF_FORMAT_PRECISE = 'float_precise'
ENDF_FORMAT_FLOAT = 'float'       # Scientific notation (e.g., " 1.234567+5")
ENDF_FORMAT_INT = 'int'           # Integer format (e.g., "         11")
ENDF_FORMAT_BLANK = 'blank'       # Blank field
ENDF_FORMAT_PRESERVE = 'preserve' # Use value's own type to determine format

#: Alias for :data:`ENDF_FORMAT_INT`, kept for the ~30 call sites that use it.
#:
#: It read "integer with zero rendered as 0 (not blank)", implying a contrast
#: with ENDF_FORMAT_INT. There has never been one: ``format_endf_data_line``
#: gave both constants the same branch, and neither has ever blanked a zero —
#: blanking is what ENDF_FORMAT_BLANK does. Making it an alias states that
#: outright, rather than leaving two names that look like a choice.
#:
#: Do not add the promised behaviour instead. Roughly forty call sites pass
#: ENDF_FORMAT_INT for fields whose zeros must be written as 0, and giving the
#: name real meaning would move every one of them.
ENDF_FORMAT_INT_ZERO = ENDF_FORMAT_INT


def format_endf_data_line(values: Sequence[Union[int, float, None]],
                         mat: int, mf: int, mt: int, line_num: int = 0,
                         formats: Optional[List[str]] = None) -> str:
    """
    Format a complete ENDF line with both data and identification parts.
    
    Args:
        values: Sequence of up to 6 numeric values for the data part
        mat: Material number
        mf: File number
        mt: Section number
        line_num: Line sequence number (optional)
        formats: Optional list of format types for each value (ENDF_FORMAT_*)
        
    Returns:
        Formatted 80-character ENDF line
    """
    # Format the data part (columns 1-66)
    parts = []

    # Apply formats if provided, otherwise use default formatting
    if formats:
        # Make sure formats list matches values length
        format_list = formats + [ENDF_FORMAT_PRESERVE] * (len(values) - len(formats))
        format_list = format_list[:len(values)]

        for value, fmt in zip(values, format_list):
            if fmt == ENDF_FORMAT_INT and value is not None:
                # ENDF_FORMAT_INT_ZERO is an alias for this, so one branch
                # serves both — as it always did, in two identical copies.
                parts.append(f"{int(value):11d}")
            elif fmt == ENDF_FORMAT_PRECISE:
                parts.append(format_endf_number_precise(value))
            elif fmt == ENDF_FORMAT_BLANK or value is None:
                parts.append("           ")
            else:
                parts.append(format_endf_number(value))
    else:
        for value in values[:6]:
            parts.append(format_endf_number(value))

    # Pad to 66 characters if needed
    data_part = ''.join(parts).ljust(66)

    return data_part + format_endf_id_columns(mat, mf, mt, line_num)


#: Highest sequence number the five-character NS field can hold. ENDF-6 counts
#: records from 1 within a section and wraps here rather than widening: the
#: record after 99999 is 1 again, not 100000.
MAX_SEQUENCE_NUMBER = 99999


def format_endf_id_columns(mat: int, mf: int, mt: int, line_num: int) -> str:
    """Columns 67-80: MAT, MF, MT and the wrapped sequence number.

    The wrap is not cosmetic. ``f"{100000:5d}"`` is six characters wide, so a
    section long enough to reach it used to be written 81 columns wide from that
    line on. Only two sections on this machine are long enough to show it —
    Ta-181's MF32 at 240 131 lines and Pu-239's at 190 445 — and both wrap
    99999 → 1, which is ``((n - 1) % 99999) + 1``.

    SEND, FEND, MEND and TEND pass their own fixed numbers through here
    unchanged: 99999 maps to itself and 0 stays 0.
    """
    if line_num > MAX_SEQUENCE_NUMBER:
        line_num = ((line_num - 1) % MAX_SEQUENCE_NUMBER) + 1
    return f"{mat:4d}{mf:2d}{mt:3d}{line_num:5d}"


_ENDF_NUMBER = re.compile(r'([-+]?\d*\.\d*)([+-]\d+)')


def parse_number(text: str) -> Union[float, int, None]:
    """
    Parse an ENDF-formatted number.
    
    ENDF uses a special format where numbers can be written in forms like:
    "1.234+5" meaning 1.234×10^5
    
    Args:
        text: The text representation of the number
        
    Returns:
        Parsed number as float or int, or None if parsing fails
    """
    text = text.strip()
    if not text:
        return None

    # Most ENDF fields omit E. Recognize that grammar before float() rather
    # than throwing a ValueError for almost every value in a LIST/TAB1 body.
    # Full matching keeps Python float syntax (including underscores, NaN,
    # infinity and explicit exponents) on its original path. One decimal
    # conversion, never mantissa * 10**exponent, preserves exact rounding.
    # A decimal point excludes integer controls; E/e excludes the explicit
    # exponent syntax used by the other fixed-width formats.
    if '.' in text and 'e' not in text and 'E' not in text:
        match = _ENDF_NUMBER.fullmatch(text)
        if match is not None:
            try:
                value = float(f"{match[1]}e{match[2]}")
            except ValueError:
                # The historical regex also matches an empty mantissa ".";
                # ".+3" has always returned None rather than raising.
                return None
            return int(value) if value.is_integer() else value
    
    try:
        # Try standard float parsing first
        value = float(text)
        # Return as int if it's a whole number
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        # Handle ENDF-specific format where "+" or "-" might be used instead of "E"
        # For example, "1.234+5" instead of "1.234E+5"
        match = _ENDF_NUMBER.search(text)
        if match:
            try:
                # Reassembled into one decimal string and converted once, NOT
                # ``mantissa * 10 ** exponent``. The multiplication rounds
                # twice -- once into the mantissa, once into the product --
                # and lands a unit in the last place away from the value the
                # digits name: ``2.427894 * 10**7`` is 24278940.000000004,
                # where ``float("2.427894e+7")`` is 24278940.0 exactly.
                # Python's own decimal-to-double conversion is correctly
                # rounded, so it gives the nearest double to what the field
                # says and nothing closer exists.
                #
                # A ulp is invisible in a printed cross section and is not
                # invisible to a fixed point: C-12's ENDF/B-VIII.1 MF3/MT5
                # writes its grid as ``24278940.0``, kika writes it back as
                # ``2.427894+7``, and the two used to decode to different
                # doubles -- which is
                # ``test_a_tape_with_mf6_comes_back_with_all_of_it[c12]``.
                value = float(f"{match.group(1)}e{match.group(2)}")
                if value.is_integer():
                    return int(value)
                return value
            except (ValueError, IndexError):
                pass
                
        # If all parsing fails
        return None


def parse_record_values(line: str) -> Tuple[Optional[Union[float, int]], ...]:
    """Six data fields of a record, retaining identification-field validation.

    LIST/TAB1 bodies need positional values, not a dict with C1..C6 keys.
    Short lines and invalid ID fields keep parse_line's existing behavior.
    """
    if len(line) >= 75:
        for field in (line[66:70], line[70:72], line[72:75]):
            if field.strip():
                int(field)
    if len(line) >= 80 and line[75:80].strip():
        int(line[75:80])
    if len(line) < 66:
        return (None,) * 6
    return (parse_number(line[:11]), parse_number(line[11:22]),
            parse_number(line[22:33]), parse_number(line[33:44]),
            parse_number(line[44:55]), parse_number(line[55:66]))


def parse_line(line: str) -> Dict[str, Any]:
    """
    Parse a standard ENDF record line into its components.
    
    Args:
        line: An 80-character ENDF line
        
    Returns:
        Dictionary with parsed components
    """
    result = {}
    
    # Parse data fields (columns 1-66)
    if len(line) >= 66:
        result = {"C1": parse_number(line[:11]), "C2": parse_number(line[11:22]),
                  "C3": parse_number(line[22:33]), "C4": parse_number(line[33:44]),
                  "C5": parse_number(line[44:55]), "C6": parse_number(line[55:66])}
    
    # Parse identification fields (columns 67-80)
    if len(line) >= 75:
        result["MAT"] = int(line[66:70]) if line[66:70].strip() else None
        result["MF"] = int(line[70:72]) if line[70:72].strip() else None
        result["MT"] = int(line[72:75]) if line[72:75].strip() else None
        
    if len(line) >= 80:
        result["SEQ"] = int(line[75:80]) if line[75:80].strip() else None
    
    return result


def parse_endf_id(line: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Parse the identification fields from an ENDF line.
    
    ENDF format specifies:
    - Columns 67-70 (0-indexed: 66-69): MAT number
    - Columns 71-72 (0-indexed: 70-71): MF number
    - Columns 73-75 (0-indexed: 72-74): MT number
    
    Args:
        line: A line from an ENDF file
        
    Returns:
        Tuple of (MAT, MF, MT) numbers
    """
    if len(line) < 75:
        return None, None, None
    
    try:
        # ENDF format has specific columns for MAT, MF, MT
        mat_str = line[66:70].strip()
        mf_str = line[70:72].strip()
        mt_str = line[72:75].strip()
        
        # Convert to integers, handling empty strings
        mat = int(mat_str) if mat_str else None
        mf = int(mf_str) if mf_str else None
        mt = int(mt_str) if mt_str else None
        
        return mat, mf, mt
    except ValueError as e:
        # This might happen if the fields contain non-numeric data
        return None, None, None
