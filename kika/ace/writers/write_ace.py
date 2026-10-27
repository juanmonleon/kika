import os
import io
import numpy as np
from kika.ace.classes.ace import Ace
from kika.ace.writers.write_header import write_header
from kika.ace.classes.xss import unpack_int_words

def write_ace(ace: Ace, filepath: str = None, overwrite: bool = False) -> str:
    # Verify header and XSS data exist
    if ace.header is None:
        raise ValueError("Ace object must have a header")
    if ace.xss_data is None or len(ace.xss_data) == 0:
        raise ValueError("Ace object must have XSS data")
    
    # Determine file path if not provided
    if filepath is None:
        if ace.filename is not None:
            base_path, extension = os.path.splitext(ace.filename)
            filepath = f"{base_path}_recon{extension}"
            print(f"Using modified original filename: {filepath}")
        else:
            zaid = getattr(ace.header, 'zaid', None) or "unknown"
            temp = getattr(ace.header, 'temperature', None) or "0K"
            filepath = f"{zaid}_{int(temp)}K.ace"
            print(f"No filepath provided. Writing to {filepath} in the current directory.")
    
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File {filepath} already exists and overwrite is False")
    
    # xss_data[0] is the FORTRAN 1-based placeholder, not part of the table.
    values = np.asarray(ace.xss_data, dtype=float)
    if values.ndim != 1:
        raise ValueError("XSS data must be a 1-D array")
    if values.size > 1:
        values = values[1:]

    is_int = unpack_int_words(getattr(ace, "xss_int_words", None), values.size)
    real = getattr(ace, "xss_real_format", None)
    if real is not None and is_int is None:
        # A table type that spells every word as a real (EPRDATA: %20.12E)
        is_int = np.zeros(values.size, dtype=bool)
    if is_int is not None:
        # The spelling of each word is known (read from the file, or set by an
        # encoder): integers as I20, reals as E20.11, and the last line as long
        # as its fields, as NJOY writes it.
        try:
            # newline='\n': ACE is LF on every platform, as NJOY writes it; text
            # mode on Windows would turn every line end into CRLF.
            with open(filepath, 'w', newline='\n', buffering=1024*1024) as f:
                f.write(write_header(ace.header) + "\n")
                for text in format_xss_lines(values, is_int, real=real or _REAL):
                    f.write(text)
            return f"Success! ACE file written to: {filepath}"
        except Exception as e:
            raise IOError(f"Error writing ACE file: {e}")

    # Without a mask: every word as a real, unless all are integers
    if np.all(np.abs(values - np.rint(values)) < 1e-12):
        values = values.astype(np.int64)
        fmt = "%20d"
    else:
        fmt = "%20.11E"

    try:
        # Open file with a large buffer and write header and data in chunks
        with open(filepath, 'w', newline='\n', buffering=1024*1024) as f:
            header_str = write_header(ace.header)
            f.write(header_str + "\n")
            
            n = values.shape[0]
            n_full_rows = n // 4
            remainder = n % 4
            
            # Reshape and write full rows (each with 4 values) using NumPy's fast formatting
            if n_full_rows > 0:
                full_rows = values[:n_full_rows*4].reshape(-1, 4)
                row_fmt = fmt * 4  # Concatenate the format string for 4 columns
                np.savetxt(f, full_rows, fmt=row_fmt, newline='\n')
            
            # Write remaining values (if any) formatted and padded to 80 characters
            if remainder:
                rem_line = ''.join(fmt % val for val in values[n_full_rows*4:])
                f.write(rem_line.ljust(80) + "\n")
            
            return f"Success! ACE file written to: {filepath}"
    except Exception as e:
        raise IOError(f"Error writing ACE file: {e}")

_INT, _REAL = "%20d", "%20.11E"
#: Line format by the integer pattern of its four words (bit 3 = first word)
_ROW_FORMATS = ["".join(_INT if (code >> (3 - k)) & 1 else _REAL for k in range(4)) for code in range(16)]
_CHUNK_ROWS = 65536


def format_xss_lines(values: np.ndarray, is_int: np.ndarray, first_row: int = 0, last_row: int = None,
                     real: str = None):
    """XSS text with each word in its own spelling, in chunks of lines ending in newline.

    ``values`` is XSS(1..n) and ``is_int`` the matching bool mask. Rows
    ``first_row..last_row`` (0-based, inclusive; default all) are formatted;
    integer words are rounded first, so a count stays a count. The last line
    of the table holds only its remaining fields, unpadded.
    """
    real = real or _REAL
    rows = _ROW_FORMATS if real == _REAL else         ["".join(_INT if (code >> (3 - k)) & 1 else real for k in range(4)) for code in range(16)]
    n = values.size
    n_rows = -(-n // 4)
    last_row = n_rows - 1 if last_row is None else last_row
    n_full = n // 4
    for a in range(first_row, last_row + 1, _CHUNK_ROWS):
        b = min(a + _CHUNK_ROWS, last_row + 1)
        full_b = min(b, n_full)
        out = []
        if a < full_b:
            v = values[4 * a:4 * full_b].copy()
            m = is_int[4 * a:4 * full_b]
            v[m] = np.rint(v[m])
            codes = (m.reshape(-1, 4) * np.array([8, 4, 2, 1])).sum(axis=1).tolist()
            out = [rows[c] % tuple(row) for c, row in zip(codes, v.reshape(-1, 4).tolist())]
        if b > n_full:  # the partial last line
            tail = range(4 * n_full, n)
            out.append("".join(_INT % round(values[k]) if is_int[k] else real % values[k] for k in tail))
        yield "\n".join(out) + "\n"


def write_ace_ascii(ace: Ace, filepath: str = None, overwrite: bool = False) -> str:
    """
    Write an Ace object to a formatted ASCII file.
    This is a convenience wrapper around write_ace.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to write
    filepath : str, optional
        The file path to write to. If None, will use the original filename 
        from which the ACE data was read, or create a name based on ZAID and temperature.
    overwrite : bool, optional
        Whether to overwrite an existing file, defaults to False
        
    Returns
    -------
    str
        A success message including the path to the written file
    """
    return write_ace(ace, filepath, overwrite)


def write_ace_checked(ace: Ace, filepath) -> None:
    """Write ``ace`` at ``filepath`` through a temporary file, read back in strict
    mode and compared word for word before it replaces ``filepath``: NXS, JXS
    and the integer words equal, the reals to the 12 digits E20.11 keeps.
    """
    from pathlib import Path

    from kika.ace.parsers.parse_ace import read_ace

    target = Path(filepath)
    temporary = target.with_name(f".{target.name}.tmp")
    write_ace(ace, str(temporary), overwrite=True)
    try:
        again = read_ace(str(temporary), strict=True)
        a = np.asarray(ace.xss_data, dtype=float)
        b = np.asarray(again.xss_data, dtype=float)
        same = (list(again.header.nxs_array) == list(ace.header.nxs_array)
                and list(again.header.jxs_array) == list(ace.header.jxs_array) and a.shape == b.shape)
        if same:
            ints = unpack_int_words(getattr(ace, "xss_int_words", None), a.size - 1)
            ints = np.zeros(a.size - 1, dtype=bool) if ints is None else np.asarray(ints, dtype=bool)
            x, y = a[1:], b[1:]
            # Half a unit in E20.11's twelfth figure is 5e-12 of a mantissa of 1, and
            # a value that rounds exactly half (B-VIII.1 Tc-98's 1.000000107585e-20
            # -> 1.00000010759e-20) lands on it, 4.5e-17 past it in binary (x and y each
            # carry ~1e-16 of their own): four materials refused to write over the one word.
            same = (np.array_equal(x[ints], y[ints])
                    and bool(np.all(np.abs(x[~ints] - y[~ints]) <= 5.001e-12 * np.abs(x[~ints]))))
        if not same:
            raise ValueError(f"{target.name} does not read back as it was written")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, target)
