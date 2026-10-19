import logging
from kika.ace.classes.ace import Ace
import numpy as np
from kika.ace.classes.header import Header
from kika.ace.classes.xss import integer_fields, pack_int_words
from kika.ace.classes.read_report import AceReadError, ReadReport, lost
from kika.ace.parsers.parse_esz import read_esz_block
from kika.ace.parsers import read_header, read_nubar_data
from kika.ace.parsers.parse_delayed import read_delayed_neutron_data
from kika.ace.parsers.parse_mtr import read_mtr_blocks
from kika.ace.parsers.parse_lqr import read_lqr_block
from kika.ace.parsers.parse_tyr import read_tyr_blocks
from kika.ace.parsers.parse_xs_locators import read_xs_locator_blocks
from kika.ace.parsers.parse_xs_data import read_xs_data_block
from kika.ace.parsers.parse_angular_locators import read_angular_locator_blocks
from kika.ace.parsers.parse_angular_distribution import read_angular_distribution_blocks
from kika.ace.parsers.parse_energy_distribution_locators import read_energy_locator_blocks
from kika.ace.parsers.parse_energy_distributions import read_energy_distribution_blocks
from kika.ace.classes.energy_distribution.container import EnergyDistributionContainer
from kika.ace.parsers.parse_gpd import read_gpd_block
from kika.ace.parsers.parse_photon_production_xs import read_production_xs_blocks
from kika.ace.parsers.parse_yield_multipliers import read_yield_multiplier_blocks
from kika.ace.parsers.parse_fission_xs import read_fission_xs_block
from kika.ace.parsers.parse_unresolved_resonance import read_unresolved_resonance_block
from kika.ace.parsers.parse_secondary_particle_types import parse_ptype_block
from kika.ace.parsers.parse_secondary_reaction_counts import parse_ntro_block
from kika.ace.parsers.parse_secondary_data_locators import parse_ixs_block
from kika.ace.parsers.parse_secondary_cross_sections import parse_hpd_block
from kika.ace.parsers.parse_photoatomic import is_photoatomic_extension, parse_photoatomic
from kika.ace.parsers.parse_dosimetry import is_dosimetry_extension, parse_dosimetry

# Setup logger
logger = logging.getLogger(__name__)

def read_ace(filename, debug=False, strict=True):
    """
    Read and parse an ACE format file.
    
    Every block is parsed when the file is read; nothing is loaded on demand.
    Once the XSS is an array, parsing all the blocks takes 5-25 % of the read
    (0.03 s of 0.19 s for U-238 B-VIII.0, 0.18 s of 0.73 s for Fe-56 Lib81)
    and the rest is turning the XSS text into numbers, so deferring blocks
    would buy little.
    
    Parameters
    ----------
    filename : str
        Path to the ACE file
    debug : bool, optional
        Whether to print debug information, defaults to False
    strict : bool, optional
        ``True`` (the default): a block or reaction that cannot be read raises
        :class:`~kika.ace.classes.read_report.AceReadError`. ``False``: the
        table keeps what could be read and ``ace.read_report`` lists what was
        not. Notes (legitimate things worth knowing) are recorded either way.
        
    Returns
    -------
    ace : Ace, PhotoatomicAce or DosimetryAce
        An Ace object containing the parsed data, or, for a photo-atomic
        table (ZAID suffix ending in ``p``), a
        :class:`~kika.ace.classes.photoatomic.PhotoatomicAce`, and for a
        dosimetry table (suffix ending in ``y``), a
        :class:`~kika.ace.classes.dosimetry.DosimetryAce`.
    """

    if debug:
        logger.debug(f"Reading ACE file: {filename}")
        
    ace = Ace()
    ace.filename = filename
    ace.header = Header()
    ace._debug = debug  # Store debug flag for later use by parsers
    ace.read_report = ReadReport(strict=strict)
    
    # The header is read line by line and the XSS straight from the file in
    # chunks: a list of lines would hold ~1.6x the file in str objects just to
    # be split again.
    with open(filename, 'rb') as file:
        lines = _read_header_lines(file)

        # Determine if it's a legacy or 2.0.1 header and read it
        if lines and "2.0.1" in lines[0][:10]:
            ace.header.format_version = "2.0.1"
        else:
            ace.header.format_version = "legacy"

        if debug:
            logger.debug(f"ACE format version: {ace.header.format_version}")

        # Read the entire header (opening and arrays)
        read_header(ace.header, lines, debug=debug)

        # Read the XSS array - essential data needed for all parsers
        ace.xss_data, ace.xss_int_words = read_xss_file(file, ace.header.nxs_array[1], spelling=True)

    # A photo-atomic table ('p') shares only the header with a neutron one:
    # its NXS and JXS mean other things, so the neutron block parsers below
    # would read garbage from it. It is decoded on its own and returned as a
    # PhotoatomicAce.
    if is_photoatomic_extension(ace.header.extension):
        return parse_photoatomic(ace.header, ace.xss_data, filename)
    # A dosimetry table ('y') is reaction cross sections only, each with its
    # own grid and laws.
    if is_dosimetry_extension(ace.header.extension):
        return parse_dosimetry(ace.header, ace.xss_data, filename, ace.xss_int_words)

    parse_blocks(ace, debug)
    return ace

def parse_blocks(ace, debug=False):
    """Decode every block of ``ace`` from its header and ``xss_data``.

    What :func:`read_ace` runs once the XSS is read; also what turns an XSS
    built by :mod:`kika.ace.encoders` back into blocks. ``ace.read_report``
    must exist (strict or not).
    """
    # Parse every block. A block that cannot be read raises AceReadError
    # naming it (strict) or is left empty and recorded in ace.read_report.
    def block(name, parser, *args):
        try:
            return parser(ace, *args, debug)
        except AceReadError:
            raise
        except (ValueError, IndexError, TypeError, KeyError, AttributeError) as e:
            lost(ace, name, "", f"block not read: {e}")
            return None

    ace.esz_block = block("ESZ", read_esz_block)
    ace.nubar = block("NU", read_nubar_data)
    ace.delayed_neutron_data = block("BDD", read_delayed_neutron_data)
    ace.reaction_mt_data = block("MTR", read_mtr_blocks)
    ace.q_values = block("LQR", read_lqr_block)
    ace.particle_release = block("TYR", read_tyr_blocks)
    ace.xs_locators = block("LSIG", read_xs_locator_blocks)
    ace.xs_data = block("SIG", read_xs_data_block)
    ace.angular_locators = block("LAND", read_angular_locator_blocks)
    ace.angular_distributions = block("AND", read_angular_distribution_blocks)
    ace.energy_distribution_locators = block("LDLW", read_energy_locator_blocks)
    ace.energy_distributions = block("DLW", read_energy_distribution_blocks)
    ace.photon_production_data = block("GPD", read_gpd_block)
    # PTYPE first: the other secondary-particle blocks depend on it
    ace.secondary_particle_types = block("PTYPE", parse_ptype_block)
    photon_xs, particle_xs = block("SIGP", read_production_xs_blocks) or (None, None)
    ace.photon_production_xs = photon_xs
    ace.particle_production_xs = particle_xs
    ace.fission_xs = block("FIS", read_fission_xs_block)
    ace.unresolved_resonance = block("UNR", read_unresolved_resonance_block)
    ace.secondary_particle_reactions = block("NTRO", parse_ntro_block)
    ace.secondary_particle_data_locations = block("IXS", parse_ixs_block)
    photon_ym, particle_ym = block("YP", read_yield_multiplier_blocks) or (None, None)
    ace.photon_yield_multipliers = photon_ym
    ace.particle_yield_multipliers = particle_ym
    ace.secondary_particle_cross_sections = block("HPD", parse_hpd_block)

    return ace


#: Lines of XSS parsed per chunk (4 values each).
_XSS_CHUNK_LINES = 65536

#: Header lines after the opening: IZAW (4), NXS (2) and JXS (4).
_HEADER_ARRAY_LINES = 10


def _read_header_lines(file):
    """
    Read the header lines of a table from a file opened in binary mode.

    The header is the two-line opening, the NC comment lines of a 2.0.1 header
    (none in a legacy one) and the IZAW, NXS and JXS arrays. Lines are decoded
    as ASCII, with any other byte replaced, and end in ``\\n``.
    """
    def line():
        return file.readline().rstrip(b"\r\n").decode("ascii", "replace") + "\n"

    lines = [line(), line()]
    n_comments = 0
    if "2.0.1" in lines[0][:10]:
        try:
            n_comments = int(lines[1].split()[3])
        except (IndexError, ValueError):
            pass  # read_header reports the malformed opening
    lines += [line() for _ in range(n_comments + _HEADER_ARRAY_LINES)]
    return lines


def read_xss_file(file, n_values, spelling=False):
    """
    Read the XSS array from a binary file positioned just after the header.

    ACE writes the XSS as 4E20.11 (integers as I20): four 20-character fields
    per 80-character line. When every line has that shape the fields are read
    through a structured dtype, a chunk of lines at a time, so the text is
    never held whole and no token strings are made: the peak is the array plus
    one chunk. Anything else (another line length, a blank or non-numeric
    field) is read again from the start by :func:`read_xss`, line by line.

    Parameters
    ----------
    file : binary file object
        Seekable, positioned at the first XSS line.
    n_values : int
        ``NXS(1)``, the declared length of the XSS array.

    spelling : bool, optional
        Also return which words are spelled as integers, packed as
        ``Ace.xss_int_words`` (see :mod:`kika.ace.classes.xss`).

    Returns
    -------
    numpy.ndarray
        ``float64`` array of length ``n_values + 1``, ``[0]`` = 0.0; with
        ``spelling=True``, ``(xss, int_words)``.
    """
    start = file.tell()
    found = _read_xss_fixed_lines(file, n_values) if n_values else None
    if found is None:
        file.seek(start)
        lines = file.read().decode("ascii", "replace").splitlines(keepends=True)
        xss = read_xss(lines, n_values)
        is_int = _spelling_from_lines(lines, xss.size - 1)
    else:
        xss, is_int = found
    if not spelling:
        return xss
    return xss, (pack_int_words(is_int) if is_int is not None else None)


def _spelling_from_lines(lines, n_values):
    """Integer spelling of each XSS token (no decimal point); None if the tokens do not count n_values."""
    if n_values:
        lines = lines[:-(-n_values // 4)]
    is_int = np.fromiter(("." not in t for t in " ".join(lines).split()), dtype=bool)
    return is_int if is_int.size == n_values else None


#: Largest single read: the share's 9p mount has failed one-off reads of ~5 MB
#: with ENOMEM (Th-232, Se-80 and W-181 in the 2026-10-14 census) that succeed
#: when repeated or smaller.
_READ_BYTES = 1 << 20


def _read_exact(file, n):
    """``n`` bytes (fewer only at the end of the file), read at most ``_READ_BYTES`` at a time."""
    parts = []
    while n > 0:
        part = file.read(min(n, _READ_BYTES))
        if not part:
            break
        parts.append(part)
        n -= len(part)
    return b"".join(parts)


def _read_xss_fixed_lines(file, n_values):
    """The fast path of :func:`read_xss_file`; ``None`` if it does not apply."""
    first = file.readline()
    width = len(first)  # 80 columns + "\n" or "\r\n"
    if width not in (81, 82) or first[80:] != b"\r\n"[82 - width:]:
        return None
    eol = first[80:]
    line_dtype = np.dtype({
        "names": ["f0", "f1", "f2", "f3", "eol"],
        "formats": ["S20", "S20", "S20", "S20", f"S{width - 80}"],
        "offsets": [0, 20, 40, 60, 80],
        "itemsize": width,
    })
    n_full, n_last = divmod(n_values, 4)

    xss = np.empty(n_values + 1, dtype=np.float64)
    xss[0] = 0.0
    is_int = np.empty(n_values, dtype=bool)
    pending = first
    done = 0  # full lines read so far
    try:
        while done < n_full:
            take = min(_XSS_CHUNK_LINES, n_full - done)
            chunk = pending + _read_exact(file, take * width - len(pending))
            pending = b""
            if len(chunk) < take * width:
                return None
            rows = np.frombuffer(chunk, dtype=line_dtype, count=take)
            if not (rows["eol"] == eol).all():
                return None
            values = xss[1 + 4 * done:1 + 4 * (done + take)]
            for k in range(4):
                values[k::4] = rows[f"f{k}"].astype(np.float64)
            fields = np.frombuffer(chunk, dtype=np.uint8).reshape(take, width)[:, :80].reshape(take, 4, 20)
            is_int[4 * done:4 * (done + take)] = integer_fields(fields).ravel()
            done += take
        if n_last:
            # n_full > 0 here: a table shorter than one line fails the width check.
            last = _read_exact(file, n_last * 20)
            if len(last) < n_last * 20:
                return None
            xss[1 + 4 * n_full:] = np.frombuffer(
                last, dtype="S20", count=n_last
            ).astype(np.float64)
            is_int[4 * n_full:] = integer_fields(np.frombuffer(last, dtype=np.uint8).reshape(n_last, 20))
    except ValueError:  # a blank or non-numeric field
        return None
    return xss, is_int


def read_xss(lines, n_values=None):
    """
    Read the XSS array of an ACE table into a float64 numpy array.

    The array uses 1-based indexing to match the FORTRAN ``XSS(i)`` of the ACE
    manual: index 0 holds a 0.0 placeholder and ``xss[i]`` is ``XSS(i)``.
    Blocks parsed from it are views of this array (see
    :mod:`kika.ace.classes.xss`).

    Parameters
    ----------
    lines : list of str
        The lines of the file after the header.
    n_values : int, optional
        ``NXS(1)``, the declared length of the XSS array. When given, only the
        lines that hold it are read, and the count is checked.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of length ``n + 1``.
    """
    if n_values:
        lines = lines[:-(-n_values // 4)]  # 4 values per line, last one partial

    # Whitespace split is ~4x faster than slicing 4E20 fields and agrees with it
    # whenever adjacent fields are separated by a blank, which every E20.11 /
    # I20 value is. A glued pair, or a token float() rejects, changes the count,
    # and that falls back to the fixed-width reading.
    # Parsed in chunks so the transient token strings stay bounded (a single
    # split of a 5.8 M-value table holds ~350 MB of str objects at once).
    try:
        values = np.concatenate([np.empty(0)] + [
            np.array(" ".join(lines[i:i + _XSS_CHUNK_LINES]).split(), dtype=np.float64)
            for i in range(0, len(lines), _XSS_CHUNK_LINES)
        ])
    except ValueError:
        values = None
    if values is None or (n_values and values.size != n_values):
        values = _read_xss_fixed_width(lines)
    if n_values and values.size != n_values:
        raise ValueError(
            f"XSS array has {values.size} values, NXS(1) declares {n_values}"
        )

    xss = np.empty(values.size + 1, dtype=np.float64)
    xss[0] = 0.0
    xss[1:] = values
    return xss


def _read_xss_fixed_width(lines):
    """4E20 field-by-field reading; skips blank and non-numeric fields."""
    values = []
    for line in lines:
        for i in range(4):
            field = line[i * 20:(i + 1) * 20].strip()
            if field:
                try:
                    values.append(float(field))
                except ValueError:
                    pass
    return np.array(values, dtype=np.float64)
