"""Run NJOY on an ENDF tape with a fixed deck and keep its ACE file and logs.

The deck is the one of :func:`run_njoy` and :func:`run_njoy_with_pendf`. Each run is
checked for the decimal-locale misread of NJOY on Windows (see
:class:`~kika.njoy.locale_guard.NjoyDecimalLocaleError`).
"""
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Optional
from .launcher import build_njoy_command
from .locale_guard import (
    LISTING_FILENAME,
    NjoyDecimalLocaleError,
    check_run,
    locale_error_message,
    read_listing,
)
from .templates import NJOY_INPUT_TEMPLATE, NJOY_INPUT_TEMPLATE_WITH_PENDF
from kika.endf import open_endf
from kika._constants import NDLIBRARY_TO_SUFFIX
from kika._utils import MeV_to_kelvin, ace_temperature_suffix

def _resolve_nd_suffix(library_name: str) -> str:
    """Return the ACE suffix for a library name, falling back to '00' if unknown."""
    import warnings
    library_key = library_name.lower().replace('-', '').replace('/', '').replace('.', '')
    if library_key in NDLIBRARY_TO_SUFFIX:
        return NDLIBRARY_TO_SUFFIX[library_key]
    warnings.warn(
        f"library_name={library_name!r} not in NDLIBRARY_TO_SUFFIX "
        f"({sorted(NDLIBRARY_TO_SUFFIX.keys())}); using fallback suffix '00'.",
        stacklevel=2,
    )
    return "00"


def _get_temperature_info(temperature: float) -> tuple[float, Optional[str]]:
    """
    Helper function to get temperature in Kelvin and corresponding suffix.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin or MeV (if < 1.0, assumed to be MeV)

    Returns
    -------
    tuple[float, str | None]
        (temperature_in_kelvin, temperature_suffix). The suffix is ``None``
        for a temperature not in ``K_TO_SUFFIX``: a caller that needs it
        asks :func:`kika._utils.ace_temperature_suffix`, which raises, and a
        caller given ``extension=`` does not need it.
    """
    # Convert temperature to Kelvin if needed
    if temperature < 1.0:
        temp_K = MeV_to_kelvin(temperature)
    else:
        temp_K = temperature

    try:
        return temp_K, ace_temperature_suffix(temp_K)
    except ValueError:
        return temp_K, None


def _suff_from_extension(extension: str) -> str:
    """Convert a user-supplied ACE extension (e.g. '06c') to the NJOY ACER
    `{suff}` slot value (e.g. '.06'). Trailing 'c' is stripped; anything
    else is taken verbatim after the leading dot."""
    ext = str(extension).strip()
    if ext.endswith("c"):
        return "." + ext[:-1]
    return "." + ext

def _generate_unique_filename(base_filename: str, ace_dir: Path, njoy_files_dir: Path, 
                             temp_suffix: str, additional_suffix: str = None) -> tuple[str, str]:
    """
    Generate a unique filename by checking for existing files and adding a suffix if needed.
    
    Parameters
    ----------
    base_filename : str
        Base filename without extension (e.g., "260560_81.02")
    ace_dir : Path
        Directory where ACE files are stored
    njoy_files_dir : Path
        Directory where NJOY files are stored
    temp_suffix : str
        Temperature suffix (e.g., ".02")
    additional_suffix : str | None
        Optional additional suffix provided by user
        
    Returns
    -------
    tuple[str, str]
        (final_base_filename, final_ace_filename)
    """
    # Start with the original base filename
    zaid_lib_part = base_filename.replace(temp_suffix, "")  # e.g., "260560_81"
    
    if additional_suffix:
        # User provided a specific suffix
        final_base = f"{zaid_lib_part}_{additional_suffix}{temp_suffix}"
        final_ace = f"{final_base}c"
    else:
        # Check if files already exist and generate unique suffix
        final_base = base_filename
        final_ace = f"{final_base}c"
        
        # Check if files exist
        ace_file_path = ace_dir / final_ace
        input_file_path = njoy_files_dir / f"{final_base}.input"
        
        if ace_file_path.exists() or input_file_path.exists():
            # Find the next available suffix
            counter = 1
            while True:
                auto_suffix = f"v{counter:02d}"
                final_base = f"{zaid_lib_part}_{auto_suffix}{temp_suffix}"
                final_ace = f"{final_base}c"
                
                ace_file_path = ace_dir / final_ace
                input_file_path = njoy_files_dir / f"{final_base}.input"
                
                if not ace_file_path.exists() and not input_file_path.exists():
                    break
                counter += 1
    
    return final_base, final_ace

def _render_title(isotope_label: str, T: float, library_name: str, njoy_version: str = "NJOY 2016.78") -> str:
    # Example: "FE56 - 293.6 K - JEFF-40 (NJOY 2016.78)"
    return f"{isotope_label} - {T:.1f} K - {library_name.upper()} ({njoy_version})"

def _render_njoy_input(mat: int, T: float, title: str, suff: str = None,
                       template: str = NJOY_INPUT_TEMPLATE, acer_iprint: int = 0) -> str:
    if suff is None:
        # Use helper function to get temperature suffix (without 'c')
        # A temperature with no suffix raises here (pass extension=), instead
        # of ".00", which would name the table by a temperature it is not at.
        temp_K, _ = _get_temperature_info(T)
        suff = ace_temperature_suffix(temp_K)
    # Fill the template with the run-specific parameters
    deck = dedent(template).format(mat=mat, T=T, title=title, suff=suff)
    if int(acer_iprint) != 0:
        # The check pass's card is "iopt iprint itype suff": only iprint moves.
        check_card = "\n 7 0 1 -1 /\n"
        if deck.count(check_card) != 1:
            raise ValueError("the deck's second ACER card is not the expected one")
        deck = deck.replace(check_card, f"\n 7 {int(acer_iprint)} 1 -1 /\n")
    return deck

# ---- 2) Runner ----
#: Re-runs when NJOY misread the deck; the same budget as RECONR's
#: (``kika.processing.njoy_reconstruct._MAX_LOCALE_ATTEMPTS``).
_MAX_LOCALE_ATTEMPTS = 8


def _run_njoy_deck(njoy_exe, workdir: Path, deck: str):
    """Run *deck* in *workdir*, again while NJOY's listing shows a misread.

    The decimal-locale misread is per run (:mod:`kika.njoy.locale_guard`), so a
    second run of the same deck on the same tapes is what fixes it; the input
    tapes are untouched by NJOY and the outputs are rewritten. What a run that
    still misreads after the last attempt left behind is checked again by
    :func:`_guard_decimal_locale`, which raises.
    """
    from .locale_guard import check_run, read_listing

    for attempt in range(1, _MAX_LOCALE_ATTEMPTS + 1):
        result = _watchedRun(njoy_exe, workdir, deck)
        check = check_run(deck, read_listing(workdir))
        crashed = _crashed_on_a_misread(result)
        if not check.corrupted and not crashed:
            return result
        if attempt < _MAX_LOCALE_ATTEMPTS:
            logging.getLogger(__name__).warning(
                "NJOY misread its deck (%s); re-running (%d/%d)",
                check.detail if check.corrupted else crashed, attempt + 1,
                _MAX_LOCALE_ATTEMPTS)
    return result


#: How often a running NJOY's listing is read for a misread echo, in seconds.
_WATCH_INTERVAL = 0.5


def _watchedRun(njoy_exe, workdir: Path, deck: str) -> subprocess.CompletedProcess:
    """One NJOY run of *deck*, stopped as soon as its listing shows a misread.

    A misread tolerance can be ``err = 0``, and RECONR may never finish on it:
    ENDF/B-VIII.1 Pu-239 ran 47 min of CPU without writing a tape, where the
    same deck read correctly takes 3.5 min. The echo is in the listing within a
    second of the start, so it is read while NJOY runs -- only up to its last
    complete line, since a line half written could read ``0.0`` for ``0.001``
    -- and the run is killed on the first misread. The listing stays behind
    for :func:`_run_njoy_deck`'s check, which then runs the deck again.
    """
    from .locale_guard import check_run, read_listing

    process = subprocess.Popen(build_njoy_command(njoy_exe), cwd=workdir,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    data = deck.encode("utf-8")
    while True:
        try:
            out, _ = process.communicate(input=data, timeout=_WATCH_INTERVAL)
            break
        except subprocess.TimeoutExpired:
            data = None
            listing = read_listing(workdir) or ""
            if check_run(deck, listing[:listing.rfind("\n") + 1]).corrupted:
                process.kill()
                out, _ = process.communicate()
                break
    return subprocess.CompletedProcess(process.args, process.returncode, out, None)


#: What gfortran prints when a formatted read meets a number it cannot parse --
#: the same decimal-locale misread, caught by the runtime instead of by the
#: listing check, in a module after the one :func:`check_run` reads (an ACER
#: card, for instance).
_MISREAD_CRASHES = ("Error during floating point read", "Bad real number")


def _crashed_on_a_misread(result) -> str | None:
    """The gfortran message when NJOY stopped on an unparsable number, else None."""
    if result.returncode == 0:
        return None
    text = (result.stdout or b"").decode("utf-8", "replace")
    return next((message for message in _MISREAD_CRASHES if message in text), None)


def _guard_decimal_locale(workdir: Path, deck: str, njoy_files_dir: Path,
                          base_filename: str) -> str | None:
    """Fail the run if NJOY read numbers other than the ones in the deck.

    An ACE file built from a misread deck looks perfectly normal and is
    wrong, which is the one outcome worth an exception.  Runs whose deck has
    no ``reconr`` card, or whose listing we cannot read, pass through
    unchecked -- see :mod:`kika.njoy.locale_guard`.

    Also keeps NJOY's own listing, which carries the error context that its
    stdout does not.
    """
    listing = read_listing(workdir)
    listing_file = None
    if listing is not None:
        listing_file = njoy_files_dir / f"{base_filename}.{LISTING_FILENAME}"
        listing_file.write_text(listing, encoding="utf-8")
    check = check_run(deck, listing)
    if check.corrupted:
        raise NjoyDecimalLocaleError(
            locale_error_message(check.detail, attempts=1), detail=check.detail
        )
    return str(listing_file) if listing_file else None


def run_njoy(
    njoy_exe: str,
    endf_path: str | Path,
    temperature: float,
    library_name: str,
    output_dir: str | Path,
    *,
    suff: str = None,
    njoy_version: str = "NJOY 2016.78",
    additional_suffix: str = None,
    extension: str | None = None,
    ace_dir: str | Path | None = None,
    xsdir_dir: str | Path | None = None,
    njoy_files_dir: str | Path | None = None,
    acer_iprint: int = 0,
) -> dict:
    """Run NJOY on one ENDF tape at one temperature and keep the ACE file it writes.

    kika copies *endf_path* to ``tape20`` of a temporary directory under
    *output_dir*, writes the deck there, runs NJOY and moves the results out. The
    deck runs these modules, in this order:

    1. ``moder``: tape 20 to a binary tape.
    2. ``reconr``: ``err`` 0.001, ``tempr`` 0 K, ``errmax`` 0.01 and ``errint``
       5e-8 (NJOY's own field names; the tolerances are fractions).
    3. ``broadr`` at *temperature*: ``errthn`` 0.001, ``thnmax`` 1e6 eV, and the
       same ``errmax`` 0.01 and ``errint`` 5e-8.
    4. ``heatr`` for MT 302, 318, 402, 442 and 444, then a second ``heatr`` pass on
       the broadened tape with print option 2, whose output tape no later module
       reads.
    5. ``purr`` at *temperature*, with 20 bins, 64 ladders and nine sigma-zero
       values from 1e10 b to 10 b.
    6. ``gaspr``.
    7. ``acer`` for fast data, which writes the ACE file with the suffix *suff*.
    8. A second ``acer`` pass (option 7) that reads that ACE file and checks it. Its
       output is the final ACE file and the xsdir file.

    Temperatures are in K and energies in eV, as NJOY reads them.

    **Decimal-locale guard.** NJOY on a Windows machine whose decimal separator is
    not ``.`` can read its own deck wrongly (see
    :class:`~kika.njoy.locale_guard.NjoyDecimalLocaleError`). After a run kika reads
    NJOY's listing and compares the three tolerances that ``reconr`` echoes there
    with the ones in the deck. A deck with no ``reconr`` card is checked against
    the echo of ``broadr`` instead, which is the case of
    :func:`run_njoy_with_pendf`. The check passes without a verdict when the
    listing is missing, has none of the echoes, or the deck's tolerances are not
    positive. While NJOY runs the listing is also read every 0.5 s, and the run is
    killed at the first mismatch. A mismatch, or a non-zero exit with
    ``Error during floating point read`` or ``Bad real number`` in NJOY's output,
    makes kika run the same deck again on the same tapes, up to 8 runs in all. If
    the listing of the last run still shows a mismatch, the function raises
    ``NjoyDecimalLocaleError``. By then the deck, the output and the listing are
    already saved, and the ACE and xsdir files are not moved. A run that ends in
    one of the two read errors on the eighth attempt does not raise: it returns
    with a non-zero ``returncode``.

    Parameters
    ----------
    njoy_exe
        Path of the NJOY executable. On Windows a UNC path into a WSL distribution
        (``wsl.localhost`` or ``wsl$``) is run through ``wsl.exe``.
    endf_path
        ENDF tape. The MAT number, the ZA and the isotope label are read from its
        headers.
    temperature
        Temperature in K, written unchanged on the ``broadr``, ``purr`` and
        ``acer`` cards. A value below 1.0 is read as MeV for the title, the folder
        names and the file names, which use the converted value in K.
    library_name
        Library label. It goes in the title (in upper case) and in the default
        folder names. Without *extension* it also picks the library number in the
        ACE file name: the name is lower-cased and stripped of ``-``, ``/`` and ``.``
        and looked up in ``NDLIBRARY_TO_SUFFIX``. An unknown name emits a warning
        and gives ``00``.
    output_dir
        Folder for the results and for the temporary run folder. It is created
        when missing. The run folder is deleted before the function returns,
        also when it raises.
    suff
        Text written in the suffix field of the first ``acer`` card. None writes
        the suffix of *temperature* (``.02`` for 293.6 K) or, with *extension*, the
        extension without its ``c`` (``.06`` for ``06c``).
    njoy_version
        Text shown in the title of the deck, as ``(NJOY 2016.78)``. kika does not
        check it against the executable.
    additional_suffix
        Used only without *extension*. A text inserted before the temperature
        suffix in the file names (``260560_81_<additional_suffix>.02``). A file of
        that name already in the folders is overwritten. None keeps the plain name
        when it is free, and otherwise inserts ``v01``, ``v02`` ... (the first
        free one) so that nothing is overwritten.
    extension
        ACE extension, such as ``06c``. Selects the fixed file names below.
        None selects the names built from *library_name* and the temperature.
    ace_dir, xsdir_dir, njoy_files_dir
        Folders for the ACE file, the xsdir file and the deck and logs. *ace_dir*
        defaults to ``<output_dir>/ace/<library_name>/<T>K`` and *njoy_files_dir*
        to ``<output_dir>/njoy_files/<library_name>/<T>K``, with *T* in K to one
        decimal. *xsdir_dir* defaults to the ``xsdir`` folder inside *ace_dir*
        with *extension*, and to *njoy_files_dir* without it. Folders are created
        when missing.
    acer_iprint
        The ``iprint`` field of the second (check) ``acer`` card, written
        unchanged. In NJOY, 0 prints the consistency checks and 1 also prints a
        dump of the whole table, which makes the listing much larger. kika
        takes the ACE and xsdir files from the same tapes either way.

    Returns
    -------
    dict
        The keys, all paths as ``str``:

        ``returncode``
            Exit code of the last NJOY run.
        ``zaid``
            ZA of the evaluation, from the tape (26056 for Fe-56).
        ``ace_filename``, ``base_filename``
            The file name of the ACE file and the stem of the other files. Without
            *extension* the stem is ``<10*ZA>_<library number><temperature
            suffix>`` (``260560_81.02``) and the ACE name adds ``c``
            (``260560_81.02c``). With *extension* the ACE name is
            ``<ZA>.<extension>`` and is also the stem (``26056.06c``). An existing
            file of that name is overwritten.
        ``njoy_input``
            ``<njoy_files_dir>/<stem>.input``: the deck that was run.
        ``njoy_output``
            ``<njoy_files_dir>/<stem>.output``: NJOY's standard output and error.
        ``njoy_listing``
            NJOY's own listing (the file ``output`` of its run folder), or None
            when NJOY wrote none. It is saved under the same name as
            ``njoy_output``, so when there is a listing it replaces the standard
            output in that file.
        ``ace_file``, ``xsdir_file``
            ``<ace_dir>/<ACE name>`` and ``<xsdir_dir>/<stem>.xsdir``. None when
            NJOY wrote no such tape, for instance after a failed run: a non-zero
            ``returncode`` does not raise.

    Raises
    ------
    ValueError
        If the MAT number or the ZA cannot be read from *endf_path*, or if
        *extension* is None and *temperature* is not within 1 K of one of the
        temperatures kika has an ACE suffix for (``K_TO_SUFFIX``). Pass
        *extension* for any other temperature.
    NjoyDecimalLocaleError
        If NJOY misread its deck on the last run (see above).
    """

    endf_path = Path(endf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # MAT, ZAID and the isotope come off the headers: open_endf indexes the
    # tape without parsing it (the full parse was ~4 s of a Fe-56 sample).
    endf = open_endf(str(endf_path))
    
    if endf.mat is not None:
        mat = endf.mat
    else:
        raise ValueError(f"Could not determine MAT number from ENDF file {endf_path}")
    
    # Get ZAID from ENDF data
    if endf.zaid is not None:
        zaid = endf.zaid
    else:
        raise ValueError(f"Could not determine ZAID from ENDF file {endf_path}. MAT: {endf.mat}")
    
    nd_suffix = _resolve_nd_suffix(library_name)

    # Get isotope symbol for the title
    isotope_symbol = endf.isotope

    # Get temperature information using helper function
    temp_K, temp_suffix_legacy = _get_temperature_info(temperature)

    # New layout: caller supplies extension + ace_dir/xsdir_dir.
    # Legacy layout: derive from library_name + temp_K (K_TO_SUFFIX).
    use_new_layout = extension is not None
    if use_new_layout:
        temp_suffix = _suff_from_extension(extension)
        ace_dir = Path(ace_dir) if ace_dir is not None else (
            output_dir / "ace" / library_name / f"{temp_K:.1f}K"
        )
        xsdir_dir = Path(xsdir_dir) if xsdir_dir is not None else ace_dir / "xsdir"
        njoy_files_dir = Path(njoy_files_dir) if njoy_files_dir is not None else (
            output_dir / "njoy_files" / library_name / f"{temp_K:.1f}K"
        )
        ace_filename = f"{zaid}.{extension}"
        xsdir_filename = f"{zaid}.{extension}.xsdir"
        base_filename = ace_filename
    else:
        temp_suffix = temp_suffix_legacy or ace_temperature_suffix(temp_K)
        ace_dir = Path(ace_dir) if ace_dir is not None else (
            output_dir / "ace" / library_name / f"{temp_K:.1f}K"
        )
        njoy_files_dir = Path(njoy_files_dir) if njoy_files_dir is not None else (
            output_dir / "njoy_files" / library_name / f"{temp_K:.1f}K"
        )
        xsdir_dir = Path(xsdir_dir) if xsdir_dir is not None else njoy_files_dir
        zaid_extended = zaid * 10
        base_filename_initial = f"{zaid_extended}_{nd_suffix}{temp_suffix}"
        base_filename, ace_filename = _generate_unique_filename(
            base_filename_initial, ace_dir, njoy_files_dir, temp_suffix, additional_suffix
        )
        xsdir_filename = f"{base_filename}.xsdir"

    ace_dir.mkdir(parents=True, exist_ok=True)
    xsdir_dir.mkdir(parents=True, exist_ok=True)
    njoy_files_dir.mkdir(parents=True, exist_ok=True)

    if suff is None:
        suff = temp_suffix

    title = _render_title(isotope_label=isotope_symbol, T=temp_K, library_name=library_name, njoy_version=njoy_version)
    njoy_input = _render_njoy_input(mat=mat, T=temperature, title=title, suff=suff,
                                    acer_iprint=acer_iprint)

    workdir = Path(tempfile.mkdtemp(prefix="njoy_", dir=output_dir))

    try:
        # NJOY expects tape files with specific unit numbers
        tape20 = workdir / "tape20"
        shutil.copy2(endf_path, tape20)

        # Write the input deck
        deck_path = workdir / "njoy.inp"
        deck_path.write_text(njoy_input)

        # Run NJOY: feed stdin from the input deck
        result = _run_njoy_deck(njoy_exe, workdir, njoy_input)

        # Save organized output files
        results = {
            "returncode": result.returncode,
            "zaid": zaid,
            "ace_filename": ace_filename,
            "base_filename": base_filename,
        }
        
        # Save NJOY input file
        input_file = njoy_files_dir / f"{base_filename}.input"
        shutil.copy2(deck_path, input_file)
        results["njoy_input"] = str(input_file)
        
        # Save NJOY output (log)
        output_file = njoy_files_dir / f"{base_filename}.output"
        output_file.write_bytes(result.stdout)
        results["njoy_output"] = str(output_file)

        results["njoy_listing"] = _guard_decimal_locale(
            workdir, njoy_input, njoy_files_dir, base_filename
        )
        
        # Process tape files (second ACER pass writes final ACE/xsdir)
        tape43 = workdir / "tape43"  # ACE file (final, from second ACER pass)
        tape44 = workdir / "tape44"  # xsdir file (final, from second ACER pass)

        if tape43.exists():
            ace_file = ace_dir / ace_filename
            shutil.move(str(tape43), ace_file)
            results["ace_file"] = str(ace_file)
        else:
            results["ace_file"] = None

        if tape44.exists():
            xsdir_file = xsdir_dir / xsdir_filename
            shutil.move(str(tape44), xsdir_file)
            results["xsdir_file"] = str(xsdir_file)
        else:
            results["xsdir_file"] = None

        return results

    finally:
        # Always clean up temporary directory
        shutil.rmtree(workdir, ignore_errors=True)


def run_njoy_with_pendf(
    njoy_exe: str,
    endf_path: str | Path,
    pendf_path: str | Path,
    temperature: float,
    library_name: str,
    output_dir: str | Path,
    *,
    suff: str = None,
    njoy_version: str = "NJOY 2016.78",
    additional_suffix: str = None,
    extension: str | None = None,
    ace_dir: str | Path | None = None,
    xsdir_dir: str | Path | None = None,
    njoy_files_dir: str | Path | None = None,
    acer_iprint: int = 0,
) -> dict:
    """Run NJOY from an existing PENDF tape instead of reconstructing one.

    The deck is the one of :func:`run_njoy` without its ``reconr`` step: after the
    first ``moder`` (ENDF tape to binary), a second ``moder`` converts the ASCII
    PENDF *pendf_path* to the binary tape that ``broadr`` reads, and the chain
    continues with ``broadr``, ``heatr``, ``purr``, ``gaspr`` and the two ``acer``
    passes with the same values. The ENDF tape is still the ENDF input of those
    modules. The files written, their names, the returned dict and the raised
    exceptions are those of :func:`run_njoy`.

    The decimal-locale guard also applies. With no ``reconr`` card in the deck, the
    check reads the tolerances echoed by ``broadr`` (``errthn``, ``errmax``,
    ``errint``). kika does not check the PENDF tape itself.

    Parameters
    ----------
    endf_path
        ENDF tape. The MAT number, the ZA and the isotope label are read from its
        headers.
    pendf_path
        ASCII PENDF tape. It is copied to ``tape26`` of the run folder and takes
        the place of the output of ``reconr``.
    njoy_exe, temperature, library_name, output_dir, suff, njoy_version
        As in :func:`run_njoy`.
    additional_suffix, extension, ace_dir, xsdir_dir, njoy_files_dir, acer_iprint
        As in :func:`run_njoy`.

    Returns
    -------
    dict
        As :func:`run_njoy`.

    Raises
    ------
    ValueError
        If the MAT number or the ZA cannot be read from *endf_path*, or if
        *extension* is None and *temperature* has no ACE suffix (see
        :func:`run_njoy`).
    NjoyDecimalLocaleError
        If NJOY misread its deck on the last run.
    """
    endf_path = Path(endf_path).resolve()
    pendf_path = Path(pendf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    endf = open_endf(str(endf_path))  # headers only; see run_njoy
    if endf.mat is None:
        raise ValueError(f"Could not determine MAT number from ENDF file {endf_path}")
    if endf.zaid is None:
        raise ValueError(f"Could not determine ZAID from ENDF file {endf_path}")
    mat = endf.mat
    zaid = endf.zaid

    nd_suffix = _resolve_nd_suffix(library_name)
    isotope_symbol = endf.isotope

    temp_K, temp_suffix_legacy = _get_temperature_info(temperature)

    use_new_layout = extension is not None
    if use_new_layout:
        temp_suffix = _suff_from_extension(extension)
        ace_dir = Path(ace_dir) if ace_dir is not None else (
            output_dir / "ace" / library_name / f"{temp_K:.1f}K"
        )
        xsdir_dir = Path(xsdir_dir) if xsdir_dir is not None else ace_dir / "xsdir"
        njoy_files_dir = Path(njoy_files_dir) if njoy_files_dir is not None else (
            output_dir / "njoy_files" / library_name / f"{temp_K:.1f}K"
        )
        ace_filename = f"{zaid}.{extension}"
        xsdir_filename = f"{zaid}.{extension}.xsdir"
        base_filename = ace_filename
    else:
        temp_suffix = temp_suffix_legacy or ace_temperature_suffix(temp_K)
        ace_dir = Path(ace_dir) if ace_dir is not None else (
            output_dir / "ace" / library_name / f"{temp_K:.1f}K"
        )
        njoy_files_dir = Path(njoy_files_dir) if njoy_files_dir is not None else (
            output_dir / "njoy_files" / library_name / f"{temp_K:.1f}K"
        )
        xsdir_dir = Path(xsdir_dir) if xsdir_dir is not None else njoy_files_dir
        zaid_extended = zaid * 10
        base_filename_initial = f"{zaid_extended}_{nd_suffix}{temp_suffix}"
        base_filename, ace_filename = _generate_unique_filename(
            base_filename_initial, ace_dir, njoy_files_dir, temp_suffix, additional_suffix,
        )
        xsdir_filename = f"{base_filename}.xsdir"

    ace_dir.mkdir(parents=True, exist_ok=True)
    xsdir_dir.mkdir(parents=True, exist_ok=True)
    njoy_files_dir.mkdir(parents=True, exist_ok=True)

    if suff is None:
        suff = temp_suffix

    title = _render_title(
        isotope_label=isotope_symbol, T=temp_K, library_name=library_name,
        njoy_version=njoy_version,
    )
    njoy_input = _render_njoy_input(
        mat=mat, T=temperature, title=title, suff=suff,
        template=NJOY_INPUT_TEMPLATE_WITH_PENDF, acer_iprint=acer_iprint,
    )

    workdir = Path(tempfile.mkdtemp(prefix="njoy_pendf_", dir=output_dir))

    try:
        tape20 = workdir / "tape20"
        tape26 = workdir / "tape26"
        shutil.copy2(endf_path, tape20)
        shutil.copy2(pendf_path, tape26)

        deck_path = workdir / "njoy.inp"
        deck_path.write_text(njoy_input)

        result = _run_njoy_deck(njoy_exe, workdir, njoy_input)

        results = {
            "returncode": result.returncode,
            "zaid": zaid,
            "ace_filename": ace_filename,
            "base_filename": base_filename,
        }

        input_file = njoy_files_dir / f"{base_filename}.input"
        shutil.copy2(deck_path, input_file)
        results["njoy_input"] = str(input_file)

        output_file = njoy_files_dir / f"{base_filename}.output"
        output_file.write_bytes(result.stdout)
        results["njoy_output"] = str(output_file)

        results["njoy_listing"] = _guard_decimal_locale(
            workdir, njoy_input, njoy_files_dir, base_filename
        )

        tape43 = workdir / "tape43"
        tape44 = workdir / "tape44"

        if tape43.exists():
            ace_file = ace_dir / ace_filename
            shutil.move(str(tape43), ace_file)
            results["ace_file"] = str(ace_file)
        else:
            results["ace_file"] = None

        if tape44.exists():
            xsdir_file = xsdir_dir / xsdir_filename
            shutil.move(str(tape44), xsdir_file)
            results["xsdir_file"] = str(xsdir_file)
        else:
            results["xsdir_file"] = None

        return results

    finally:
        shutil.rmtree(workdir, ignore_errors=True)