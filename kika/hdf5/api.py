"""The doors ``kika.read(..., format="hdf5")`` and ``kika.write(..., format="hdf5")`` open.

Reading dispatches on the file's ``filetype``: ``data_neutron`` →
:class:`~kika.nuclear_data.model.suite.ReactionSuite`, ``data_thermal`` → a suite of
interaction ``thermalNeutronScatteringLaw``. ``data_photon`` is transcribed by
the record layer and not decoded into the model, which has no photo-atomic
suite; reading one raises and says so. ``data_wmp`` is
the parked windowed multipole.
"""
from __future__ import annotations

from pathlib import Path

from .exceptions import HDF5FormatError, HDF5UnsupportedError

__all__ = ["read", "write", "filetype", "HDF5_SIGNATURE"]

#: The first eight bytes of every HDF5 file without a user block.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def filetype(path) -> str:
    """The root ``filetype`` attribute of an OpenMC file, ``data_`` prefix included.

    Parameters
    ----------
    path
        The ``.h5`` file.

    Returns
    -------
    str
        For example ``"data_neutron"``, ``"data_thermal"``, ``"data_photon"`` or
        ``"data_wmp"``. The value is returned as stored; it is not checked against
        that list.

    Raises
    ------
    HDF5FormatError
        If the root has no ``filetype`` attribute.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("filetype")
    if value is None:
        raise HDF5FormatError("no root 'filetype' attribute; not an OpenMC data file", path=path)
    return value.decode() if isinstance(value, bytes) else str(value)


def read(path, **kwargs):
    """One OpenMC HDF5 file decoded into the model.

    The file type decides what is decoded: ``data_neutron`` and ``data_thermal``
    files are read; ``data_photon`` and ``data_wmp`` are not (see *Raises*).

    Parameters
    ----------
    path
        The ``.h5`` file.
    **kwargs
        Accepted so that this function has the signature ``kika.read`` passes
        through; none is used.

    Returns
    -------
    ReactionSuite
        For ``data_neutron``, the processed suite (``heated_<T>`` cross sections,
        one grid per temperature; OpenMC stores energies in eV and cross
        sections in barn). For ``data_thermal``, a suite of interaction
        ``thermalNeutronScatteringLaw``. Its ``report`` attribute lists what the
        decoder could not carry over.

    Raises
    ------
    HDF5FormatError
        If the root has no ``filetype``, or the file breaks the layout the decoder
        needs.
    HDF5UnsupportedError
        If the file type is ``data_photon`` (the model has no photo-atomic suite;
        :mod:`kika.hdf5.records` reads it as plain records) or any type other
        than neutron and thermal.
    """
    path = Path(path)
    kind = filetype(path)
    if kind == "data_neutron":
        from .neutron_decode import readNeutron

        suite, _ = readNeutron(path)
        return suite
    if kind == "data_thermal":
        from .thermal_decode import readThermal

        suite, _ = readThermal(path)
        return suite
    if kind == "data_photon":
        raise HDF5UnsupportedError(
            "a photo-atomic file has no home in the model yet; "
            "kika.hdf5.records reads and writes it as records")
    raise HDF5UnsupportedError(f"filetype {kind!r} is not read by kika")


def write(suite, directory, temperatures=None, **kwargs):
    """Write a suite as an OpenMC file in ``directory`` and index it.

    A suite already in the processed shape (read from HDF5, or built by
    ``kika.hdf5.writer.writeSuite``) is serialised as it is; an evaluation
    is processed first (:func:`kika.hdf5.processing.processSuite`; a thermal
    evaluation, :func:`kika.hdf5.thermal_processing.processThermal`).

    The file is written under a temporary name in ``directory``, read back and
    compared with what was meant, and only then moved over any file of the same
    name. A ``cross_sections.xml`` in ``directory`` is created or updated to index
    it. The file is named ``<material>.h5`` after its top-level group unless
    ``fileName`` is given.

    Parameters
    ----------
    suite
        A processed suite, or an evaluation decoded from ENDF or GNDS.
    directory
        Created if absent.
    temperatures
        Temperatures to write, in K. For an evaluation they are the temperatures
        processed; ``None`` is 293.6 K (group ``294K``) for a neutron evaluation
        and every temperature the evaluation tabulates for a thermal one. For a
        processed suite they restrict the ``heated_<T>`` styles written (names such as ``"294K"``
        or numbers); the default is every temperature it has.
    **kwargs
        ``report`` (a ``ConversionReport`` to fill, default a new one), ``index``
        (``False`` leaves ``cross_sections.xml`` alone, default ``True``) and
        ``fileName``. For an evaluation, any other keyword reaches
        ``processSuite``; for a thermal evaluation they must include ``endf``
        (the path of its tape) and ``name`` (the OpenMC table name, such as
        ``"c_H_in_H2O"``), and may include ``neutron`` and the keywords of
        ``processThermal``.

    Returns
    -------
    ConversionReport
        What the conversion approximated, lost or warned about.

    Raises
    ------
    ValueError
        If processing options are given for a neutron suite that is already
        processed (a processed thermal suite ignores extra keywords), or a thermal evaluation is given without ``endf`` and ``name``.
    HDF5FormatError
        If the file read back differs from what was written; the temporary file is
        removed.
    """
    from .writer import writeSuite

    return writeSuite(suite, directory, temperatures=temperatures, **kwargs)
