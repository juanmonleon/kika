"""The doors ``kika.read(..., format="hdf5")`` and ``kika.write(..., format="hdf5")`` open.

Reading dispatches on the file's ``filetype``: ``data_neutron`` →
:class:`~kika.nuclear_data.model.ReactionSuite`, ``data_thermal`` → a suite of
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
    """The root ``filetype`` attribute, ``data_`` prefix included."""
    import h5py

    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("filetype")
    if value is None:
        raise HDF5FormatError("no root 'filetype' attribute; not an OpenMC data file", path=path)
    return value.decode() if isinstance(value, bytes) else str(value)


def read(path, **kwargs):
    """One OpenMC HDF5 file into the model, with ``.report`` set."""
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
    :func:`kika.hdf5.writer.writeSuite`) is serialised as it is; an evaluation
    is processed first. Returns the :class:`ConversionReport`.
    """
    from .writer import writeSuite

    return writeSuite(suite, directory, temperatures=temperatures, **kwargs)
