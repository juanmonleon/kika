"""OpenMC's HDF5 nuclear data — one more door into and out of the model.

The format OpenMC's continuous-energy transport reads (``cross_sections.xml``
plus one ``.h5`` per nuclide, thermal material or element): **processed** data,
every temperature of a nuclide in one file, organised by reaction and product.
It plays the part ACE plays for MCNP. The plan is
``kika-workspace/docs/library/openmc_hdf5_roadmap.md``; the format reference is
``openmc_hdf5_format_survey.md`` beside it, built from OpenMC v0.16.0's source.

OpenMC is **not** a dependency of kika, not even an optional one (roadmap D3):
the files are read and written with h5py, against what OpenMC's C++ reads, and
checking against a real OpenMC happens outside the library
(``kika-workspace/kika_dev/hdf5/``).

Layers, bottom up:

- :mod:`kika.hdf5.records` — the HDF5 tree as plain records, nothing interpreted;
- :mod:`kika.hdf5.library` — ``cross_sections.xml``, and the overlay that
  points some materials at kika's files and leaves the rest where they are;
- the neutron and thermal decoders and encoders, reached through
  ``kika.read(path, format="hdf5")`` and ``kika.write(suite, dir, format="hdf5")``.

Nothing here imports h5py or :mod:`kika.nuclear_data.model` at module scope:
``import kika.hdf5`` must not wake either (``kika/hdf5/tests/test_dormancy.py``).
"""
from __future__ import annotations

__all__ = ["records", "library", "open", "read", "write"]

#: Lazily-resolved names -> the module holding each, as in ``kika/__init__.py``.
_LAZY = {
    "open": "library",
    "Library": "library",
    "CrossSectionsXml": "library",
    "read": "api",
    "write": "api",
    "writeSuite": "neutron_encode",
    "patch_library": "patch",
}

_SUBMODULES = ("records", "library", "names", "functions", "distributions",
               "neutron_decode", "neutron_encode", "thermal_decode",
               "thermal_encode", "patch", "api", "tables")


def __getattr__(name):
    import importlib

    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_SUBMODULES))
