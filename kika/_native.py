"""Loading kika's compiled extensions, and saying so when one is missing.

Every ``*_native`` extension has a Python (NumPy) definition that gives the same
results, so kika runs without them -- slower, sometimes several times slower.
That used to happen in silence: an install whose build failed, or a checkout
whose extensions were never built or are out of date, ran the slow path and
nothing said so. :func:`load` is the one place every extension is loaded from,
and it warns once per extension with what is slower and how to fix it.

Not for an extension kika cannot do without (``_thermal_native``): that one
raises where it is used.

This module is a leaf. It imports nothing from kika.
"""
import warnings
from importlib import import_module

__all__ = ["NativeExtensionWarning", "load", "status", "native_status"]


class NativeExtensionWarning(RuntimeWarning):
    """A compiled extension is missing or out of date; kika runs its Python code."""


#: name -> "loaded", "missing" or "out of date", for every extension asked for.
_STATUS = {}

_FIX = (
    "To fix it: in a source checkout, run `python build_native.py` from the "
    "repository root (it needs a C compiler: MSVC on Windows, gcc or clang "
    "elsewhere; MinGW gcc on PATH is tried automatically on Windows); "
    "with pip, reinstall the wheel: `pip install --force-reinstall --no-deps kika-nd`. "
    "kika.native_status() lists every extension. To silence this knowingly: "
    "warnings.filterwarnings('ignore', category=kika.NativeExtensionWarning)."
)


def load(name, api_version, what):
    """The compiled module *name*, or None after a warning.

    *api_version* is the ``API_VERSION`` the Python side was written against;
    a module that states another one is an out-of-date build and is not used.
    None for an extension that states no version (``_doppler_native``).
    *what* says in a few words what runs slower without it.
    """
    try:
        module = import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        _STATUS[name] = "missing"
        warnings.warn(f"kika's compiled extension {name} is not built, so {what} runs "
                      f"in Python: the same results, slower. {_FIX}",
                      NativeExtensionWarning, stacklevel=2)
        return None
    if api_version is not None and getattr(module, "API_VERSION", None) != api_version:
        _STATUS[name] = "out of date"
        warnings.warn(f"kika's compiled extension {name} is out of date (API version "
                      f"{getattr(module, 'API_VERSION', None)}, this kika needs {api_version}), "
                      f"so {what} runs in Python: the same results, slower. {_FIX}",
                      NativeExtensionWarning, stacklevel=2)
        return None
    _STATUS[name] = "loaded"
    return module


def status():
    """``{extension: "loaded" | "missing" | "out of date"}`` for every extension.

    Imports the modules that load them first, so the answer does not depend on
    what the session happened to use.
    """
    for module in ("kika._records", "kika.processing.doppler.core", "kika.processing.heating",
                   "kika.processing.tabulation.angular", "kika.processing.resonances._rm_acceleration",
                   "kika.processing.resonances.coulomb"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NativeExtensionWarning)
            import_module(module)
    result = dict(sorted(_STATUS.items()))
    try:
        thermal = import_module("kika.processing._thermal_native")
        result["kika.processing._thermal_native"] = (
            "loaded" if getattr(thermal, "API_VERSION", None) == 3 else "out of date")
    except ModuleNotFoundError:
        result["kika.processing._thermal_native"] = "missing"
    return result


#: The public name, ``kika.native_status()``.
native_status = status
