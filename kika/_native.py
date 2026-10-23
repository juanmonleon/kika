"""Loading kika's compiled extensions, and saying so when kika runs without them.

Every ``*_native`` extension but the thermal kernel has a Python (NumPy)
definition that gives the same results, so kika runs without them -- slower,
sometimes several times slower. That used to happen in silence: an install
whose build failed, or a checkout whose extensions were never built or are out
of date, ran the slow path and nothing said so.

:func:`load` is the one place they are loaded from. The first time any of them
is missing or out of date it gives one warning, :data:`MESSAGE`, the same
whichever it is: the cause and the fix are the same for all of them, so it is
said once per session, not once per extension. :func:`status` says which.

The thermal kernel has no Python definition and raises where it is used, with
the same fix (:data:`FIX`).

This module is a leaf. It imports nothing from kika.
"""
import warnings
from importlib import import_module

__all__ = ["NativeExtensionWarning", "MESSAGE", "FIX", "load", "status", "native_status"]


class NativeExtensionWarning(RuntimeWarning):
    """kika is running its Python code where a compiled (C) extension is missing."""


#: How to get the compiled extensions back.
FIX = ("To fix it: in a source checkout, run `python build_native.py` from the repository "
       "root (it needs a C compiler: MSVC on Windows, gcc or clang elsewhere; MinGW gcc on "
       "PATH is tried automatically on Windows); with pip, reinstall the wheel: "
       "`pip install --force-reinstall --no-deps kika-nd`.")

#: The one warning, whichever extension is missing or out of date.
MESSAGE = ("kika is running without some of its compiled (C) extensions, which are missing or "
           "out of date, and uses its Python code instead: the same results, slower. " + FIX
           + " kika.native_status() says which extensions are affected. To silence this "
           "knowingly: warnings.filterwarnings('ignore', category=kika.NativeExtensionWarning).")

#: name -> "loaded", "missing" or "out of date", for every extension asked for.
_STATUS = {}
_warned = False


def _fallback(name, state):
    global _warned
    _STATUS[name] = state
    if not _warned:
        _warned = True
        warnings.warn(MESSAGE, NativeExtensionWarning, stacklevel=3)
    return None


def load(name, api_version=None):
    """The compiled module *name*, or None (and the warning) when kika must use Python.

    *api_version* is the ``API_VERSION`` the Python side was written against;
    a module that states another one is an out-of-date build and is not used.
    None for an extension that states no version (``_doppler_native``).
    """
    try:
        module = import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return _fallback(name, "missing")
    if api_version is not None and getattr(module, "API_VERSION", None) != api_version:
        return _fallback(name, "out of date")
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
