"""A missing or out-of-date compiled extension is said, with the fix, never silent."""
import subprocess
import sys
import types
import warnings

import pytest

import kika
from kika._native import NativeExtensionWarning, load


def test_a_missing_extension_warns_with_the_fix_and_gives_none():
    with pytest.warns(NativeExtensionWarning) as caught:
        assert load("kika._no_such_native", 1, "something") is None
    message = str(caught[0].message)
    assert "kika._no_such_native is not built" in message
    assert "python build_native.py" in message and "pip install --force-reinstall" in message


def test_an_out_of_date_extension_warns_and_is_not_used(monkeypatch):
    stale = types.ModuleType("kika._stale_native")
    stale.API_VERSION = 1
    monkeypatch.setitem(sys.modules, "kika._stale_native", stale)
    with pytest.warns(NativeExtensionWarning, match="out of date"):
        assert load("kika._stale_native", 2, "something") is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load("kika._stale_native", 1, "something") is stale


def test_an_error_inside_the_extension_is_not_taken_for_a_missing_one(monkeypatch):
    def broken(name, *args, **kwargs):
        raise ModuleNotFoundError("No module named 'numpy_core_x'", name="numpy_core_x")
    monkeypatch.setattr("kika._native.import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        load("kika._records_native", 1, "something")


def test_the_built_extensions_are_reported():
    status = kika.native_status()
    assert set(status) >= {"kika._records_native", "kika.processing._thermal_native",
                           "kika.processing.resonances._rm_native"}
    assert set(status.values()) <= {"loaded", "missing", "out of date"}


BLOCKED = r"""
import sys, warnings
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "kika._records_native":
            raise ModuleNotFoundError("No module named " + name, name=name)
sys.meta_path.insert(0, Block())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import kika._records as R
print(len([w for w in caught if w.category.__name__ == "NativeExtensionWarning"]))
print(R.parse_number(" 1.234567+5"), R.format_endf_number(0.12640975), R._records_native)
"""


def test_without_the_extension_kika_says_so_and_reads_the_same():
    out = subprocess.run([sys.executable, "-c", BLOCKED], capture_output=True, text=True,
                         check=True, cwd=str(__import__("pathlib").Path(kika.__file__).parents[1]))
    count, values = out.stdout.splitlines()
    assert count == "1"
    assert values == "123456.7  0.12640975 None"
