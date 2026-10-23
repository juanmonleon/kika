"""A missing or out-of-date compiled extension is said, with the fix, never silent."""
import subprocess
import sys
import types
import warnings

import pytest

import kika
import kika._native
from kika._native import MESSAGE, NativeExtensionWarning, load


@pytest.fixture
def fresh(monkeypatch):
    """A session that has not warned yet."""
    monkeypatch.setattr(kika._native, "_warned", False)
    monkeypatch.setattr(kika._native, "_STATUS", {})


def test_one_generic_warning_with_the_fix_whichever_extension(fresh, monkeypatch):
    stale = types.ModuleType("kika._stale_native")
    stale.API_VERSION = 1
    monkeypatch.setitem(sys.modules, "kika._stale_native", stale)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert load("kika._no_such_native", 1) is None
        assert load("kika._stale_native", 2) is None
        assert load("kika._other_missing_native") is None
    assert [str(w.message) for w in caught] == [MESSAGE]
    assert caught[0].category is NativeExtensionWarning
    assert "python build_native.py" in MESSAGE and "pip install --force-reinstall" in MESSAGE
    assert kika._native._STATUS == {"kika._no_such_native": "missing",
                                    "kika._stale_native": "out of date",
                                    "kika._other_missing_native": "missing"}


def test_a_current_extension_is_used_without_a_warning(fresh, monkeypatch):
    current = types.ModuleType("kika._current_native")
    current.API_VERSION = 1
    monkeypatch.setitem(sys.modules, "kika._current_native", current)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load("kika._current_native", 1) is current


def test_an_error_inside_the_extension_is_not_taken_for_a_missing_one(monkeypatch):
    def broken(name, *args, **kwargs):
        raise ModuleNotFoundError("No module named 'numpy_core_x'", name="numpy_core_x")
    monkeypatch.setattr("kika._native.import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        load("kika._records_native", 1)


def test_the_built_extensions_are_reported():
    status = kika.native_status()
    assert set(status) >= {"kika._records_native", "kika.processing._thermal_native",
                           "kika.processing.resonances._rm_native"}
    assert set(status.values()) <= {"loaded", "missing", "out of date"}


BLOCKED = r"""
import sys, warnings
class Block:
    def find_spec(self, name, path=None, target=None):
        if name in ("kika._records_native", "kika.processing.resonances._rm_native"):
            raise ModuleNotFoundError("No module named " + name, name=name)
sys.meta_path.insert(0, Block())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import kika._records as R
    import kika.processing.resonances._rm_acceleration
print(len([w for w in caught if w.category.__name__ == "NativeExtensionWarning"]))
print(R.parse_number(" 1.234567+5"), R.format_endf_number(0.12640975), R._records_native)
"""


def test_without_two_extensions_kika_says_so_once_and_reads_the_same():
    out = subprocess.run([sys.executable, "-c", BLOCKED], capture_output=True, text=True,
                         check=True, cwd=str(__import__("pathlib").Path(kika.__file__).parents[1]))
    count, values = out.stdout.splitlines()
    assert count == "1"
    assert values == "123456.7  0.12640975 None"
