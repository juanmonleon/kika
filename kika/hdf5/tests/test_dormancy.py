"""``import kika`` and ``import kika.hdf5`` wake neither h5py nor the model.

h5py is a required dependency (roadmap D2) and still imported lazily, as every
format is: the cluster pipeline, the app and every notebook import kika, and
none of them should pay for HDF5 until they open a file. Run in a subprocess
for the reason ``kika/nuclear_data/model/tests/test_dormancy.py`` gives.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(body: str) -> str:
    completed = subprocess.run([sys.executable, "-c", textwrap.dedent(body)],
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout.strip()


def test_import_kika_does_not_import_h5py_or_kika_hdf5():
    assert _run("""
        import sys
        import kika
        leaked = sorted(m for m in sys.modules if m == "h5py" or m.startswith(("h5py.", "kika.hdf5")))
        assert not leaked, leaked
        print("dormant")
    """) == "dormant"


def test_import_kika_hdf5_wakes_neither_h5py_nor_the_model():
    assert _run("""
        import sys
        import kika.hdf5
        import kika.hdf5.records, kika.hdf5.library, kika.hdf5.names
        leaked = sorted(m for m in sys.modules
                        if m == "h5py" or m.startswith(("h5py.", "kika.nuclear_data.model")))
        assert not leaked, leaked
        print("dormant")
    """) == "dormant"


def test_h5py_is_there_when_a_file_is_opened():
    """Dormant is not absent: the dependency is required, and installed."""
    assert _run("""
        import h5py
        print("present")
    """) == "present"
