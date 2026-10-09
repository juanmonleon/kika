"""Runs inside FUDGE's interpreter, never kika's (see ``test_decay_sublibrary.py``).

Given ``PAYLOAD`` -- a JSON object with an ENDF decay or fission-yield tape
(``endf``) and, optionally, the GNDS kika wrote from it (``kika``) -- it prints
one line ``KIKA-ORACLE-JSON:{...}`` with:

- ``gnds``: the GNDS FUDGE's ``endfFileToGNDS`` writes from the tape;
- ``endfFromFudge``: FUDGE's ``toENDF6`` of that GNDS;
- ``endfFromKika``: FUDGE's ``toENDF6`` of kika's GNDS (or ``kikaError``).
"""
import json
import os
import tempfile
import traceback

from brownies.legacy.converting import endfFileToGNDS
import brownies.legacy.toENDF6.PoPs_toENDF6.database  # noqa: F401  (registers toENDF6)
import brownies.legacy.toENDF6.PoPs_toENDF6.decays.decayData  # noqa: F401
import brownies.legacy.toENDF6.toENDF6  # noqa: F401

payload = json.loads(PAYLOAD)  # noqa: F821  (prepended by the test)
folder = tempfile.mkdtemp()
tape = os.path.join(folder, "tape.endf")
with open(tape, "w") as handle:
    handle.write(payload["endf"])

results = endfFileToGNDS.endfFileToGNDS(tape, toStdOut=False, skipBadData=True)
top = results.get("fissionFragmentData") or results.get("PoPs")
ours = os.path.join(folder, "fudge.xml")
top.saveToFile(ours)
out = {"gnds": open(ours).read()}


def toENDF6(path):
    text = open(path).read()
    if "<fissionFragmentData" in text[:400]:
        from fudge import fissionFragmentData as fissionFragmentDataModule
        data = fissionFragmentDataModule.FissionFragmentData.readXML_file(path)
        return data.toENDF6("eval", {"verbosity": 0})
    from PoPs import database as databaseModule
    return databaseModule.read(path).toENDF6("eval", {"verbosity": 0})


try:
    out["endfFromFudge"] = toENDF6(ours)
except Exception:                                     # noqa: BLE001
    out["fudgeError"] = traceback.format_exc()[-1500:]
if payload.get("kika"):
    theirs = os.path.join(folder, "kika.xml")
    with open(theirs, "w") as handle:
        handle.write(payload["kika"])
    try:
        out["endfFromKika"] = toENDF6(theirs)
    except Exception:                                 # noqa: BLE001
        out["kikaError"] = traceback.format_exc()[-1500:]
print("KIKA-ORACLE-JSON:" + json.dumps(out))
