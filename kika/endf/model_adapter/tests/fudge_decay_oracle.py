"""Runs inside FUDGE's interpreter, never kika's (see ``test_decay_sublibrary.py``).

Given ``PAYLOAD`` -- a JSON object with an ENDF decay or fission-yield tape
(``endf``) and, optionally, the GNDS kika wrote from it (``kika``) -- it prints
one line ``KIKA-ORACLE-JSON:{...}`` with:

- ``gnds``: the GNDS FUDGE's ``endfFileToGNDS`` writes from the tape;
- ``endfFromFudge`` / ``endfFromKika``: FUDGE's ``toENDF6`` of its own GNDS and
  of kika's (or ``fudgeError`` / ``kikaError``);
- ``yieldsFromFudge`` / ``yieldsFromKika``: the fission product yields FUDGE
  reads from each file, ``{elapsedTime: [[energy, {nuclides, values,
  variances}], ...]}`` (or ``...Error``).
"""
import json
import os
import tempfile
import traceback

from brownies.legacy.converting import endfFileToGNDS
import brownies.legacy.toENDF6.PoPs_toENDF6.database  # noqa: F401  (registers toENDF6)
import brownies.legacy.toENDF6.PoPs_toENDF6.decays.decayData  # noqa: F401
import brownies.legacy.toENDF6.PoPs_toENDF6.fissionFragmentData.productYield  # noqa: F401
import brownies.legacy.toENDF6.toENDF6  # noqa: F401
from fudge import GNDS_file

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


def _productYield(top):
    if hasattr(top, "productYields"):
        return top.productYields[0] if len(top.productYields) else None
    for key in top.keys():
        data = getattr(top[key], "fissionFragmentData", None)
        if data is not None and len(data.productYields):
            return data.productYields[0]
    return None


def _yields(y, common):
    try:
        names = list(y.nuclides.data)
    except Exception:                                    # noqa: BLE001
        names = list(common)
    variances = (list(y.uncertainty.form().matrix().constructArray().diagonal())
                 if y.uncertainty else [])
    return {"nuclides": names, "values": [float(v) for v in y.values],
            "variances": [float(v) for v in variances]}


def yieldsOf(top):
    productYield = _productYield(top)
    if productYield is None:
        return None
    common = list(productYield.nuclides.data) if productYield.nuclides is not None else []
    dump = {}
    for elapsed in productYield.elapsedTimes:
        rows = []
        if len(getattr(elapsed, "incidentEnergies", None) or []):
            for incident in elapsed.incidentEnergies:
                rows.append([float(incident.energy[0].value), _yields(incident.yields, common)])
        else:
            rows.append([None, _yields(elapsed.yields, common)])
        dump[elapsed.label] = rows
    return dump


def run(path, name):
    try:
        top = GNDS_file.read(path)
    except Exception:                                    # noqa: BLE001
        out[f"{name}Error"] = traceback.format_exc()[-1500:]
        return
    try:
        out[f"yieldsFrom{name}"] = yieldsOf(top)
    except Exception:                                    # noqa: BLE001
        out[f"yieldsFrom{name}Error"] = traceback.format_exc()[-1500:]
    try:
        out[f"endfFrom{name}"] = top.toENDF6("eval", {"verbosity": 0})
    except Exception:                                    # noqa: BLE001
        out[f"endfFrom{name}Error"] = traceback.format_exc()[-1500:]


run(ours, "Fudge")
if payload.get("kika"):
    theirs = os.path.join(folder, "kika.xml")
    with open(theirs, "w") as handle:
        handle.write(payload["kika"])
    run(theirs, "Kika")
print("KIKA-ORACLE-JSON:" + json.dumps(out))
