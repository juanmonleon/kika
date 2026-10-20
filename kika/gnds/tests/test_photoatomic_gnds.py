"""Photo-atomic and atomic-relaxation GNDS ↔ the model (the photon plan's F2b).

Always run, from the committed iron micro-tapes:

* an atomic-relaxation PoPs read from ENDF, written as GNDS and read back, is
  the same model, number for number;
* a photo-atomic suite read from ENDF, written as GNDS and read back, keeps every
  cross section, sum and coherent/incoherent form; and written again from what
  was read, it reads back identical (GNDS → model → GNDS is a fixed point).

With the NNDC ENDF/B-VIII.1 GNDS files (``$KIKA_PHOTON_GNDS``, or
``../NuclearData/gnds/endfb81`` beside the repo), over all 100 + 100:

1. NNDC GNDS → model → GNDS → model gives the same model;
2. the same evaluation read from its ENDF tape and from NNDC's GNDS agree on
   every number, once FUDGE's two conventions are allowed for: a regions1d split
   at a repeated point is the same function as one region, and FUDGE drops the
   leading (edge, 0) point of a photoelectric subshell. The anomalous factors are
   checked against the tape's own sections (MT 506 real, MT 505 imaginary,
   ENDF-6 §27).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.nuclear_data.model.compare import modelDifferences

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
PHOTOAT = DATA / "micro_fe_photoatomic.endf"
RELAX = DATA / "micro_fe_relaxation.endf"
#: Bookkeeping that is not the data being compared.
IGNORE = ("report", "provenance", "styles", "documentation")


def _merged(regions):
    out = []
    for nbt, law in regions:
        if out and out[-1][1] == law:
            out[-1] = (nbt, law)
        else:
            out.append((int(nbt), int(law)))
    return out


def _table(f):
    if f is None:
        return None
    x, y, regions = f.toEndfRegions()
    return np.asarray(x, float), np.asarray(y, float), _merged(regions)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return (a[0].shape == b[0].shape and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            and a[2] == b[2])


def _edgeZero(a, b) -> bool:
    """*a* is *b* with the leading (edge, 0) point FUDGE drops at a subshell edge."""
    return (a is not None and b is not None and a[0].size == b[0].size + 1 and a[1][0] == 0.0
            and a[0][0] == a[0][1] and np.array_equal(a[0][1:], b[0]) and np.array_equal(a[1][1:], b[1]))


def _byMt(suite):
    return {r.ENDF_MT: r for r in list(suite.reactions) + list(suite.sums)}


_PARTS = ("formFactor", "realAnomalousFactor", "imaginaryAnomalousFactor", "scatteringFactor")


def _contentDifferences(a, b, edgeZeroAllowed=False):
    """Every MT whose cross section or scattering form differs between two photo-atomic suites."""
    out = []
    ma, mb = _byMt(a), _byMt(b)
    if set(ma) != set(mb):
        out.append(f"MTs {sorted(set(ma) ^ set(mb))}")
    for mt in sorted(set(ma) & set(mb)):
        ta, tb = _table(ma[mt].crossSection["eval"]), _table(mb[mt].crossSection["eval"])
        if not (_same(ta, tb) or (edgeZeroAllowed and _edgeZero(ta, tb))):
            out.append(f"MT{mt} crossSection")
        da, db = ma[mt].doubleDifferentialCrossSection, mb[mt].doubleDifferentialCrossSection
        fa = da["eval"] if da is not None and len(da) else None
        fb = db["eval"] if db is not None and len(db) else None
        if (fa is None) != (fb is None) or (fa is not None and type(fa) is not type(fb)):
            out.append(f"MT{mt} doubleDifferentialCrossSection")
        elif fa is not None:
            out += [f"MT{mt} {p}" for p in _PARTS
                    if not _same(_table(getattr(fa, p, None)), _table(getattr(fb, p, None)))]
    return out


# ---------------------------------------------------------------------------
# Always run
# ---------------------------------------------------------------------------

def test_relaxation_from_endf_writes_and_reads_back_the_same(tmp_path):
    pops = kika.read(str(RELAX))
    out = tmp_path / "relax.xml"
    kika.write(pops, str(out), format="gnds")
    back = kika.read(str(out))
    assert modelDifferences(pops, back, ignore=IGNORE) == []
    configuration = back.chemicalElements["Fe"].atomic["1s1/2"]
    modes = configuration.decayData.decayModes.decayModes
    assert {m.mode for m in modes} == {"atomicRelaxation"}
    assert all(p.pid in ("photon", "e-") for p in (m.decayPath.decays[0].products[0] for m in modes))


def test_photoatomic_from_endf_keeps_its_data_through_gnds(tmp_path):
    suite = kika.read(str(PHOTOAT))
    out = tmp_path / "photo.xml"
    kika.write(suite, str(out), format="gnds")
    back = kika.read(str(out))
    assert (back.projectile, back.target, back.interaction) == ("photon", "Fe", "atomic")
    assert _contentDifferences(suite, back) == []


def test_photoatomic_gnds_is_a_fixed_point(tmp_path):
    first = tmp_path / "first.xml"
    kika.write(kika.read(str(PHOTOAT)), str(first), format="gnds")
    once = kika.read(str(first))
    second = tmp_path / "second.xml"
    kika.write(once, str(second), format="gnds")
    twice = kika.read(str(second))
    for name in ("reactions", "sums"):
        assert modelDifferences(list(getattr(once, name)), list(getattr(twice, name)), ignore=IGNORE) == []
    assert modelDifferences(once.PoPs, twice.PoPs, ignore=IGNORE) == []


# ---------------------------------------------------------------------------
# The NNDC files
# ---------------------------------------------------------------------------

def _root():
    root = os.environ.get("KIKA_PHOTON_GNDS")
    return Path(root) if root else Path(__file__).resolve().parents[5] / "NuclearData" / "gnds" / "endfb81"


def _tapesRoot():
    return _root().parents[1]


def _pairs():
    root = _root()
    files = sorted((root / "photoat").glob("*.gnds.xml")) + sorted((root / "atomic_relax").glob("*.gnds.xml"))
    return files


@pytest.mark.parametrize("path", _pairs() or [None], ids=lambda p: p.name if p else "none")
def test_nndc_gnds_writes_and_reads_back_the_same(path, tmp_path):
    if path is None:
        pytest.skip("no NNDC photon GNDS ($KIKA_PHOTON_GNDS or ../NuclearData/gnds/endfb81)")
    first = kika.read(str(path))
    out = tmp_path / path.name
    kika.write(first, str(out), format="gnds")
    back = kika.read(str(out))
    if hasattr(first, "reactions"):
        for name in ("reactions", "sums"):
            assert modelDifferences(list(getattr(first, name)), list(getattr(back, name)), ignore=IGNORE) == []
        assert modelDifferences(first.PoPs, back.PoPs, ignore=IGNORE) == []
    else:
        assert modelDifferences(first, back, ignore=IGNORE) == []


@pytest.mark.parametrize("path", _pairs() or [None], ids=lambda p: p.name if p else "none")
def test_endf_and_nndc_gnds_agree(path):
    if path is None:
        pytest.skip("no NNDC photon GNDS ($KIKA_PHOTON_GNDS or ../NuclearData/gnds/endfb81)")
    sublibrary = path.parent.name
    tape = _tapesRoot() / sublibrary / "endfb81" / path.name.replace(".gnds.xml", "")
    if not tape.exists():
        pytest.skip(f"no ENDF tape {tape}")
    gnds, endf = kika.read(str(path)), kika.read(str(tape))
    if sublibrary == "atomic_relax":
        a = next(iter(endf.chemicalElements.values())).atomic.configurations
        b = next(iter(gnds.chemicalElements.values())).atomic.configurations
        assert [c.subshell for c in a] == [c.subshell for c in b]
        for x, y in zip(a, b):
            assert (x.electronNumber, x.bindingEnergy.value) == (y.electronNumber, y.bindingEnergy.value)
            mx = x.decayData.decayModes.decayModes if x.decayData else []
            my = y.decayData.decayModes.decayModes if y.decayData else []
            assert [(m.mode, m.probability, [p.pid for p in m.decayPath.decays[0].products]) for m in mx] == \
                   [(m.mode, m.probability, [p.pid for p in m.decayPath.decays[0].products]) for m in my]
        return
    differences = [d for d in _contentDifferences(endf, gnds, edgeZeroAllowed=True)
                   if not d.endswith("AnomalousFactor")]
    assert differences == []
    # the anomalous factors against the tape's own sections (ENDF-6 §27)
    from kika.endf.read_endf import read_endf

    mf27 = read_endf(str(tape)).mf[27].mt
    coherent = _byMt(gnds)[502].doubleDifferentialCrossSection["eval"]
    for mt, part in ((506, "realAnomalousFactor"), (505, "imaginaryAnomalousFactor")):
        if mt in mf27:
            x, y, _ = _table(getattr(coherent, part))
            np.testing.assert_array_equal(x, mf27[mt].x)
            np.testing.assert_array_equal(y, mf27[mt].y)
