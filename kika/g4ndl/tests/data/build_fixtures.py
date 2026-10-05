"""Build the committed G4NDL fixtures. Run by hand, never by the test suite.

Two kinds of fixture, laid out as **mini libraries** — each is a directory with
``Elastic/CrossSection`` and ``Elastic/FS`` exactly as a real library has them,
so the reader under test opens them the way it opens a real one:

``JEFF-4.0/`` and ``G4NDL-4.7.1/``
    Real files, copied **byte for byte** (still zlib-compressed) from the IAEA
    JEFF-4.0 G4NDL translation (Mendoza & Cano-Ott) and from Geant4's own
    G4NDL 4.7.1. Chosen by ``kika-workspace/myworkspace/G4NDL/inventory.py`` as
    the smallest file showing each construct; ``REAL`` below says which.
    Kept small on purpose (~130 kB): Fe-56 and anything else full-size is
    read from the whole library by the ``tape``-marked tests instead
    (``KIKA_G4NDL``, see the repository ``conftest.py``).

``synthetic/``
    Hand-written token streams for what no real file shows (``repFlag=0``, the
    ``G4NDL`` header, a laboratory frame, ``tempdep != 0``) and for the inputs a
    reader must refuse. ``SYNTHETIC`` holds the text; each one carries exactly
    one property, named in ``CLAIMS``.

Usage::

    python build_fixtures.py --jeff  /path/to/JEFF-4.0 \\
                             --g4ndl /path/to/G4NDL4.7.1

Re-running is idempotent: real files are re-copied and must hash to the values
recorded in ``SHA256``; the synthetic library is rewritten from this file.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import zlib
from pathlib import Path

HERE = Path(__file__).parent

#: (library, isotope stem) -> why it is here. Both the CrossSection and the FS
#: file of each isotope are copied.
REAL = {
    ("JEFF-4.0", "1_1_Hydrogen"): "repFlag=1 (Legendre only); one repeated incident energy",
    ("JEFF-4.0", "2_3_Helium"): "repFlag=2 (tabulated only); the smallest real file",
    ("JEFF-4.0", "6_12_Carbon"): "repFlag=3; mu interpolation INT=4 (LOGLIN)",
    ("JEFF-4.0", "7_14_Nitrogen"): "repFlag=3; two energy-interpolation regions, INT=3 then INT=2",
    ("JEFF-4.0", "27_58m1_Cobalt"): "isomer name <Z>_<A>m<M>_<Element>",
    ("G4NDL-4.7.1", "6_nat_Carbon"): "natural-element name <Z>_nat_<Element>",
}

SUBDIRS = ("Elastic/CrossSection", "Elastic/FS")

#: sha256 of every committed real file, filled from the first build. A change
#: here means the source library changed, and that is news.
SHA256 = {
    "G4NDL-4.7.1/Elastic/CrossSection/6_nat_Carbon.z": "0f976388f60166171da0198006aa5274666aa1e50449d86941c847df09fb50bb",
    "G4NDL-4.7.1/Elastic/FS/6_nat_Carbon.z": "e01353eed87ac0fcb8ff16d97f9de6cde947232bf4c249efcef5d665325de9ca",
    "JEFF-4.0/Elastic/CrossSection/1_1_Hydrogen.z": "cd49090965075c4c5effac4874268999c22d0d21dc31687a31b5f51a48b3a710",
    "JEFF-4.0/Elastic/CrossSection/27_58m1_Cobalt.z": "54a4db3266b4f2810ed55098cab9edd7eaa9d8eb86901e917cb4fc7d790a046b",
    "JEFF-4.0/Elastic/CrossSection/2_3_Helium.z": "722b470617ba0c132f0c85f9a5eca96066afbcb062d8802c8e5228938160cd22",
    "JEFF-4.0/Elastic/CrossSection/6_12_Carbon.z": "92c27f2c3fcc3a499e0fa8033d2cf430eb2c0d3d364e38eacdd2450ca2c9d175",
    "JEFF-4.0/Elastic/CrossSection/7_14_Nitrogen.z": "d4b47c72d7e038d531d48d7191bf518e1d5cfe9b3e24e44450e189d3ff5d8441",
    "JEFF-4.0/Elastic/FS/1_1_Hydrogen.z": "593a3de8db65cd60e95a509fe9e11c0dbadc61e3198d34eb211e044b58aa9c0a",
    "JEFF-4.0/Elastic/FS/27_58m1_Cobalt.z": "1bc76133ffc3009dd0c64f87011af2937399512e33d891f6f9575b8a9577a97a",
    "JEFF-4.0/Elastic/FS/2_3_Helium.z": "59161930e1b39747d920e59784cbca0359bd3ffbb8588972e28a4e58377afe9c",
    "JEFF-4.0/Elastic/FS/6_12_Carbon.z": "996d24020dec7fce45e7987a3942f4ffb38f61c62d0626d7cb03630f38e8f061",
    "JEFF-4.0/Elastic/FS/7_14_Nitrogen.z": "45da433a1bf8b33916c0d06cffe27cb8ee41618f6675425979e32be8aa761f82",
}

_CS_OK = "0 0\n3\n1.0e-5 4.0 1.0e6 3.0 2.0e7 1.0\n"
_LEG_OK = """\
1 55.4544 2
2
1 2 2
0.0 1.0e-5 0 1 0.0
0.0 2.0e7 0 2 0.1 0.01
"""

#: isotope stem -> (FS text, CrossSection text). Plain text, not compressed,
#: except where SYNTHETIC_COMPRESSED says so.
SYNTHETIC = {
    # repFlag=0: the consumer reads a second frameFlag after the three fields.
    "1_1_Hydrogen": ("G4NDL synthetic\n0 0.999167 2\n2\n",
                     "G4NDL synthetic\n" + _CS_OK),
    # Laboratory frame, a non-zero temperature and tempdep=1 on one record.
    "3_6_Lithium": ("1 5.9634 1\n2\n1 2 2\n"
                    "293.6 1.0e-5 0 1 0.0\n293.6 2.0e7 1 1 0.2\n", _CS_OK),
    # Text and .z both present with different content: the .z must win.
    # The text says repFlag=1; the .z (SYNTHETIC_COMPRESSED) says repFlag=0.
    "4_9_Berylium": (_LEG_OK, _CS_OK),
    # Interpolation code 6 does not exist; Geant4 throws on it.
    "5_10_Boron": ("1 9.9269 2\n2\n1 2 6\n"
                   "0.0 1.0e-5 0 1 0.0\n0.0 2.0e7 0 1 0.1\n", _CS_OK),
    # Truncated: the second Legendre record stops after its coefficient count.
    "5_11_Boron": ("1 10.9147 2\n2\n1 2 2\n0.0 1.0e-5 0 1 0.0\n0.0 2.0e7 0 2\n",
                   _CS_OK),
    # A negative cross section.
    "6_13_Carbon": (_LEG_OK, "0 0\n3\n1.0e-5 4.0 1.0e6 -3.0 2.0e7 1.0\n"),
    # An integer field written as a float: NE = "2.0".
    "7_15_Nitrogen": ("1 14.8713 2\n2.0\n1 2 2\n"
                      "0.0 1.0e-5 0 1 0.0\n0.0 2.0e7 0 1 0.1\n", _CS_OK),
    # Tokens after the last record.
    "8_17_Oxygen": (_LEG_OK + "9 9 9\n", _CS_OK),
    # repFlag=3 whose tabulated block does not start at the Legendre end.
    "9_19_Fluorine": ("3 18.8352 2\n"
                      "2\n1 2 2\n0.0 1.0e-5 0 1 0.0\n0.0 2.0e7 0 1 0.1\n"
                      "2\n1 2 2\n"
                      "0.0 2.1e7 0 2 1 2 2 -1.0 0.5 1.0 0.5\n"
                      "0.0 1.5e8 0 2 1 2 2 -1.0 0.4 1.0 0.6\n", _CS_OK),
    # A CrossSection whose N promises more pairs than the file holds.
    "10_20_Neon": (_LEG_OK, "0 0\n4\n1.0e-5 4.0 1.0e6 3.0 2.0e7 1.0\n"),
}

#: stem -> FS text written as <stem>.z, beside the plain text above.
SYNTHETIC_COMPRESSED = {
    "4_9_Berylium": "0 8.9348 2\n2\n",
}

#: What each synthetic isotope is for. test_fixtures.py asserts each claim.
CLAIMS = {
    "1_1_Hydrogen": "repflag0_with_header",
    "3_6_Lithium": "lab_frame_tempdep",
    "4_9_Berylium": "z_and_text_disagree",
    "5_10_Boron": "unknown_interpolation",
    "5_11_Boron": "truncated_fs",
    "6_13_Carbon": "negative_cross_section",
    "7_15_Nitrogen": "integer_as_float",
    "8_17_Oxygen": "trailing_tokens",
    "9_19_Fluorine": "transition_mismatch",
    "10_20_Neon": "cs_short_count",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_real(sources: dict[str, Path]) -> dict[str, str]:
    hashes = {}
    for (lib, stem), _why in REAL.items():
        for sub in SUBDIRS:
            src = sources[lib] / sub / f"{stem}.z"
            dst = HERE / lib / sub / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            key = str(dst.relative_to(HERE))
            hashes[key] = sha256(dst)
            if SHA256.get(key) and SHA256[key] != hashes[key]:
                raise SystemExit(f"{key}: source changed, sha256 {hashes[key]}")
    return hashes


def write_synthetic() -> None:
    root = HERE / "synthetic"
    if root.exists():
        shutil.rmtree(root)
    for stem, (fs, cs) in SYNTHETIC.items():
        for sub, text in (("Elastic/FS", fs), ("Elastic/CrossSection", cs)):
            p = root / sub / stem
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="ascii")
    for stem, text in SYNTHETIC_COMPRESSED.items():
        (root / "Elastic/FS" / f"{stem}.z").write_bytes(zlib.compress(text.encode("ascii"), 9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jeff", type=Path, required=True)
    ap.add_argument("--g4ndl", type=Path, required=True)
    a = ap.parse_args()
    hashes = copy_real({"JEFF-4.0": a.jeff, "G4NDL-4.7.1": a.g4ndl})
    write_synthetic()
    for k, v in sorted(hashes.items()):
        print(f'    "{k}": "{v}",')


if __name__ == "__main__":
    main()
