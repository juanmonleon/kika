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

``inelastic/JEFF-4.0/`` and ``inelastic/G4NDL-4.7.1/``
    Real ``Inelastic/`` files, byte for byte, kept apart from the elastic
    mini libraries so that those keep testing an elastic-only library. Chosen
    by ``kika-workspace/myworkspace/G4NDL/phase10/`` as the smallest file
    showing each section type and law (``INELASTIC`` says which), plus one
    whole isotope (Pu-244 of G4NDL 4.7.1: its ``Inelastic/CrossSection`` and
    every channel, 2.3 kB) for the total and the sums, and two level schemes
    from ``Inelastic/Gammas`` (G4NDL 4.7.1 only: JEFF-4.0 ships none). ~120 kB.

``capture/G4NDL-4.7.1/``
    Real ``Capture/`` files, byte for byte: whole isotopes (the cross section
    and the one final state each has), the smallest showing each final-state
    construct (``CAPTURE`` says which). ~27 kB.

``fission/G4NDL-4.7.1/``
    Real ``Fission/`` files, byte for byte: whole isotopes (σ, ``FS`` and every
    chance file each has), the smallest showing each construct, and one
    ``Fission/FF`` alone (``FISSION`` says which). ~105 kB, 80 of them
    Fm-255's ``FS``: the smallest file with the photons and an LF=5 delayed
    spectrum.

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

#: (library, subdir under Inelastic/, file) -> why it is here.
INELASTIC = {
    ("G4NDL-4.7.1", "F18", "94_244_Plutonium.z"): "base FS: MF4 isotropic + MF5 LF=9 (not modelled)",
    ("G4NDL-4.7.1", "F23", "47_109_Silver.z"): "composite F23, the lumped MT103 alone",
    ("G4NDL-4.7.1", "F05", "80_196_Mercury.z"): "photons 12 (LO=1), 14 isotropic, 15; MF5 LF=1 with INT=1",
    ("G4NDL-4.7.1", "F05", "56_132_Barium.z"): "MF5 LF=1 modelled: MF4 + MF5 as one uncorrelated",
    ("JEFF-4.0", "F18", "97_247_Berkelium.z"): "MF6 LAW=1 LANG=1",
    ("G4NDL-4.7.1", "F04", "64_156_Gadolinium.z"): "MF4 tabulated; MF5 NK=2",
    ("JEFF-4.0", "F01", "8_18_Oxygen.z"): "the smallest F01",
    ("G4NDL-4.7.1", "F22", "13_27_Aluminum.z"): "MF6 LAW=1 LANG=2 (Kalbach-Mann)",
    ("JEFF-4.0", "F18", "79_197_Gold.z"): "MF4 Legendre; photons 13",
    ("G4NDL-4.7.1", "F26", "26_58_Iron.z"): "MF6 LAW=2",
    ("G4NDL-4.7.1", "F01", "63_151_Europium.z"): "photons 12 LO=2 (cascade)",
    ("G4NDL-4.7.1", "F24", "7_14_Nitrogen.z"): "photons 14 Legendre (LTT=1)",
    ("JEFF-4.0", "F25", "5_10_Boron.z"): "MF6 LAW=4 (recoil)",
    ("G4NDL-4.7.1", "F01", "28_64_Nickel.z"): "F01 with MT4 and partials; MF6 LAW=3",
    ("JEFF-4.0", "F02", "94_240_Plutonium.z"): "MF6 LAW=0",
    ("JEFF-4.0", "F27", "4_9_Berylium.z"): "MF6 LAW=7",
}

#: Whole isotopes: Inelastic/CrossSection and every channel file.
INELASTIC_ISOTOPES = {("G4NDL-4.7.1", "94_244_Plutonium"): "the total and its parts"}

#: Inelastic/Gammas level schemes (plain text).
GAMMAS = {("G4NDL-4.7.1", "z6.a15"): "the smallest non-empty",
          ("G4NDL-4.7.1", "z55.a120"): "levels out of order"}

#: (library, isotope stem) -> why it is here. Its Capture/CrossSection and its
#: final state (Capture/FSMF6 or Capture/FS, whichever it has) are copied.
CAPTURE = {
    ("G4NDL-4.7.1", "1_2_Hydrogen"): "FS: one photon line, isotropic, no MF15 read",
    ("G4NDL-4.7.1", "7_14_Nitrogen"): "FS: 59 lines, MF14 Legendre, MF15 spectra read",
    ("G4NDL-4.7.1", "1_1_Hydrogen"): "FSMF6: LAW=2 photon and its LAW=4 deuteron recoil",
    ("G4NDL-4.7.1", "7_15_Nitrogen"): "FSMF6: one LAW=1 photon (the shape of 507 of 508)",
}

#: (library, isotope stem) -> why it is here: every Fission/ file it has.
FISSION = {
    ("G4NDL-4.7.1", "90_230_Thorium"): "the smallest: nu total only, energy release, MF5 LF=7",
    ("G4NDL-4.7.1", "94_244_Plutonium"): "chances FC and SC, each with its own MF4 + MF5",
    ("G4NDL-4.7.1", "88_223_Radium"): "one delayed family, MF4 tabulated (repFlag=2)",
    ("G4NDL-4.7.1", "100_255_Fermium"): "photons 12/14/15; six LF=5 delayed spectra; four chances",
}
#: (library, file) -> why: a Fission/FF without the rest of its isotope.
FISSION_FF = {
    ("G4NDL-4.7.1", "90_227_Thorium.z"): "the smallest FF: MF8/454 and 459, one energy each",
}

#: sha256 of every committed real file, filled from the first build. A change
#: here means the source library changed, and that is news.
SHA256 = {
    "fission/G4NDL-4.7.1/Fission/CrossSection/100_255_Fermium.z": "9919b4b1a0c4b4e0cb7896b0f38a807dc97737c84bc4ac9da8f9c27ff1e66aca",
    "fission/G4NDL-4.7.1/Fission/CrossSection/88_223_Radium.z": "e500108deb325889aebdc9b74aefea97264692666a397f417ed875beaaa2ca57",
    "fission/G4NDL-4.7.1/Fission/CrossSection/90_230_Thorium.z": "70312086f0a9cdf9c083249e1cbce4620fe6eb89edcfc6014f1f175e2d8d2826",
    "fission/G4NDL-4.7.1/Fission/CrossSection/94_244_Plutonium.z": "edb5bc5acc8045ebe721b41083e132b9efdc64198f1b1b8e617098fafbc266c2",
    "fission/G4NDL-4.7.1/Fission/FC/100_255_Fermium.z": "092a544173623b70ae51d57a27006b9c3cd634ed9037aa831f2edc54cbf1971b",
    "fission/G4NDL-4.7.1/Fission/FC/94_244_Plutonium.z": "23d4798ac1bbbc92d9cf0b5bac147aa6c0a6336115ecc815fb994ba125d2620c",
    "fission/G4NDL-4.7.1/Fission/FF/90_227_Thorium.z": "1f58b8adebd60e0aefbcd787bb210804b36bbdff8c696830b53259015dddcdd8",
    "fission/G4NDL-4.7.1/Fission/FS/100_255_Fermium.z": "16186be9af90beed2c72377dcb803cc8c4bd001b66f11eb47ff3b907370b1689",
    "fission/G4NDL-4.7.1/Fission/FS/88_223_Radium.z": "46d4d4b55645538e1abbeede2954f0d7d5fc9831bf56b8ac2c9841a3e443460b",
    "fission/G4NDL-4.7.1/Fission/FS/90_230_Thorium.z": "7a2bc6fbeacc950c425d1aa5980f3486c0f55d45f117b02b415da9e544e3e159",
    "fission/G4NDL-4.7.1/Fission/FS/94_244_Plutonium.z": "def5d24c04ecb7cd22dd1d4752ee3b768255f451fd486e343716b2329c466deb",
    "fission/G4NDL-4.7.1/Fission/LC/100_255_Fermium.z": "bf3bbc006532dee3ff8f16e7cb28ab0bfe32760dc7c9d382af27a4c5f8423ed0",
    "fission/G4NDL-4.7.1/Fission/SC/100_255_Fermium.z": "91c79e6eae4629724fd70c013e4223a7e543488755a072da6dcea0f0926c7604",
    "fission/G4NDL-4.7.1/Fission/SC/94_244_Plutonium.z": "3a5f61604effeea6679325a08f62d81b668aef241acf9108673f5d104140cd40",
    "fission/G4NDL-4.7.1/Fission/TC/100_255_Fermium.z": "06663170009babb3d52816aa41ee12bde1302969c6805faff6a1ac8306cb1bb6",
    "capture/G4NDL-4.7.1/Capture/CrossSection/1_1_Hydrogen.z": "4593a3fb885eacd3dc71eadb200e6a98e34d9d66680663a5f8497413a4001a88",
    "capture/G4NDL-4.7.1/Capture/CrossSection/1_2_Hydrogen.z": "e40c5a72fedd83643107c3de866be1538b5048c0d84ddca366e5bdd48422f786",
    "capture/G4NDL-4.7.1/Capture/CrossSection/7_14_Nitrogen.z": "682cb37a9e557a484e0087fbe8ae863f732e1b832d7c59854c3335c62b16b871",
    "capture/G4NDL-4.7.1/Capture/CrossSection/7_15_Nitrogen.z": "ad6f759338459dd37ccaa30003e3915032f958730fdda155e39c94a2c9012060",
    "capture/G4NDL-4.7.1/Capture/FS/1_2_Hydrogen.z": "77e78d5fecff23fc6cde05842c83c1270bf473ea531bf2cfd874ef99b66466f2",
    "capture/G4NDL-4.7.1/Capture/FS/7_14_Nitrogen.z": "39f871220ce7a47ac9ddc079293d73588e24a46daef10f3635851dd3ea255638",
    "capture/G4NDL-4.7.1/Capture/FSMF6/1_1_Hydrogen.z": "a9d64186446ba312b2ed845d063794db710054ff0f6f78918039fc067294546e",
    "capture/G4NDL-4.7.1/Capture/FSMF6/7_15_Nitrogen.z": "b9d023deb7df0088ea11f60bf438ec0b673ad499c5200aa74b572c82be735c85",
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
    "inelastic/G4NDL-4.7.1/Inelastic/CrossSection/94_244_Plutonium.z": "afda8ff6dd4ecc7ec69baa5278abae9b62c594b06ff6eae87a0314e572c1bf93",
    "inelastic/G4NDL-4.7.1/Inelastic/F01/28_64_Nickel.z": "200002676a4e04223373e1fec59095739e9b6f75ed09aff338c4ae4b4eb0202b",
    "inelastic/G4NDL-4.7.1/Inelastic/F01/63_151_Europium.z": "213f2fef0f3321ff557621229d23064bd9d42e51569455872cb022a32810b2de",
    "inelastic/G4NDL-4.7.1/Inelastic/F01/94_244_Plutonium.z": "1e694b9838a5532f9f7b10f068e83bc58262699e86e43518a668f41c1359791d",
    "inelastic/G4NDL-4.7.1/Inelastic/F04/64_156_Gadolinium.z": "c5841490795dce66312015c7fe74e4fd369681c979ab8e8425215c0ceb5e9383",
    "inelastic/G4NDL-4.7.1/Inelastic/F04/94_244_Plutonium.z": "3484115c3af622f14411645bcb07b64c767c0e0419b1885a78eee9d56f27c260",
    "inelastic/G4NDL-4.7.1/Inelastic/F05/56_132_Barium.z": "dd4ceb21afbd3e00d42c40786ac52f5ef079d16c3f99e47f780f4e505fd2bfae",
    "inelastic/G4NDL-4.7.1/Inelastic/F05/80_196_Mercury.z": "147b916b3ccaa77da9ab51caa90e4942301457554330cc048c49b81ecccc3a62",
    "inelastic/G4NDL-4.7.1/Inelastic/F05/94_244_Plutonium.z": "86af1d426f28267d6112e9917eb14cedb41df0c2608fb6c4874cedb5b00f3660",
    "inelastic/G4NDL-4.7.1/Inelastic/F18/94_244_Plutonium.z": "ea82a8b7c184b301a200138ab75cfc3e30b2190de1ef4976d054e7b80d5da24a",
    "inelastic/G4NDL-4.7.1/Inelastic/F22/13_27_Aluminum.z": "4d8fe0b2deaa631b17bceaf51d007478f43a65bb24ef6fbe2451d5e7c1e809aa",
    "inelastic/G4NDL-4.7.1/Inelastic/F23/47_109_Silver.z": "443f7c54d715328d11771c3690c6cb45ac47d278f543a3464155ebf46a61eeff",
    "inelastic/G4NDL-4.7.1/Inelastic/F24/7_14_Nitrogen.z": "0d68670af182125c2027168fa53036e017383466ada71b039d932f65dd8ee148",
    "inelastic/G4NDL-4.7.1/Inelastic/F26/26_58_Iron.z": "76fa6f229601f65f60167060c170333d9e259da45a306f635a2445b8e74277e1",
    "inelastic/G4NDL-4.7.1/Inelastic/Gammas/z55.a120": "c475283b46d01df233d2b95398efcd03b54c50bdbe350b0f0e6f4ee84401c239",
    "inelastic/G4NDL-4.7.1/Inelastic/Gammas/z6.a15": "fec14ab141b545fa360042678c3fb1617172629b9136dcec5189f751d016e311",
    "inelastic/JEFF-4.0/Inelastic/F01/8_18_Oxygen.z": "75a6fb6f7fa838193def79d40f767190c9476412a1d0318cd5dcc446c6ceef21",
    "inelastic/JEFF-4.0/Inelastic/F02/94_240_Plutonium.z": "5b00d43081a069d3a1e2e9bdfc3f79b81aeedf3c4266cadd6844ad170a83d634",
    "inelastic/JEFF-4.0/Inelastic/F18/79_197_Gold.z": "a0ffbda3fea74f4d76aa0cf93ef12bd1c77f7fc0ff1dbfd1f31f53ca12cb1d1d",
    "inelastic/JEFF-4.0/Inelastic/F18/97_247_Berkelium.z": "fbbe473d2335298074132e6a6ca1f68dd93d7aaf1b17a8a1506bbb4047712903",
    "inelastic/JEFF-4.0/Inelastic/F25/5_10_Boron.z": "96c54417a24ed30da53f06450ac03f3bd42a3243778979f2c197c7058bd17032",
    "inelastic/JEFF-4.0/Inelastic/F27/4_9_Berylium.z": "a51160244e93fb66e9d01919f3ebdedc9b997c13c8793b10fad63adba1318814",
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


def copy_inelastic(sources: dict[str, Path]) -> dict[str, str]:
    files = [(lib, f"Inelastic/{sub}/{name}") for (lib, sub, name) in INELASTIC]
    for (lib, stem) in INELASTIC_ISOTOPES:
        for d in sorted((sources[lib] / "Inelastic").iterdir()):
            if d.name != "Gammas" and (d / f"{stem}.z").is_file():
                files.append((lib, f"Inelastic/{d.name}/{stem}.z"))
    files += [(lib, f"Inelastic/Gammas/{name}") for (lib, name) in GAMMAS]
    hashes = {}
    for lib, rel in files:
        dst = HERE / "inelastic" / lib / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[lib] / rel, dst)
        key = str(dst.relative_to(HERE).as_posix())
        hashes[key] = sha256(dst)
        if SHA256.get(key) and SHA256[key] != hashes[key]:
            raise SystemExit(f"{key}: source changed, sha256 {hashes[key]}")
    return hashes


def copy_capture(sources: dict[str, Path]) -> dict[str, str]:
    hashes = {}
    for (lib, stem) in CAPTURE:
        for sub in ("CrossSection", "FSMF6", "FS"):
            src = sources[lib] / "Capture" / sub / f"{stem}.z"
            if not src.is_file():
                continue
            dst = HERE / "capture" / lib / "Capture" / sub / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            key = str(dst.relative_to(HERE).as_posix())
            hashes[key] = sha256(dst)
            if SHA256.get(key) and SHA256[key] != hashes[key]:
                raise SystemExit(f"{key}: source changed, sha256 {hashes[key]}")
    return hashes


def copy_fission(sources: dict[str, Path]) -> dict[str, str]:
    files = [(lib, f"Fission/FF/{name}") for (lib, name) in FISSION_FF]
    for (lib, stem) in FISSION:
        for d in sorted((sources[lib] / "Fission").iterdir()):
            if (d / f"{stem}.z").is_file():
                files.append((lib, f"Fission/{d.name}/{stem}.z"))
    hashes = {}
    for lib, rel in files:
        dst = HERE / "fission" / lib / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[lib] / rel, dst)
        key = str(dst.relative_to(HERE).as_posix())
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
    sources = {"JEFF-4.0": a.jeff, "G4NDL-4.7.1": a.g4ndl}
    hashes = copy_real(sources)
    hashes.update(copy_inelastic(sources))
    hashes.update(copy_capture(sources))
    hashes.update(copy_fission(sources))
    write_synthetic()
    for k, v in sorted(hashes.items()):
        print(f'    "{k}": "{v}",')


if __name__ == "__main__":
    main()
