"""Photo-atomic ACE tables (``.84p``, ``.14p``): the decoder against the manual.

The synthetic tables are laid out word by word from Hughes, LA-UR-13-27377
(Tables 1-4), in both layouts: MCPLIB (NXS(6)=0, form factors on MCNP's
fixed grids) and EPRDATA (NXS(6)>0, grids in the file, subshell and
relaxation blocks). Each array is written from a known value, so the test
checks the decoder put every word where the manual says it belongs, with the
logs undone and MeV turned into eV.

The real table, ``26000.14p`` from LANL's eprdata14, is looked for in
``$KIKA_PHOTOAT/eprdata14`` or ``../kika-files/photoat/eprdata14`` next to
the repo, and compared against ENDF/B-VIII.0's Fe photo-atomic tape when that
is there too.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.classes.ace import Ace
from kika.ace.classes.photoatomic import PhotoatomicAce
from kika.ace.parsers.parse_photoatomic import (
    MCPLIB_COHERENT_X,
    MCPLIB_INCOHERENT_X,
    is_photoatomic_extension,
)
from kika.ace.tests._synthetic import write_table as write_neutron_table

E_MEV = [1.0e-3, 1.0e-2, 1.0, 100.0]
INC = [0.5, 2.0, 3.0, 0.4]
# Not 1.0 anywhere: ln(1 b) = 0.0 is the stored zero, so ACE cannot write
# exactly one barn (LA-UR-13-27377, Note 1) and neither can this table.
COH = [9.0, 1.2, 0.03, 1.0e-5]
PE = [800.0, 60.0, 0.02, 1.0e-6]
PAIR = [0.0, 0.0, 0.0, 4.5]          # zero below threshold: stored as 0.0, not a log
HEAT_MEV = [1.0e-3, 9.0e-3, 0.5, 90.0]
FLO = {"edge": [7.112e-3, 7.112e-3], "phi": [0.3, 1.0],
       "yield": [0.1, 0.25], "line": [6.4e-3, 7.06e-3]}
SHELL_EL = [2.0, 8.0]
SHELL_BE = [7.1e-3, 7.2e-4]
SHELL_PR = [0.2, 1.0]
PROFILES = [  # (JJ, pz, pdf, cdf)
    (2, [0.0, 1.0, 10.0], [1.0, 0.5, 0.0], [0.0, 0.8, 1.0]),
    (1, [0.0, 2.0, 20.0], [2.0, 0.1, 0.0], [0.0, 0.9, 1.0]),
]
SUB_DESIG = [1, 2]
SUB_EL = [2.0, 2.0]
SUB_BE = [7.117e-3, 8.51e-4]
SUB_CV = [0.1, 0.2]
SUB_PE = {1: [0.0, 50.0, 0.015, 1.0e-6], 2: [700.0, 9.0, 0.004, 1.0e-7]}
XPROB = {1: [(3, 0, 6.40e-3, 0.3), (2, 2, 5.4e-3, 1.0)], 2: [(4, 4, 7.0e-4, 1.0)]}


def _ln(values):
    return [math.log(v) if v != 0.0 else 0.0 for v in values]


def build_photoatomic(layout: str):
    """Return ``(nxs, jxs, xss)`` for a Z=26 table; ``xss`` 1-based."""
    nes = len(E_MEV)
    xss, jxs, nxs = [0.0], [0] * 33, [0] * 17

    def here():
        return len(xss)

    jxs[1] = here()
    xss += _ln(E_MEV) + _ln(INC) + _ln(COH) + _ln(PE) + _ln(PAIR)
    jxs[2] = here()
    if layout == "mcplib":
        xss += list(np.linspace(0.0, 26.0, MCPLIB_INCOHERENT_X.size))
        jxs[3] = here()
        xss += list(np.linspace(0.0, 30.0, MCPLIB_COHERENT_X.size))
        xss += list(np.linspace(26.0, 0.1, MCPLIB_COHERENT_X.size))
    else:
        xss += [0.0, 1.0, 1.0e9] + [0.0, 13.0, 26.0]
        jxs[3] = here()
        xss += [0.0, 0.5, 2.0, 1.0e9] + [0.0, 1.0, 2.0, 2.5] + [26.0, 20.0, 5.0, 0.0]
    jxs[4] = here()
    xss += FLO["edge"] + FLO["phi"] + FLO["yield"] + FLO["line"]
    jxs[5] = here()
    xss += HEAT_MEV
    jxs[6] = here()
    xss += SHELL_EL
    jxs[7] = here()
    xss += SHELL_BE
    jxs[8] = here()
    xss += SHELL_PR
    jxs[9] = here()
    swd, locs = [], []
    for jj, pz, pdf, cdf in PROFILES:
        locs.append(len(swd) + 1)
        swd += [jj, len(pz)] + pz + pdf + cdf
    xss += locs
    jxs[10] = here()
    xss += swd

    if layout == "eprdata":
        ns = len(SUB_DESIG)
        jxs[11] = here()
        xss += SUB_DESIG + SUB_EL + SUB_BE + SUB_CV + [len(XPROB[d]) for d in SUB_DESIG]
        jxs[12], jxs[13], jxs[14], jxs[15] = (jxs[11] + k * ns for k in range(1, 5))
        jxs[16] = here()
        for d in SUB_DESIG:
            xss += _ln(SUB_PE[d])
        jxs[17] = here()
        offs, words = [], []
        for d in SUB_DESIG:
            offs.append(len(words))
            for t in XPROB[d]:
                words += list(t)
        xss += offs
        jxs[18] = here()
        xss += words
        jxs[19] = here()                 # ESZE: one electron energy, six zeros
        xss += [1.0e-3] + [0.0] * (5 + ns)
        nxs[7], nxs[8] = ns, 1

    nxs[1] = len(xss) - 1
    nxs[2] = 26
    nxs[3] = nes
    nxs[4] = len(FLO["edge"])
    nxs[5] = len(PROFILES)
    nxs[6] = 0 if layout == "mcplib" else 3
    return nxs, jxs, xss


def write_photoatomic(path: Path, layout: str, zaid: str = "26000.84p",
                      nxs=None, jxs=None, xss=None) -> Path:
    """Write a legacy-header ACE file (Table 1 of the manual) from the arrays."""
    if nxs is None:
        nxs, jxs, xss = build_photoatomic(layout)
    lines = [f"{zaid:>10}{55.367:12.6f}{0.0:12.4E} {'10/10/26':<10}",
             f"{'synthetic photo-atomic table for kika tests':<70}{'mat2600':>10}"]
    for _ in range(4):
        lines.append(f"{0:7d}{0.0:11.0f}" * 4)
    for k in range(1, 17, 8):
        lines.append("".join(f"{v:9d}" for v in nxs[k:k + 8]))
    for k in range(1, 33, 8):
        lines.append("".join(f"{v:9d}" for v in jxs[k:k + 8]))
    words = xss[1:]
    for k in range(0, len(words), 4):
        lines.append("".join(f"{float(v):20.11E}" for v in words[k:k + 4]))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(params=["mcplib", "eprdata"])
def table(request, tmp_path):
    path = write_photoatomic(tmp_path / f"fe_{request.param}.ace", request.param,
                             zaid="26000.84p" if request.param == "mcplib" else "26000.14p")
    return request.param, read_ace(str(path))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext, expected", [(".84p", True), (".14p", True), (".12P", True),
                                           (".80c", False), (".00u", False), (".24h", False),
                                           (None, False)])
def test_photoatomic_is_the_p_class(ext, expected):
    assert is_photoatomic_extension(ext) is expected


def test_a_neutron_table_still_reads_as_ace(tmp_path):
    ace = read_ace(str(write_neutron_table(tmp_path / "n.ace")))
    assert isinstance(ace, Ace)
    assert not isinstance(ace, PhotoatomicAce)


# ---------------------------------------------------------------------------
# Every block, both layouts
# ---------------------------------------------------------------------------

def test_eszg_is_exponentiated_and_in_ev(table):
    _, t = table
    assert isinstance(t, PhotoatomicAce)
    assert t.z == 26
    np.testing.assert_allclose(t.energies, np.array(E_MEV) * 1e6, rtol=1e-10)
    np.testing.assert_allclose(t.incoherent, INC, rtol=1e-10)
    np.testing.assert_allclose(t.coherent, COH, rtol=1e-10)
    np.testing.assert_allclose(t.photoelectric, PE, rtol=1e-10)
    # A stored 0.0 is a zero cross section, not ln(1 b) = 0.
    np.testing.assert_allclose(t.pair_production, PAIR, rtol=1e-10)
    assert t.pair_production[0] == 0.0
    np.testing.assert_allclose(t.heating, np.array(HEAT_MEV) * 1e6, rtol=1e-10)


def test_cross_sections_method(table):
    _, t = table
    xs = t.cross_sections()
    assert set(xs) == {"total", "coherent", "incoherent", "photoelectric", "pair_production"}
    E, total = xs["total"]
    np.testing.assert_allclose(E, np.array(E_MEV) * 1e6, rtol=1e-10)
    np.testing.assert_allclose(total, np.add.reduce([INC, COH, PE, PAIR]), rtol=1e-10)


def test_form_factors(table):
    layout, t = table
    if layout == "mcplib":
        np.testing.assert_array_equal(t.incoherent_x, MCPLIB_INCOHERENT_X)
        np.testing.assert_array_equal(t.coherent_x, MCPLIB_COHERENT_X)
        assert MCPLIB_INCOHERENT_X.size == 21 and MCPLIB_COHERENT_X.size == 55
        assert t.incoherent_scattering_function[-1] == pytest.approx(26.0)
        assert t.coherent_integrated_form_factor[-1] == pytest.approx(30.0)
        assert t.coherent_form_factor[0] == pytest.approx(26.0)
        assert t.coherent_form_factor[-1] == pytest.approx(0.1)
    else:
        np.testing.assert_allclose(t.incoherent_x, [0.0, 1.0, 1.0e9])
        np.testing.assert_allclose(t.incoherent_scattering_function, [0.0, 13.0, 26.0])
        np.testing.assert_allclose(t.coherent_x, [0.0, 0.5, 2.0, 1.0e9])
        np.testing.assert_allclose(t.coherent_integrated_form_factor, [0.0, 1.0, 2.0, 2.5])
        np.testing.assert_allclose(t.coherent_form_factor, [26.0, 20.0, 5.0, 0.0])


def test_fluorescence_and_compton(table):
    _, t = table
    np.testing.assert_allclose(t.fluorescence_edge_energies, np.array(FLO["edge"]) * 1e6)
    np.testing.assert_allclose(t.fluorescence_ejection_probabilities, FLO["phi"])
    np.testing.assert_allclose(t.fluorescence_yields, FLO["yield"])
    np.testing.assert_allclose(t.fluorescence_energies, np.array(FLO["line"]) * 1e6)
    np.testing.assert_allclose(t.shell_electrons, SHELL_EL)
    np.testing.assert_allclose(t.shell_binding_energies, np.array(SHELL_BE) * 1e6)
    np.testing.assert_allclose(t.shell_probabilities, SHELL_PR)
    assert len(t.compton_profiles) == 2
    for got, (jj, pz, pdf, cdf) in zip(t.compton_profiles, PROFILES):
        assert got.interpolation == jj
        np.testing.assert_allclose(got.pz, pz)
        np.testing.assert_allclose(got.pdf, pdf)
        np.testing.assert_allclose(got.cdf, cdf)


def test_eprdata_subshells_and_relaxation(table):
    layout, t = table
    if layout == "mcplib":
        assert t.subshell_photoelectric == {} and t.relaxation == {}
        assert t.unparsed_blocks == []
        return
    np.testing.assert_array_equal(t.subshell_designators, SUB_DESIG)
    np.testing.assert_allclose(t.subshell_electrons, SUB_EL)
    np.testing.assert_allclose(t.subshell_binding_energies, np.array(SUB_BE) * 1e6)
    np.testing.assert_allclose(t.subshell_vacancy_probabilities, SUB_CV)
    np.testing.assert_array_equal(t.subshell_num_transitions, [2, 1])
    for d in SUB_DESIG:
        np.testing.assert_allclose(t.subshell_photoelectric[d], SUB_PE[d], rtol=1e-10)
    assert t.subshell_photoelectric[1][0] == 0.0           # below the K edge
    k = t.relaxation[1]
    assert [(r.designator_1, r.designator_2) for r in k] == [(3, 0), (2, 2)]
    assert k[0].energy == pytest.approx(6.4e3)
    assert k[1].cumulative_probability == 1.0
    assert [(r.designator_1, r.designator_2) for r in t.relaxation[2]] == [(4, 4)]
    xs = t.cross_sections(subshells=True)
    assert {"photoelectric_K", "photoelectric_L1"} <= set(xs)
    assert t.unparsed_blocks == ["ESZE"]


def test_kika_read_returns_the_photoatomic_table(tmp_path):
    import kika
    path = write_photoatomic(tmp_path / "26000.84p", "mcplib")
    result = kika.read(str(path))
    assert isinstance(result, PhotoatomicAce)
    assert result.z == 26


# ---------------------------------------------------------------------------
# Strictness
# ---------------------------------------------------------------------------

def test_a_pointer_past_the_table_raises(tmp_path):
    nxs, jxs, xss = build_photoatomic("mcplib")
    jxs[5] = nxs[1] - 1                       # LHNM would run past NXS(1)
    path = write_photoatomic(tmp_path / "bad.ace", "mcplib", nxs=nxs, jxs=jxs, xss=xss)
    with pytest.raises(ValueError, match="LHNM"):
        read_ace(str(path))


def test_a_jinc_of_the_wrong_length_raises(tmp_path):
    nxs, jxs, xss = build_photoatomic("mcplib")
    jxs[3] += 1
    path = write_photoatomic(tmp_path / "bad.ace", "mcplib", nxs=nxs, jxs=jxs, xss=xss)
    with pytest.raises(ValueError, match="JINC"):
        read_ace(str(path))


def test_an_eprdata_jcoh_that_does_not_split_in_three_raises(tmp_path):
    nxs, jxs, xss = build_photoatomic("eprdata")
    jxs[4] -= 1
    path = write_photoatomic(tmp_path / "bad.ace", "eprdata", nxs=nxs, jxs=jxs, xss=xss)
    with pytest.raises(ValueError, match="JCOH"):
        read_ace(str(path))


# ---------------------------------------------------------------------------
# The real table
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[3]
PHOTOAT_DIR = Path(os.environ.get("KIKA_PHOTOAT", REPO.parent / "kika-files" / "photoat"))
EPRDATA14_FE = PHOTOAT_DIR / "eprdata14" / "26000.14p"
ENDF_FE = PHOTOAT_DIR / "photo_2600_26-Fe-0.dat"


@pytest.mark.tape
def test_eprdata14_fe_agrees_with_endf_b8_photoatomic():
    """eprdata14 (EPICS2014) against ENDF/B-VIII.0 (EPICS2017): same physics, close numbers.

    Incoherent scattering was not re-evaluated between the two and agrees to
    the digits written; coherent and photoelectric moved by a few per cent at
    most, which is the evaluation and not the decoding.
    """
    if not EPRDATA14_FE.is_file():
        pytest.skip(f"{EPRDATA14_FE} not found")
    t = read_ace(str(EPRDATA14_FE))
    assert isinstance(t, PhotoatomicAce)
    assert (t.z, t.layout, t.energies.size, len(t.compton_profiles)) == (26, "eprdata", 2020, 7)
    np.testing.assert_array_equal(t.subshell_designators, np.arange(1, 11))
    assert t.subshell_binding_energies[0] == pytest.approx(7083.4)
    assert t.coherent_form_factor[0] == pytest.approx(26.0)
    assert t.incoherent_scattering_function[-1] == pytest.approx(26.0)
    if not ENDF_FE.is_file():
        return
    from kika.endf import photoatomic_cross_sections, read_endf
    endf = photoatomic_cross_sections(read_endf(str(ENDF_FE)))
    E = np.array([1e4, 1e5, 1e6, 1e7])
    pairs = {"incoherent": (504, 1e-6), "coherent": (502, 0.03), "photoelectric": (522, 0.01),
             "pair_production": (516, 1e-5), "total": (501, 0.01)}
    ace = t.cross_sections()
    for name, (mt, rtol) in pairs.items():
        np.testing.assert_allclose(np.interp(E, *ace[name]), np.interp(E, *endf[mt]),
                                   rtol=rtol, err_msg=name)


MCPLIB84_FE = PHOTOAT_DIR / "mcplib84" / "26000.84p"


@pytest.mark.tape
def test_mcplib84_fe_reads_and_agrees_with_eprdata14():
    """The MCPLIB layout on a real table: ``26000.84p`` from LANL's mcplib84,
    cut out of the library at its xsdir address.

    Until this test the MCPLIB half of the parser had only met synthetic
    tables. mcplib84 and eprdata14 both come from EPDL, so on the processes
    they share the cross sections agree to a few parts in a thousand; what is
    checked here is that the MCPLIB block layout (no subshells, no relaxation)
    is read into the same places.
    """
    if not MCPLIB84_FE.is_file():
        pytest.skip(f"{MCPLIB84_FE} not found")
    t = read_ace(str(MCPLIB84_FE))
    assert isinstance(t, PhotoatomicAce)
    assert (t.z, t.layout) == (26, "mcplib")
    assert not t.unparsed_blocks
    assert t.coherent_form_factor[0] == pytest.approx(26.0)
    assert t.heating is not None and t.heating.size == t.energies.size
    if not EPRDATA14_FE.is_file():
        return
    epr = read_ace(str(EPRDATA14_FE)).cross_sections()
    mcp = t.cross_sections()
    E = np.array([1e4, 1e5, 1e6, 1e7])
    for name in ("coherent", "incoherent", "photoelectric", "total"):
        np.testing.assert_allclose(np.interp(E, *mcp[name]), np.interp(E, *epr[name]),
                                   rtol=5e-3, err_msg=name)
