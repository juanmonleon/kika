"""``kika.identify()`` — every format kika reads, told apart by content.

The covariance files are written by kika's own writers, so a change in what a
writer emits that stops the identifier recognising it fails here. The Serpent
and NJOY samples are short, hand-written heads in the shape their readers'
regexes require; ``kika/serpent`` ships no output files to test against.

Each sample is also checked against every *other* kind in the table, which is
the property that matters: a heuristic that recognises its own format but also
claims a neighbour's is worse than none.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import kika
from kika._identify import KINDS, identify
from kika._read import UnknownFormatError

HERE = Path(__file__).resolve().parent
KIKA = HERE.parent


SERPENT_RES = """\

% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
"""

SERPENT_SENS = """\
% Number of different perturbed materials

SENS_N_MAT = 1;

% List of perturbed materials

SENS_MAT_LIST = [
'total'
];
"""

SERPENT_DET = """\

DET1                      =  [
    1    1    1    1    1    1    1    1    1    1    5.53000E-01 0.00813
];

DET1E                      = [
  1.00000E-11  1.00000E-01  5.00000E-02
];
"""

SERPENT_HIS = """\

% Cycle-wise history:

HIS_TIME = [
  1.00000E+00 2.00000E+00
];
"""

SERPENT_DEP = """\

% Material compositions (2.1.32 -- Mon Jan  1 00:00:00 2024)

BU = [  0.00000E+00  1.00000E-01 ];
DAYS = [  0.00000E+00  3.46010E+00 ];

ZAI = [
   922350
   922380
   0
];
"""

SERPENT_INPUT = """\
/* A pin cell */
set title "pin"
% materials
mat fuel -10.4
 92235.09c 0.03
 92238.09c 0.97
surf 1 cyl 0.0 0.0 0.4
cell 1 0 fuel -1
cell 2 0 outside 1
set bc 2
"""

MCNP_INPUT = """\
Bare sphere
1 1 -18.7 -1 imp:n=1
2 0 1 imp:n=0

1 so 8.74

m1 92235.80c 1.0
kcode 10000 1.0 50 250
ksrc 0 0 0
"""

NJOY_DECK = """\
-- a two-step deck
moder
 20 -21/
reconr
 -21 -22/
 'pendf tape'/
 1306 3/
 .005/
 0/
stop
"""

CSV = """\
sample,keff,keff_unc
1,1.0012,0.0003
2,0.9987,0.0003
3,1.0004,0.0003
"""


def _covariance():
    from kika.cov.cross_section_covariance import CrossSectionCovariance
    cov = CrossSectionCovariance(
        num_groups=4, energy_grid=[1.0e-5, 1.0, 1.0e3, 1.0e6, 2.0e7],
        energy_unit="eV", metadata={"awr": 55.45443, "temperature": 293.6})
    cov.add_matrix(26056, 2, 26056, 2, np.diag([0.01, 0.02, 0.03, 0.04]),
                   is_relative=True)
    cov.cross_sections[(26056, 2)] = np.array([1.0, 2.0, 3.0, 4.0])
    return cov


@pytest.fixture(scope="module")
def samples(tmp_path_factory, micro_tape, micro_cov_tape, h2_gnds):
    """kind -> a path holding one file of that kind (or a list of them)."""
    from kika.cov import write_boxer, write_covfil, write_coverx

    d = tmp_path_factory.mktemp("identify")
    cov = _covariance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # precision notices from the writers
        write_covfil(cov, str(d / "covfil.txt"))
        write_boxer(cov, str(d / "boxer.txt"))
        write_coverx(cov, str(d / "coverx.bin"))
        write_coverx(cov, str(d / "coverx.txt"), fmt="text")

    texts = {
        "serpent-res": SERPENT_RES, "serpent-sens": SERPENT_SENS,
        "serpent-det": SERPENT_DET, "serpent-his": SERPENT_HIS,
        "serpent-dep": SERPENT_DEP, "serpent-input": SERPENT_INPUT,
        "mcnp-input": MCNP_INPUT, "njoy-input": NJOY_DECK, "csv": CSV,
    }
    out = {kind: [d / f"{kind}.dat"] for kind in texts}
    for kind, text in texts.items():
        out[kind][0].write_text(text, encoding="utf-8")

    sdf = KIKA / "sensitivities" / "tests" / "data" / "sdf"
    mcnp = KIKA / "mcnp" / "tests" / "data"
    out.update({
        "endf": [micro_tape, micro_cov_tape],
        "gnds": [h2_gnds],
        "covfil": [d / "covfil.txt"],
        "boxer": [d / "boxer.txt"],
        "coverx": [d / "coverx.bin", d / "coverx.txt"],
        "sdf": sorted(sdf.glob("*.sdf")),
        "mcnp-mctal": sorted((mcnp / "mctal").glob("*.m")),
    })
    out["mcnp-input"] += [mcnp / "input" / "inputfile_test_1.i",
                          *sorted(sdf.glob("pertfile_*_PERT.i"))]
    return out


def _cases():
    return [k for k in KINDS if k not in ("ace", "g4ndl")]


@pytest.mark.parametrize("kind", _cases())
def test_each_kind_is_recognised_as_itself(samples, kind):
    assert samples[kind], f"no sample for {kind}"
    for path in samples[kind]:
        assert identify(path) == kind, path


def test_an_ace_file_is_recognised(fe56_ace):
    assert identify(fe56_ace) == "ace"


def test_a_g4ndl_directory_goes_through_sniff_format(tmp_path):
    (tmp_path / "Elastic" / "CrossSection").mkdir(parents=True)
    (tmp_path / "Elastic" / "FS").mkdir()
    assert identify(tmp_path) == "g4ndl"


def test_identify_is_on_the_package_without_waking_the_model():
    import subprocess, sys
    code = ("import sys, kika; kika.identify; "
            "assert 'kika.nuclear_data.model' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
    assert kika.identify is identify


def test_the_identifier_agrees_with_sniff_format_where_both_answer(samples):
    for kind in ("endf", "gnds"):
        for path in samples[kind]:
            assert kika.sniff_format(path) == identify(path)


def test_an_unrecognised_text_file_says_what_was_tried(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("just some notes\nnothing to see\n", encoding="utf-8")
    with pytest.raises(UnknownFormatError, match="Checked:"):
        identify(path)


def test_a_binary_file_that_is_not_coverx_is_refused(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8)
    with pytest.raises(UnknownFormatError, match="COVERX"):
        identify(path)


def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(UnknownFormatError, match="empty"):
        identify(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        identify(tmp_path / "nope")


def test_a_groupr_tape_is_named_rather_than_called_covfil(samples, tmp_path):
    """GENDF from groupr: grouped like a COVFIL, but MF6 where MF33 would be."""
    lines = samples["covfil"][0].read_text().splitlines()
    swapped = [ln[:70] + " 6" + ln[72:] if len(ln) >= 75 and ln[70:72] == "33" else ln
               for ln in lines]
    path = tmp_path / "gendf"
    path.write_text("\n".join(swapped) + "\n")
    with pytest.raises(UnknownFormatError, match="GENDF"):
        identify(path)


# ----------------------------------------------------------------------
# Cases the first scan of a real workspace got wrong
# ----------------------------------------------------------------------

def test_a_thermal_ace_table_is_refused_by_name(tmp_path):
    """It was called MCNP input: its numeric cards look like cells."""
    path = tmp_path / "referenceTape71"
    path.write_text("  hh2o.90t    0.999167  2.5300E-08   07/19/20\n"
                    " 1 2 3 4 5 6 7 8\n 1 2 3 4 5 6 7 8\n", encoding="utf-8")
    with pytest.raises(UnknownFormatError, match="thermal-scattering ACE"):
        identify(path)


def test_errorr_mf35_output_is_refused_with_the_reason(samples, tmp_path):
    lines = samples["covfil"][0].read_text().splitlines()
    swapped = [ln[:70] + "35" + ln[72:] if len(ln) >= 75 and ln[70:72] == "33" else ln
               for ln in lines]
    path = tmp_path / "errorr35"
    path.write_text("\n".join(swapped) + "\n")
    with pytest.raises(UnknownFormatError, match="MF35"):
        identify(path)


def test_an_mcnp_file_of_materials_alone_is_mcnp_input(tmp_path):
    path = tmp_path / "perturbed.i"
    path.write_text("c generated\nc\nm300000 nlib=06c\n    26056 0.9\n    26054 0.1\n"
                    "c perturbed\nm42 nlib=06c\n    26056 1.0\n", encoding="utf-8")
    assert identify(path) == "mcnp-input"


@pytest.mark.parametrize("text", [
    "1 2 .30  2/\n1 0 1. 1.  2.6 4.0 5.0 3.3 0.0/\n'title'/\n/\n3/\n1.0E+02 2.0E+07/\n",
    "all: build\ninclude common.mk\n%.o: %.c\n\tcc -c $<\nset -e\n",
    "#!/bin/bash\n# Include the deck\nset -e\ncat > input <<EOF\n",
])
def test_slash_cards_makefiles_and_scripts_are_not_input_decks(tmp_path, text):
    path = tmp_path / "f"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UnknownFormatError):
        identify(path)


def test_xml_that_is_not_a_gnds_root_is_not_gnds(tmp_path):
    path = tmp_path / "schema.xsd"
    path.write_text('<?xml version="1.0"?>\n<xs:schema xmlns:xs="x"></xs:schema>\n')
    with pytest.raises(UnknownFormatError):
        identify(path)


def test_comma_separated_numbers_are_csv_not_boxer_or_endf(tmp_path):
    rows = ["exp_key,offset,lo,hi,level_rms,n,sigma_norm,author,tau"] + [
        f"1033200{i},-0.013082754521073776,-0.013082754521073776,"
        f"-0.013082754521073775,0.2021811,{i},0.1,Smith,{i}" for i in range(6)]
    path = tmp_path / "offsets.csv"
    path.write_text("\n".join(rows) + "\n")
    assert identify(path) == "csv"


def test_a_serpent_output_listing_is_refused_rather_than_called_mcnp(tmp_path):
    """The run's text log, whose tables read as MCNP cells (all 59 NEA examples)."""
    path = tmp_path / "bwr.out"
    path.write_text(
        "\n --- Table  1: Summary of nuclide data: \n\n"
        " Data for 19 nuclides included in calculation:\n\n"
        "   1  0  1001.03c  1  TRA  1  1  0  0.99917  293.6\n"
        "   2  0  8016.03c  1  TRA  8  16  0  15.8575  293.6\n"
        "   3  0  92235.03c  1  TRA  92  235  0  233.025  293.6\n",
        encoding="utf-8")
    with pytest.raises(UnknownFormatError, match="Serpent output listing"):
        identify(path)
