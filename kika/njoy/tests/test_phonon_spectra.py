"""The phonon spectra of a TSL evaluation, read as LEAPR's input (roadmap E4-r2).

The fixtures are copies of four inputs ENDF/B-VIII.1 distributes beside its TSL
tapes (``NuclearData/tsl/endfb81``), one per shape the reader has to take:

- ``tsl-026_Fe_056.leapr``: an NJOY deck around LEAPR, ZA written ``26056.``
  and the alpha grid over several records with no ``/``;
- ``tsl-SiO2-beta.leapr``: the module name quoted, a title card with no ``/``,
  and a secondary scatterer (``nss=1, b7=0``) with its own spectra;
- ``tsl-7Liin7LiD-mixed-DOS.txt``: FLASSH, rows labelled ``/DOS k``, followed by
  the energy grid it was evaluated on;
- ``tsl-ZrinZrC_flassh-DOS.txt``: FLASSH, rows labelled by atom (``/2Zr1``) and
  a grid specification card after them.

Over the whole of ENDF/B-VIII.1 and VIII.0 (213 inputs) the reader gives 1 332
spectra; the two NJOY+NCrystal decks (CF2, iel=99) are refused by name.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.njoy.phonon_spectra import phononSpectra

DATA = Path(__file__).parent / "data"


def test_a_leapr_deck_gives_one_spectrum_per_temperature():
    spectra = phononSpectra(DATA / "tsl-026_Fe_056.leapr")
    assert [s.temperature for s in spectra] == [20.0, 80.0, 293.6, 400.0, 600.0, 800.0]
    first = spectra[0].spectrum
    assert first.xs[1] == pytest.approx(0.0005) and first.xs[0] == 0.0
    assert len(first.xs) == 79 and first.axes is not None
    assert spectra[0].extra == {"mat": 101, "za": 26056}


def test_the_secondary_scatterer_has_its_own_spectra():
    spectra = phononSpectra(DATA / "tsl-SiO2-beta.leapr")
    atoms = {s.atom for s in spectra}
    assert atoms == {"principal", "secondary"}
    principal = [s for s in spectra if s.atom == "principal"]
    secondary = [s for s in spectra if s.atom == "secondary"]
    assert len(principal) == len(secondary)
    assert not np.array_equal(principal[0].spectrum.ys, secondary[0].spectrum.ys)


@pytest.mark.parametrize("name, atoms", [
    ("tsl-7Liin7LiD-mixed-DOS.txt", ["DOS 1", "DOS 2"]),
    ("tsl-ZrinZrC_flassh-DOS.txt", ["2Zr1", "1C1"]),
])
def test_a_flassh_dos_gives_one_spectrum_per_atom(name, atoms):
    spectra = phononSpectra(DATA / name)
    assert [s.atom for s in spectra] == atoms
    assert all(s.temperature is None for s in spectra)
    assert all(s.spectrum.ys[0] == 0.0 for s in spectra)


def test_an_ncrystal_deck_is_refused_by_name(tmp_path):
    deck = tmp_path / "tsl-CinCF2.leapr"
    deck.write_text("leapr\n20\n'C in CF2' /\n12 1 100 /\n1051 6000 0 0 1.00e-300 -600./\n"
                    "11.907856 4.724172 1 99 0 0 /\n0 /\n'ORNL_CF2.ncmat' /\n")
    with pytest.raises(ValueError, match="NCrystal"):
        phononSpectra(deck)
