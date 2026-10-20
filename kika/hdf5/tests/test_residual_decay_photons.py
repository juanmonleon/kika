"""A level's photons stated under its residual reach the processed reaction.

FUDGE writes NNDC's Cl-35 MT51 as ``n + (Cl35_e1 -> Cl35 + photon)``: the
photon is in the residual's own outputChannel, not beside the neutron. The
processed suite used to read the reaction's products alone, so the HDF5 file
had no photons for 51-91, 600-849 and the heating deposited each level's
energy (KERMA x2.3 above 5 MeV once the balance used QM).
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from kika.hdf5.processing import _decayPhotons


def _product(pid, *children):
    return NS(pid=pid, label=pid, outputChannel=NS(products=list(children)) if children else None)


def test_the_photons_of_a_residuals_decay_are_found_at_any_depth():
    first, second = _product("photon"), _product("photon")
    residual = _product("Cl35_e2", _product("Cl35_e1", _product("Cl35"), second), first)
    reaction = NS(outputChannel=NS(products=[_product("n"), residual]))
    assert _decayPhotons(reaction) == [first, second]


def test_a_reaction_without_a_decaying_residual_has_none():
    reaction = NS(outputChannel=NS(products=[_product("n"), _product("Fe56"), _product("photon")]))
    assert _decayPhotons(reaction) == []


@pytest.mark.tape
def test_nndcs_cl35_levels_carry_their_photons_under_the_residual():
    tapes = os.environ.get("KIKA_LIB_TAPES")
    stated = os.environ.get("KIKA_GNDS_LIB")
    root = Path(stated) if stated else (Path(tapes).parent / "gnds" / "endfb81" if tapes else None)
    path = root / "n-017_Cl_035.endf.gnds.xml" if root else None
    if path is None or not path.is_file():
        pytest.skip("NNDC's GNDS Cl-35 not found (set KIKA_LIB_TAPES or KIKA_GNDS_LIB)")
    import kika

    suite = kika.read(str(path), format="gnds", covariances=False)
    for mt in (51, 52, 601, 801):
        reaction = suite.reactions.byENDF_MT(mt)
        assert not any(p.pid == "photon" for p in reaction.outputChannel.products), mt
        assert _decayPhotons(reaction), mt


@pytest.mark.parametrize("form", ["Branching3d", "Unspecified"])
def test_a_residual_photon_that_states_no_spectrum_is_not_reported_lost(form):
    # Fe-56 MT51's level photon is a Branching3d (its cascade is in PoPs and
    # _cascadePhotons writes it); O-17 MT51's is Unspecified (the tape states
    # no photon). Neither is a loss.
    from kika.hdf5.processing import _products
    from kika.nuclear_data import model as m
    from kika.processing.tabulation import TabulationOptions

    photon = m.Product(pid="photon", label="photon")
    photon.distribution = m.Distribution()
    photon.distribution["eval"] = getattr(m, form)(label="eval", productFrame=m.Frame.lab)
    level = NS(pid="Fe56_e1", label="Fe56_e1", outputChannel=NS(products=[photon]))
    reaction = NS(ENDF_MT=51, outputChannel=NS(products=[level], fissionFragmentData=None))
    channel = NS(products=NS(products=[]))
    report = m.ConversionReport()
    _products(reaction, channel, 51, 55.454, 26056, TabulationOptions(), report, pops=None)
    assert report.losses == []
