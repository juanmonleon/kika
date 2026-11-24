"""A level reaction's residual is the level, and a two-body residual recoils, as FUDGE writes them."""
from __future__ import annotations

import collections
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import kika

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"


def _reaction(suite, mt):
    return suite.reactions[mt]


def _written(tmp_path, tape):
    suite = kika.read(DATA / tape)
    path = tmp_path / "out.xml"
    report = kika.write(suite, path, format="gnds")
    reactions = {r.get("ENDF_MT"): r for r in ET.parse(path).getroot().iter("reaction")}
    return suite, report, reactions


def _products(reaction):
    return list(reaction.find("outputChannel").find("products"))


def _recoilOf(product):
    node = product.find("distribution").find("angularTwoBody")
    recoil = None if node is None else node.find("recoil")
    return None if recoil is None else recoil.get("href")


def test_an_mf6_residual_stated_as_the_ground_state_is_the_level():
    suite = kika.read(DATA / "micro_li6_mf6.endf")
    residual = _reaction(suite, 52).outputChannel.products[1]
    assert (residual.pid, residual.label) == ("Li6_e2", "Li6_e2")
    # No photon data for this level: its decay is the ground state alone, as NNDC's.
    assert [p.pid for p in residual.outputChannel.products] == ["Li6"]
    record = suite.reactions[52].provenance.headerFields["mf6"]["products"][1]
    assert (record["pid"], record["label"]) == ("Li6_e2", "Li6_e2")


def test_mf6_photons_of_a_level_are_the_residuals_decay():
    suite = kika.read(DATA / "micro_be9_mf6.endf")
    channel = _reaction(suite, 701).outputChannel
    assert [p.pid for p in channel.products] == ["H3", "Li7_e1"]
    assert [p.pid for p in channel.products[1].outputChannel.products] == ["Li7", "photon"]
    assert channel.genre == "twoBody"


def _sections(path):
    out = collections.defaultdict(list)
    for line in open(path, encoding="latin-1"):
        try:
            mf, mt = int(line[70:72]), int(line[72:75])
        except ValueError:
            continue
        if mf and mt:
            out[(mf, mt)].append(line[:66])
    return out


@pytest.mark.parametrize("tape", ["micro_li6_mf6.endf", "micro_be9_mf6.endf"])
def test_the_tape_comes_back_the_same(tape, tmp_path):
    source = DATA / tape
    kika.write(kika.read(source), tmp_path / "out.endf", format="endf")
    before, after = _sections(source), _sections(tmp_path / "out.endf")
    for key in before:
        if key[0] in (3, 6, 12):
            assert after[key] == before[key], f"MF{key[0]}/MT{key[1]}"


def test_an_unspecified_residual_is_written_as_the_ejectiles_recoil(tmp_path):
    _, report, reactions = _written(tmp_path, "micro_w186_covcheck.endf")
    neutron, residual = _products(reactions["51"])
    assert residual.get("pid") == "W186_e1"
    assert _recoilOf(residual) == ("../../../../product[@label='n']/distribution/"
                                   "angularTwoBody[@label='eval']")
    assert residual.find("outputChannel") is not None
    assert any("written as the ejectile's recoil" in m for m in report.warnings)


def test_a_missing_residual_is_added_after_the_ejectile(tmp_path):
    _, report, reactions = _written(tmp_path, "micro_u235_mf6.endf")
    products = _products(reactions["800"])
    assert [p.get("pid") for p in products][:2] == ["He4", "Th232"]
    assert _recoilOf(products[1]).startswith("../../../../product[@label='He4']")
    assert products[1].find("multiplicity").find("constant1d").get("value") == "1"


def test_elastic_still_gets_the_targets_recoil(tmp_path):
    suite, _, reactions = _written(tmp_path, "micro_fe56_xs_and_angular.endf")
    products = _products(reactions["2"])
    assert [p.get("pid") for p in products] == ["n", suite.target]
    assert _recoilOf(products[1]) is not None
    assert len(list(suite.reactions[2].outputChannel.products)) == 1


def test_writers_that_rebuild_mf6_still_find_the_photon_in_the_decay():
    from kika.endf.model_adapter.photon_lines import mf6Products

    suite = kika.read(DATA / "micro_be9_mf6.endf")
    assert [p.label for p in mf6Products(suite.reactions[701])] == ["H3", "Li7_e1", "photon"]
