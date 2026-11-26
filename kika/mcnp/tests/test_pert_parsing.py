import pytest
from kika.mcnp.parse_input import read_mcnp, _read_PERT


def test_input_parsing_single_card(tmp_path):
    # Test a simple input file with one PERT card.
    content = "PERT1:n CELL=1,2,3 MAT=5 RHO=0.95 METHOD=2 RXN=101 ERG=0.0 10.0\n"
    file = tmp_path / "input_test1.i"
    file.write_text(content)
    
    inst = read_mcnp(str(file))
    perturbation = inst.perturbation.pert.get(1)
    assert perturbation is not None
    assert perturbation.particle == "n"
    assert perturbation.cell == [1, 2, 3]
    assert perturbation.material == 5
    assert perturbation.rho == 0.95
    assert perturbation.method == 2
    assert perturbation.reaction == 101
    assert perturbation.energy == (0.0, 10.0)

def test_input_parsing_multiline_card(tmp_path):
    # Test a PERT card spanning multiple lines with '&' continuation.
    content = (
        "PERT2:f CELL=1,2,3 &\n"
        "MAT=5 RHO=1.0 &\n"
        "METHOD=2 RXN=102 ERG=0.5 9.5\n"
    )
    file = tmp_path / "input_test2.i"
    file.write_text(content)
    
    inst = read_mcnp(str(file))
    perturbation = inst.perturbation.pert.get(2)
    assert perturbation is not None
    assert perturbation.particle == "f"
    assert perturbation.cell == [1, 2, 3]
    assert perturbation.material == 5
    assert perturbation.rho == 1.0
    assert perturbation.method == 2
    assert perturbation.reaction == 102
    assert perturbation.energy == (0.5, 9.5)

def test_PERT_parsing_valid():
    # Test direct parsing of a valid PERT card.
    lines = ["PERT3:p CELL=4,5 MAT=7 RHO=0.88 METHOD=3 RXN=103 ERG=1.0 11.0"]
    pert_obj, new_index = _read_PERT(lines, 0)
    assert pert_obj is not None
    assert pert_obj.id == 3
    assert pert_obj.particle == "p"
    assert pert_obj.cell == [4,5]
    assert pert_obj.material == 7
    assert pert_obj.rho == 0.88
    assert pert_obj.method == 3
    assert pert_obj.reaction == 103
    assert pert_obj.energy == (1.0, 11.0)
    assert new_index == 1

def test_PERT_parsing_invalid_header():
    # Test that a card with an invalid header returns None.
    lines = ["XYZ3:p CELL=4,5 MAT=7 RHO=0.88 METHOD=3 RXN=103 ERG=1.0 11.0"]
    pert_obj, new_index = _read_PERT(lines, 0)
    assert pert_obj is None

def test_PERT_parsing_incomplete_erg():
    # Test a card with incomplete ERG values so that energy is set to None.
    lines = ["PERT4:a CELL=1 MAT=5 RHO=1.0 METHOD=1 RXN=100 ERG=5.0"]
    pert_obj, new_index = _read_PERT(lines, 0)
    assert pert_obj is not None
    # Energy should be None because only one value is provided.
    assert pert_obj.energy is None


@pytest.mark.parametrize("rxn", ["RXN=51 52", "RXN=51,52", "RXN= 51 52", "RXN = 51, 52", "rxn=51,52"])
def test_every_mt_of_rxn_is_read_and_the_next_keyword_still_is(rxn):
    pert, _ = _read_PERT([f"PERT1:n CELL=1 MAT=2 RHO=-7.8 METHOD=2 {rxn} ERG=1 2"], 0)
    assert pert.reactions == (51, 52)
    assert pert.reaction is None
    assert pert.energy == (1.0, 2.0)


@pytest.mark.parametrize("cell", ["CELL=1 2 3", "CELL=1,2,3", "CELL=1, 2, 3", "CELL= 1 ,2 3"])
def test_cells_take_blanks_and_commas(cell):
    pert, _ = _read_PERT([f"PERT1:n {cell} MAT=2 RHO=1 METHOD=2 RXN=2"], 0)
    assert pert.cell == [1, 2, 3]
    assert pert.reaction == 2


@pytest.mark.parametrize("erg", ["ERG=1 2", "ERG=1,2", "ERG= 1, 2"])
def test_the_energy_bounds_take_blanks_and_commas(erg):
    pert, _ = _read_PERT([f"PERT1:n CELL=1 MAT=2 RHO=1 METHOD=2 RXN=2 {erg}"], 0)
    assert pert.energy == (1.0, 2.0)


def test_a_card_in_lowercase_is_read(tmp_path):
    deck = tmp_path / "deck.i"
    deck.write_text("pert7:N cell=1 mat=2 rho=-7.8 method=2 rxn=102 erg=1 2\n")
    pert = read_mcnp(str(deck)).perturbation.pert[7]
    assert pert.particle == "n"
    assert pert.reaction == 102
    assert pert.energy == (1.0, 2.0)


def test_a_card_continued_by_leading_blanks_is_read():
    lines = ["PERT1:n CELL=1 MAT=2 RHO=1\n", "      METHOD=2 RXN=51 52\n", "      ERG=1 2\n"]
    pert, end = _read_PERT(lines, 0)
    assert pert.reactions == (51, 52)
    assert pert.energy == (1.0, 2.0)
    assert end == 3


def test_the_whole_range_of_a_sum_mt_stands_for_it():
    levels = " ".join(str(mt) for mt in range(51, 92))
    pert, _ = _read_PERT([f"PERT1:n CELL=1 MAT=2 RHO=1 METHOD=2 RXN={levels}"], 0)
    assert pert.reaction == 4
    assert len(pert.reactions) == 41


def test_several_mts_are_refused_where_a_profile_is_grouped_by_mt(tmp_path):
    deck = tmp_path / "deck.i"
    deck.write_text(
        "PERT1:n CELL=1 MAT=2 RHO=1 METHOD=2 RXN=2 ERG=1 2\n"
        "PERT2:n CELL=1 MAT=2 RHO=1 METHOD=2 RXN=51,52 ERG=1 2\n"
    )
    perturbation = read_mcnp(str(deck)).perturbation
    with pytest.raises(ValueError, match=r"PERT2 RXN=51 52"):
        perturbation._group_perts_by_reaction(2)


def test_read_file_cards_are_followed_beside_the_input(tmp_path):
    (tmp_path / "perts").mkdir()
    (tmp_path / "perts" / "PERT.i").write_text(
        "c kika:pert_zaid=26056 pert_mat=2\n"
        "PERT1:n CELL=1 MAT=2 RHO=1 METHOD=2 RXN=51,52 ERG=1 2\n"
        "read file=more.i\n"
    )
    (tmp_path / "perts" / "more.i").write_text("PERT2:n CELL=1 MAT=2 RHO=1 METHOD=2 RXN=2 ERG=1 2\n")
    deck = tmp_path / "deck.i"
    deck.write_text("title\nREAD FILE=perts/PERT.i\nm2 26056.80c 1\n")
    inp = read_mcnp(str(deck))
    assert sorted(inp.perturbation.pert) == [1, 2]
    assert inp.perturbation.pert[1].reactions == (51, 52)
    assert inp.perturbation.pert[1].zaid == 26056
    assert 2 in inp.materials.by_id


def test_a_missing_read_file_is_named(tmp_path):
    deck = tmp_path / "deck.i"
    deck.write_text("READ FILE=PERT.i\n")
    with pytest.raises(FileNotFoundError, match="PERT.i"):
        read_mcnp(str(deck))


def test_read_cards_in_a_loop_are_refused(tmp_path):
    (tmp_path / "a.i").write_text("read file=b.i\n")
    (tmp_path / "b.i").write_text("read file=a.i\n")
    with pytest.raises(ValueError, match="loop"):
        read_mcnp(str(tmp_path / "a.i"))

