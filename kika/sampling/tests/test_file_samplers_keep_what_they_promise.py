"""The file-level samplers: energy windows, xsdir rewrites, worker errors and tape discovery."""

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling import ace_perturbation, ace_perturbation_separate
from kika.sampling.endf_perturbation import perturb_ENDF_files
from kika.sampling.reprocess_endf_to_ace import _construct_endf_file_paths

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = DATA / "micro_fe56_structural.endf"
GRID_MEV = [1.0e-11, 1.0, 20.0]
FACTORS = np.array([[1.1, 0.9, 1.2, 0.8]])


@pytest.mark.parametrize("module", [ace_perturbation, ace_perturbation_separate])
def test_a_bin_outside_every_window_gets_factor_one(module):
    masked = module._mask_factors_by_energy_range_ace(
        FACTORS, [2, 102], GRID_MEV, [(0.0, 1.0e6)])

    assert np.allclose(masked, [[1.1, 1.0, 1.2, 1.0]])


class _Recorder:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message)

    def info(self, message, *args, **kwargs):
        pass


def test_a_reaction_with_a_non_positive_factor_is_reported_not_applied():
    assert ace_perturbation._apply_factors_to_mt(
        None, 102, np.array([0.0, 1.0]), np.array(GRID_MEV)) == "+0.000e+00"


def test_the_run_logs_every_reaction_a_sample_left_unperturbed(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(ace_perturbation, "_get_logger", lambda: recorder)
    monkeypatch.setattr(ace_perturbation, "_process_sample",
                        lambda *args: [(102, "+0.000e+00")])
    tasks = [("t.ace", np.ones(2), 0, GRID_MEV, [102], ".", None, None, None)]

    ace_perturbation._run_samples(tasks, nprocs=1, verbose=False)

    assert any("0001" in w and "MT=102" in w and "not perturbed" in w
               for w in recorder.warnings)


def test_only_xsdir_rewrites_what_perturb_endf_files_wrote(tmp_path):
    zaid = read_endf(str(TAPE)).zaid
    snippet_dir = tmp_path / "ace" / "0001" / "xsdir"
    snippet_dir.mkdir(parents=True)
    entry = f"{zaid}.80c 55.454 ace/0001/{zaid}.80c 0 1 1 123456 0 0 2.5852E-08"
    (snippet_dir / f"{zaid}.80c.xsdir").write_text(entry + "\n")
    baseline = tmp_path / "xsdir_base"
    baseline.write_text(f"directory\n{zaid}.80c 55.454 old/path 0 1 1 999 0 0 2.5852E-08\n")

    perturb_ENDF_files(str(TAPE), [[2]], [], 1, output_dir=str(tmp_path),
                       xsdir_file=str(baseline), only_xsdir=True)

    master = tmp_path / "xsdir" / "xsdir_base_0001"
    assert master.is_file()
    assert f"ace/0001/{zaid}.80c" in master.read_text()
    assert "old/path" not in master.read_text()


def test_the_summary_file_is_never_taken_for_the_tape(tmp_path, monkeypatch):
    sample = tmp_path / "endf" / "26056" / "0001"
    sample.mkdir(parents=True)
    (sample / "jeff40_260560_0001_summary.txt").write_text("summary\n")
    (sample / "jeff40_260560_0001.txt").write_text("tape\n")
    from kika.sampling import reprocess_endf_to_ace as module
    listdir = module.os.listdir
    monkeypatch.setattr(module.os, "listdir", lambda path: sorted(listdir(path), reverse=True))

    found = _construct_endf_file_paths(str(tmp_path), 26056, 1, "jeff40_260560")

    assert [Path(path).name for path, _, _ in found] == ["jeff40_260560_0001.txt"]


def test_a_sample_that_fails_in_a_worker_raises(tmp_path):
    tasks = [(str(tmp_path / "missing.ace"), np.ones(2), 0, GRID_MEV, [2], str(tmp_path),
              None, None, None)]

    with pytest.raises(RuntimeError, match="0001"):
        ace_perturbation._run_samples(tasks, nprocs=2, verbose=False)
