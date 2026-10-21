"""The QM of a continuum (MT91, MT649, ...) derived from the levels of its series.

QM is the Q to the residual's ground state, a property of the channel: every
level of the series states the same one, and the continuum takes it. When the
levels agree that is the channel's QM exactly, a note; when they do not (JEFF-4.0
Fe-56: MT649 400 eV from MT601), taking the lowest is a choice, an approximation
-- which the PENDF publication refuses, so the distinction decides whether a
PENDF can be written from a GNDS file.
"""
from types import SimpleNamespace as NS

from kika.endf.model_adapter.derive import reactions as r
from kika.nuclear_data.model import ConversionReport


def _level(mt, qi, decay):
    """A discrete level MT whose residual decays with Q ``decay``."""
    residual = NS(pid="Fe56_e1", outputChannel=NS(Q=NS(value=decay)))
    return NS(ENDF_MT=mt, label=f"MT{mt}", outputChannel=NS(Q=NS(value=qi), products=[residual]))


def _derived(monkeypatch, levels):
    monkeypatch.setattr(r, "_lr", lambda residual, label, report: 0)
    context = NS(suite=NS(reactions=levels))
    report = ConversionReport()
    qm, lr = r._continuum(None, 91, -4.5e6, context, report)
    return qm, report


def test_levels_that_agree_give_the_channels_qm_as_a_note(monkeypatch):
    # QI = -E_level and the residual's decay releases E_level: QM = 0 for (n,n'),
    # here with the seven-digit rounding of real fields.
    levels = [_level(51, -846778.0, 846778.0), _level(52, -2085100.0, 2085101.0),
              _level(53, -2658600.0, 2658599.0)]
    qm, report = _derived(monkeypatch, levels)
    assert qm == 0.0
    assert report.isClean and any("MT91" in w and "every level" in w for w in report.warnings)


def test_levels_that_disagree_are_an_approximation(monkeypatch):
    levels = [_level(51, -846778.0, 846778.0), _level(52, -2085100.0, 2085500.0)]
    qm, report = _derived(monkeypatch, levels)
    assert qm == 0.0
    assert not report.isClean and any("do not all state" in a for a in report.approximations)
