"""Dense scheduling must retain the reference path for ineligible backends."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import kika
from kika.hdf5 import processing as P
from kika.nuclear_data import model as m
from kika.processing.doppler import DopplerOptions, core, _blocks


@pytest.mark.parametrize("mode", ["native", "absent", "python", "negative", "small-memory"])
def test_dense_schedule_respects_native_eligibility(mode,monkeypatch):
    tape=Path(__file__).resolve().parents[2]/"endf/tests/data/micro_fe56_structural.endf"
    suite=kika.read(tape,covariances=False)
    for reaction in suite.reactions:
        if reaction.ENDF_MT in (2,102):
            x=np.geomspace(1e-5,1e4,513)
            y=np.full(len(x),-1. if mode=="negative" else 1.)
            reaction.crossSection["recon"]=m.XYs1d(xs=x,ys=y,label="recon")
    monkeypatch.setattr(P,"CHUNK_KNOTS",32)
    monkeypatch.setattr(_blocks,"_DENSE_SOURCE",128)
    monkeypatch.setattr(core,"_native",None if mode=="absent" else SimpleNamespace(prepare_blocks=True))
    options=DopplerOptions(backend="python" if mode=="python" else "auto",
                          max_work_bytes=32*1024**2 if mode=="small-memory" else 64*1024**2)
    entries,leaves,lo,hi,tasks=P._broadenPlan(suite,2250.,options)
    assert len(leaves)==2
    if mode=="native":
        assert len(tasks)==2
        assert all(len(keys)==1 and domain==(lo,hi) for keys,domain in tasks)
    else:
        assert len(tasks)>2
        assert all(len(keys)==2 for keys,domain in tasks)
    # No source ordinate or style is altered by choosing a schedule.
    for key in leaves:
        np.testing.assert_array_equal(entries[key].crossSection["recon"].ys,y)
