"""The URR tables a perturbation run shares among its samples are processSuite's.

Computed once on the unperturbed evaluation, reconstructed and broadened as each
sample is, so the stopping rule of the ladders is measured on the same smooth
cross sections: a file written with them is the file processSuite writes alone.
"""
from types import SimpleNamespace

from kika.sampling.model_perturbation import _hdf5Pieces


def test_the_shared_tables_give_the_file_processsuite_writes(cf252_b81_tape, tmp_path):
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.read_endf import read_endf
    from kika.endf.reconstruction import context_from_endf
    from kika.hdf5 import records as R
    from kika.hdf5.processing import urrTablesFor
    from kika.hdf5.writer import writeSuite

    suite, _ = decodeReactionSuite(read_endf(str(cf252_b81_tape)))
    context, _ = context_from_endf(read_endf(str(cf252_b81_tape), mf_numbers=[1, 2]), suite)
    tables = urrTablesFor(suite, [293.6], context=context)
    writeSuite(suite, tmp_path, temperatures=[293.6], fileName="alone.h5", index=False,
               endf=cf252_b81_tape)
    writeSuite(suite, tmp_path, temperatures=[293.6], fileName="shared.h5", index=False,
               endf=cf252_b81_tape, urrTables=tables)
    assert R.compareTrees(R.readFile(tmp_path / "alone.h5"),
                          R.readFile(tmp_path / "shared.h5"), limit=5) == []


def test_the_ace_emitter_reuses_the_hdf5_emitters_tables():
    suite = SimpleNamespace(resonances=SimpleNamespace(unresolved=object()))
    options = SimpleNamespace(omit=(), temperatures=(293.6,), thnmax="njoy", thinning=1e-3)
    computed = {"urr": object(), "key": ((293.6,), "njoy", 1e-3)}
    assert _hdf5Pieces(options, suite, None, None, computed) is computed
    assert _hdf5Pieces(SimpleNamespace(**{**vars(options), "omit": ("urr",)}), suite,
                       None, None, computed) is None
