"""A thermal ACE table (THERMR + ACER) read into the processed thermal forms.

The table here is built by hand in the type-1 layout, small enough to check
every number; the real ones are checked against the official library
(H in H2O at 294 K and 600 K came out
identical dataset by dataset).
"""
from __future__ import annotations

import numpy as np

from kika.hdf5 import records as R
from kika.hdf5.ace_thermal import nuclidesFromPairs, readAceTable, thermalForms
from kika.nuclear_data import model as m


def _table(tmp_path):
    nE, nMu = 2, 2                      # incident energies; NXS(3) = nMu + 1 for continuous
    energies = [1e-11, 1e-6]            # MeV
    xs = [20.0, 10.0]
    # ITIE at 1: n, E..., sigma...
    itie = [nE] + energies + xs
    itxe_start = 1 + len(itie)
    # ITXE: locators (LOCC) then counts, then per incident energy NOut x (E', p, c, mu1, mu2)
    nOut = [2, 2]
    blocks = []
    for i in range(nE):
        blocks.append([1e-12, 0.5e12, 0.25, -0.5, 0.5,
                       2e-12, 0.5e12, 1.0, 0.2, -0.2])
    locc = []
    pos = itxe_start + 2 * nE
    for b in blocks:
        locc.append(pos - 1)            # xss[locc+1] is the first E'
        pos += len(b)
    itxe = locc + nOut + [v for b in blocks for v in b]
    itce_start = itxe_start + len(itxe)
    itce = [2, 1e-9, 2e-9, 1e-6, 3e-6]  # coherent: n, edges (MeV), factors (MeV b)
    xss = itie + itxe + itce
    nxs = [len(xss), 0, nMu + 1, 0, 4, -1, 2] + [0] * 9
    jxs = [1, 0, itxe_start, itce_start] + [0] * 28
    lines = [" hh2o.20t   0.999167  2.5300E-08   10/11/26",
             "test table" + " " * 60 + "mat1001"]
    pairs = [(1001, 0.999167)] + [(0, 0.0)] * 15
    for k in range(4):
        lines.append("".join(f"{z:7d}{a:11.6f}" for z, a in pairs[4 * k:4 * k + 4]))
    ints = nxs + jxs
    for k in range(0, 48, 8):
        lines.append("".join(f"{v:9d}" for v in ints[k:k + 8]))
    for k in range(0, len(xss), 4):
        lines.append("".join(f"{v:20.11E}" for v in xss[k:k + 4]))
    path = tmp_path / "t.20t"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_table_reads_into_the_processed_forms(tmp_path):
    ace = readAceTable(_table(tmp_path))
    assert ace.name == "hh2o.20t" and ace.kT == 2.53e-8 and ace.pairs == [(1001, 0.999167)]
    inXs, inDist, elXs, elDist, energyMax = thermalForms(ace)
    assert np.allclose(inXs.xs, [1e-5, 1.0]) and np.allclose(inXs.ys, [20.0, 10.0])
    assert energyMax == 1.0
    assert isinstance(inDist, m.EnergyAngularMC)
    spectrum = inDist.energy.function1ds[0]
    # The CDF started at 0.25, so a 0 eV point with isotropic cosines was added.
    assert spectrum.xs[0] == 0.0 and spectrum.cdf[0] == 0.0 and len(spectrum) == 3
    cosines = inDist.energyAngular.function2ds[0].function1ds
    assert len(cosines) == 3
    assert np.allclose(cosines[2].xs, [-0.2, 0.2])          # sorted
    assert cosines[1].numberOfDiscrete == 2
    assert isinstance(elXs, m.CoherentElasticXS) and isinstance(elDist, m.CoherentElasticMC)
    assert np.allclose(elXs.braggEdges, [1e-3, 2e-3])
    assert np.allclose(elXs.structureFactors, [1.0, 3.0])


def test_the_forms_encode_as_openmc_lays_them_out(tmp_path):
    from kika.hdf5.distributions import encodeCorrelated

    inXs, inDist, *_ = thermalForms(readAceTable(_table(tmp_path)))
    group = encodeCorrelated(inDist, typeName="incoherent_inelastic")
    assert R.text(R.attr(group, "type")) == "incoherent_inelastic"
    eout = group["energy_out"]
    assert eout.data.shape == (5, 6)
    assert list(R.attr(eout, "n_discrete_lines")) == [0, 0]
    assert np.all(eout.data[3] == 0)                         # discrete cosines
    assert group["mu"].data.shape == (3, 12)


def test_nuclides_as_openmc_lists_them():
    assert nuclidesFromPairs([(6000, 11.9)]) == ["C0", "C12", "C13"]
    assert nuclidesFromPairs([(1001, 1.0)]) == ["H1"]
