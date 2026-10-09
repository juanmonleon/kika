"""G6: the covariance sections' ENDF header -- MAT, ZA, AWR and MF34's LTT.

§25.2.2 identifies a covariance by the ``href`` of what it is about, not by a
material header, so a covariance read from GNDS has no ZA/AWR/MAT and the
MF31/33/34/35 encoders refused it ("the section header would be invented").
They are not invented here: they are the suite's own, derived in G1, as every
other section of the tape has them. MF34's LTT is 1, the only value ENDF-6
admits there (Legendre-coefficient covariances, §34.2).

Everything else of an MF31-35 section -- the LB of each sub-subsection, MAT1,
MT1, NL, NI -- the encoders already take from the model: the row and column
links, the matrix shape and its grids. MF32 from GNDS is not written (roadmap
E1: it is rewritten from the section the ENDF decoder kept, and a GNDS suite
has none), and says so.
"""
from __future__ import annotations

from . import DERIVED, DerivationContext, register

__all__ = ["deriveCovarianceSection"]


def deriveCovarianceSection(section, path, context: DerivationContext, report):
    from kika.nuclear_data.model import EndfProvenance

    row = getattr(section, "rowData", None)
    mf = getattr(row, "ENDF_MF", None) if row is not None else None
    if mf is None:
        return None
    header = {"ltt": 1} if int(mf) == 34 else {}
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, headerFields=header)


def _isCovarianceSection(node) -> bool:
    from kika.nuclear_data.model.covariances import CovarianceSection

    return isinstance(node, CovarianceSection)


register("covariances", _isCovarianceSection, deriveCovarianceSection)
