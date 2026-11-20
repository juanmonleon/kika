"""Covariance matrices of nuclear data: reading, conditioning and collapsing them.

A :class:`CrossSectionCovariance` holds the cross-section (MF33) covariances and a
:class:`LegendreCovariance` the Legendre-coefficient (MF34) ones, each read from COVERX,
BOXER or COVFIL files or built from an evaluation. The package also repairs a matrix that
cannot be sampled (see :mod:`kika.cov.conditioning`) and collapses covariances onto a
coarser group structure (see :mod:`kika.cov.multigroup`).
"""
from .parse_covmat import read_coverx, write_coverx, read_covfil, write_covfil, read_boxer, write_boxer, read_scale_covmat, read_njoy_covmat
from .legendre_covariance import LegendreCovariance, MF34CovMat
from .cross_section_covariance import CrossSectionCovariance, CovMat, TransferResult
from .decomposition import nearest_psd_higham, verify_psd_projection

__all__ = [
    'read_coverx',
    'write_coverx',
    'read_covfil',
    'write_covfil',
    'read_boxer',
    'write_boxer',
    'read_scale_covmat',
    'read_njoy_covmat',
    'LegendreCovariance',
    'MF34CovMat',
    'CrossSectionCovariance',
    'CovMat',
    'TransferResult',
    'nearest_psd_higham',
    'verify_psd_projection',
]
