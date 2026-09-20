# Multigroup covariance analysis package
#
# The four ``plot_mg_*`` functions used to be re-exported from here, through a
# deprecation shim (``plotting_mg.py``) onto a 1553-line legacy module
# (``legacy_mg_plotting.py``). That import is what kept the legacy module alive
# on every ``import kika.cov.multigroup``, DeprecationWarning included, although
# nothing called it: the four plotting methods on
# :class:`MultigroupLegendreCovariance` have gone straight to
# :mod:`kika.plotting.multigroup_covariance` since that module was written.
# Both files were removed in September 2026. Reach for the functions at
# :mod:`kika.plotting.multigroup_covariance`, or for the methods on the class.

from .mg_legendre_covariance import MultigroupLegendreCovariance, MGMF34CovMat
from .collapse import collapse_to_multigroup, MF34_to_MG

__all__ = [
    'MultigroupLegendreCovariance',
    'MGMF34CovMat',
    'collapse_to_multigroup',
    'MF34_to_MG',
]
