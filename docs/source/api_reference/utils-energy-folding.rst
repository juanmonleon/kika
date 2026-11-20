kika.utils.energy_folding
=========================

Energy resolution of a time-of-flight measurement, and the folding of cross-sections
and angular distributions with it.

.. automodule:: kika.utils.energy_folding
   :members: EnergyFoldingConfig, tof_energy_resolution, compute_energy_resolution_tof, fold_cross_section, fold_angular_distribution, endf_angular_distribution, compute_folded_differential_xs, compute_unfolded_differential_xs

``kika.utils`` exports these names, and :func:`kika.algebra.gaussian_fold_nodes` and
:func:`kika.algebra.fold_tabulated`, which belong to :mod:`kika.algebra`.

Debug logging
-------------

.. currentmodule:: kika.utils

.. autofunction:: configure_ace_debug_logging

.. autofunction:: configure_endf_debug_logging

.. autofunction:: get_endf_logger
