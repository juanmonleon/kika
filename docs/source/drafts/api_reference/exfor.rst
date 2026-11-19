kika.exfor
==========

Experimental data from EXFOR: datasets loaded from an X4Pro database or from JSON files,
the conversions between the centre-of-mass and laboratory frames, and plots of the data
against an evaluation.

.. currentmodule:: kika.exfor

Configuring where the data is
-----------------------------

.. autofunction:: configure

.. autofunction:: get_config

Reading datasets
----------------

.. autofunction:: read_exfor

.. autofunction:: read_all_exfor

.. autofunction:: read_exfor_from_database

.. autoclass:: X4ProDatabase

.. autoclass:: X4ProDataset

.. autofunction:: parse_reacode_fields

.. autofunction:: exfor_quantity

.. autofunction:: canonical_unit_factor

.. py:data:: EXFOR_QUANTITY_CODES

   The quantity codes of the X4Pro database, each with its description.

.. py:data:: EXFOR_QUANTITY_WILDCARDS

   The combined codes that match several quantities (``"DA*"``), each with its description.

.. py:data:: QUANTITY_FAMILIES

   Related quantity codes, grouped by family.

.. py:data:: EXFOR_URL_BASE

   The address of the EXFOR database at the IAEA, for links to an entry.

What a dataset is
-----------------

.. autoclass:: ExforEntry

.. autoclass:: ExforAngularDistribution

.. autoclass:: ExforCrossSection

.. autoclass:: ExforExperiment

Frames and angles
-----------------

.. autofunction:: cos_cm_from_cos_lab

.. autofunction:: cos_lab_from_cos_cm

.. autofunction:: jacobian_cm_to_lab

.. autofunction:: jacobian_lab_to_cm

.. autofunction:: transform_lab_to_cm

.. autofunction:: transform_cm_to_lab

.. autofunction:: angle_deg_to_cos

.. autofunction:: cos_to_angle_deg

Plots
-----

.. autofunction:: plot_exfor_angular

.. autofunction:: plot_exfor_ace_comparison

.. autofunction:: plot_multiple_energies

Earlier functions
-----------------

The functions of ``kika.exfor.AD_utils`` are kept for existing scripts. New code uses
the classes above.

.. autofunction:: load_exfor_data

.. autofunction:: load_all_exfor_data

.. autofunction:: extract_experiment_info

.. autofunction:: extract_angular_distribution

.. autofunction:: calculate_differential_cross_section

.. autofunction:: cosine_to_angle_degrees

.. autofunction:: angle_degrees_to_cosine

.. autofunction:: plot_angular_distribution

.. autofunction:: plot_combined_angular_distribution

.. autofunction:: plot_individual_energy_comparisons

.. autofunction:: plot_combined_angular_distribution_multi_ace

.. autofunction:: plot_all_energies_comparison
