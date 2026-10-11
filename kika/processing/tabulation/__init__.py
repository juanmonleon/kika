"""ACER-like tabulation: evaluated distributions → Monte Carlo tables (roadmap H3, D6).

What a continuous-energy transport code samples is not what an evaluation
states. ACER (NJOY) turns Legendre series, laws and correlated forms into
tables in mu and E' with their cumulative; this package does the same on the
model, so any processed writer can use it -- the OpenMC HDF5 writer today, an
ACE writer if one comes. The format packages only serialise its output.

The forms it produces are GNDS's ``MonteCarlo_cdf`` ones (``xs_pdf_cdf1d``,
``energyAngularMC``, the Monte Carlo ``KalbachMann``) plus the two processed
laws with no GNDS node (:class:`~kika.nuclear_data.model.LevelEnergy`,
:class:`~kika.nuclear_data.model.DiscretePhotonEnergy`), filed under the
``MonteCarlo`` label.

The rule for a Legendre series that turns negative is ACER's: the negative
densities are set to zero and the table renormalised; the conversion report
says how many tables needed it (:attr:`TabulationOptions.negative`).
"""
from .tables import NEGATIVE_POLICIES, TabulationOptions, normalisedTable, pdfTable, refine
from .angular import legendreDensity, muTable, tabulateAngular
from .energy import mapFunction2d, tabulateEnergy
from .kalbach import kalbachSlope, separationEnergy
from .correlated import tabulateAngularEnergy, tabulateEnergyAngular, tabulateKalbachMann
from .distribution import (MONTE_CARLO_LABEL, TabulationContext, isLevelMT, levelEnergy,
                           tabulateDistribution)

__all__ = ["TabulationOptions", "NEGATIVE_POLICIES", "refine", "normalisedTable", "pdfTable",
           "legendreDensity", "muTable", "tabulateAngular", "mapFunction2d", "tabulateEnergy",
           "kalbachSlope", "separationEnergy", "tabulateKalbachMann", "tabulateEnergyAngular",
           "tabulateAngularEnergy", "TabulationContext", "tabulateDistribution", "isLevelMT",
           "levelEnergy", "MONTE_CARLO_LABEL"]
