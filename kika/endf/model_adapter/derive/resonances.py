"""MF2/151 for a suite whose resonances hold the scattering radius alone.

A GNDS ``resonances`` node with a ``scatteringRadius`` and no ``resolved`` or
``unresolved`` region is ENDF's LRU=0 range: LRP=0, and MF2/151 states one
isotope with one range whose only parameters are the target spin and AP
(ENDF-102 §2.1, the "scattering radius only" case). Every field of that range
is in the model:

=============  ===========================================================
``ZAI``        the target's ZA (one isotope, abundance 1)
``EL``/``EH``  the radius' own energy domain (GNDS ``constant1d``), else the
               evaluation's ``projectileEnergyDomain``, reported
``SPI``        the target's spin, from PoPs
``AP``         the constant radius, fm → ENDF's 10⁻¹² cm
``LFW``        0: a range with no widths has no fission widths to flag
=============  ===========================================================

Resolved and unresolved parameters need fields the model has no node for (QX,
LRX, LAD and the particle-pair columns), so they are not derived here, and the
writer reports MF2 as lost. A radius that depends on energy is not either: an
LRU=0 range carries a constant AP. A target spin PoPs does not state is refused,
never assumed.
"""
from __future__ import annotations

from typing import Optional

from . import DERIVED, DerivationContext, register

__all__ = ["deriveRadiusOnlyResonances"]


def _targetSpin(suite) -> Optional[float]:
    pops = getattr(suite, "PoPs", None)
    particle = pops.particles.get(suite.target) if pops is not None else None
    spin = getattr(particle, "spin", None)
    value = getattr(spin, "value", spin)
    return None if value is None else float(value)


def _domain(radius, suite, report):
    if radius.domainMin is not None and radius.domainMax is not None:
        return float(radius.domainMin), float(radius.domainMax)
    from ..mf1_header import _evaluatedStyle

    style = _evaluatedStyle(suite)
    domain = getattr(style, "projectileEnergyDomain", None)
    if domain is None:
        return None
    report.approximated(
        f"MF2/151: the scattering radius states no energy domain, so its range "
        f"is written over the evaluation's projectile domain "
        f"{domain.min:g}-{domain.max:g} eV")
    return float(domain.min), float(domain.max)


def deriveRadiusOnlyResonances(resonances, path, context: DerivationContext, report):
    """The provenance :func:`encodeMF2MT151` writes an LRU=0 range from, or ``None``."""
    from kika.nuclear_data.model import EndfProvenance
    from kika.nuclear_data.model.resonances import radiusToEndf

    if resonances.resolved or resonances.unresolved is not None:
        return None
    radius = resonances.scatteringRadius
    if radius is None:
        return None
    if radius.isEnergyDependent or radius.constant is None:
        report.lost(
            "MF2/151: the scattering radius depends on energy and there is no "
            "resonance range for it to belong to; an LRU=0 range states a "
            "constant AP, so MF2 is not written")
        return None
    suite = context.suite
    spin = _targetSpin(suite)
    if spin is None:
        report.lost(
            f"MF2/151: PoPs states no spin for the target {suite.target!r}, and "
            f"SPI is not assumed, so MF2 is not written")
        return None
    domain = _domain(radius, suite, report)
    if domain is None:
        report.lost("MF2/151: neither the scattering radius nor the evaluation "
                    "states an energy domain, so MF2 is not written")
        return None
    region = {"kind": "radiusOnly", "lru": 0, "lrf": 0, "nro": 0, "naps": 0,
              "el": domain[0], "eh": domain[1], "spi": spin,
              "ap": float(radiusToEndf(radius.constant)), "isotope_index": 0}
    header = {"nis": 1,
              "isotopes": [{"za": context.za, "abn": 1.0, "lfw": 0, "ner": 1}],
              "regions": [region]}
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=context.za,
                          awr=context.awr, headerFields=header)


def _isResonances(node) -> bool:
    from kika.nuclear_data.model import Resonances

    return isinstance(node, Resonances)


register("resonances", _isResonances, deriveRadiusOnlyResonances)
