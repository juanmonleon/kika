"""ACE's secondary-neutron blocks → the model's Monte Carlo forms (ACE roadmap A3).

AND, DLW, TYR, NU, DNU, BDD and DNED become products of the reactions
:func:`~kika.ace.model_adapter.decode.decodeAce` builds, in the forms the HDF5
decoder gives an OpenMC file (:mod:`kika.hdf5.distributions`), labelled
``MonteCarlo``, in eV:

=====================================  =========================================
ACE                                    model
=====================================  =========================================
elastic AND                            :class:`AngularTwoBody`
LAND = 0 / AND record                  :class:`Isotropic2d` / ``XYs2d`` of
                                       ``xs_pdf_cdf1d`` in mu (32 equiprobable
                                       bins: a histogram, CDF k/32)
law 2                                  :class:`DiscretePhotonEnergy`
law 3, 33                              :class:`LevelEnergy`
law 1, 4                               ``XYs2d`` of ``xs_pdf_cdf1d`` in E'
                                       (law 1: its equiprobable bins)
law 5, 7, 9, 11                        :class:`GeneralEvaporation`,
                                       :class:`SimpleMaxwellianFission`,
                                       :class:`Evaporation`, :class:`Watt`
law 44                                 :class:`KalbachMann`
law 61                                 :class:`EnergyAngularMC`
law 66                                 :class:`NBodyPhaseSpace`
LNW chain with P(E)                    :class:`DistributionMixture`, one
                                       ``Weighted`` per law (a chain of one too,
                                       as OpenMC keeps its ``applicability``)
TY                                     multiplicity (a constant, or the TY > 100
                                       table) and the frame (TY < 0: CM)
NU / DNU, BDD, DNED                    the prompt (or total) fission neutron, the
                                       total-ν ``multiplicitySum`` (MT 452), and
                                       one :class:`DelayedNeutron` per group
=====================================  =========================================

How this differs from OpenMC's ``IncidentNeutron.from_ace``, on purpose:

- an isotropic reaction is :class:`Isotropic2d`, not a two-point table: the
  file says "isotropic", and the C++ reads a missing angle the same way;
- a delayed group's multiplicity is DNU(E)·P_g(E), not renormalised so that
  the group probabilities sum to one: the file's probabilities are kept;
- a DNED group with a chain keeps every law (OpenMC keeps the first);
- negative cross sections are not set to zero here.

Laws 22, 24 and 67 have no model form (and no witness in any library on the
share): they are reported, not converted. What the forms cannot hold and the
ACE writer needs back (the TY, the 32-bin form, law 66's tabulated part...)
goes to the ``attributes`` of an :class:`AceProvenance` on the product.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from kika.nuclear_data.model import (AceProvenance, ConversionReport, Distribution, Isotropic2d,
                                     Multiplicity, PhysicalQuantity, Product, Xs_pdf_cdf1d, XYs1d,
                                     XYs2d)

#: ACE energies are MeV; the model's are eV (OpenMC ``EV_PER_MEV``).
EV_PER_MEV = 1.0e6
#: ACE decay constants are per shake (1e-8 s).
PER_SHAKE_TO_PER_S = 1.0e8
MONTE_CARLO_LABEL = "MonteCarlo"
TOTAL_NUBAR_LABEL = "total fission neutron multiplicity"
FISSION_MTS = (18, 19, 20, 21, 38)


def _m():
    from kika.nuclear_data import model

    return model


# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------

def _function(x, y, nbt=(), interp=(), xScale=EV_PER_MEV, yScale=1.0):
    """A TAB1 (Table 30's NR, NBT, INT, NE, x, y) as XYs1d or Regions1d."""
    m = _m()
    x = np.asarray(x, dtype=float) * xScale
    y = np.asarray(y, dtype=float) * yScale
    nbt = [int(v) for v in nbt]
    interp = [int(v) for v in interp]
    if not nbt:
        nbt, interp = [len(x)], [2]
    if len(nbt) == 1:
        return XYs1d(xs=x, ys=y, interpolation=m.ENDF_INT_TO_INTERPOLATION[interp[0]])
    return m.Regions1d.fromEndfRegions(x, y, list(zip(nbt, interp)))


def _applicability(law):
    return _function(law.applicability_energies, law.applicability_probabilities,
                     law.applicability_nbt, law.applicability_interp)


def _incident(functions, nbt, interp, n):
    m = _m()
    nbt = [int(v) for v in nbt] or [n]
    interp = [int(v) for v in interp] or [2]
    return m.fromEndfTab2(functions, list(zip(nbt, interp)))


def _interpolation(code):
    m = _m()
    return m.Interpolation.flat if int(code) == 1 else m.Interpolation.linlin


def _eoutTable(t: dict, incident: float) -> Xs_pdf_cdf1d:
    """One outgoing-energy table of laws 4/44/61: discrete lines first, pdf in 1/eV."""
    nd = int(t["n_discrete"])
    pdf = np.array(t["pdf"], dtype=float)
    pdf[nd:] = pdf[nd:] / EV_PER_MEV
    # A table of lines only (NJOY writes INTT' = 10·ND, INTT = 0) is a histogram, as OpenMC writes it
    intt = 1 if nd == len(pdf) else t["intt"]
    return Xs_pdf_cdf1d(xs=np.asarray(t["e_out"], dtype=float) * EV_PER_MEV, pdf=pdf,
                        cdf=np.array(t["cdf"], dtype=float), interpolation=_interpolation(intt),
                        numberOfDiscrete=nd, outerDomainValue=float(incident) * EV_PER_MEV)


def _isotropicTable(outer) -> Xs_pdf_cdf1d:
    """Isotropic as a table, where a form needs one (OpenMC ``Uniform.to_tabular``)."""
    m = _m()
    return Xs_pdf_cdf1d(xs=np.array([-1.0, 1.0]), pdf=np.array([0.5, 0.5]), cdf=np.array([0.0, 1.0]),
                        interpolation=m.Interpolation.flat, outerDomainValue=float(outer))


def _equiprobableTable(edges, outer) -> Xs_pdf_cdf1d:
    """32 equiprobable cosine bins as a histogram (OpenMC ``AngleDistribution.from_ace``)."""
    m = _m()
    edges = np.asarray(edges, dtype=float)
    n = edges.size - 1
    pdf = np.zeros(n + 1)
    pdf[:n] = 1.0 / (n * np.diff(edges))
    return Xs_pdf_cdf1d(xs=edges, pdf=pdf, cdf=np.linspace(0.0, 1.0, n + 1),
                        interpolation=m.Interpolation.flat, outerDomainValue=float(outer))


# ----------------------------------------------------------------------
# Angular distributions (AND, ANDP)
# ----------------------------------------------------------------------

def angularForm(dist):
    """An AND record → ``Isotropic2d`` or an ``XYs2d`` of ``xs_pdf_cdf1d`` in mu; None for LOCB = -1."""
    m = _m()
    kind = type(dist).__name__
    if kind == "KalbachMannAngularDistribution":
        return None
    if kind == "IsotropicAngularDistribution":
        return Isotropic2d()
    energies = np.asarray(dist._energies, dtype=float)
    tables = []
    for k, e in enumerate(energies):
        outer = float(e) * EV_PER_MEV
        form = (dist.energy_forms[k] if getattr(dist, "energy_forms", None) else
                ("equiprobable" if kind == "EquiprobableAngularDistribution" else "tabulated"))
        if form == "isotropic":
            tables.append(_isotropicTable(outer))
        elif form == "equiprobable":
            edges = dist._cosine_bins[k] if kind == "EquiprobableAngularDistribution" else dist._cosine_grid[k]
            tables.append(_equiprobableTable(edges, outer))
        else:
            tables.append(Xs_pdf_cdf1d(xs=np.asarray(dist._cosine_grid[k], dtype=float),
                                       pdf=np.asarray(dist._pdf[k], dtype=float),
                                       cdf=np.asarray(dist._cdf[k], dtype=float),
                                       interpolation=_interpolation(dist.interpolation[k]),
                                       outerDomainValue=outer))
    return XYs2d(function1ds=tables, label=MONTE_CARLO_LABEL)


# ----------------------------------------------------------------------
# Energy laws (DLW)
# ----------------------------------------------------------------------

def _law4(law):
    tables = [_eoutTable(t, e) for t, e in zip(law.distributions, law.incident_energies)]
    return _incident(tables, law.nbt, law.interp, len(tables))


def _law1(law):
    m = _m()
    tables = []
    for t, e in zip(law.distribution_data, law.incident_energies):
        edges = np.asarray(t["e_out"], dtype=float) * EV_PER_MEV
        n = edges.size - 1
        pdf = np.zeros(n + 1)
        pdf[:n] = 1.0 / (n * np.diff(edges))
        tables.append(Xs_pdf_cdf1d(xs=edges, pdf=pdf, cdf=np.linspace(0.0, 1.0, n + 1),
                                   interpolation=m.Interpolation.flat,
                                   outerDomainValue=float(e) * EV_PER_MEV))
    return _incident(tables, law.nbt, law.interp, len(tables))


def _law44(law, frame):
    m = _m()
    f, r, a = [], [], []
    for t, e in zip(law.distributions, law.incident_energies):
        table = _eoutTable(t, e)
        f.append(table)
        x = table.xs
        r.append(XYs1d(xs=x, ys=np.asarray(t["r"], dtype=float), interpolation=table.interpolation,
                       outerDomainValue=table.outerDomainValue))
        a.append(XYs1d(xs=x, ys=np.asarray(t["a"], dtype=float), interpolation=table.interpolation,
                       outerDomainValue=table.outerDomainValue))
    n = len(f)
    return m.KalbachMann(f=_incident(f, law.nbt, law.interp, n), r=_incident(r, law.nbt, law.interp, n),
                         a=_incident(a, law.nbt, law.interp, n), label=MONTE_CARLO_LABEL, productFrame=frame)


def _law61(law, frame):
    m = _m()
    byLC = {int(t["lc"]): t for t in law.angular_tables}
    energy, inner = [], []
    for t, e in zip(law.distributions, law.incident_energies):
        table = _eoutTable(t, e)
        energy.append(table)
        mus = []
        for xOut, lc in zip(table.xs, t["lc"]):
            if int(lc) == 0:
                mus.append(_isotropicTable(xOut))
                continue
            a = byLC[int(lc)]
            mus.append(Xs_pdf_cdf1d(xs=np.asarray(a["cosines"], dtype=float),
                                    pdf=np.asarray(a["pdf"], dtype=float),
                                    cdf=np.asarray(a["cdf"], dtype=float),
                                    interpolation=_interpolation(a["jj"]), outerDomainValue=float(xOut)))
        inner.append(XYs2d(function1ds=mus, outerDomainValue=table.outerDomainValue))
    energyForm = _incident(energy, law.nbt, law.interp, len(energy))
    xys3d = m.XYs3d(function2ds=inner)
    if isinstance(energyForm, XYs2d):
        xys3d.interpolation = energyForm.interpolation
        xys3d.interpolationQualifier = energyForm.interpolationQualifier
    return m.EnergyAngularMC(energy=energyForm, energyAngular=xys3d, label=MONTE_CARLO_LABEL, productFrame=frame)


def _pq(value, unit="eV"):
    return PhysicalQuantity(value=float(value), unit=unit)


def _spectrum(law):
    """Laws 5/7/9/11 → the §18.3 parametrised spectra, parameters in eV (OpenMC's units)."""
    m = _m()
    if law.law == 7:
        return m.SimpleMaxwellianFission(
            U=_pq(law.restriction_energy * EV_PER_MEV),
            theta=_function(law.incident_energies, law.temperatures, law.temp_nbt, law.temp_interp,
                            yScale=EV_PER_MEV))
    if law.law == 9:
        return m.Evaporation(
            U=_pq(law.restriction_energy * EV_PER_MEV),
            theta=_function(law.incident_energies, law.temperatures, law.temp_nbt, law.temp_interp,
                            yScale=EV_PER_MEV))
    if law.law == 11:
        return m.Watt(
            U=_pq(law.restriction_energy * EV_PER_MEV),
            a=_function(law.a_incident_energies, law.a_values, law.a_nbt, law.a_interp, yScale=EV_PER_MEV),
            b=_function(law.b_incident_energies, law.b_values, law.b_nbt, law.b_interp, yScale=1.0 / EV_PER_MEV))
    if law.law == 5:
        x = np.asarray(law.equiprob_values, dtype=float)
        return m.GeneralEvaporation(
            theta=_function(law.incident_energies, law.temperatures, law.temp_nbt, law.temp_interp,
                            yScale=EV_PER_MEV),
            g=XYs1d(xs=x, ys=np.full(x.size, 1.0 / max(x.size - 1, 1)), interpolation=m.Interpolation.flat))
    raise ValueError(f"LAW={law.law} is not a parametrised spectrum")


def lawForm(law, angular, frame, awr: float, report: ConversionReport, where: str):
    """One energy law (with the reaction's AND form, for the uncorrelated ones) → a model form."""
    m = _m()
    uncorrelated = None
    if law.law == 2:
        uncorrelated = m.DiscretePhotonEnergy(primaryFlag=int(law.lp), energy=float(law.eg) * EV_PER_MEV,
                                              atomicWeightRatio=float(awr))
    elif law.law == 3:
        uncorrelated = m.LevelEnergy(threshold=float(law.aplusoaabsq) * EV_PER_MEV, massRatio=float(law.asquare))
    elif law.law == 33:
        uncorrelated = m.LevelEnergy(threshold=float(law.threshold) * EV_PER_MEV,
                                     massRatio=float(law.energy_factor))
    elif law.law == 4:
        uncorrelated = _law4(law)
    elif law.law == 1:
        uncorrelated = _law1(law)
    elif law.law in (5, 7, 9, 11):
        uncorrelated = _spectrum(law)
    elif law.law == 66:
        from kika._constants import NEUTRON_MASS_AMU

        return m.Uncorrelated(angular=Isotropic2d(),
                              energy=m.NBodyPhaseSpace(numberOfProducts=int(law.npsx),
                                                       mass=_pq(float(law.ap) * NEUTRON_MASS_AMU, "amu")),
                              label=MONTE_CARLO_LABEL, productFrame=frame)
    elif law.law == 44:
        return _law44(law, frame)
    elif law.law == 61:
        return _law61(law, frame)
    else:
        report.unsupportedNode(f"{where}: ACE energy law {law.law} has no model form; not converted")
        return None
    return m.Uncorrelated(angular=angular if angular is not None else Isotropic2d(), energy=uncorrelated,
                          label=MONTE_CARLO_LABEL, productFrame=frame)


def chainDistribution(chain, angular, frame, awr, report, where, applicability=True):
    """An LNW chain → a ``Distribution`` holding one form, or a ``DistributionMixture``."""
    m = _m()
    weighted = []
    for law in chain:
        form = lawForm(law, angular, frame, awr, report, where)
        if form is None:
            return None
        weighted.append(m.Weighted(weight=_applicability(law) if applicability else None, functional=form))
    distribution = Distribution()
    if len(weighted) == 1 and not applicability:
        distribution[MONTE_CARLO_LABEL] = weighted[0].functional
    else:
        distribution[MONTE_CARLO_LABEL] = m.DistributionMixture(weighted=weighted, label=MONTE_CARLO_LABEL)
    return distribution


# ----------------------------------------------------------------------
# Multiplicities
# ----------------------------------------------------------------------

def _constant(n):
    m = _m()
    return m.Polynomial1d(coefficients=np.array([float(n)]), domainMin_=0.0, domainMax_=float("inf"),
                          label=MONTE_CARLO_LABEL)


def _yieldTable(y):
    return _function(y.energies, y.yields, y.nbt, y.interp)


def _nuForm(nu):
    """A NU/DNU record → Polynomial1d (coefficients in powers of E/eV) or a table."""
    m = _m()
    if nu.format == "polynomial":
        c = np.array(nu.polynomial.coefficients, dtype=float)
        for i in range(c.size):
            c[i] *= EV_PER_MEV ** (-i)
        return m.Polynomial1d(coefficients=c, domainMin_=0.0, domainMax_=float("inf"), label=MONTE_CARLO_LABEL)
    t = nu.tabulated
    return _function(t.energies, t.nubar_values, [p[0] for p in t.interpolation_regions],
                     [p[1] for p in t.interpolation_regions])


def _abscissae(f):
    return f.xs if isinstance(f, XYs1d) else f.toEndfRegions()[0]


def _scaled(f, factor):
    """``factor * f`` on f's own points and laws."""
    m = _m()
    if isinstance(f, XYs1d):
        return XYs1d(xs=f.xs.copy(), ys=f.ys * factor, interpolation=f.interpolation)
    x, y, pairs = f.toEndfRegions()
    return m.Regions1d.fromEndfRegions(x, np.asarray(y) * factor, pairs)


def _multiplicity(form):
    form.label = MONTE_CARLO_LABEL
    multiplicity = Multiplicity()
    multiplicity[MONTE_CARLO_LABEL] = form
    return multiplicity


# ----------------------------------------------------------------------
# Products of the neutron reactions
# ----------------------------------------------------------------------

def _neutron(multiplicityForm, distribution, attributes):
    product = Product(pid="n", label="n", multiplicity=_multiplicity(multiplicityForm), distribution=distribution)
    product.provenance = AceProvenance(attributes=attributes)
    return product


def attachAceSecondaries(ace, suite, report: Optional[ConversionReport] = None) -> ConversionReport:
    """Give each reaction of ``suite`` (from :func:`decodeAce`) its secondary neutrons.

    Elastic gets its angular distribution; every reaction with secondary
    neutrons its neutron product; the fission reactions their prompt (or
    total) and delayed neutrons, and the suite the total ν.
    """
    m = _m()
    report = report if report is not None else ConversionReport()
    awr = float(ace.header.atomic_weight_ratio)
    ad, ed = ace.angular_distributions, ace.energy_distributions

    elastic = suite.findReactionByENDF_MT(2)
    if elastic is not None and ad is not None and ad.elastic is not None:
        angular = angularForm(ad.elastic)
        distribution = Distribution()
        distribution[MONTE_CARLO_LABEL] = m.AngularTwoBody(
            angular=angular, productFrame=m.Frame.centerOfMass, label=MONTE_CARLO_LABEL)
        elastic.outputChannel.products.products.append(
            _neutron(_constant(1), distribution, {"LOCB": ace.angular_locators.elastic_scattering}))

    mts = [int(v) for v in ace.reaction_mt_data.incident_neutron]
    tys = dict(zip(mts, (int(v) for v in ace.particle_release.incident_neutron)))
    delayed = _delayedNeutrons(ace, awr, report)
    total = None
    for mt in (int(v) for v in ace.reaction_mt_data.secondary_neutron_mt):
        reaction = suite.findReactionByENDF_MT(mt)
        if reaction is None:
            report.lost(f"MT{mt}: secondary neutrons in DLW, but no reaction to attach them to")
            continue
        ty = tys[mt]
        frame = m.Frame.centerOfMass if ty < 0 else m.Frame.lab
        dist = ad.incident_neutron.get(mt) if ad is not None else None
        angular = angularForm(dist) if dist is not None else None
        distribution = chainDistribution(ed.incident_neutron.get(mt, []), angular, frame, awr, report,
                                         f"DLW MT{mt}")
        attributes = {"TY": ty, "angular": type(dist).__name__ if dist is not None else None}
        channel = reaction.outputChannel
        if ty == 19:
            prompt, total, mode = _fissionNeutron(ace)
            if prompt is None:
                report.lost(f"MT{mt}: TY=19 but no NU block; the fission neutrons are not converted")
                continue
            attributes["emission_mode"] = mode
            channel.products.products.append(_neutron(prompt, distribution, attributes))
            if delayed:
                channel.fissionFragmentData = m.FissionFragmentData()
                for d in delayed:
                    channel.fissionFragmentData.delayedNeutrons.append(d)
            continue
        if abs(ty) > 100:
            form = _yieldTable(ed.neutron_yields[mt])
        else:
            form = _constant(abs(ty))
        channel.products.products.append(_neutron(form, distribution, attributes))

    if total is not None:
        fission = next((suite.findReactionByENDF_MT(mt) for mt in FISSION_MTS
                        if suite.findReactionByENDF_MT(mt) is not None), None)
        multiplicitySum = m.MultiplicitySum(label=TOTAL_NUBAR_LABEL, multiplicity=_multiplicity(total),
                                            ENDF_MT=452)
        if fission is not None:
            multiplicitySum.summands.append(m.Add(
                href=f"/reactionSuite/reactions/reaction[@label='{fission.label}']"
                     f"/outputChannel/products/product[@label='n']/multiplicity"))
        suite.sums.multiplicitySums.append(multiplicitySum)
    return report


def _fissionNeutron(ace):
    """``(prompt-or-total ν, total ν or None, emission mode)`` from NU, as OpenMC reads it."""
    nu = ace.nubar
    if nu is None or not nu.has_nubar or nu.total is None:
        return None, None, None
    if nu.has_both_nu_types:
        return _nuForm(nu.prompt), _nuForm(nu.total), "prompt"
    return _nuForm(nu.total), None, ("prompt" if nu.has_delayed else "total")


def _delayedNeutrons(ace, awr, report) -> List:
    """One DelayedNeutron per BDD group: rate in 1/s, multiplicity DNU(E)·P_g(E), DNED spectrum."""
    m = _m()
    nu = ace.nubar
    bdd = ace.delayed_neutron_data
    if nu is None or not nu.has_delayed or bdd is None or not bdd.precursors:
        return []
    dnu = _nuForm(nu.delayed)
    out = []
    chains = ace.energy_distributions.delayed_neutron
    for g, p in enumerate(bdd.precursors):
        prob = np.asarray(p.probabilities, dtype=float)
        if np.all(prob == prob[0]):
            form = _scaled(dnu, prob[0])
        else:
            pg = _function(p.energies, prob, [r[0] for r in p.interpolation_regions],
                           [r[1] for r in p.interpolation_regions])
            top = min(dnu.domainMax, pg.domainMax)
            x = np.union1d(_abscissae(dnu), _abscissae(pg))
            x = x[x <= top]
            form = XYs1d(xs=x, ys=np.asarray(dnu.evaluate(x)) * np.asarray(pg.evaluate(x)))
        chain = chains[g] if g < len(chains) else []
        distribution = chainDistribution(chain, None, m.Frame.lab, awr, report, f"DNED group {g + 1}",
                                         applicability=len(chain) > 1)
        product = _neutron(form, distribution, {"group": g + 1, "probabilities": prob.tolist()})
        out.append(m.DelayedNeutron(label=str(g + 1),
                                    rate=PhysicalQuantity(value=float(p.decay_constant) * PER_SHAKE_TO_PER_S,
                                                          unit="1/s"),
                                    product=product))
    return out
