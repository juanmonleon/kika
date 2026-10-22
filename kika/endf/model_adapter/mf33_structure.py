"""MF33's declared groups and covariance sums, without inventing member matrices."""
from __future__ import annotations

from dataclasses import replace


def _mixed(components):
    from kika.nuclear_data.model import Mixed

    # GNDS components are addressable by label within their parent.
    for index, component in enumerate(components):
        component.label = str(index)
    return Mixed(components=components, label="eval")


def decodeSums(section, report, xsSections):
    """Keep every NC LTY=0 range as a reference form, alongside any explicit NI."""
    from kika.nuclear_data.model import CovarianceSection, DataLink, Mixed, Sum, Summand
    from .covariances import decodeMF33MT, reactionHref, _sectionProvenance

    provenance = _sectionProvenance(section)
    output = []
    for sub in section.subsections:
        if sub.mat1 not in (0, section._mat):
            report.lost(f"MF33/MT{section.number}: NC-containing block with external "
                        f"MAT1={sub.mat1} cannot be linked from this evaluation; not relabelled as local")
            continue
        forms = []
        if sub.ni_records:
            ni = replace(sub, nc=0, nc_records=[])
            explicit, _ = decodeMF33MT(replace(section, _subsections=[ni], _nl=1),
                                      report, xsSections=xsSections)
            forms.extend(s.form for s in explicit)
        for nc in sub.nc_records:
            if nc.lty != 0:
                report.lost(f"MF33/MT{section.number}: NC LTY={nc.lty} ratio to a standard "
                            "has no model form; its explicit NI components are retained")
                continue
            if len(nc.ci) != len(nc.xmti):
                raise ValueError(f"MF33/MT{section.number}: unequal NC coefficients and targets")
            terms = []
            for coefficient, target in zip(nc.ci, nc.xmti):
                if float(target) != int(target):
                    raise ValueError(f"MF33/MT{section.number}: nonintegral NC target {target}")
                mt = int(target)
                terms.append(Summand(
                    href=f"/covarianceSuite/covarianceSections/covarianceSection[@label='MF33-MT{mt}']",
                    coefficient=float(coefficient), ENDF_MFMT=f"33,{mt}"))
            forms.append(Sum(summands=terms, label="eval", domainMin=nc.e1,
                             domainMax=nc.e2, domainUnit="eV"))
        if not forms:
            continue
        mt, other = int(section.number), int(sub.mt1)
        label = f"MF33-MT{mt}" + (f"-MT{other}" if other != mt else "")
        # A single GNDS form can retain disjoint rules for the same block.
        previous = next((s for s in output if s.label == label), None)
        if previous is not None:
            components = previous.form.components if isinstance(previous.form, Mixed) else [previous.form]
            previous.form = _mixed(components + forms)
            continue
        output.append(CovarianceSection(label=label,
            rowData=DataLink(reactionHref(mt), ENDF_MFMT=f"33/{mt}"),
            columnData=DataLink(reactionHref(other), ENDF_MFMT=f"33/{other}") if other != mt else None,
            form=forms[0] if len(forms) == 1 else _mixed(forms),
            provenance=provenance))
    return output, report


def attachLumps(endf, suite, report):
    """Represent MTL membership in the companion reactionSuite's native sums."""
    from kika.algebra import add
    from kika.algebra.laws import interval_laws
    from kika.nuclear_data.model import (Add, CrossSection, CrossSectionSum, OutputChannel,
        Q, ReactionId, Summands, XYs1d, crossSectionAxes)
    from kika.nuclear_data.model.endf_conversion import reactionHref

    mf33 = endf.mf.get(33)
    if mf33 is None:
        return set()
    byMT = {r.ENDF_MT:r for r in list(suite.reactions)+list(suite.sums)}
    groups = {}
    for mt, section in mf33.mt.items():
        if section._mtl:
            if section.subsections:
                report.lost(f"MF33/MT{mt}: lumped member also carries subsections; inconsistent membership")
                continue
            groups.setdefault(int(section._mtl), []).append(mt)
    retained = set()
    mf3 = endf.mf.get(3)
    for lump, members in sorted(groups.items()):
        if not 851 <= lump <= 870 or lump not in mf33.mt:
            report.lost(f"MF33 group MT{lump}: invalid or absent lump section")
            continue
        if any(mt not in byMT for mt in members) or mf3 is None:
            report.lost(f"MF33 group MT{lump}: missing member cross sections; membership cannot be linked")
            continue
        links=[Add(reactionHref(byMT[mt])+"/crossSection") for mt in members]
        if lump in byMT:
            previous=byMT[lump]
            if not isinstance(previous,CrossSectionSum) or list(previous.summands)!=links:
                raise ValueError(f"MF33 group MT{lump}: conflicts with an existing quantity")
            retained.update(members)
            continue
        tables=[]
        for mt in members:
            s=mf3.mt.get(mt)
            if s is None:
                break
            # Preserve each member's interpolation and threshold steps through algebra.add.
            laws=interval_laws(len(s.energies),s.energy_interpolation)
            tables.append((s.energies,s.cross_sections,laws))
        if len(tables) != len(members):
            report.lost(f"MF33 group MT{lump}: missing MF3 member; no partial sum was created")
            continue
        x,y=add(tables)
        if any(any(law > 2 for law in laws) for _,_,laws in tables):
            from kika.algebra.arithmetic import LINEARIZATION_TOLERANCE
            report.approximated(f"MF33 group MT{lump}: nonlinear MF3 member curves "
                                f"linearized to {LINEARIZATION_TOLERANCE:g} for the nominal sum; "
                                "original members and covariance declarations are unchanged")
        cross=CrossSection(); cross["eval"]=XYs1d(x,y,label="eval",axes=crossSectionAxes())
        # As in GNDS's lumped sums, Q=0 is bookkeeping, not a transport channel.
        node=CrossSectionSum(id=ReactionId(label=f"MT{lump}",ENDF_MT=lump),
            crossSection=cross, outputChannel=OutputChannel(Q=Q(value=0.,unit="eV")),
            summands=Summands(links))
        suite.sums.append(node)
        retained.update(members)
    return retained


def bindReactionLinks(covariances, reactions):
    """Point rows and columns to the actual sum/reaction containers, not guessed MTs."""
    from kika.nuclear_data.model.endf_conversion import reactionHref
    byMT={r.ENDF_MT:r for r in list(reactions.reactions)+list(reactions.sums)}
    for section in covariances.covarianceSections:
        for link in (section.rowData,section.columnData):
            if (link is not None and link.ENDF_MF==33 and link.ENDF_MT in byMT
                    and not link.href.startswith("$")):
                link.href=reactionHref(byMT[link.ENDF_MT])+"/crossSection"


def bindSumLinks(covariances, report):
    """GNDS sum terms point at covariance forms, including mixed and derived forms."""
    from kika.nuclear_data.model import CovarianceMatrix, Mixed, Sum
    byMT={s.rowData.ENDF_MT:s for s in covariances.covarianceSections
          if s.rowData is not None and s.rowData.ENDF_MF==33 and s.columnData is None}
    tags={CovarianceMatrix:"covarianceMatrix",Mixed:"mixed",Sum:"sum"}
    for section in covariances.covarianceSections:
        if section.rowData is None or section.rowData.ENDF_MF != 33:
            continue
        forms=section.form.components if isinstance(section.form,Mixed) else [section.form]
        for form in forms:
            if not isinstance(form,Sum):
                continue
            for term in form.summands:
                mt=int(term.ENDF_MFMT.split(",")[1])
                target=byMT.get(mt)
                if target is None:
                    report.warn(f"{section.label}: NC target MF33/MT{mt} has no self covariance form; reference retained unresolved")
                    continue
                target.form.label=target.form.label or "eval"
                tag=tags[type(target.form)]
                term.href=(f"/covarianceSuite/covarianceSections/covarianceSection[@label='{target.label}']"
                           f"/{tag}[@label='{target.form.label}']")
