"""GNDS-2.1 §14.1.1 ``reactionSuite`` — the evaluation root.

Attributes (§14.1.1, all required): ``evaluation``, ``format``, ``projectile``,
``projectileFrame``, ``target``, ``interaction``. Children: ``externalFiles``,
``styles``, ``PoPs``, ``resonances``, ``reactions``, ``orphanProducts``,
``sums``, ``fissionComponents``, ``productions``, ``incompleteReactions``,
``applicationData``.

**Every child is present as an empty container, never as ``None`` -- except
``resonances``,** which is ``None`` when the evaluation states no resonance
data. §14.1.2
notes that most children are optional so libraries can set their own
requirements, which is exactly what lets kika ship a partially-filled hierarchy
honestly — a slot that exists and is empty says "kika models this and this file
has none", while a missing attribute says nothing at all and fails three frames
away. The list containers even define ``__bool__`` as ``True`` so that
``if suite.reactions:`` cannot silently read *empty* as *absent*.

MF12-15 have slots here that were later filled (the photon products of a
reaction, ``orphanProducts`` and ``multiplicitySum``), as MF5 and MF6 had
filled theirs. Filling them restructured nothing, which was the claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .conversion import ConversionReport
from .covariances import CovarianceSuite
from .cross_section_forms import EVAL_LABEL
from .enums import Frame
from .functions import Regions1d, XYs1d
from .pops import PoPs
from .reactions import (FissionComponents, IncompleteReactions, OrphanProducts,
                        Productions, Reaction, Reactions, Sums)
from .resonances import Resonances
from .styles import Styles

__all__ = ["ExternalFile", "ExternalFiles", "ApplicationData", "ReactionSuite",
           "CROSS_SECTION_UNITS"]

#: §14.1.1 ``interaction``. The spec's three values.
INTERACTIONS = ("nuclear", "atomic", "thermalNeutronScatteringLaw")

#: This library targets GNDS 2.1. FUDGE 6.10.0 emits 2.0, so a file read from
#: FUDGE will not match and the reader is expected to warn rather than pretend.
GNDS_FORMAT = "2.1"


def _eV(value: float) -> str:
    """An energy in eV, rendered at a scale a human reads without counting zeros."""
    for scale, unit in ((1e6, "MeV"), (1e3, "keV")):
        if abs(value) >= scale:
            return f"{value / scale:g} {unit}"
    return f"{value:g} eV"


#: What :meth:`ReactionSuite.cross_section` hands back, as ``(dependent,
#: independent)`` units. Not a preference — it is what every consumer of those
#: bare arrays already assumes, from the reconstructor to the chi2 pipeline.
CROSS_SECTION_UNITS = ("b", "eV")


def _checkUnits(form: object, expected: Tuple[str, str], what: str) -> None:
    """Refuse to hand back bare arrays whose axes say something else.

    **Why this exists, and why it has never fired.** A unit lives on an
    ``axis`` (§5.1.2) and this method returns two numpy arrays, so the unit is
    dropped at exactly this line. That is fine while every file agrees on
    ``b``/``eV`` — and they do: a census of all 558 neutron evaluations of
    ENDF/B-VIII.1-GNDS found **35 259 ``crossSection`` axes, every one ``b``**,
    and **475 003 ``energy_in`` axes, every one ``eV``** (2026-08-13). The ENDF
    and ACE adapters both fill :func:`~kika.nuclear_data.model.axes.
    crossSectionAxes`, which states the same pair.

    So the guarantee is in *the data*, not in the code, and this puts it in the
    code. ``ScatteringRadius.constant`` is what the other outcome looks like:
    the same field meant 5.444 fm down one path and 0.5444 of a different unit
    down the other, for a year, because nothing asked.

    **It raises rather than converting.** kika does not rescale on read
    anywhere — the reconstructor works in ENDF's units throughout — and a
    convenience accessor is the last place to start. The model still holds the
    data and its axes; a caller meeting this reads ``form.axes`` and decides.

    An *unstated* unit is not an error. ``axes is None`` and ``unit == ""``
    both mean the source said nothing, which §2.3.3 distinguishes from saying
    the wrong thing, and inventing a complaint about silence would fire on
    every hand-built form in the test suite.
    """
    axes = getattr(form, "axes", None)
    if axes is None or getattr(axes, "href", None):
        return
    try:
        found = (axes.dependent.unit, axes.independent[0].unit)
    except (KeyError, IndexError, AttributeError):
        return
    wrong = [(f, e) for f, e in zip(found, expected) if f and f != e]
    if wrong:
        raise ValueError(
            f"{what} states its axes in "
            f"({found[0] or 'unstated'}, {found[1] or 'unstated'}) and this "
            f"shortcut returns bare arrays that every consumer reads as "
            f"({expected[0]}, {expected[1]}). kika does not rescale on read, so "
            f"the conversion is yours: reach for the form and its `axes` "
            f"directly."
        )


def _pointwise(form: object) -> Tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` arrays out of whichever 1d container a form turned out to be.

    Copies, so a caller who edits what they were handed does not edit the
    evaluation underneath it.
    """
    if isinstance(form, XYs1d):
        return form.xs.copy(), form.ys.copy()
    if isinstance(form, Regions1d):
        xs, ys, _ = form.toEndfRegions()
        return xs, ys
    raise TypeError(
        f"a {type(form).__name__} is not a tabulated function, so it has no "
        f"(E, sigma) pair to hand back. Forms such as ResonancesWithBackground "
        f"and Reference have to be reconstructed or dereferenced first; reach "
        f"for the form itself rather than this shortcut."
    )


@dataclass
class ExternalFile:
    """§14.1.1. A file this evaluation refers to — usually its covarianceSuite."""

    label: str
    path: str
    checksum: Optional[str] = None
    algorithm: Optional[str] = None   # §3.4.3: 'md5' or 'sha1'


@dataclass
class ExternalFiles:
    """The container, present even when empty."""

    files: List[ExternalFile] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self):
        return iter(self.files)

    def __bool__(self) -> bool:
        return True  # present-but-empty is not absent

    def byLabel(self, label: str) -> ExternalFile:
        for entry in self.files:
            if entry.label == label:
                return entry
        raise KeyError(f"no external file labelled {label!r}")


@dataclass
class ApplicationData:
    """§14.1.1. Application-specific data GNDS does not interpret."""

    entries: List[object] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return True


@dataclass
class ReactionSuite:
    """§14.1.1. One evaluation, of one projectile on one target.

    The root of the model. It is returned by the decoders (``kika.read`` on an
    ENDF, ACE or GNDS file) or built by hand from its fields. Every child is
    present as an empty container, never None, except ``resonances``, which is None
    when the evaluation states no resonance data. Energies are in eV and cross
    sections in b as the evaluations state them; kika does not rescale on read.

    Attributes
    ----------
    evaluation : str
        The evaluation identifier (library and version).
    projectile, target : str
        The projectile and the target, as the evaluation names them.
    projectileFrame : Frame
        The frame of the projectile's energy; a string is converted.
    interaction : {"nuclear", "atomic", "thermalNeutronScatteringLaw"}
        The kind of evaluation. Any other value raises ``ValueError`` on
        construction.
    format : str
        The GNDS version of the structure, ``"2.1"`` by default.
    externalFiles : ExternalFiles
        Files the evaluation refers to, usually its covariance suite.
    styles : Styles
        The styles of the data; the form labels of a cross section (``'eval'``,
        ``'recon'``) refer to them. ``styleLabels`` lists their labels.
    PoPs : PoPs
        The particle properties.
    resonances : Resonances, optional
        The resolved and unresolved resonance data, or None.
    reactions : Reactions
        The exclusive reactions, indexed by ENDF MT (``reactions[102]``) or by label.
    orphanProducts, sums, fissionComponents, productions, incompleteReactions
        The other reaction containers of GNDS, with the same access as *reactions*:
        products without a parent reaction, cross-section and multiplicity sums
        (MT 1, MT 4, and so on), the fission chances, production reactions and
        incomplete reactions.
    applicationData : ApplicationData
        Application-specific data that GNDS does not interpret.
    covarianceSuite : CovarianceSuite, optional
        The covariances read beside the evaluation (a separate root in GNDS), or
        None.
    provenance
        The ENDF header (MF 1 / MT 451) kept so that the encoder can write it back,
        or None.
    report : ConversionReport, optional
        What the decoder lost, approximated or refused; None for a suite built by
        hand.
    """

    evaluation: str
    projectile: str
    target: str
    projectileFrame: Frame = Frame.lab
    interaction: str = "nuclear"
    format: str = GNDS_FORMAT

    externalFiles: ExternalFiles = field(default_factory=ExternalFiles)
    styles: Styles = field(default_factory=Styles)
    PoPs: PoPs = field(default_factory=PoPs)
    resonances: Optional[Resonances] = None
    reactions: Reactions = field(default_factory=Reactions)
    orphanProducts: OrphanProducts = field(default_factory=OrphanProducts)
    sums: Sums = field(default_factory=Sums)
    fissionComponents: FissionComponents = field(default_factory=FissionComponents)
    productions: Productions = field(default_factory=Productions)
    incompleteReactions: IncompleteReactions = field(default_factory=IncompleteReactions)
    applicationData: ApplicationData = field(default_factory=ApplicationData)

    #: Not a GNDS child. kika keeps the covariances it has read beside the
    #: evaluation for convenience; in a GNDS file they are a separate root that
    #: `externalFiles` points at, and the writer must emit them that way.
    covarianceSuite: Optional[CovarianceSuite] = None

    #: Not a GNDS node either -- the MF1/451 header, kept so the encoder can
    #: write it back byte for byte. See Reaction.provenance.
    provenance: Optional[object] = None

    #: Not a GNDS node either. What the decode did beyond producing this object:
    #: what it lost, approximated or refused. The decoders keep returning it as
    #: the second element of a tuple -- that surface is unchanged -- but a tuple
    #: element is discarded by every caller who is in a hurry, and "which MFs did
    #: this evaluation carry that kika cannot read?" is a question the object has
    #: to be able to answer on its own. `kika.read()` fills this in; `repr` and
    #: `summary()` show its counts, so a partial decode announces itself.
    report: Optional[ConversionReport] = None

    def __post_init__(self) -> None:
        self.projectileFrame = Frame(self.projectileFrame)
        if self.interaction not in INTERACTIONS:
            raise ValueError(
                f"interaction must be one of {INTERACTIONS}, got {self.interaction!r}"
            )

    # ------------------------------------------------------------------

    def reactionByLabel(self, label: str) -> Reaction:
        """The exclusive reaction with this label.

        Parameters
        ----------
        label
            The reaction label, searched in ``reactions`` only.

        Returns
        -------
        Reaction

        Raises
        ------
        KeyError
            If no reaction of ``reactions`` has the label.
        """
        return self.reactions.byLabel(label)

    def reactionByENDF_MT(self, mt: int) -> Reaction:
        """Look-up by MT, for the ENDF adapters.

        Searches ``reactions``, ``sums``, ``fissionComponents``, ``productions`` and
        ``incompleteReactions``, in that order, and returns the first match.
        MT 1 and MT 4 are summed quantities and not exclusive reactions, and a
        caller asking for "MT 1" means the sum.

        Parameters
        ----------
        mt
            The ENDF MT number.

        Returns
        -------
        Reaction
            The first reaction with that ``ENDF_MT``.

        Raises
        ------
        KeyError
            If no container holds the MT.
        """
        for container in (self.reactions, self.sums, self.fissionComponents,
                          self.productions, self.incompleteReactions):
            try:
                return container.byENDF_MT(mt)
            except KeyError:
                continue
        raise KeyError(f"no reaction with ENDF_MT {mt} anywhere in this suite")

    def findReactionByENDF_MT(self, mt: int) -> Optional[Reaction]:
        """The same look-up as :meth:`reactionByENDF_MT`, returning ``None`` if absent.

        An encoder asking for the MT it is about wants :meth:`reactionByENDF_MT`,
        because its absence is a bug. A decoder asking whether there is a reaction
        to hang an MF 4 section on has a legitimate "no": an ENDF file may carry an
        MF 4 / MT section with no MF 3 / MT.

        Parameters
        ----------
        mt
            The ENDF MT number.

        Returns
        -------
        Reaction or None
        """
        try:
            return self.reactionByENDF_MT(mt)
        except KeyError:
            return None

    @property
    def hasResonances(self) -> bool:
        """True when the evaluation states resonance data (``resonances`` is not None)."""
        return self.resonances is not None

    def styleLabels(self) -> List[str]:
        """The labels of ``styles``."""
        return self.styles.labels

    def cross_section(
        self, mt: int, form: str = EVAL_LABEL
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(E, sigma)`` for one MT, as two arrays: the shortcut to a cross section.

        The long road, ``suite.reactions[102].crossSection.evaluated.xs``, says
        which *form* it is reading. This goes through :meth:`reactionByENDF_MT`,
        so summed quantities such as MT 1 and MT 4 work here even though they are
        not in ``reactions``.

        A ``Regions1d`` form is flattened onto one grid by its ``toEndfRegions``,
        which drops the boundary point the regions share. **The regions' differing
        interpolation rules do not survive that flattening**: the returned pair is
        faithful at its own nodes and nowhere else, so a value between two nodes
        must come from ``form.evaluate(E)``, which interpolates each region by its
        own rule.

        **The arrays are in eV and b**, and that is checked: a form whose axes
        state another unit raises instead of being rescaled. An unstated unit
        (no axes, or an empty unit) is not an error.

        Parameters
        ----------
        mt
            The ENDF MT number, searched as in :meth:`reactionByENDF_MT`.
        form
            The label of the cross-section form to read: ``'eval'`` by default, or
            for instance ``'recon'`` for a reconstructed one.

        Returns
        -------
        tuple of numpy.ndarray
            ``(E, sigma)``: the energies in eV and the cross sections in b, copies
            that may be edited without touching the evaluation.

        Raises
        ------
        KeyError
            If the MT is absent, or the reaction has no form with that label (the
            message lists the available labels).
        ValueError
            If the axes of the form state a unit other than eV and b.
        TypeError
            If the form is not a tabulated function (for instance a resonance
            form or a reference): reconstruct or dereference it first.
        """
        reaction = self.reactionByENDF_MT(mt)
        # Raises with the available labels listed -- CrossSection.__getitem__
        # already writes that message, so it is not rewritten here.
        chosen = reaction.crossSection[form]
        _checkUnits(chosen, CROSS_SECTION_UNITS,
                    f"the {form!r} cross section of MT{mt}")
        return _pointwise(chosen)

    def resolve_reference(self, reference) -> object:
        """The cross-section form a §16.1.1 ``reference`` points at, in this suite.

        Resolved by **labels**, the container's and the form's, never by the
        element tag in the href: the tag says how a GNDS writer happened to spell
        the form (a one-region ``regions1d`` is written as its ``XYs1d``), and
        the model holds the form whatever it is spelt as. Searches ``reactions``,
        ``sums``, ``fissionComponents``, ``productions`` and ``orphanProducts``.

        Parameters
        ----------
        reference
            A ``Reference`` form, whose ``href`` names a container label and a form
            label (``[@label='...']`` twice or more; the first and the last are used).

        Returns
        -------
        object
            The form itself (not a copy) of the first reaction whose label is the
            container label.

        Raises
        ------
        ValueError
            If the ``href`` names fewer than two labels.
        KeyError
            If no reaction of the containers above has the container label, or it
            has no form with the form label.
        """
        import re

        labels = re.findall(r"\[@label='([^']*)'\]", getattr(reference, "href", "") or "")
        if len(labels) < 2:
            raise ValueError(f"cannot resolve {reference!r}: the href does not name "
                             f"a container label and a form label")
        owner, formLabel = labels[0], labels[-1]
        for container in (self.reactions, self.sums, self.fissionComponents,
                          self.productions, self.orphanProducts):
            for reaction in container:
                if reaction.label == owner:
                    return reaction.crossSection[formLabel]
        raise KeyError(f"no reaction labelled {owner!r} for {reference!r}")

    def production_cross_section(
        self, production, form: str = EVAL_LABEL
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(E, σ_production)`` of one ``production``: its cross section times its multiplicity.

        *production* is a production reaction or its label. ENDF states the same
        quantity two ways -- MF10 as a cross section, MF9 as a multiplicity of
        the reaction's MF3 -- and the model keeps the difference (a ``reference``
        to the reaction's σ for MF9), so a perturbed MF3 reaches the production.
        This is the one place that undoes it: the reference is followed, and the
        product's multiplicity is applied.

        The grid is the union of both tables' nodes inside the common domain,
        and each factor is evaluated there by **its own** interpolation rule, so
        the values are exact at every node; between nodes the product of two
        interpolants is not itself one of ENDF's laws, the caveat
        :meth:`cross_section` gives for flattened regions.

        **A reference read at the evaluated form is refused where the resonances
        could reach it.** There the reaction's evaluated form is the MF3
        background, and MF2 adds the resonances only on reconstruction: Am-241
        capture to Am-242m reads 0 at thermal. The reconstruction gives the
        production a ``recon`` form that follows the reaction's, so ask for that.
        A table that starts above the resonance range, or a tape whose MF3
        already holds the resonances (LRP=2), is read as it is.

        Parameters
        ----------
        production
            A production reaction of ``productions`` or its label.
        form
            The label of the cross-section form (and of the multiplicity form,
            when the multiplicity has one of that label; else its default form).
            ``'eval'`` by default.

        Returns
        -------
        tuple of numpy.ndarray
            ``(E, sigma)``: energies in eV and the production cross sections in b.
            The energies are the union of the nodes of both tables inside their
            common domain; the cross section and the multiplicity are each
            evaluated by their own interpolation rule.

        Raises
        ------
        ValueError
            If the reaction does not have exactly one product other than the
            photon, or that product has no multiplicity; if the cross section
            states units other than eV and b; if it refers to a reaction form that
            lacks the resonance contribution (see above).
        KeyError
            If the production label or the form label is not found.
        TypeError
            If the cross-section form is not a tabulated function.
        """
        if isinstance(production, str):
            production = self.productions.byLabel(production)
        chosen = production.crossSection[form]
        if type(chosen).__name__ == "Reference":
            chosen = self.resolve_reference(chosen)
            self._refuseBackgroundOnly(production, form, chosen)
        _checkUnits(chosen, CROSS_SECTION_UNITS,
                    f"the {form!r} cross section of {production.label}")
        x, _ = _pointwise(chosen)
        products = [p for p in production.outputChannel.products if p.pid != "photon"]
        if len(products) != 1 or products[0].multiplicity is None:
            raise ValueError(f"{production.label} has {len(products)} products; a "
                             f"production makes exactly one, with a multiplicity")
        stated = products[0].multiplicity
        multiplicity = stated[form] if form in stated else stated.form
        grid = [x]
        if isinstance(multiplicity, (XYs1d, Regions1d)):
            grid.append(_pointwise(multiplicity)[0])
        lo = max(float(chosen.domainMin), float(multiplicity.domainMin))
        hi = min(float(chosen.domainMax), float(multiplicity.domainMax))
        energies = np.unique(np.concatenate(grid))
        energies = energies[(energies >= lo) & (energies <= hi)]
        values = (np.asarray(chosen.evaluate(energies), dtype=float)
                  * np.asarray(multiplicity.evaluate(energies), dtype=float))
        return energies, values

    def _refuseBackgroundOnly(self, production, form: str, resolved) -> None:
        """Raise when *resolved*, the reaction form a reference reached, may lack MF2's part."""
        header = getattr(self.provenance, "headerFields", None) or {}
        if header.get("lrp") == 2:
            return
        domain = self.resonances.domain if self.resonances is not None else None
        if domain is None or float(resolved.domainMin) >= float(domain[1]):
            return
        try:
            chain = self.styles.chain(form)
        except (KeyError, ValueError):
            chain = []
        if any(type(style).__name__ == "CrossSectionReconstructed" for style in chain):
            return
        raise ValueError(
            f"{production.label} refers to its reaction's {form!r} cross section, which "
            f"below {float(domain[1]):g} eV is the MF3 background without the resonance "
            f"contribution; reconstruct the suite (kika.processing.resonances) and ask "
            f"for form='recon'")

    def summary(self) -> str:
        """A few lines saying what is actually in here, including what is missing."""
        lines = [
            f"{self.projectile} + {self.target}   "
            f"[{self.evaluation or 'no evaluation id'}]  GNDS {self.format}",
            f"  styles       {self.styles.labels or 'none'}",
            f"  reactions    {len(self.reactions)}  MT{self.reactions.ENDF_MTs}",
        ]
        for name in ("sums", "fissionComponents", "productions",
                     "orphanProducts", "incompleteReactions"):
            container = getattr(self, name)
            if len(container):
                lines.append(f"  {name:<12} {len(container)}  MT{container.ENDF_MTs}")
        lines.append(f"  PoPs         {len(self.PoPs)} particles")
        if self.resonances is None:
            lines.append("  resonances   none")
        else:
            domain = self.resonances.domain
            span = "no region" if domain is None else f"{_eV(domain[0])} - {_eV(domain[1])}"
            lines.append(
                f"  resonances   {len(self.resonances.resolved)} resolved, "
                f"{'1' if self.resonances.unresolved is not None else 'no'} unresolved"
                f"   [{span}]"
            )
        lines.append(
            "  covariances  none"
            if self.covarianceSuite is None
            else f"  covariances  {len(self.covarianceSuite)} sections"
        )
        lines.append(
            f"  decode       {self.report.summary()}" if self.report is not None
            else "  decode       not recorded"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = [
            f"{self.projectile} + {self.target}",
            f"{self.evaluation!r}" if self.evaluation else "no evaluation id",
            f"{len(self.reactions)} reactions",
        ]
        if self.resonances is not None and self.resonances.domain is not None:
            parts.append(f"resonances to {_eV(self.resonances.domain[1])}")
        if self.covarianceSuite is not None:
            parts.append(f"{len(self.covarianceSuite)} covariance sections")
        if self.report is not None and not self.report.isClean:
            parts.append(self.report.summary())
        return f"<ReactionSuite {' | '.join(parts)}>"
