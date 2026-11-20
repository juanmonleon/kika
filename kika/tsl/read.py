"""One TSL material, whatever format it came in: ENDF-6, GNDS, ACE, OpenMC HDF5, G4NDL.

:func:`read_tsl` returns a :class:`TSLMaterial`: the model ``reactionSuite``
(what :func:`kika.read` gives) and, for the processed formats, the
:class:`~kika.tsl.processed.ProcessedTSL` it was built from, which keeps the
outgoing-neutron tables the model has no node for.

The questions a viewer asks are answered the same way for every format:

- :meth:`TSLMaterial.elastic` -- σ_coh and σ_inc at a temperature, from the
  model's laws (evaluations, and the exact parts of processed files) or from
  the processed tables;
- :meth:`TSLMaterial.inelastic` -- σ_inel(E) at a temperature, which only a
  processed file holds (an evaluation has S(α,β); see :mod:`kika.tsl.thermr`);
- :meth:`TSLMaterial.outgoing` and :meth:`TSLMaterial.elastic_cosines` -- the
  outgoing-neutron tables, processed files only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .processed import BraggTable, OutgoingTable, ProcessedTSL, processed_suite

__all__ = ["TSL_FORMATS", "PROCESSED_FORMATS", "TSLMaterial", "detect_tsl_format", "read_tsl"]

TSL_FORMATS = ("endf", "gnds", "ace", "openmc", "g4ndl")
PROCESSED_FORMATS = ("ace", "openmc", "g4ndl")


def detect_tsl_format(path: Union[str, Path]) -> Optional[str]:
    """The TSL format of *path*, or None when it is not a TSL file kika reads.

    Decided on content: an NSUB=12 ENDF tape, a GNDS ``reactionSuite`` whose
    interaction is the TNSL one, an ACE thermal table, an OpenMC
    ``data_thermal`` file, or a G4NDL ``ThermalScattering`` folder.

    Parameters
    ----------
    path
        A file, or a folder for G4NDL. A file that cannot be opened gives None.

    Returns
    -------
    str or None
        One of ``TSL_FORMATS`` (``"endf"``, ``"gnds"``, ``"ace"``, ``"openmc"``,
        ``"g4ndl"``), or None. An ENDF-6 tape is ``"endf"`` only when its NSUB is
        12; a GNDS file only when its text names the thermal neutron scattering
        interaction.
    """
    from kika._read import _headLines, _looksLikeEndf
    from .ace import looks_like_ace_thermal
    from .openmc import HDF5_SIGNATURE

    path = Path(path)
    if path.is_dir():
        from .g4ndl import g4ndl_thermal_root
        return "g4ndl" if g4ndl_thermal_root(path) else None
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096)
    except OSError:
        return None
    if head.startswith(HDF5_SIGNATURE):
        from .openmc import is_openmc_thermal
        return "openmc" if is_openmc_thermal(path) else None
    text = head.decode("utf-8", errors="replace")
    if text.lstrip().startswith("<"):
        return "gnds" if 'interaction="thermalNeutronScatteringLaw"' in text else None
    lines = _headLines(path, 6)
    if looks_like_ace_thermal(lines):
        return "ace"
    if _looksLikeEndf(lines):
        from .library import _nsub_of
        return "endf" if _nsub_of(path) == 12 else None
    return None


@dataclass
class TSLMaterial:
    """A TSL material in the model, with its processed tables when it has them.

    Returned by :func:`read_tsl`. The methods answer the same questions for every
    format, and give None for what the format does not hold.

    Attributes
    ----------
    format
        One of ``TSL_FORMATS``.
    suite
        The model ``reactionSuite`` with the thermal scattering interaction.
    paths
        The files read, as strings (for G4NDL, the files of the material).
    processed
        The processed tables, for the formats in ``PROCESSED_FORMATS``; None
        for ENDF-6 and GNDS.
    member
        The *member* the caller passed to :func:`read_tsl` for a processed format
        (ACE, OpenMC, G4NDL), stored as given; None when none was passed, and
        for ENDF-6 and GNDS.
    report
        The report of the conversion into the model, when the reader gives one.
    endf
        The decoded ENDF-6 tape, for ``format="endf"`` only; None otherwise.
    """

    format: str
    suite: object
    paths: List[str]
    processed: Optional[ProcessedTSL] = None
    member: Optional[str] = None
    report: object = None
    #: The endf object an ENDF tape was decoded from, for the callers that read
    #: ENDF-only fields (the B array, LLN) off it.
    endf: object = field(default=None, repr=False)

    @property
    def is_processed(self) -> bool:
        """True when ``processed`` holds tables (ACE, OpenMC and G4NDL materials)."""
        return self.processed is not None

    # -- the forms ----------------------------------------------------------

    def _evaluated(self, kind):
        for reaction in self.suite.reactions:
            dd = reaction.doubleDifferentialCrossSection
            if dd is None:
                continue
            for label in dd:
                form = dd[label]
                if isinstance(form, kind) and label == "eval":
                    return form
        return None

    def temperatures(self) -> Dict[str, List[float]]:
        """The temperatures (K) of each part: ``coherent``, ``incoherent``, ``inelastic``."""
        from kika.nuclear_data.model import CoherentElastic, IncoherentElastic, IncoherentInelastic, Gridded3d

        if self.processed is not None:
            blocks = self.processed.temperatures
            return {
                "coherent": [b.temperature for b in blocks if b.coherent is not None],
                "incoherent": [b.temperature for b in blocks if b.incoherent is not None],
                "inelastic": [b.temperature for b in blocks if b.inelastic_xs is not None],
            }
        out = {"coherent": [], "incoherent": [], "inelastic": []}
        coherent = self._evaluated(CoherentElastic)
        if coherent is not None:
            out["coherent"] = [float(t) for t in coherent.temperatures]
        incoherent = self._evaluated(IncoherentElastic)
        if incoherent is not None:
            out["incoherent"] = [float(t) for t in incoherent.temperatures]
        inelastic = self._evaluated(IncoherentInelastic)
        if inelastic is not None:
            kernel = inelastic.principal.selfScatteringKernel.kernel
            if isinstance(kernel, Gridded3d):
                out["inelastic"] = [float(t) for t in np.asarray(kernel.grids[0].values, float)]
        return out

    def bragg(self, temperature: float) -> Optional[BraggTable]:
        """The Bragg edges and cumulative S at *temperature*, or None.

        Parameters
        ----------
        temperature
            In K. A processed file matches it to a tabulated one within a relative
            1e-6; an evaluation needs it to be exactly a tabulated value. Nothing is
            interpolated in temperature.

        Returns
        -------
        BraggTable or None
            Edge energies in eV and cumulative S in eV·b. None when the material has
            no coherent elastic part or does not hold that temperature.
        """
        from kika.nuclear_data.model import CoherentElastic

        if self.processed is not None:
            try:
                return self.processed.at(temperature).coherent
            except KeyError:
                return None
        form = self._evaluated(CoherentElastic)
        if form is None:
            return None
        try:
            return BraggTable(edges=form.braggEdges(), s=form.cumulativeS(temperature))
        except KeyError:
            return None

    def elastic(self, temperature: float, energies: Sequence[float]) -> Dict[str, Optional[np.ndarray]]:
        """σ_coh and σ_inc (b) at *energies* (eV) and a temperature the file holds.

        A part the file does not hold at that temperature is None.

        Parameters
        ----------
        temperature
            In K. Coherent: as in :meth:`bragg`. Incoherent of a processed file: a
            tabulated temperature within a relative 1e-6. Incoherent of an
            evaluation: interpolated in W'(T) between its tabulated temperatures,
            and None outside the table.
        energies
            Incident energies in eV.

        Returns
        -------
        dict
            ``{"coherent": σ_coh, "incoherent": σ_inc}``, each an array of barns of
            the shape of *energies*, or None. The coherent value is S(E_i)/E (zero
            below the first edge). A tabulated incoherent σ of a processed file is
            interpolated linearly, is NaN below the first tabulated energy and
            zero above the last; an OpenMC file gives the closed form instead.
        """
        from kika.nuclear_data.model import IncoherentElastic

        e = np.asarray(energies, dtype=float)
        bragg = self.bragg(temperature)
        out = {"coherent": bragg.cross_section(e) if bragg is not None else None,
               "incoherent": None}
        if self.processed is not None:
            try:
                table = self.processed.at(temperature).incoherent
            except KeyError:
                table = None
            if table is not None:
                out["incoherent"] = table.cross_section(e)
            return out
        form = self._evaluated(IncoherentElastic)
        if form is not None:
            try:
                out["incoherent"] = form.crossSection(e, temperature)
            except KeyError:
                pass
        return out

    def inelastic(self, temperature: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """σ_inel(E) at *temperature* as (E eV, σ b), from a processed file; None otherwise.

        Parameters
        ----------
        temperature
            In K; a tabulated temperature within a relative 1e-6.

        Returns
        -------
        tuple of numpy.ndarray or None
            Incident energies in eV and σ_inel in barns, on the file's own grid. None
            for an ENDF-6 or GNDS evaluation (it holds S(α,β), not σ_inel), for a
            temperature the file does not hold, and for a block with no inelastic σ.
        """
        if self.processed is None:
            return None
        try:
            block = self.processed.at(temperature)
        except KeyError:
            return None
        if block.inelastic_xs is None:
            return None
        return block.inelastic_energies, block.inelastic_xs

    def outgoing(self, temperature: float) -> Optional[OutgoingTable]:
        """The inelastic outgoing-neutron table at *temperature* (processed files only).

        Parameters
        ----------
        temperature
            In K; a tabulated temperature within a relative 1e-6.

        Returns
        -------
        OutgoingTable or None
            Read from the file on first request. None for an evaluation, for a
            temperature the file does not hold, and when the block has no table.
        """
        if self.processed is None:
            return None
        try:
            return self.processed.at(temperature).outgoing()
        except KeyError:
            return None

    def elastic_cosines(self, temperature: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Incoherent elastic equiprobable cosines (incident E, cosines) at *temperature*.

        Parameters
        ----------
        temperature
            In K; a tabulated temperature within a relative 1e-6.

        Returns
        -------
        tuple of numpy.ndarray or None
            Incident energies in eV, and an array with one row of equiprobable
            cosines per energy. None for an evaluation, for a temperature the file
            does not hold, and when the incoherent elastic has no cosines (OpenMC
            keeps the bound σ and W'(T) instead).
        """
        if self.processed is None:
            return None
        try:
            table = self.processed.at(temperature).incoherent
        except KeyError:
            return None
        if table is None or table.cosines is None or table.cosine_energies is None:
            return None
        return table.cosine_energies, table.cosines


def read_tsl(source: Union[str, Path, Sequence[Union[str, Path]]], *, format: Optional[str] = None,
             member: Optional[str] = None) -> TSLMaterial:
    """Read one TSL material.

    *source* is a file, or for ACE several files of one material (one per
    temperature), or for G4NDL the ``ThermalScattering`` folder (or the G4NDL
    root) with *member* naming the material (``h_water``).

    Parameters
    ----------
    source
        What to read, as above. Only the first file decides the format; the
        formats other than ACE read that first file alone.
    format
        One of ``TSL_FORMATS``, in any case. None detects it from the first
        file with :func:`detect_tsl_format`.
    member
        The G4NDL material name. Required for ``"g4ndl"``. ACE and OpenMC only
        store it on the result; ENDF-6 and GNDS ignore it.

    Returns
    -------
    TSLMaterial
        Its ``processed`` is set for ACE, OpenMC and G4NDL, and None for ENDF-6
        and GNDS. GNDS is read without covariances.

    Raises
    ------
    ValueError
        If *source* is an empty list, if the format is not one of
        ``TSL_FORMATS`` or cannot be detected, or if the format is G4NDL and
        *member* is empty.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    if not sources:
        raise ValueError("no file given")
    first = sources[0]
    format = (format or detect_tsl_format(first) or "").lower()
    if format not in TSL_FORMATS:
        raise ValueError(f"{first} is not a TSL file kika reads (ENDF-6 NSUB=12, GNDS TNSL, "
                         "ACE thermal, OpenMC thermal HDF5 or G4NDL ThermalScattering)")
    paths = [str(p) for p in sources]
    if format == "ace":
        from .ace import read_ace_thermal
        processed = read_ace_thermal(sources)
    elif format == "openmc":
        from .openmc import read_openmc_thermal
        processed = read_openmc_thermal(first)
    elif format == "g4ndl":
        from .g4ndl import read_g4ndl_thermal
        if not member:
            raise ValueError("a G4NDL thermal material is named by member= (e.g. 'h_water')")
        processed = read_g4ndl_thermal(first, member)
        paths = processed.paths
    else:
        processed = None
    if processed is not None:
        suite, report = processed_suite(processed)
        return TSLMaterial(format=format, suite=suite, paths=paths, processed=processed,
                           member=member, report=report)
    if format == "endf":
        from kika.endf.model_adapter.decode import decodeReactionSuite
        from kika.endf.read_endf import read_endf

        endf = read_endf(str(first))
        suite, report = decodeReactionSuite(endf)
        return TSLMaterial(format=format, suite=suite, paths=paths, report=report, endf=endf)
    import kika

    suite = kika.read(str(first), format="gnds", covariances=False)
    return TSLMaterial(format=format, suite=suite, paths=paths, report=getattr(suite, "report", None))

