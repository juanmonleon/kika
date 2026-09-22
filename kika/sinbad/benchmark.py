"""Opening a benchmark: the entry, the calculations beside it, and the joins between.

:func:`read` is the whole entry point. Give it the path of a benchmark XML file
-- or of the directory holding one -- and it returns a
:class:`SinbadBenchmark`, with every calculations file in ``calculations/``
already opened and joined to it.

**Why the two files are one object here.** §1 of ``Data_structures.md`` splits
them on purpose: adding a calculation must not touch the benchmark. But nobody
reads a benchmark to look at an experiment in isolation -- they read it to
compare something with it, and the comparison lives in the calculations file
while the measurement it divides by lives in the benchmark. So the files stay
separate on disk and the labels resolve across them, in the direction the
format states: a calculations file may use the benchmark's labels, never the
other way round.

**The sha1 is not decoration.** A calculations file records the checksum of the
benchmark it was written against. :attr:`Calculations.matches_benchmark` says
whether the benchmark has changed since, and :meth:`SinbadBenchmark.check`
collects it for every calculations file at once. A mismatch is not an error --
the numbers are still readable, and the answer may well be that nothing
relevant moved -- but a C/E that silently divides by a table someone has since
edited is exactly what the checksum exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from kika.sinbad._constants import CALCULATIONS_DIRNAME
from kika.sinbad._xml import Document, LabelIndex
from kika.sinbad.data_object import DataObject, DataObjectCollection
from kika.sinbad.entry import (
    Absence,
    Calculation,
    Comparison,
    CoordinateFrame,
    Detector,
    Documentation,
    ExternalFile,
    Identification,
    Issue,
    Normalisation,
    Position,
    RadiationSource,
    Status,
    ValueConvention,
)
from kika.sinbad.exceptions import (
    AmbiguousLabelError,
    LabelNotFoundError,
    SinbadError,
    SinbadFormatError,
)
from kika.sinbad.uncertainty import correlation, covariance

__all__ = ["SinbadBenchmark", "Calculations", "read"]


class _Named:
    """A dict-like view over objects that have a ``label``."""

    def __init__(self, objects: Sequence[Any], what: str):
        self._objects = list(objects)
        self._what = what

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._objects)

    def __contains__(self, label: object) -> bool:
        return any(getattr(o, "label", None) == label for o in self._objects)

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            return self._objects[key]
        for obj in self._objects:
            if getattr(obj, "label", None) == key:
                return obj
        matches = [o for o in self._objects if key in getattr(o, "label", "")]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise AmbiguousLabelError(
                f"{key!r} matches {len(matches)} {self._what}s "
                f"({', '.join(getattr(m, 'label', '') for m in matches)}); "
                f"use the full label"
            )
        raise LabelNotFoundError(
            f"no {self._what} {key!r}; have {[getattr(o, 'label', '') for o in self._objects]}"
        )

    @property
    def labels(self) -> List[str]:
        return [getattr(o, "label", "") for o in self._objects]

    def to_dataframe(self):
        """The objects as a table, one row each, attributes as columns."""
        import pandas as pd  # noqa: PLC0415

        rows = []
        for obj in self._objects:
            row = {
                key: value
                for key, value in vars(obj).items()
                if not key.startswith("_") and not isinstance(value, (dict, tuple, list))
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"<{len(self._objects)} {self._what}: {', '.join(self.labels[:5])}{'...' if len(self._objects) > 5 else ''}>"


class _Blocks:
    """Reading of the blocks a benchmark and a calculations file have in common."""

    def _read_common(self, document: Document, index: LabelIndex) -> None:
        root = document.root
        self.documentation = Documentation._read(root.find("documentation"))
        self.files = _Named(
            [ExternalFile._read(f) for f in root.findall("externalFiles/externalFile")],
            "external file",
        )
        self.normalisations = _Named(
            [Normalisation._read(n) for n in root.findall("definitions/normalisation")],
            "normalisation",
        )
        self.conventions = _Named(
            [ValueConvention._read(c) for c in root.findall("definitions/valueConvention")],
            "value convention",
        )


class SinbadBenchmark(_Blocks):
    """
    One SINBAD benchmark entry, read from its XML file.

    Parameters
    ----------
    path : str or pathlib.Path
        The benchmark XML file, or a directory holding exactly one.
    calculations : bool, default True
        Also open every file in ``calculations/`` beside it.
    entry_root : str or pathlib.Path, optional
        The entry repository directory the ``externalFile`` paths are relative
        to. Only needed to resolve or verify those files.

    Attributes
    ----------
    id, short_code, title, abstract : str
    identification : Identification
    documentation : Documentation
    status : Status
    data : DataObjectCollection
    detectors, positions, frames, sources : dict-like views
    calculations : dict-like view of Calculations

    Examples
    --------
    >>> import kika.sinbad as sinbad                     # doctest: +SKIP
    >>> b = sinbad.read("Benchmarks/asp_fe88/asp_fe88.xml")   # doctest: +SKIP
    >>> print(b.summary())                               # doctest: +SKIP
    >>> b.data(quantity="reaction rate").to_dataframe()  # doctest: +SKIP
    >>> b["reactionRate-S32"].uncertainty_budget.to_dataframe()   # doctest: +SKIP
    """

    def __init__(
        self,
        path: Union[str, Path],
        calculations: bool = True,
        entry_root: Optional[Union[str, Path]] = None,
    ):
        self.document = Document(path)
        if self.document.kind != "benchmark":
            raise SinbadFormatError(
                f"{self.document.path} is a calculations file; open its benchmark "
                f"and reach it through .calculations"
            )
        self.entry_root = Path(entry_root).expanduser() if entry_root else None
        self.index = LabelIndex([self.document])

        root = self.document.root
        self.format_version: Optional[str] = root.get("formatVersion")
        self.id: str = root.get("id", "")
        self.short_code: str = root.get("shortCode", "")
        #: The path of the entry inside the SINBAD repository, as the file records it.
        self.entry_path: Optional[str] = root.get("path")
        #: The revision of that repository the numbers were transcribed at.
        self.revision: Optional[str] = root.get("revision")

        self.identification = Identification._read(root.find("identification"))
        self.status = Status._read(root.find("status"))
        self._read_common(self.document, self.index)

        self.frames = _Named(
            [CoordinateFrame._read(f)
             for f in root.findall("experiment/coordinateFrames/coordinateFrame")],
            "coordinate frame",
        )
        self.positions = _Named(
            [Position._read(p) for p in root.findall("experiment/positions/position")],
            "position",
        )
        self.detectors = _Named(
            [Detector._read(d) for d in root.findall("experiment/detectors/detector")],
            "detector",
        )
        self.sources = _Named(
            [RadiationSource._read(s)
             for s in root.findall("experiment/radiationSources/radiationSource")],
            "radiation source",
        )
        self.absences: Tuple[Absence, ...] = tuple(
            Absence(a.get("kind", ""), " ".join((a.text or "").split()))
            for a in root.findall("absences/absence")
        )
        self.data = DataObjectCollection([
            DataObject(d, self.index, self)
            for d in root.findall("experiment/dataObjects/dataObject")
        ])

        self._calculations: List["Calculations"] = []
        if calculations:
            self._open_calculations()

    # -- construction -----------------------------------------------------

    def _open_calculations(self) -> None:
        folder = self.document.path.parent / CALCULATIONS_DIRNAME
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.xml")):
            self._calculations.append(Calculations(path, benchmark=self))

    @property
    def calculations(self) -> _Named:
        """The calculations files beside the benchmark, by label."""
        return _Named(self._calculations, "calculations file")

    # -- identity ---------------------------------------------------------

    @property
    def title(self) -> str:
        return self.documentation.title

    @property
    def abstract(self) -> str:
        return self.documentation.abstract

    @property
    def year(self) -> Optional[int]:
        return self.documentation.year

    @property
    def issues(self) -> Tuple[Issue, ...]:
        """What the entry says is wrong or inconsistent in its own sources."""
        return self.status.issues

    # -- look-up ----------------------------------------------------------

    def resolve(self, label: str) -> Any:
        """
        Any object of the entry, by its label.

        Data objects first, then detectors, positions, frames, sources,
        normalisations, conventions and files; then the calculations files.

        Raises
        ------
        LabelNotFoundError
        """
        for view in (
            self.data, self.detectors, self.positions, self.frames, self.sources,
            self.normalisations, self.conventions, self.files,
        ):
            if label in view:
                return view[label]
        for calculations in self._calculations:
            if label == calculations.label:
                return calculations
            found = calculations._resolve_own(label)
            if found is not None:
                return found
        raise LabelNotFoundError(f"no object labelled {label!r} in {self.short_code or self.id}")

    def __getitem__(self, label: str) -> Any:
        return self.resolve(label)

    def __contains__(self, label: object) -> bool:
        try:
            self.resolve(str(label))
        except LabelNotFoundError:
            return False
        return True

    # -- the numbers ------------------------------------------------------

    @property
    def measurements(self) -> DataObjectCollection:
        """The measured data objects -- what the experiment produced."""
        return self.data(nature="measured")

    def to_dataframe(self, convention: Optional[str] = None, **filters):
        """
        The measured tables, stacked, one row per point.

        Parameters
        ----------
        convention : str, optional
            Bring every value to this ``valueConvention`` first. This is the
            argument that makes tables stored in different conventions
            comparable; without it, they are returned as stored.
        **filters
            Passed to :meth:`DataObjectCollection.__call__` -- ``quantity``,
            ``nature``, ``reaction``, ``detector``.

        Examples
        --------
        >>> b.to_dataframe("backgroundSubtracted").head()    # doctest: +SKIP
        """
        selection = self.data(**filters) if filters else self.measurements
        return selection.to_dataframe(convention=convention)

    def covariance(self, labels: Optional[Sequence[str]] = None, relative: bool = False):
        """
        The covariance of the measured points, from the uncertainty budgets.

        Parameters
        ----------
        labels : sequence of str, optional
            Which data objects to include. Default: every measured object that
            holds a table and a budget.
        relative : bool, default False

        Returns
        -------
        matrix : numpy.ndarray
        index : list of (label, position)

        Examples
        --------
        >>> matrix, index = b.covariance()               # doctest: +SKIP
        >>> matrix.shape                                 # doctest: +SKIP
        (55, 55)
        """
        return covariance(self._covariance_objects(labels), relative=relative)

    def correlation(self, labels: Optional[Sequence[str]] = None):
        """The correlation matrix of the same points. See :meth:`covariance`."""
        return correlation(self._covariance_objects(labels))

    def _covariance_objects(self, labels: Optional[Sequence[str]]) -> List[DataObject]:
        if labels is not None:
            return [self.data[label] for label in labels]
        selected = [
            obj for obj in self.measurements
            if obj.kind == "table" and obj.uncertainty_budget is not None
        ]
        if not selected:
            raise SinbadError("no measured object holds both a table and a budget")
        return selected

    def ce(self, wide: bool = False, **filters):
        """
        Every published comparison, from every calculations file, as one table.

        Parameters
        ----------
        wide : bool, default False
            Pivot to positions x (file, column).
        **filters
            ``calculations`` and/or ``reaction`` to narrow it.

        Returns
        -------
        pandas.DataFrame
            Long form: calculations, comparison, reaction, position,
            shieldThickness, series, value.
        """
        import pandas as pd  # noqa: PLC0415

        frames = [c.ce() for c in self._calculations]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        table = pd.concat(frames, ignore_index=True)
        for key, value in filters.items():
            table = table[table[key] == value]
        if wide:
            return table.pivot_table(
                index=["reaction", "position", "shieldThickness"],
                columns=["calculations", "series"],
                values="value",
            )
        return table

    # -- what the file says about itself ----------------------------------

    def check(self):
        """
        Collect what can be checked without opening the entry repository.

        Returns
        -------
        pandas.DataFrame
            One row per statement the files make about themselves: the sha1
            each calculations file records for this benchmark, and whether it
            still holds.
        """
        import pandas as pd  # noqa: PLC0415

        digest = self.document.checksum()
        rows = [{
            "what": "benchmark file",
            "name": self.document.path.name,
            "expected": digest,
            "found": digest,
            "ok": True,
        }]
        for calculations in self._calculations:
            rows.append({
                "what": "calculations -> benchmark sha1",
                "name": calculations.label,
                "expected": calculations.benchmark_checksum,
                "found": digest,
                "ok": calculations.matches_benchmark,
            })
        return pd.DataFrame(rows)

    def verify_files(self, entry_root: Optional[Union[str, Path]] = None):
        """
        Check every ``externalFile`` checksum against the entry repository.

        Parameters
        ----------
        entry_root : str or pathlib.Path, optional
            The entry directory. Defaults to the one given when opening.

        Returns
        -------
        pandas.DataFrame
            One row per file: its role, its path, and whether it is present and
            unchanged. Files inside an archive are reported as not checked.
        """
        import pandas as pd  # noqa: PLC0415

        root = Path(entry_root).expanduser() if entry_root else self.entry_root
        if root is None:
            raise SinbadError(
                "no entry_root: give one here or when opening the benchmark"
            )
        rows = []
        for handle in self.files:
            row = {
                "label": handle.label, "role": handle.role, "path": handle.path,
                "ok": None, "note": "",
            }
            try:
                row["ok"] = handle.verify(root)
            except SinbadError as exc:
                row["note"] = str(exc)
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """
        A readable description of the entry: what it is, and what it holds.

        Examples
        --------
        >>> print(b.summary())                           # doctest: +SKIP
        """
        identification = self.identification
        lines = [
            f"{self.short_code or self.id} -- {self.title}",
            f"  id              {self.id}",
        ]
        if identification.domain:
            lines.append(
                f"  keys            {identification.domain} {identification.purpose} "
                f"{identification.geometry} {identification.material} "
                f"{identification.source} {identification.sequence} "
                f"{identification.measurement}"
            )
        if self.revision:
            lines.append(f"  revision        {self.revision[:12]}")
        if self.status.availability:
            quality = f", {self.status.quality.rating}" if self.status.quality else ""
            lines.append(f"  status          {self.status.availability}{quality}")
        if self.documentation.authors:
            names = ", ".join(a.name for a in self.documentation.authors)
            lines.append(f"  authors         {names}")
        if self.year:
            lines.append(f"  year            {self.year}")
        lines.append("")
        lines.append(f"  detectors       {len(self.detectors)}: {', '.join(self.detectors.labels)}")
        lines.append(f"  positions       {len(self.positions)}")
        lines.append(f"  data objects    {len(self.data)}")
        for quantity in self.data.quantities:
            selection = self.data(quantity=quantity)
            lines.append(f"     {quantity:<28s} {len(selection)}")
        measured = [
            obj for obj in self.measurements
            if obj.kind == "table" and obj.uncertainty_budget is not None
        ]
        if measured:
            points = sum(int(np.sum(~np.isnan(obj.values))) for obj in measured)
            lines.append(f"  measured points {points} in {len(measured)} tables")
        if self._calculations:
            lines.append("")
            for calculations in self._calculations:
                state = "" if calculations.matches_benchmark else "  [benchmark has changed]"
                codes = ", ".join(
                    c.name for c in calculations.documentation.computer_codes
                )
                lines.append(f"  calculations    {calculations.label} ({codes}){state}")
        if self.issues:
            lines.append("")
            lines.append(f"  issues          {len(self.issues)} recorded in the entry")
        if self.absences:
            lines.append(f"  absences        {len(self.absences)} things the entry does not have")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<SinbadBenchmark {self.short_code or self.id!r}: {len(self.data)} data "
            f"objects, {len(self._calculations)} calculations file(s)>"
        )


class Calculations(_Blocks):
    """
    One set of calculations of a benchmark: one code, one version, one report.

    Opened through its benchmark, not directly -- it needs the benchmark to
    resolve the labels it borrows.

    Attributes
    ----------
    label : str
    runs : dict-like view of Calculation
    data : DataObjectCollection
    comparisons : dict-like view of Comparison
    matches_benchmark : bool

    Examples
    --------
    >>> calc = b.calculations["enea-tort-3.2"]           # doctest: +SKIP
    >>> calc.runs.labels                                 # doctest: +SKIP
    >>> calc.ce().head()                                 # doctest: +SKIP
    """

    def __init__(self, path: Union[str, Path], benchmark: SinbadBenchmark):
        self.document = Document(path)
        if self.document.kind != "calculations":
            raise SinbadFormatError(f"{self.document.path} is not a calculations file")
        self.benchmark = benchmark
        #: Labels resolve here first, then in the benchmark -- §1.
        self.index = LabelIndex([self.document, benchmark.document])

        root = self.document.root
        self.format_version: Optional[str] = root.get("formatVersion")
        self.label: str = root.get("label", self.document.path.stem)

        reference = root.find("benchmark")
        self.benchmark_id: Optional[str] = reference.get("id") if reference is not None else None
        self.benchmark_short_code: Optional[str] = (
            reference.get("shortCode") if reference is not None else None
        )
        self.benchmark_checksum: Optional[str] = (
            reference.get("checksum") if reference is not None else None
        )

        self._read_common(self.document, self.index)
        self.issues: Tuple[Issue, ...] = tuple(
            Issue._read(i) for i in root.findall("issues/issue")
        )
        self.runs = _Named(
            [Calculation._read(c) for c in root.findall("calculation")], "calculation"
        )
        self.data = DataObjectCollection([
            DataObject(d, self.index, self) for d in root.findall("dataObjects/dataObject")
        ])
        self.comparisons = _Named(
            [Comparison(c, self) for c in root.findall("comparisons/comparison")],
            "comparison",
        )

    # -- the tie to the benchmark -----------------------------------------

    @property
    def matches_benchmark(self) -> bool:
        """
        Whether the benchmark file still has the sha1 this file was written against.

        ``True`` when they agree or when the file records no checksum.
        """
        if not self.benchmark_checksum:
            return True
        return self.benchmark.document.checksum() == self.benchmark_checksum

    def _resolve_own(self, label: str) -> Optional[Any]:
        for view in (self.data, self.runs, self.comparisons, self.normalisations,
                     self.conventions, self.files):
            if label in view:
                return view[label]
        return None

    def resolve(self, label: str) -> Any:
        """Any object of this file, or of its benchmark."""
        found = self._resolve_own(label)
        if found is not None:
            return found
        return self.benchmark.resolve(label)

    def __getitem__(self, label: str) -> Any:
        return self.resolve(label)

    def __contains__(self, label: object) -> bool:
        return self._resolve_own(str(label)) is not None

    # -- views ------------------------------------------------------------

    def ce(self):
        """
        The published comparisons of this file, long form.

        Returns
        -------
        pandas.DataFrame
            calculations, comparison, operator, reaction, position,
            shieldThickness, series, value. ``series`` is the comparison's
            value column -- one per library.
        """
        import pandas as pd  # noqa: PLC0415

        rows = []
        for comparison in self.comparisons:
            table = comparison.table
            if table is None:
                continue
            denominator = None
            try:
                denominator = self.resolve(comparison.denominator_label)
            except LabelNotFoundError:
                pass
            reaction = getattr(denominator, "reaction", None)
            positions = table.positions
            thickness = table["shieldThickness"] if "shieldThickness" in table else None
            for column in table.value_columns:
                values = table[column.name]
                for i in range(table.nrows):
                    rows.append({
                        "calculations": self.label,
                        "comparison": comparison.label,
                        "operator": comparison.operator,
                        "reaction": reaction,
                        "position": str(positions[i]) if positions is not None else str(i),
                        "shieldThickness": float(thickness[i]) if thickness is not None else np.nan,
                        "series": column.name,
                        "value": float(values[i]),
                    })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """What was run, with what, and what it produced."""
        lines = [f"{self.label} -- {self.documentation.title}"]
        if self.documentation.authors:
            lines.append(
                f"  authors         {', '.join(a.name for a in self.documentation.authors)}"
            )
        lines.append(f"  benchmark       {self.benchmark_short_code or self.benchmark_id}")
        if not self.matches_benchmark:
            lines.append("                  [the benchmark file has changed since]")
        for run in self.runs:
            lines.append(
                f"  run             {run.label}: {run.code} "
                f"{run.library or run.evaluated_library or ''}"
            )
        lines.append(f"  data objects    {len(self.data)}")
        lines.append(f"  comparisons     {len(self.comparisons)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<Calculations {self.label!r}: {len(self.runs)} run(s), "
            f"{len(self.comparisons)} comparison(s)>"
        )


def read(
    path: Union[str, Path],
    calculations: bool = True,
    entry_root: Optional[Union[str, Path]] = None,
) -> SinbadBenchmark:
    """
    Open a SINBAD benchmark.

    Parameters
    ----------
    path : str or pathlib.Path
        The benchmark XML file, or a directory holding exactly one. A
        directory whose name matches the file (``asp_fe88/asp_fe88.xml``) is
        the layout ``Benchmarks/`` uses, and is found without ambiguity.
    calculations : bool, default True
        Open the files in ``calculations/`` beside it as well.
    entry_root : str or pathlib.Path, optional
        Directory of the entry repository, for resolving ``externalFile``
        paths and verifying their checksums.

    Returns
    -------
    SinbadBenchmark

    Raises
    ------
    SinbadFormatError
        If the path holds no benchmark file, or more than one.

    Examples
    --------
    >>> import kika.sinbad as sinbad                     # doctest: +SKIP
    >>> b = sinbad.read("Benchmarks/asp_fe88")           # doctest: +SKIP
    >>> b.short_code                                     # doctest: +SKIP
    'asp_fe88'
    """
    path = Path(path).expanduser()
    if path.is_dir():
        candidates = [p for p in sorted(path.glob("*.xml"))]
        named = [p for p in candidates if p.stem == path.name]
        if named:
            candidates = named
        if not candidates:
            raise SinbadFormatError(f"no XML file in {path}")
        if len(candidates) > 1:
            raise SinbadFormatError(
                f"{path} holds {len(candidates)} XML files "
                f"({', '.join(p.name for p in candidates)}); name the one you mean"
            )
        path = candidates[0]
    return SinbadBenchmark(path, calculations=calculations, entry_root=entry_root)
