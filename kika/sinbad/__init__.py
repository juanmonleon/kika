"""
SINBAD shielding benchmarks, in the XML format of the SINBAD v2 data-format group.

Give :func:`read` the path of a benchmark file and you get an object that
answers what the entry says::

    >>> import kika.sinbad as sinbad
    >>> b = sinbad.read("Benchmarks/asp_fe88/asp_fe88.xml")
    >>> print(b.summary())
    >>> b.detectors.labels
    ['det-Au197', 'det-Rh103', 'det-In115', 'det-S32-pressed', ...]
    >>> b.data(quantity="reaction rate").to_dataframe()
    >>> b["reactionRate-S32"].uncertainty_budget.to_dataframe()
    >>> matrix, index = b.covariance()
    >>> b.calculations["enea-tort-3.2"].ce()

Everything in an entry is reached the same way, because the format stores it
the same way: **every set of numbers is a ``dataObject``**, with a label, a
quantity, a nature, its corrections and its uncertainty budget. The measured
reaction rates of one foil, the source distribution, the assembly geometry, a
calculated table and a group structure are all data objects, and
``b["label"]`` finds any of them -- including the ones in the calculations
files beside the benchmark, which are opened with it.

What the format is
------------------
A benchmark entry is one XML file, ``sinbad``, plus one file per set of
calculations, ``sinbadCalculations``, in ``calculations/`` beside it. The nodes
are GNDS-2.1's where GNDS has them (``table``, ``grid``, ``gridded1d``,
``documentation``, ``externalFile``), SFCOMPO's for materials and geometry, and
SINBAD's own for what neither had. ``Structures/Data_structures.md`` in the
``sinbadv2-data-format`` repository is the specification, and its section
numbers are cited throughout this subpackage.

This reader is written against draft **v0.3** (2026-09-19). The format is work
in progress: a container it does not model yet opens as
:class:`~kika.sinbad.content.UnknownContent`, which keeps the raw element, so a
later draft still reads.

Three things worth knowing before using it
------------------------------------------
**Conventions.** Two tables of the same quantity may be stored differently --
in the ASPIS pilot, sulphur ``asMeasured`` with a 2 % background correction
declared and not applied, gold ``backgroundSubtracted`` with it applied.
:meth:`~kika.sinbad.DataObject.corrected` and the ``convention`` argument of
:meth:`~kika.sinbad.SinbadBenchmark.to_dataframe` are how they are made
comparable; ignoring them compares two different things.

**Covariance is computed, not stored.** The format keeps uncertainty
*components* with a correlation scope, and a matrix only when the entry
published one. :func:`covariance` builds the matrix from the components, which
is the one calculation the specification asks a reader to do.

**The benchmark and its calculations are tied by sha1.** Each calculations file
records the checksum of the benchmark it was written against;
:attr:`~kika.sinbad.Calculations.matches_benchmark` and
:meth:`~kika.sinbad.SinbadBenchmark.check` say whether it still holds.

Relation to the rest of kika
----------------------------
This is a different repository of benchmarks from :mod:`kika.benchmarks`, which
holds ICSBEP/DICE criticality entries; nothing is shared between them. What is
shared is GNDS: :meth:`~kika.sinbad.content.Gridded.to_gnds` and
:func:`~kika.sinbad.gnds.check_units` translate into
:mod:`kika.nuclear_data.model`, and
:meth:`~kika.sinbad.content.Materials.to_kika_materials` into
:mod:`kika.materials`. Those imports happen at call time, so reading an entry
costs nothing but the standard library, numpy, and pandas when a frame is
asked for.
"""

from kika.sinbad._constants import (
    ABSENCE_KINDS,
    CORRELATION_SCOPES,
    NATURES,
    QUANTITIES,
    vocabularies,
)
from kika.sinbad.benchmark import Calculations, SinbadBenchmark, read
from kika.sinbad.content import (
    Axes,
    Axis,
    Column,
    Content,
    Double,
    FactorChain,
    Geometry,
    Grid,
    GridContent,
    Gridded,
    Material,
    Materials,
    Polynomial1d,
    Table,
    UnknownContent,
    XYs1d,
)
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
    BenchmarkMismatchError,
    ContentTypeError,
    LabelNotFoundError,
    SinbadError,
    SinbadFormatError,
)
from kika.sinbad.plotting import plot_ce, plot_correlation, plot_profile
from kika.sinbad.uncertainty import (
    Correction,
    UncertaintyBudget,
    UncertaintyComponent,
    correlation,
    covariance,
)

#: Open a benchmark. Alias of :func:`kika.sinbad.read`, for symmetry with
#: ``kika.read_endf`` and friends; ``read`` is the name to prefer.
open_benchmark = read

__all__ = [
    # the entry points
    "read",
    "open_benchmark",
    "SinbadBenchmark",
    "Calculations",
    # data objects and their content
    "DataObject",
    "DataObjectCollection",
    "Content",
    "Table",
    "Column",
    "Grid",
    "GridContent",
    "Gridded",
    "Axes",
    "Axis",
    "XYs1d",
    "Polynomial1d",
    "Materials",
    "Material",
    "Geometry",
    "FactorChain",
    "Double",
    "UnknownContent",
    # entry blocks
    "Identification",
    "Documentation",
    "Status",
    "Issue",
    "Absence",
    "ExternalFile",
    "Normalisation",
    "ValueConvention",
    "CoordinateFrame",
    "Position",
    "Detector",
    "RadiationSource",
    "Calculation",
    "Comparison",
    # uncertainty
    "Correction",
    "UncertaintyBudget",
    "UncertaintyComponent",
    "covariance",
    "correlation",
    # plots
    "plot_profile",
    "plot_ce",
    "plot_correlation",
    # vocabularies
    "vocabularies",
    "QUANTITIES",
    "NATURES",
    "CORRELATION_SCOPES",
    "ABSENCE_KINDS",
    # errors
    "SinbadError",
    "SinbadFormatError",
    "LabelNotFoundError",
    "AmbiguousLabelError",
    "ContentTypeError",
    "BenchmarkMismatchError",
]
