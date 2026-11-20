"""A run's own account of itself: events, not lines.

The perturbation pipelines this library grew up with logged the way most
pipelines do -- a ``logging.FileHandler`` and a few hundred f-strings -- and the
result was a file a person could grep and nothing could read. Nothing in it
said which sample a line was about, how long a stage took, or which block a
warning concerned; that had to be recovered from the shape of the text.

This module records **events** instead. An :class:`Event` says what happened
(*kind*), to what (*subject*, *sample*), when, for how long, and carries a
JSON payload with the numbers -- dimensions, rank, λ_min/λ_max, realised
covariance error, the files written. The events are written as
``run.log.jsonl`` for a program (the app's timeline, a dashboard, a test), and
the report a person reads (:mod:`kika.sampling.run_report`, ``report.html``) is
built from those same events, so it can be rebuilt from the file at any time.

The Python ``logging`` interface is kept: a :class:`RunLog` given a *logger*
forwards every event to it as one line, so a caller that already has a log
file keeps getting one.

Kinds are a closed set (:data:`EVENT_KINDS`). A consumer that filters on
``kind == "drawn"`` has to be able to rely on the spelling.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Event", "RunLog", "RunCancelled", "EVENT_KINDS", "LEVELS"]

#: What an event can be. In the order a run produces them, roughly.
EVENT_KINDS = (
    "started",      # the call, with its arguments
    "read",         # the source file decoded to the model
    "reconstructed",  # the resonance region rebuilt by NJOY RECONR
    "request",      # what the request became once normalised
    "assembled",    # the covariance blocks, with their dimensions
    "inspected",    # the pre-flight's findings, per block
    "conditioned",  # one repair step actually applied
    "drawn",        # one block drawn, with its diagnostics
    "applied",      # one realisation put on the model
    "checked",      # a check on a realisation (positivity, sum rule, normalisation)
    "emitted",      # one file written for one sample
    "written",      # a run-level file written (factors table, plan, metadata)
    "note",         # something true of the run that no output file states
    "warning",
    "error",
    "cancelled",    # stopped on request; what was written so far is kept
    "finished",
)

LEVELS = ("info", "warning", "error")


class RunCancelled(Exception):
    """Raised by whoever drives a run to stop it at the next event.

    The run records a ``cancelled`` event, not an ``error``, and still writes
    its log and report. Nothing in kika raises it by itself: a caller raises it,
    for instance from the ``logger`` given to ``RunLog``, whose methods the log
    calls on every event, and it propagates out of ``perturbFromModel``.
    """


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _jsonable(value):
    """The payload as JSON can hold it. Numpy scalars and arrays included."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class Event:
    """One thing that happened during a run."""

    kind: str
    message: str
    at: str
    level: str = "info"
    subject: Optional[str] = None
    sample: Optional[int] = None
    seconds: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "level": self.level, "at": self.at,
            "sample": self.sample, "subject": self.subject,
            "seconds": self.seconds, "message": self.message,
            "payload": _jsonable(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(kind=data["kind"], message=data.get("message", ""),
                   at=data.get("at", ""), level=data.get("level", "info"),
                   subject=data.get("subject"), sample=data.get("sample"),
                   seconds=data.get("seconds"),
                   payload=dict(data.get("payload") or {}))

    def render(self) -> str:
        """One line for a person. The clock, the kind, who, what, how long."""
        clock = self.at[11:23] if len(self.at) >= 23 else self.at
        mark = {"info": " ", "warning": "!", "error": "E"}[self.level]
        where = ""
        if self.sample is not None:
            where += f"#{self.sample:04d} "
        if self.subject:
            where += f"{self.subject} "
        tail = f"  ({self.seconds:.3f} s)" if self.seconds is not None else ""
        scalars = {
            key: value for key, value in self.payload.items()
            if isinstance(value, (int, float, str, bool)) and not isinstance(value, bool)
            or isinstance(value, bool)
        }
        detail = ""
        if scalars:
            bits = []
            for key, value in scalars.items():
                if isinstance(value, float):
                    bits.append(f"{key}={value:.4g}")
                else:
                    bits.append(f"{key}={value}")
            detail = "  | " + ", ".join(bits)
        return f"{clock} {mark} {self.kind:<11} {where}{self.message}{tail}{detail}"


class RunLog:
    """The events of one run, in order, and the two files they become.

    Parameters
    ----------
    logger
        An optional ``logging.Logger`` (or anything with ``info``/``warning``/
        ``error``). Every event is forwarded to it as its rendered line, so
        the old interface keeps working for a caller that has one. An
        exception raised by the logger propagates out of the call that
        recorded the event.
    label
        A name for the run (``perturbFromModel`` passes its ``labelPrefix``).
        It is kept on the ``label`` attribute. The default is an empty string.

    Attributes
    ----------
    events : list
        The recorded ``Event`` objects, in order. An ``Event`` has ``kind``,
        ``message``, ``at`` (ISO 8601 time), ``level``, ``subject``, ``sample``,
        ``seconds`` and ``payload``.
    stage : str or None
        The stage under way: set by ``timed``, and by the pipeline between
        timed blocks, so an exception raised outside one still says where.
    """

    def __init__(self, logger=None, *, label: str = "") -> None:
        self.label = label
        self.events: List[Event] = []
        self._logger = logger
        self._started = time.perf_counter()
        #: The stage under way: set by :meth:`timed` and by the pipeline between
        #: timed blocks, so an exception raised outside one still says where.
        self.stage: Optional[str] = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def event(self, kind: str, message: str, *, subject: Optional[str] = None,
              sample: Optional[int] = None, seconds: Optional[float] = None,
              level: str = "info", **payload) -> Event:
        """Record one event and forward it to the logger, if any.

        Parameters
        ----------
        kind
            One of ``EVENT_KINDS``.
        message
            The text of the event.
        subject
            What the event is about (a block, a file); stored as a string.
        sample
            The sample number the event concerns.
        seconds
            How long the thing took, in seconds.
        level
            ``"info"``, ``"warning"`` or ``"error"``. For the kinds
            ``"warning"`` and ``"error"`` an ``"info"`` level becomes that kind.
        **payload
            The numbers and names the event carries. They stay as given in
            memory; ``to_dicts`` and ``write`` convert them to JSON (NumPy
            values and arrays become Python ones, paths become strings, other
            objects their ``str``).

        Returns
        -------
        Event
            The recorded event.

        Raises
        ------
        ValueError
            If *kind* is not in ``EVENT_KINDS`` or *level* is not one of the
            three above.
        """
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}; known: {EVENT_KINDS}")
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; known: {LEVELS}")
        if kind in ("warning", "error") and level == "info":
            level = kind
        event = Event(kind=kind, message=message, at=_now(), level=level,
                      subject=None if subject is None else str(subject),
                      sample=sample, seconds=seconds, payload=dict(payload))
        self.events.append(event)
        self._forward(event)
        return event

    def note(self, message: str, **payload) -> Event:
        """Record a ``"note"`` event: something true of the run that no file states.

        Parameters
        ----------
        message
            The text of the note.
        **payload
            Carried by the event, as in ``event``. A ``subject`` or ``sample``
            keyword is passed on as such.

        Returns
        -------
        Event
            The recorded event.
        """
        return self.event("note", message, **payload)

    def warning(self, message: str, *, subject=None, sample=None, **payload) -> Event:
        """Record a ``"warning"`` event, at level ``"warning"``.

        Parameters
        ----------
        message, subject, sample, **payload
            As in ``event``.

        Returns
        -------
        Event
            The recorded event.
        """
        return self.event("warning", message, subject=subject, sample=sample,
                          level="warning", **payload)

    def error(self, message: str, *, subject=None, sample=None, **payload) -> Event:
        """Record an ``"error"`` event, at level ``"error"``. It does not raise.

        Parameters
        ----------
        message, subject, sample, **payload
            As in ``event``.

        Returns
        -------
        Event
            The recorded event.
        """
        return self.event("error", message, subject=subject, sample=sample,
                          level="error", **payload)

    @contextmanager
    def timed(self, kind: str, message: str, *, subject=None, sample=None,
              **payload) -> Iterator[Dict[str, Any]]:
        """Record *kind* when the block finishes, with how long it took.

        Yields a dict the caller may fill with payload as it learns things;
        whatever is in it at exit is recorded. An exception inside becomes an
        ``error`` event naming the stage and is re-raised, so a run that died
        says where. A ``RunCancelled`` is re-raised with no error event. The
        ``stage`` attribute is set to *kind* on entry.

        Parameters
        ----------
        kind, message, subject, sample
            As in ``event``; the event is recorded at the end of the block with
            ``seconds`` set to the time the block took.
        **payload
            Carried by the event, together with what the block put in the
            yielded dict.

        Yields
        ------
        dict
            Initially empty; its entries join the event's payload.

        Raises
        ------
        ValueError
            If *kind* is not in ``EVENT_KINDS`` (when the block finishes without
            an exception).
        """
        extra: Dict[str, Any] = {}
        self.stage = kind
        start = time.perf_counter()
        try:
            yield extra
        except RunCancelled:
            raise
        except Exception as failure:
            self.error(f"{message}: {type(failure).__name__}: {failure}",
                       subject=subject, sample=sample,
                       seconds=time.perf_counter() - start, stage=kind,
                       exception=type(failure).__name__)
            raise
        self.event(kind, message, subject=subject, sample=sample,
                   seconds=time.perf_counter() - start, **payload, **extra)

    def absorb(self, events) -> None:
        """Take in events recorded elsewhere, as they were.

        A worker process keeps its own :class:`RunLog` and sends the events
        back; they are appended here with their own timestamps and forwarded
        to the logger like anything recorded directly.

        Parameters
        ----------
        events
            An iterable of ``Event`` objects or of their ``to_dict`` form.
        """
        for item in events:
            event = item if isinstance(item, Event) else Event.from_dict(item)
            self.events.append(event)
            self._forward(event)

    def _forward(self, event: Event) -> None:
        if self._logger is None:
            return
        line = event.render()
        if event.level == "error":
            self._logger.error(line)
        elif event.level == "warning":
            self._logger.warning(line)
        else:
            self._logger.info(line)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def of(self, kind: str) -> List[Event]:
        """The events of one kind, in order; empty when there are none.

        Parameters
        ----------
        kind
            The ``kind`` to select (not validated).

        Returns
        -------
        list
            The matching ``Event`` objects.
        """
        return [event for event in self.events if event.kind == kind]

    def problems(self) -> List[Event]:
        """Every event that is not plain information."""
        return [event for event in self.events if event.level != "info"]

    @property
    def elapsed(self) -> float:
        """Seconds since the log was created (wall clock, not the sum of events)."""
        return time.perf_counter() - self._started

    def summary(self) -> str:
        """One line: the number of events, the count of each kind, and the problems.

        Returns
        -------
        str
            ``"N event(s): <count> <kind>, ..."``, with ``"; M warning(s)/error(s)"``
            appended when some event is not at level ``"info"``.
        """
        counts: Dict[str, int] = {}
        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        parts = [f"{count} {kind}" for kind, count in counts.items()]
        problems = self.problems()
        head = f"{len(self.events)} event(s): " + ", ".join(parts)
        if problems:
            head += f"; {len(problems)} warning(s)/error(s)"
        return head

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Every event as a JSON-ready dict, in order (what ``write`` puts in the log).

        Each dict has the keys ``kind``, ``level``, ``at``, ``sample``,
        ``subject``, ``seconds``, ``message`` and ``payload``.
        """
        return [event.to_dict() for event in self.events]

    # ------------------------------------------------------------------
    # The report: what a person reads
    # ------------------------------------------------------------------

    def verdict(self) -> str:
        """One line: did it work. The first line of the report, and the last."""
        errors = self.of("error")
        finished = self.of("finished")
        warnings = [e for e in self.problems() if e.level == "warning"]
        started = self.of("started")
        dry = bool(started and started[0].payload.get("dryRun"))
        what = "DRY RUN" if dry else "RUN"
        cancelled = self.of("cancelled")
        if cancelled:
            return f"{what} STOPPED: {cancelled[0].message}"
        if errors:
            first = errors[0]
            where = f" at stage '{first.payload['stage']}'" if "stage" in first.payload else ""
            return f"{what} FAILED{where}: {first.message}"
        if not finished:
            return f"{what} DID NOT FINISH: the log ends after '{self.events[-1].kind}'" \
                if self.events else f"{what} NEVER STARTED"
        n = finished[0].payload.get("nSamples")
        head = f"{what} OK" + (f": {n} sample(s)" if n is not None else "")
        if warnings:
            head += f", {len(warnings)} warning(s) -- read them below"
        return head

    def report(self):
        """The run's report, a ``PerturbationReport`` built from the events.

        Its ``to_html`` renders the page that ``write`` saves as ``report.html``.
        """
        from kika.sampling.run_report import PerturbationReport

        return PerturbationReport.from_log(self)

    def write(self, directory, *, stem: str = "run") -> Tuple[Path, Path]:
        """``<stem>.log.jsonl`` for a program and ``report.html`` for a person.

        The events are written first: a report that cannot be rendered must not
        cost the run its log, so a rendering failure leaves a page that says so.
        Existing files of those names are overwritten.

        Parameters
        ----------
        directory
            Where to write; created when missing.
        stem
            The log file is ``<stem>.log.jsonl``; the page is always
            ``report.html``.

        Returns
        -------
        tuple of pathlib.Path
            ``(log path, report path)``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        jsonl = directory / f"{stem}.log.jsonl"
        page = directory / "report.html"
        with jsonl.open("w", encoding="utf-8") as handle:
            for record in self.to_dicts():
                handle.write(json.dumps(record) + "\n")
        try:
            body = self.report().to_html()
        except Exception:  # pragma: no cover - a renderer bug
            import html
            import traceback

            body = ("<!doctype html><meta charset=\"utf-8\"><title>kika run report</title>"
                    f"<h1>{html.escape(self.verdict())}</h1><p>The report could not be "
                    f"rendered; the events are in {html.escape(jsonl.name)}.</p><pre>"
                    f"{html.escape(traceback.format_exc())}</pre>")
        page.write_text(body, encoding="utf-8")
        return jsonl, page

    @classmethod
    def read(cls, path) -> "RunLog":
        """A log back from its ``.jsonl``.

        Parameters
        ----------
        path
            A log file written by ``write``. Blank lines are skipped.

        Returns
        -------
        RunLog
            A new log with the file's events. Its ``label`` is empty and it has
            no logger.
        """
        log = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                log.events.append(Event.from_dict(json.loads(line)))
        return log

    def __repr__(self) -> str:
        return f"RunLog({self.summary()})"
