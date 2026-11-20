"""NJOY Input Deck reader — parse existing decks into structured data.

Public API
----------
- ``read_deck(text)``      — parse a string
- ``read_deck_file(path)`` — parse a file
- ``ParsedDeck``           — result container
- ``ParsedModule``         — single-module container
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Comprehensive set of all NJOY module names (lowercase).
NJOY_MODULE_NAMES: set[str] = {
    "moder", "reconr", "broadr", "unresr", "heatr", "thermr",
    "groupr", "gaminr", "errorr", "covr", "dtfr", "ccccr",
    "matxsr", "resxsr", "acer", "powr", "wimsr", "plotr",
    "viewr", "mixr", "purr", "leapr", "gaspr",
}


@dataclass
class ParsedModule:
    """One module of a deck read by ``read_deck``: its original text and its values.

    Attributes
    ----------
    name
        Module name in lower case (``"moder"``).
    raw_lines
        The module's lines exactly as they were in the deck: the ``--`` comment
        lines above the module name, the name line and every card after it up to
        the next module (or the ``stop`` line).
    params
        The values of the cards, keyed by NJOY's field names (``nin``, ``err``),
        with a ``_line_map`` entry that gives, for each field, the indices of its
        card among the module's cards (comment lines and the name line not
        counted). None when no values could be extracted: the module has no
        parser (``mixr``) or its parser failed.
    """

    name: str              # Lowercase module name (e.g. "moder")
    raw_lines: list[str]   # Original lines (comments + name + cards)
    params: dict | None = None  # Extracted parameters; None for unknown modules


@dataclass
class ParsedDeck:
    """An NJOY deck split into modules, as returned by ``read_deck``.

    Only the cards of the modules with a parser are turned into values; all the
    text is kept, so ``render`` returns the deck as it was read, up to the
    ``stop`` line.

    Attributes
    ----------
    preamble
        The lines before the first module (and before the ``--`` comment lines
        that belong to it). With no module, the lines before the ``stop`` line,
        or the whole text when there is none.
    modules
        The modules in deck order.
    stop_line
        The last line whose first word is ``stop``, or None when there is none.
        The lines after it are dropped.
    """

    preamble: list[str] = field(default_factory=list)
    modules: list[ParsedModule] = field(default_factory=list)
    stop_line: str | None = None

    def render(self) -> str:
        """Reconstruct the exact original text from raw_lines."""
        all_lines = list(self.preamble)
        for mod in self.modules:
            all_lines.extend(mod.raw_lines)
        if self.stop_line is not None:
            all_lines.append(self.stop_line)
        return "\n".join(all_lines)

    def __getitem__(self, name: str) -> ParsedModule:
        """Get first module by name (case-insensitive)."""
        name_lower = name.lower()
        for mod in self.modules:
            if mod.name == name_lower:
                return mod
        raise KeyError(f"No module named {name!r}")

    def get_all(self, name: str) -> list[ParsedModule]:
        """Return every module called *name*, in deck order.

        Parameters
        ----------
        name
            Module name; the case is ignored.

        Returns
        -------
        list[ParsedModule]
            The matching modules, an empty list when there are none.
        """
        name_lower = name.lower()
        return [mod for mod in self.modules if mod.name == name_lower]


def read_deck(text: str) -> ParsedDeck:
    """Split an NJOY deck into modules and parse the cards of each one.

    A module starts at a line whose first word is the name of an NJOY module
    (``moder``, ``reconr``, ``mixr`` ...), in any case, and runs to the next one;
    the ``--`` comment lines directly above it belong to it. Every line is kept
    as text in ``ParsedModule.raw_lines``. The cards are parsed into
    ``ParsedModule.params`` for the modules kika has a parser for.

    Parameters
    ----------
    text
        The deck, one card per line.

    Returns
    -------
    ParsedDeck
        The preamble, the modules and the ``stop`` line. A module with no parser,
        or whose cards the parser rejects, keeps ``params=None`` and a warning is
        emitted.
    """
    from .modules import parse_module

    lines = text.split("\n")

    # Locate module-name lines and the stop line
    module_indices: list[tuple[int, str]] = []
    stop_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        first_word = stripped.split()[0] if stripped else ""
        if first_word in NJOY_MODULE_NAMES:
            module_indices.append((i, first_word))
        elif first_word == "stop":
            stop_idx = i

    if not module_indices:
        return ParsedDeck(
            preamble=lines[: stop_idx] if stop_idx is not None else lines[:],
            modules=[],
            stop_line=lines[stop_idx] if stop_idx is not None else None,
        )

    # Walk backwards from each module name to attach preceding -- comments
    block_starts: list[tuple[int, int, str]] = []  # (block_start, name_idx, name)
    for line_idx, name in module_indices:
        start = line_idx
        while start > 0 and lines[start - 1].strip().startswith("--"):
            start -= 1
        block_starts.append((start, line_idx, name))

    # Determine end index for each block
    blocks: list[tuple[int, int, int, str]] = []  # (start, end, name_idx, name)
    for i, (start, name_idx, name) in enumerate(block_starts):
        if i + 1 < len(block_starts):
            end = block_starts[i + 1][0]
        elif stop_idx is not None:
            end = stop_idx
        else:
            end = len(lines)
        blocks.append((start, end, name_idx, name))

    preamble = lines[: blocks[0][0]]
    stop_line = lines[stop_idx] if stop_idx is not None else None

    modules: list[ParsedModule] = []
    for start, end, name_idx, name in blocks:
        raw_lines = lines[start:end]

        # Extract data card lines (skip comments and the module-name line)
        card_lines: list[str] = []
        for ln in raw_lines:
            stripped = ln.strip()
            if stripped.startswith("--"):
                continue
            first_word = stripped.lower().split()[0] if stripped else ""
            if first_word == name:
                continue
            card_lines.append(ln)

        params = parse_module(name, card_lines)
        modules.append(ParsedModule(name=name, raw_lines=raw_lines, params=params))

    return ParsedDeck(preamble=preamble, modules=modules, stop_line=stop_line)


def read_deck_file(path: str | Path) -> ParsedDeck:
    """Read an NJOY deck from a UTF-8 file and parse it with ``read_deck``.

    Parameters
    ----------
    path
        Path of the deck file.

    Returns
    -------
    ParsedDeck
        As ``read_deck`` returns it.
    """
    return read_deck(Path(path).read_text(encoding="utf-8"))
