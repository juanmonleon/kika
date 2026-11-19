"""
Base class for all EXFOR entries.

This module provides ExforEntry, the base class for EXFOR data with
metadata stored as plain dictionaries for simplicity.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import re

from kika.exfor._constants import SCHEMA_VERSION
from kika._constants import ATOMIC_MASS


def _parse_target_from_reaction(reaction: str) -> tuple:
    """
    Parse target isotope and ZAID from EXFOR reaction notation.

    Examples:
        "26-FE-56(N,EL)26-FE-56" -> ("Fe56", 26056)
        "26-FE-0(N,EL)26-FE-0" -> ("Fe0", 26000)

    Returns:
        Tuple of (target_string, target_zaid)
    """
    match = re.match(r"(\d+)-([A-Z]+)-(\d+)", reaction)
    if not match:
        return ("Unknown", 0)

    z = int(match.group(1))
    symbol = match.group(2).capitalize()
    a = int(match.group(3))

    target = f"{symbol}{a}" if a > 0 else f"{symbol}0"
    zaid = z * 1000 + a

    return (target, zaid)


def _parse_projectile_from_reaction(reaction: str) -> str:
    """Parse projectile from EXFOR reaction notation."""
    match = re.search(r"\(([A-Z]+),", reaction)
    if match:
        return match.group(1).lower()
    return "n"


def _parse_process_from_reaction(reaction: str) -> str:
    """Parse process from EXFOR reaction notation."""
    match = re.search(r",([A-Z]+)\)", reaction)
    if match:
        return match.group(1)
    return "EL"


def _normalize_units(units: Dict[str, str]) -> Dict[str, str]:
    """Normalize unit strings to standard format."""
    energy_map = {
        "ev": "eV", "kev": "keV", "mev": "MeV",
        "EV": "eV", "KEV": "keV", "MEV": "MeV",
        "M": "MeV",
    }
    angle_map = {
        "adeg": "deg", "ADEG": "deg", "DEG": "deg",
        "cos": "cos", "COS": "cos",
    }
    xs_map = {
        "b/sr": "b/sr", "B/SR": "b/sr",
        "mb/sr": "mb/sr", "MB/SR": "mb/sr",
        "ub/sr": "ub/sr", "UB/SR": "ub/sr", "MUB/SR": "ub/sr",
    }

    return {
        "energy": energy_map.get(units.get("energy", "MeV"), units.get("energy", "MeV")),
        "angle": angle_map.get(units.get("angle", "deg"), units.get("angle", "deg")),
        "cross_section": xs_map.get(
            units.get("cross_section") or units.get("dsig", "b/sr"),
            units.get("cross_section") or units.get("dsig", "b/sr")
        ),
    }


@dataclass
class ExforEntry:
    """
    Metadata common to every EXFOR subentry, kept as plain dicts that map onto JSON.

    The subclasses add the data: :class:`~kika.exfor.ExforAngularDistribution`,
    :class:`~kika.exfor.ExforCrossSection` and :class:`~kika.exfor.ExforExperiment`.
    A user gets one from the readers and the database, not by building it.

    Attributes
    ----------
    entry : str
        EXFOR entry number, for example ``"27673"``.
    subentry : str
        EXFOR subentry number, for example ``"002"``.
    quantity : str
        EXFOR quantity code, for example ``"DA"`` for an angular distribution.
    citation : dict
        Bibliographic data: ``authors`` (a list of names) and ``year``, which
        :attr:`label` uses.
    reaction : dict
        Keys ``target`` (``"Fe56"``), ``target_zaid`` (``26056``), ``projectile``
        (``"n"``), ``process`` (``"EL"``) and ``notation`` (the EXFOR reaction
        string).
    facility, method : dict
        Facility and measurement metadata, as the source gives them.
    detector, sample : dict or None
        Detector and sample metadata, None when the source has none.
    """

    # Core identifiers
    entry: str
    subentry: str
    quantity: str  # 'DA', 'SIG', etc.

    # Metadata as plain dicts
    citation: Dict[str, Any] = field(default_factory=dict)
    reaction: Dict[str, Any] = field(default_factory=dict)
    facility: Dict[str, Any] = field(default_factory=dict)
    method: Dict[str, Any] = field(default_factory=dict)
    detector: Optional[Dict[str, Any]] = None
    sample: Optional[Dict[str, Any]] = None

    # =========================================================================
    # Properties for convenient access
    # =========================================================================

    @property
    def label(self) -> str:
        """Return formatted label (e.g., 'Kinney et al. (1976)')."""
        authors = self.citation.get("authors", [])
        year = self.citation.get("year", "")

        if not authors:
            return f"Unknown ({year})"

        first_author = authors[0]
        # Handle format like "W.E.Kinney" -> "Kinney"
        if "." in first_author:
            surname = first_author.split(".")[-1].strip()
        elif " " in first_author:
            surname = first_author.split()[-1]
        else:
            surname = first_author

        if len(authors) > 1:
            return f"{surname} et al. ({year})"
        return f"{surname} ({year})"

    @property
    def target(self) -> str:
        """Return target isotope (e.g., 'Fe56')."""
        return self.reaction.get("target", "Unknown")

    @property
    def zaid(self) -> int:
        """Return target ZAID (e.g., 26056)."""
        return self.reaction.get("target_zaid", 0)

    @property
    def target_mass(self) -> float:
        """Return target mass in amu from ATOMIC_MASS dictionary."""
        zaid = self.zaid
        if zaid in ATOMIC_MASS:
            return ATOMIC_MASS[zaid]
        # For natural abundance (ZAID ends in 000), try average mass
        z = zaid // 1000
        natural_zaid = z * 1000
        if natural_zaid in ATOMIC_MASS:
            return ATOMIC_MASS[natural_zaid]
        # Fallback: estimate from ZAID
        a = zaid % 1000
        return float(a) if a > 0 else float(z * 2)

    @property
    def process(self) -> str:
        """Return reaction process (e.g., 'EL')."""
        return self.reaction.get("process", "EL")

    @property
    def projectile(self) -> str:
        """Return projectile (e.g., 'n')."""
        return self.reaction.get("projectile", "n")

    @property
    def is_natural_target(self) -> bool:
        """
        Check if the experiment target is a natural element (not a specific isotope).

        Natural targets have ZAID ending in 000 (e.g., Fe-nat = 26000).
        Specific isotopes have the mass number (e.g., Fe-56 = 26056).

        Returns
        -------
        bool
            True if target is natural element, False if specific isotope.
        """
        zaid = self.zaid
        return zaid > 0 and (zaid % 1000) == 0

    @property
    def dataset_id(self) -> str:
        """
        Return the full dataset ID (entry + subentry).

        Examples
        --------
        >>> entry.entry = "23365"
        >>> entry.subentry = "005"
        >>> entry.dataset_id
        '23365005'
        """
        return f"{self.entry}{self.subentry}"

    @property
    def exfor_url(self) -> str:
        """
        Return the EXFOR URL for this entry on the IAEA website.

        Format: https://www-nds.iaea.org/exfor/ENTRY.SUBENTRY
        Example: Entry 23365, subentry 005 -> "https://www-nds.iaea.org/exfor/23365.005"

        Returns
        -------
        str
            URL to the EXFOR entry on the IAEA website
        """
        from kika.exfor._constants import EXFOR_URL_BASE
        subentry_formatted = self.subentry.zfill(3)
        return f"{EXFOR_URL_BASE}{self.entry}.{subentry_formatted}"

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "schema_version": SCHEMA_VERSION,
            "entry": self.entry,
            "subentry": self.subentry,
            "quantity": self.quantity,
            "citation": self.citation,
            "reaction": self.reaction,
            "facility": self.facility,
            "method": self.method,
        }
        if self.detector is not None:
            result["detector"] = self.detector
        if self.sample is not None:
            result["sample"] = self.sample
        return result

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """
        Write the object to a JSON file, from :meth:`to_dict`.

        A subclass writes its own fields too. An existing file is overwritten.

        Parameters
        ----------
        filepath : str
            Path of the file to write.
        indent : int
            Indentation of the JSON text, in spaces.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def _parse_base_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse base fields from JSON data.

        Handles both old and new JSON formats, converting reaction string
        to dict format if needed.
        """
        # Handle reaction - can be string (old format) or dict (new format)
        reaction_data = data.get("reaction")
        if isinstance(reaction_data, str):
            target, zaid = _parse_target_from_reaction(reaction_data)
            projectile = _parse_projectile_from_reaction(reaction_data)
            process = _parse_process_from_reaction(reaction_data)
            reaction = {
                "target": target,
                "target_zaid": zaid,
                "projectile": projectile,
                "process": process,
                "notation": reaction_data,
            }
        else:
            reaction = reaction_data or {}

        return {
            "entry": data.get("entry", "Unknown"),
            "subentry": data.get("subentry", "Unknown"),
            "quantity": data.get("quantity", "DA"),
            "citation": data.get("citation", {}),
            "reaction": reaction,
            "facility": data.get("facility", {}),
            "method": data.get("method", {}),
            "detector": data.get("detector"),
            "sample": data.get("sample"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExforEntry":
        """
        Build the metadata of an entry from a dict with the layout of the JSON files.

        Missing keys default to ``"Unknown"`` for ``entry`` and ``subentry``, to
        ``"DA"`` for ``quantity``, to empty dicts for the metadata and to None for
        ``detector`` and ``sample``. A ``reaction`` given as an EXFOR string is
        split into target, ZAID, projectile and process.

        Parameters
        ----------
        data : dict
            The values, as in a JSON file.

        Returns
        -------
        ExforEntry
            An instance of the class it is called on.
        """
        fields = cls._parse_base_fields(data)
        return cls(**fields)

    @classmethod
    def from_json(cls, filepath: str) -> "ExforEntry":
        """
        Read an entry from a JSON file, choosing the class from ``quantity``.

        A file whose ``quantity`` is ``"DA"`` (or missing) gives an
        :class:`~kika.exfor.ExforAngularDistribution`, whichever class this is
        called on. Any other quantity gives an instance of the class it is called on,
        with the metadata only: the data of the file are not read.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.

        Returns
        -------
        ExforEntry
            The entry.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If the file is not valid JSON.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Dispatch to appropriate subclass based on quantity
        quantity = data.get("quantity", "DA")

        if quantity == "DA":
            # Import here to avoid circular import
            from kika.exfor.angular_distribution import ExforAngularDistribution
            return ExforAngularDistribution.from_dict(data)

        # Default to base class for unknown quantities
        return cls.from_dict(data)
