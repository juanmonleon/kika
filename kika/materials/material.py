from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

from .._constants import ATOMIC_MASS, ATOMIC_NUMBER_TO_SYMBOL, NATURAL_ABUNDANCE
from .._utils import zaid_to_symbol, symbol_to_zaid

# Density unit normalisation helpers.
_DENSITY_UNIT_ALIASES: Dict[str, str] = {
    "g/cc": "g/cc",
    "g/cm3": "g/cc",
    "g/cm^3": "g/cc",
    "g/cm³": "g/cc",
    "kg/m3": "kg/m3",
    "kg/m^3": "kg/m3",
    "kg/m³": "kg/m3",
    "atoms/b-cm": "atoms/b-cm",
    "atoms/barn-cm": "atoms/b-cm",
    "a/b-cm": "atoms/b-cm",
    "atoms/b·cm": "atoms/b-cm",
}

_DENSITY_CONVERSIONS: Dict[tuple[str, str], float] = {
    ("g/cc", "g/cc"): 1.0,
    ("g/cc", "kg/m3"): 1000.0,
    ("kg/m3", "kg/m3"): 1.0,
    ("kg/m3", "g/cc"): 0.001,
}


# MCNP data class letter -> m-card library keyword. 'y' (dosimetry) is kika's own key, not an
# m-card keyword; 't' (S(a,b)) is absent because those tables go on the mt card.
_LIB_KEY_BY_CLASS: Dict[str, str] = {
    "c": "nlib", "d": "nlib", "p": "plib", "u": "pnlib", "e": "elib", "h": "hlib",
    "o": "dlib", "r": "tlib", "s": "slib", "a": "alib", "y": "ylib",
}


def _normalize_density_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    key = unit.strip().lower().replace(" ", "")
    return _DENSITY_UNIT_ALIASES.get(key, key)


def _read_lines(path) -> List[str]:
    """Lines of a text file with their own endings, any byte sequence kept as is."""
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.readlines()


def _write_lines(path, lines: List[str]) -> None:
    """Write lines read by :func:`_read_lines` back byte for byte."""
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.writelines(lines)


def _line_ending(lines: List[str]) -> str:
    """The file's line ending, from its first ended line; ``"\\n"`` when there is none."""
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
    return "\n"


def _validate_fraction_type(fraction_type: str) -> None:
    """Validate that fraction_type is exactly 'ao' or 'wo'."""
    if fraction_type not in ('ao', 'wo'):
        raise ValueError(f"Fraction type must be exactly 'ao' or 'wo', got '{fraction_type}'")


def _is_natural_zaid(zaid: int) -> bool:
    """Return True for natural elements (mass number exactly 0 -> ZAID ends with '000')."""
    return zaid > 1000 and zaid % 1000 == 0


@dataclass
class Nuclide:
    """Lightweight nuclide entry for a material.

    Attributes
    ----------
    zaid : int
        ZAID identifier (e.g. 92235 for U-235).
    fraction : float
        Composition fraction stored as a positive value; interpretation is driven by
        the parent ``Material.fraction_type``.
    libs : dict
        Per-nuclide library overrides keyed by library keyword (e.g. ``'nlib'``, ``'plib'``).
    """

    zaid: int
    fraction: float
    libs: Dict[str, str] = field(default_factory=dict)

    @property
    def element(self) -> Optional[str]:
        """Element symbol for atomic number ``zaid // 1000``, or None if unknown."""
        atomic_number = self.zaid // 1000
        return ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)

    @property
    def symbol(self) -> str:
        """Return the element-mass symbol (e.g., 'Fe56', 'U235')."""
        return zaid_to_symbol(self.zaid)

    @property
    def is_natural(self) -> bool:
        """True for a natural element: ``zaid`` above 1000 and a multiple of 1000."""
        return _is_natural_zaid(self.zaid)

    def convert_natural_element(self) -> None:
        """Print an isotopic breakdown of a natural element.

        Raises
        ------
        ValueError
            If ``zaid`` does not represent a natural element or if abundance data
            are missing.
        """
        if not self.is_natural:
            raise ValueError(
                f"Nuclide with ZAID {self.zaid} is not a natural element (mass number must be 0, e.g. ZAID ends with '000')"
            )

        atomic_number = self.zaid // 1000
        atomic_key = atomic_number * 1000

        if atomic_key not in NATURAL_ABUNDANCE:
            raise ValueError(f"No natural abundance data available for element ZAID {self.zaid}")

        original_fraction = abs(self.fraction)
        natural_isotopes = NATURAL_ABUNDANCE[atomic_key]

        if self.fraction < 0:
            total_mass_per_atom = 0.0
            isotope_masses = {}
            for iso_zaid, atomic_abundance in natural_isotopes.items():
                iso_mass = ATOMIC_MASS.get(iso_zaid, iso_zaid % 1000)
                isotope_masses[iso_zaid] = iso_mass
                total_mass_per_atom += atomic_abundance * iso_mass

            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                iso_mass = isotope_masses[iso_zaid]
                iso_mass_fraction = (atomic_abundance * iso_mass) / total_mass_per_atom
                iso_mass_density = -(original_fraction * iso_mass_fraction)
                print(f"{iso_zaid} {iso_mass_density:.6e}")
        else:
            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                fraction = original_fraction * atomic_abundance
                print(f"{iso_zaid} {fraction:.6e}")

    def __str__(self) -> str:
        """Return a user-friendly string representation of the nuclide."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"{'Nuclide: ' + self.symbol:^70}")
        lines.append("=" * 70)

        # Format attributes in two columns
        lines.append(f"{'ZAID:':<25} {self.zaid}")
        lines.append(f"{'Element:':<25} {self.element or 'Unknown'}")
        lines.append(f"{'Mass Number:':<25} {self.zaid % 1000}")
        lines.append(f"{'Fraction:':<25} {self.fraction:.6e}")
        lines.append(f"{'Is Natural:':<25} {self.is_natural}")

        if self.libs:
            lines.append("")
            lines.append(f"{'Library Overrides:'}")
            for key, value in sorted(self.libs.items()):
                lines.append(f"  {key:<23} {value}")
        else:
            lines.append("")
            lines.append(f"{'Library Overrides:':<25} None")

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the same formatted representation as __str__."""
        return self.__str__()


class NuclideAccessor:
    """Dictionary-like accessor that allows retrieving nuclides by ZAID or symbol.

    This wrapper allows accessing nuclides using either:
    - Symbol (str): e.g., 'Fe56', 'U235', 'Fe' (natural)
    - ZAID (int): e.g., 26056, 92235, 26000 (natural)

    String symbols are normalized through ZAID conversion for consistency,
    so 'Fe', 'fe', and 26000 all map to the same nuclide.

    Examples
    --------
    >>> material.nuclide['Fe56']  # Access by symbol
    >>> material.nuclide[26056]    # Access by ZAID
    >>> material.nuclide['Fe']     # Access natural Fe
    >>> material.nuclide[26000]    # Access natural Fe by ZAID

    Parameters
    ----------
    nuclide_dict
        The dictionary of :class:`~kika.mcnp.material.Nuclide` entries, keyed by
        symbol, that the accessor wraps. It is not copied: changes through the
        accessor change it.
    """

    def __init__(self, nuclide_dict: Dict[str, Nuclide]):
        self._dict = nuclide_dict

    def __getitem__(self, key: Union[str, int]) -> Nuclide:
        """Get nuclide by symbol (str) or ZAID (int)."""
        if isinstance(key, int):
            # Convert ZAID to symbol
            symbol = zaid_to_symbol(key)
        else:
            # Normalize string through ZAID conversion for consistency
            # This ensures 'Fe', 'fe', 'FE' all map to the same key
            zaid = symbol_to_zaid(key)
            symbol = zaid_to_symbol(zaid)
        return self._dict[symbol]

    def __setitem__(self, key: Union[str, int], value: Nuclide) -> None:
        """Set nuclide by symbol (str) or ZAID (int)."""
        if isinstance(key, int):
            # Convert ZAID to symbol
            symbol = zaid_to_symbol(key)
        else:
            # Normalize string through ZAID conversion
            zaid = symbol_to_zaid(key)
            symbol = zaid_to_symbol(zaid)
        self._dict[symbol] = value

    def __delitem__(self, key: Union[str, int]) -> None:
        """Delete nuclide by symbol (str) or ZAID (int)."""
        if isinstance(key, int):
            # Convert ZAID to symbol
            symbol = zaid_to_symbol(key)
        else:
            # Normalize string through ZAID conversion
            zaid = symbol_to_zaid(key)
            symbol = zaid_to_symbol(zaid)
        del self._dict[symbol]

    def __contains__(self, key: Union[str, int]) -> bool:
        """Check if nuclide exists by symbol (str) or ZAID (int)."""
        if isinstance(key, int):
            # Convert ZAID to symbol
            symbol = zaid_to_symbol(key)
        else:
            # Normalize string through ZAID conversion
            try:
                zaid = symbol_to_zaid(key)
                symbol = zaid_to_symbol(zaid)
            except ValueError:
                return False
        return symbol in self._dict

    def __iter__(self):
        """Iterate over nuclide symbols."""
        return iter(self._dict)

    def __len__(self) -> int:
        """Return number of nuclides."""
        return len(self._dict)

    def keys(self):
        """Return view of nuclide symbols."""
        return self._dict.keys()

    def values(self):
        """Return view of nuclide objects."""
        return self._dict.values()

    def items(self):
        """Return view of (symbol, nuclide) pairs."""
        return self._dict.items()

    def get(self, key: Union[str, int], default=None):
        """Get nuclide by symbol or ZAID with optional default.

        Parameters
        ----------
        key
            Symbol or ZAID, normalised as in ``accessor[key]``.
        default
            Returned when *key* is not present or is not a valid symbol.

        Returns
        -------
        ~kika.mcnp.material.Nuclide or object
            The nuclide, or *default*.
        """
        try:
            return self[key]
        except (KeyError, ValueError):
            return default


@dataclass
class Material:
    """General-purpose material representation.

    Fractions are stored as non-negative numbers; ``fraction_type`` records whether
    they represent atomic (``'ao'``) or weight (``'wo'``) fractions. This class provides
    methods for managing nuclide composition, density, and metadata for a material.

    Parameters
    ----------
    id : int
        Material identifier number.
    nuclide : dict, optional
        Mapping of ZAID to :class:`Nuclide` objects.
    libs : dict, optional
        Default library keywords (e.g. ``'nlib'``). Nuclide libraries override these defaults.
    name : str, optional
        Optional descriptive name.
    fraction_type : str, optional
        ``'ao'`` for atomic fractions or ``'wo'`` for weight fractions.
    density : float, optional
        Density magnitude paired with ``density_unit``.
    density_unit : str, optional
        Density unit (``'g/cc'`` or ``'kg/m3'``).
    temperature : float, optional
        Material temperature if available in Kelvin.
    metadata : dict, optional
        Arbitrary metadata tags.
    """

    id: int
    nuclide: Union[Dict[str, Nuclide], NuclideAccessor] = field(default_factory=dict)  # Keyed by symbol, accessible by symbol or ZAID
    libs: Dict[str, str] = field(default_factory=dict)
    name: Optional[str] = None
    fraction_type: str = "ao"
    density: Optional[float] = None
    density_unit: Optional[str] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_fraction_type(self.fraction_type)
        self.libs = dict(self.libs)

        # Convert nuclide to internal dict if needed, then wrap with accessor
        if isinstance(self.nuclide, dict) and not isinstance(self.nuclide, NuclideAccessor):
            nuclide_dict = dict(self.nuclide)
            for nuc in nuclide_dict.values():
                nuc.fraction = float(abs(nuc.fraction))
                nuc.libs = dict(nuc.libs)
            self.nuclide = NuclideAccessor(nuclide_dict)
        elif isinstance(self.nuclide, NuclideAccessor):
            for nuc in self.nuclide.values():
                nuc.fraction = float(abs(nuc.fraction))
                nuc.libs = dict(nuc.libs)

        if self.density is not None:
            self.density = float(self.density)
        self.density_unit = _normalize_density_unit(self.density_unit)

    @property
    def is_atomic(self) -> bool:
        """True when ``fraction_type`` is ``'ao'`` (atomic fractions)."""
        return self.fraction_type == "ao"

    @property
    def is_weight(self) -> bool:
        """True when ``fraction_type`` is ``'wo'`` (weight fractions)."""
        return self.fraction_type == "wo"

    @property
    def density_with_units(self) -> Optional[str]:
        """Return density formatted with units for display.

        Returns
        -------
        str or None
            e.g., "10.97 g/cc" or None if not set.
        """
        if self.density is not None and self.density_unit is not None:
            return f"{self.density:.6g} {self.density_unit}"
        return None

    def _infer_lib_key_from_suffix(self, library: str) -> Optional[str]:
        """Library keyword for the data class letter that ends *library* (``'80c'`` -> ``'nlib'``), or None."""
        if not library:
            return None
        return _LIB_KEY_BY_CLASS.get(library[-1].lower())

    def set_name(self, name: Optional[str]) -> None:
        """Set or change the material name.

        Parameters
        ----------
        name : str or None
            New name for the material, or None to clear.

        Returns
        -------
        None
        """
        self.name = name

    def set_id(self, new_id: int) -> None:
        """Change the material ID.

        Parameters
        ----------
        new_id : int
            New material ID.

        Returns
        -------
        None
        """
        self.id = int(new_id)

    def set_library(self, lib_key: str, lib_value: str) -> None:
        """Set a material-level library default.

        Parameters
        ----------
        lib_key : str
            Library keyword (e.g., 'nlib', 'plib', 'ylib').
        lib_value : str
            Library suffix value (e.g., '80c', '12p').

        Returns
        -------
        None
        """
        self.libs[lib_key] = lib_value

    def remove_library(self, lib_key: str) -> None:
        """Remove a material-level library setting.

        Parameters
        ----------
        lib_key : str
            Library keyword to remove (e.g., 'nlib', 'plib', 'ylib').

        Returns
        -------
        None
        """
        self.libs.pop(lib_key, None)

    def add_nuclide(self, nuclide: Union[int, str], fraction: float, fraction_type: str,
                    library: Optional[str] = None) -> None:
        """Add or update a nuclide in the material.

        Parameters
        ----------
        nuclide : int or str
            ZAID identifier (int, e.g. 92235) or element-mass symbol (str, e.g. 'U235', 'Fe56').
            For natural elements, use symbol without mass number (e.g. 'Fe' -> 26000).
        fraction : float
            Fraction value; the absolute value is stored and interpreted according
            to the provided ``fraction_type``.
        fraction_type : str
            Fraction type for this addition: ``'ao'`` for atomic fractions or ``'wo'`` for
            weight fractions. It must be the material's own type when the material
            holds other nuclides (convert first with :meth:`to_atomic_fraction` or
            :meth:`to_weight_fraction`); a material with no other nuclide takes it.
        library : str, optional
            MCNP library suffix (e.g. ``'80c'``, ``'710nc'``, ``'12p'``), a nuclide-level
            override. Its last letter, the MCNP data class, picks the keyword it is
            stored under: ``c``/``d`` neutron (``nlib``), ``p`` photoatomic
            (``plib``), ``u`` photonuclear (``pnlib``), ``e`` electron (``elib``),
            ``h`` proton (``hlib``), ``o`` deuteron (``dlib``), ``r`` triton
            (``tlib``), ``s`` helium-3 (``slib``), ``a`` alpha (``alib``) and ``y``
            dosimetry (``ylib``). Thermal-scattering tables (``t``) belong on the
            MCNP ``mt`` card, not here.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If nuclide is a string and does not represent a valid element symbol,
            if *fraction_type* differs from the type of the nuclides already in the
            material, or if the class letter of *library* is none of the above.

        Examples
        --------
        >>> mat = Material(id=1)
        >>> mat.add_nuclide('U235', 1.0, 'ao')  # Add U-235 as atomic fraction
        >>> mat.add_nuclide(92238, 0.5, 'ao')   # Add U-238 using ZAID
        >>> mat.add_nuclide('Fe', 1.0, 'ao')    # Add natural Fe (26000)
        """
        # Convert symbol to zaid if necessary
        if isinstance(nuclide, str):
            zaid = symbol_to_zaid(nuclide)
        else:
            zaid = nuclide

        _validate_fraction_type(fraction_type)
        symbol = zaid_to_symbol(zaid)
        others = [s for s in self.nuclide if s != symbol]
        if others and fraction_type != self.fraction_type:
            raise ValueError(
                f"Material {self.id} holds '{self.fraction_type}' fractions; adding {symbol} "
                f"as '{fraction_type}' would reinterpret the others. Convert the material first."
            )
        self.fraction_type = fraction_type

        lib_key = self._infer_lib_key_from_suffix(library) if library else None
        if library and lib_key is None:
            raise ValueError(
                f"Library suffix '{library}' of {symbol} ends in an MCNP data class kika does not "
                f"place on an m card (known: {', '.join(sorted(_LIB_KEY_BY_CLASS))})"
            )

        if symbol in self.nuclide:
            nuc = self.nuclide[symbol]
            nuc.fraction = float(abs(fraction))
        else:
            nuc = Nuclide(zaid=zaid, fraction=float(abs(fraction)))
            self.nuclide[symbol] = nuc

        if lib_key is not None and library is not None:
            nuc.libs[lib_key] = library

    def add_element(self, element: str, fraction: float, fraction_type: str = "ao",
                    library: Optional[str] = None) -> None:
        """Add a natural element to the material.

        This method is a convenience for adding natural elements by symbol only (without
        mass number). For specific isotopes, use ``add_nuclide()``.

        Parameters
        ----------
        element : str
            Element symbol (e.g., 'Fe', 'U', 'O'). Will be converted to natural ZAID
            (mass number = 0, e.g., 'Fe' -> 26000).
        fraction : float
            Fraction value; the absolute value is stored and interpreted according
            to the provided ``fraction_type``.
        fraction_type : str, optional
            Fraction type: ``'ao'`` for atomic fractions or ``'wo'`` for weight fractions.
            Default is ``'ao'``.
        library : str, optional
            MCNP library suffix (e.g. ``'80c'``, ``'70c'``, ``'12p'``).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If element is not a recognized element symbol.

        Examples
        --------
        >>> mat = Material(id=1)
        >>> mat.add_element('Fe', 1.0, 'ao')      # Add natural Fe with atomic fraction
        >>> mat.add_element('O', 2.0, 'wo')       # Add natural O with weight fraction
        """
        self.add_nuclide(element, fraction, fraction_type, library)

    def remove_nuclide(self, nuclide: Union[int, str]) -> None:
        """Remove a nuclide from the material.

        Parameters
        ----------
        nuclide : int or str
            ZAID identifier or symbol of the nuclide to remove.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the nuclide is not found in the material.
        """
        del self.nuclide[nuclide]

    def update_nuclide_fraction(self, nuclide: Union[int, str], fraction: float) -> None:
        """Update the fraction of an existing nuclide.

        Parameters
        ----------
        nuclide : int or str
            ZAID identifier or symbol of the nuclide to update.
        fraction : float
            New fraction value (absolute value will be used).

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the nuclide is not found in the material.
        """
        nuc = self.nuclide[nuclide]
        nuc.fraction = float(abs(fraction))

    def clear_nuclides(self) -> None:
        """Remove all nuclides from the material.

        Returns
        -------
        None
        """
        self.nuclide._dict.clear()

    def _get_nuclide_symbol(self, zaid: int) -> Optional[str]:
        atomic_number = zaid // 1000
        return ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)

    def get_effective_library(self, nuclide: Union[int, str], lib_key: str = "nlib") -> Optional[str]:
        """Return the effective MCNP library for a given nuclide.

        Parameters
        ----------
        nuclide : int or str
            ZAID identifier or symbol.
        lib_key : str, optional
            Library keyword to query (e.g. ``'nlib'``, ``'plib'``, ``'ylib'``).

        Returns
        -------
        str or None
            Nuclide-level override if present, otherwise the material default.
        """
        # Convert to symbol if ZAID provided
        if isinstance(nuclide, int):
            symbol = zaid_to_symbol(nuclide)
        else:
            symbol = nuclide

        nuc = self.nuclide.get(symbol)
        if nuc is not None and lib_key in nuc.libs:
            return nuc.libs[lib_key]
        return self.libs.get(lib_key)

    def set_density(self, density: float, unit: str) -> None:
        """Set the material density with explicit units.

        Parameters
        ----------
        density : float
            Density magnitude.
        unit : str
            ``'g/cc'``, ``'kg/m3'`` or ``'atoms/b-cm'``, case-insensitive and ignoring
            spaces; the aliases ``'g/cm3'``, ``'g/cm^3'``, ``'kg/m^3'``,
            ``'atoms/barn-cm'`` and ``'a/b-cm'`` are accepted and stored under the
            canonical name.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If *unit* is none of the three.
        """
        normalized_unit = _normalize_density_unit(unit)
        if normalized_unit not in {"g/cc", "kg/m3", "atoms/b-cm"}:
            raise ValueError(f"Unsupported density unit '{unit}'. Use 'g/cc', 'kg/m3', or 'atoms/b-cm'.")
        self.density = float(density)
        self.density_unit = normalized_unit

    def set_temperature(self, temperature: float) -> None:
        """Set the material temperature in Kelvin.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        None
        """
        self.temperature = float(temperature)

    def density_in(self, unit: str) -> float:
        """Return the current density expressed in a different unit.

        Parameters
        ----------
        unit : str
            Target unit string ('g/cc' or 'kg/m3').

        Returns
        -------
        float
            Density value in the requested unit.

        Raises
        ------
        ValueError
            If density is not set or unit is unsupported.
        """
        if self.density is None or self.density_unit is None:
            raise ValueError("Density or density_unit not set.")
        target_unit = _normalize_density_unit(unit)
        if target_unit not in {"g/cc", "kg/m3", "atoms/b-cm"}:
            raise ValueError(f"Unsupported density unit '{unit}'. Use 'g/cc', 'kg/m3', or 'atoms/b-cm'.")

        # Direct conversion between g/cc and kg/m3
        if self.density_unit != "atoms/b-cm" and target_unit != "atoms/b-cm":
            factor = _DENSITY_CONVERSIONS[(self.density_unit, target_unit)]
            return self.density * factor

        # Identity
        if self.density_unit == target_unit:
            return self.density

        # Composition-aware conversions involving atoms/b-cm
        if self.density_unit == "atoms/b-cm" and target_unit in ("g/cc", "kg/m3"):
            mass_density = self._calculate_mass_density()
            if mass_density is None:
                raise ValueError("Cannot convert atoms/b-cm to g/cc: insufficient composition data.")
            if target_unit == "kg/m3":
                return mass_density * 1000.0
            return mass_density

        if self.density_unit in ("g/cc", "kg/m3") and target_unit == "atoms/b-cm":
            atomic_density = self._calculate_atomic_density()
            if atomic_density is None:
                raise ValueError("Cannot convert to atoms/b-cm: insufficient composition data.")
            return atomic_density

    def convert_density(self, unit: str) -> None:
        """Convert density in-place to the requested unit.

        Parameters
        ----------
        unit : str
            Target unit string ('g/cc' or 'kg/m3').

        Returns
        -------
        None
        """
        converted = self.density_in(unit)
        self.density = converted
        self.density_unit = _normalize_density_unit(unit)

    def _get_effective_atomic_mass(self, zaid: int) -> float:
        """Return the effective atomic mass, handling natural elements.

        Parameters
        ----------
        zaid : int
            ZAID identifier for which to retrieve mass.

        Returns
        -------
        float
            Atomic mass; for natural elements the abundance-weighted mean is used.
        """
        # Check if this is a natural element (ZAID mass number equals 0)
        if _is_natural_zaid(zaid):
            atomic_number = zaid // 1000
            atomic_key = atomic_number * 1000  # e.g., 6000 for carbon

            # Handle natural elements using abundance data
            if atomic_key in NATURAL_ABUNDANCE:
                natural_isotopes = NATURAL_ABUNDANCE[atomic_key]
                weighted_mass = 0.0

                for iso_zaid, abundance in natural_isotopes.items():
                    # Get isotope mass from constants or approximate
                    iso_mass = ATOMIC_MASS.get(iso_zaid, iso_zaid % 1000)
                    weighted_mass += abundance * iso_mass

                return weighted_mass

        # For specific isotopes or fallback
        atomic_number = zaid // 1000
        mass_number = zaid % 1000

        # Construct the isotope key
        isotope_key = atomic_number * 1000 + mass_number

        # Get atomic mass from constants or use a default approximation
        atomic_mass = ATOMIC_MASS.get(isotope_key)
        if atomic_mass is None:
            # Fallback to approximate mass if exact isotope not found
            atomic_mass = float(mass_number) if mass_number > 0 else 1.0

        return atomic_mass

    def to_weight_fraction(self) -> Material:
        """Convert atomic fractions to weight fractions, in place.

        The stored fractions are always positive; ``fraction_type`` is what says they
        are atomic or weight. Each fraction ``x`` becomes ``x * m / sum(x * m)``, so
        the result sums to 1, with ``m`` the atomic mass of the nuclide (the
        abundance-weighted mean for a natural element). Without a tabulated mass,
        ``m`` is the mass number for an isotope and 1.0 for a natural element with
        no abundance data. A material that is already ``'wo'`` is returned unchanged,
        without renormalising.

        Returns
        -------
        Material
            This material, with ``fraction_type`` ``'wo'``.

        Raises
        ------
        ValueError
            If the total of ``x * m`` is zero or negative (for instance, no nuclides).
        """
        if self.is_weight:
            return self

        mass_sum = 0.0
        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            mass_sum += nuclide.fraction * atomic_mass

        if mass_sum <= 0:
            raise ValueError(f"Cannot convert to weight fractions: total mass calculated as {mass_sum}")

        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            nuclide.fraction = nuclide.fraction * atomic_mass / mass_sum

        self.fraction_type = "wo"
        return self

    def to_atomic_fraction(self) -> Material:
        """Convert weight fractions to atomic fractions, in place.

        The stored fractions are always positive; ``fraction_type`` is what says they
        are atomic or weight. Each fraction ``w`` becomes ``(w / m) / sum(w / m)``, so
        the result sums to 1, with ``m`` the atomic mass of the nuclide (the
        abundance-weighted mean for a natural element). Without a tabulated mass,
        ``m`` is the mass number for an isotope and 1.0 for a natural element with
        no abundance data. A material that is already ``'ao'`` is returned unchanged,
        without renormalising.

        Returns
        -------
        Material
            This material, with ``fraction_type`` ``'ao'``.

        Raises
        ------
        ValueError
            If the total of ``w / m`` is zero or negative (for instance, no nuclides).
        """
        if self.is_atomic:
            return self

        atomic_sum = 0.0
        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            if atomic_mass <= 0:
                mass_number = nuclide.zaid % 1000
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            atomic_sum += nuclide.fraction / atomic_mass

        if atomic_sum <= 0:
            raise ValueError("Cannot convert to atomic fractions: atomic_sum calculated as 0 or negative.")

        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            if atomic_mass <= 0:
                mass_number = nuclide.zaid % 1000
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            nuclide.fraction = (nuclide.fraction / atomic_mass) / atomic_sum

        self.fraction_type = "ao"
        return self

    def normalize(self, fraction_type: str = "ao") -> Material:
        """Normalize all fractions so they sum to 1.0.

        This method first converts all fractions to the specified fraction_type (atomic or weight),
        then rescales them proportionally so that the total sum equals 1.0. This is useful after
        manually adding nuclides with arbitrary fractional values, such as defining water
        as H with fraction 2.0 and O with fraction 1.0, then normalizing.

        Parameters
        ----------
        fraction_type : str, optional
            Target fraction type for normalization: ``'ao'`` for atomic fractions or
            ``'wo'`` for weight fractions. Default is ``'ao'``.

        Returns
        -------
        Material
            Self reference for chaining.

        Raises
        ------
        ValueError
            If total fraction sum is zero or negative, or if fraction_type is invalid.

        Examples
        --------
        >>> mat = Material(id=1)
        >>> mat.add_nuclide('H', 2.0, 'ao')
        >>> mat.add_nuclide('O', 1.0, 'ao')
        >>> mat.normalize('ao')  # Normalize to atomic fractions (default)
        >>> mat.normalize('wo')  # Or normalize to weight fractions
        """
        # Validate fraction type
        _validate_fraction_type(fraction_type)

        # Convert to target fraction type if needed
        if fraction_type == "wo" and self.is_atomic:
            self.to_weight_fraction()
        elif fraction_type == "ao" and self.is_weight:
            self.to_atomic_fraction()

        # Now normalize
        total_fraction = sum(nuc.fraction for nuc in self.nuclide.values())

        if total_fraction <= 0:
            raise ValueError(f"Cannot normalize: total fraction sum is {total_fraction} (must be positive)")

        if total_fraction != 1.0:
            for nuclide in self.nuclide.values():
                nuclide.fraction = nuclide.fraction / total_fraction

        return self

    def to_integer_fractions(self, precision: int = 6) -> Material:
        """Convert fractions to integer ratios (inverse of normalize).

        This method converts normalized fractions back to integer ratios using
        rational number approximation. For example, water with normalized atomic
        fractions (H: 0.6667, O: 0.3333) becomes (H: 2, O: 1).

        Parameters
        ----------
        precision : int, optional
            Maximum denominator for the rational approximation. Higher values
            give more accurate results but may produce larger integers.
            Default is 6 (max denominator of 10^6).

        Returns
        -------
        Material
            Self reference for chaining.

        Notes
        -----
        - Works for both atomic and weight fractions (fractions are just numbers).
        - Results are approximate due to float precision.
        - The fractions will still sum to an integer total, not necessarily 1.0.

        Examples
        --------
        >>> mat = Material(id=1)
        >>> mat.add_nuclide('H', 0.6667, 'ao')
        >>> mat.add_nuclide('O', 0.3333, 'ao')
        >>> mat.to_integer_fractions()
        >>> # H: 2.0, O: 1.0
        """
        from fractions import Fraction
        from math import gcd
        from functools import reduce

        if len(self.nuclide) == 0:
            return self

        max_denom = 10 ** precision

        # Convert each fraction to a rational approximation
        rationals = {}
        for symbol, nuc in self.nuclide.items():
            frac = Fraction(nuc.fraction).limit_denominator(max_denom)
            rationals[symbol] = frac

        # Find LCM of all denominators
        denominators = [f.denominator for f in rationals.values()]

        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        common_denom = reduce(lcm, denominators, 1)

        # Multiply all fractions by the LCM to get integers
        integer_fracs = {}
        for symbol, frac in rationals.items():
            integer_fracs[symbol] = int(frac * common_denom)

        # Find GCD of all numerators to reduce to smallest integers
        numerators = list(integer_fracs.values())
        common_factor = reduce(gcd, numerators)

        if common_factor > 1:
            for symbol in integer_fracs:
                integer_fracs[symbol] //= common_factor

        # Update the nuclide fractions
        for symbol, int_frac in integer_fracs.items():
            self.nuclide[symbol].fraction = float(int_frac)

        return self

    def expand_natural_elements(self, elements: Optional[Union[str, List[str]]] = None) -> Material:
        """Expand natural elements into isotopic composition using abundance data.

        Parameters
        ----------
        elements : str or list[str], optional
            Specific natural element symbol(s) to expand (e.g., 'Fe', 'C').
            If ``None``, all natural elements in the material are expanded.

        Returns
        -------
        Material
            Self reference for chaining.

        Raises
        ------
        ValueError
            If a requested element is missing, not natural, or lacks abundance data.
        """
        original_fraction_type = self.fraction_type

        if self.is_weight:
            self.to_atomic_fraction()

        if elements is not None:
            # Convert to list of symbols
            symbols_list = [elements] if isinstance(elements, str) else elements
            natural_symbols: List[str] = []

            for elem in symbols_list:
                # Convert element symbol to natural symbol (e.g., 'Fe' -> 'Fe0')
                zaid = symbol_to_zaid(elem)
                symbol = zaid_to_symbol(zaid)

                if symbol not in self.nuclide:
                    raise ValueError(f"Element '{elem}' (symbol '{symbol}') not found in material {self.id}")
                if not _is_natural_zaid(zaid):
                    raise ValueError(
                        f"Element '{elem}' is not a natural element (must have mass number 0)"
                    )
                natural_symbols.append(symbol)
        else:
            # Find all natural elements
            natural_symbols = [sym for sym, nuc in self.nuclide.items() if nuc.is_natural]

        for symbol in natural_symbols:
            nuclide = self.nuclide[symbol]
            zaid = nuclide.zaid
            atomic_number = zaid // 1000

            atomic_key = atomic_number * 1000
            if atomic_key not in NATURAL_ABUNDANCE:
                raise ValueError(f"No natural abundance data available for element {symbol}")

            original_fraction = nuclide.fraction
            natural_isotopes = NATURAL_ABUNDANCE[atomic_key]
            new_isotopes: Dict[str, Nuclide] = {}

            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                fraction = original_fraction * atomic_abundance
                iso_symbol = zaid_to_symbol(iso_zaid)
                new_isotopes[iso_symbol] = Nuclide(
                    zaid=iso_zaid,
                    fraction=fraction,
                    libs=dict(nuclide.libs),
                )

            for iso_symbol, isotope in new_isotopes.items():
                self.nuclide[iso_symbol] = isotope
            del self.nuclide[symbol]

        if original_fraction_type == "wo":
            self.to_weight_fraction()

        return self

    def _format_fraction(self, fraction: float) -> str:
        """Format a fraction value for MCNP output.

        Parameters
        ----------
        fraction : float
            Fraction value to format.

        Returns
        -------
        str
            Formatted fraction in scientific notation.
        """
        return f"{fraction:.6e}"

    def _card_suffix(self, nuclide: Nuclide) -> str:
        """The single suffix a nuclide's m-card entry carries: its neutron library when it has one."""
        # Listing a nuclide once per library would count its atoms once per library.
        chosen = "nlib" if "nlib" in nuclide.libs else sorted(nuclide.libs)[0]
        dropped = {k: v for k, v in nuclide.libs.items() if k != chosen}
        if dropped:
            warnings.warn(
                f"Material {self.id}: {nuclide.symbol} is written as {nuclide.zaid}."
                f"{nuclide.libs[chosen]}; its other libraries {dropped} have no place on an "
                f"MCNP m card and are not written",
                stacklevel=3,
            )
        return nuclide.libs[chosen]

    def to_mcnp(self) -> str:
        """Serialise the material as an MCNP ``m`` card.

        The output includes MCNP comments with:
        - Material name (if set) as a comment before the material card
        - Density in both g/cc and atoms/b-cm (if set) as a comment after libraries
        - Nuclide symbol after each fraction line

        Returns
        -------
        str
            MCNP material card formatted as a multi-line string.
        """
        output_lines = []

        # Add material name as comment if present
        if self.name:
            output_lines.append(f"c KIKA_MAT_NAME: {self.name}")

        # Build main material line
        main_line = f"m{self.id}"
        lib_parts = [f"{k}={v}" for k, v in sorted(self.libs.items())]
        if lib_parts:
            main_line += " " + " ".join(lib_parts)

        # Add density comment if density is set
        if self.density is not None and self.density_unit is not None:
            try:
                density_gcc = self.density_in("g/cc")
                atomic_density = self._calculate_atomic_density()
                if atomic_density is not None:
                    main_line += f"  $ KIKA_DENSITY: {density_gcc:.6e} g/cc, {atomic_density:.6e} atoms/b-cm"
                else:
                    main_line += f"  $ KIKA_DENSITY: {density_gcc:.6e} g/cc"
            except (ValueError, TypeError):
                # Conversion to g/cc may fail for atoms/b-cm without sufficient
                # composition data — output what we have
                if self.density_unit == "atoms/b-cm":
                    mass_density = self._calculate_mass_density()
                    if mass_density is not None:
                        main_line += f"  $ KIKA_DENSITY: {mass_density:.6e} g/cc, {self.density:.6e} atoms/b-cm"
                    else:
                        main_line += f"  $ KIKA_DENSITY: {self.density:.6e} atoms/b-cm"
                else:
                    main_line += f"  $ KIKA_DENSITY: {self.density:.6e} {self.density_unit}"

        output_lines.append(main_line)

        sign = -1.0 if self.is_weight else 1.0
        for symbol, nuclide in sorted(self.nuclide.items()):
            zaid = nuclide.zaid
            # Get element symbol for comment
            nuclide_symbol = nuclide.symbol

            zaid_lib = f"{zaid}.{self._card_suffix(nuclide)}" if nuclide.libs else str(zaid)
            output_lines.append(
                f"     {zaid_lib:<12} {self._format_fraction(sign * nuclide.fraction):>14}   $ {nuclide_symbol:<6}"
            )

        return "\n".join(output_lines)

    def to_serpent(self) -> str:
        """Serialise the material as a Serpent material card.

        Serpent format::

            mat <name> <density> [rgb R G B] [tmp <T>] [moder <lib> <ZA>]
                <zaid>.<lib> <fraction>

        The material name is used if set, otherwise ``'mat<id>'``.
        Density uses negative values for mass density (g/cc) and positive
        for atomic density (atoms/b-cm), following Serpent convention.

        Returns
        -------
        str
            Serpent material card formatted as a multi-line string.
        """
        output_lines = []

        mat_name = self.name if self.name else f"mat{self.id}"
        mat_name = mat_name.replace(" ", "_")

        # Density: preserve original unit
        if self.metadata.get("serpent_density_sum"):
            density_str = "sum"
        elif self.density is not None and self.density_unit is not None:
            if self.density_unit == "atoms/b-cm":
                density_str = f"{self.density:.6e}"
            else:
                density_gcc = self._mass_density_gcc()
                if density_gcc is not None:
                    density_str = f"{-density_gcc:.6e}"
                else:
                    density_str = f"{-self.density:.6e}"
        else:
            density_str = "-1.000000e+00"

        mat_line = f"mat {mat_name} {density_str}"

        # Serpent modifiers
        if self.metadata.get("rgb"):
            r, g, b = self.metadata["rgb"]
            mat_line += f" rgb {r} {g} {b}"

        if self.temperature is not None:
            mat_line += f" tmp {self.temperature:.1f}"

        if self.metadata.get("tms") is not None:
            mat_line += f" tms {self.metadata['tms']:.1f}"

        if self.metadata.get("tft") is not None:
            tmin, tmax = self.metadata["tft"]
            mat_line += f" tft {tmin:.1f} {tmax:.1f}"

        if self.metadata.get("moder"):
            for lib_name, za in self.metadata["moder"]:
                mat_line += f" moder {lib_name} {za}"

        if self.metadata.get("burn") is not None:
            mat_line += f" burn {self.metadata['burn']}"

        if self.metadata.get("vol") is not None:
            mat_line += f" vol {self.metadata['vol']:.6e}"

        if self.metadata.get("fix") is not None:
            fix_id, fix_t = self.metadata["fix"]
            mat_line += f" fix {fix_id} {fix_t:.1f}"

        output_lines.append(mat_line)

        # Output nuclides
        sign = -1.0 if self.is_weight else 1.0

        for symbol, nuclide in sorted(self.nuclide.items()):
            zaid = nuclide.zaid

            lib_suffix = None
            if nuclide.libs:
                lib_suffix = nuclide.libs.get('nlib', next(iter(nuclide.libs.values()), None))
            elif self.libs:
                lib_suffix = self.libs.get('nlib', next(iter(self.libs.values()), None))

            if lib_suffix:
                zaid_str = f"{zaid}.{lib_suffix}"
            else:
                zaid_str = str(zaid)

            fraction_str = f"{sign * nuclide.fraction:.6e}"
            output_lines.append(f"{zaid_str:<12} {fraction_str:>14}")

        return "\n".join(output_lines)

    def _effective_avg_atomic_mass(self) -> Optional[float]:
        """Return the effective average atomic mass for this material.

        For atomic fractions: M_eff = sum(x_i * M_i) / sum(x_i)
        For weight fractions: M_eff = sum(w_i) / sum(w_i / M_i)

        Returns
        -------
        float or None
            Average atomic mass in g/mol, or None if not calculable.
        """
        total_weight = 0.0
        total_fraction = 0.0

        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            if atomic_mass <= 0:
                return None

            if self.is_atomic:
                total_weight += nuclide.fraction * atomic_mass
                total_fraction += nuclide.fraction
            else:
                total_weight += nuclide.fraction / atomic_mass
                total_fraction += nuclide.fraction

        if total_fraction <= 0:
            return None

        if self.is_atomic:
            return total_weight / total_fraction
        else:
            return total_fraction / total_weight

    def _mass_density_gcc(self) -> Optional[float]:
        """Return mass density in g/cc without going through density_in (avoids recursion).

        Returns
        -------
        float or None
        """
        if self.density is None or self.density_unit is None:
            return None
        if self.density_unit == "g/cc":
            return self.density
        if self.density_unit == "kg/m3":
            return self.density * 0.001
        return None  # atoms/b-cm cannot be converted without composition

    def _calculate_atomic_density(self) -> Optional[float]:
        """Calculate atomic density in atoms/barn-cm from mass density.

        Returns
        -------
        float or None
            Atomic density in atoms/b-cm, or None if calculation is not possible.
        """
        if self.density is None or self.density_unit is None:
            return None

        if self.density_unit == "atoms/b-cm":
            return self.density

        density_gcc = self._mass_density_gcc()
        if density_gcc is None:
            return None

        avg_atomic_mass = self._effective_avg_atomic_mass()
        if avg_atomic_mass is None:
            return None

        N_A = 6.02214076e23
        atomic_density = (density_gcc * N_A / avg_atomic_mass) * 1e-24
        return atomic_density

    def _calculate_mass_density(self) -> Optional[float]:
        """Calculate mass density in g/cc from atomic density.

        Returns
        -------
        float or None
            Mass density in g/cc, or None if calculation is not possible.
        """
        if self.density is None or self.density_unit is None:
            return None

        if self.density_unit in ("g/cc", "kg/m3"):
            return self._mass_density_gcc()

        if self.density_unit != "atoms/b-cm":
            return None

        avg_atomic_mass = self._effective_avg_atomic_mass()
        if avg_atomic_mass is None:
            return None

        N_A = 6.02214076e23
        # Reverse of: atomic_density = (density_gcc * N_A / M) * 1e-24
        density_gcc = (self.density * avg_atomic_mass / N_A) * 1e24
        return density_gcc

    def __str__(self) -> str:
        """Return a user-friendly string representation of the material.

        Returns
        -------
        str
            Formatted string showing material properties and composition.
        """
        lines = []
        lines.append("=" * 50)
        lines.append(f"{'Material (ID: ' + str(self.id) + ')':^50}")
        lines.append("=" * 50)
        lines.append("")

        # Material name if set
        if self.name:
            lines.append(f"Name: {self.name}")

        # Density with units
        if self.density_with_units:
            lines.append(f"Density: {self.density_with_units}")

        # Temperature
        if self.temperature is not None:
            lines.append(f"Temperature: {self.temperature} K")

        # MCNP Libraries (material-level)
        if self.libs:
            lib_str = ", ".join(f"{k}={v}" for k, v in sorted(self.libs.items()))
            lines.append(f"Libraries: {lib_str}")

        # Total nuclides count
        lines.append(f"Total Nuclides: {len(self.nuclide)}")
        lines.append("")

        # Fraction type before the table
        fraction_type_label = "Atomic" if self.fraction_type == "ao" else "Weight"
        lines.append(f"Fraction Type: {fraction_type_label}")
        lines.append("")

        # Check if any nuclide has specific libraries
        has_nuclide_libs = any(nuc.libs for nuc in self.nuclide.values())

        # Nuclide composition table
        if self.nuclide:
            lines.append("-" * 50)
            if has_nuclide_libs:
                # Show libraries column only if there are nuclide-specific libraries
                lines.append(f"{'ZAID':^8} | {'Element':^8} | {'Fraction':^14} | {'Libraries':^12}")
            else:
                lines.append(f"{'ZAID':^8} | {'Element':^8} | {'Fraction':^14}")
            lines.append("-" * 50)

            # Sort by ZAID for consistent ordering
            sorted_nuclides = sorted(self.nuclide.items(), key=lambda x: x[1].zaid)

            for symbol, nuc in sorted_nuclides:
                element = nuc.element if nuc.element else "?"

                if has_nuclide_libs:
                    if nuc.libs:
                        lib_str = ", ".join(f"{v}" for k, v in sorted(nuc.libs.items()))
                        if len(lib_str) > 10:
                            lib_str = lib_str[:9] + "..."
                    else:
                        lib_str = "-"
                    lines.append(f"{nuc.zaid:^8} | {element:^8} | {nuc.fraction:^14.6e} | {lib_str:^12}")
                else:
                    lines.append(f"{nuc.zaid:^8} | {element:^8} | {nuc.fraction:^14.6e}")

            lines.append("-" * 50)

        # Usage information
        lines.append("")
        lines.append("Available methods:")
        lines.append("- .to_weight_fraction() - Convert material to weight fractions")
        lines.append("- .to_atomic_fraction() - Convert material to atomic fractions")
        lines.append("- .expand_natural_elements() - Expand natural elements to isotopes")
        lines.append("- .to_integer_fractions() - Convert normalized fractions to integers")
        lines.append("- .to_mcnp() - Export as MCNP material card")
        lines.append("- .to_serpent() - Export as Serpent material card")
        lines.append("- .copy(new_id) - Create a copy with a new material ID")
        lines.append("")
        lines.append("Examples of accessing data:")
        lines.append("- .nuclide['Fe56'] - Access nuclide by symbol")
        lines.append("- .nuclide[26056] - Access nuclide by ZAID")
        lines.append("- .nuclide['Fe'] or .nuclide[26000] - Access natural Fe")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the same formatted representation as __str__.

        Returns
        -------
        str
            Formatted material representation.
        """
        return self.__str__()

    def copy(self, new_id: int) -> Material:
        """Create an exact copy of this material with a new ID.

        Parameters
        ----------
        new_id : int
            Material ID for the new copy.

        Returns
        -------
        Material
            New material instance with all properties copied.
        """
        new_material = Material(
            id=new_id,
            nuclide={sym: Nuclide(zaid=nuc.zaid, fraction=nuc.fraction, libs=dict(nuc.libs)) for sym, nuc in self.nuclide.items()},
            libs=dict(self.libs),
            name=self.name,
            fraction_type=self.fraction_type,
            density=self.density,
            density_unit=self.density_unit,
            temperature=self.temperature,
            metadata=deepcopy(self.metadata),
        )
        return new_material


@dataclass
class MaterialCollection:
    """Container of materials keyed by ID, built by hand or read from an input file.

    :meth:`~kika.mcnp.material.MaterialCollection.from_mcnp`,
    :meth:`~kika.mcnp.material.MaterialCollection.from_serpent` and
    :func:`~kika.materials.read_serpent_materials` return one. Iterating gives the
    materials in insertion order.

    Attributes
    ----------
    by_id
        The materials, keyed by ``Material.id``. Serpent materials are keyed by
        their position in the file (1, 2, ...); ``by_name`` indexes them by name.
    """

    by_id: Dict[int, Material] = field(default_factory=dict)

    def __iter__(self) -> Iterable[Material]:
        return iter(self.by_id.values())

    def __len__(self) -> int:
        return len(self.by_id)

    def add_material(self, material: Material) -> None:
        """Add a material to the collection.

        Parameters
        ----------
        material : Material
            Material to add.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a material with the same ID already exists.
        """
        if material.id in self.by_id:
            raise ValueError(f"Material ID {material.id} already exists in collection")
        self.by_id[material.id] = material

    def to_mcnp(self) -> str:
        """Serialise all materials as MCNP material cards.

        Returns
        -------
        str
            All material cards concatenated with newlines.
        """
        return "\n".join(material.to_mcnp() for material in self.by_id.values())

    def to_serpent(self) -> str:
        """Serialise all materials as Serpent material cards.

        Returns
        -------
        str
            All material cards concatenated with blank lines between them.
        """
        return "\n\n".join(material.to_serpent() for material in self.by_id.values())

    def to_file(self, filepath: str, material_ids: Optional[List[int]] = None,
                header: bool = True) -> None:
        """Export materials to a new file in MCNP format.

        This creates a new file containing only material cards. Use this when
        you want to export materials separately from a full MCNP input file.

        Parameters
        ----------
        filepath : str
            Path to the output file.
        material_ids : list of int, optional
            Specific material IDs to write. If None, writes all materials.
        header : bool, optional
            Whether to include KIKA header/footer comments. Default is True.

        Returns
        -------
        None

        Examples
        --------
        >>> collection.to_file("materials.txt")  # Export all materials
        >>> collection.to_file("subset.txt", material_ids=[1, 2, 3])  # Export specific ones
        """
        from .._constants import MCNPY_HEADER, MCNPY_FOOTER

        # Determine which materials to write
        if material_ids is None:
            materials_to_write = list(self.by_id.values())
        else:
            materials_to_write = [self.by_id[mid] for mid in material_ids if mid in self.by_id]

        content_lines = []

        if header:
            content_lines.append(MCNPY_HEADER)
            content_lines.append("c \n")

        for material in materials_to_write:
            content_lines.append(material.to_mcnp() + "\n")
            content_lines.append("c \n")

        if header:
            content_lines.append(MCNPY_FOOTER)

        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.writelines(content_lines)

    def _materials_equal(self, mat1: Material, mat2: Material) -> bool:
        """Compare two materials for equality.

        Two materials are considered equal if they have the same id, name,
        fraction type, material libraries and nuclides with the same fractions
        and libraries.

        Parameters
        ----------
        mat1 : Material
            First material to compare.
        mat2 : Material
            Second material to compare.

        Returns
        -------
        bool
            True if materials are equal, False otherwise.
        """
        if mat1.id != mat2.id:
            return False
        if mat1.fraction_type != mat2.fraction_type:
            return False
        if mat1.libs != mat2.libs or mat1.name != mat2.name:
            return False
        if len(mat1.nuclide) != len(mat2.nuclide):
            return False

        # Compare nuclides by ZAID
        # Note: Nuclide.libs is a dict, so we need to compare the entire dict
        nuclides1 = {n.zaid: (n.fraction, frozenset(n.libs.items())) for n in mat1.nuclide.values()}
        nuclides2 = {n.zaid: (n.fraction, frozenset(n.libs.items())) for n in mat2.nuclide.values()}

        if set(nuclides1.keys()) != set(nuclides2.keys()):
            return False

        for zaid, (frac1, libs1) in nuclides1.items():
            frac2, libs2 = nuclides2[zaid]
            # Use relative tolerance for fraction comparison
            if abs(frac1 - frac2) > 1e-10 * max(abs(frac1), abs(frac2), 1e-10):
                return False
            if libs1 != libs2:
                return False

        return True

    def write_to_mcnp(self, input_filepath: str, output_filepath: Optional[str] = None,
                      material_ids: Optional[List[int]] = None,
                      force_rewrite: bool = False) -> List[int]:
        """Update materials in an MCNP input file, only modifying those that changed.

        Each material card is replaced where it stands, together with the
        ``KIKA_MAT_NAME`` comment and the KIKA banner that an earlier write put around
        it, so writing again replaces the same block. Every other line, materials
        not in this collection included, is copied byte for byte, and the file keeps
        its own line endings.

        Parameters
        ----------
        input_filepath : str
            Path to the original MCNP input file.
        output_filepath : str, optional
            Path for the output file. If None, updates the input file in place.
        material_ids : list of int, optional
            Specific material IDs to consider for update. If None, considers all
            materials in this collection.
        force_rewrite : bool, optional
            If True, rewrites every considered material that is in the file even when
            it has not changed. Default is False.

        Returns
        -------
        list of int
            List of material IDs that were actually updated (changed or added).

        Notes
        -----
        - A material is changed when its nuclides, fractions, fraction type, nuclide
          or material libraries, or name differ from the card in the file.
        - Materials in the collection but not in the file are added after the last
          material card; materials in the file but not in the collection are left
          unchanged.

        Examples
        --------
        >>> updated = collection.write_to_mcnp("input.i")
        >>> updated = collection.write_to_mcnp("input.i", "input_modified.i")
        >>> updated = collection.write_to_mcnp("input.i", force_rewrite=True)
        """
        from .parse_mcnp import read_material
        from .._constants import MCNPY_HEADER, MCNPY_FOOTER
        import re

        if output_filepath is None:
            output_filepath = input_filepath
        lines = _read_lines(input_filepath)
        eol = _line_ending(lines)
        header = MCNPY_HEADER.splitlines()
        footer = MCNPY_FOOTER.splitlines()

        if material_ids is None:
            ids_to_consider = set(self.by_id.keys())
        else:
            ids_to_consider = set(mid for mid in material_ids if mid in self.by_id)

        # (mat_id, block start, block end, material read from the file)
        blocks = []
        i = 0
        while i < len(lines):
            match = re.match(r"m(\d+)(?=\s|$)", lines[i].strip().lower())
            if not match:
                i += 1
                continue
            existing, end = read_material(lines, i)
            start = i
            if start > 0 and "kika_mat_name:" in lines[start - 1].lower():
                start -= 1
            if [l.strip() for l in lines[max(start - 3, 0):start]] == header:
                start -= 3
            if [l.strip() for l in lines[end:end + 3]] == footer:
                end += 3
            blocks.append((int(match.group(1)), start, end, existing))
            i = max(end, i + 1)

        def block_lines(material: Material) -> List[str]:
            text = MCNPY_HEADER + material.to_mcnp() + "\n" + MCNPY_FOOTER
            return [line + eol for line in text.splitlines()]

        materials_updated = []
        for mat_id, start, end, existing in reversed(blocks):
            if mat_id not in ids_to_consider:
                continue
            new_material = self.by_id[mat_id]
            if not force_rewrite and self._materials_equal(existing, new_material):
                continue
            lines[start:end] = block_lines(new_material)
            materials_updated.append(mat_id)

        found = {mat_id for mat_id, _, _, _ in blocks}
        new_ids = sorted(ids_to_consider - found)
        if new_ids:
            insert_at = len(lines)
            if blocks:
                # the replacements above moved the end of the last block
                insert_at = blocks[-1][2]
                shift = sum(
                    len(block_lines(self.by_id[mid])) - (e - s)
                    for mid, s, e, _ in blocks if mid in materials_updated
                )
                insert_at += shift
            if insert_at > 0 and not lines[insert_at - 1].endswith(("\n", "\r")):
                lines[insert_at - 1] += eol
            added = []
            for mat_id in new_ids:
                added.extend(block_lines(self.by_id[mat_id]))
                materials_updated.append(mat_id)
            lines[insert_at:insert_at] = added

        _write_lines(output_filepath, lines)
        return sorted(materials_updated)

    @classmethod
    def from_mcnp(cls, path: str) -> MaterialCollection:
        """Create a MaterialCollection from an MCNP input file.

        Parameters
        ----------
        path : str
            Path to MCNP input file.

        Returns
        -------
        MaterialCollection
            Collection of materials parsed from the file.
        """
        from ..mcnp.parse_input import read_mcnp

        input_obj = read_mcnp(path)
        return input_obj.materials

    @classmethod
    def from_serpent(cls, path: str) -> MaterialCollection:
        """Create a MaterialCollection from a Serpent input file.

        Parameters
        ----------
        path : str
            Path to Serpent input file.

        Returns
        -------
        MaterialCollection
            Collection of materials parsed from the file.
        """
        from .parse_serpent import read_serpent_materials

        return read_serpent_materials(path)

    @property
    def by_name(self) -> Dict[str, Material]:
        """Return materials indexed by name.

        Useful for Serpent materials which use string identifiers.

        Returns
        -------
        dict of str to Material
            Materials keyed by their name attribute.
        """
        return {mat.name: mat for mat in self.by_id.values() if mat.name}

    def write_to_serpent(self, input_filepath: str,
                         output_filepath: Optional[str] = None) -> None:
        """Update materials in a Serpent input file.

        Reads the original file, replaces material definitions with updated
        versions, and preserves all non-material content.

        Parameters
        ----------
        input_filepath : str
            Path to the original Serpent input file.
        output_filepath : str, optional
            Path for the output file. If None, updates the input file in place.
        """
        from .parse_serpent import _find_material_spans

        if output_filepath is None:
            output_filepath = input_filepath

        lines = _read_lines(input_filepath)
        eol = _line_ending(lines)

        spans = _find_material_spans(lines)
        name_to_mat = self.by_name

        # Process in reverse to maintain line positions
        for mat_name, start, end in reversed(spans):
            if mat_name in name_to_mat:
                text = name_to_mat[mat_name].to_serpent()
                lines[start:end] = [line + eol for line in text.splitlines()]

        _write_lines(output_filepath, lines)

    def __str__(self) -> str:
        """Return a user-friendly string representation of the collection.

        Returns
        -------
        str
            Formatted string showing materials summary and table.
        """
        header_width = 80
        lines = []
        lines.append("=" * header_width)
        lines.append(f"{'Materials Collection':^{header_width}}")
        lines.append("=" * header_width)

        materials_count = len(self.by_id)
        lines.append(f"\nTotal Materials: {materials_count}\n")

        if materials_count > 0:
            # Table header
            table_width = 80
            lines.append("-" * table_width)
            lines.append(f"{'ID':^12} | {'Nuclides':^10} | {'Type':^10} | {'Libraries':^38}")
            lines.append("-" * table_width)

            # Table rows
            for mat_id in sorted(self.by_id.keys()):
                material = self.by_id[mat_id]
                nuclide_count = len(material.nuclide)

                frac_type = "Atomic" if material.is_atomic else "Weight"

                # Gather all libraries (material-level and nuclide-level)
                libs_set = set()
                if material.libs:
                    for k, v in material.libs.items():
                        libs_set.add(v)

                for nuclide in material.nuclide.values():
                    if nuclide.libs:
                        for k, v in nuclide.libs.items():
                            libs_set.add(v)

                lib_str = ", ".join(sorted(libs_set)) if libs_set else "default"
                if len(lib_str) > 36:
                    lib_str = lib_str[:33] + "..."

                lines.append(f"{mat_id:^12} | {nuclide_count:^10} | {frac_type:^10} | {lib_str:^38}")

            lines.append("-" * table_width)

        # Available methods
        lines.append("\nAvailable methods:")
        lines.append("  .add_material(material)  - Add a material to the collection")
        lines.append("  .to_mcnp()               - Export all materials as MCNP cards")
        lines.append("  .to_serpent()            - Export all materials as Serpent cards")
        lines.append("  .by_id[material_id]      - Access a specific material by ID")

        # Examples
        lines.append("\nExamples of accessing data:")
        lines.append("  collection.by_id[1]      - Access material with ID 1")
        lines.append("  for mat in collection:   - Iterate over all materials")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the same formatted representation as __str__.

        Returns
        -------
        str
            Formatted collection representation.
        """
        return self.__str__()
