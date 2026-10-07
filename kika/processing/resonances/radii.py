"""Validated immutable radius functions; no silent interpolation fallback."""
from dataclasses import dataclass
import numpy as np
from kika.algebra import evaluate, interval_laws


@dataclass(frozen=True)
class RadiusFunction:
    constant: float | None = None
    energies: tuple[float, ...] = ()
    values: tuple[float, ...] = ()
    interpolation: tuple[tuple[int, int], ...] = ()

    def evaluate(self, energy):
        x = np.asarray(energy, dtype=float)
        if self.constant is not None:
            return np.full_like(x, self.constant)
        grid = np.asarray(self.energies)
        if np.any(x < grid[0]) or np.any(x > grid[-1]):
            raise ValueError("radius table does not cover evaluation/reference energies")
        return np.asarray(evaluate(grid, self.values, self._laws(), x))

    def _laws(self):
        """One law per interval; NBT is the 1-based last point of a region."""
        return interval_laws(len(self.energies), self.interpolation)


    def difference(self, reference, energy):
        """R(reference)-R(E), using local divided differences near reference.

        log1p/expm1 preserve changes smaller than the radius' absolute ULP.
        Histograms keep their declared sided values; no smoothing is applied.
        """
        x = np.asarray(energy, dtype=float)
        values = self.evaluate(x)
        result = np.asarray(self.evaluate(reference)-values).copy()
        if self.constant is not None:
            return result
        grid, radii = np.asarray(self.energies), np.asarray(self.values)
        i = np.clip(np.searchsorted(grid, x, side="right")-1, 0, len(grid)-2)
        laws = self._laws()[i]
        near = (abs(reference-x) < .25*x) & (reference >= grid[i]) & (reference <= grid[i+1])
        for law in (2, 3, 4, 5):
            mask = near & (laws == law)
            if not np.any(mask):
                continue
            a, b = grid[i[mask]], grid[i[mask]+1]
            ra, rb = radii[i[mask]], radii[i[mask]+1]
            delta = reference-x[mask]
            if law == 2:
                result[mask] = delta*(rb-ra)/(b-a)
            elif law == 3:
                result[mask] = np.log1p(delta/x[mask])*(rb-ra)/np.log(b/a)
            elif law == 4:
                result[mask] = values[mask]*np.expm1(delta*np.log(rb/ra)/(b-a))
            else:
                result[mask] = values[mask]*np.expm1(np.log1p(delta/x[mask])*np.log(rb/ra)/np.log(b/a))
        return result


def prepare_radius(radius):
    if isinstance(radius, (int, float, np.number)):
        value = float(radius)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("radius must be finite and positive [fm]")
        return RadiusFunction(constant=value)
    if radius is None or radius.unit not in (None, "fm"):
        raise ValueError("missing or unnormalized radius [fm]")
    if not radius.isEnergyDependent:
        return prepare_radius(radius.constant)
    x, y = np.asarray(radius.energies, dtype=float), np.asarray(radius.values, dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or len(x) < 2 or x.shape != y.shape
            or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))
            or np.any(np.diff(x) <= 0) or np.any(x < 0) or np.any(y <= 0)):
        raise ValueError("invalid radius table")
    pairs = tuple(tuple(p) for p in (radius.interpolation or ()))
    if not pairs or pairs[-1][0] != len(x):
        raise ValueError("radius interpolation must cover every point")
    previous = 1
    for nbt, law in pairs:
        if not isinstance(nbt, (int, np.integer)) or not previous < nbt <= len(x) or law not in (1,2,3,4,5):
            raise ValueError("invalid radius interpolation region/law")
        if law in (3, 5) and np.any(x[previous-1:nbt] <= 0):
            raise ValueError("log-energy radius interpolation needs positive energies")
        previous = nbt
    return RadiusFunction(energies=tuple(x), values=tuple(y), interpolation=pairs)
