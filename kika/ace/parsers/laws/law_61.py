import numpy as np

from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.angle_energy import TabulatedAngleEnergyDistribution
from kika.ace.classes.xss import xss_ints


class Law61ParseError(ValueError):
    """Exception raised for errors in parsing Law 61 data."""
    pass


def parse_tabulated_angle_energy_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int,
                                              debug: bool = False) -> TabulatedAngleEnergyDistribution:
    """
    Parse a tabulated angle-energy distribution (Law 61).

    Table 45: LAW=61
    - LDAT(1): N_R (number of interpolation regions)
    - LDAT(2..1+N_R): NBT(l) interpolation parameters
    - LDAT(2+N_R..1+2*N_R): INT(l) interpolation schemes
    - LDAT(2+2*N_R): N_E (number of incident energies)
    - LDAT(3+2*N_R..2+2*N_R+N_E): E(l) (incident energies)
    - LDAT(3+2*N_R+N_E..2+2*N_R+2*N_E): L(l) (distribution locations)

    Table 46: secondary energy distribution at each incident energy, at JED + L - 1
    - INTT' = 10*N_D + INTT, N_p, E_out(1:N_p), PDF(1:N_p), CDF(1:N_p), LC(1:N_p)

    Table 47: angular distribution at JED + |LC| - 1 (LC = 0: isotropic, no table)
    - JJ, N_p, Cos_out(1:N_p), PDF(1:N_p), CDF(1:N_p)

    The arrays are views of the XSS. Outgoing-energy points of one or several
    incident energies often share an angular table: each LC is read once,
    and ``angular_tables`` lists them in the order they first appear.

    Raises
    ------
    Law61ParseError
        If a locator or a table runs past the XSS array.
    """
    xss = ace.xss_data
    size = len(xss)
    distribution = TabulatedAngleEnergyDistribution(
        law=base_dist.law,
        idat=base_dist.idat,
        applicability_energies=base_dist.applicability_energies,
        applicability_probabilities=base_dist.applicability_probabilities,
        applicability_nbt=base_dist.applicability_nbt,
        applicability_interp=base_dist.applicability_interp,
    )
    jed = getattr(base_dist, "jed", None)
    if jed is None:
        raise Law61ParseError("Missing JED value required for parsing Law 61 distribution")
    if idat_idx >= size:
        raise Law61ParseError(f"Index {idat_idx} out of bounds for XSS data with length {size}")

    try:
        n_r = int(xss[idat_idx])
        idx = idat_idx + 1
        distribution.n_interp_regions = n_r
        if n_r > 0:
            distribution.nbt = xss_ints(xss, idx, n_r)
            distribution.interp = xss_ints(xss, idx + n_r, n_r)
            idx += 2 * n_r
        n_e = int(xss[idx])
        idx += 1
        if n_e <= 0:
            raise Law61ParseError(f"Invalid N_E value: {n_e} (must be positive)")
        if idx + 2 * n_e > size:
            raise Law61ParseError(f"N_E={n_e} incident energies and locators at {idx} run past the XSS array ({size})")
        distribution.n_energies = n_e
        distribution.incident_energies = xss[idx:idx + n_e]
        distribution.distribution_locations = xss_ints(xss, idx + n_e, n_e)
    except IndexError as e:
        raise Law61ParseError(f"LAW=61 header at {idat_idx}: {e}")

    distribution.distributions = []
    all_lcs = []
    for i, loc in enumerate(distribution.distribution_locations):
        if loc <= 0:
            raise Law61ParseError(f"distribution {i + 1}: invalid location (L={loc})")
        k = jed + loc - 1
        if k + 1 >= size:
            raise Law61ParseError(f"Distribution index {k} (JED={jed} + L={loc}) out of bounds for XSS data with length {size}")
        intt_prime = int(xss[k])
        n_p = int(xss[k + 1])
        a = k + 2
        if a + 4 * n_p > size:
            raise Law61ParseError(f"energy distribution {i + 1} of N_p={n_p} at {k} runs past the XSS array ({size})")
        lcs = np.rint(xss[a + 3 * n_p:a + 4 * n_p]).astype(np.int64)
        all_lcs.append(lcs)
        distribution.distributions.append({
            'intt': intt_prime % 10,
            'n_discrete': intt_prime // 10,
            'n_points': n_p,
            'e_out': xss[a:a + n_p],
            'pdf': xss[a + n_p:a + 2 * n_p],
            'cdf': xss[a + 2 * n_p:a + 3 * n_p],
            'lc': lcs.tolist(),
        })
    distribution.angular_tables = _angular_tables(xss, jed, all_lcs)
    return distribution


def _angular_tables(xss, jed: int, all_lcs) -> list:
    """The Table 47 angular distributions the LCs point to, each read once.

    Listed in the order their LC first appears; LC = 0 (isotropic) has no table.
    """
    lcs = np.concatenate(all_lcs) if all_lcs else np.empty(0, dtype=np.int64)
    lcs = lcs[lcs != 0]
    if lcs.size == 0:
        return []
    _, first = np.unique(lcs, return_index=True)
    lcs = lcs[np.sort(first)]
    size = len(xss)
    ks = jed + np.abs(lcs) - 1
    if ks.max() + 1 >= size:
        bad = int(lcs[np.argmax(ks + 1 >= size)])
        raise Law61ParseError(
            f"Angular distribution of LC={bad} (JED={jed} + |LC| - 1) "
            f"out of bounds for XSS data with length {size}")
    jjs = np.rint(xss[ks]).astype(np.int64)
    nps = np.rint(xss[ks + 1]).astype(np.int64)
    ends = ks + 2 + 3 * nps
    if ends.max() > size:
        bad = int(np.argmax(ends > size))
        raise Law61ParseError(
            f"angular distribution LC={int(lcs[bad])} of N_p={int(nps[bad])} at {int(ks[bad])} "
            f"runs past the XSS array ({size})")
    return [
        {
            'lc': lc,  # the locator the 'lc' lists of the distributions refer to
            'jj': jj,
            'n_points': n,
            'cosines': xss[a:a + n],
            'pdf': xss[a + n:a + 2 * n],
            'cdf': xss[a + 2 * n:a + 3 * n],
        }
        for lc, jj, n, a in zip(lcs.tolist(), jjs.tolist(), nps.tolist(), (ks + 2).tolist())
    ]
