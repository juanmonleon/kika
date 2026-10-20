"""Regenerate ``bmm_compton_profiles.npz``: the Biggs-Mendelsohn-Mann Compton profiles.

F. Biggs, L. B. Mendelsohn and J. B. Mann, "Hartree-Fock Compton profiles for the
elements", Atomic Data and Nuclear Data Tables 16, 201-309 (1975): J(p_z) per
shell on a common grid of 31 projected momenta (atomic units), unnormalised, as
the paper prints them (three significant digits).

The transcription read here is NJOY3's (dryad, ``src/njoy/dryad/external/
ComptonProfiles/src/biggs_mendelsohn_mann.hpp``, BSD licence), which is how
eprdata25 got its profiles. One entry of it is a C++ expression and not a
number: Eu (Z = 63), shell 2p1/2, p_z = 30 a.u. reads ``4.61-03``, which C++
evaluates as 4.61 - 3 = 1.61. The paper's value is 4.61E-03, and that is the one
written here: a Fortran-style exponent without its E.

Usage: ``python make_bmm_table.py <path to biggs_mendelsohn_mann.hpp>``.
"""
import re
import sys
from pathlib import Path

import numpy as np


def _number(token: str) -> float:
    token = token.strip()
    fortran = re.fullmatch(r"([0-9.]+)([-+]\d+)", token)
    if fortran:                       # "4.61-03": the paper's 4.61E-03
        return float(f"{fortran.group(1)}E{fortran.group(2)}")
    return float(token)


def load(source: Path):
    text = source.read_text()
    pz = [float(v) for v in re.search(r"momentum = \{(.*?)\};", text, re.S).group(1).split(",") if v.strip()]
    shells_text = text[text.index("biggs_mendelsohn_mann_shells"):text.index("biggs_mendelsohn_mann_data")]
    shells = {int(m.group(1)): re.findall(r'"([^"]+)"', m.group(2))
              for m in re.finditer(r"\{\s*(\d+),\s*\{(.*?)\}\s*\}\s*(?:,|\n\})", shells_text, re.S)}
    data_text = text[text.index("biggs_mendelsohn_mann_data"):]
    starts = [(m.start(), int(m.group(1))) for m in re.finditer(r"\n  \{\s*(\d+),\s*\{", data_text)]
    data = {}
    for k, (at, z) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(data_text)
        data[z] = [[_number(v) for v in block.replace("\n", " ").split(",") if v.strip()]
                   for block in re.findall(r"\{([^{}]*)\}", data_text[at:end])]
    return np.asarray(pz), shells, data


def main(source):
    pz, shells, data = load(Path(source))
    arrays = {"pz": pz}
    for z in sorted(shells):
        profiles = np.asarray(data[z], dtype=float)
        if profiles.shape != (len(shells[z]), pz.size):
            raise ValueError(f"Z={z}: {len(shells[z])} shells but profiles of shape {profiles.shape}")
        arrays[f"labels_{z}"] = np.asarray(shells[z])
        arrays[f"J_{z}"] = profiles
    out = Path(__file__).with_name("bmm_compton_profiles.npz")
    np.savez_compressed(out, **arrays)
    print(out, len(shells), "elements")


if __name__ == "__main__":
    main(sys.argv[1])
