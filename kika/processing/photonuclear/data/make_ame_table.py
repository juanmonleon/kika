"""Regenerate ``ame2020_masses.npz``: the atomic masses of the 2020 Atomic Mass Evaluation.

W. J. Huang, M. Wang, F. G. Kondev, G. Audi and S. Naimi, "The AME 2020 atomic
mass evaluation (I)", Chinese Physics C 45, 030002 (2021); M. Wang et al., (II),
ibid. 030003. The file read is the evaluation's ``mass_1.mas20`` as the AMDC
distributes it (``https://www-nds.iaea.org/amdc/``; OpenMC ships the same file).

Fixed-width lines (its header states ``a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,
f13.5,1x,f10.5,1x,a2,f13.5,f11.5,1x,i3,1x,f13.6,f12.6``): Z in columns 10-14, A in
15-19 and the atomic mass as an integer number of u (107-109) and micro-u
(111-123). A ``#`` marks a value estimated from systematics rather than measured;
it is kept, as a number, like every other.

The table written holds ``za`` (1000 Z + A) and ``mass_u`` (atomic mass, u).

Usage: ``python make_ame_table.py <path to mass_1.mas20>``.
"""
import sys
from pathlib import Path

import numpy as np


def read(path):
    za, mass = [], []
    started = False
    for line in Path(path).read_text().splitlines():
        if not started:
            # the data start after the column header, on the line of the neutron
            if len(line) > 110 and line[20:23].strip() == "n" and line[14:19].strip() == "1":
                started = True
            else:
                continue
        if len(line) < 123 or not line[9:19].strip():
            continue
        z, a = int(line[9:14]), int(line[14:19])
        whole = line[106:109].strip()
        micro = line[110:123].strip().replace("#", ".")
        if not whole or not micro:
            continue
        za.append(1000 * z + a)
        mass.append(int(whole) + float(micro) * 1.0e-6)
    return np.array(za, dtype=np.int64), np.array(mass, dtype=float)


if __name__ == "__main__":
    za, mass = read(sys.argv[1])
    out = Path(__file__).with_name("ame2020_masses.npz")
    np.savez_compressed(out, za=za, mass_u=mass)
    print(f"{za.size} nuclides -> {out}")
