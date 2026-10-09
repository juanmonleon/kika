"""A small, hand-laid continuous-energy neutron ACE table for tape-free tests.

Every block is written word by word from the ACE manual (LA-UR-19-29016,
section 4), so a parser that disagrees with the manual disagrees with this
table. It carries:

* ESZ on three energies; MTR = (16, 51, 102), or (102,) with
  ``secondaries=False``, the shape of H-1 (NXS(5)=0);
* LQR, TYR (one TY per MTR reaction), LSIG/SIG;
* LAND/AND: tabulated elastic, LOCB=-1 for MT16, a tabulated MT51 that reuses
  the elastic's first table;
* LDLW/DLW: LAW=44 for MT16 and LAW=3 for MT51, each with IDAT > 1, which is
  what NJOY writes and what puts the LAW=44 tables away from IDAT.
"""
from __future__ import annotations

from pathlib import Path

ENERGIES = [1.0e-11, 1.0, 20.0]
AWR = 55.4544


def _xs_block(values):
    return [1, len(values)] + list(values)


def build_table(secondaries: bool = True):
    """Return ``(nxs, jxs, xss)``; ``xss`` is 1-based (``xss[0]`` unused)."""
    xss = [0.0]
    jxs = [0] * 33
    nxs = [0] * 17

    def here():
        return len(xss)

    mts = [16, 51, 102] if secondaries else [102]
    n_sec = 2 if secondaries else 0

    # ESZ: E, total, absorption, elastic, heating (Table 5)
    jxs[1] = here()
    xss += ENERGIES + [5.0, 4.0, 3.0] + [1.0, 0.5, 0.1] + [4.0, 3.5, 2.9] + [0.0, 0.0, 0.0]

    jxs[3] = here()
    xss += mts                                          # MTR (Table 10)
    jxs[4] = here()
    xss += [{16: -11.2, 51: -0.8467, 102: 7.6}[m] for m in mts]  # LQR (Table 11)
    jxs[5] = here()
    xss += [{16: -2, 51: -1, 102: 0}[m] for m in mts]   # TYR (Table 12)

    # LSIG/SIG: every reaction on the full grid (Tables 13-15)
    sig = [_xs_block([0.1 * (k + 1)] * 3) for k in range(len(mts))]
    jxs[6] = here()
    loca, pos = [], 1
    for block in sig:
        loca.append(pos)
        pos += len(block)
    xss += loca
    jxs[7] = here()
    for block in sig:
        xss += block

    # LAND/AND (Tables 16-20). Tabulated tables, JJ=2 (lin-lin, as NJOY writes)
    jxs[8] = here()
    and_words = []
    elastic = [2] + [ENERGIES[0], ENERGIES[2]] + [None, None]       # NE, E, LC
    table_a = [2, 3, -1.0, 0.0, 1.0, 0.25, 0.5, 0.75, 0.0, 0.375, 1.0]
    table_b = [2, 3, -1.0, 0.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.625, 1.0]
    loc_el = 1
    loc_a = loc_el + len(elastic)
    loc_b = loc_a + len(table_a)
    elastic[3], elastic[4] = -loc_a, -loc_b
    and_words += elastic + table_a + table_b
    land = [loc_el]
    if secondaries:
        loc_51 = len(and_words) + 1
        and_words += [1, 1.0, -loc_a]                    # MT51: one energy, reuses table_a
        land += [-1, loc_51]                             # MT16 in the DLW (LAW=44)
    xss += land
    jxs[9] = here()
    xss += and_words

    # LDLW/DLW (Tables 22-25, 36-37, eq. 2-3)
    jxs[10] = here()
    if secondaries:
        dlw = []
        # MT16, LAW=44. Header (9 words) then LDAT at IDAT=10
        dlw += [0, 44, 10, 0, 2, ENERGIES[0], ENERGIES[2], 1.0, 1.0]
        ldat = [0, 2, 11.3, 20.0, None, None]            # NR, NE, E, L
        l1 = 10 + len(ldat)
        t1 = [2, 2, 0.0, 0.1, 10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        l2 = l1 + len(t1)
        t2 = [2, 3, 0.0, 2.0, 8.0, 0.125, 0.125, 0.125, 0.0, 0.25, 1.0,
              0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
        ldat[4], ldat[5] = l1, l2
        dlw += ldat + t1 + t2
        # MT51, LAW=3
        locc_51 = len(dlw) + 1
        idat_51 = locc_51 + 9
        dlw += [0, 3, idat_51, 0, 2, ENERGIES[0], ENERGIES[2], 1.0, 1.0]
        dlw += [(AWR + 1) / AWR * 0.8467, (AWR / (AWR + 1)) ** 2]
        xss += [1, locc_51]
        jxs[11] = here()
        xss += dlw
    else:
        jxs[10] = 0

    jxs[22] = len(xss) - 1                               # END
    nxs[1] = len(xss) - 1
    nxs[2] = 26056
    nxs[3] = len(ENERGIES)
    nxs[4] = len(mts)
    nxs[5] = n_sec
    return nxs, jxs, xss


def write_table(path: Path, secondaries: bool = True, header: str = "legacy") -> Path:
    """Write the table as an ACE file, with a legacy or a 2.0.1 opening (Tables 1-2)."""
    nxs, jxs, xss = build_table(secondaries)
    lines = []
    if header == "2.0.1":
        lines.append(f"{'2.0.1':<10}{'26056.800nc':>24}{'ENDF/B-VIII.0':>24}")
        lines.append(f"{AWR:12.6f}{2.5301e-08:12.4E} {'2018-05-02':<10}{2:10d}")
    lines.append(f"{'26056.00c':>10}{AWR:12.6f}{2.5301e-08:12.4E} {'10/09/26':<10}")
    lines.append(f"{'synthetic table for kika tests':<70}{'mat2631':>10}")
    for _ in range(4):
        lines.append(f"{0:7d}{0.0:11.0f}" * 4)
    for k in range(1, 17, 8):
        lines.append("".join(f"{v:9d}" for v in nxs[k:k + 8]))
    for k in range(1, 33, 8):
        lines.append("".join(f"{v:9d}" for v in jxs[k:k + 8]))
    words = xss[1:]
    for k in range(0, len(words), 4):
        lines.append("".join(f"{float(v):20.11E}" for v in words[k:k + 4]))
    path.write_text("\n".join(lines) + "\n")
    return path
