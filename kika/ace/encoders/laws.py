"""Encoders of the ACE energy laws: the inverse of ``kika.ace.parsers.laws``.

Each ``encode_*`` appends to a :class:`~kika.ace.encoders._words.Words` the
words its parser reads (LA-UR-19-29016, Tables 30-52), in the order NJOY
writes them. Every locator inside a law (IDAT, LNW, the L(l) of laws 4, 22,
44, 61 and 67, the LC of law 61, the LMU of law 67) is relative to JED, the
first word of the DLW-like block (DLW, DLWP, DLWH, DNED), so it is a
``Words`` position: it is computed from where the data is actually put,
never copied from the locators the parser kept.
"""
from __future__ import annotations

import numpy as np

from kika.ace.encoders._words import Words


def _name(law) -> str:
    return law if isinstance(law, str) else f"LAW={law}"


def _interp_regions(w: Words, nbt, interp, law) -> None:
    """NR, NBT(NR), INT(NR)."""
    nbt = [] if nbt is None else list(nbt)
    interp = [] if interp is None else list(interp)
    if len(nbt) != len(interp):
        raise ValueError(f"{_name(law)}: {len(nbt)} NBT but {len(interp)} INT")
    w.int(len(nbt))
    w.ints(nbt)
    w.ints(interp)


def _same_length(law, what, *arrays) -> int:
    n = {len(a) for a in arrays}
    if len(n) != 1:
        raise ValueError(f"{_name(law)}: {what} have lengths {[len(a) for a in arrays]}")
    return n.pop()


def _table_header(w: Words, dist, law) -> int:
    """NR, NBT, INT, NE, E(NE), then NE reserved locators L(l); returns the index of L(1)."""
    _interp_regions(w, dist.nbt, dist.interp, law)
    energies = np.asarray(dist.incident_energies, dtype=float)
    w.int(len(energies))
    w.reals(energies)
    return w.reserve(len(energies))


def _energy_table(w: Words, d: dict, law, extra=()) -> None:
    """INTT' = 10 N_D + INTT, N_p, E_out, PDF, CDF and the ``extra`` arrays (Tables 36, 44, 46)."""
    arrays = [d['e_out'], d['pdf'], d['cdf']] + [d[k] for k in extra]
    n_p = _same_length(law, "E_out/PDF/CDF" + "".join(f"/{k}" for k in extra), *arrays)
    w.int(10 * int(d.get('n_discrete', 0)) + int(d['intt']))
    w.int(n_p)
    for a in arrays:
        w.reals(a)


def _law1(dist, w):
    # Table 32: NR, NBT, INT, NE, E, NET, E_out tables of NET boundaries
    _interp_regions(w, dist.nbt, dist.interp, 1)
    tables = [np.asarray(d['e_out'], dtype=float) for d in dist.distribution_data]
    energies = np.asarray(dist.incident_energies, dtype=float)
    if len(tables) != len(energies):
        raise ValueError(f"LAW=1: {len(energies)} incident energies but {len(tables)} E_out tables")
    net = {len(t) for t in tables}
    if len(net) > 1:
        raise ValueError(f"LAW=1: the E_out tables have different lengths {sorted(net)}")
    w.int(len(energies))
    w.reals(energies)
    w.int(net.pop() if net else 0)
    for t in tables:
        w.reals(t)


def _law2(dist, w):
    # Table 33: LP, EG
    w.int(dist.lp)
    w.real(dist.eg)


def _law3(dist, w):
    # Table 34: (A+1)/A |Q|, (A/(A+1))^2
    w.real(dist.aplusoaabsq, dist.asquare)


def _law33(dist, w):
    # Ec, C (NJOY acelcp)
    w.real(dist.threshold, dist.energy_factor)


def _law4(dist, w, extra=()):
    # Tables 35-36 (law 4) and 43-44 (law 44, extra = R, A): header, then the tables in order
    law = dist.law
    if len(dist.distributions) != len(dist.incident_energies):
        raise ValueError(f"LAW={law}: {len(dist.incident_energies)} incident energies "
                         f"but {len(dist.distributions)} tables")
    loc = _table_header(w, dist, law)
    padding = list(getattr(dist, "padding", None) or [])
    for i, d in enumerate(dist.distributions):
        w.set(loc + i, w.next)
        _energy_table(w, d, law, extra)
        if i < len(padding) and padding[i]:
            w.int(*([0] * padding[i]))  # NJOY's unused room after the table


def _law44(dist, w):
    _law4(dist, w, extra=('r', 'a'))


def _temperature_table(w, dist, law):
    """NR, NBT, INT, NE, E(NE), theta(NE) (Tables 37-39)."""
    _interp_regions(w, dist.temp_nbt, dist.temp_interp, law)
    n = _same_length(law, "E/theta", dist.incident_energies, dist.temperatures)
    w.int(n)
    w.reals(dist.incident_energies)
    w.reals(dist.temperatures)


def _law5(dist, w):
    # Table 37: temperature table, NET, X(NET)
    _temperature_table(w, dist, 5)
    w.int(len(dist.equiprob_values))
    w.reals(dist.equiprob_values)


def _law7(dist, w):
    # Tables 38-39: temperature table, U
    _temperature_table(w, dist, dist.law)
    w.real(dist.restriction_energy)


def _law11(dist, w):
    # Table 40: a(E) table, b(E) table, U
    for p in ('a', 'b'):
        _interp_regions(w, getattr(dist, f"{p}_nbt"), getattr(dist, f"{p}_interp"), 11)
        e, v = getattr(dist, f"{p}_incident_energies"), getattr(dist, f"{p}_values")
        w.int(_same_length(11, f"E_{p}/{p}", e, v))
        w.reals(e)
        w.reals(v)
    w.real(dist.restriction_energy)


def _law22(dist, w):
    # Table 41: header with LOCE(l), then NF, P(NF), T(NF), C(NF) per incident energy
    if len(dist.function_data) != len(dist.incident_energies):
        raise ValueError(f"LAW=22: {len(dist.incident_energies)} incident energies "
                         f"but {len(dist.function_data)} function sets")
    loc = _table_header(w, dist, 22)
    for i, f in enumerate(dist.function_data):
        w.set(loc + i, w.next)
        w.int(_same_length(22, "P/T/C", f['p'], f['t'], f['c']))
        w.reals(f['p'])
        w.reals(f['t'])
        w.reals(f['c'])


def _law24(dist, w):
    # Table 42: NR, NBT, INT, NE, E, NET, NE tables of NET multipliers
    tables = [np.asarray(t, dtype=float) for t in dist.multiplier_tables]
    if len(tables) != len(dist.incident_energies):
        raise ValueError(f"LAW=24: {len(dist.incident_energies)} incident energies "
                         f"but {len(tables)} multiplier tables")
    net = {len(t) for t in tables}
    if len(net) > 1:
        raise ValueError(f"LAW=24: the multiplier tables have different lengths {sorted(net)}")
    _interp_regions(w, dist.nbt, dist.interp, 24)
    w.int(len(tables))
    w.reals(dist.incident_energies)
    w.int(net.pop() if net else 0)
    for t in tables:
        w.reals(t)


def _law61(dist, w):
    """Tables 45-47. NJOY writes each E' table (with its LC column) followed
    by the angular tables first referenced from it, in order of first use; a
    table already written is pointed to again. LC keeps the sign it had
    (NJOY writes tabulated tables as negative LC); LC = 0 is isotropic.
    """
    if len(dist.distributions) != len(dist.incident_energies):
        raise ValueError(f"LAW=61: {len(dist.incident_energies)} incident energies "
                         f"but {len(dist.distributions)} tables")
    tables = {}
    for t in dist.angular_tables:
        if t['lc'] in tables:
            raise ValueError(f"LAW=61: two angular tables share LC={t['lc']}")
        tables[t['lc']] = t
    written = {}  # stored LC -> new position
    loc = _table_header(w, dist, 61)
    for i, d in enumerate(dist.distributions):
        w.set(loc + i, w.next)
        lcs = [int(v) for v in d['lc']]
        _energy_table(w, d, 61)
        if len(lcs) != len(d['e_out']):
            raise ValueError(f"LAW=61: table {i + 1} has {len(d['e_out'])} points but {len(lcs)} LC")
        lc_at = w.reserve(len(lcs))
        for k, lc in enumerate(lcs):
            if lc == 0:
                continue
            if lc not in written:
                t = tables.get(lc)
                if t is None:
                    raise ValueError(f"LAW=61: LC={lc} of table {i + 1} names no angular table")
                written[lc] = w.next
                w.int(t['jj'])
                w.int(_same_length(61, "cosine/PDF/CDF", t['cosines'], t['pdf'], t['cdf']))
                w.reals(t['cosines'])
                w.reals(t['pdf'])
                w.reals(t['cdf'])
            w.set(lc_at + k, -written[lc] if lc < 0 else written[lc])
    unused = set(tables) - set(written)
    if unused:
        raise ValueError(f"LAW=61: angular tables {sorted(unused)} are referenced by no LC")


def _law66(dist, w):
    # Table 48: NPSX, Ap, and (NJOY 2016 on) INTT, N_P, xi, PDF, CDF
    w.int(dist.npsx)
    w.real(dist.ap)
    if len(dist.xi_grid) == 0:
        return
    w.int(dist.intt)
    w.int(_same_length(66, "xi/PDF/CDF", dist.xi_grid, dist.pdf, dist.cdf))
    w.reals(dist.xi_grid)
    w.reals(dist.pdf)
    w.reals(dist.cdf)


def _law67(dist, w):
    """Tables 49-51: header with L(l); per incident energy INTMU, NMU,
    XMU(NMU), LMU(NMU), followed by the NMU energy tables INTEP, NPEP,
    E_p, PDF, CDF in the order of the cosines.
    """
    aes = dist.angle_energy_distributions
    if len(aes) != len(dist.incident_energies):
        raise ValueError(f"LAW=67: {len(dist.incident_energies)} incident energies "
                         f"but {len(aes)} angle tables")
    loc = _table_header(w, dist, 67)
    for i, a in enumerate(aes):
        w.set(loc + i, w.next)
        eds = a['energy_distributions']
        n = _same_length(67, "XMU/energy tables", a['cosines'], eds)
        w.int(a['intmu'])
        w.int(n)
        w.reals(a['cosines'])
        lmu = w.reserve(n)
        for j, e in enumerate(eds):
            if e is None:
                raise ValueError(f"LAW=67: incident energy {i + 1}, cosine {j + 1} has no energy table")
            w.set(lmu + j, w.next)
            w.int(e['intep'])
            w.int(_same_length(67, "E_p/PDF/CDF", e['e_out'], e['pdf'], e['cdf']))
            w.reals(e['e_out'])
            w.reals(e['pdf'])
            w.reals(e['cdf'])


_ENCODERS = {
    1: _law1, 2: _law2, 3: _law3, 4: _law4, 5: _law5, 7: _law7, 9: _law7,
    11: _law11, 22: _law22, 24: _law24, 33: _law33, 44: _law44, 61: _law61,
    66: _law66, 67: _law67,
}


def encode_law_data(dist, w: Words) -> None:
    """Append the LDAT of one energy law (Tables 32-51) to ``w``."""
    law = int(getattr(dist, "law", 0))
    encoder = _ENCODERS.get(law)
    if encoder is None:
        raise ValueError(f"LAW={law} is not an ACE energy law kika can encode")
    encoder(dist, w)


def encode_law_chain(laws: list, w: Words) -> None:
    """Append the law chain of one reaction (Table 30) to ``w``.

    Each law is its record LNW, LAW, IDAT, NR, NBT, INT, NE, E(NE), P(NE)
    followed by its LDAT, as NJOY writes it: IDAT points right after the
    record and LNW to the next law's record (0 for the last).
    """
    if not laws:
        raise ValueError("an energy-law chain needs at least one law")
    for k, dist in enumerate(laws):
        lnw = w.reserve()
        w.int(dist.law)
        idat = w.reserve()
        energies = np.asarray(dist.applicability_energies, dtype=float)
        probabilities = np.asarray(dist.applicability_probabilities, dtype=float)
        _interp_regions(w, dist.applicability_nbt, dist.applicability_interp, dist.law)
        w.int(_same_length(dist.law, "applicability E/P", energies, probabilities))
        w.reals(energies)
        w.reals(probabilities)
        w.set(idat, w.next)
        encode_law_data(dist, w)
        if k < len(laws) - 1:
            w.set(lnw, w.next)
    gap = int(getattr(laws[-1], "padding_after", 0) or 0)
    if gap:
        w.int(*([0] * gap))  # NJOY's unused room before the next record


def encode_energy_dependent_yield(y, w: Words) -> None:
    """Append a TY > 100 yield record (Table 52): NR, NBT, INT, NE, E(NE), Y(NE)."""
    _interp_regions(w, y.nbt, y.interp, "TY>100 yield")
    w.int(_same_length("TY>100 yield", "E/Y", y.energies, y.yields))
    w.reals(y.energies)
    w.reals(y.yields)
