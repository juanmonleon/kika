"""Decode a photo-atomic ACE table into a :class:`PhotoatomicAce`.

The layout followed is the one written down in H. G. Hughes, *An
Electron/Photon/Relaxation Data Library for MCNP6*, LA-UR-13-27377 (2013),
Tables 1-4, which documents EPRDATA and, for the blocks the two share, the
older MCPLIB tables too. Every block is a view of positions the JXS array
names, so the decoding is a list of slices; what makes it worth a module is
the checks around them.

**It refuses rather than guesses.** A pointer past NXS(1), a JINC or JCOH
whose length does not split into its two or three arrays, or an energy grid
that is not increasing raises :class:`ValueError` naming the block. A
photo-atomic table read wrongly gives plausible-looking numbers, and no
downstream code would notice.

**The zero convention.** ESZG and SPHEL hold natural logarithms, and a cross
section that is truly zero (pair production below threshold) cannot be
logged. ACE's convention is that a stored 0.0 *is* a zero cross section, not
ln(1 b) (LA-UR-13-27377, Note 1). The decoder follows it.
"""
from __future__ import annotations

import numpy as np

from kika.ace.classes.photoatomic import (
    EV_PER_MEV,
    ComptonProfile,
    PhotoatomicAce,
    RelaxationTransitionAce,
)

#: MCPLIB's fixed abscissae for the incoherent scattering function (21) and
#: the coherent form factors (55), in 1/Å. The old tables do not write them:
#: MCNP has them in its source, and so must every reader.
MCPLIB_INCOHERENT_X = np.array([
    0.0, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0,
])
MCPLIB_COHERENT_X = np.array([
    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12,
    0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
    0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4,
    3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6,
    5.8, 6.0,
])

#: EPRDATA's electron blocks, which this decoder lists and does not read.
_ELECTRON_BLOCKS = {19: "ESZE", 20: "EXCIT", 21: "ELASI", 22: "ELAS",
                    23: "EION", 24: "BREMI", 25: "BREME", 26: "BREML"}


def is_photoatomic_extension(extension) -> bool:
    """True for a ZAID suffix such as ``.84p`` or ``.14p``.

    MCNP's class letter ``p`` is photo-atomic and nothing else: photonuclear
    is ``u``, protons ``h``, electrons ``e``.
    """
    return bool(extension) and str(extension).strip().lower().endswith("p")


def _block(xss, nxs1: int, start: int, n: int, name: str) -> np.ndarray:
    """``XSS(start) ... XSS(start+n-1)`` as a view, or a ValueError naming *name*."""
    if n < 0:
        raise ValueError(f"{name}: negative length {n}")
    if n == 0:
        return xss[1:1]
    if start < 1 or start + n - 1 > nxs1:
        raise ValueError(
            f"{name}: XSS({start}..{start + n - 1}) runs outside the table "
            f"(NXS(1) = {nxs1}); the pointer or the counter is wrong"
        )
    return xss[start:start + n]


def _from_logs(values: np.ndarray) -> np.ndarray:
    """exp() of every entry except the 0.0 that stands for a zero cross section."""
    out = np.zeros_like(values, dtype=float)
    nonzero = values != 0.0
    out[nonzero] = np.exp(values[nonzero])
    return out


def parse_photoatomic(header, xss: np.ndarray, filename=None) -> PhotoatomicAce:
    """Decode the XSS of a photo-atomic table whose header is already read."""
    nxs = list(header.nxs_array)
    jxs = list(header.jxs_array)
    if len(nxs) < 17 or len(jxs) < 33:
        raise ValueError("photo-atomic ACE: the NXS or JXS array is short")
    nxs1 = nxs[1]
    if xss is None or xss.size != nxs1 + 1:
        raise ValueError(f"photo-atomic ACE: XSS holds {0 if xss is None else xss.size - 1} values, NXS(1) = {nxs1}")

    table = PhotoatomicAce(filename=filename, header=header, xss_data=xss)
    table.z = int(nxs[2])
    table.format_flag = int(nxs[6])
    table.layout = "eprdata" if nxs[6] > 0 else "mcplib"
    nes, nflo, nsh = int(nxs[3]), int(nxs[4]), int(nxs[5])
    if nes <= 0:
        raise ValueError(f"photo-atomic ACE: NXS(3) = {nes}, no energy grid")

    # ESZG: ln E, ln σ_inc, ln σ_coh, ln σ_pe, ln σ_pair  (Table 4)
    eszg = _block(xss, nxs1, jxs[1], 5 * nes, "ESZG")
    ln_e = eszg[:nes]
    table.energies = np.exp(ln_e) * EV_PER_MEV
    if np.any(np.diff(table.energies) < 0.0):
        raise ValueError("ESZG: the photon energy grid is not increasing")
    table.incoherent = _from_logs(eszg[nes:2 * nes])
    table.coherent = _from_logs(eszg[2 * nes:3 * nes])
    table.photoelectric = _from_logs(eszg[3 * nes:4 * nes])
    table.pair_production = _from_logs(eszg[4 * nes:5 * nes])

    _read_form_factors(table, xss, nxs1, jxs)

    # JFLO: NF edges × (binding energy, ejection probability, yield, line energy)
    if nflo > 0:
        flo = _block(xss, nxs1, jxs[4], 4 * nflo, "JFLO")
        table.fluorescence_edge_energies = flo[:nflo] * EV_PER_MEV
        table.fluorescence_ejection_probabilities = flo[nflo:2 * nflo]
        table.fluorescence_yields = flo[2 * nflo:3 * nflo]
        table.fluorescence_energies = flo[3 * nflo:] * EV_PER_MEV

    # LHNM: heating numbers, MeV per collision, *not* logged
    table.heating = _block(xss, nxs1, jxs[5], nes, "LHNM") * EV_PER_MEV

    if nsh > 0:
        _read_compton(table, xss, nxs1, jxs, nsh)

    if table.layout == "eprdata":
        _read_subshells(table, xss, nxs1, nxs, jxs, nes)
        table.unparsed_blocks = [name for j, name in _ELECTRON_BLOCKS.items() if jxs[j] > 0]
        if jxs[19] > 0:
            start = int(jxs[19])
            table.electron_nxs = [int(v) for v in nxs[8:13]]
            table.electron_words = np.array(_block(xss, nxs1, start, nxs1 - start + 1, "electron blocks"))
            table.electron_jxs = [int(jxs[k]) - start if jxs[k] > 0 else None for k in range(19, 33)]
            # The EION header names each subshell block by its XSS index: kept
            # counted from the first electron word, so the blocks can move.
            nssh = int(nxs[7])
            if jxs[23] > 0 and nssh:
                at = int(jxs[23]) - start + nssh
                table.electron_words[at:at + 2 * nssh] -= start - 1
    return table


def _read_form_factors(table: PhotoatomicAce, xss, nxs1, jxs) -> None:
    """JINC and JCOH. MCPLIB: values only, on the fixed grids; EPRDATA: x first."""
    jinc, jcoh, jflo = jxs[2], jxs[3], jxs[4]
    if table.layout == "mcplib":
        n_inc, n_coh = MCPLIB_INCOHERENT_X.size, MCPLIB_COHERENT_X.size
        if jcoh - jinc != n_inc:
            raise ValueError(f"JINC: MCPLIB layout has {n_inc} values, JXS(3)-JXS(2) = {jcoh - jinc}")
        table.incoherent_x = MCPLIB_INCOHERENT_X.copy()
        table.incoherent_scattering_function = _block(xss, nxs1, jinc, n_inc, "JINC")
        coh = _block(xss, nxs1, jcoh, 2 * n_coh, "JCOH")
        table.coherent_x = MCPLIB_COHERENT_X.copy()
        table.coherent_integrated_form_factor = coh[:n_coh]
        table.coherent_form_factor = coh[n_coh:]
        return

    # EPRDATA: N_inc = (JXS(3)-JXS(2))/2 and N_coh = (JXS(4)-JXS(3))/3 (Table 2).
    span_inc, span_coh = jcoh - jinc, jflo - jcoh
    if span_inc <= 0 or span_inc % 2:
        raise ValueError(f"JINC: JXS(3)-JXS(2) = {span_inc} does not split into x and S(x)")
    if span_coh <= 0 or span_coh % 3:
        raise ValueError(f"JCOH: JXS(4)-JXS(3) = {span_coh} does not split into x, I(x) and F(x)")
    n_inc, n_coh = span_inc // 2, span_coh // 3
    inc = _block(xss, nxs1, jinc, 2 * n_inc, "JINC")
    table.incoherent_x = inc[:n_inc]
    table.incoherent_scattering_function = inc[n_inc:]
    coh = _block(xss, nxs1, jcoh, 3 * n_coh, "JCOH")
    table.coherent_x = coh[:n_coh]
    table.coherent_integrated_form_factor = coh[n_coh:2 * n_coh]
    table.coherent_form_factor = coh[2 * n_coh:]


def _read_compton(table: PhotoatomicAce, xss, nxs1, jxs, nsh: int) -> None:
    """LNEPS, LBEPS, LPIPS, LSWD and SWD: Compton Doppler broadening by shell."""
    table.shell_electrons = _block(xss, nxs1, jxs[6], nsh, "LNEPS")
    table.shell_binding_energies = _block(xss, nxs1, jxs[7], nsh, "LBEPS") * EV_PER_MEV
    table.shell_probabilities = _block(xss, nxs1, jxs[8], nsh, "LPIPS")
    offsets = _block(xss, nxs1, jxs[9], nsh, "LSWD")
    for k in range(nsh):
        start = jxs[10] + int(round(offsets[k])) - 1
        jj, n = _block(xss, nxs1, start, 2, f"SWD shell {k + 1}")
        n = int(round(n))
        body = _block(xss, nxs1, start + 2, 3 * n, f"SWD shell {k + 1}")
        table.compton_profiles.append(ComptonProfile(
            interpolation=int(round(jj)), pz=body[:n], pdf=body[n:2 * n], cdf=body[2 * n:]))


def _read_subshells(table: PhotoatomicAce, xss, nxs1, nxs, jxs, nes: int) -> None:
    """SUBSH, SPHEL, RELO and XPROB: EPRDATA's per-subshell photon data."""
    ns = int(nxs[7])
    if ns <= 0 or jxs[11] <= 0:
        return
    subsh = _block(xss, nxs1, jxs[11], 5 * ns, "SUBSH")
    designators = np.rint(subsh[:ns]).astype(int)
    table.subshell_designators = designators
    table.subshell_electrons = subsh[ns:2 * ns]
    table.subshell_binding_energies = subsh[2 * ns:3 * ns] * EV_PER_MEV
    table.subshell_vacancy_probabilities = subsh[3 * ns:4 * ns]
    table.subshell_num_transitions = np.rint(subsh[4 * ns:]).astype(int)

    if jxs[16] > 0:
        sphel = _block(xss, nxs1, jxs[16], ns * nes, "SPHEL")
        for k, designator in enumerate(designators):
            table.subshell_photoelectric[int(designator)] = _from_logs(sphel[k * nes:(k + 1) * nes])

    if jxs[17] > 0 and jxs[18] > 0:
        offsets = _block(xss, nxs1, jxs[17], ns, "RELO")
        for k, designator in enumerate(designators):
            nt = int(table.subshell_num_transitions[k])
            words = _block(xss, nxs1, jxs[18] + int(round(offsets[k])), 4 * nt,
                           f"XPROB subshell {designator}")
            table.relaxation[int(designator)] = [
                RelaxationTransitionAce(int(round(words[4 * t])), int(round(words[4 * t + 1])),
                                        float(words[4 * t + 2]) * EV_PER_MEV, float(words[4 * t + 3]))
                for t in range(nt)
            ]
