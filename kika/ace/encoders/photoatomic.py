"""Encode a :class:`~kika.ace.classes.photoatomic.PhotoatomicAce` into an EPRDATA table.

The inverse of :func:`kika.ace.parsers.parse_photoatomic.parse_photoatomic` for
the EPRDATA layout (Hughes, LA-UR-13-27377, Tables 1-4; NXS(6) = 3 for the
``eprdata14``/``eprdata25`` generation): the blocks are laid out in JXS order,

    ESZG  ln E, ln σ_inc, ln σ_coh, ln σ_pe, ln σ_pair   (5·NES)
    JINC  x, S(x)                                         (2·NXS(13))
    JCOH  x, I(x), F(x)                                   (3·NXS(14))
    JFLO  edges, ejection probabilities, yields, lines    (4·NXS(4))
    LHNM  heating numbers, MeV                            (NES)
    LNEPS, LBEPS (MeV), LPIPS, LSWD, SWD                  (Compton, NXS(5) shells)
    SUBSH designators, electrons, binding (MeV), vacancy probabilities,
          numbers of transitions                          (5·NXS(7), JXS(11)-(15))
    SPHEL ln σ per subshell on ESZG's grid                (NXS(7)·NES)
    RELO, XPROB                                           (offsets; 4 words per transition)
    electron blocks, JXS(19) onwards

with energies in MeV, a zero cross section written as 0.0 (not as a log), LSWD
1-based from SWD and RELO 0-based from XPROB. The electron blocks are written
as the table carries them (:attr:`PhotoatomicAce.electron_words`).
"""
from __future__ import annotations

import numpy as np

from kika.ace.classes.photoatomic import EV_PER_MEV, PhotoatomicAce

#: NXS(6) of the EPRDATA generation kika writes (eprdata14, eprdata25).
EPRDATA_FORMAT = 3


def _logs(sigma) -> np.ndarray:
    """ln σ, with a zero cross section written as 0.0 (LA-UR-13-27377, Note 1)."""
    sigma = np.asarray(sigma, dtype=float)
    out = np.zeros_like(sigma)
    positive = sigma > 0.0
    if np.any(sigma < 0.0):
        raise ValueError("a photo-atomic cross section is negative")
    out[positive] = np.log(sigma[positive])
    return out


def encode_photoatomic(table: PhotoatomicAce, header=None) -> PhotoatomicAce:
    """A copy of ``table`` whose header and XSS are laid out from its decoded blocks.

    ``header`` (default: the table's own) gives the identification lines; NXS
    and JXS are recomputed. Raises :class:`ValueError` on blocks whose lengths
    do not agree.
    """
    from copy import copy

    nes = table.energies.size
    for name in ("incoherent", "coherent", "photoelectric", "pair_production", "heating"):
        if getattr(table, name).size != nes:
            raise ValueError(f"{name} has {getattr(table, name).size} values for {nes} energies")
    ns = len(table.subshell_designators)
    nsh = len(table.compton_profiles)
    nflo = table.fluorescence_edge_energies.size

    words, jxs, nxs = [], [0] * 33, [0] * 17

    def block(index, values):
        jxs[index] = len(words) + 1
        words.extend(np.asarray(values, dtype=float).ravel().tolist())

    block(1, np.concatenate([np.log(table.energies / EV_PER_MEV), _logs(table.incoherent), _logs(table.coherent),
                             _logs(table.photoelectric), _logs(table.pair_production)]))
    block(2, np.concatenate([table.incoherent_x, table.incoherent_scattering_function]))
    block(3, np.concatenate([table.coherent_x, table.coherent_integrated_form_factor, table.coherent_form_factor]))
    block(4, np.concatenate([table.fluorescence_edge_energies / EV_PER_MEV, table.fluorescence_ejection_probabilities,
                             table.fluorescence_yields, table.fluorescence_energies / EV_PER_MEV]))
    block(5, table.heating / EV_PER_MEV)
    if nsh:
        block(6, table.shell_electrons)
        block(7, table.shell_binding_energies / EV_PER_MEV)
        block(8, table.shell_probabilities)
        offsets, at = [], 1
        for p in table.compton_profiles:
            offsets.append(at)
            at += 2 + 3 * p.pz.size
        block(9, offsets)
        block(10, np.concatenate([np.concatenate([[p.interpolation, p.pz.size], p.pz, p.pdf, p.cdf])
                                  for p in table.compton_profiles]))
    if ns:
        block(11, table.subshell_designators)
        block(12, table.subshell_electrons)
        block(13, table.subshell_binding_energies / EV_PER_MEV)
        block(14, table.subshell_vacancy_probabilities)
        block(15, table.subshell_num_transitions)
        block(16, np.concatenate([_logs(table.subshell_photoelectric[int(d)]) for d in table.subshell_designators]))
        offsets, at, xprob = [], 0, []
        for d, nt in zip(table.subshell_designators, table.subshell_num_transitions):
            transitions = table.relaxation.get(int(d), [])
            if len(transitions) != int(nt):
                raise ValueError(f"subshell {d}: {len(transitions)} transitions, SUBSH says {nt}")
            offsets.append(at)
            at += 4 * len(transitions)
            for t in transitions:
                xprob += [t.designator_1, t.designator_2, t.energy / EV_PER_MEV, t.cumulative_probability]
        block(17, offsets)
        block(18, xprob)
    if table.electron_words.size:
        start = len(words) + 1
        words.extend(np.asarray(table.electron_words, dtype=float).tolist())
        for k, rel in zip(range(19, 33), table.electron_jxs):
            jxs[k] = start + rel if rel is not None else 0
        nxs[8:13] = list(table.electron_nxs)

    xss = np.concatenate(([0.0], np.asarray(words, dtype=float)))
    nxs[1], nxs[2], nxs[3], nxs[4], nxs[5] = xss.size - 1, int(table.z), nes, nflo, nsh
    nxs[6], nxs[7] = int(table.format_flag or EPRDATA_FORMAT), ns
    nxs[13], nxs[14] = table.incoherent_x.size, table.coherent_x.size
    out = copy(table)
    h = copy(header if header is not None else table.header)
    h.nxs_array, h.jxs_array = nxs, jxs
    out.header, out.xss_data = h, xss
    return out
