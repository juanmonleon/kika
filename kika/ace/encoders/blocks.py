"""The neutron-table blocks of an ``Ace``, one encoder per parser.

Each ``encode_X`` is the inverse of ``parsers/parse_X``: it reads the objects
the parser builds and writes the block's words (manual LA-UR-19-29016,
section 4), with locators relative to the block (see ``_words``). Values are
copied bit for bit; counts, MTs and locators are recomputed from what is
written, never copied from the object, so a changed object gives a
consistent block. The layout (``layout.py``) places the blocks.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from kika.ace.classes.ixs_keys import ixs_keys
from kika.ace.encoders._words import Words
from kika.ace.encoders.laws import encode_energy_dependent_yield, encode_law_chain


# --- ESZ, NU, MTR, LQR, TYR (Tables 5-12) -------------------------------------

def encode_esz(ace) -> Words:
    esz = ace.esz_block
    w = Words()
    for arr in (esz.energies, esz.total_xs, esz.absorption_xs, esz.elastic_xs, esz.heating_numbers):
        w.reals(arr)
    return w


def _encode_nu_record(nu, w: Words) -> None:
    """One ν record (Table 6): LNU = 1 polynomial or 2 tabulated."""
    if nu.format == "polynomial":
        c = nu.polynomial.coefficients
        w.int(1, len(c))
        w.reals(c)
    elif nu.format == "tabulated":
        t = nu.tabulated
        w.int(2, len(t.interpolation_regions))
        w.ints([nbt for nbt, _ in t.interpolation_regions])
        w.ints([i for _, i in t.interpolation_regions])
        w.int(len(t.energies))
        w.reals(t.energies)
        w.reals(t.nubar_values)
    else:
        raise ValueError(f"nu record of unknown format {nu.format!r}")


def encode_nu(ace) -> Optional[Words]:
    """NU (Table 6): total ν alone, or KNU < 0 then prompt and total."""
    nu = ace.nubar
    if nu is None or not nu.has_nubar or nu.total is None:
        return None
    w = Words()
    if nu.has_both_nu_types:
        prompt = Words()
        _encode_nu_record(nu.prompt, prompt)
        w.int(-len(prompt))
        w.extend(prompt)
    _encode_nu_record(nu.total, w)
    return w


def encode_dnu(ace) -> Optional[Words]:
    nu = ace.nubar
    if nu is None or not nu.has_delayed or nu.delayed is None:
        return None
    w = Words()
    _encode_nu_record(nu.delayed, w)
    return w


def _ints_block(values) -> Words:
    w = Words()
    w.ints(values)
    return w


def _reals_block(values) -> Words:
    w = Words()
    w.reals(values)
    return w


# --- LSIG/SIG (Tables 13-15), FIS ----------------------------------------------

def encode_sig(ace) -> Tuple[Words, Words, Optional[int]]:
    """LSIG and SIG in MTR order; also the SIG position of MT=18 (for JXS(21))."""
    lsig, sig = Words(), Words()
    fis_at = None
    for mt in ace.reaction_mt_data.incident_neutron:
        mt = int(mt)
        r = ace.cross_section.reaction[mt]
        lsig.int(sig.next)
        if mt == 18:
            fis_at = sig.next
        xs = r._xs_entries
        sig.int(r.energy_idx + 1, len(xs))
        sig.reals(xs)
    return lsig, sig, fis_at


# --- LAND/AND, LANDP/ANDP, LANDH/ANDH (Tables 16-21) ---------------------------

def _encode_and_record(dist, w: Words) -> int:
    """One angular record (NE, E, LC, then the tables in order); returns its LOCB."""
    kind = type(dist).__name__
    if kind == "KalbachMannAngularDistribution":
        return -1
    energies = np.asarray(dist._energies, dtype=float)
    if kind == "IsotropicAngularDistribution":
        if energies.size == 0:
            return 0
        locb = w.next
        w.int(energies.size)
        w.reals(energies)
        w.int(*([0] * energies.size))
        return locb
    if kind == "EquiprobableAngularDistribution":
        forms = dist.energy_forms or ["equiprobable"] * energies.size
        tables = [(f, (dist._cosine_bins[k],)) for k, f in enumerate(forms)]
    elif kind == "TabulatedAngularDistribution":
        forms = dist.energy_forms or ["tabulated"] * energies.size
        tables = [(f, (dist._cosine_grid[k], dist._pdf[k], dist._cdf[k], dist.interpolation[k]))
                  for k, f in enumerate(forms)]
    else:
        raise ValueError(f"angular distribution of MT={dist.mt}: {kind} cannot be written")
    locb = w.next
    w.int(energies.size)
    w.reals(energies)
    lc_at = w.reserve(energies.size)
    for k, (form, data) in enumerate(tables):
        if form == "isotropic":
            continue
        if form == "equiprobable":
            w.set(lc_at + k, w.next)
            w.reals(data[0])
        else:
            mu, pdf, cdf, jj = data
            w.set(lc_at + k, -w.next)
            w.int(jj, len(mu))
            w.reals(mu)
            w.reals(pdf)
            w.reals(cdf)
    return locb


def encode_and(ace) -> Tuple[Words, Words]:
    """LAND (elastic, then the NR reactions with secondary neutrons) and AND."""
    ad = ace.angular_distributions
    land, and_ = Words(), Words()
    land.int(_encode_and_record(ad.elastic, and_))
    for mt in ace.reaction_mt_data.secondary_neutron_mt:
        land.int(_encode_and_record(ad.incident_neutron[int(mt)], and_))
    return land, and_


def encode_andp(ace) -> Tuple[Words, Words]:
    ad = ace.angular_distributions
    land, and_ = Words(), Words()
    for mt in ace.reaction_mt_data.photon_production:
        land.int(_encode_and_record(ad.photon_production[int(mt)], and_))
    return land, and_


def encode_andh(ace, i: int) -> Tuple[Words, Words]:
    ad = ace.angular_distributions
    land, and_ = Words(), Words()
    for key in ixs_keys(ace.reaction_mt_data.particle_production[i]):
        land.int(_encode_and_record(ad.particle_production[i][key], and_))
    return land, and_


# --- LDLW/DLW, LDLWP/DLWP, LDLWH/DLWH, DNEDL/DNED (Tables 29-30, 52) ------------

def encode_dlw(ace) -> Tuple[Words, Words, dict]:
    """LDLW and DLW; also the new TY of each reaction whose yield sits in DLW.

    A TY > 100 yield record sits just before its reaction's law chain, and
    |TY| - 100 is its position in DLW (KY = JED + |TY| - 101); the sign
    (frame) is kept.
    """
    ed = ace.energy_distributions
    ldlw, dlw = Words(), Words()
    new_ty = {}
    for mt in ace.reaction_mt_data.secondary_neutron_mt:
        mt = int(mt)
        y = ed.neutron_yields.get(mt)
        if y is not None:
            new_ty[mt] = 100 + dlw.next
            encode_energy_dependent_yield(y, dlw)
        ldlw.int(dlw.next)
        encode_law_chain(ed.incident_neutron[mt], dlw)
    return ldlw, dlw, new_ty


def encode_dlwp(ace) -> Tuple[Words, Words]:
    ed = ace.energy_distributions
    ldlw, dlw = Words(), Words()
    for mt in ace.reaction_mt_data.photon_production:
        ldlw.int(dlw.next)
        encode_law_chain(ed.photon_production[int(mt)], dlw)
    return ldlw, dlw


def encode_dlwh(ace, i: int) -> Tuple[Words, Words]:
    ed = ace.energy_distributions
    ldlw, dlw = Words(), Words()
    for key in ixs_keys(ace.reaction_mt_data.particle_production[i]):
        ldlw.int(dlw.next)
        encode_law_chain(ed.particle_production[i][key], dlw)
    return ldlw, dlw


def encode_dned(ace) -> Tuple[Words, Words]:
    ldlw, dlw = Words(), Words()
    for chain in ace.energy_distributions.delayed_neutron:
        ldlw.int(dlw.next)
        encode_law_chain(chain, dlw)
    return ldlw, dlw


# --- UNR, BDD (Tables 26-28, 22) -----------------------------------------------

def encode_unr(ace) -> Optional[Words]:
    u = ace.unresolved_resonance
    if u is None or not u.has_data:
        return None
    w = Words()
    m = u.table_length
    w.int(len(u.energies), m, u.interpolation, u.inelastic_flag, u.other_absorption_flag, u.factors_flag)
    w.reals(u.energies)
    for t in u.tables:
        for arr in (t.cumulative_probabilities, t.total_xs, t.elastic_xs, t.fission_xs, t.capture_xs,
                    t.heating_numbers):
            if len(arr) != m:
                raise ValueError(f"UNR table at E={t.energy} has {len(arr)} entries, not M={m}")
            w.reals(arr)
    return w


def encode_bdd(ace) -> Optional[Words]:
    d = ace.delayed_neutron_data
    if d is None or not d.has_delayed_neutron_data or not d.precursors:
        return None
    w = Words()
    for p in d.precursors:
        w.real(p.decay_constant)
        w.int(len(p.interpolation_regions))
        w.ints([nbt for nbt, _ in p.interpolation_regions])
        w.ints([i for _, i in p.interpolation_regions])
        w.int(len(p.energies))
        w.reals(p.energies)
        w.reals(p.probabilities)
    return w


# --- photons: GPD, MTRP, LSIGP/SIGP, YP (Tables 23, 47-49, 55) -----------------

def encode_gpd(ace) -> Optional[Words]:
    g = ace.photon_production_data
    if g is None:
        return None
    w = Words()
    w.reals(g.total_xs)
    if g.outgoing_energies:
        for group in g.outgoing_energies:
            w.reals(group)
    return w


def _encode_production_xs(xs, w: Words) -> None:
    """One SIGP/SIGH record: MFTYPE 12/16 (yield on MTMULT) or 13 (σ on the grid)."""
    if xs.mftype in (12, 16):
        w.int(xs.mftype, xs.mtmult, len(xs.interpolation_bounds))
        w.ints(xs.interpolation_bounds)
        w.ints(xs.interpolation_schemes)
        w.int(len(xs.energies))
        w.reals(xs.energies)
        w.reals(xs.yields)
    elif xs.mftype == 13:
        w.int(13, xs.energy_grid_index, len(xs.cross_sections))
        w.reals(xs.cross_sections)
    else:
        raise ValueError(f"production cross section of MT={xs.mt}: MFTYPE={xs.mftype} cannot be written")


def encode_sigp(ace) -> Tuple[Words, Words]:
    lsig, sig = Words(), Words()
    for mt in ace.reaction_mt_data.photon_production:
        lsig.int(sig.next)
        _encode_production_xs(ace.photon_production_xs.cross_sections[int(mt)], sig)
    return lsig, sig


def _encode_yield_mts(mts) -> Words:
    w = Words()
    w.int(len(mts))
    w.ints(mts)
    return w


def encode_yp(ace) -> Words:
    """YP (Table 55): NJOY writes it whenever there is photon production, NYP = 0 included."""
    y = ace.photon_yield_multipliers
    return _encode_yield_mts(y.multiplier_mts if y is not None else [])


# --- particles: PTYPE, NTRO, and per particle HPD...YH (Tables 56-61) ------------
# One SIGH, ANDH and DLWH record per MTRH entry, found by its key: an MT may
# repeat in MTRH (kika.ace.classes.ixs_keys).

def encode_hpd(ace, i: int) -> Words:
    d = ace.secondary_particle_cross_sections.particle_data[i + 1]
    w = Words()
    w.int(d.energy_grid_index, len(d.xs_values))
    w.reals(d.xs_values)
    w.reals(d.heating_numbers)
    return w


def encode_sigh(ace, i: int) -> Tuple[Words, Words]:
    lsig, sig = Words(), Words()
    xs = ace.particle_production_xs
    for key in ixs_keys(ace.reaction_mt_data.particle_production[i]):
        lsig.int(sig.next)
        _encode_production_xs(xs.particle_cross_sections[i + 1][key], sig)
    return lsig, sig


def encode_yh(ace, i: int) -> Words:
    y = ace.particle_yield_multipliers
    mts = (y.particle_multipliers.get(i + 1, []) if y is not None else [])
    return _encode_yield_mts(mts)
