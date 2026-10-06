"""What each layer-1 finding means: one entry per ``check`` name.

Written in ASCII like the summaries; :meth:`CheckDescription.to_dict` and the
pages give them with Greek letters (:mod:`.symbols`).

The reports' legend and the desktop app's detail panel both read this table, so
there is one text for each check. Thresholds are taken from the constants the
checks use. ``test_every_check_is_described`` keeps the table and the checks in
step: a new ``check`` name without an entry here fails it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

from .findings import DEFECT, NOTE, WARN
from .symbols import to_symbols

__all__ = ["CheckDescription", "CHECKS", "describe"]


@dataclass(frozen=True)
class CheckDescription:
    """``levels`` maps each level the check can give to when it gives it."""

    title: str
    mf: Tuple[int, ...]
    levels: Mapping[str, str]
    description: str

    def to_dict(self) -> Dict[str, object]:
        """Plain data, with Greek letters and math signs (the texts here are ASCII)."""
        return {"title": to_symbols(self.title), "mf": list(self.mf),
                "levels": {lv: to_symbols(self.levels[lv])
                           for lv in (DEFECT, WARN, NOTE) if lv in self.levels},
                "description": to_symbols(self.description)}


def _pct(x: float) -> str:
    return f"{x * 100:g} %"


def _build() -> Dict[str, CheckDescription]:
    from . import covariances as c
    from .mf35 import SUM_RULE_DEFECT

    ALL = (31, 32, 33, 34, 35, 40)
    XS = (31, 33, 34, 40)  # the NI/NC-record files
    D = CheckDescription
    return {
        # ---- structure ------------------------------------------------------
        "parse_error": D(
            "Section dropped by the parser", ALL,
            {DEFECT: "always"},
            "kika's ENDF parser could not read this section and left it out, so nothing "
            "downstream sees it. Usually a record that breaks the format (a count that does "
            "not match what follows); the evidence carries the parser's message."),
        "count_mismatch": D(
            "A count does not match the content", XS + (35,),
            {DEFECT: "always"},
            "A declared count (NL, NMT1, NI, NC, NCI, NK, or the number of (L, L1) blocks "
            "NL/NL1 imply) differs from what the section holds. A processor that trusts the "
            "count reads a different set of blocks than the evaluator wrote."),
        "nt_mismatch": D(
            "NT inconsistent with NE and LS", XS + (35,),
            {DEFECT: "always"},
            "The number of values NT is not what the other counts of the record imply "
            "(NE and LS for LB=5, the two grids for LB=6, NP for LB 0-4, 8 and 9). The "
            "record cannot be decoded unambiguously."),
        "lb_invalid": D(
            "LB not allowed in this file", XS,
            {DEFECT: "always"},
            "The covariance form LB is not one ENDF-6 defines for this file (MF34 allows "
            "only LB = 0, 1, 2, 5, 6). The sub-subsection cannot be interpreted."),
        "lty_invalid": D(
            "LTY not defined", (31, 33, 40),
            {DEFECT: "always"},
            "An NC sub-subsection declares an LTY other than 0-3, so the derivation it "
            "describes is unknown."),
        "lct_invalid": D(
            "LCT not 0, 1 or 2", (34,),
            {WARN: "always"},
            "The frame flag of an MF34 block is not one ENDF-6 defines; which frame the "
            "Legendre coefficients refer to is left open."),
        "l_out_of_range": D(
            "Legendre order out of range", (34,),
            {DEFECT: "always"},
            "A block (L, L1) has an order outside 1..NL (or NL1), beyond the LTT of the "
            "section, or L > L1 within one MT, where only the upper triangle may be stored."),
        "duplicate_block": D(
            "The same block written twice", XS,
            {DEFECT: "always"},
            "The same (MT, MT1) or (L, L1) appears twice in one section. kika keeps one of "
            "them; which one the evaluator meant is not knowable from the file."),
        "grid_not_increasing": D(
            "Energy grid not increasing", XS + (35,),
            {DEFECT: "always"},
            "The energies of a block decrease somewhere. Bins with negative width have no "
            "meaning, and the projection onto a union grid is undefined."),
        "grid_repeated_point": D(
            "Repeated grid point", XS + (35,),
            {WARN: "always"},
            "Two consecutive energies are equal, giving a bin of zero width. Harmless if "
            "its values are ignored, but the covariance of that bin has no energy range."),
        "grid_too_short": D(
            "Grid with fewer than two points", XS + (35,),
            {DEFECT: "always"},
            "A block whose grid cannot hold a single bin: there is no covariance in it."),
        "decode_error": D(
            "Record that cannot be decoded", XS + (32,),
            {DEFECT: "always"},
            "kika could not turn the record into a matrix (in MF32, the resonance-parameter "
            "decoder skipped the range, e.g. an NNN that does not add up)."),
        "lt_not_decoded": D(
            "LB 0-2 with LT > 0 not decoded", (34,),
            {NOTE: "always"},
            "An MF34 LB 0-2 record with a second table (LT > 0) is kept verbatim but not "
            "checked."),
        "not_decoded": D(
            "Kept but not checked", (32,),
            {NOTE: "always"},
            "Part of an MF32 range whose form kika keeps verbatim but cannot assemble, so "
            "no value check ran on it."),
        "ls1_in_cross_block": D(
            "Symmetric storage (LS=1) in a cross block", XS,
            {DEFECT: "the stored triangle is not all zeros",
             NOTE: "the stored triangle is all zeros, so mirroring it changes nothing"},
            "LS=1 says the stored triangle is to be mirrored, which is only right for a "
            "symmetric matrix. A cross block (L != L1, or MT != MT1) is not symmetric: the "
            "stored triangle may be valid, but the matrix kika and other processors build "
            "by mirroring it is not what the evaluator meant, and it often gives |rho| > 1 "
            "(Ne-20 of JEFF-4.0: 0.877 stored, 4.63 mirrored)."),
        # ---- values ---------------------------------------------------------
        "correlation_out_of_bounds": D(
            "|rho| > 1", ALL,
            {DEFECT: f"|rho| > 1 + {c.RHO_DEFECT:g}",
             NOTE: f"1 + {c.RHO_ROUNDING:g} < |rho| <= 1 + {c.RHO_DEFECT:g} (rounding)"},
            "A correlation beyond +-1 means the block is not a covariance matrix: no "
            "random variables have it. In a cross block the variances come from the two "
            "self blocks on a common grid. When the block is LS=1, the evidence says "
            "whether the stored triangle alone already exceeds 1 or the mirror creates it."),
        "negative_variance": D(
            "Negative variance", ALL,
            {DEFECT: "always"},
            "A diagonal element below zero. No variance can be negative; kika's MF34 "
            "assembly sets these to zero without saying so, which hides the fault."),
        "inert_rows": D(
            "Rows with no uncertainty", ALL,
            {NOTE: "always"},
            "Rows that are exactly zero (max == min == 0): those energies or parameters are "
            "given no uncertainty. Legal, and common at the edges of a grid; a sampler "
            "leaves them at their central value."),
        "covariance_without_variance": D(
            "Covariance on a row without variance", ALL,
            {DEFECT: "always"},
            "A row with zero variance but non-zero covariances. Its correlations are "
            "infinite, and the block cannot be positive semi-definite."),
        "asymmetric_self_block": D(
            "Self block not symmetric", ALL,
            {DEFECT: f"relative asymmetry > {c.ASYM_DEFECT:g}",
             NOTE: f"relative asymmetry > {c.ASYM_NOTE:g} (rounding)"},
            "A self block stored in full (LS=0, or MF32) differs from its transpose. A "
            "covariance matrix is symmetric; which triangle is meant is not knowable."),
        "not_positive_semidefinite": D(
            "Not positive semi-definite", ALL,
            {DEFECT: f"|lambda_min| / lambda_max > {c.PSD_DEFECT:g}, or a warning whose "
                     f"clipping changes some sigma by {_pct(c.PSD_IMPACT_DEFECT)} or more",
             WARN: f"{c.PSD_NOTE:g} <= |lambda_min| / lambda_max <= {c.PSD_DEFECT:g}, "
                   f"not explained by rounding, and clipping changes sigma by "
                   f"{_pct(c.PSD_IMPACT_NOTE)} to {_pct(c.PSD_IMPACT_DEFECT)}",
             NOTE: f"|lambda_min| / lambda_max < {c.PSD_NOTE:g}, or explained by the rounding "
                   f"of the file, or clipping changes no sigma by {_pct(c.PSD_IMPACT_NOTE)} or more"},
            "The block has negative eigenvalues: some combination of the quantities has a "
            "negative variance. Samplers have to repair it (clip the eigenvalues), and the "
            "repair changes the uncertainties. The level says how far from PSD the block is "
            "and how much the repair would change sigma; the evidence names the record "
            "already indefinite on its own. MF32 is judged on its correlation matrix."),
        "psd_not_evaluated": D(
            "PSD not evaluated (too large)", (32,),
            {NOTE: "always"},
            "An MF32 block with more parameters than the eigenvalue limit; its positive "
            "semi-definiteness was not computed."),
        "relative_uncertainty_above_one": D(
            "Relative uncertainty above 100 %", (31, 32, 33),
            {NOTE: "always"},
            "sigma_rel > 1. Not a fault: a cross section has no upper bound. But a normal "
            "draw gives negative values with the probability in the evidence; a lognormal "
            "sampler avoids it."),
        "implausible_relative_uncertainty": D(
            "Implausible relative uncertainty", (31, 33),
            {WARN: f"sigma_rel > {c.RELATIVE_IMPLAUSIBLE:g} where the central value is at "
                   f"least {_pct(c.RELATIVE_THRESHOLD_ZONE)} of the reaction's maximum"},
            "An uncertainty larger than any of ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 shows "
            "outside threshold regions. The kind of value a unit or format error produces "
            "(an absolute LB=8 summed as relative gave 53 such bins in one tape)."),
        "variance_exceeds_physical_bound": D(
            "Variance above a physical bound", (34, 35),
            {DEFECT: "always"},
            "MF34: |a_l| <= 1 for l >= 1, so sigma(a_l) > 1 is impossible whatever the "
            "distribution (Popoviciu); the relative sigma is scaled by the smallest |a_l| "
            "in the bin. MF35: a probability P has var(P) <= P(1 - P), with P from MF5."),
        "central_values_unavailable": D(
            "Central values not available", (31, 33, 34, 35, 40),
            {NOTE: "always"},
            "The magnitude checks need the central values (MF1, MF3/PENDF, MF4, MF5) and "
            "they were not read or are not in the tape; MF40's are in MF10, which kika "
            "does not read. Those checks did not run."),
        "mixed_needs_cross_sections": D(
            "Mixed absolute/relative block checked in part", (31, 33, 40),
            {NOTE: "always"},
            "The block sums absolute (LB=8/9) and relative components. Adding them needs "
            "the cross sections; without a PENDF only the relative part was checked."),
        "mixed_absolute_relative": D(
            "MF34 block mixes absolute and relative", (34,),
            {WARN: "always"},
            "An LB=0 (absolute) record summed with relative ones. kika sums them all as "
            "relative, so the assembled block is wrong by the absolute part; only the "
            "relative part was checked."),
        # ---- completeness ---------------------------------------------------
        "missing_self_block": D(
            "Cross covariances without the self block", XS,
            {DEFECT: "always"},
            "The section has covariances with other reactions (or orders, or final states) "
            "but none with itself. Correlations cannot be formed, and the cross blocks "
            "cannot be checked."),
        "missing_partner": D(
            "The other side of a cross block is missing", XS,
            {DEFECT: "the partner MT, order or final state has no section or no self block",
             NOTE: "the cross block is all zeros"},
            "A cross block names an MT1 (or order L1) that has no section or no variance, "
            "or that is a component of a lumped reaction. The block cannot be placed in a "
            "joint matrix."),
        "order_without_variance": D(
            "Order declared without a self block", (34,),
            {NOTE: "always"},
            "NL counts a Legendre order that has no (L, L) block. Legal: that order is "
            "given no uncertainty."),
        "cross_block_below_diagonal": D(
            "Cross block stored under the diagonal", (31, 33, 34),
            {NOTE: "always"},
            "A cross block written in the section of the larger MT (or L). ENDF-6 asks for "
            "the upper triangle; kika places it transposed, so nothing is lost."),
        "symmetric_block_repeated": D(
            "Cross block written both ways", (31, 33, 34),
            {NOTE: "always"},
            "The same cross block appears as (X, Y) and as (Y, X), and both agree."),
        "symmetric_block_conflict": D(
            "Cross block written both ways, differently", (31, 33, 34),
            {DEFECT: "always"},
            "(X, Y) and (Y, X) are both given and disagree on the union grid. kika keeps "
            "whichever it places last, without saying so."),
        "mat1_is_own_mat": D(
            "MAT1 set to the material itself", (31, 33),
            {NOTE: "always"},
            "MAT1 equals the tape's MAT where ENDF-6 asks for 0. Read as the same material."),
        "external_material": D(
            "Covariance with another material", (31, 33, 40),
            {NOTE: "always"},
            "Blocks with MAT1 != 0 relate this material to another one; a single file "
            "cannot check them."),
        "unresolved_reference": D(
            "Reference that does not resolve", (31, 33),
            {DEFECT: "always"},
            "An NC (LTY=0) derivation names an MT without a section, without a variance, "
            "itself, or a component of a lumped reaction; or a lumped MT outside 851-870; "
            "or an LTY=1 standard that is missing. The derived covariance kika assembles is "
            "smaller than the one the file declares."),
        "nc_misplaced": D(
            "NC sub-subsection in a cross block", (31, 33),
            {DEFECT: "always"},
            "LTY 0 or 1 may only appear in a self block; in a cross block its meaning is "
            "undefined."),
        "nc_ranges_overlap": D(
            "Overlapping derivation ranges", (31, 33),
            {DEFECT: "always"},
            "Two LTY=0 derivations of the same MT cover overlapping energy ranges, so the "
            "covariance there is defined twice."),
        "nc_chained": D(
            "Derivation from a derived reaction", (31, 33),
            {DEFECT: "a constituent is itself derived over an overlapping range",
             NOTE: "the chain passes through another range"},
            "An MT is derived from one that is itself derived. kika's resolver ignores the "
            "energy ranges of the derivations, so a chain can resolve differently from "
            "what the file describes."),
        "nc_only_first_resolved": D(
            "Only the first derivation resolved", (31, 33),
            {NOTE: "always"},
            "Several different LTY=0 rules for one MT; kika resolves only the first."),
        "ratio_to_standard": D(
            "Covariance relative to a standard", (31, 33),
            {NOTE: "always"},
            "LTY 1-3 (ratio to a standard cross section) is not resolved by kika."),
        "lumped_without_components": D(
            "Lumped reaction without components", (33,),
            {WARN: "always"},
            "A lumped covariance MT (851-870) that no section names as its MTL, so the "
            "cross section its relative covariance refers to cannot be summed."),
        "grid_coverage": D(
            "Covariance grid shorter than MF3", (31, 33),
            {NOTE: "always"},
            "The covariance does not reach the whole range where the cross section is "
            "non-zero. Outside it the reaction is given no uncertainty."),
        # ---- MF32 -----------------------------------------------------------
        "correlation_index_out_of_range": D(
            "Correlation index outside the matrix", (32,),
            {DEFECT: "always"},
            "A compact (INTG) correlation points to a parameter beyond NNN, so it belongs "
            "to no parameter of the range."),
        "formalism_mismatch": D(
            "Formalism differs from MF2", (32,),
            {DEFECT: "always"},
            "The range's LRU/LRF differ from those of the same range in MF2: the "
            "parameters the covariance is about are not the ones File 2 uses."),
        "range_not_in_mf2": D(
            "Range not in MF2", (32,),
            {WARN: "always"},
            "MF32 has a resonance range that MF2 does not have."),
        "range_limits_differ_from_mf2": D(
            "Range limits differ from MF2", (32,),
            {NOTE: "always"},
            "The range exists in MF2 with the same formalism, but its energy limits differ."),
        "parameters_not_in_mf2": D(
            "Resonances not in MF2", (32,),
            {WARN: "always"},
            "Resonance energies in MF32 that MF2 does not list: the covariance is about "
            "parameters the cross section is not built from."),
        # ---- MF35 -----------------------------------------------------------
        "missing_distribution": D(
            "Spectrum covariance without the spectrum", (35,),
            {DEFECT: "always"},
            "MF35/MT has no MF5/MT: a covariance of a spectrum the file does not give."),
        "sum_rule_violated": D(
            "Sum rule violated", (35,),
            {DEFECT: f"max_i |sum_j C_ij| / max |C| > {SUM_RULE_DEFECT:g}"},
            "The covariance of a normalised spectrum must have zero row sums: the "
            "probabilities add to one in every sample. A matrix that breaks it does not "
            "describe a normalised spectrum."),
        "variance_where_distribution_is_zero": D(
            "Variance where the spectrum is zero", (35,),
            {WARN: "always"},
            "Bins with an uncertainty where MF5 gives P = 0 over the whole band."),
        "band_empty": D(
            "Empty or reversed band", (35,),
            {DEFECT: "the band's upper energy is below the lower", WARN: "E1 == E2"},
            "An incident-energy band of zero or negative width."),
        "band_gap": D(
            "Gap between bands", (35,),
            {WARN: "always"},
            "Consecutive incident-energy bands leave a gap: no covariance there."),
        "band_overlap": D(
            "Overlapping bands", (35,),
            {DEFECT: "always"},
            "Consecutive incident-energy bands overlap, so two covariances apply there."),
        "band_coverage": D(
            "Bands do not cover MF5", (35,),
            {NOTE: "always"},
            "The bands do not reach every incident energy of MF5."),
    }


CHECKS: Dict[str, CheckDescription] = _build()


def describe(checks: Iterable[str]) -> List[Tuple[str, CheckDescription]]:
    """The entries of *checks* in first-seen order; unknown names are skipped."""
    out, seen = [], set()
    for name in checks:
        if name in CHECKS and name not in seen:
            seen.add(name)
            out.append((name, CHECKS[name]))
    return out
