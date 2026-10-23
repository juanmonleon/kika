"""What each data finding means: one entry per ``check`` id of :mod:`kika.endf.checks.data`.

The same table shape as the covariance legend (:mod:`..descriptions`), read by
the same pages and by the desktop app; ``test_every_check_is_described``
keeps it in step with the checks. Thresholds come from the constants the
checks use. Written in ASCII; the pages give Greek letters.
"""
from __future__ import annotations

from typing import Dict

from ..descriptions import CheckDescription
from ..findings import DEFECT, NOTE, WARN

__all__ = ["DATA_CHECKS"]


def _g(x: float) -> str:
    import re

    return re.sub(r"e([+-])0(\d)", r"e\1\2", f"{x:g}").replace("e+", "e")


def _pct(x: float) -> str:
    return f"{x * 100:g} %"


def _build() -> Dict[str, CheckDescription]:
    from . import general as g
    from . import xs
    from . import distributions as dist
    from . import photons as ph
    from . import resonances as res
    from . import thermal as th
    from . import decay as dec
    from . import photoatomic as pa

    D = CheckDescription
    ANY = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 23, 26, 27, 28)
    TABS = (1, 3, 4, 5, 6, 12, 13, 14, 15, 23, 26, 27)
    out = {
        # ---- reading ----------------------------------------------------------
        "file_unreadable": D(
            "File that could not be read", ANY, {DEFECT: "always"},
            "kika's parser failed on the whole file, so nothing in it was checked and nothing "
            "downstream can use it. The evidence carries the parser's message."),
        "gnds_translated": D(
            "Checked through kika's GNDS translation", (1,), {NOTE: "always"},
            "A GNDS evaluation is checked as the ENDF-6 tape kika writes from it. A finding can then "
            "come from the evaluation or from the translation; the GNDS file itself is not changed."),
        "section_unreadable": D(
            "Section dropped by the parser", ANY, {DEFECT: "always"},
            "One section of the file could not be read and was left out: a record that breaks "
            "the format (a count that does not match what follows, a field out of place)."),
        # ---- MF1/451 ------------------------------------------------------------
        "header_missing": D(
            "No MF1/MT451", (1,), {DEFECT: "always"},
            "Without its header a tape does not say what material, projectile, sublibrary or "
            "energy range it is; every processor needs it."),
        "head_za_mismatch": D(
            "ZA of a section differs from the header", ANY, {DEFECT: "always"},
            "Every section repeats the material's ZA in its HEAD record (CHECKR's TEST2). A "
            "different one is a section of another material, or a typing error."),
        "head_awr_mismatch": D(
            "AWR of a section differs from the header", ANY,
            {WARN: f"more than {_g(g.HEAD_AWR_WARN)} relative",
             NOTE: f"more than {_g(g.HEAD_AWR_NOTE)} relative"},
            "Every section repeats the mass ratio AWR; codes take it from whichever section they "
            "read, so two values give two kinematics."),
        "header_flags": D(
            "Header flags contradict the tape", (1,),
            {DEFECT: "LRP says MF2 is given and it is not",
             WARN: "LRP = -1 with an MF2; LFI = 0 with fission; LIS and ELIS disagree; AWI not 1 "
                   "on a neutron sublibrary",
             NOTE: "LFI = 1 without fission"},
            "MF1/451's flags say what a processor should look for: LRP whether there are resonance "
            "parameters, LFI whether there is fission, LIS/ELIS the target's state, AWI the "
            "projectile."),
        "directory_mismatch": D(
            "Directory and sections disagree", (1,),
            {DEFECT: "NXC does not match the entries read",
             WARN: "a section missing from the directory, or listed and absent"},
            "MF1/451 lists every section of the tape (CHECKR's TESTD). A reader that trusts the "
            "directory skips a section it does not list, or looks for one that is not there."),
        "directory_line_count": D(
            "Directory line counts", (1,), {NOTE: "always"},
            "The NC of a directory entry is not the number of lines of its section. No processor "
            "uses NC to read, so this only says the directory was not regenerated."),
        "section_split": D(
            "Section split across the tape", ANY, {DEFECT: "always"},
            "The lines of one section appear in two places. ENDF-6 sections are contiguous; a "
            "reader takes the first part and loses the rest."),
        "tape_materials": D(
            "More than one material", (1,), {NOTE: "always"},
            "The file holds several materials; the check looked at the first."),
        "awr_mass": D(
            "AWR against AME2020", (1,),
            {WARN: f"more than {_pct(g.AWR_WARN)} relative",
             NOTE: f"more than {_g(g.AWR_NOTE)} relative"},
            "AWR (atomic mass over neutron mass) against the AME2020 atomic mass of the target's "
            "ground state. Every threshold and recoil energy scales with it; evaluations made with "
            "older mass tables differ at the 1e-5 level."),
        "emax_coverage": D(
            "Cross sections do not reach EMAX", (1, 3),
            {WARN: "no cross section reaches EMAX", NOTE: "cross sections go beyond EMAX"},
            "EMAX is the upper limit of the evaluation. Below it every reaction should be given; "
            "a code reads a table that stops early as zero."),
        # ---- nu-bar -------------------------------------------------------------
        "nubar_missing": D(
            "Fission without nu-bar", (1,), {DEFECT: "always"},
            "A fissionable material needs MF1/MT452 (and MT452 whenever MT455 or MT456 is given): "
            "without it a code cannot tell how many neutrons fission releases."),
        "nubar_representation": D(
            "Unknown nu-bar representation", (1,), {DEFECT: "always"},
            "LNU must be 1 (polynomial) or 2 (table)."),
        "nubar_coverage": D(
            "nu-bar stops before fission does", (1,), {WARN: "always"},
            "The nu-bar table ends below the last energy of the fission cross section; above it "
            "codes either hold its last value or read zero."),
        "nubar_range": D(
            "nu-bar out of range", (1,),
            {DEFECT: "negative", WARN: f"above {_g(g.NUBAR_MAX)} (delayed: above {_g(g.NUBAR_DELAYED_MAX)})"},
            "Neutrons per fission are positive and below FIZCON's bounds."),
        "nubar_sum": D(
            "Total nu-bar is not prompt plus delayed", (1,),
            {WARN: f"more than {_g(g.NUBAR_SUM_WARN)} of MT452", NOTE: f"more than {_g(g.NUBAR_SUM_NOTE)}"},
            "MT452 = MT455 + MT456 at every energy (FIZCON's SUM452). Codes use either the total "
            "or the two parts, so a gap changes the neutrons per fission between codes."),
        "delayed_constants": D(
            "Non-positive decay constant", (1,), {DEFECT: "always"},
            "A precursor family decays with lambda > 0 (FUDGE: fatal)."),
        "delayed_abundances": D(
            "Precursor abundances do not sum to 1", (1,),
            {WARN: f"off by more than {_g(g.ABUNDANCE_WARN)}", NOTE: f"off by more than {_g(g.ABUNDANCE_NOTE)}"},
            "With energy-dependent decay data (LDG = 1) MF1/455 gives each family's abundance; "
            "they are fractions of one delayed neutron (ACER's consis: 1e-3)."),
        "delayed_spectra_missing": D(
            "Delayed nu-bar without spectra", (1, 5), {DEFECT: "always"},
            "MF1/455 gives delayed neutrons but there is no MF5/455 to give them an energy. "
            "Found in ENDF/B-VIII.1 Ra-223."),
        "delayed_groups_mismatch": D(
            "Precursor families differ between MF1 and MF5", (1, 5), {DEFECT: "always"},
            "MF5/455 has one spectrum per precursor family of MF1/455; different counts cannot "
            "be paired (FUDGE's translator refuses them)."),
        # ---- MT458 --------------------------------------------------------------
        "fission_energy_missing": D(
            "Fission without MT458", (1,), {NOTE: "always"},
            "Without the components of the fission energy release, heating has to assume how "
            "the energy is shared (NJOY's HEATR says so too)."),
        "fission_energy_record": D(
            "MT458 too short", (1,), {DEFECT: "always"},
            "MT458 holds fewer than the 18 values of its nine thermal components."),
        "fission_energy_negative": D(
            "Negative fission energy component", (1,),
            {DEFECT: "a component, or a tabulated component, negative", WARN: "an uncertainty negative"},
            "Every component of the fission energy release (fragments, neutrons, photons, betas, "
            "neutrinos) is an energy carried away: it is not negative. The LFC = 1 tables are "
            "checked too, which no other code does."),
        "fission_energy_sum": D(
            "MT458 components do not add up", (1,),
            {WARN: f"more than {_g(g.FER_SUM_WARN)} of ET", NOTE: f"more than {_g(g.FER_SUM_NOTE)}"},
            "ET is the sum of the first seven components and ER = ET - ENU (FIZCON's CKS458)."),
        # ---- tables -------------------------------------------------------------
        "table_empty": D(
            "Table with no points", TABS, {DEFECT: "always"}, "A TAB1 that holds nothing."),
        "table_interpolation": D(
            "Interpolation regions malformed", TABS, {DEFECT: "always"},
            "NBT strictly increasing with the last equal to NP, and every INT a defined law "
            "(1-5, or 6 in MF3) -- CHECKR's TESTIN. A reader cannot tell which law applies where."),
        "table_blank_pairs": D(
            "Blank pairs at the end of a table", TABS, {DEFECT: "always"},
            "NP counts more pairs than the record holds; the blank fields read as (0, 0) after the "
            "last real point. Found in JEFF-4.0 MF3 (Mo-100, Ru-101, Nd-142, ...)."),
        "table_grid_order": D(
            "Abscissae decrease", TABS, {DEFECT: "always"},
            "A TAB1's abscissae never decrease; where they do, the table has no reading."),
        "table_grid_repeat": D(
            "Abscissa repeated three times", TABS, {NOTE: "with values that are not all equal"},
            "Two equal abscissae make a discontinuity, from the first value to the second. A "
            "third copy with another value is not used by any reading (FIZCON flags it)."),
        "table_log_nonpositive": D(
            "Logarithmic law over non-positive values", TABS, {DEFECT: "always"},
            "INT = 3, 4, 5 take the logarithm of x, y or both: a non-positive x, or a negative y, "
            "has none (FIZCON's TEST1)."),
        "table_log_zero": D(
            "Logarithmic law towards zero", TABS, {NOTE: "always"},
            "A log-y law with a value of exactly zero at one end: undefined, and read as zero over "
            "the interval by kika, as a jump by others."),
        # ---- MF3 ----------------------------------------------------------------
        "xs_negative": D(
            "Negative cross section", (3,),
            {DEFECT: "outside the resolved range, or in a reaction that is not a background",
             NOTE: "a background inside the resolved range of a tape with resonance parameters"},
            "A cross section is not negative. Inside the resolved range MF3 is a background to "
            "which MF2's resonances are added, and it may be negative there; whether the sum is "
            "positive is the reconstruction's check."),
        "xs_below_threshold": D(
            "Cross section below its threshold", (3,),
            {DEFECT: f"more than {_pct(xs.THRESHOLD_DEFECT)} of the threshold below it",
             WARN: f"more than {_g(xs.THRESHOLD_WARN)} below it"},
            "A reaction with QI < 0 cannot take place below E_th = -QI (A + a)/A (relativistically "
            "a little more). The lower of the two formulas is used, so an evaluator who rounds "
            "either way is not flagged. NJOY moves such points up."),
        "xs_threshold_step": D(
            "Cross section starts with a step", (3,), {NOTE: "always"},
            "A threshold reaction whose first point, at its threshold, is not zero. Physically the "
            "cross section opens from zero; NJOY inserts a jump, FUDGE calls it severe."),
        "xs_coverage": D(
            "Reaction stops before EMAX", (3,), {WARN: "always"},
            "The table ends below EMAX at a non-zero value, so the reaction drops to zero there."),
        "q_mass": D(
            "Q value against AME2020", (3,),
            {DEFECT: "QM = 0 where the masses give a Q beyond the warning",
             WARN: f"off by more than {_g(xs.Q_WARN_EV / 1e3)} keV and {_pct(xs.Q_WARN_REL)} of Q",
             NOTE: f"off by more than {_g(xs.Q_NOTE_EV / 1e3)} keV and {_pct(xs.Q_NOTE_REL)} of Q"},
            "QM is the ground-state Q of the products the MT names, which the AME2020 atomic "
            "masses give (FUDGE's check, 1 keV). Evaluations made with older mass tables differ by "
            "keV; a unit error or a QM left at zero differ by MeV."),
        "level_qm": D(
            "Levels of one reaction with different QM", (3,), {WARN: "always"},
            "MT51-90, 600-648 ... are levels of one reaction and share its ground-state QM."),
        "level_energy": D(
            "Level below the ground state", (3,), {DEFECT: "always"},
            "The excitation of a level, QM - QI, cannot be negative."),
        "level_order": D(
            "Levels out of order", (3,), {WARN: "always"},
            "Levels take consecutive MTs in increasing energy (FIZCON's CHK_LEVEL)."),
        "level_continuum_overlap": D(
            "Continuum starts below the last level", (3,), {NOTE: "always"},
            "The continuum of a reaction (MT91, 649, ...) starts in excitation below its last "
            "discrete level. Evaluators do this where the discrete levels are known to be "
            "incomplete; it is a choice, said here because the two then overlap."),
        "level_gap": D(
            "Gap in the level MTs", (3,), {NOTE: "always"},
            "A level MT is skipped (MT n given without MT n-1); NJOY's HEATR calls it a probable "
            "ENDF error, but evaluators sometimes leave a level out on purpose."),
        "xs_sum": D(
            "A total differs from the sum of its parts", (3,),
            {WARN: f"by more than {_pct(xs.SUM_MATERIAL_REL)} and {_g(xs.SUM_MATERIAL_B * 1e3)} mb at some energy",
             NOTE: f"by more than {_g(xs.SUM_NOTE)} relative"},
            "MT1, MT3, MT4, MT16, MT18, MT27, MT101 and MT103-107 are sums of other reactions. "
            "Codes rebuild them from the parts (kika and NJOY) or use them as written (a "
            "transport code reading the total), so a gap means two answers. On a tape whose MF2 "
            "adds resonances (LRP = 1) only the energies above the resonance ranges are compared: "
            "below, MF3 holds backgrounds, and the sums of the reconstructed cross sections are "
            "the reconstruction's check."),
    }
    out.update(dist.descriptions(D, _g, _pct))
    out.update(ph.descriptions(D, _g, _pct))
    out.update(res.descriptions(D, _g, _pct))
    for module in (th, dec, pa):
        out.update(module.descriptions(D, _g, _pct))
    return out


DATA_CHECKS: Dict[str, CheckDescription] = _build()
