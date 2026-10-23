"""The method section of a data-check report, built from the checks' constants.

Short on purpose, like the covariance one (:func:`..export.method_sections`):
one sentence of context per family, then one line per rule. What each check
means is in the legend (:mod:`.descriptions`).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..findings import DEFECT, NOTE, WARN
from .descriptions import _g, _pct

Rule = Tuple[Optional[str], str]


def method_sections() -> List[Tuple[str, str, List[Rule]]]:
    from . import distributions as dist
    from . import general as g
    from . import photons as ph
    from . import resonances as res
    from . import thermal as th
    from . import decay as dec
    from . import photoatomic as pa
    from . import xs

    sections = [
        ("Scope",
         "Layer 1: every file of the evaluation but the covariances, as it is written, read with "
         "kika's ENDF parser. Static checks only: nothing is reconstructed or processed, and "
         "nothing in the file is changed. No finding blocks anything.",
         [(DEFECT, "the data are not what ENDF-6 or physics allows, and using them gives a wrong "
                   "result."),
          (WARN, "can change a result, and needs the user's judgement."),
          (NOTE, "true and worth knowing, but not a fault: rounding, a convention, a choice."),
          (None, "Every threshold below was set by sweeping ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 "
                 "and reading what each level catches; a threshold not yet measured gives notes "
                 "only.")]),
        ("Header and nu-bar",
         "MF1: the header against the tape, AME2020 and itself; nu-bar, delayed neutrons and the "
         "fission energy release.",
         [(DEFECT, "a section of another ZA; LRP says MF2 exists and it does not; delayed nu-bar "
                   "without MF5/455 or with a different number of families; a negative nu-bar, "
                   "decay constant or fission energy component."),
          (WARN, f"AWR off AME2020 by more than {_pct(g.AWR_WARN)}; MT452 != MT455 + MT456 by more "
                 f"than {_g(g.NUBAR_SUM_WARN)}; ET or ER off its components by more than "
                 f"{_g(g.FER_SUM_WARN)}; nu-bar above {_g(g.NUBAR_MAX)}; a section missing from "
                 "the directory."),
          (NOTE, "smaller disagreements, down to rounding; directory line counts.")]),
        ("Tables",
         "Every TAB1 of every file, before its values are used.",
         [(DEFECT, "malformed interpolation regions, decreasing abscissae, blank pairs read as "
                   "zeros, a logarithmic law over a non-positive value."),
          (NOTE, "an abscissa given three times with different values.")]),
        ("Cross sections",
         "MF3 against kinematics, masses and its own sums.",
         [(DEFECT, f"a negative cross section outside the resolved range; a cross section more than "
                   f"{_pct(xs.THRESHOLD_DEFECT)} below its threshold; a level under the ground state; "
                   "QM left at zero."),
          (WARN, f"below threshold by more than {_g(xs.THRESHOLD_WARN)}; QM off AME2020 by more than "
                 f"{_g(xs.Q_WARN_EV / 1e3)} keV and {_pct(xs.Q_WARN_REL)}; a total off its partials by "
                 f"{_pct(xs.SUM_MATERIAL_REL)} and {_g(xs.SUM_MATERIAL_B * 1e3)} mb; levels out of order; "
                 "a reaction that stops at a non-zero value below EMAX."),
          (NOTE, f"QM off by more than {_g(xs.Q_NOTE_EV / 1e3)} keV; a total off by more than "
                 f"{_g(xs.SUM_NOTE)}; negative backgrounds; threshold steps; skipped level MTs.")]),
    ]
    sections += res.method(_g, _pct)
    sections += dist.method(_g, _pct)
    sections += ph.method(_g, _pct)
    for module in (th, dec, pa):
        sections += module.method(_g, _pct)
    return sections
