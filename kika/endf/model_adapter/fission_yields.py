"""ENDF MF8/MT454 and MT459 (fission product yields) ↔ §18.4 ``productYields``.

Roadmap E7c. The yields live on the target nuclide of a standalone PoPs (a
fission yield sublibrary, NSUB=5 or 11) or on the decaying nuclide of a decay
tape that also states them (36 of ENDF/B-VIII.1's, Cf-252 among them). The
GNDS shape is FUDGE's (``ENDF_ITYPE_1``/``ENDF_ITYPE_5``): MT454 is
``elapsedTime label="initial"`` at 0 s, MT459 ``label="unspecified"``; a
neutron-induced evaluation lists one ``incidentEnergy`` per ENDF energy, a
spontaneous one puts the yields under the elapsed time; the products are one
list on the ``productYield`` when every energy names the same ones, in the
same order, and a list per ``yields`` otherwise (U-235 of B-VIII.1: 1 247,
1 260 and 1 237 products at its three energies). The uncertainty is the
variance DY².

What GNDS does not hold -- each energy's interpolation flag I, and the exact
spelling of every field (``5.37962-12`` where the canonical form has seven
digits) -- is kept the way :mod:`.decay_sublibrary` keeps MT457's: a rebuild
from the model plus field overrides that apply only while the rebuilt field is
unchanged.
"""
from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional, Tuple

from kika.nuclear_data.model import (CUMULATIVE, INDEPENDENT, ConversionReport, ElapsedTime,
                                     IncidentEnergy, PhysicalQuantity, ProductYield, Yields,
                                     pidFromZA, zaFromPid)

__all__ = ["decodeFissionYields", "encodeFissionYields", "FPY_KEY", "fissionProductPid",
           "zaAndStateOf"]

FPY_KEY = "mf8_fpy"
_LABELS = {454: INDEPENDENT, 459: CUMULATIVE}


def fissionProductPid(zafp: float, fps: float) -> str:
    """``(ZAFP, FPS)`` → FUDGE's id: ``Cu68``, or ``Cu68_m1`` for an isomer."""
    pid = pidFromZA(int(round(zafp)))
    return f"{pid}_m{int(round(fps))}" if fps else pid


def zaAndStateOf(pid: str) -> Tuple[int, int]:
    base, _, state = pid.partition("_m")
    return zaFromPid(base), int(state) if state else 0


def decodeFissionYields(mf8, induced: bool, sourcePath=None,
                        report: Optional[ConversionReport] = None
                        ) -> Tuple[Optional[ProductYield], dict]:
    """MF8/MT454 and /459 → one ``productYield`` labelled ``eval``, plus its book."""
    from .decay_sublibrary import _overrides, _rawSection

    report = report if report is not None else ConversionReport()
    sections = {mt: mf8.mt[mt] for mt in (454, 459) if mt in getattr(mf8, "mt", {})}
    if not sections:
        return None, {}
    lists = []
    for section in sections.values():
        for block in section.energies:
            lists.append([fissionProductPid(e.zafp, e.fps) for e in block.entries])
    common = all(names == lists[0] for names in lists)
    productYield = ProductYield(label="eval", nuclides=list(lists[0]) if common else None)
    book: Dict[str, dict] = {}
    for mt, section in sections.items():
        label = _LABELS[mt]
        if mt == 454:
            time = PhysicalQuantity(value=0.0, unit="s", label=label if induced else "eval")
        else:
            time = "unspecified"
        elapsed = ElapsedTime(label=label, time=time)
        perEnergy = induced or len(section.energies) != 1
        if perEnergy and not induced:
            report.warn(f"MF8/MT{mt}: a spontaneous-fission evaluation with "
                        f"{len(section.energies)} energies; its yields are listed per energy")
        for index, block in enumerate(section.energies):
            entries = block.entries
            yields = Yields(values=[e.y for e in entries],
                            uncertainty=[e.dy * e.dy for e in entries],
                            nuclides=None if common else [fissionProductPid(e.zafp, e.fps)
                                                          for e in entries])
            if perEnergy:
                elapsed.incidentEnergies.append(IncidentEnergy(
                    label=str(index), yields=yields,
                    energy=PhysicalQuantity(value=float(block.energy), unit="eV",
                                            label=str(index))))
            else:
                elapsed.yields = yields
        productYield.elapsedTimes.append(elapsed)
        entry = {"interp": [block.interpolation for block in section.energies],
                 "pad": (section.pad.pairs, section.pad.values, section.pad.interp),
                 "za": section._za, "awr": section._awr, "perEnergy": perEnergy}
        rebuilt = str(_section(productYield, elapsed, mt, entry, section._mat)).split("\n")[:-1]
        source = (_rawSection(sourcePath, 8, mt) if sourcePath is not None else None) \
            or str(section).split("\n")[:-1]
        entry["overrides"] = _overrides(rebuilt, source, report, f"MF8/MT{mt}")
        book[str(mt)] = entry
    return productYield, book


def _section(productYield: ProductYield, elapsed: ElapsedTime, mt: int, entry: dict, mat):
    from kika.endf.classes.mf8.yields import FissionYields, MF8FissionYields
    from kika.endf.utils import PadStyle

    blocks = ([(ie.energy.value, ie.yields) for ie in elapsed.incidentEnergies]
              if elapsed.incidentEnergies else [(0.0, elapsed.yields)])
    interp = list(entry.get("interp") or [])
    section = MF8FissionYields(number=mt, _za=float(entry["za"]), _awr=float(entry["awr"]),
                               _mat=mat)
    if "pad" in entry:
        section.pad = PadStyle(*entry["pad"])
    for index, (energy, yields) in enumerate(blocks):
        names = yields.nuclides if yields.nuclides is not None else productYield.nuclides
        values = []
        variances = yields.uncertainty or [0.0] * len(yields.values)
        for name, y, variance in zip(names, yields.values, variances):
            za, state = zaAndStateOf(name)
            values += [float(za), float(state), float(y), sqrt(variance) if variance > 0 else 0.0]
        flag = interp[index] if index < len(interp) else (2 if elapsed.incidentEnergies else 0)
        section.energies.append(FissionYields(energy=float(energy), interpolation=int(flag),
                                              values=values))
    return section


def encodeFissionYields(productYield: ProductYield, za: float, awr: float, book: dict, mat: int,
                        report: ConversionReport) -> List[tuple]:
    """``productYield`` → ``[(8, 454, section), (8, 459, section)]`` (those it has)."""
    from .decay_sublibrary import _TextSection, _applyOverrides

    out = []
    for mt in (454, 459):
        elapsed = productYield.elapsedTime(_LABELS[mt])
        if elapsed is None:
            continue
        entry = dict(book.get(str(mt)) or {})
        entry.setdefault("za", za)
        entry.setdefault("awr", awr)
        lines = str(_section(productYield, elapsed, mt, entry, mat)).split("\n")[:-1]
        lines = _applyOverrides(lines, entry.get("overrides"), report, f"MF8/MT{mt}")
        out.append((8, mt, _TextSection(lines, mat)))
    for elapsed in productYield.elapsedTimes:
        if elapsed.label not in _LABELS.values():
            report.lost(f"productYield elapsedTime {elapsed.label!r}: ENDF states only the "
                        f"independent (MT454) and cumulative (MT459) yields")
    return out
