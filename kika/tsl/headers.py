"""What a TSL library listing needs from a GNDS, ACE, OpenMC or G4NDL material.

The ENDF-6 tapes are read by :func:`kika.tsl.library.read_tsl_header`. These
are its counterparts for the other formats, each cheap enough to run over a
whole library: a GNDS file's first kilobytes, an ACE table's header lines, an
OpenMC file's attributes, a G4NDL material's small cross-section files.

Every one returns :class:`~kika.tsl.library.TSLTape` entries with ``format``
set, so a library of any format is listed by compound the same way.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .naming import tsl_naming, tsl_naming_from_openmc, tsl_naming_from_title

__all__ = ["gnds_tsl_header", "ace_tsl_tapes", "openmc_tsl_header", "g4ndl_tsl_tapes",
           "is_gnds_tsl"]

#: How far into a GNDS file its forms' heads are looked for. The S(α,β) table
#: of the incoherent inelastic form comes after them, and is never read here.
_GNDS_HEAD_BYTES = 4 << 20


def _floats(text: str) -> List[float]:
    out = []
    for token in text.split():
        try:
            out.append(float(token))
        except ValueError:
            pass
    return out


def is_gnds_tsl(path: Union[str, Path]) -> bool:
    try:
        with open(path, "rb") as handle:
            head = handle.read(2048).decode("utf-8", errors="replace")
    except OSError:
        return False
    return "<reactionSuite" in head and 'interaction="thermalNeutronScatteringLaw"' in head


def _gnds_name(file_name: str) -> str:
    """``tsl-HinH2O.endf.gnds.xml`` → ``tsl-HinH2O.endf`` (what the naming rules read)."""
    name = re.sub(r"\.gnds\.xml$|\.xml$", "", file_name, flags=re.IGNORECASE)
    return name if name.lower().endswith(".endf") else name + ".endf"


def _section(text: str, tag: str) -> Optional[str]:
    start = text.find(f"<{tag}")
    if start < 0:
        return None
    end = text.find(f"</{tag}>", start)
    return text[start:end if end > 0 else len(text)]


def gnds_tsl_header(path: Union[str, Path]):
    """A GNDS TSL file as a library lists it, from its head (S(α,β) is not read)."""
    from .library import TSLTape

    path = Path(path)
    with open(path, "rb") as handle:
        text = handle.read(_GNDS_HEAD_BYTES).decode("utf-8", errors="replace")
    suite = re.search(r"<reactionSuite\b([^>]*)>", text)
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', suite.group(1))) if suite else {}
    evaluated = re.search(r"<evaluated\b([^>]*)>", text)
    eattrs = dict(re.findall(r'(\w+)="([^"]*)"', evaluated.group(1))) if evaluated else {}
    author = re.search(r'<author name="([^"]*)"', text)
    compat = re.search(r"<endfCompatible><!\[CDATA\[(.*?)\]\]>", text, re.DOTALL)
    description = compat.group(1).split("\n") if compat else []
    zsymam = description[0][:11].strip() if description else None
    mat_match = re.search(r"MATERIAL\s+(\d+)", compat.group(1)) if compat else None

    tape: Dict[str, object] = dict(elastic=None, elastic_temperatures=[], temperatures=[])
    coherent = _section(text, "thermalNeutronScatteringLaw_coherentElastic")
    incoherent = _section(text, "thermalNeutronScatteringLaw_incoherentElastic")
    inelastic = _section(text, "thermalNeutronScatteringLaw_incoherentInelastic")
    elastic_t: List[float] = []
    if coherent:
        grids = re.findall(r'<grid index="(\d)" label="(\w+)"[^>]*>\s*<values[^>]*>([^<]*)</values>', coherent)
        for _index, label, values in grids:
            if label == "temperature":
                elastic_t = _floats(values)
            elif label == "energy_in":
                tape["bragg_edges"] = len(_floats(values))
    if incoherent:
        w = re.search(r"<values[^>]*>([^<]*)</values>", incoherent)
        if w and not elastic_t:
            elastic_t = _floats(w.group(1))[0::2]
    tape["elastic"] = ("both" if coherent and incoherent else "coherent" if coherent
                       else "incoherent" if incoherent else None)
    tape["elastic_temperatures"] = elastic_t
    if inelastic:
        head = re.search(r"<thermalNeutronScatteringLaw_incoherentInelastic\b([^>]*)>", inelastic)
        hattrs = dict(re.findall(r'(\w+)="([^"]*)"', head.group(1))) if head else {}
        tape["lat"] = 1 if hattrs.get("calculatedAtThermal") == "true" else 0
        atoms = re.findall(r'<scatteringAtom pid="([^"]+)" numberPerMolecule="([^"]+)"([^>]*)>', inelastic)
        masses = re.findall(r'<mass value="([^"]+)" unit="amu"', inelastic)
        principal_index = next((i for i, a in enumerate(atoms) if 'primaryScatterer="true"' in a[2]), 0)
        if atoms:
            tape["principal_atoms"] = float(atoms[principal_index][1])
        if masses and len(masses) > principal_index:
            tape["principal_awr"] = float(masses[principal_index]) / 1.00866491595
        kernel = _section(inelastic, "selfScatteringKernel") or ""
        symmetric = re.search(r'<selfScatteringKernel symmetric="(\w+)"', inelastic)
        if symmetric:
            tape["lasym"] = 0 if symmetric.group(1) == "true" else 1
        grids = re.findall(r'<grid index="(\d)" label="(\w+)"[^>]*>\s*<values[^>]*>([^<]*)</values>', kernel)
        for _index, label, values in grids:
            if label == "temperature":
                tape["temperatures"] = _floats(values)
            elif label == "beta":
                tape["n_beta"] = len(_floats(values))
            elif label == "alpha":
                tape["n_alpha"] = len(_floats(values))
        tape["secondaries"] = [{"law": None, "free_xs": None, "awr": None, "atoms": float(a[1])}
                               for i, a in enumerate(atoms) if i != principal_index]
    naming = tsl_naming(_gnds_name(path.name), zsymam)
    return TSLTape(path=str(path), file_name=path.name, size=path.stat().st_size,
                   mat=int(mat_match.group(1)) if mat_match else 0, za=0.0, awr=0.0, nsub=12,
                   zsymam=zsymam, name=naming.name, principal=naming.principal,
                   compound=naming.compound, variant=naming.variant, name_source=naming.source,
                   lab=None, author=author.group(1) if author else None,
                   evaluation_date=eattrs.get("date"), description=description,
                   format="gnds", library=attrs.get("evaluation"), title=attrs.get("target"),
                   **tape)


def ace_tsl_tapes(paths: Sequence[Union[str, Path]]) -> List:
    """ACE thermal tables grouped into materials: one entry per material, all its temperatures."""
    from .ace import read_ace_thermal_header
    from .library import TSLTape

    groups: Dict[tuple, list] = defaultdict(list)
    for path in paths:
        header = read_ace_thermal_header(path)
        stem = header.zaid.rsplit(".", 1)[0]
        groups[(str(Path(path).parent), stem, header.name)].append((header, Path(path)))
    tapes = []
    for (_parent, stem, name), members in groups.items():
        members.sort(key=lambda m: m[0].temperature)
        header = members[0][0]
        naming = tsl_naming_from_title(name, stem)
        mode = header.elastic_mode
        temperatures = [m[0].temperature for m in members]
        tapes.append(TSLTape(
            path=str(members[0][1]), file_name=f"{stem}.*t" if len(members) > 1 else members[0][1].name,
            size=sum(p.stat().st_size for _h, p in members), mat=_mat(header.mat), za=0.0,
            awr=header.awr, nsub=12, zsymam=None, name=stem, principal=naming.principal,
            compound=naming.compound, variant=naming.variant, name_source=naming.source,
            evaluation_date=header.date, description=[header.title],
            elastic={"coherent": "coherent", "incoherent": "incoherent", "mixed": "both"}.get(mode or ""),
            elastic_temperatures=temperatures if mode else [], temperatures=temperatures,
            principal_awr=header.awr, format="ace", library=header.library, title=name,
            paths=[str(p) for _h, p in members], member=stem,
            details={"zaids": [h.zaid for h, _p in members], "nuclides": header.izaw,
                     "outgoing": header.outgoing_kind, "elastic_mode": mode}))
    return tapes


def _mat(text: str) -> int:
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) if match else 0


def openmc_tsl_header(path: Union[str, Path]):
    """An OpenMC thermal HDF5 file as a library lists it (attributes and temperatures only)."""
    from .library import TSLTape
    from .openmc import K_EV, _h5py, _text

    path = Path(path)
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        group_name = next(iter(f.keys()))
        group = f[group_name]
        awr = float(group.attrs.get("atomic_weight_ratio", 0.0))
        nuclides = [_text(n) for n in group.attrs.get("nuclides", [])]
        temperatures, elastic_t, kinds = [], [], set()
        bragg = None
        for key in group["kTs"]:
            t = round(float(group["kTs"][key][()]) / K_EV, 1)
            temperatures.append(t)
            node = group[key]
            if "elastic" in node:
                elastic_t.append(t)
                xs = node["elastic"]["xs"]
                kind = _text(xs.attrs.get("type", b""))
                if kind == "Sum":
                    kinds.update(_text(xs[k].attrs.get("type", b"")) for k in xs)
                else:
                    kinds.add(kind)
                if bragg is None and "CoherentElastic" in kinds:
                    data = xs if kind == "CoherentElastic" else next(
                        xs[k] for k in xs if _text(xs[k].attrs.get("type", b"")) == "CoherentElastic")
                    bragg = int(data.shape[1])
    coherent = "CoherentElastic" in kinds
    incoherent = bool(kinds & {"IncoherentElastic", "Tabulated1D"})
    naming = tsl_naming_from_openmc(group_name)
    order = sorted(range(len(temperatures)), key=lambda i: temperatures[i])
    return TSLTape(
        path=str(path), file_name=path.name, size=path.stat().st_size, mat=0, za=0.0, awr=awr,
        nsub=12, zsymam=None, name=naming.name, principal=naming.principal,
        compound=naming.compound, variant=naming.variant, name_source=naming.source,
        elastic="both" if coherent and incoherent else "coherent" if coherent
        else "incoherent" if incoherent else None,
        elastic_temperatures=sorted(elastic_t), bragg_edges=bragg,
        temperatures=[temperatures[i] for i in order], principal_awr=awr,
        format="openmc", title=group_name, member=group_name,
        details={"nuclides": nuclides})


def g4ndl_tsl_tapes(root: Union[str, Path]) -> List:
    """Every material of a G4NDL ``ThermalScattering`` folder."""
    from .g4ndl import _bragg_blocks, _pairs_blocks, g4ndl_thermal_naming, list_g4ndl_thermal
    from .library import TSLTape

    root = Path(root)
    tapes = []
    for name, files in list_g4ndl_thermal(root).items():
        _stem, principal, compound, variant = g4ndl_thermal_naming(name)
        temperatures: List[float] = []
        elastic_t: List[float] = []
        source = ""
        bragg = None
        if "Inelastic" in files:
            source, blocks = _pairs_blocks(files["Inelastic"])
            temperatures = [b[0] for b in blocks]
        if "Incoherent" in files:
            source, blocks = _pairs_blocks(files["Incoherent"])
            elastic_t = [b[0] for b in blocks]
        if "Coherent" in files:
            fs = root / "Coherent" / "FS" / f"{name}.z"
            if fs.is_file():
                blocks = _bragg_blocks(fs)
                elastic_t = sorted(set(elastic_t) | {b[0] for b in blocks})
                bragg = int(blocks[0][1].edges.size) if blocks else None
        coherent, incoherent = "Coherent" in files, "Incoherent" in files
        size = sum(p.stat().st_size for process in files
                   for p in (root / process / "CrossSection" / f"{name}.z",
                             root / process / "FS" / f"{name}.z") if p.is_file())
        tapes.append(TSLTape(
            path=str(root), file_name=name, size=size, mat=0, za=0.0, awr=0.0, nsub=12,
            zsymam=None, name=name, principal=principal, compound=compound, variant=variant,
            name_source="g4ndl",
            elastic="both" if coherent and incoherent else "coherent" if coherent
            else "incoherent" if incoherent else None,
            elastic_temperatures=elastic_t, bragg_edges=bragg,
            temperatures=temperatures or elastic_t, format="g4ndl", library=source or None,
            title=name, member=name, details={"processes": sorted(files)}))
    return tapes
