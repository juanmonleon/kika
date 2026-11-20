"""The photo-atomic processing entry point: three evaluations of an element → its EPRDATA table.

An EPRDATA table is the photon interactions, the atomic relaxation and the
electron interactions of one element, so the three ENDF sublibraries are all
required: photoat (NSUB=3), atomic_relax (NSUB=6) and electrons (NSUB=113).
:func:`process_photoatomic` reads them (paths or decoded models), builds every
block (:func:`~kika.ace.processing.photoatomic.photoatomicTable`,
:func:`~kika.ace.processing.photoatomic.electrons.electronBlocks`) and returns a
:class:`PhotoatomicProcessing` whose :meth:`~PhotoatomicProcessing.write_ace`
writes the table. There are no numerical options: the method is fixed.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional

__all__ = ["PhotoatomicProcessing", "process_photoatomic"]


def _model(source, what: str):
    import kika

    if isinstance(source, (str, Path)):
        return kika.read(str(source))
    if source is None:
        raise ValueError(f"process_photoatomic needs the {what} evaluation: an EPRDATA table is photons, "
                         f"atomic relaxation and electrons together")
    return source


def _z(model) -> Optional[int]:
    from kika.nuclear_data.model import PoPs

    if isinstance(model, PoPs):
        elements = [e for e in model.chemicalElements.values() if getattr(e, "atomic", None) is not None]
        return elements[0].Z if elements else None
    element = model.PoPs.chemicalElements.get(model.target) if model.PoPs is not None else None
    if element is not None:
        return element.Z
    particle = model.PoPs.particles.get(model.target) if model.PoPs is not None else None
    return getattr(particle, "Z", None)


class PhotoatomicProcessing:
    """The EPRDATA table of an element, built; ``write_ace`` writes it.

    ``table`` is the ``PhotoatomicAce`` (every
    block decoded, energies in eV); ``notes`` lists what the build decided that a
    user should know (Compton shells joined or dropped, transitions whose energy
    from the binding energies is not positive).

    :func:`~kika.endf.process_photoatomic` returns one.

    Parameters
    ----------
    table : PhotoatomicAce
        The table, with energies in eV.
    notes : list of str
        The messages of the build.
    """

    def __init__(self, table, notes: List[str]):
        self.table = table
        self.notes = notes

    def write_ace(self, directory, *, extension: Optional[str] = None, xsdir: bool = True,
                  replace: bool = False) -> Path:
        """Write the table as ``<1000 Z><extension>`` (``26000.00p``) into ``directory``.

        Every word as ``%20.12E``, the EPRDATA layout (NXS(6) = 3). The file is read
        back in strict mode and compared word for word before it replaces one, and
        its line is added to ``directory/xsdir`` unless ``xsdir`` is false. The table
        file is written first and replaces any file of the same name; ``replace``
        governs only the xsdir entry.

        Parameters
        ----------
        directory
            Output folder, created if missing.
        extension
            Suffix of the table, with or without the leading dot (``"00p"`` or
            ``".00p"``). It is stored on ``table.header``, so it stays the
            suffix of later calls. ``None`` keeps the table's own (``.00p``).
        xsdir
            Add the line of the table to ``directory/xsdir``, created if
            missing.
        replace
            When the table name is already in the xsdir, ``False`` refuses and
            ``True`` replaces that entry. It does nothing when ``xsdir`` is
            false.

        Returns
        -------
        pathlib.Path
            The table file written, in *directory*.

        Raises
        ------
        ~kika.endf.ProcessingOutputError
            ``ace-table-exists`` when the xsdir already names the table and
            *replace* is false (the table file has then been written, and
            only the xsdir is unchanged); ``ace-failed`` for any other
            failure.
        """
        from kika.ace.encoders.photoatomic import encode_photoatomic
        from kika.ace.writers.write_ace import write_ace_checked
        from kika.ace.xsdir import XsdirEntryExists, add_xsdir_entry, build_xsdir_line
        from kika.endf.process import ProcessingOutputError

        directory = Path(directory)
        table = self.table
        if extension is not None:
            table.header.extension = extension if extension.startswith(".") else "." + extension
        try:
            ace = encode_photoatomic(table)
            name = f"{ace.header.zaid}{ace.header.extension}"
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / name
            write_ace_checked(ace, target)
        except Exception as error:
            raise ProcessingOutputError("ace-failed", f"{type(error).__name__}: {error}") from error
        if xsdir:
            line = build_xsdir_line(name, ace.header.atomic_weight_ratio, name, int(ace.header.nxs_array[1]), 0.0)
            try:
                add_xsdir_entry(directory / "xsdir", line, replace=replace)
            except XsdirEntryExists as error:
                raise ProcessingOutputError("ace-table-exists", str(error)) from error
        return target


def process_photoatomic(photoat, relaxation, electrons) -> PhotoatomicProcessing:
    """The EPRDATA table of an element from its three ENDF evaluations.

    Parameters
    ----------
    photoat, relaxation, electrons
        The photo-atomic (NSUB=3), atomic-relaxation (NSUB=6) and electro-atomic
        (NSUB=113) tapes, each a path or the model that ``kika.read`` returns for
        it, all of the same element.

    Returns
    -------
    PhotoatomicProcessing
        The built table and the notes of the build; ``write_ace`` writes it.

    Raises
    ------
    ValueError
        When one evaluation is ``None``, is not the sublibrary it should be, or
        the three are not of one element.
    """
    from kika import __version__
    from kika.nuclear_data.model import ATOMIC_INTERACTION, PoPs

    from .electrons import electronBlocks
    from .tables import photoatomicTable

    photo = _model(photoat, "photo-atomic")
    relax = _model(relaxation, "atomic relaxation")
    elec = _model(electrons, "electro-atomic")
    if getattr(photo, "interaction", None) != ATOMIC_INTERACTION or photo.projectile != "photon":
        raise ValueError("photoat is not a photo-atomic evaluation (NSUB=3)")
    if not isinstance(relax, PoPs) or not relax.chemicalElements:
        raise ValueError("relaxation is not an atomic relaxation evaluation (NSUB=6)")
    if getattr(elec, "interaction", None) != ATOMIC_INTERACTION or elec.projectile != "e-":
        raise ValueError("electrons is not an electro-atomic evaluation (NSUB=113)")
    zs = {_z(photo), _z(relax), _z(elec)}
    if len(zs) != 1 or None in zs:
        raise ValueError(f"the three evaluations are not of one element (Z = {sorted(z for z in zs if z)})")
    comment = f"{photo.target} photo-atomic, relaxation and electro-atomic data (kika {__version__})"
    table, notes = photoatomicTable(photo, relax, extension=".00p", comment=comment[:70],
                                    date=datetime.date.today().strftime("%d/%m/%y"))
    table.electron_nxs, table.electron_words, table.electron_jxs = electronBlocks(elec)
    messages = list(notes.compton)
    if notes.nonPositiveTransitions:
        messages.append(f"{notes.nonPositiveTransitions} relaxation transition(s) have an energy <= 0 from the "
                        f"binding energies; written as they come out")
    return PhotoatomicProcessing(table, messages)
