"""
NJOY Input Deck builder.

Provides the fluent ``InputDeck`` class for notebook/scripting use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .modules import GENERATORS, generate_module
from .viewr_defs import ViewrPlot
from .plotr_defs import PlotrJob


#: The parameters that name a tape unit, whichever module carries them.
_TAPE_KEYS = frozenset({
    "nin", "nout", "nendf", "npend", "ngout", "ngout1", "ngout2", "ngend", "ngendf",
    "nace", "ndir", "nplot", "nmatx",
})


class InputDeck:
    """Builder for NJOY input decks — add modules, then render or save.

    Usage::

        deck = InputDeck()
        deck.moder(nin=20, nout=-25)
        deck.reconr(isotope="U235", err=0.001)
        deck.broadr(isotope="U235", temperatures=[293.6])
        deck.acer(isotope="U235", tempd=293.6, suff=0.03)
        print(deck)
        deck.save("u235.inp")
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._modules: list[str] = []
        # Auto-chaining state
        self._endf_tape: int | None = None
        self._pendf_tape: int | None = None
        self._gendf_tape: int | None = None
        self._leapr_tape: int | None = None
        self._plotr_tape: int | None = None
        self._next_tape: int = -20
        # Every unit a module has named, so that an automatic tape never lands
        # on one the caller chose: moder(nout=-21) and then reconr() would
        # otherwise both get -21, and RECONR would read and write one tape.
        self._used_units: set[int] = set()

    # ------------------------------------------------------------------
    # Auto-chaining helpers
    # ------------------------------------------------------------------

    def _alloc_tape(self) -> int:
        """Allocate the next available tape number (negative = binary)."""
        self._next_tape -= 1
        while abs(self._next_tape) in self._used_units:
            self._next_tape -= 1
        self._used_units.add(abs(self._next_tape))
        return self._next_tape

    def _resolve_input(self, explicit: int | None, attr: str, label: str) -> int:
        """Resolve an input tape: explicit value > tracked state > error."""
        if explicit is not None:
            return explicit
        tracked = getattr(self, attr)
        if tracked is not None:
            return tracked
        raise ValueError(
            f"No {label} tape available. Call the preceding module first "
            f"or pass the tape number explicitly."
        )

    # ------------------------------------------------------------------
    # Module methods — each maps friendly kwargs → internal param dict
    # ------------------------------------------------------------------

    def moder(self, *, nin: int = 20, nout: int | None = None) -> InputDeck:
        """Add a MODER step, which converts a tape between ASCII and binary.

        Parameters
        ----------
        nin
            Input unit, an NJOY unit number (negative for a binary tape, positive
            for an ASCII one).
        nout
            Output unit. None allocates the next free unit counting down from
            -21, skipping the units the deck already names through the tape
            parameters of earlier steps (``nin``, ``nout``, ``nendf``, ``npend``,
            ``ngout``, ``ngout1``, ``ngout2``, ``ngend``, ``ngendf``, ``nace``,
            ``ndir``, ``nplot`` and ``nmatx``). Units given as ``ngam1``,
            ``ngam2``, ``ngen1`` to ``ngen8``, ``nplt``, ``nplt0``, ``infile``
            or ``nps`` are not recorded, so an allocated unit can coincide with
            one of them. Later steps take this tape as their ENDF tape.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.
        """
        nout = nout if nout is not None else self._alloc_tape()
        self._append("MODER", {"nin": nin, "nout": nout})
        self._endf_tape = nout
        return self

    def reconr(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        isotope: str = "U235",
        err: float = 0.001,
        tempr: float | None = None,
        errmax: float | None = None,
        errint: float | None = None,
    ) -> InputDeck:
        """Add a RECONR step, which reconstructs the pointwise cross sections.

        Writes one material. A tolerance left out is not written, so NJOY applies
        its own default.

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        npend
            Output PENDF tape. None allocates the next free unit (see ``moder``).
            Later steps take it as their PENDF tape.
        isotope
            Isotope name as ``load_isotopes`` spells it (``"U235"``), turned into
            its MAT number. An unknown name gives the MAT of U-235 without an
            error.
        err
            Reconstruction tolerance, as a fraction.
        tempr
            Reconstruction temperature in K. None writes 0 when ``errmax`` or
            ``errint`` is given and nothing otherwise.
        errmax, errint
            NJOY's ``errmax`` and ``errint`` (fractions). ``errint`` given alone
            writes ``errmax`` as 10 times ``err``.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* is None and no ``moder`` step comes before.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        npend = npend if npend is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "mat": isotope,
            "err": str(err),
        }
        if tempr is not None:
            p["tempr"] = tempr
        if errmax is not None:
            p["errmax"] = errmax
        if errint is not None:
            p["errint"] = errint
        self._append("RECONR", p)
        self._pendf_tape = npend
        return self

    def broadr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        isotope: str = "U235",
        temperatures: list[float] | str = "293.6",
        errthn: float = 0.001,
        thnmax: float | None = None,
        errmax: float | None = None,
        errint: float | None = None,
        istart: str | None = None,
        istrap: str | None = None,
        temp1: float | None = None,
    ) -> InputDeck:
        """Add a BROADR step, which Doppler-broadens the cross sections.

        An optional value left out is not written, so NJOY applies its own default.
        The optional fields of the cards are written in NJOY's order, so giving a
        later one also writes the earlier ones (with the values below).

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        nin
            Input PENDF tape. None takes the output of the last step that wrote one.
        nout
            Output PENDF tape. None allocates the next free unit (see ``moder``).
            Later steps take it as their PENDF tape.
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235).
        temperatures
            Final temperatures in K: a list, or a string of values separated by
            spaces. They are written as ``str`` gives them.
        errthn
            Thinning tolerance, as a fraction.
        thnmax
            Upper energy limit of the thinning, in eV. Written when given, or as 1
            when ``errmax`` or ``errint`` is given without it.
        errmax, errint
            NJOY's ``errmax`` and ``errint`` (fractions). ``errint`` given alone
            writes ``errmax`` as 10 times ``errthn``.
        istart, istrap
            NJOY's ``istart`` and ``istrap`` options, ``"no"`` or ``"yes"``, written
            as 0 or 1. Giving ``temp1`` writes both (0 for the one left out) and
            giving ``istrap`` writes ``istart`` too.
        temp1
            NJOY's ``temp1``, in K.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* or *nin* is None and no earlier step provides that tape.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "mat": isotope,
            "temp2": temp_str,
            "errthn": str(errthn),
        }
        if thnmax is not None:
            p["thnmax"] = thnmax
        if errmax is not None:
            p["errmax"] = errmax
        if errint is not None:
            p["errint"] = errint
        if istart is not None:
            p["istart"] = istart
        if istrap is not None:
            p["istrap"] = istrap
        if temp1 is not None:
            p["temp1"] = temp1
        self._append("BROADR", p)
        self._pendf_tape = nout
        return self

    def heatr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        nplot: int = 0,
        isotope: str = "U235",
        mtk: list[int] | str = "302",
        ntemp: int = 0,
        local: str | None = None,
        iprint: str | None = None,
        ed: float | None = None,
    ) -> InputDeck:
        """Add a HEATR step, which computes heating (KERMA) and damage energies.

        An optional value left out is not written, so NJOY applies its own default.
        The optional fields of the card are written in NJOY's order, so giving a
        later one also writes the earlier ones: ``ed``, ``iprint`` or ``local``
        writes ``nqa`` as 0 and ``ntemp`` (0 if not given), and ``ed`` or
        ``iprint`` also writes ``local`` (0 if not given); ``ed`` also writes
        ``iprint`` (0 if not given).

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        nin
            Input PENDF tape. None takes the output of the last step that wrote one.
        nout
            Output PENDF tape. None allocates the next free unit (see ``moder``).
            Later steps take it as their PENDF tape.
        nplot
            Plot tape unit, always written (0 for none).
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235).
        mtk
            MT numbers of the partial heating reactions: a list, or a string of
            numbers separated by spaces. Their count is written as ``npk``.
        ntemp
            NJOY's ``ntemp``. Written when it is not 0, and as 0 when ``ed``,
            ``iprint`` or ``local`` is given.
        local
            ``"Transported"`` (written as 0) or ``"Deposited"`` (1), NJOY's
            ``local`` option. Any other value is not written, except that 0 is
            written when ``ed`` or ``iprint`` is given.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer.
        ed
            NJOY's ``ed``, the displacement energy.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* or *nin* is None and no earlier step provides that tape.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(mtk, list):
            mtk_str = " ".join(str(m) for m in mtk)
        else:
            mtk_str = str(mtk)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "nplot": str(nplot),
            "matd": isotope,
            "mtk": mtk_str,
        }
        if ntemp:
            p["ntemp"] = ntemp
        if local is not None:
            p["local"] = local
        if iprint is not None:
            p["iprint"] = iprint
        if ed is not None:
            p["ed"] = ed
        self._append("HEATR", p)
        self._pendf_tape = nout
        return self

    def thermr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        matde: int = 0,
        isotope: str = "U235",
        nbin: int = 8,
        iin: int = 2,
        icoh: int = 0,
        iform: int | None = None,
        natom: int | None = None,
        mtref: int | None = None,
        temperatures: list[float] | str = "293.6",
        tol: float = 0.01,
        emax: float = 4.0,
        iprint: str | None = None,
    ) -> InputDeck:
        """Add a THERMR step, which adds thermal scattering cross sections.

        ``nbin``, ``iin``, ``icoh``, ``tol`` and ``emax`` are always written, with
        the defaults below (which are kika's). The other optional fields are
        written in NJOY's order, so giving a later one writes the earlier ones
        (``iform`` as 0, ``natom`` as 1, ``mtref`` as 222).

        Parameters
        ----------
        nendf
            Tape holding the thermal scattering law. None writes 0 (free gas) when
            ``matde`` is 0, and otherwise takes the output of the last ``leapr``
            step.
        nin
            Input PENDF tape. None takes the output of the last step that wrote one.
        nout
            Output PENDF tape. None allocates the next free unit (see ``moder``).
            Later steps take it as their PENDF tape.
        matde
            MAT number of the scatterer on *nendf*; 0 for a free gas.
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into the MAT number
            of the material on the PENDF tape (an unknown name gives the MAT of
            U-235).
        nbin
            Number of equiprobable angles.
        iin, icoh
            NJOY's inelastic and elastic options.
        iform, natom, mtref
            NJOY's ``iform``, ``natom`` and ``mtref``.
        temperatures
            Temperatures in K: a list, or a string of values separated by spaces.
        tol
            Tolerance, as a fraction.
        emax
            Upper energy limit of the thermal treatment, in eV.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nin* is None and no earlier step provides a PENDF tape, or if
            *nendf* is None, ``matde`` is not 0 and no ``leapr`` step comes before.
        """
        # nendf is the thermal/LEAPR tape; 0 means free-gas
        if nendf is None:
            if matde == 0:
                nendf = 0
            else:
                nendf = self._resolve_input(None, '_leapr_tape', 'LEAPR/thermal')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "matde": matde,
            "matdp": isotope,
            "nbin": nbin,
            "iin": iin,
            "icoh": icoh,
            "temperatures": temp_str,
            "tol": tol,
            "emax": emax,
        }
        if iform is not None:
            p["iform"] = iform
        if natom is not None:
            p["natom"] = natom
        if mtref is not None:
            p["mtref"] = mtref
        if iprint is not None:
            p["iprint"] = iprint
        self._append("THERMR", p)
        self._pendf_tape = nout
        return self

    def purr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        isotope: str = "U235",
        temperatures: list[float] | str = "293.6",
        sigz: list[float] | str = "1e10",
        nbin: int = 20,
        nladr: int = 64,
        iprint: str | None = None,
        nunx: int | None = None,
    ) -> InputDeck:
        """Add a PURR step, which computes unresolved-resonance probability tables.

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        nin
            Input PENDF tape. None takes the output of the last step that wrote one.
        nout
            Output PENDF tape. None allocates the next free unit (see ``moder``).
            Later steps take it as their PENDF tape.
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235).
        temperatures
            Temperatures in K: a list, or a string of values separated by spaces.
        sigz
            Sigma-zero values in barns: a list, or a string of values separated by
            spaces.
        nbin, nladr
            NJOY's number of probability bins and of resonance ladders, always
            written.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. Written
            when given, or as 1 when ``nunx`` is given without it.
        nunx
            NJOY's ``nunx``; written when given.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* or *nin* is None and no earlier step provides that tape.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        if isinstance(sigz, list):
            sigz_str = " ".join(str(s) for s in sigz)
        else:
            sigz_str = str(sigz)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "matd": isotope,
            "temp": temp_str,
            "sigz": sigz_str,
            "nbin": str(nbin),
            "nladr": str(nladr),
        }
        if iprint is not None:
            p["iprint"] = iprint
        if nunx is not None:
            p["nunx"] = nunx
        self._append("PURR", p)
        self._pendf_tape = nout
        return self

    def unresr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
        isotope: str = "U235",
        temperatures: list[float] | str = "293.6",
        sigz: list[float] | str = "1e10",
        iprint: str | None = None,
    ) -> InputDeck:
        """Add an UNRESR step, which computes effective self-shielded cross sections.

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        nin
            Input PENDF tape. None takes the output of the last step that wrote one.
        nout
            Output PENDF tape. None allocates the next free unit (see ``moder``).
            Later steps take it as their PENDF tape.
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235).
        temperatures
            Temperatures in K: a list, or a string of values separated by spaces.
        sigz
            Sigma-zero values in barns: a list, or a string of values separated by
            spaces.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. Not
            written when None.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* or *nin* is None and no earlier step provides that tape.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        if isinstance(sigz, list):
            sigz_str = " ".join(str(s) for s in sigz)
        else:
            sigz_str = str(sigz)
        p: dict[str, Any] = {
            "nendf": nendf,
            "nin": nin,
            "nout": nout,
            "matd": isotope,
            "temp": temp_str,
            "sigz": sigz_str,
        }
        if iprint is not None:
            p["iprint"] = iprint
        self._append("UNRESR", p)
        self._pendf_tape = nout
        return self

    def gaspr(
        self,
        *,
        nendf: int | None = None,
        nin: int | None = None,
        nout: int | None = None,
    ) -> InputDeck:
        """Add a GASPR step, which adds gas-production cross sections.

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        nin
            Input PENDF tape. None takes the output of the last step that wrote one.
        nout
            Output PENDF tape. None allocates the next free unit (see ``moder``).
            Later steps take it as their PENDF tape.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* or *nin* is None and no earlier step provides that tape.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        nin = self._resolve_input(nin, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        self._append("GASPR", {"nendf": nendf, "nin": nin, "nout": nout})
        self._pendf_tape = nout
        return self

    def groupr(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        ngout1: int = 0,
        ngout2: int | None = None,
        isotope: str = "U235",
        ign: int = 3,
        igg: int = 0,
        iwt: int = 3,
        lord: int = 0,
        temperatures: list[float] | str = "293.6",
        sigz: list[float] | str = "1e10",
        title: str | None = None,
        reactions: list[tuple] | None = None,
        egn: list[float] | None = None,
        egg: list[float] | None = None,
        eb: float | None = None,
        tb: float | None = None,
        ec: float | None = None,
        tc: float | None = None,
        iprint: str | None = None,
        ismooth: int | None = None,
    ) -> InputDeck:
        """Add a GROUPR step, which computes multigroup cross sections.

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        npend
            Input PENDF tape. None takes the output of the last step that wrote one.
        ngout1
            Input GENDF tape, 0 for none.
        ngout2
            Output GENDF tape. None allocates the next free unit (see ``moder``).
            ``ccccr``, ``matxsr``, ``wimsr``, ``dtfr`` and ``powr`` steps take it
            as their input.
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235).
        ign, igg
            NJOY's neutron and photon group structures. With ``ign`` 1 the
            boundaries come from ``egn``, and with ``igg`` 1 from ``egg``.
        iwt
            NJOY's weight function option. 0, 1 and negative values are refused;
            4 needs ``eb``, ``tb``, ``ec`` and ``tc``.
        lord
            Legendre order.
        temperatures
            Temperatures in K: a list, or a string of values separated by spaces.
        sigz
            Sigma-zero values in barns: a list, or a string of values separated by
            spaces.
        title
            Title card. None writes ``multigroup data for <isotope>``.
        reactions
            Reaction cards, written once for each temperature: a list of tuples
            ``(mfd,)``, ``(mfd, mtd)`` or ``(mfd, mtd, name)``. None writes the
            single card ``3 /``.
        egn, egg
            Neutron and photon group boundaries in eV, as NJOY reads them (a list,
            or for ``egn`` a string of values separated by spaces). ``egn`` is written
            only when ``ign`` is 1, and when it is None the placeholders ``{NGN} /``
            and ``{EGN} /`` are written for the user to fill in. ``egg`` is written
            only when ``igg`` is 1, and is then required.
        eb, tb, ec, tc
            Parameters of the analytic flux of ``iwt`` 4, in the units NJOY reads.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer.
        ismooth
            NJOY's ``ismooth``. Giving it writes ``iprint`` too (1 if None).

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* or *npend* is None and no earlier step provides that tape;
            if ``iwt`` is 0, 1 or negative; if ``iwt`` is 4 and one of ``eb``,
            ``tb``, ``ec``, ``tc`` is None; or if ``igg`` is 1 and ``egg`` is None.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        npend = self._resolve_input(npend, '_pendf_tape', 'PENDF')
        ngout2 = ngout2 if ngout2 is not None else self._alloc_tape()
        if isinstance(temperatures, list):
            temp_str = " ".join(str(t) for t in temperatures)
        else:
            temp_str = str(temperatures)
        if isinstance(sigz, list):
            sigz_str = " ".join(str(s) for s in sigz)
        else:
            sigz_str = str(sigz)
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "ngout1": ngout1,
            "ngout2": ngout2,
            "matb": isotope,
            "ign": ign,
            "igg": igg,
            "iwt": iwt,
            "lord": lord,
            "temperatures": temp_str,
            "sigz": sigz_str,
        }
        if title is not None:
            p["title"] = title
        if reactions is not None:
            p["reactions"] = reactions
        if egn is not None:
            p["egn"] = egn
        if egg is not None:
            p["egg"] = egg
        if eb is not None:
            p["eb"] = eb
        if tb is not None:
            p["tb"] = tb
        if ec is not None:
            p["ec"] = ec
        if tc is not None:
            p["tc"] = tc
        if iprint is not None:
            p["iprint"] = iprint
        if ismooth is not None:
            p["ismooth"] = ismooth
        self._append("GROUPR", p)
        self._gendf_tape = ngout2
        return self

    def gaminr(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        ngam1: int = 0,
        ngam2: int | None = None,
        isotope: str = "U235",
        igg: int = 7,
        iwt: int = 3,
        lord: int = 0,
        title: str | None = None,
        reactions: list[tuple] | None = None,
        egg: list[float] | None = None,
        iprint: str | None = None,
    ) -> InputDeck:
        """Add a GAMINR step, which computes multigroup photoatomic cross sections.

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        npend
            Input PENDF tape. None takes the output of the last step that wrote one.
        ngam1
            Input GENDF tape, 0 for none.
        ngam2
            Output GENDF tape. None allocates the next free unit (see ``moder``).
            Later ``ccccr``, ``matxsr``, ``wimsr``, ``dtfr`` and ``powr`` steps take
            it as their input.
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235).
        igg
            NJOY's photon group structure. With 1 the boundaries come from ``egg``.
        iwt
            NJOY's weight function option. 1 is refused.
        lord
            Legendre order.
        title
            Title card. None writes ``photoatomic data for <isotope>``.
        reactions
            Reaction cards: a list of tuples ``(mfd,)``, ``(mfd, mtd)`` or
            ``(mfd, mtd, name)``. None writes the single card ``-1 /``.
        egg
            Photon group boundaries in eV, as NJOY reads them; required, and written,
            only when ``igg`` is 1.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. Not
            written when None.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* or *npend* is None and no earlier step provides that tape;
            if ``iwt`` is 1; or if ``igg`` is 1 and ``egg`` is None.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        npend = self._resolve_input(npend, '_pendf_tape', 'PENDF')
        ngam2 = ngam2 if ngam2 is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "ngam1": ngam1,
            "ngam2": ngam2,
            "matb": isotope,
            "igg": igg,
            "iwt": iwt,
            "lord": lord,
        }
        if title is not None:
            p["title"] = title
        if reactions is not None:
            p["reactions"] = reactions
        if egg is not None:
            p["egg"] = egg
        if iprint is not None:
            p["iprint"] = iprint
        self._append("GAMINR", p)
        self._gendf_tape = ngam2
        return self

    def errorr(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        ngout: int = 0,
        nout: int | None = None,
        isotope: str = "U235",
        ign: int = 1,
        iwt: int = 6,
        iprint: str | None = None,
        irelco: int = 1,
        mprint: int = 0,
        tempin: float = 300.0,
        mfcov: int = 33,
        irespr: int | None = None,
        dap: float | None = None,
        egn: list[float] | None = None,
    ) -> InputDeck:
        """Add an ERRORR step, which processes covariance data into multigroup form.

        The step does not update the tapes later steps take by default.

        Parameters
        ----------
        nendf
            Input ENDF tape. None takes the output of the last ``moder`` step.
        npend
            Input PENDF tape. None takes the output of the last step that wrote one.
        ngout
            Input GENDF tape, 0 for none.
        nout
            Output tape. None allocates the next free unit (see ``moder``).
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235).
        ign
            NJOY's neutron group structure. With 1 or 19 the boundaries come from
            ``egn``.
        iwt
            NJOY's weight function option. 4 is refused: the analytic flux
            parameters it needs cannot be passed here.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. None
            writes 1.
        irelco, mprint
            NJOY's ``irelco`` and ``mprint`` options.
        tempin
            NJOY's ``tempin``, in K.
        mfcov
            NJOY's ``mfcov``: the MF number of the covariance file to process.
        irespr, dap
            NJOY's ``irespr`` and ``dap``, written only when given. Giving ``dap``
            writes ``irespr`` as 1 if it is None, and ``legord``, ``ifissp`` and
            ``efmean`` as 0.
        egn
            Group boundaries in eV, as NJOY reads them. Giving them forces ``ign``
            to 1. With ``ign`` 1 or 19 and no ``egn``, the placeholders ``{NGN} /``
            and ``{EGN} /`` are written for the user to fill in.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nendf* or *npend* is None and no earlier step provides that tape,
            or if ``iwt`` is 4.
        """
        nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
        npend = self._resolve_input(npend, '_pendf_tape', 'PENDF')
        nout = nout if nout is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "ngout": ngout,
            "nout": nout,
            "matd": isotope,
            "ign": ign,
            "iwt": iwt,
            "irelco": irelco,
            "mprint": mprint,
            "tempin": tempin,
            "mfcov": mfcov,
        }
        if iprint is not None:
            p["iprint"] = iprint
        if irespr is not None:
            p["irespr"] = irespr
        if dap is not None:
            p["dap"] = dap
        if egn is not None:
            p["egn"] = egn
        self._append("ERRORR", p)
        return self

    def acer(
        self,
        *,
        nendf: int | None = None,
        npend: int | None = None,
        ngend: int = 0,
        nace: int = 40,
        ndir: int = 41,
        isotope: str = "U235",
        tempd: float = 293.6,
        suff: float = 0.03,
        iopt: int = 1,
        itype: int = 1,
        iprint: str | None = None,
        # Fast (iopt=1) params:
        newfor: int | None = None,
        iopp: int | None = None,
        ismooth: int | None = None,
        thin1: float | None = None,
        thin2: float | None = None,
        thin3: float | None = None,
        # Thermal (iopt=2) params:
        tname: str | None = None,
        iza: list[int] | str | None = None,
        mti: int | None = None,
        nbint: int = 16,
        mte: int = 0,
        ielas: int = 0,
        nmix: int = 1,
        emax: float = 1000.0,
        iwt: int = 2,
        # Extra iz,aw pairs:
        nxtra: list[tuple[int, float]] | None = None,
    ) -> InputDeck:
        """Add an ACER step, which writes an ACE file.

        ``iopt`` selects the cards written after the title: 1 fast data, 2 thermal
        data, 3 a material and a temperature card, 4 and 5 a material card, and 7
        and 8 none (NJOY's options for an existing ACE file). The branch is chosen
        by the absolute value of ``iopt``; the signed value is written.

        Parameters
        ----------
        nendf
            Input ENDF tape. For ``iopt`` 7 and 8, 0 when None; otherwise None
            takes the output of the last ``moder`` step.
        npend
            Input PENDF tape. For ``iopt`` 7 and 8, 0 when None; otherwise None
            takes the output of the last step that wrote one.
        ngend
            Input GENDF tape, 0 for none.
        nace, ndir
            Output ACE tape and output xsdir tape (units 40 and 41 by default).
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235). Not used for ``iopt`` 7 and 8.
        tempd
            Temperature in K, written as it is. Not used for ``iopt`` 7 and 8.
        suff
            ACE suffix, as a number: 0.03 is written ``.03``. A positive value is
            truncated to two decimals, a negative one is written as it is and 0
            is written ``.00``.
        iopt
            NJOY's ACER option (see above). Another value raises ``ValueError``.
        itype
            NJOY's ``itype``: the ACE file type.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. None
            writes 1.
        newfor, iopp, ismooth
            Fast data (``iopt`` 1) only: NJOY's ``newfor``, ``iopp`` and ``ismooth``,
            written as 1 when None.
        thin1, thin2, thin3
            Fast data only: the three thinning values of NJOY. With ``thin1`` None the
            card is written empty and NJOY applies its defaults.
        tname
            Thermal data (``iopt`` 2) only: the ZAID name of the thermal table.
            Required for ``iopt`` 2.
        iza
            Thermal data only: the ZA values of the moderator, as a list, a string
            of values separated by spaces, or one value. Required for ``iopt`` 2.
        mti
            Thermal data only: the MT of the inelastic reaction. Required for
            ``iopt`` 2.
        nbint, mte, ielas, nmix, emax, iwt
            Thermal data only: the fields of NJOY's thermal card after ``mti``
            (``emax`` in eV). They are written only when one of them differs from
            its default (16, 0, 0, 1, 1000.0, 2), and then every field up to the last
            one that differs is written.
        nxtra
            Pairs ``(iz, aw)`` of extra nuclides. Their count goes on the option
            card and the pairs on a card after the title.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If ``iopt`` is not 7 or 8 and *nendf* or *npend* is None with no earlier
            step to take it from; if ``iopt`` is not 1, 2, 3, 4, 5, 7 or 8 in
            absolute value; or if ``iopt`` is 2 and ``tname``, ``iza`` or ``mti``
            is None.
        """
        abs_iopt = abs(iopt)
        # iopt 7/8 read existing ACE files — no ENDF/PENDF needed
        if abs_iopt in (7, 8):
            if nendf is None:
                nendf = 0
            if npend is None:
                npend = 0
        else:
            nendf = self._resolve_input(nendf, '_endf_tape', 'ENDF')
            npend = self._resolve_input(npend, '_pendf_tape', 'PENDF')
        p: dict[str, Any] = {
            "nendf": nendf,
            "npend": npend,
            "ngend": ngend,
            "nace": nace,
            "ndir": ndir,
            "matd": isotope,
            "tempd": tempd,
            "suff": suff,
            "iopt": str(iopt),
            "itype": itype,
        }
        if iprint is not None:
            p["iprint"] = iprint
        # Fast params (iopt=1)
        if newfor is not None:
            p["newfor"] = newfor
        if iopp is not None:
            p["iopp"] = iopp
        if ismooth is not None:
            p["ismooth"] = ismooth
        if thin1 is not None:
            p["thin1"] = thin1
        if thin2 is not None:
            p["thin2"] = thin2
        if thin3 is not None:
            p["thin3"] = thin3
        # Thermal params — only add when provided or iopt=2
        if tname is not None:
            p["tname"] = tname
        if iza is not None:
            p["iza"] = iza
        if mti is not None:
            p["mti"] = mti
        if nbint != 16:
            p["nbint"] = nbint
        if mte != 0:
            p["mte"] = mte
        if ielas != 0:
            p["ielas"] = ielas
        if nmix != 1:
            p["nmix"] = nmix
        if emax != 1000.0:
            p["emax"] = emax
        if iwt != 2:
            p["iwt"] = iwt
        # Extra iz,aw pairs
        if nxtra is not None:
            p["nxtra"] = nxtra
        self._append("ACER", p)
        return self

    def covr(
        self,
        *,
        nin: int,
        nout: int = 0,
        nplot: int = 0,
        cases: list[list[int]] | None = None,
        # Plot mode params
        icolor: int = 0,
        tlev: list[float] | None = None,
        epmin: float | None = None,
        irelco: int = 1,
        noleg: int = 0,
        nstart: int = 1,
        ndiv: int = 1,
        # Library mode params
        matype: int = 3,
        hlibid: str = "",
        hdescr: str = "",
    ) -> InputDeck:
        """Add a COVR step, which plots or formats the covariance data of ERRORR.

        ``nout`` selects the mode: a positive value writes a covariance library
        (library mode, with ``matype``, ``hlibid`` and ``hdescr``); 0 or a negative
        value writes plots (plot mode, with ``nplot`` and the options after it). The
        options of the other mode are ignored.

        Parameters
        ----------
        nin
            Input ERRORR tape. There is no default.
        nout
            Output library tape (library mode), or 0 for plot mode.
        nplot
            Plot mode only: the plot tape unit.
        cases
            One card per entry, each a list of numbers written as given. None
            writes one card holding the placeholder ``{MAT}`` for the user to
            fill in.
        icolor
            Plot mode only: NJOY's colour option. With 2, ``tlev`` gives the
            colour levels.
        tlev
            Plot mode only: the levels, used when ``icolor`` is 2.
        epmin
            Plot mode only: NJOY's ``epmin``, written unchanged (NJOY reads it in
            eV); ``0.0`` is written when None.
        irelco, noleg, nstart, ndiv
            Plot mode only: NJOY's options of the same name.
        matype
            Library mode only: NJOY's library type.
        hlibid, hdescr
            Library mode only: the library identifier and description, written
            between quotes. Empty by default.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.
        """
        p: dict[str, Any] = {"nin": nin, "nout": nout}
        if cases is not None:
            p["cases"] = cases
        if nout > 0:
            # Library mode
            p["matype"] = matype
            p["hlibid"] = hlibid
            p["hdescr"] = hdescr
        else:
            # Plot mode
            p["nplot"] = nplot
            p["icolor"] = icolor
            if tlev is not None:
                p["tlev"] = tlev
            if epmin is not None:
                p["epmin"] = epmin
            p["irelco"] = irelco
            p["noleg"] = noleg
            p["nstart"] = nstart
            p["ndiv"] = ndiv
        self._append("COVR", p)
        return self

    def leapr(
        self,
        *,
        mat: int,
        za: int,
        awr: float,
        spr: float,
        alphas: list[float],
        betas: list[float],
        temperatures: list[dict],
        nout: int | None = None,
        title: str = "thermal scattering law via leapr",
        npr: int = 1,
        iel: int = 0,
        ncold: int = 0,
        nsk: int = 0,
        nss: int = 0,
        b7: int = 0,
        aws: float = 0.0,
        sps: float = 0.0,
        mss: int = 0,
        lat: int = 1,
        isabt: int = 0,
        ilog: int = 0,
        smin: float | None = None,
        nphon: int = 100,
        iprint: str | None = None,
        secondary_temperatures: list[dict] | None = None,
        comments: list[str] | None = None,
    ) -> InputDeck:
        """Add a LEAPR step, which builds a thermal scattering law from a spectrum.

        The fields of the cards from ``npr`` on follow NJOY's order and an optional
        one is written only when it or a later one differs from its default
        (written as the signature shows), so NJOY applies the same default
        otherwise. The material, ZA and grids have no default.

        Parameters
        ----------
        mat
            MAT number (an integer, not a name) of the thermal material.
        za
            ZA of the principal scatterer.
        awr
            Atomic weight ratio of the principal scatterer.
        spr
            Free-atom cross section of the principal scatterer, in barns.
        alphas, betas
            The alpha and beta grids, in NJOY's dimensionless units. Their lengths
            are written as ``nalpha`` and ``nbeta``.
        temperatures
            One dict per temperature of the principal scatterer, in order. The
            keys are ``temp`` (K; a negative value reuses the previous spectrum and
            ends the dict), ``delta``, ``ni``, ``rho`` (a list), ``twt``, ``c``,
            ``tbeta``; optionally ``nd`` with ``energies`` and ``weights`` when
            ``nd`` is above 0; and, when ``nsk`` is not 0, ``nka``, ``dka``,
            ``kappas`` (when ``nka`` is above 0) and, for ``nsk`` 2, ``cfrac``.
        nout
            Output tape. None allocates the next free unit (see ``moder``). A later
            ``thermr`` step takes it as its thermal tape.
        title
            Title card.
        npr, iel, ncold, nsk
            NJOY's fields of those names.
        nss, b7, aws, sps, mss
            NJOY's secondary-scatterer fields.
        lat, isabt, ilog, smin
            NJOY's fields of those names. ``smin`` is written when given.
        nphon
            NJOY's number of phonon terms.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. None
            writes 0.
        secondary_temperatures
            Same dicts as *temperatures*, for the secondary scatterer. Written
            only when ``nss`` is above 0 and ``b7`` is 0.
        comments
            Accepted and not written.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.
        """
        nout = nout if nout is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "nout": nout,
            "title": title,
            "mat": mat,
            "za": za,
            "awr": awr,
            "spr": spr,
            "alphas": alphas,
            "betas": betas,
            "temperatures": temperatures,
            "npr": npr,
            "iel": iel,
            "ncold": ncold,
            "nsk": nsk,
            "nss": nss,
            "b7": b7,
            "aws": aws,
            "sps": sps,
            "mss": mss,
            "lat": lat,
            "isabt": isabt,
            "ilog": ilog,
            "nphon": nphon,
        }
        if smin is not None:
            p["smin"] = smin
        if iprint is not None:
            p["iprint"] = iprint
        if secondary_temperatures is not None:
            p["secondary_temperatures"] = secondary_temperatures
        if comments is not None:
            p["comments"] = comments
        self._append("LEAPR", p)
        self._leapr_tape = nout
        return self

    def resxsr(
        self,
        *,
        nout: int | None = None,
        materials: list[dict],
        maxt: int = 1,
        efirst: float = 4.0,
        elast: float = 500.0,
        eps: float = 0.001,
        huse: str = "NJOY",
        ivers: int = 0,
        comments: list[str] | None = None,
    ) -> InputDeck:
        """Add a RESXSR step, which writes a resonance cross-section file.

        Parameters
        ----------
        nout
            Output tape. None allocates the next free unit (see ``moder``).
        materials
            One dict per material, with the keys ``isotope`` (a name as
            ``load_isotopes`` spells it, used as the name on the card and turned
            into the MAT number; an unknown name gives the MAT of U-235) and
            ``unit`` (the PENDF tape of that material, written as given).
        maxt
            NJOY's maximum number of temperatures per material.
        efirst, elast
            NJOY's ``efirst`` and ``elast`` fields, written unchanged. NJOY reads
            them as the lower and upper energy limits in eV.
        eps
            NJOY's ``eps`` field, written unchanged. NJOY reads it as the thinning
            tolerance, a fraction.
        huse
            User identification text, written between quotes.
        ivers
            NJOY's version number of the file.
        comments
            Lines of text written between quotes; their count is written on the
            card.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.
        """
        nout = nout if nout is not None else self._alloc_tape()
        mat_list: list[dict] = []
        for m in materials:
            from .isotopes import get_mat_number
            isotope = m["isotope"]
            mat_list.append({
                "name": isotope,
                "mat": get_mat_number(isotope),
                "unit": m["unit"],
            })
        p: dict[str, Any] = {
            "nout": nout,
            "materials": mat_list,
            "maxt": maxt,
            "efirst": efirst,
            "elast": elast,
            "eps": eps,
            "huse": huse,
            "ivers": ivers,
        }
        if comments is not None:
            p["comments"] = comments
        self._append("RESXSR", p)
        return self

    def ccccr(
        self,
        *,
        nin: int | None = None,
        nisot: int = 50,
        nbrks: int = 0,
        ndlay: int = 0,
        isotopes: list[dict],
        ngroup: int,
        maxord: int = 0,
        ifopt: int = 1,
        lprint: int = 0,
        ivers: int = 0,
        huse: str = "",
        hsetid: str = "",
        nggrup: int = 0,
        # ISOTXS params
        nsblok: int = 1,
        maxup: int = 0,
        maxdn: int | None = None,
        ichix: int = -1,
        spec: list[float] | None = None,
        isotxs_params: list[dict] | None = None,
        # BRKOXS params
        nti: int = -1,
        nzi: int = -1,
        atem: list[float] | None = None,
        asig: list[float] | None = None,
    ) -> InputDeck:
        """Add a CCCCR step, which writes CCCC interface files from a GENDF tape.

        Parameters
        ----------
        nin
            Input GENDF tape. None takes the output of the last ``groupr`` or
            ``gaminr`` step.
        nisot, nbrks, ndlay
            Output units of the ISOTXS, BRKOXS and DLAYXS files, 0 for none. The
            ISOTXS cards are written when ``nisot`` is above 0 and the BRKOXS
            cards when ``nbrks`` is above 0.
        isotopes
            One dict per isotope, with the keys ``hisnm``, ``habsid``, ``hident``,
            ``hmat`` (texts), ``imat`` (an integer) and ``xspo``.
        ngroup
            Number of energy groups.
        maxord, ifopt, lprint, ivers, huse, hsetid, nggrup
            NJOY's fields of those names; ``huse`` and ``hsetid`` are text.
        nsblok, maxup, maxdn, ichix
            ISOTXS only: NJOY's fields. ``maxdn`` is ``ngroup`` when None.
        spec
            ISOTXS only: the ``ngroup`` values of the spectrum card, written when
            ``ichix`` is 1 or more.
        isotxs_params
            ISOTXS only: one dict per isotope with the keys ``kbr``, ``amass``,
            ``efiss``, ``ecapt``, ``temp``, ``sigpot`` and ``adens``. None writes
            none.
        nti, nzi
            BRKOXS only: the numbers of temperatures and of sigma-zero values.
        atem, asig
            BRKOXS only: the temperatures and the sigma-zero values, written when
            ``nti`` or ``nzi`` is above 0.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nin* is None and no ``groupr`` or ``gaminr`` step comes before.
        """
        nin = self._resolve_input(nin, '_gendf_tape', 'GENDF')
        p: dict[str, Any] = {
            "nin": nin,
            "nisot": nisot,
            "nbrks": nbrks,
            "ndlay": ndlay,
            "isotopes": isotopes,
            "ngroup": ngroup,
            "nggrup": nggrup,
            "maxord": maxord,
            "ifopt": ifopt,
            "lprint": lprint,
            "ivers": ivers,
            "huse": huse,
            "hsetid": hsetid,
        }
        if nisot > 0:
            p["nsblok"] = nsblok
            p["maxup"] = maxup
            p["maxdn"] = maxdn if maxdn is not None else ngroup
            p["ichix"] = ichix
            if spec is not None:
                p["spec"] = spec
            if isotxs_params is not None:
                p["isotxs_params"] = isotxs_params
            else:
                p["isotxs_params"] = []
        if nbrks > 0:
            p["nti"] = nti
            p["nzi"] = nzi
            if atem is not None:
                p["atem"] = atem
            if asig is not None:
                p["asig"] = asig
        self._append("CCCCR", p)
        return self

    def matxsr(
        self,
        *,
        ngen1: int | None = None,
        ngen2: int = 0,
        nmatx: int | None = None,
        ivers: int = 0,
        huse: str = "",
        hsetid: list[str] | None = None,
        particles: list[dict] | None = None,
        data_types: list[dict] | None = None,
        materials: list[dict] | None = None,
        ngen3: int = 0, ngen4: int = 0, ngen5: int = 0,
        ngen6: int = 0, ngen7: int = 0, ngen8: int = 0,
    ) -> InputDeck:
        """Add a MATXSR step, which writes a MATXS library from GENDF tapes.

        Parameters
        ----------
        ngen1
            First input GENDF tape. None takes the output of the last ``groupr``
            or ``gaminr`` step.
        ngen2, ngen3, ngen4, ngen5, ngen6, ngen7, ngen8
            Further input GENDF tapes, 0 for none. Zeros at the end of ``ngen3`` to
            ``ngen8`` are left off the card.
        nmatx
            Output MATXS tape. None allocates the next free unit (see ``moder``).
        ivers
            NJOY's version number of the file.
        huse
            User identification text, written between quotes.
        hsetid
            Lines of the set identification, written between quotes. None writes
            ``matxs library``.
        particles
            One dict per particle, with the keys ``name`` and ``ngrp`` (its number
            of groups). Required.
        data_types
            One dict per data type, with the keys ``name``, ``jinp`` and
            ``joutp``. Required.
        materials
            One dict per material, with the keys ``hmat``, ``matno`` (a MAT number,
            or an isotope name as ``load_isotopes`` spells it; an unknown name gives
            the MAT of U-235) and, optionally, ``matgg``. Required.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *particles*, *data_types* or *materials* is None, or if *ngen1* is
            None and no ``groupr`` or ``gaminr`` step comes before.
        """
        ngen1 = self._resolve_input(ngen1, '_gendf_tape', 'GENDF')
        nmatx = nmatx if nmatx is not None else self._alloc_tape()
        if particles is None:
            raise ValueError("matxsr() requires 'particles'")
        if data_types is None:
            raise ValueError("matxsr() requires 'data_types'")
        if materials is None:
            raise ValueError("matxsr() requires 'materials'")
        if hsetid is None:
            hsetid = ["matxs library"]
        p: dict[str, Any] = {
            "ngen1": ngen1,
            "ngen2": ngen2,
            "nmatx": nmatx,
            "ivers": ivers,
            "huse": huse,
            "hsetid": hsetid,
            "particles": particles,
            "data_types": data_types,
            "materials": materials,
            "ngen3": ngen3,
            "ngen4": ngen4,
            "ngen5": ngen5,
            "ngen6": ngen6,
            "ngen7": ngen7,
            "ngen8": ngen8,
        }
        self._append("MATXSR", p)
        return self

    def wimsr(
        self,
        *,
        ngendf: int | None = None,
        nout: int | None = None,
        isotope: str = "U235",
        rdfid: float | None = None,
        iverw: int = 4,
        igroup: int = 0,
        # Card 4 options
        ntemp: int = 0,
        nsigz: int = 0,
        sgref: float = 1e10,
        ires: int = 0,
        sigp: float = 0.0,
        mti: int = 0,
        mtc: int = 0,
        ip1opt: int = 1,
        inorf: int = 0,
        isof: int = 0,
        ifprod: int = 0,
        jp1: int = 0,
        # Lambdas
        lambdas: list[float] | None = None,
        # Optional
        iprint: str | None = None,
        burnup: dict | None = None,
        p1flx: list[float] | None = None,
        # Custom groups (igroup=9)
        ngnd: int | None = None,
        nfg: int | None = None,
        nrg: int | None = None,
        igref: int | None = None,
    ) -> InputDeck:
        """Add a WIMSR step, which writes WIMS library data from a GENDF tape.

        Parameters
        ----------
        ngendf
            Input GENDF tape. None takes the output of the last ``groupr`` or
            ``gaminr`` step.
        nout
            Output tape. None allocates the next free unit (see ``moder``).
        isotope
            Isotope name as ``load_isotopes`` spells it, turned into its MAT number
            (an unknown name gives the MAT of U-235).
        rdfid
            NJOY's ``rdfid``. None writes the MAT number.
        iverw, igroup
            NJOY's fields. ``igroup`` 9 selects custom groups and needs ``ngnd``,
            ``nfg`` and ``nrg``.
        ntemp, nsigz, sgref, ires, sigp, mti, mtc, ip1opt, inorf, isof, ifprod, jp1
            NJOY's fields of the material card, always written.
        lambdas
            The Goldstein lambdas: a list, or a string of values separated by
            spaces. None writes 13 zeros.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. None
            writes 0.
        burnup
            A dict that turns on the burnup cards (NJOY's ``iburn`` is written
            as 1). Keys: ``ntis``, ``efiss``, and optionally ``capture`` and
            ``decay`` (each a pair of numbers) and ``fission_products`` (a list of
            such pairs).
        p1flx
            The current spectrum values: a list, or a string of values separated
            by spaces. Required when ``jp1`` is above 0.
        ngnd, nfg, nrg, igref
            Custom groups, used only when ``igroup`` is 9. ``igref`` is optional.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *ngendf* is None and no ``groupr`` or ``gaminr`` step comes before;
            if ``igroup`` is 9 and ``ngnd``, ``nfg`` or ``nrg`` is None; or if
            ``jp1`` is above 0 and ``p1flx`` is None.
        """
        ngendf = self._resolve_input(ngendf, '_gendf_tape', 'GENDF')
        nout = nout if nout is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "ngendf": ngendf,
            "nout": nout,
            "mat": isotope,
            "iverw": iverw,
            "igroup": igroup,
            "ntemp": ntemp,
            "nsigz": nsigz,
            "sgref": sgref,
            "ires": ires,
            "sigp": sigp,
            "mti": mti,
            "mtc": mtc,
            "ip1opt": ip1opt,
            "inorf": inorf,
            "isof": isof,
            "ifprod": ifprod,
            "jp1": jp1,
        }
        if rdfid is not None:
            p["rdfid"] = rdfid
        if lambdas is not None:
            p["lambdas"] = lambdas
        if iprint is not None:
            p["iprint"] = iprint
        if burnup is not None:
            p["iburn"] = 1
            p["burnup"] = burnup
        if p1flx is not None:
            p["p1flx"] = p1flx
        if igroup == 9:
            if ngnd is not None:
                p["ngnd"] = ngnd
            if nfg is not None:
                p["nfg"] = nfg
            if nrg is not None:
                p["nrg"] = nrg
            if igref is not None:
                p["igref"] = igref
        self._append("WIMSR", p)
        return self

    def dtfr(
        self,
        *,
        nin: int | None = None,
        nout: int | None = None,
        npend: int | None = None,
        nplot: int = 0,
        iedit: int = 0,
        nlmax: int = 1,
        ng: int = 30,
        # DTF table params (iedit=0)
        iptotl: int | None = None,
        ipingp: int | None = None,
        itabl: int | None = None,
        edit_names: list[str] | None = None,
        edits: list[list[int]] | None = None,
        ntherm: int = 0,
        mti: int | None = None,
        mtc: int = 0,
        nlc: int = 0,
        # Gamma
        nptabl: int = 0,
        ngp: int = 0,
        # Materials
        materials: list[dict] | None = None,
        # Options
        iprint: str | None = None,
        ifilm: int = 0,
    ) -> InputDeck:
        """Add a DTFR step, which writes DTF-IV transport tables from GENDF data.

        Parameters
        ----------
        nin
            Input GENDF tape. None takes the output of the last ``groupr`` or
            ``gaminr`` step.
        nout
            Output tape, an ASCII (positive) unit. None allocates the next free
            unit (see ``moder``) and takes its absolute value.
        npend
            PENDF tape for point-data plots. None takes the output of the last step
            that wrote one, or 0 if there is none.
        nplot
            Plot tape unit, 0 for none.
        iedit
            0 writes the DTF-table cards; 1 (CLAW) writes only ``nlmax ng /`` in
            their place. Either way the ``nptabl ngp /`` card and the material
            cards follow.
        nlmax, ng
            NJOY's Legendre order and number of groups.
        iptotl, ipingp, itabl
            DTF-table cards (``iedit`` 0 only). None writes ``len(edit_names) +
            3``, ``iptotl + 1`` and ``iptotl + nlmax * ng``.
        edit_names
            The names of the edits (``iedit`` 0 only), written between quotes. Their
            count must be ``iptotl - 3``.
        edits
            One ``[jpos, mt, mult]`` list per edit (``iedit`` 0 only).
        ntherm, mti, mtc, nlc
            Thermal fields (``iedit`` 0 only). ``mti`` is required when ``ntherm``
            is above 0; ``mtc`` and ``nlc`` are then written too.
        nptabl, ngp
            NJOY's gamma-table fields.
        materials
            One dict per material, with the keys ``name``, ``isotope`` (or ``mat``;
            a name as ``load_isotopes`` spells it, or a MAT number), ``jsigz``
            (default 1) and ``dtemp`` (K, default 293.6).
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. None
            writes 0.
        ifilm
            NJOY's ``ifilm``.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *nin* is None and no ``groupr`` or ``gaminr`` step comes before; if
            ``ntherm`` is above 0 and ``mti`` is None; or if the number of
            ``edit_names`` is not ``iptotl - 3``.
        """
        nin = self._resolve_input(nin, '_gendf_tape', 'GENDF')
        # DTF output is coded (ASCII) text — use positive tape number
        if nout is None:
            nout = abs(self._alloc_tape())
        # Optionally use PENDF tape for point-data plots
        if npend is None:
            npend = self._pendf_tape or 0
        p: dict[str, Any] = {
            "nin": nin,
            "nout": nout,
            "npend": npend,
            "nplot": nplot,
            "iedit": iedit,
            "nlmax": nlmax,
            "ng": ng,
            "nptabl": nptabl,
            "ngp": ngp,
            "ifilm": ifilm,
        }
        if iprint is not None:
            p["iprint"] = iprint
        if iedit == 0:
            if iptotl is not None:
                p["iptotl"] = iptotl
            if edit_names is not None:
                p["edit_names"] = edit_names
            if edits is not None:
                p["edits"] = edits
            if ipingp is not None:
                p["ipingp"] = ipingp
            if itabl is not None:
                p["itabl"] = itabl
            p["ntherm"] = ntherm
            if ntherm > 0:
                if mti is None:
                    raise ValueError("dtfr() with ntherm>0 requires 'mti'")
                p["mti"] = mti
                p["mtc"] = mtc
                p["nlc"] = nlc
        if materials is not None:
            p["materials"] = materials
        self._append("DTFR", p)
        return self

    def powr(
        self,
        *,
        ngendf: int | None = None,
        nout: int | None = None,
        lib: int = 1,
        iprint: str | None = None,
        iclaps: int = 0,
        # lib=1 and lib=2: materials list
        materials: list[dict] | None = None,
        # lib=3 (CPM) specific
        nlib: int = 1,
        idat: int = 0,
        iopt: int = 0,
        mode: int = 0,
        if5: int = 0,
        if4: int = 0,
        mat_list: list[str] | None = None,
        nuclides: list[dict] | None = None,
        burnup_data: list | None = None,
    ) -> InputDeck:
        """Add a POWR step, which writes an EPRI-CELL or EPRI-CPM library.

        Parameters
        ----------
        ngendf
            Input GENDF tape. None takes the output of the last ``groupr`` or
            ``gaminr`` step.
        nout
            Output tape. None allocates the next free unit (see ``moder``).
        lib
            1 fast library, 2 thermal library, 3 CPM library.
        iprint
            ``"min"``, ``"max"`` or ``"check"`` (0, 1, 2), or an integer. None
            writes 0.
        iclaps
            NJOY's ``iclaps``.
        materials
            ``lib`` 1 and 2 only: one dict per material. Both need ``matd`` (an
            isotope name as ``load_isotopes`` spells it). Fast (1): optionally
            ``rtemp``, ``iff``, ``nsgz``, ``izref``, and ``word`` and ``fsn``
            (texts). Thermal (2): optionally ``idtemp``, ``name``, ``itrc``,
            ``mti``, ``mtc``, ``xi``, ``alpha``, ``mubar``, ``nu``, ``kf``, ``kc``,
            ``lambda`` and ``sigma_s``.
        nlib, idat, iopt, mode, if5, if4
            ``lib`` 3 only: NJOY's fields of those names.
        mat_list
            ``lib`` 3 only: isotope names, written when ``iopt`` is 0.
        nuclides
            ``lib`` 3 only: one dict per nuclide, with the optional keys ``nina``,
            ``ntemp``, ``nsigz``, ``sgref``, ``ires``, ``sigp``, ``mti``, ``mtc``,
            ``ip1opt``, ``inorf``, ``pos``, ``posr`` and ``lambdas``.
        burnup_data
            ``lib`` 3 only: the lines of the burnup cards, written when ``if5`` is
            above 0.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.

        Raises
        ------
        ValueError
            If *ngendf* is None and no ``groupr`` or ``gaminr`` step comes before;
            if ``lib`` is not 1, 2 or 3; or if a material of ``lib`` 1 or 2 has no
            ``matd``.
        """
        ngendf = self._resolve_input(ngendf, '_gendf_tape', 'GENDF')
        nout = nout if nout is not None else self._alloc_tape()
        p: dict[str, Any] = {
            "ngendf": ngendf,
            "nout": nout,
            "lib": lib,
            "iclaps": iclaps,
        }
        if iprint is not None:
            p["iprint"] = iprint
        if lib in (1, 2):
            if materials is not None:
                p["materials"] = materials
        elif lib == 3:
            p["nlib"] = nlib
            p["idat"] = idat
            p["iopt"] = iopt
            p["mode"] = mode
            p["if5"] = if5
            p["if4"] = if4
            if mat_list is not None:
                p["mat_list"] = mat_list
            if nuclides is not None:
                p["nuclides"] = nuclides
            if burnup_data is not None:
                p["burnup_data"] = burnup_data
        self._append("POWR", p)
        return self

    def plotr(
        self,
        *,
        nplt: int | None = None,
        nplt0: int = 0,
        plot_def: PlotrJob | None = None,
    ) -> InputDeck:
        """Add a PLOTR step, which writes plot commands from ENDF, PENDF or GENDF data.

        Parameters
        ----------
        nplt
            Output plot-command tape, an ASCII (positive) unit. None allocates the
            next free unit (see ``moder``) and takes its absolute value. A later
            ``viewr`` step without ``infile`` reads it.
        nplt0
            Input plot-command tape, 0 for none.
        plot_def
            Plot commands written after the module card. Its own ``nplt`` and
            ``nplt0`` fields are not read. None writes only the module card.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.
        """
        if nplt is None:
            nplt = abs(self._alloc_tape())
        p: dict[str, Any] = {"nplt": nplt, "nplt0": nplt0}
        if plot_def is not None:
            p["plot_def"] = plot_def
        self._append("PLOTR", p)
        self._plotr_tape = nplt
        return self

    def viewr(
        self,
        *,
        infile: int | None = None,
        nps: int,
        plot_def: ViewrPlot | None = None,
    ) -> InputDeck:
        """Add a VIEWR step, which turns plot commands into a PostScript file.

        Parameters
        ----------
        infile
            Input plot-command unit. With *plot_def* the value is replaced by 5
            (standard input, where the commands are written). Otherwise None takes
            the output tape of the last ``plotr`` step, or 5 when there is none.
        nps
            Output PostScript unit. There is no default.
        plot_def
            Plot commands written after the module card. None writes only the
            module card.

        Returns
        -------
        InputDeck
            The deck itself, so that calls chain.
        """
        if infile is None:
            if plot_def is not None:
                infile = 5
            elif self._plotr_tape is not None:
                infile = self._plotr_tape
            else:
                infile = 5
        p: dict[str, Any] = {"infile": infile, "nps": nps}
        if plot_def is not None:
            p["plot_def"] = plot_def
        self._append("VIEWR", p)
        return self

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _append(self, name: str, params: dict[str, Any]) -> None:
        for key, value in params.items():
            if key in _TAPE_KEYS and isinstance(value, int) and value != 0:
                self._used_units.add(abs(value))
        self._lines.extend(generate_module(name, params))
        self._modules.append(name)

    def render(self) -> str:
        """Return the full NJOY input deck as a string."""
        if not self._lines:
            return ""
        return "\n".join(self._lines + ["stop"])

    def save(self, path: str | Path) -> None:
        """Write the rendered deck to *path* as UTF-8 text, replacing the file.

        Parameters
        ----------
        path
            Destination file. Its folder must exist. The text is that of
            ``render``: with no trailing newline, and empty for a deck with no
            module.
        """
        Path(path).write_text(self.render(), encoding="utf-8")

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        chain = " -> ".join(self._modules) if self._modules else "empty"
        return f"InputDeck({len(self._modules)} modules: {chain})"
