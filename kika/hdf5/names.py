"""The names the format fixes: temperatures, nuclides, reaction labels.

Every table here is OpenMC v0.16.0's, copied rather than imported (OpenMC is
not a dependency, roadmap D3), with its source line. The C++ re-opens groups by
the names these functions produce, so a name that differs by one character is a
file that fails to load (``openmc_hdf5_format_survey.md`` §2.2).
"""
from __future__ import annotations

from typing import Tuple

from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, SYMBOL_TO_ATOMIC_NUMBER

__all__ = ["K_BOLTZMANN", "EV_PER_MEV", "HDF5_VERSION", "temperatureName",
           "temperatureFromName", "gndsName", "parseGndsName", "reactionLabel",
           "REACTION_NAME", "FISSION_MTS", "KEEP_REDUNDANT_MTS", "SUBSHELLS",
           "isFission", "isDisappearance", "isInelasticScatter"]

#: eV/K. ``openmc.data.K_BOLTZMANN`` (from the ``endf`` package OpenMC uses).
#: kika's own ``BOLTZMANN_CONSTANT`` is the CODATA-2018 value to two more
#: digits; this one is what produced the official kTs and must be used to name
#: their groups.
K_BOLTZMANN = 8.617333262e-05
EV_PER_MEV = 1.0e6

#: ``openmc/data/__init__.py:2-4``; the C++ compares only the major number.
HDF5_VERSION = (3, 0)

#: ``reaction.py:82``.
FISSION_MTS = (18, 19, 20, 21, 38)

#: Redundant reactions the official exporter keeps (``neutron.py:406-411``).
KEEP_REDUNDANT_MTS = (4, 16, 103, 104, 105, 106, 107, 203, 204, 205, 206, 207,
                      301, 444, 901)

#: ``photon.py:32-35``.
SUBSHELLS = (None, 'K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
             'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3',
             'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'P1', 'P2', 'P3', 'P4',
             'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'Q1', 'Q2', 'Q3')

#: ``reaction.py:29-73``.
REACTION_NAME = {1: '(n,total)', 2: '(n,elastic)', 3: "(n,nonelastic)",
                 4: '(n,level)', 5: '(n,misc)', 11: '(n,2nd)', 16: '(n,2n)',
                 17: '(n,3n)', 18: '(n,fission)', 19: '(n,f)', 20: '(n,nf)',
                 21: '(n,2nf)', 22: '(n,na)', 23: '(n,n3a)', 24: '(n,2na)',
                 25: '(n,3na)', 27: '(n,absorption)', 28: '(n,np)', 29: '(n,n2a)',
                 30: '(n,2n2a)', 32: '(n,nd)', 33: '(n,nt)', 34: '(n,n3He)',
                 35: '(n,nd2a)', 36: '(n,nt2a)', 37: '(n,4n)', 38: '(n,3nf)',
                 41: '(n,2np)', 42: '(n,3np)', 44: '(n,n2p)', 45: '(n,npa)',
                 91: '(n,nc)', 101: '(n,disappear)', 102: '(n,gamma)',
                 103: '(n,p)', 104: '(n,d)', 105: '(n,t)', 106: '(n,3He)',
                 107: '(n,a)', 108: '(n,2a)', 109: '(n,3a)', 111: '(n,2p)',
                 112: '(n,pa)', 113: '(n,t2a)', 114: '(n,d2a)', 115: '(n,pd)',
                 116: '(n,pt)', 117: '(n,da)', 152: '(n,5n)', 153: '(n,6n)',
                 154: '(n,2nt)', 155: '(n,ta)', 156: '(n,4np)', 157: '(n,3nd)',
                 158: '(n,nda)', 159: '(n,2npa)', 160: '(n,7n)', 161: '(n,8n)',
                 162: '(n,5np)', 163: '(n,6np)', 164: '(n,7np)', 165: '(n,4na)',
                 166: '(n,5na)', 167: '(n,6na)', 168: '(n,7na)', 169: '(n,4nd)',
                 170: '(n,5nd)', 171: '(n,6nd)', 172: '(n,3nt)', 173: '(n,4nt)',
                 174: '(n,5nt)', 175: '(n,6nt)', 176: '(n,2n3He)',
                 177: '(n,3n3He)', 178: '(n,4n3He)', 179: '(n,3n2p)',
                 180: '(n,3n2a)', 181: '(n,3npa)', 182: '(n,dt)',
                 183: '(n,npd)', 184: '(n,npt)', 185: '(n,ndt)',
                 186: '(n,np3He)', 187: '(n,nd3He)', 188: '(n,nt3He)',
                 189: '(n,nta)', 190: '(n,2n2p)', 191: '(n,p3He)',
                 192: '(n,d3He)', 193: '(n,3Hea)', 194: '(n,4n2p)',
                 195: '(n,4n2a)', 196: '(n,4npa)', 197: '(n,3p)',
                 198: '(n,n3p)', 199: '(n,3n2pa)', 200: '(n,5n2p)', 203: '(n,Xp)',
                 204: '(n,Xd)', 205: '(n,Xt)', 206: '(n,X3He)', 207: '(n,Xa)',
                 301: 'heating', 444: 'damage-energy',
                 501: 'photon-total', 502: 'coherent-scatter',
                 504: 'incoherent-scatter', 515: 'pair-production-electron',
                 516: 'pair-production', 517: 'pair-production-nuclear',
                 522: 'photoelectric',
                 649: '(n,pc)', 699: '(n,dc)', 749: '(n,tc)', 799: '(n,3Hec)',
                 849: '(n,ac)', 891: '(n,2nc)', 901: 'heating-local'}
REACTION_NAME.update({i: f'(n,n{i - 50})' for i in range(51, 91)})
REACTION_NAME.update({i: f'(n,p{i - 600})' for i in range(600, 649)})
REACTION_NAME.update({i: f'(n,d{i - 650})' for i in range(650, 699)})
REACTION_NAME.update({i: f'(n,t{i - 700})' for i in range(700, 749)})
REACTION_NAME.update({i: f'(n,3He{i - 750})' for i in range(750, 799)})
REACTION_NAME.update({i: f'(n,a{i - 800})' for i in range(800, 849)})
REACTION_NAME.update({i: f'(n,2n{i - 875})' for i in range(875, 891)})
REACTION_NAME.update({534 + i: f'photoelectric-{shell}' for i, shell in enumerate(SUBSHELLS[1:])})


def reactionLabel(mt: int) -> str:
    """``@label`` of ``reaction_<MT>``: the table's name, or the MT as text (``reaction.py:933-936``)."""
    return REACTION_NAME.get(int(mt), str(int(mt)))


def temperatureName(kT: float) -> str:
    """The group name of a temperature given as kT in eV: ``f"{round(kT/k_B)}K"``.

    293.6 K is ``"294K"``. The C++ derives the temperatures from the *values*
    in ``kTs/`` and re-opens ``energy/{T}K`` with this exact spelling
    (``nuclide.cpp:86-90, 226``), so it is not a choice.
    """
    return f"{int(round(kT / K_BOLTZMANN))}K"


def temperatureFromName(name: str) -> float:
    """``"294K"`` → 294.0. The *name*, which is rounded; the exact kT is in ``kTs/``."""
    if not name.endswith("K"):
        raise ValueError(f"{name!r} is not a temperature group name")
    return float(name[:-1])


def gndsName(Z: int, A: int, metastable: int = 0) -> str:
    """``U235``, ``Am242_m1`` — ``openmc.data.gnds_name``."""
    base = f"{ATOMIC_NUMBER_TO_SYMBOL[int(Z)]}{int(A)}"
    return f"{base}_m{int(metastable)}" if metastable else base


def parseGndsName(name: str) -> Tuple[int, int, int]:
    """``"Am242_m1"`` → ``(95, 242, 1)``."""
    base, _, meta = name.partition("_m")
    symbol = base.rstrip("0123456789")
    if symbol not in SYMBOL_TO_ATOMIC_NUMBER or symbol == base:
        raise ValueError(f"{name!r} is not a GNDS nuclide name")
    return SYMBOL_TO_ATOMIC_NUMBER[symbol], int(base[len(symbol):]), int(meta or 0)


def isFission(mt: int) -> bool:
    """``src/endf.cpp:53-57``."""
    return mt in FISSION_MTS


def isDisappearance(mt: int) -> bool:
    """``src/endf.cpp:59-71``: what the C++ adds to absorption besides fission."""
    return (101 <= mt <= 117 or 600 <= mt <= 849
            or mt in (155, 182, 191, 192, 193, 197))


def isInelasticScatter(mt: int) -> bool:
    """``src/endf.cpp:73-90`` (``MISC`` is 5, ``N_2N0..N_2NC`` 875-891)."""
    if mt < 100:
        return not isFission(mt) and mt >= 5 and mt != 27
    if mt <= 200:
        return not isDisappearance(mt)
    return 875 <= mt <= 891
