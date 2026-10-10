# Processing with NJOY

```{admonition} At a glance
:class: note
**Computes:** ACE files from an ENDF tape (or from an ENDF tape and a PENDF), and
pointwise cross-sections reconstructed by NJOY RECONR, by running a local NJOY with a
fixed, checked input deck.

**Applies to:** neutron ENDF-6 tapes; NJOY 2016.

**Entry points:** `kika.njoy.run_njoy.run_njoy`, `kika.njoy.run_njoy.run_njoy_with_pendf`,
`kika.processing.njoy_reconstruct.njoy_reconstruct`,
`kika.processing.njoy_pendf_cache.get_or_create_pendf`; decks: `kika.njoy.InputDeck`,
`kika.njoy.reader.read_deck`.

**Status:** stable.
```

## Background

NJOY (MacFarlane et al., 2017) is the reference code that turns an evaluated ENDF-6 file
into the libraries transport codes read. It runs as a chain of modules on numbered
"tapes": RECONR reconstructs the resonance region and linearises every cross-section
into a PENDF (pointwise ENDF), BROADR Doppler-broadens it to a temperature, HEATR adds
heating and damage cross-sections, PURR adds probability tables for the unresolved range,
GASPR adds gas-production cross-sections, and ACER writes the ACE file a Monte Carlo
code reads. Each module reads a few cards of input; the deck is the sequence.

kika does not reimplement this chain. It runs NJOY, with an input it writes the same way
every time, and checks that NJOY read what it was given.

## What it computes

`run_njoy(njoy_exe, endf_path, temperature, library_name, output_dir)` produces, for one
tape and one temperature, an ACE file, its `xsdir` line and NJOY's listing, in
`output_dir/ace/` and `output_dir/njoy_files/`. `run_njoy_with_pendf` does the same from
a PENDF the caller supplies (a perturbed one, for instance), without running RECONR.
`njoy_reconstruct` runs RECONR alone and returns the reconstructed cross-sections by MT.

## Method

**1. The deck.** For a tape with material number MAT at temperature $T$ (in K), the
deck is the following, with `{mat}`, `{T}`, the title and the ACE suffix `{suff}`
substituted and nothing else changed:

```text
moder
 20 -25
reconr
 -25 -21
'{title}'/
 {mat} 0 0
 0.001 0 0.01 5e-08
 0 /
broadr
 -25 -21 -22
 {mat} 1 0 0 0
 0.001 1e+06 0.01 5e-08
 {T}
 0 /
heatr
 -25 -22 -21 /
 {mat} 5 0 0 0 0 /
 302 318 402 442 444 /
heatr
 -25 -22 -23 /
 {mat} 6 0 1 0 2 /
 302 303 402 442 443 444 /
purr
 -25 -21 -24
 {mat} 1 9 20 64 1 0 /
 {T} /
 1e+10 1e+8 1e+6 1e+4 1e+3 3e+2 1e+2 3e+1 1e+1 /
 0 /
gaspr
 -25 -24 -21  /
acer
 -25 -21 0 40 41
 1 0 1 {suff} /
'{title}'/
 {mat} {T}
 1 1 1
 /
acer
 0 40 0 43 44
 7 1 1 -1 /
'{title}'/
stop
```

RECONR and BROADR work to a fractional tolerance of $10^{-3}$; PURR uses nine background
cross-sections from $10^{10}$ to 10 b; the second ACER call reads back the ACE file the
first one wrote, checks it, and writes the final ACE file and its `xsdir` line (tapes 43
and 44). With a PENDF supplied, the RECONR step is replaced by a MODER that reads the
PENDF from tape 26 and converts it to the binary tape BROADR expects.

**2. Material, temperature and suffix.** MAT is read from the tape's first material
section. A temperature below 1 is taken as $kT$ in MeV and converted to K. The ACE
suffix is the caller's extension, or the standard one for the temperature (`.02` for
293.6 K, `.03` for 300 K, `.06` for 600 K and so on, matched to within 1 K, `.00`
otherwise), and the library name gives the evaluation's part of the table name (`00`
with a warning for a library kika does not know).

**3. The run, and the check that NJOY read the deck.** NJOY is run in a temporary
directory with the tape as `tape20`. RECONR echoes the three tolerances of its fourth
card into the listing; kika reads them back and compares them with the ones it sent
(to $5.05 \times 10^{-4}$ in the absolute tolerance). A mismatch means NJOY misread its
input; the same deck is run again, up to 8 times, and if every run misread, the run
fails with `NjoyDecimalLocaleError` instead of producing an ACE file. A run whose deck
has no RECONR is checked through BROADR's echo instead. NJOY's listing is kept beside
the ACE file.

**4. RECONR alone.** `njoy_reconstruct` runs MODER and RECONR at the requested
tolerance $\varepsilon$ (default $10^{-3}$), with $\varepsilon_\text{max} = 10\varepsilon$
and $\varepsilon_\text{int} = 5 \times 10^{-5}\varepsilon$, reads the PENDF and returns
its MF 3 sections. When RECONR stops on an "ill-behaved threshold", the tolerance is
doubled and the run repeated, up to 10 %; beyond that the NJOY error is raised.

**5. A cached PENDF for derived covariances.** An MF 33 section can state a covariance
as a combination of other reactions' (NC sub-subsections, LTY=0). Resolving it needs
the reconstructed cross-sections, so `get_or_create_pendf` runs RECONR once per tape,
keyed by the SHA-256 of the tape, and later calls read the cached PENDF.

## Conventions and choices

- **One fixed deck.** Every sample of an ensemble, and every tape of a library, is
  processed by the same chain with the same parameters; only MAT, $T$, the title and the
  suffix change. A difference between two ACE files is then a difference in the data.
- **PURR, not THERMR.** The chain adds probability tables for the unresolved resonance
  range and no thermal scattering law treatment; a tape that needs S($\alpha$,$\beta$) is
  processed separately.
- **The input is checked, not trusted.** On Windows, some NJOY builds switch to the
  system's decimal separator after writing their first binary tape and then misread
  numbers such as `2.330248+2`. The run still completes and writes a plausible file, so
  kika compares what NJOY echoed with what it sent and re-runs on a mismatch.
- **A looser RECONR tolerance is taken only on NJOY's own error**, and stops at 10 %:
  a coarser reconstruction is no longer useful, and kika raises instead of returning it.
- **The executable is the caller's**: the argument, else `$NJOY_EXECUTABLE`, else `njoy`
  on the path. kika ships no NJOY.

## Limits and failure modes

- No NJOY executable found: `NjoyNotFoundError` (for the PENDF cache) or the subprocess
  error.
- A tape with no MAT in its header: `ValueError`.
- NJOY misread its input on every attempt: `NjoyDecimalLocaleError`
  (`NjoyDecimalLocaleReconstructError` for RECONR alone).
- RECONR fails at 10 % tolerance: the NJOY error, with the tail of its listing.
- The deck builder (`InputDeck`, for the 22 NJOY modules) writes cards from the
  arguments given and does not check them against NJOY's rules.

## Validation

- The decimal-misread check: a clean listing passes, a corrupt one is caught with the
  reason, a missing listing never blocks a run, a tolerance below the printed precision
  is not flagged, a PENDF deck is checked through BROADR
  (`kika/processing/tests/test_njoy_locale_guard.py`, 10 tests, passed 2026-10-11).
- RECONR reconstruction of Fe-56 and U-238, streaming, and missing executable or tape
  (`kika/processing/tests/test_njoy_reconstruct.py`). `TODO(check)`: the reconstruction
  tests need NJOY and `--deep`; not re-run for this page.
- ACE produced inside a sampling run
  (`kika/sampling/tests/test_ace_is_an_emitter.py`). `TODO(check)`: the end-to-end case
  needs NJOY and a full tape; not re-run.
- The deck builder, the deck reader and `run_njoy` itself have no dedicated tests.

## Diagnostics

`run_njoy` returns a dictionary: `returncode`, `zaid`, `ace_file` and `xsdir_file` (or
`None` when NJOY wrote none), and NJOY's `njoy_input`, `njoy_output` and `njoy_listing`. **A non-zero return code, or an ACE
file missing, means read the listing.** `njoy_reconstruct_stream` yields `log`,
`warning`, `pendf` and `result` events; a warning names a tolerance that was loosened.

## Implementation

- `kika/njoy/templates.py`: the deck, with and without RECONR.
- `kika/njoy/run_njoy.py`: `run_njoy`, `run_njoy_with_pendf`, the suffix rules, the
  re-run loop.
- `kika/njoy/locale_guard.py`: reading the echoed tolerances back from the listing;
  `probe_executable` measures how often a binary misreads.
- `kika/njoy/launcher.py`: the command line (Windows and WSL included).
- `kika/processing/njoy_reconstruct.py`: RECONR alone, with the tolerance ladder.
- `kika/processing/njoy_pendf_cache.py`: the cached PENDF for derived MF 33 covariances.
- `kika/njoy/deck.py`, `reader.py`, `modules/`: building and reading arbitrary decks;
  `viewr_defs.py`, `plotr_defs.py`: VIEWR and PLOTR plot definitions.

## References

- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791.
- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: §33 (MF 33, NC sub-subsections).
