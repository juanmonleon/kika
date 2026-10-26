# From an evaluation to transport files

How to take an evaluation from the library to the files a transport code reads (ACE for
MCNP and Serpent, HDF5 for OpenMC), and how to read the report kika writes about the run.
The example processes the ENDF/B-VIII.1 evaluation of Fe-56; the outputs shown are what it
printed. How each stage computes what it does is in
{doc}`../../methods/processing/processing-chain`, and every option of the processing is in
{doc}`../processing-an-evaluation`.

## Get the evaluation

{py:func}`~kika.endf.remote.download_endf` downloads an evaluation from the IAEA Nuclear
Data Services and keeps a copy in kika's cache:

```python
from kika.endf.remote import download_endf

tape = download_endf("Fe56", save_to=".", library="endfb8.1")    # Fe56_endfb8.1_n.endf
```

## Process it

{py:func}`kika.process` reads the tape, reconstructs the resonance region, Doppler-broadens
to room temperature, computes the unresolved-range probability tables when there is an
unresolved range, and checks every stage. `write` then writes the files and reads each one
back before keeping it. The processing uses several processes, so a script that calls it
on Windows or macOS needs the `if __name__ == "__main__":` guard:

```python
import kika

def main():
    run = kika.process("Fe56_endfb8.1_n.endf")
    run.write("out/", ["ace", "hdf5"])
    run.report_document().save("out/Fe56.html")

if __name__ == "__main__":
    main()
```

```text
out/
├── 26056.02c             ACE table at 293.6 K
├── xsdir                 its xsdir line
├── Fe56.h5               OpenMC HDF5
├── cross_sections.xml    the index OpenMC reads
└── Fe56.html             the report
```

The processing took 38 s and writing the two files 98 s on a 12-core desktop (measured
2026-10-26). `kika.process("...", temperatures=[293.6, 600, 900])` processes several
temperatures at once: the HDF5 file holds them all, and each ACE table is one temperature
(`.02c`, `.06c`, `.09c`).

## Read the report

The report is one self-contained HTML page; `report_document().to_text()` gives the same
in plain text, which is what is quoted here. It opens with the answer:

```text
 [WARN] PROCESSED, WITH WARNINGS TO REVIEW
   Fe-56 (ENDF/B-VIII.1) at 293.6 K. Files written and read back: ACE (1 table), OpenMC HDF5. 1
   warning, 1 note.
```

`OK` means every check passed, `WARN` that something needs a look, and `FAIL` that a stage
did not finish (and its files were not written). Then one line per stage:

```text
           Stage                                Time        What was done
   ------  -----------------------------------  ----------  ----------------------------------------------------------
   [WARN]  Reading the evaluation               1.1 s       ENDF/B-VIII.1 evaluation of Fe-56 (MAT 2631), 77 reactions.
   [NOTE]  Cross sections at 0 K                3.4 s       312 resolved resonances (Reich-Moore) up to 850 keV; ...
   [ OK ]  Doppler broadening                   30.7 s      To 293.6 K, up to 850 keV; 73,484 points at 293.6 K.
   [ -- ]  Unresolved-range probability tables  -           The file has no unresolved resonance range; ...
   [ -- ]  Heating and damage                   -           No stage of its own: the ACE and HDF5 files carry
                                                            KERMA (MT301) and damage (MT444), computed when they
                                                            are written.
   [ OK ]  Files written                        1 min 26 s  Written and read back: ACE (26056.02c); OpenMC HDF5 (Fe56.h5).
```

The checks under each stage say what was verified. For the 0 K cross-sections, for
instance:

```text
     [ OK ] Tables reproduce the reference cross sections: 77 reactions; worst difference 45.4 % of
            the tolerance (MT102 (n,g) at 671.2 keV).
     [NOTE] The evaluation's totals equal the sum of their parts: MT4 differs from the sum of its
            parts, by up to 35x the tolerance.
     [ OK ] Integrals over energy are preserved: 77 reactions; worst relative difference 0.00325 %
            (MT102 (n,g)).
```

The tolerance is 0.1 % of the cross-section (five times tighter below 0.5 eV), from the
evaluation to the file; *45.4 % of the tolerance* means the worst table is within 0.045 %.
The note says the evaluation's own MT 4 is not the sum of its levels: kika rebuilds every
total from its parts, as NJOY does, and only compares the evaluation's.

## What the warning is

The warning comes from the data checks run on the evaluation as it is read, and the
*Findings* section lists them:

```text
   [WARN] The evaluation passes kika's data checks
          The evaluation has defects or warnings (listed). The output carries them as the evaluation
          states them.
        - Warning, Negative Legendre distribution (MF4 MT2): MT2: the Legendre series is negative at
          E = 1.557 MeV: negative area 0.0023, minimum -0.0181 (and at 1 other incident energy of
          3960)
        - Note, Negative cross section (MF3 MT2): MT2: negative background at 15 points inside the
          resolved range (down to -3.156 mb at 801 keV); the resonances of MF2 are added to it
        ...
```

These are properties of the evaluation, not of the processing: the elastic angular
distribution of ENDF/B-VIII.1 Fe-56 goes slightly negative near 1.56 MeV, and the files
carry it as stated. Whether it matters is a judgement about the application.
{doc}`../checking-evaluations` explains every check.

## The choices kika made

The section *Choices made for you* lists every setting the run used without being told,
with its reason:

```text
   - Temperatures: 293.6 K
     Why: Room temperature, as the official libraries state it (293.6 K), unless others were asked
     for.
   - Accuracy: Every final table within 0.1 % of the cross section, five times tighter below 0.5 eV,
     from the evaluation to the file. Half of it goes to the 0 K reconstruction, 5 % to the
     broadening, the rest to thinning the grid.
     Why: NJOY applies its tolerance once in each module, so its errors add up; kika splits one
     tolerance between the stages, so the tables are within the one stated.
   - Broadening range: Up to 850 keV; above it the 0 K cross sections are used.
   - ACE tables: Named ZAID plus the suffix of the temperature (.02c at 293.6 K), legacy header, one
     xsdir line each.
```

The few options that exist are physical: `temperatures`, and `heating`. By default the ACE
and HDF5 files carry KERMA and damage, computed when they are written; `heating=True` also
puts HEATR-style MT 301 and MT 444 in a PENDF, and `heating=False` leaves them out of every
file (an ACE table's heating numbers are then 0, and an OpenMC heating tally has nothing to
read).

## Use the files

The ACE table comes with its xsdir line: add it to the `xsdir` of the data your MCNP or
Serpent installation reads, with the path of the table. For OpenMC, point the
`OPENMC_CROSS_SECTIONS` environment variable at `out/cross_sections.xml`, or merge the
entry into your library's index.

## From the command line

The same run, without Python:

```bash
kika process Fe56_endfb8.1_n.endf --to ace hdf5 -o out/
```

It writes the report beside the files as `Fe56_endfb8.1_n.report.html`, and exits with
status 1 when a stage did not finish, so it can sit in a script that processes a whole
library: `kika process endfb81/neutrons/ --to ace -o lib/`.
