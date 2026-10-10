# Reading and writing ENDF special data

```{admonition} At a glance
:class: note
**Reads:** the parts of ENDF-6 beyond cross-sections and secondary distributions:
thermal neutron scattering laws (MF 7), photon production (MF 12-15), radioactive decay
(MF 8/MT 457), fission product yields (MF 8/MT 454, 459), and the photo-atomic and atomic
relaxation sublibraries (MF 23, 27, 28).

**Writes:** all of them back to ENDF, and to GNDS through the model.

**Into:** the canonical model (thermal scattering, photons, decay, yields); plain arrays
for photo-atomic data.

**Entry points:** {py:func}`kika.read`, {py:func}`kika.write`;
`kika.endf.photoatomic_summary`, `kika.endf.photoatomic_cross_sections`,
`kika.endf.form_factors`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

These data are spread over ENDF-6 files and sublibraries of their own (ENDF-6 Ch. 7, 8,
12-15, 23, 27, 28): a thermal scattering tape (NSUB=12) states $S(\alpha, \beta, T)$ and the
elastic laws of a moderator; a reaction's photons are listed in MF 12 or MF 13 with their
angular (MF 14) and continuum (MF 15) distributions; a decay sublibrary (NSUB=4) holds one
nuclide and how it decays; a fission-yield sublibrary (NSUB=5, 11) its products. GNDS holds
the same physics in other shapes, and kika's mapping follows FUDGE's wherever FUDGE follows
the GNDS specification, so a file kika writes and the distributed GNDS files describe the
same data the same way.

## Coverage

| Part | Into the model |
|---|---|
| MF 7/MT 2 coherent elastic (LTHR=1, 3) | `CoherentElastic` with its cumulative $S$ table over temperature and energy |
| MF 7/MT 2 incoherent elastic (LTHR=2, 3) | `IncoherentElastic`: bound cross-section and Debye-Waller integral |
| MF 7/MT 4 incoherent inelastic | $S(\alpha, \beta, T)$ as a three-dimensional grid (temperature, $\beta$, $\alpha$), the free and bound cross-sections, the effective temperatures, the secondary scatterers (short-collision-time or free-gas) |
| MF 12 LO=1, MF 13, MF 14, MF 15 | photon products of the reaction: discrete, primary and continuum photons with their multiplicities and distributions; totals as multiplicity sums |
| MF 12 LO=2 | the level's decay data in the particle database, with branching |
| MF 8/MT 457 (NSUB=4) | a `PoPs` nuclide with half-life, spin, parity and decay data |
| MF 8/MT 454, 459 (NSUB=5, 11) | independent and cumulative product yields per incident energy |
| MF 23, 27, 28 (NSUB=3, 6) | read in the parse layer; arrays through `kika.endf.photoatomic_*` |

## Reading

**Thermal scattering.** Atom identities come from MF 7/MT 451, matched by mass, never from
the tape's file name. The bound cross-section of the principal atom is
$\sigma_\text{bound} = \sigma_\text{free}\,((A+1)/A)^2$, and its mass $A\,m_n$. ENDF stores
$S$ as $(\beta, T, \alpha)$; the model holds it transposed, (temperature, $\beta$, $\alpha$).

**Photons.** A photon hangs on the reaction's output channel, except for a discrete-level
reaction (MT 51-90, 600-648, ...), where it is the decay of the excited residual and goes
into the residual's channel. The photons of a stated sum go to an orphan product that refers
to the sum. A product is matched to its MF 14 entry by its energies $(E_\gamma, E_s)$, never
by position. MF 13 states production cross-sections $\sigma_\gamma$; the model holds
multiplicities, $\sigma_\gamma/\sigma$, computed on the union of the two grids after making
both lin-lin; where $\sigma = 0$ the multiplicity is 0 and the report says so.

**Decay and yields.** A decay evaluation is a nuclide in a standalone `PoPs`. MT 454 is the
independent yield at elapsed time 0, MT 459 the cumulative one; a neutron-induced evaluation
lists one incident energy per ENDF energy. The uncertainty is held as the variance
$\Delta Y^2$.

## Assumptions and defaults

- Thermal scattering: LI=0 is kept as written (not rewritten to 2), because kika reproduces
  the tape.
- Photons: MF 14 with LI=1, or an isotropic record, is an isotropic distribution; $a_0 = 1$
  is written out for Legendre photon distributions.

## Ambiguous and inconsistent input

- **Photons are paired with MF 14 entries by energy**, $(E_\gamma, E_s)$ within
  $10^{-4}$ eV (FUDGE's tolerance), never by position. A reaction whose photons have no MF 14
  at all is reported as a loss; a single photon left without a match is not reported.
- **MF 13 reached by division** is an approximation, and the report says so.

## Refused

- Thermal scattering: LLN=1 (logarithm of $S$ stored), a diffusive secondary scatterer
  ($B(6i+1)=2$), and $\alpha$ grids that differ between $\beta$ values, each by name.
  FUDGE refuses the same three.

## Writing

Everything the model holds is written back. What GNDS has no node for is kept beside the
model and restored on the way out:

- **Thermal scattering**: the B array as written and the exact digits of the scalars.
- **Photons**: an MF 13 section is written back as its original text while the
  multiplicities and the cross-section it was divided by are unchanged (checked by
  SHA-256); once either changes, it is rebuilt.
- **Decay and yields**: the spectrum normalisations (FD, FC), average energies, the order of
  the lines, each energy's interpolation flag and the exact text of fields are kept as a
  layout plus **field overrides**: after a section is rebuilt from the model, a field that
  still differs from the tape is restored only if the rebuilt text is unchanged, so an
  override restores a digit the model cannot express and never undoes an edit.

## What the round trip keeps and loses

Decay sublibraries: ENDF to the model and back to ENDF, byte for byte; model to GNDS and
back to the model, identical (`kika/endf/model_adapter/tests/test_decay_sublibrary.py`; the
whole ENDF/B-VIII.1 decay sublibrary under `-m tape`, `TODO(check)`: not re-run here).
Thermal scattering and photons come back with the sections unchanged where the model was
not edited.

## Report

Entries a user meets: photon multiplicities set to 0 where $\sigma = 0$ (MF 13); MF 13 as an
approximation; a reaction whose photons have no MF 14 (a loss); refused thermal scattering
forms.

## Validation

All passed on 2026-10-11 (94 passed, 17 skipped that need full tapes):
`kika/endf/model_adapter/tests/test_thermal_scattering_round_trip.py`,
`test_thermal_scattering_elastic_xs.py`, `test_photon_round_trip.py`,
`test_photon_cascades.py`, `test_photon_ace.py`, `test_decay_sublibrary.py`,
`test_fission_yields.py`, and `kika/endf/tests/test_photoatomic.py`. Comparisons against
NJOY RECONR on a kika-written photon tape (`test_photon_reconr.py`) and against FUDGE's
decay translation need NJOY or FUDGE: `TODO(check)`, not re-run.

## Implementation

- `kika/endf/model_adapter/thermal_scattering.py`: MF 7.
- `kika/endf/model_adapter/photons.py`: MF 12-15.
- `kika/endf/model_adapter/decay_sublibrary.py`: MT 457 and the sublibrary writer;
  `fission_yields.py`: MT 454, 459.
- `kika/endf/photoatomic.py`: photo-atomic arrays; `kika/endf/parsers/parse_photoatomic.py`
  and `kika/endf/classes/mf23/`, `mf27/`, `mf28/`: the parse layer.
- `kika/nuclear_data/photon_production.py`: `PhotonProduction`, lines, cascades and
  continua of a reaction for display.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: Ch. 7, 8, 12-15, 23, 27, 28.
- C. Mattoon, B. Beck and G. Gert, "Managing and Processing Nuclear Data Libraries with
  FUDGE", *EPJ Web of Conferences* 284, 14010 (2023), doi:10.1051/epjconf/202328414010.
