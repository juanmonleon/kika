# Mesh tally test files

Real MCNP output from two permissively licensed projects, kept here as
fixtures for `kika.mcnp.read_meshtal`.

| File | Source | What it holds |
|---|---|---|
| `mcnptools_col_cf.msht` | LANL MCNPTools, `doc/.../meshtal` (BSD-3-Clause) | MCNP6 (ld=06/01/17) MESHTAL text: FMESH 14-84, rectangular and cylindrical, energy and time bins with totals, `out=col` and `out=cf` |
| `mcnptools_col_cf.inp` | same repository, `inpgen` | the input deck that wrote it |
| `mcnp631_xdmf_crop.h5` | OpenMC fundamental testing, `0.15.3/fundamental_testing/7224_ss/4/mcnp_run/c_source-batch.h5`, commit 775dc31f (MIT) | MCNP 6.3.1 HDF5 runtape written with `out=xdmf`, **cropped**: `/config_control` attributes and `/results/mesh_tally/mesh_tally_4` with x bins 0-4, y bins 28-31 and z bins 40-42 of the original 50x60x90 mesh (datasets sliced, attributes unchanged) |

Own runs (kika project, no third-party licence): one MCNP 6.3.0 fixed-source
problem written twice, differing only in FMESH `out=`.

| File | What it holds |
|---|---|
| `mcnp630_check_xdmf.inp`, `mcnp630_check_sci.inp` | the two decks (rect + cyl FMESH with energy/time bins, TMESH 71 on FMESH 34's grid, every MCTAL bin axis) |
| `mcnp630_xdmf.h5` | `/config_control` attributes and `/results` of the `out=xdmf` runtape (the rest of the 27 MB runtape dropped) |
| `mcnp630_colsci_cfsci.msht` | the MESHTAL of the `out=colsci`/`cfsci` run |
| `../mctal/mcnp630_check.m` | the MCTAL of the `out=xdmf` run |
| `mcnp61_check.inp`, `mcnp61_col_ij.msht`, `../mctal/mcnp61_check.m` | the same problem under MCNP 6.1 (`out=col`, FMESH 64 as `out=ij`) |

## Licences

### LANL MCNPTools (BSD-3-Clause)

#### Copyright

Copyright 2022.  Triad National Security, LLC.  All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for
Los Alamos National Laboratory (LANL), which is operated by Triad National
Security, LLC for the U.S. Department of Energy/National Nuclear Security
Administration.  All rights in the program are reserved by Triad National
Security, LLC, and the U.S.  Department of Energy/National Nuclear Security
Administration.  The Government is granted for itself and others acting on its
behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare derivative works, distribute copies to the public, perform
publicly and display publicly, and to permit others to do so.

#### License

This program is open source under the BSD-3 License.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#### Trademark Notice

MCNP® and Monte Carlo N-Particle® are registered trademarks owned by Los Alamos
National Security, LLC, manager and operator of Los Alamos National Laboratory.
Any third party use of such registered marks should be properly attributed to
Triad National Security, LLC, including the use of the ® designation as
appropriate. Any questions regarding licensing, proper use, and/or proper
attribution of Triad National Security, LLC marks should be directed to
[trademarks@lanl.gov](mailto:trademarks@lanl.gov).

### OpenMC fundamental testing (MIT)

```
MIT License

Copyright (c) 2026 OpenMC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
