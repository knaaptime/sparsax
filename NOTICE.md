# Licensing notice

`sparsax`'s own source code is licensed **BSD-3-Clause** (see `LICENSE.txt`).
That covers everything in this repository.

**A compiled `sparsax` is a different matter.** The extension module links
against SuiteSparse, parts of which are licensed **GPL-2.0-or-later**. Binary
distributions — the wheels on PyPI, the conda-forge package, or anything you
build yourself — are combined works that bundle or link that code, so
**redistributing a built `sparsax` obliges you to comply with the GPL**, not
merely with BSD-3-Clause.

BSD-3-Clause is GPL-compatible, so making the combination is permitted. But
compatibility only means the combination is *allowed*; it does not make the
result permissive. It does not.

## Sobol direction numbers

`src/cpp/sobol_directions.h` holds the first 256 dimensions of Joe and Kuo's
direction numbers (`new-joe-kuo-6.21201`, from
<https://web.maths.unsw.edu.au/~fkuo/sobol/>), which `argmax_logprob` uses for its
draws: S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better
two-dimensional projections", *SIAM J. Sci. Comput.* 30, 2635-2654, 2008. Their
licence applies to those numbers, in source and in compiled form:

```
Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the names of the copyright holders nor the names of the
      University of New South Wales and the University of Waikato
      and its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## What gets linked

`sparsax_cpp` links CHOLMOD, KLU, and UMFPACK directly, and those pull in the
rest transitively. Licenses as declared by SuiteSparse 7.10.1 (conda-forge
packaging; other distributors package the same upstream sources):

| Component | License |
|---|---|
| **UMFPACK** | **GPL-2.0-or-later** |
| **CHOLMOD** | LGPL-2.1-or-later **AND GPL-2.0-or-later** AND Apache-2.0 |
| KLU, BTF | LGPL-2.1-or-later |
| AMD, COLAMD, CAMD, CCOLAMD, SuiteSparse_config | BSD-3-Clause |
| METIS (vendored by CHOLMOD's Partition module) | Apache-2.0 |

CHOLMOD is split by module: Check, Cholesky, Core, Partition, and Utility are
LGPL-2.1-or-later, while **MatrixOps, Modify, and Supernodal are
GPL-2.0-or-later**.

## Which GPL code `sparsax` actually reaches

Not theoretically — these are called on ordinary code paths:

- **CHOLMOD Modify**, via `cholmod_updown`, is what `update_solve` and
  `update_solve_bcoo` are built on, and via `cholmod_rowdel`, what
  `argmax_logprob` is built on.
- **CHOLMOD Supernodal**, via `cholmod_factorize` under CHOLMOD's default
  `AUTO` strategy, which selects the supernodal factorization for suitable
  matrices. `set_options(supernodal=...)` selects it explicitly.
- **UMFPACK**, in its entirety, behind `umf_solve`, `umf_logdet`, `umf_factor`,
  `umf_solve_factor`, and `umf_logdet_factor`.

The project's own wheel build opts into these deliberately: the Windows job
installs `suitesparse[gpl]` from vcpkg precisely to get `cholmod_updown`.

## What this means for you

**Using `sparsax`** — running it, importing it, doing research with it, and
publishing results — is unaffected. The GPL governs distribution, not use.

**Redistributing a built `sparsax`** — vendoring the wheel, shipping it inside
an application, bundling it in a container image you distribute — means
distributing GPL-2.0-or-later code, with the obligations that carry. In
particular you cannot ship it as part of a proprietary product distributed
under terms that conflict with the GPL. If you are redistributing commercially,
get this checked properly rather than relying on this file.

**Depending on `sparsax` in your own permissively-licensed library** is fine as
far as *your* source goes — a dependency edge is not a combined work. It stops
being fine the moment you distribute a binary artifact with SuiteSparse inside
it.

## There is no permissive-only build today

A build reaching no GPL code is possible in principle but is not a supported
configuration: it would require dropping the UMFPACK backend entirely, dropping
`update_solve` / `update_solve_bcoo` and `argmax_logprob` (CHOLMOD Modify), and forcing
`supernodal=off` so `cholmod_factorize` stays in the LGPL simplicial path. That
removes a large part of why the package exists, so it is not offered. Open an
issue if you have a concrete need for it.

## Corrections

This file is a good-faith summary by the maintainer, not legal advice, and it
describes SuiteSparse as currently packaged. If you believe any of it is wrong,
please open an issue — a licensing error here is worth fixing quickly.
