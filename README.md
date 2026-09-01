[![Continuous Integration](https://github.com/knaaptime/sparsax/actions/workflows/unittests.yml/badge.svg)](https://github.com/knaaptime/sparsax/actions/workflows/unittests.yml)
[![codecov](https://codecov.io/gh/knaaptime/sparsax/branch/main/graph/badge.svg)](https://codecov.io/gh/knaaptime/sparsax/branch/main/graph/badge.svg)


# sparsax

JAX & PyTensor-native sparse direct solvers backed by [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).
`solve`, `logdet`, and friends are [XLA FFI](https://docs.jax.dev/en/latest/ffi.html) custom
calls into **CHOLMOD** (Cholesky, for symmetric positive definite matrices) and **KLU** and
**UMFPACK** (LU, for general non-symmetric matrices), so sparse factorizations run at full
native speed **inside `@jax.jit`** (and `lax.scan` / `lax.fori_loop`) — no Python callback
overhead.

## Why?

No open-source JIT framework (JAX, PyTorch, TensorFlow) exposes a sparse direct factorization
as a compilable primitive. For iterative workloads that factorize the same sparsity pattern
many times with changing values — Gibbs samplers, MCMC for Bayesian spatial econometrics,
shift-invert Krylov methods — paying the Python/SuiteSparse boundary on every iteration
dominates the runtime. `sparsax` moves the whole loop to C++ behind a JIT-compiled JAX
function, with the symbolic analysis (fill-reducing ordering + elimination tree) cached and
reused across calls, and the numeric factorization skipped entirely when the values haven't
changed.

Benchmarks (M-series macOS, 2D grid Laplacian, values changing every iteration): CHOLMOD via
`sparsax` matches a hand-written scikit-sparse Python loop per iteration and is ~2.7× faster
than `scipy.sparse.linalg.splu`, while running entirely inside `jax.jit`.

## Features

- **CHOLMOD** — `solve`, `logdet`, `selinv` (selected inverse), `factor_solve`,
  `sample_gaussian`, `update_solve` (rank-k update/downdate) for symmetric positive definite
  `A`.
- **KLU** — `lu_solve`, `lu_logdet`, `lu_factor` / `lu_solve_factor` for general
  non-symmetric `A` (e.g. `A = I − ρW` with a row-standardised spatial weights matrix).
- **UMFPACK** — `umf_solve`, `umf_logdet`, `umf_factor` / `umf_solve_factor`: the same API
  over SuiteSparse's multifrontal LU, for the same non-symmetric `A`. The two LU backends are
  interchangeable and differ only in which graph they are fast on — see
  *Two LU backends: KLU vs UMFPACK* below.
- **JIT / `lax.scan`**: the symbolic analysis is computed once and reused across iterations.
- **`vmap`**: `jax.vmap(solve)` lowers to a *single* native FFI call that loops over the
  batch in C++ (reusing the cached analysis), rather than XLA per-iteration dispatch.
  Composes with `grad` (`vmap(grad(solve))` batches too).
- **Autodiff**: `solve` has a custom reverse-mode VJP in both `Ax` and `b`; `logdet` has one
  in `Ax` via the selected inverse. `lu_solve` is differentiable in `Ax` and `b` (transpose
  solve), as is `umf_solve`; `lu_logdet` and `umf_logdet` are forward-only. Together
  `solve`/`logdet` give the gradient of a
  Gaussian log-density, so a precision matrix's values can be fit by HMC/NUTS, VI, or
  empirical-Bayes optimization.
- **Factor-once primitives** — `factor_solve` / `sample_gaussian` (CHOLMOD), `lu_factor` /
  `lu_solve_factor` (KLU), and `umf_factor` / `umf_solve_factor` (UMFPACK) factorize once and
  serve an unbounded sequence of solves against the same factor, inside `lax.fori_loop` and
  under `vmap`.
- **Multiple right-hand sides**: `b` may be `(n,)` or `(n, n_rhs)`.
- **PyMC / PyTensor** frontend (optional extra) for the same CHOLMOD core, so PyMC's default
  NUTS sampler can differentiate through the sparse solve and log-determinant without going
  through JAX/XLA.

## Installation

```bash
conda env create -f environment.yml   # suitesparse, jax, nanobind, cmake, ...
conda activate sparsax
pip install --no-build-isolation .
```

For the PyTensor / PyMC frontend:

```bash
pip install "sparsax[pytensor]"
```

> `jax.config.update("jax_enable_x64", True)` is required — CHOLMOD and KLU are float64.

## Quick start: CHOLMOD (symmetric positive definite `A`)

```python
import jax
import jax.numpy as jnp
import numpy as np
import sparsax

jax.config.update("jax_enable_x64", True)   # required: CHOLMOD is float64

# SPD matrix in COO form. Entries with Ai <= Aj are used (upper triangle);
# pass the full symmetric matrix or just its upper triangle.
Ai = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
Aj = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
Ax = jnp.array([4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0])
b  = jnp.array([1.0, 2.0, 3.0])

x  = sparsax.solve(Ai, Aj, Ax, b)          # eager
ld = sparsax.logdet(Ai, Aj, Ax, n=3)       # log|A| from the same factorization
z  = sparsax.selinv(Ai, Aj, Ax, n=3)       # (A^-1) on A's sparsity pattern

@jax.jit
def gibbs_step(Ax, b):
    x  = sparsax.solve(Ai, Aj, Ax, b)      # full CHOLMOD speed, no callbacks
    ld = sparsax.logdet(Ai, Aj, Ax, n=3)   # factorization shared with solve
    return x, ld
```

## Quick start: KLU (general non-symmetric `A`)

```python
import jax
import jax.numpy as jnp
import numpy as np
import sparsax

jax.config.update("jax_enable_x64", True)

# General (non-symmetric) matrix in COO form. The full matrix is used —
# no upper-triangle folding. A common case is A = I - rho * W.
n = 3
Ai = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
Aj = np.array([0, 1, 0, 1, 1, 2], dtype=np.int32)
Ax = jnp.array([1.0, -0.2, -0.3, 1.0, -0.1, 1.0])   # I - rho W, say
b  = jnp.array([1.0, 2.0, 3.0])

x  = sparsax.lu_solve(Ai, Aj, Ax, b)        # eager
ld = sparsax.lu_logdet(Ai, Aj, Ax, n=n)     # log|det(A)| from the KLU factor

@jax.jit
def step(Ax, b):
    return sparsax.lu_solve(Ai, Aj, Ax, b)  # full KLU speed in JIT
```

KLU shares the same caching story as CHOLMOD: the sparsity pattern's fill-reducing analysis is
computed once and cached, and under `jax.vmap` the whole batch is solved in a single native FFI
call. `lu_solve` is differentiable in `Ax` and `b` (transpose solve); `lu_logdet` is forward-only
(the non-symmetric logdet gradient is the full selected inverse, which KLU does not expose —
use `logdet` for symmetric `A` when a gradient is needed).

## Two LU backends: KLU vs UMFPACK

`umf_*` mirrors `lu_*` function for function — same arguments, same results, same JIT / `vmap` /
autodiff behaviour — over SuiteSparse **UMFPACK** instead of KLU:

```python
x  = sparsax.umf_solve(Ai, Aj, Ax, b)       # drop-in for lu_solve
ld = sparsax.umf_logdet(Ai, Aj, Ax, n=n)    # log|det(A)|, from get_determinant
```

They differ only in which graph they are fast on, and the difference runs both ways:

- **KLU** has a third stage UMFPACK lacks — `klu_analyze` → `klu_factor` → **`klu_refactor`** —
  whose refactor reuses the symbolic analysis *and the pivot ordering*, so a refactor at a new
  ρ is pure numeric work. That saves 74–86% of a full factorization while the graph is sparse.
- **UMFPACK** re-pivots inside every `numeric()` call (it is multifrontal with partial
  pivoting, so the pivot sequence depends on the values), but it is frontal and uses BLAS3
  where KLU is a scalar circuit-simulation code with no supernodal path. Once fronts get wide,
  that wins by a lot.

Measured through `sparsax` on row-standardised distance-band graphs at *n* = 3,000, timing
`logdet` over 12 values of ρ on one pattern:

| mean degree | nnz | `lu_logdet` (ms) | `umf_logdet` (ms) | winner |
|---:|---:|---:|---:|:--|
| 2.0 | 9,134 | **0.37** | 0.79 | KLU, 2.1× |
| 7.9 | 26,662 | **1.03** | 2.77 | KLU, 2.7× |
| 21.2 | 66,750 | 7.35 | **6.15** | UMFPACK, 1.2× |
| 65.2 | 198,726 | 48.10 | **15.12** | UMFPACK, 3.2× |
| 160.0 | 483,092 | 154.19 | **85.03** | UMFPACK, 1.8× |

> **Route by measurement, not by a threshold.** The crossover **moves with *n*** — mean degree
> ≈30 at *n* = 1,500 falling to ≈15 at *n* = 12,000 — so a hardcoded cutoff is wrong at both
> ends. A factorization is being computed anyway: time one with each backend and route on the
> result. The probe costs roughly 5–10% of a ρ-node budget, and it is dearest exactly where the
> choice matters least.

One further asymmetry: `umf_logdet` reads the determinant straight out of
`umfpack_*_get_determinant`, in mantissa/exponent form (`det = Mx · 10^Ex`). KLU exposes no
equivalent, so `lu_logdet` reconstructs it from `U`'s diagonal and the row scale factors as a
sum of *n* logarithms. Both are overflow-safe and agree to ~13 digits; UMFPACK's is the more
accurate of the two at large *n*, since it does not accumulate rounding over *n* terms
(at *n* = 4,000 with `det = 1e4000`, UMFPACK is exact to printed precision and the log-sum
drifts by ~3e-14 relative). Neither LU logdet is differentiable — that gradient is the full
selected inverse, which neither backend exposes.

Iterative refinement is **off** on the UMFPACK path (its own default is two steps), so an
`umf_solve` costs the same triangular work as its KLU counterpart and the two can be timed
against each other fairly.

## How it works

`solve` and `logdet` (and their `lu_*` / `umf_*` counterparts) are XLA FFI custom calls into
CHOLMOD, KLU, and UMFPACK. Each backend keeps its own registry of symbolic analyses keyed on the
sparsity pattern, so repeated calls
with the same pattern only pay for the numeric refactorization — and calls with unchanged
values skip even that, sharing one factorization between `solve` and `logdet`. There are no
handles to manage and nothing to pass through JIT boundaries; the caching is transparent.

For the "hold a numeric factor open" case — e.g. a shift-invert Krylov basis that reuses one
factorization across an unbounded sequence of solves whose RHS depends on the previous solve's
output — `factor` / `solve_factor` / `logdet_factor` (CHOLMOD), `lu_factor` /
`lu_solve_factor` / `lu_logdet_factor` (KLU), and `umf_factor` / `umf_solve_factor` /
`umf_logdet_factor` (UMFPACK) return an opaque token carrying a strong reference
to the numeric factor, guaranteeing reuse inside `lax.fori_loop` without relying on a host-side
cache surviving JIT-compiled iterations.

## Factor once, do everything: `factor_solve` / `sample_gaussian`

`solve` and `logdet` are separate primitives, so a Gibbs sweep that needs a posterior mean, a
correlated draw, and a log-determinant factors the *same* A several times — and under `vmap`
that is one factorization per solve **per batch element**. `factor_solve` factors A once and
serves every requested solve (each a *chain* of `MODE_*` codes) plus an optional `logdet` from
that single factor. Under `vmap` it lowers to one batched FFI call that factors **once per
element**, whatever the number of chains.

```python
# Gibbs Gaussian step: posterior mean, a draw ~ N(mean, A^-1), and log|A|,
# from ONE factorization. eta = mean + P' L^-T z  (since A = P' L L' P).
eta, mean, ld = sparsax.sample_gaussian(Ai, Aj, Ax, b, z, want_logdet=True)

# ...or spell it out with the general primitive:
(mean, w), ld = sparsax.factor_solve(
    Ai, Aj, Ax,
    [(b, sparsax.MODE_A),                             # A^-1 b
     (z, (sparsax.MODE_LT, sparsax.MODE_PT))],        # P' L^-T z  (chain)
    want_logdet=True)
eta = mean + w
```

Each `rhs` entry is `(b, modes)` where `modes` is one `MODE_*` or a sequence applied left to
right. `sparsax.factorization_count()` reports how many real factorizations have happened —
handy for confirming the fusion. Benchmarked Gibbs draw (mean + sample + logdet) vmapped over a
batch of *different* A's: **4× fewer factorizations and ~3.3–3.6× faster** than issuing the
separate `solve`/`logdet` primitives. `factor_solve` is forward-only (no autodiff rule); use
`solve`/`logdet` when you need gradients.

The KLU analogue is `lu_factor` + `lu_solve_factor` (+ `lu_logdet_factor`): factorize once,
then issue an unbounded sequence of solves (and a logdet) against the same LU factor inside a
JIT-compiled loop, without re-hashing `Ax` or risking LRU eviction mid-sweep.

## Gradients & the selected inverse

`solve` and `logdet` are the two halves of a Gaussian log-density's gradient, so a precision
matrix `A(θ)` with a fixed pattern and θ-dependent values can be fit by gradient-based
inference — HMC/NUTS (via numpyro/blackjax, or PyMC's JAX sampling backend), VI, or
empirical-Bayes/MAP optimization:

```python
def neg_log_post(Ax):                                  # up to constants
    quad = b @ sparsax.solve(Ai, Aj, Ax, b)            # b' A^-1 b   (solve VJP)
    return 0.5 * quad - 0.5 * sparsax.logdet(Ai, Aj, Ax, n)   # log|A| (logdet VJP)

grad_Ax = jax.grad(neg_log_post)(Ax)                   # works under jit / vmap
```

`logdet`'s reverse-mode rule uses that `d log|A| / dA = A^{-1}`, evaluated **only at `A`'s
sparsity pattern** by Takahashi's selected-inversion recurrence over the Cholesky factor —
never the dense inverse. That quantity is exposed directly:

```python
z   = sparsax.selinv(Ai, Aj, Ax, n)   # z[k] == (A^-1)[Ai[k], Aj[k]]
var = z[Ai == Aj]                      # diag(A^-1): Gaussian marginal variances
```

`selinv` shares the factorization cache, is JIT-compilable and `vmap`-able, and costs one
selected-inversion pass over the factor (`O(nnz(L))`-ish), not `n` solves. `factor_solve` /
`sample_gaussian` remain forward-only. `lu_solve` and `umf_solve` carry the analogous
reverse-mode rule (a transpose solve for the VJP); neither `lu_logdet` nor `umf_logdet` is
differentiable.

## PyMC / PyTensor (NUTS)

The same CHOLMOD core is exposed as a **PyTensor** frontend for PyMC's default backend, so
gradient-based samplers (NUTS) can differentiate through the sparse solve and log-determinant
without going through JAX/XLA. It's an optional extra — the base package stays JAX-only:

```python
import pytensor.tensor as pt
import sparsax.pytensor as cjpt

Ax = pt.dvector("Ax")                 # the precision-matrix values (θ-dependent)
# Gaussian log-density (up to constants); grad flows into Ax
logp = -0.5 * pt.dot(b, cjpt.solve(Ai, Aj, Ax, b)) + 0.5 * cjpt.logdet(Ai, Aj, Ax, n)
g = pt.grad(logp, Ax)                 # solve VJP + logdet (selected-inverse) VJP
```

`cjpt.solve`, `cjpt.logdet`, and `cjpt.selinv` mirror the JAX functions and carry the same
reverse-mode rules (`solve` in `Ax`/`b`, `logdet` in `Ax`); the pattern `(Ai, Aj)` is
non-differentiable data. On PyTensor ≥ 3.1 the gradient routes through `Op.pullback`; older
versions use `grad`.

> **Prefer the C backend, not numba.** These Ops implement a pure-Python `perform` (it calls
> the native core, which holds no GIL and does the real work — the CHOLMOD call dominates).
> PyTensor's **C backend** (`FAST_RUN`, the default) calls that `perform` directly with no
> penalty. PyTensor's **numba backend** cannot JIT a Python `perform`, so it falls back to
> *object mode* and prints a `UserWarning` on every call — functionally correct but with
> per-call overhead. Keep PyMC on its default sampler (C backend) to use these Ops, and switch
> to the JAX frontend above if you deliberately run PyMC's JAX/numba linker.

## JAX sparse (`BCOO`)

JAX's native sparse type is `jax.experimental.sparse.BCOO`, whose `.indices` is `(nnz, 2)` and
`.data` is `(nnz,)`. Convenience wrappers accept one directly (every backend alike):

```python
from jax.experimental import sparse as jsparse
A = jsparse.BCOO.fromdense(A_dense)         # or build however you like

# CHOLMOD
x  = sparsax.solve_bcoo(A, b)
ld = sparsax.logdet_bcoo(A)
x  = sparsax.update_solve_bcoo(A, C, b)     # rank-k update/downdate

# KLU
x  = sparsax.lu_solve_bcoo(A, b)
ld = sparsax.lu_logdet_bcoo(A)

# UMFPACK
x  = sparsax.umf_solve_bcoo(A, b)
ld = sparsax.umf_logdet_bcoo(A)
```

## License

`sparsax`'s own source is **BSD-3-Clause** (`LICENSE.txt`).

**Binary distributions are not.** The extension links SuiteSparse, and parts of
SuiteSparse are **GPL-2.0-or-later** — UMFPACK in full, plus CHOLMOD's MatrixOps,
Modify, and Supernodal modules. The wheels on PyPI, the conda-forge package, and
anything you build yourself are combined works that bundle that code, so
**redistributing a built `sparsax` carries GPL obligations**, not just BSD ones.
BSD-3-Clause is GPL-compatible, so making the combination is fine — but the result
is not permissive.

This is not incidental: `update_solve` is built on CHOLMOD's `cholmod_updown`
(Modify), CHOLMOD's default `AUTO` strategy selects the Supernodal factorization,
and the whole `umf_*` family is UMFPACK.

*Using* `sparsax` — running it, doing research with it, publishing results — is
unaffected; the GPL governs distribution, not use. Depending on it from your own
permissively-licensed source is likewise fine. It matters when you ship a binary
artifact with SuiteSparse inside it, especially in a proprietary product.

See [`NOTICE.md`](NOTICE.md) for the per-component breakdown and what a
permissive-only build would cost. None of this is legal advice; if you are
redistributing commercially, get it checked properly.
