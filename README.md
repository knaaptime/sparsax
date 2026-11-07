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

- **CHOLMOD** — `solve`, `logdet`, `selinv` (selected inverse), `rcond`, `factor_solve`,
  `sample_gaussian`, `update_solve` (rank-k update/downdate) for symmetric positive definite
  `A`, and `argmax_logprob` (GHK choice probabilities of a sparse-precision Gaussian).
- **KLU** — `lu_solve`, `lu_logdet`, `lu_slogdet`, `lu_rcond`, `lu_factor` / `lu_solve_factor` for general
  non-symmetric `A` (e.g. `A = I − ρW` with a row-standardised spatial weights matrix).
- **UMFPACK** — `umf_solve`, `umf_logdet`, `umf_factor` / `umf_solve_factor`: the same API
  over SuiteSparse's multifrontal LU, for the same non-symmetric `A`. The two LU backends are
  interchangeable and differ only in which graph they are fast on — see
  *Two LU backends: KLU vs UMFPACK* below.
- **JIT / `lax.scan`**: the symbolic analysis is computed once and reused across iterations.
- **`vmap`**: `jax.vmap(solve)` lowers to a *single* native FFI call that loops over the
  batch in C++ (reusing the cached analysis), rather than XLA per-iteration dispatch. A batch
  of right-hand sides against fixed values is one multi-column solve on one factor.
  Composes with `grad` (`vmap(grad(solve))` batches too).
- **Autodiff**: `solve` has a custom reverse-mode VJP in both `Ax` and `b`; `logdet` has one
  in `Ax` via the selected inverse. `lu_solve` is differentiable in `Ax` and `b` (transpose
  solve), as is `umf_solve`; `lu_logdet` and `umf_logdet` are not differentiable. Together
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
call. `lu_solve` is differentiable in `Ax` and `b` (transpose solve); `lu_logdet` is not differentiable
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

The crossover moves with *n*, from a mean degree of about 30 at *n* = 1,500 to about 15 at
*n* = 12,000, so no cutoff on degree works at both ends. One number does decide it: KLU's flops
per factor entry, roughly the size of a typical front. `lu_backend` computes it and returns the
faster backend's functions:

```python
lu = sparsax.lu_backend(Ai, Aj, n)          # workload="logdet" if that is the main use
x  = lu.solve(Ai, Aj, Ax, b)                # as lu_solve; also factor, logdet, rcond, ...
```

It chooses UMFPACK above 66 flops per entry, or 47 for a log-determinant, where UMFPACK's lead
starts sooner. The choice depends on the pattern alone, not on the values or a timing, so a fixed
seed reproduces the same draws, which timing the two backends would not: a sampler turns a
difference in rounding into a different chain. On 19 rook and k-NN patterns with `n` from 400 to
25,600 it chose the faster backend on 18 for each workload, and the misses cost at most 6%
(`benchmarks/lu_backends.py`, M3 Max). Measuring a pattern costs one KLU factorization, once.

One further asymmetry: `umf_logdet` reads the determinant straight out of
`umfpack_*_get_determinant`, in mantissa/exponent form (`det = Mx · 10^Ex`). KLU exposes no
equivalent, so `lu_logdet` reconstructs it from `U`'s diagonal and the row scale factors as a
sum of *n* logarithms. Both are overflow-safe and agree to ~13 digits; UMFPACK's is the more
accurate of the two at large *n*, since it does not accumulate rounding over *n* terms
(at *n* = 4,000 with `det = 1e4000`, UMFPACK is exact to printed precision and the log-sum
drifts by ~3e-14 relative). Neither LU logdet is differentiable — that gradient is the full
selected inverse, which neither backend exposes.

`lu_slogdet` and `umf_slogdet` also return the sign, as `numpy.linalg.slogdet` does, which shows
when a parameter has carried `I − ρW` across a singularity. `rcond`, `lu_rcond` and `umf_rcond`
read a cheap reciprocal condition estimate off the factor: the smallest pivot over the largest
(squared for CHOLMOD, which reports it that way). It is a rough indicator, not a bound, and the
values are not comparable across backends, because the LU backends scale the rows first.

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

Tokens are plain arrays with no destructor, so each backend keeps only its 64 newest
(`set_token_cache_size` changes it) and releases the rest; a released token raises a
"stale factor token" error. Hold a token across at most that many newer factorizations of its
backend, or refactor. A token works only with its own backend's functions and a `b` of the
factor's size; anything else raises an error.

Once a pattern's value cache is full (`set_lu_cache_size`, `set_umf_cache_size`,
`set_num_cache_size`; 32 by default), a factorization at new values reuses the storage of the
factor it evicts, provided no token or in-flight solve still holds it: CHOLMOD refactors it in
place, and KLU runs `klu_refactor` on its pivot sequence (1.2–1.8× cheaper than `klu_factor`),
falling back to a full factorization if the reciprocal pivot growth collapses. UMFPACK's symbolic
analysis uses the values of the first call on a pattern, which lets it choose the symmetric
strategy for `I − ρW` (1.35–1.95× faster numeric factorizations than a pattern-only analysis).

CHOLMOD's analysis tries AMD, and METIS when AMD's fill looks poor. `set_options(ordering="nesdis")`
uses nested dissection instead (`"amd"`, `"metis"` and `"best"` also work). On 2-D grids and on
k-NN and distance-band graphs with 30,000 to 400,000 nodes, that cut fill by 8–21%, the time of
a factorization by 5–18% and of a selected inverse by 4–23%, for an analysis 5–11× slower, which
is paid once per pattern. It changes no answers, and it stays opt-in because the gain is modest
and small problems may never repay the analysis.

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
separate `solve`/`logdet` primitives. `factor_solve` has no autodiff rule; use
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

`selinv` shares the factorization cache and is JIT-compilable and `vmap`-able. On a supernodal
factor, CHOLMOD's automatic choice except on very sparse matrices, it treats each supernode as
one dense block with BLAS and costs 1.2–1.9× a factorization (M-series macOS, grids and a k-NN
graph up to `n = 360,000`): 53 ms on a 2-D grid with `n = 90,000`, 132 ms on a 3-D grid with
`n = 43,000`. A simplicial factor runs one column at a time. So does every factor on Windows,
where sparsax has no direct BLAS link: it first converts a supernodal factor to simplicial, and
those two grids take 556 ms and 7.8 s. `factor_solve` / `sample_gaussian` have no autodiff rule.

`solve`, `lu_solve` and `umf_solve` differentiate by implicit differentiation
(`jax.lax.custom_linear_solve`): every derivative of `A x = b` is another solve on the same
cached factor (a transpose solve for the LU backends), so derivatives of **any order** work in
forward and reverse mode, including exact Hessians and Hessian-vector products for Newton and
trust-region methods. `logdet`'s rule is first-order only, and neither `lu_logdet` nor
`umf_logdet` is differentiable.

## Choice probabilities: `argmax_logprob`

`argmax_logprob` scores which component of a sparse-precision Gaussian is largest:
`log P(argmax_k x_k = j)` for `x ~ N(mean, A⁻¹)`. That is the choice probability of a
multinomial probit whose utilities form a Gaussian Markov random field, such as a spatial
multinomial probit with `A = (I − ρW)'(I − ρW)`. It runs the GHK simulator on the sparse factor
of `A` rather than on the dense covariance of the utility differences, so a draw costs
`O(nnz(L))` instead of `O(n²)`: the differences' precision is sparse minus a rank-one term with
a closed-form factor, and each path deletes its chosen index from one cached factor with
`cholmod_rowdel`.

```python
# mean: one row per decision-maker; chosen[p]: the index they picked
masked = mean.at[jnp.arange(len(chosen)), chosen].set(-jnp.inf)
comp = jnp.argsort(-masked, axis=1)[:, :30]        # the 30 likeliest rivals, drawn first
logp = sparsax.argmax_logprob(Ai, Aj, Ax, mean, chosen, comp, n_draws=256, keys=ids)
grad = jax.grad(lambda m: sparsax.argmax_logprob(Ai, Aj, Ax, m, chosen, comp).sum())(mean)
```

Draws are a fixed function of `(seed, keys[p], draw, step)`, so the simulated log-likelihood is
reproducible across calls and batchings and smooth in `mean`, which simulated maximum
likelihood needs. GHK's variance depends on the order of the draws, and it grows quickly with
`n` unless the rivals most likely to beat the chosen index come first: pass them as
`competitors`, which are moved to the end of the elimination order (one `cholmod_rowdel` each
plus a dense block) while everything else keeps the fill-reducing order.

The first 256 steps of each path take scrambled Sobol points instead of independent draws,
competitors first, scrambled per `(seed, keys[p])` so that the estimate stays unbiased. With
`n = 4,900`, 30 competitors and 128 draws, the median variance of the log-probability fell 34×
when the means spread widely across alternatives (standard deviation 2), 5.4× at 1 and 2.2× at
0.5, the hardest case; the RMS error against a 16,384-draw reference fell from 0.12 to 0.026,
from 0.43 to 0.24, and from 1.8 to 1.4. The gradient's variance fell 9.5× in the first case.
This takes about 5% more time, and up to about 30% more on small problems, where more of the
steps are scrambled. Keep `n_draws` a power of two, which is where the points balance:
`draws="iid"` restores independent draws.

The gradient with
respect to `mean` is a reverse sweep through the recursion at about 1.8× the cost of the value.
There is no gradient with respect to `A`, so one that flows through `Ax` raises an error: wrap
`Ax` in `jax.lax.stop_gradient` to hold it fixed, or difference in the parameter of `A`, which
the fixed draws keep smooth. With `n = 5,000`, 4,000 paths, 100 draws and 30 competitors, a value
takes about 10 s and a value-and-gradient about 17 s on 16 threads (M-series macOS).

## When a factorization fails

A factorization can fail on the numbers: the matrix is not positive definite, it is singular,
or a value is not finite. By default that raises an error, which aborts the whole computation,
`vmap` batch included. A sampler needs something else: it should reject a proposal outside the
valid region, and its other chains should carry on.

```python
sparsax.set_options(on_failure="nan")
```

With this option the failed element returns NaN and the rest of the batch is unaffected, in all
three backends and in the PyTensor frontend. Its gradient is NaN too, and a token from a failed
factorization gives NaN from every function that takes it. Bad arguments, stale tokens and
running out of memory still raise. The option applies to the whole process, takes effect at
once in compiled functions, and leaves the caches alone.

## Threads

The native libraries are called without a global lock, so concurrent calls (one per MCMC
chain, say) run in parallel. The wheels link a multithreaded BLAS (Accelerate on macOS, the
threaded build of OpenBLAS on Linux and Windows) but no OpenMP runtime, so they cannot clash with
the OpenMP runtime of another library in the same process. With many concurrent callers, cap BLAS
threads to avoid oversubscription, e.g. `OPENBLAS_NUM_THREADS=1` on Linux and Windows or
`VECLIB_MAXIMUM_THREADS=1` on macOS.

A single solve with many right-hand sides (128 columns or more) is itself split into
64-column panels solved on every hardware thread, for all three backends;
`SPARSAX_NUM_THREADS` caps it. The factor is only read, and each worker brings its own
workspaces. With 4,000 right-hand sides on an `n = 3,600` k-NN system, 16 threads take
the solve from 0.24 s to 0.018 s (KLU), 0.63 s to 0.061 s (UMFPACK), and 0.21 s / 0.47 s
to 0.026 s / 0.032 s (CHOLMOD simplicial / supernodal).

A `vmap` over the values of `A` runs on several threads too, with one factor per element, as
long as each thread gets enough work to repay starting it (from a batch of two at
`n = 900`), and `SPARSAX_NUM_THREADS` caps this as well. A vmapped log-density and its gradient
over 16 chains went from 118 ms to 17 ms on a 2-D grid with `n = 10,000`, from 1.45 s to 0.20 s
at `n = 90,000`, from 1.60 s to 0.26 s on a k-NN graph with `n = 100,000`, and from 3.5 s to
0.91 s on a 3-D grid with `n = 43,000`, whose large supernodes already use BLAS threads (M3
Max). Such a batch counts as many concurrent callers for the advice on BLAS threads above.
Threads asking for values already being factored wait for that factor rather than repeat the
work, so chains started at one point factor once. The scripts in `benchmarks/` produce the
timings in this README.

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

These Ops implement a Python `perform` that calls the native core, which releases the GIL while
it works, so other Python threads keep running. From PyTensor 3 the default linker is Numba, even
with `mode="FAST_RUN"`. It cannot compile a Python `perform`, so it runs these Ops in object mode
and prints a `UserWarning` when it compiles the function, not on each call; the results are
correct (checked on PyTensor 3.3.2). `mode="JAX"` does not work, because the Ops register no JAX
conversion: to run on JAX, use the JAX frontend above.

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
