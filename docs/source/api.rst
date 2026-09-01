.. _api_ref:

.. currentmodule:: sparsax

API reference
=============

The public API is split across two modules: the JAX frontend in
:mod:`sparsax` and the optional PyTensor frontend in :mod:`sparsax.pytensor`.

JAX frontend (``sparsax``)
--------------------------

Sparse Cholesky (CHOLMOD) solvers for symmetric positive definite matrices,
exposed as XLA FFI custom calls that run at full native speed inside
``@jax.jit`` (and ``lax.scan`` / ``lax.fori_loop``). Symbolic analysis is
cached on the sparsity pattern, so repeated solves with the same pattern only
pay for the numeric refactorization.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   solve
   logdet
   selinv
   factor_solve
   sample_gaussian
   update_solve
   solve_bcoo
   logdet_bcoo
   update_solve_bcoo
   set_options
   clear_cache
   cache_size
   factorization_count

Factor tokens (hold a numeric factor open)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Factor SPD ``A`` once, then issue an unbounded sequence of solves + a logdet
against the held factor without re-hashing ``Ax`` or refactoring. This is the
"hold the factor open" primitive that :func:`factor_solve` (which fuses a
*fixed* set of solves into one call) cannot express: a recurrence
``V_{j+1} = A^{-1}(G V_j)`` (a shift-invert Krylov basis) needs the previous
solve's output as the next solve's input. Forward-only (no autodiff); use
:func:`solve` / :func:`logdet` when gradients are needed.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   factor
   solve_factor
   logdet_factor
   set_num_cache_size

Sparse LU (KLU)
~~~~~~~~~~~~~~~

Sparse LU solver for general (non-symmetric) matrices, e.g.
``A = I - rho W`` for a row-standardised spatial weights matrix. See
`Sparse LU (UMFPACK)`_ below for the second LU backend and when to prefer it.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   lu_solve
   lu_solve_bcoo
   lu_logdet
   lu_logdet_bcoo

KLU factor tokens
~~~~~~~~~~~~~~~~~

The non-symmetric analogue of the CHOLMOD factor tokens: factor general ``A``
once via KLU, then issue an unbounded sequence of solves (``A x = b`` or
``A^T x = b``) and a logdet against the held factor. The token holds a strong
reference to the numeric factor, so the per-pattern LRU cannot evict it while
the token is live. Forward-only (no autodiff).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   lu_factor
   lu_solve_factor
   lu_logdet_factor
   set_lu_cache_size

Sparse LU (UMFPACK)
~~~~~~~~~~~~~~~~~~~

The second LU backend, interchangeable with the KLU functions above — same
arguments, same results, same JIT/vmap/autodiff behaviour — and differing only
in which graph it is fast on. KLU has a refactor stage that reuses its pivot
ordering across value changes and wins several-fold while the graph is sparse;
UMFPACK re-pivots on every numeric factorization but is multifrontal and uses
BLAS3, and wins several-fold once the graph densifies. The crossover falls as
``n`` grows (mean degree around 30 at ``n = 1,500``, around 15 at
``n = 12,000``), so route between them by timing one numeric factorization with
each rather than by a fixed degree threshold.

:func:`umf_logdet` reads ``log|det(A)|`` from ``umfpack_*_get_determinant``,
which returns the determinant in mantissa/exponent form. KLU exposes no
equivalent, so :func:`lu_logdet` reconstructs it from ``U``'s diagonal and the
row scale factors as a sum of ``n`` logarithms. Both are overflow-safe and
agree closely; UMFPACK's is the more accurate at large ``n``, having no
rounding accumulated over ``n`` terms.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   umf_solve
   umf_solve_bcoo
   umf_logdet
   umf_logdet_bcoo

UMFPACK factor tokens
~~~~~~~~~~~~~~~~~~~~~

The UMFPACK counterpart of the KLU factor tokens, with identical semantics:
factor once, then issue an unbounded sequence of solves (``A x = b`` or
``A^T x = b``) and a logdet against the held factor. Forward-only (no autodiff).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   umf_factor
   umf_solve_factor
   umf_logdet_factor
   set_umf_cache_size

Solve modes
~~~~~~~~~~~

Constants selecting which system is solved by :func:`solve` and
:func:`update_solve`. The factorization is ``P'LL'P = A``, so e.g. sampling
from ``N(0, A^{-1})`` is ``solve(..., z, mode=MODE_LT)`` followed by
``solve(..., ., mode=MODE_PT)``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MODE_A
   MODE_LDLT
   MODE_LD
   MODE_DLT
   MODE_L
   MODE_LT
   MODE_D
   MODE_P
   MODE_PT

PyTensor frontend (``sparsax.pytensor``)
----------------------------------------

A second frontend over the same cached CHOLMOD core, for use with PyMC's
default (PyTensor) backend — including gradient-based samplers like NUTS.
These are pure-Python ``Op``\ s whose ``perform`` calls the numpy-callable
core; they do **not** go through JAX/XLA. Install with
``pip install sparsax[pytensor]``.

.. currentmodule:: sparsax.pytensor

.. autosummary::
   :toctree: generated/
   :nosignatures:

   solve
   logdet
   selinv

PyTensor Ops
~~~~~~~~~~~~

The ``Op`` classes backing the convenience functions above. Documented for
users who need to construct or introspect them directly.

.. autoclass:: CholmodSolve
   :no-index:
   :no-undoc-members:

.. autoclass:: CholmodLogdet
   :no-index:
   :no-undoc-members:

.. autoclass:: CholmodSelinv
   :no-index:
   :no-undoc-members:
