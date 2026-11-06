# Benchmarks

These scripts produce the timings quoted in the top-level README. They are not run in CI.
Pass `--quick` for a smoke test on small problems. The numbers depend on the machine and its
BLAS; those in the README come from an Apple M3 Max with Accelerate.

| Script | What it measures |
|---|---|
| `selected_inverse.py` | factorization, `selinv`, and a log-density with its gradient |
| `chains.py` | a `vmap` over the values of `A`, on one thread and on several |
| `shared_factor.py` | cached solves from several Python threads on one large pattern |
| `orderings.py` | fill and timings under each `set_options(ordering=...)` |
| `lu_backends.py` | whether `lu_backend` picks the faster of KLU and UMFPACK |
| `ghk_draws.py` | Sobol against independent draws in `argmax_logprob` |
| `accelerate_dpotri.cpp` | why `selinv` does not call `dpotri`: threaded Accelerate returns wrong values |
