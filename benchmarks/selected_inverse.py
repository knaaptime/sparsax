"""Factorization, selected inverse, and a log-density with its gradient, on new values each time."""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sparsax
from problems import grid, knn, upper_coo

QUICK = "--quick" in sys.argv
PROBLEMS = (
    [("2-D n=900", grid(30, 2), 5), ("3-D n=1k", grid(10, 3), 5)]
    if QUICK
    else [
        ("2-D n=10k", grid(100, 2), 7),
        ("2-D n=90k", grid(300, 2), 5),
        ("2-D n=360k", grid(600, 2), 3),
        ("3-D n=8k", grid(20, 3), 5),
        ("3-D n=43k", grid(35, 3), 3),
        ("kNN(8) n=100k", knn(100_000, 8), 3),
    ]
)

for name, A, reps in PROBLEMS:
    Ai, Aj, Ax = upper_coo(A)
    n = A.shape[0]
    b = jnp.ones(n)

    def logp(ax):
        x = sparsax.solve(Ai, Aj, ax, b)
        return -0.5 * b @ x + 0.5 * sparsax.logdet(Ai, Aj, ax, n)

    value_and_grad = jax.jit(jax.value_and_grad(logp))
    jax.block_until_ready(value_and_grad(jnp.asarray(Ax)))
    factor, selinv, both = [], [], []
    for r in range(reps):
        ax = jnp.asarray(Ax * (1.0 + 0.01 * (r + 1)))
        t0 = time.perf_counter()
        jax.block_until_ready(sparsax.logdet(Ai, Aj, ax, n))
        factor.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        jax.block_until_ready(sparsax.selinv(Ai, Aj, ax, n))
        selinv.append(time.perf_counter() - t0)
        ax = jnp.asarray(Ax * (2.0 + 0.01 * (r + 1)))
        t0 = time.perf_counter()
        jax.block_until_ready(value_and_grad(ax))
        both.append(time.perf_counter() - t0)
    f, s, v = (1e3 * np.median(t) for t in (factor, selinv, both))
    print(
        f"{name:14s} factor {f:8.1f} ms   selinv {s:8.1f} ms ({s / f:4.1f}x)   "
        f"logp+grad {v:8.1f} ms",
        flush=True,
    )
