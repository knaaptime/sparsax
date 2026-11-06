"""Cached solves from T Python threads on a large pattern: lookups and locks show here."""

import sys
import threading
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import sparsax

n = 20_000 if "--quick" in sys.argv else 1_000_000
A = sp.diags(
    [
        -np.ones(n - 2),
        -np.ones(n - 1),
        np.full(n, 5.0),
        -np.ones(n - 1),
        -np.ones(n - 2),
    ],
    [-2, -1, 0, 1, 2],
).tocoo()
Ai, Aj, Ax = A.row.astype(np.int32), A.col.astype(np.int32), jnp.asarray(A.data)
b = jnp.ones(n)
print(f"nnz = {A.nnz}", flush=True)
for name, solve in (
    ("cholmod", sparsax.solve),
    ("klu", sparsax.lu_solve),
    ("umfpack", sparsax.umf_solve),
):
    f = jax.jit(lambda: solve(Ai, Aj, Ax, b))
    jax.block_until_ready(f())
    rates = {}
    for threads in (1, 2, 4, 8):
        stop = time.perf_counter() + 2.0
        counts = [0] * threads

        def work(i):
            while time.perf_counter() < stop:
                jax.block_until_ready(f())
                counts[i] += 1

        pool = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
        t0 = time.perf_counter()
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        rates[threads] = sum(counts) / (time.perf_counter() - t0)
    print(
        f"{name:8s} "
        + "  ".join(f"{t}T {r:7.1f}/s ({r / rates[1]:.1f}x)" for t, r in rates.items()),
        flush=True,
    )
