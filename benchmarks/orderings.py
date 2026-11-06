"""Each ordering's analysis time, fill, and downstream timings, against "auto"."""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sparsax
from problems import distance_band, grid, knn, upper_coo

QUICK = "--quick" in sys.argv
PROBLEMS = (
    [("2-D grid n=2,500", grid(50, 2))]
    if QUICK
    else [
        ("2-D grid n=250k", grid(500, 2)),
        ("3-D grid n=43k", grid(35, 3)),
        ("kNN(8) n=100k", knn(100_000, 8)),
        ("kNN(8) n=400k", knn(400_000, 8)),
        ("band degree~30 n=30k", distance_band(30_000, 30)),
        ("kNN(30) n=30k", knn(30_000, 30)),
    ]
)

for name, A in PROBLEMS:
    Ai, Aj, Ax = upper_coo(A)
    n = A.shape[0]
    print(f"{name}: nnz(A) = {A.nnz}", flush=True)
    base = None
    for ordering in ("auto", "amd", "metis", "nesdis", "best"):
        sparsax.set_options(ordering=ordering)
        t0 = time.perf_counter()
        info = sparsax._cpp.analysis_info(Ai, Aj, n)
        analyze = time.perf_counter() - t0
        factor, selinv = [], []
        for r in range(5):
            ax = jnp.asarray(Ax * (1.0 + 0.01 * (r + 1)))
            t0 = time.perf_counter()
            jax.block_until_ready(sparsax.logdet(Ai, Aj, ax, n))
            factor.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            jax.block_until_ready(sparsax.selinv(Ai, Aj, ax, n))
            selinv.append(time.perf_counter() - t0)
        f, s = np.median(factor), np.median(selinv)
        if base is None:
            base = (info["lnz"], f, s)
        print(
            f"  {ordering:7s} [{info['ordering']:6s}] analyze {analyze * 1e3:7.0f} ms  "
            f"nnz(L) {info['lnz'] / base[0]:4.2f}x  factor {f / base[1]:4.2f}x  "
            f"selinv {s / base[2]:4.2f}x",
            flush=True,
        )
sparsax.set_options(ordering="auto")
