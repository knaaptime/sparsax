"""Does sparsax.lu_backend pick the faster LU backend?  A factorization and one solve or logdet."""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import sparsax
from scipy.spatial import cKDTree

QUICK = "--quick" in sys.argv
REPS = 3 if QUICK else 7


def rook(side):
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    W = sp.kron(path, eye) + sp.kron(eye, path)
    return (sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W).tocoo()


def directed_knn(n, k, seed=0):
    points = np.random.default_rng(seed).uniform(size=(n, 2))
    _, neighbours = cKDTree(points).query(points, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    return sp.coo_matrix(
        (np.full(n * k, 1.0 / k), (rows, neighbours[:, 1:].ravel())), shape=(n, n)
    )


PATTERNS = (
    [("rook n=400", rook(20)), ("kNN(8) n=400", directed_knn(400, 8))]
    if QUICK
    else [
        ("rook n=1,600", rook(40)),
        ("rook n=6,400", rook(80)),
        ("rook n=25,600", rook(160)),
        *[
            (f"kNN({k}) n={n:,}", directed_knn(n, k))
            for n in (400, 1600, 6400)
            for k in (4, 8, 16, 32, 64, 96)
            if k < n / 8
        ],
    ]
)


def timed(fn, rows, cols, W, n, b):
    """Median time of `fn` on new values each time, so that every call factors."""
    samples = []
    for r in range(REPS):
        rho = 0.3 + 0.05 * r
        vals = jnp.asarray(np.concatenate([np.ones(n), -rho * W.data]))
        t0 = time.perf_counter()
        jax.block_until_ready(fn(rows, cols, vals, b))
        samples.append(time.perf_counter() - t0)
    return np.median(samples)


right = {"solve": 0, "logdet": 0}
cost = {"solve": [], "logdet": []}
for name, W in PATTERNS:
    n = W.shape[0]
    rows = np.concatenate([np.arange(n), W.row]).astype(np.int32)
    cols = np.concatenate([np.arange(n), W.col]).astype(np.int32)
    work = sparsax._cpp.lu_work(rows, cols, n)
    density = work["flops"] / (work["lnz"] + work["unz"])
    b = jnp.ones(n)
    line = f"{name:18s} d={density:8.1f} "
    for workload, klu_fn, umf_fn in (
        ("solve", sparsax.lu_solve, sparsax.umf_solve),
        (
            "logdet",
            lambda i, j, x, _: sparsax.lu_logdet(i, j, x, n),
            lambda i, j, x, _: sparsax.umf_logdet(i, j, x, n),
        ),
    ):
        # Each pass starts cold, so the cache cannot favour either backend.
        sparsax.clear_cache()
        klu = timed(klu_fn, rows, cols, W, n, b)
        umf = timed(umf_fn, rows, cols, W, n, b)
        chosen = sparsax.lu_backend(rows, cols, n, workload=workload).name
        times = {"klu": klu, "umfpack": umf}
        right[workload] += chosen == min(times, key=times.get)
        cost[workload].append(times[chosen] / min(klu, umf))
        line += (
            f"| {workload:6s} KLU {klu * 1e3:9.2f} UMF {umf * 1e3:9.2f} ms "
            f"-> {chosen:7s} ({times[chosen] / min(klu, umf):4.2f}x best) "
        )
    print(line, flush=True)
total = len(PATTERNS)
for workload in right:
    print(
        f"{workload}: faster backend chosen on {right[workload]} of {total}; "
        f"worst cost {max(cost[workload]):.2f}x, median {np.median(cost[workload]):.2f}x"
    )
