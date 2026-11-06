"""jit(vmap(value_and_grad(logp))) over chains, on one thread and on several."""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sparsax
from problems import grid, knn, upper_coo

SERIAL, DEFAULT = 2**62, -1  # see sparsax_cpp.set_batch_work_per_thread
QUICK = "--quick" in sys.argv
PROBLEMS = (
    [("2-D n=900", grid(30, 2))]
    if QUICK
    else [
        ("2-D n=900", grid(30, 2)),
        ("2-D n=10k", grid(100, 2)),
        ("2-D n=90k", grid(300, 2)),
        ("3-D n=8k", grid(20, 3)),
        ("3-D n=43k", grid(35, 3)),
        ("kNN(8) n=100k", knn(100_000, 8)),
    ]
)

for name, A in PROBLEMS:
    Ai, Aj, Ax = upper_coo(A)
    n = A.shape[0]
    b = jnp.ones(n)
    Ax = jnp.asarray(Ax)

    def logp(scale):
        x = sparsax.solve(Ai, Aj, scale * Ax, b)
        return -0.5 * b @ x + 0.5 * sparsax.logdet(Ai, Aj, scale * Ax, n)

    evaluate = jax.jit(jax.vmap(jax.value_and_grad(logp)))
    row = []
    for chains in (4, 8, 16, 32):
        base = jnp.linspace(1.0, 1.5, chains)
        jax.block_until_ready(evaluate(base))
        times = {}
        for label, setting in (("serial", SERIAL), ("threads", DEFAULT)):
            sparsax._cpp.set_batch_work_per_thread(setting)
            samples = []
            for r in range(15 if n <= 10_000 else 5):
                offset = 1.0 + 1e-3 * (r + 1) + (0.5 if label == "threads" else 0.0)
                t0 = time.perf_counter()
                jax.block_until_ready(evaluate(base * offset))
                samples.append(time.perf_counter() - t0)
            times[label] = np.median(samples)
        row.append(
            f"{chains}: {times['serial'] * 1e3:7.1f} -> {times['threads'] * 1e3:7.1f} ms "
            f"({times['serial'] / times['threads']:4.1f}x)"
        )
    print(f"{name:14s} " + "  ".join(row), flush=True)
