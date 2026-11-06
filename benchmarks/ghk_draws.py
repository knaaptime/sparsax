"""Sobol against independent draws in argmax_logprob, on a spatial multinomial probit."""

import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sparsax
from problems import spatial_precision

QUICK = "--quick" in sys.argv
SIDE, PATHS, SEEDS, REFERENCE = (12, 12, 4, 2**10) if QUICK else (70, 60, 12, 2**14)


def setup(rho, spread, competitors):
    A = spatial_precision(SIDE, rho)
    Ai, Aj, Ax = A.row.astype(np.int32), A.col.astype(np.int32), jnp.asarray(A.data)
    n = A.shape[0]
    rng = np.random.default_rng(0)
    mean = np.tile(rng.normal(0, spread, n), (PATHS, 1))
    mean += rng.normal(0, 0.25 * spread, (PATHS, n))
    z = jnp.asarray(rng.normal(size=(n, PATHS)))
    noise = sparsax.sample_gaussian(Ai, Aj, Ax, jnp.zeros((n, PATHS)), z)[0]
    chosen = (mean + np.asarray(noise).T).argmax(1)
    rivals = mean.copy()
    rivals[np.arange(PATHS), chosen] = -np.inf
    comp = np.argsort(-rivals, axis=1)[:, :competitors]
    return Ai, Aj, Ax, jnp.asarray(mean), chosen, comp


def log_p(problem, n_draws, seed, draws):
    Ai, Aj, Ax, mean, chosen, comp = problem
    return np.asarray(
        sparsax.argmax_logprob(
            Ai, Aj, Ax, mean, chosen, comp, n_draws=n_draws, seed=seed, draws=draws
        )
    )


for rho, spread in ((0.5, 2.0), (0.5, 1.0), (0.5, 0.5), (0.8, 2.0)):
    problem = setup(rho, spread, 30)
    reference = log_p(problem, REFERENCE, 991, "sobol")
    print(
        f"rho={rho} sd(means)={spread}, mean log p {reference.mean():.2f}", flush=True
    )
    for n_draws in (32, 128, 512):
        reps = {
            draws: np.stack([log_p(problem, n_draws, s, draws) for s in range(SEEDS)])
            for draws in ("iid", "sobol")
        }
        ratio = reps["iid"].var(0, ddof=1) / reps["sobol"].var(0, ddof=1)
        rms = {d: np.sqrt(np.mean((r - reference) ** 2)) for d, r in reps.items()}
        print(
            f"  {n_draws:4d} draws: variance reduction {np.median(ratio):6.1f}x   "
            f"RMS error {rms['iid']:.3f} -> {rms['sobol']:.3f}",
            flush=True,
        )
