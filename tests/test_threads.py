"""Solves issued from concurrent threads, for all three backends.

Chain-per-thread MCMC samplers call sparsax from several threads at once, often
with bit-identical values (chains started at one rho share a cached factor).
None of the backends takes a global lock, so these tests pin that concurrent
calls stay exact on a shared factor, on distinct factors, and through a shared
factor token.
"""

from concurrent.futures import ThreadPoolExecutor

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax

THREADS = 8
SOLVES = 40


def _rook_W(side):
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    W = (sp.kron(eye, path) + sp.kron(path, eye)).tocsr()
    return sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W


def _run_threads(work):
    with ThreadPoolExecutor(THREADS) as pool:
        return list(pool.map(work, range(THREADS)))


# --- KLU and UMFPACK: A = I - rho W (non-symmetric) -------------------------


def lu_system(side):
    """Pattern of I ∪ W, fixed across rho, with values and a dense reference."""
    W = _rook_W(side).tocoo()
    n = side * side
    Ai = np.concatenate([np.arange(n), W.row]).astype(np.int32)
    Aj = np.concatenate([np.arange(n), W.col]).astype(np.int32)
    Wd = W.toarray()

    def values(rho):
        return np.concatenate([np.ones(n), -rho * W.data])

    def dense(rho):
        return np.eye(n) - rho * Wd

    return Ai, Aj, values, dense, n


LU_BACKENDS = {
    "klu": (sparsax.lu_solve, sparsax.lu_factor, sparsax.lu_solve_factor),
    "umfpack": (sparsax.umf_solve, sparsax.umf_factor, sparsax.umf_solve_factor),
}


@pytest.mark.parametrize("backend", sorted(LU_BACKENDS))
def test_lu_shared_factor_concurrent_solves_are_exact(backend):
    """Every thread solves with one cached factor (identical values)."""
    lu_solve, _, _ = LU_BACKENDS[backend]
    Ai, Aj, values, dense, n = lu_system(24)
    Ax = jnp.asarray(values(0.4))
    A = dense(0.4)
    solve = jax.jit(lambda ax, b: lu_solve(Ai, Aj, ax, b))
    solve(Ax, jnp.zeros(n))  # compile once, before the threads start

    def work(t):
        rng = np.random.default_rng(t)
        worst = 0.0
        for _ in range(SOLVES):
            b = rng.normal(size=n)
            x = np.asarray(solve(Ax, jnp.asarray(b)))
            worst = max(worst, np.max(np.abs(A @ x - b)))
        return worst

    assert max(_run_threads(work)) < 1e-10


@pytest.mark.parametrize("backend", sorted(LU_BACKENDS))
def test_lu_distinct_factors_concurrent_solves_are_exact(backend):
    """Each thread solves with its own factor (its own rho)."""
    lu_solve, _, _ = LU_BACKENDS[backend]
    Ai, Aj, values, dense, n = lu_system(24)
    solve = jax.jit(lambda ax, b: lu_solve(Ai, Aj, ax, b))
    solve(jnp.asarray(values(0.1)), jnp.zeros(n))

    def work(t):
        rho = 0.1 + 0.08 * t
        Ax = jnp.asarray(values(rho))
        A = dense(rho)
        rng = np.random.default_rng(100 + t)
        worst = 0.0
        for _ in range(SOLVES):
            b = rng.normal(size=n)
            x = np.asarray(solve(Ax, jnp.asarray(b)))
            worst = max(worst, np.max(np.abs(A @ x - b)))
        return worst

    assert max(_run_threads(work)) < 1e-10


@pytest.mark.parametrize("backend", sorted(LU_BACKENDS))
def test_lu_shared_token_concurrent_solves_are_exact(backend):
    """Threads solve through one factor token, including transpose solves."""
    _, lu_factor, lu_solve_factor = LU_BACKENDS[backend]
    Ai, Aj, values, dense, n = lu_system(24)
    tok = lu_factor(Ai, Aj, values(0.6), n)
    A = dense(0.6)

    def work(t):
        rng = np.random.default_rng(200 + t)
        worst = 0.0
        for i in range(SOLVES):
            b = rng.normal(size=n)
            trans = bool(i % 2)
            x = np.asarray(lu_solve_factor(tok, b, trans=trans))
            M = A.T if trans else A
            worst = max(worst, np.max(np.abs(M @ x - b)))
        return worst

    assert max(_run_threads(work)) < 1e-10


# --- CHOLMOD: P = s I - 0.25 (W + W^T), symmetric positive definite ---------


def spd_system(side):
    W = _rook_W(side)
    S = (W + W.T).tocoo()
    n = side * side
    upper = S.row <= S.col
    Ai = np.concatenate([np.arange(n), S.row[upper]]).astype(np.int32)
    Aj = np.concatenate([np.arange(n), S.col[upper]]).astype(np.int32)
    Sd = S.toarray()
    s_data = S.data[upper]

    def values(s):
        return np.concatenate([np.full(n, s), -0.25 * s_data])

    def dense(s):
        return s * np.eye(n) - 0.25 * Sd

    return Ai, Aj, values, dense, n


def test_cholmod_shared_factor_concurrent_solves_are_exact():
    Ai, Aj, values, dense, n = spd_system(24)
    Ax = jnp.asarray(values(1.2))
    P = dense(1.2)
    solve = jax.jit(lambda ax, b: sparsax.solve(Ai, Aj, ax, b))
    solve(Ax, jnp.zeros(n))

    def work(t):
        rng = np.random.default_rng(300 + t)
        worst = 0.0
        for _ in range(SOLVES):
            b = rng.normal(size=n)
            x = np.asarray(solve(Ax, jnp.asarray(b)))
            worst = max(worst, np.max(np.abs(P @ x - b)))
        return worst

    assert max(_run_threads(work)) < 1e-10


def test_cholmod_distinct_factor_solve_and_logdet_are_exact():
    """Each thread runs factor_solve (mean, draw factor part, logdet) at its own s."""
    Ai, Aj, values, dense, n = spd_system(24)

    @jax.jit
    def step(ax, b, z):
        return sparsax.factor_solve(
            Ai,
            Aj,
            ax,
            [(b, sparsax.MODE_A), (z, (sparsax.MODE_LT, sparsax.MODE_PT))],
            want_logdet=True,
        )

    step(jnp.asarray(values(1.5)), jnp.zeros(n), jnp.zeros(n))

    def work(t):
        s = 1.1 + 0.2 * t
        Ax = jnp.asarray(values(s))
        P = dense(s)
        logdet = np.linalg.slogdet(P)[1]
        rng = np.random.default_rng(400 + t)
        worst = 0.0
        for _ in range(SOLVES // 4):
            b, z = rng.normal(size=n), rng.normal(size=n)
            (mean, w), ld = step(Ax, jnp.asarray(b), jnp.asarray(z))
            worst = max(worst, np.max(np.abs(P @ np.asarray(mean) - b)))
            # w = P' L^-T z has covariance P^{-1}: z'z == w' P w.
            w = np.asarray(w)
            worst = max(worst, abs(w @ P @ w - z @ z) / (z @ z))
            worst = max(worst, abs(float(ld) - logdet) / abs(logdet))
        return worst

    assert max(_run_threads(work)) < 1e-9


def test_cholmod_shared_token_concurrent_solves_are_exact():
    Ai, Aj, values, dense, n = spd_system(24)
    tok = sparsax.factor(Ai, Aj, values(1.3), n)
    P = dense(1.3)
    logdet = np.linalg.slogdet(P)[1]

    def work(t):
        rng = np.random.default_rng(500 + t)
        worst = 0.0
        for _ in range(SOLVES):
            b = rng.normal(size=n)
            x = np.asarray(sparsax.solve_factor(tok, b))
            worst = max(worst, np.max(np.abs(P @ x - b)))
        worst = max(worst, abs(float(sparsax.logdet_factor(tok)) - logdet))
        return worst

    assert max(_run_threads(work)) < 1e-9


def test_cholmod_concurrent_update_solve_and_selinv_are_exact():
    """Threads interleave update_solve and selinv over shared and distinct factors."""
    Ai, Aj, values, dense, n = spd_system(16)
    scales = (1.2, 1.6)
    Pinv = {s: np.linalg.inv(dense(s)) for s in scales}

    def work(t):
        s = scales[t % 2]
        P = dense(s)
        rng = np.random.default_rng(600 + t)
        worst = 0.0
        for _ in range(8):
            c, b = 0.1 * rng.normal(size=n), rng.normal(size=n)
            x = np.asarray(sparsax.update_solve(Ai, Aj, values(s), c, b))
            worst = max(worst, np.max(np.abs((P + np.outer(c, c)) @ x - b)))
            z = np.asarray(sparsax.selinv(Ai, Aj, values(s), n))
            worst = max(worst, np.max(np.abs(z - Pinv[s][Ai, Aj])))
        return worst

    assert max(_run_threads(work)) < 1e-9
