"""CHOLMOD with the factorization forced simplicial and forced supernodal.

CHOLMOD's automatic choice leaves small test matrices simplicial, so the
supernodal path needs its own tests: each numeric factorization copies the
symbolic factor, the log-determinant is read off supernodal columns, and the
selected inverse and rank updates convert the factor to simplicial LDL'.
"""

from concurrent.futures import ThreadPoolExecutor

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax


@pytest.fixture(params=["simplicial", "supernodal"])
def mode(request):
    sparsax.set_options(supernodal=request.param)
    yield request.param
    sparsax.set_options(supernodal="auto")


def spd_system(side=24):
    """P = s I - 0.5 S for grid adjacency S: positive definite for s > 2."""
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    S = (sp.kron(eye, path) + sp.kron(path, eye)).tocoo()
    n = side * side
    upper = S.row <= S.col
    Ai = np.concatenate([np.arange(n), S.row[upper]]).astype(np.int32)
    Aj = np.concatenate([np.arange(n), S.col[upper]]).astype(np.int32)
    s_upper = S.data[upper]
    Sd = S.toarray()

    def values(s):
        return np.concatenate([np.full(n, s), -0.5 * s_upper])

    def dense(s):
        return s * np.eye(n) - 0.5 * Sd

    return Ai, Aj, values, dense, n


def test_factor_solve_mean_draw_and_logdet(mode):
    Ai, Aj, values, dense, n = spd_system()
    P = dense(2.5)
    rng = np.random.default_rng(1)
    b, z = rng.normal(size=n), rng.normal(size=n)
    sparsax.clear_cache()
    c0 = sparsax.factorization_count()
    (mean, w), ld = sparsax.factor_solve(
        Ai,
        Aj,
        values(2.5),
        [(b, sparsax.MODE_A), (z, (sparsax.MODE_LT, sparsax.MODE_PT))],
        want_logdet=True,
    )
    jax.block_until_ready((mean, w, ld))
    assert sparsax.factorization_count() - c0 == 1
    np.testing.assert_allclose(mean, np.linalg.solve(P, b), rtol=1e-9)
    # w = P' L^-T z has covariance P^{-1}, so w' P w == z' z exactly.
    w = np.asarray(w)
    np.testing.assert_allclose(w @ P @ w, z @ z, rtol=1e-9)
    np.testing.assert_allclose(float(ld), np.linalg.slogdet(P)[1], rtol=1e-11)


def test_token_solves_and_logdet(mode):
    Ai, Aj, values, dense, n = spd_system()
    P = dense(3.0)
    b = np.linspace(-1.0, 1.0, n)
    tok = sparsax.factor(Ai, Aj, values(3.0), n)
    x = sparsax.solve_factor(tok, b, mode=sparsax.MODE_P)
    x = sparsax.solve_factor(tok, x, mode=sparsax.MODE_L)
    x = sparsax.solve_factor(tok, x, mode=sparsax.MODE_LT)
    x = sparsax.solve_factor(tok, x, mode=sparsax.MODE_PT)
    np.testing.assert_allclose(x, np.linalg.solve(P, b), rtol=1e-9)
    np.testing.assert_allclose(
        float(sparsax.logdet_factor(tok)), np.linalg.slogdet(P)[1], rtol=1e-11
    )


def test_selinv_and_update_solve(mode):
    Ai, Aj, values, dense, n = spd_system(16)
    P = dense(2.5)
    z = np.asarray(sparsax.selinv(Ai, Aj, values(2.5), n))
    np.testing.assert_allclose(z, np.linalg.inv(P)[Ai, Aj], atol=1e-11)

    rng = np.random.default_rng(2)
    C = 0.2 * rng.normal(size=(n, 2))
    b = rng.normal(size=n)
    x, ld = sparsax.update_solve(Ai, Aj, values(2.5), C, b, return_logdet=True)
    Pu = P + C @ C.T
    np.testing.assert_allclose(x, np.linalg.solve(Pu, b), rtol=1e-9)
    np.testing.assert_allclose(float(ld), np.linalg.slogdet(Pu)[1], rtol=1e-11)


def test_concurrent_shared_and_distinct_factors(mode):
    """Eight threads over four values of s: each factor is shared by two threads."""
    Ai, Aj, values, dense, n = spd_system()
    scales = (2.5, 2.7, 2.9, 3.1)
    refs = {s: (dense(s), np.linalg.slogdet(dense(s))[1]) for s in scales}
    solve = jax.jit(lambda ax, b: sparsax.solve(Ai, Aj, ax, b))
    solve(jnp.asarray(values(scales[0])), jnp.zeros(n))

    def work(t):
        s = scales[t % 4]
        P, logdet = refs[s]
        Ax = jnp.asarray(values(s))
        rng = np.random.default_rng(10 + t)
        worst = 0.0
        for _ in range(20):
            b = rng.normal(size=n)
            x = np.asarray(solve(Ax, jnp.asarray(b)))
            worst = max(worst, np.max(np.abs(P @ x - b)))
        ld = float(sparsax.logdet(Ai, Aj, Ax, n))
        return max(worst, abs(ld - logdet) / abs(logdet))

    with ThreadPoolExecutor(8) as pool:
        assert max(pool.map(work, range(8))) < 1e-9


def test_indefinite_matrix_raises(mode):
    Ai, Aj, values, dense, n = spd_system(16)
    with pytest.raises(Exception, match="positive definite"):
        sparsax.logdet(Ai, Aj, values(1.0), n).block_until_ready()
