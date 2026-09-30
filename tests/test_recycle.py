"""Numeric factors evicted from a full cache are refactored in place.

Once a pattern's value cache is full, a factorization at new values reuses the
storage of the factor it evicts -- CHOLMOD refactors it, KLU runs klu_refactor
on its pivot sequence -- provided nothing else holds that factor.  These tests
shrink the caches so every call recycles, and pin the results to dense algebra.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax


def _rook_W(side):
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    W = (sp.kron(eye, path) + sp.kron(path, eye)).tocsr()
    return (sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W).tocoo()


@pytest.fixture
def tiny_caches():
    sparsax.set_lu_cache_size(2)
    sparsax.set_umf_cache_size(2)
    sparsax.set_num_cache_size(2)
    yield
    sparsax.set_lu_cache_size(32)
    sparsax.set_umf_cache_size(32)
    sparsax.set_num_cache_size(32)
    sparsax.clear_cache()


def _lu_system():
    W = _rook_W(8)
    n = W.shape[0]
    Ai = np.concatenate([np.arange(n), W.row]).astype(np.int32)
    Aj = np.concatenate([np.arange(n), W.col]).astype(np.int32)
    return Ai, Aj, lambda rho: np.concatenate([np.ones(n), -rho * W.data]), W, n


@pytest.mark.parametrize("backend", ["klu", "umfpack"])
def test_lu_refactors_stay_exact(tiny_caches, backend):
    solve = {"klu": sparsax.lu_solve, "umfpack": sparsax.umf_solve}[backend]
    Ai, Aj, values, W, n = _lu_system()
    b = np.random.default_rng(0).standard_normal((n, 3))
    f = jax.jit(lambda ax, b: solve(Ai, Aj, ax, b))
    for rho in np.linspace(-0.9, 0.9, 13):  # every call past the second recycles
        A = np.eye(n) - rho * W.toarray()
        x = np.asarray(f(jnp.asarray(values(rho)), jnp.asarray(b)))
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-12, atol=1e-12)
        # Transposed solve (the VJP) from the same recycled factor.
        _, vjp = jax.vjp(
            lambda bb: solve(Ai, Aj, jnp.asarray(values(rho)), bb), jnp.asarray(b)
        )
        g = np.asarray(vjp(jnp.asarray(b))[0])
        np.testing.assert_allclose(g, np.linalg.solve(A.T, b), rtol=1e-12, atol=1e-12)


def test_cholmod_refactors_stay_exact(tiny_caches):
    W = _rook_W(8)
    S = ((W + W.T) / 2).tocoo()
    n = S.shape[0]
    up = S.row <= S.col
    Ai = np.concatenate([np.arange(n), S.row[up]]).astype(np.int32)
    Aj = np.concatenate([np.arange(n), S.col[up]]).astype(np.int32)
    b = np.random.default_rng(1).standard_normal(n)
    for mode in ("simplicial", "supernodal"):
        sparsax.set_options(supernodal=mode)
        sparsax.set_num_cache_size(2)
        for rho in np.linspace(-0.9, 0.9, 11):
            ax = jnp.asarray(np.concatenate([np.ones(n), -rho * S.data[up]]))
            A = np.eye(n) - rho * S.toarray()
            x = np.asarray(sparsax.solve(Ai, Aj, ax, jnp.asarray(b)))
            np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-12, atol=1e-12)
            ld = float(sparsax.logdet(Ai, Aj, ax, n))
            np.testing.assert_allclose(ld, np.linalg.slogdet(A)[1], rtol=1e-12)
    sparsax.set_options(supernodal="auto")


def test_held_klu_factor_is_not_recycled(tiny_caches):
    """A token keeps its factor even after the cache has cycled past it."""
    Ai, Aj, values, W, n = _lu_system()
    b = jnp.asarray(np.random.default_rng(2).standard_normal(n))
    tok = sparsax.lu_factor(Ai, Aj, jnp.asarray(values(0.4)), n)
    for rho in np.linspace(-0.8, 0.8, 9):  # evict and recycle repeatedly
        sparsax.lu_solve(Ai, Aj, jnp.asarray(values(rho)), b).block_until_ready()
    x = np.asarray(sparsax.lu_solve_factor(tok, b))
    A = np.eye(n) - 0.4 * W.toarray()
    np.testing.assert_allclose(x, np.linalg.solve(A, np.asarray(b)), rtol=1e-12)


def test_klu_unstable_refactor_falls_back_to_pivoting(tiny_caches):
    """Reusing diagonal pivots on a zero-diagonal matrix fails; KLU re-pivots."""
    Ai = np.array([0, 0, 1, 1], dtype=np.int32)
    Aj = np.array([0, 1, 0, 1], dtype=np.int32)
    b = jnp.asarray([1.0, 2.0])
    diag_dominant = jnp.asarray([2.0, 1.0, 1.0, 2.0])
    sparsax.lu_solve(Ai, Aj, diag_dominant, b).block_until_ready()
    sparsax.lu_solve(Ai, Aj, diag_dominant * 1.5, b).block_until_ready()  # fills cache
    swap = jnp.asarray([0.0, 1.0, 1.0, 0.0])  # nonsingular, zero diagonal
    x = np.asarray(sparsax.lu_solve(Ai, Aj, swap, b))
    np.testing.assert_allclose(x, [2.0, 1.0], atol=1e-14)


@pytest.mark.parametrize("backend", ["cholmod", "klu", "umfpack"])
def test_token_tables_are_bounded(backend):
    """Only the newest tokens pin factors; an older one fails as stale."""
    if backend == "cholmod":
        W = _rook_W(6)
        S = ((W + W.T) / 2).tocoo()
        n = S.shape[0]
        up = S.row <= S.col
        Ai = np.concatenate([np.arange(n), S.row[up]]).astype(np.int32)
        Aj = np.concatenate([np.arange(n), S.col[up]]).astype(np.int32)
        values = lambda rho: np.concatenate([np.ones(n), -rho * S.data[up]])  # noqa: E731
        factor, solve_factor = sparsax.factor, sparsax.solve_factor
    else:
        Ai, Aj, values, _, n = _lu_system()
        factor, solve_factor = {
            "klu": (sparsax.lu_factor, sparsax.lu_solve_factor),
            "umfpack": (sparsax.umf_factor, sparsax.umf_solve_factor),
        }[backend]
    b = jnp.ones(n)
    sparsax.set_token_cache_size(3)
    try:
        tokens = [
            factor(Ai, Aj, jnp.asarray(values(r)), n) for r in (0.1, 0.2, 0.3, 0.4, 0.5)
        ]
        for tok in tokens[-3:]:  # the newest three still solve
            assert np.all(np.isfinite(np.asarray(solve_factor(tok, b))))
        with pytest.raises(Exception, match="stale factor token"):
            np.asarray(solve_factor(tokens[0], b))
    finally:
        sparsax.set_token_cache_size(64)
        sparsax.clear_cache()
