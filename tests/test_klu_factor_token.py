"""Tests for KLU factor-token + logdet primitives.

lu_factor / lu_solve_factor / lu_logdet_factor are the non-symmetric analogue
of the CHOLMOD factor/solve_factor/logdet_factor token primitives: hold a KLU
numeric factor open across an unbounded sequence of solves + a logdet without
refactoring. lu_logdet computes log|det(A)| from the KLU factor's U diagonal
(O(n) C-side), needed for non-symmetric spatial models where log|I - rho W|
appears in the log-likelihood Jacobian and W is row-standardised.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax


def nonsym_matrix(n, rho=0.5, seed=0):
    """A = I - rho W, W row-standardised random — non-symmetric, general."""
    rng = np.random.default_rng(seed)
    W = rng.random((n, n))
    W /= W.sum(1, keepdims=True)
    A = np.eye(n) - rho * W
    As = sp.coo_matrix(A)
    return (
        As.row.astype(np.int32),
        As.col.astype(np.int32),
        As.data.astype(np.float64),
        A,
    )


@pytest.fixture
def nonsym():
    return nonsym_matrix(16)


class TestLuLogdet:
    def test_matches_numpy(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        ld = sparsax.lu_logdet(Ai, Aj, Ax, A.shape[0])
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_sign_independent(self, nonsym):
        """lu_logdet returns log|det|, so sign(A) must not matter."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        ld1 = float(sparsax.lu_logdet(Ai, Aj, Ax, n))
        ld2 = float(sparsax.lu_logdet(Ai, Aj, -Ax, n))
        np.testing.assert_allclose(ld1, ld2, rtol=1e-12)

    def test_under_jit(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]

        @jax.jit
        def f(Ax):
            return sparsax.lu_logdet(Ai, Aj, Ax, n)

        np.testing.assert_allclose(float(f(Ax)), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_shares_factor_with_lu_solve(self, nonsym):
        """lu_solve + lu_logdet with identical Ax factorize once."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        b = np.ones(n)
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        x = sparsax.lu_solve(Ai, Aj, Ax, b)
        ld = sparsax.lu_logdet(Ai, Aj, Ax, n)
        jax.block_until_ready((x, ld))
        assert sparsax.factorization_count() - c0 == 1

    def test_singular_raises(self):
        Ai = np.array([0, 1], dtype=np.int32)
        Aj = np.array([0, 1], dtype=np.int32)
        Ax = np.array([0.0, 0.0])
        with pytest.raises(Exception):
            sparsax.lu_logdet(Ai, Aj, Ax, 2)


class TestLuFactorSolveLogdet:
    def test_factor_solve_match_lu_solve(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(0)
        b = rng.normal(size=n)
        tok = sparsax.lu_factor(Ai, Aj, Ax, n)
        x = sparsax.lu_solve_factor(tok, b)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10)

    def test_factor_logdet_match(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        tok = sparsax.lu_factor(Ai, Aj, Ax, n)
        ld = sparsax.lu_logdet_factor(tok)
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_token_shape(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        tok = sparsax.lu_factor(Ai, Aj, Ax, A.shape[0])
        assert np.asarray(tok).shape == (2,)
        assert np.asarray(tok).dtype == np.int64

    def test_trans_solve(self, nonsym):
        """lu_solve_factor(trans=True) solves A^T x = b."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(1)
        b = rng.normal(size=n)
        tok = sparsax.lu_factor(Ai, Aj, Ax, n)
        x = sparsax.lu_solve_factor(tok, b, trans=True)
        np.testing.assert_allclose(x, np.linalg.solve(A.T, b), rtol=1e-10)

    def test_multi_rhs(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(2)
        B = rng.normal(size=(n, 5))
        tok = sparsax.lu_factor(Ai, Aj, Ax, n)
        X = sparsax.lu_solve_factor(tok, B)
        assert X.shape == B.shape
        np.testing.assert_allclose(X, np.linalg.solve(A, B), rtol=1e-10)

    def test_reuse_many_solves_one_factorization(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(3)
        bs = [rng.normal(size=n) for _ in range(5)]
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        tok = sparsax.lu_factor(Ai, Aj, Ax, n)
        xs = [sparsax.lu_solve_factor(tok, b) for b in bs]
        ld = sparsax.lu_logdet_factor(tok)
        jax.block_until_ready(xs + [ld])
        assert sparsax.factorization_count() - c0 == 1
        for i, b in enumerate(bs):
            np.testing.assert_allclose(xs[i], np.linalg.solve(A, b), rtol=1e-10)


class TestJIT:
    def test_jit_factor_and_solves(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(4)
        b = rng.normal(size=n)

        @jax.jit
        def f(Ax, b):
            tok = sparsax.lu_factor(Ai, Aj, Ax, n)
            x1 = sparsax.lu_solve_factor(tok, b)
            x2 = sparsax.lu_solve_factor(tok, 2.0 * b)
            ld = sparsax.lu_logdet_factor(tok)
            return x1, x2, ld

        x1, x2, ld = f(Ax, b)
        np.testing.assert_allclose(x1, np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(x2, np.linalg.solve(A, 2.0 * b), rtol=1e-10)
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_jit_changing_values(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        b = np.ones(n)
        f = jax.jit(
            lambda Ax, b: sparsax.lu_solve_factor(sparsax.lu_factor(Ai, Aj, Ax, n), b)
        )
        for scale in (1.0, 2.5, 0.7):
            np.testing.assert_allclose(
                f(scale * Ax, b), np.linalg.solve(scale * A, b), rtol=1e-10
            )


class TestForiLoop:
    def test_krylov_recurrence_factors_once(self, nonsym):
        """V_{j+1} = A^{-1}(G V_j) via fori_loop of lu_solve_factor — one factor."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(5)
        G = rng.normal(size=(n, n))
        v0 = rng.normal(size=n)

        ref = v0.copy()
        for _ in range(5):
            ref = np.linalg.solve(A, G @ ref)

        @jax.jit
        def krylov(Ax, v0):
            tok = sparsax.lu_factor(Ai, Aj, Ax, n)

            def body(j, v):
                return sparsax.lu_solve_factor(tok, G @ v)

            return jax.lax.fori_loop(0, 5, body, v0)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        out = krylov(Ax, v0)
        jax.block_until_ready(out)
        assert sparsax.factorization_count() - c0 == 1
        np.testing.assert_allclose(out, ref, atol=1e-9)


class TestVmap:
    def test_vmap_over_b_one_factor(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(6)
        bs = rng.normal(size=(6, n))

        @jax.jit
        def f(Ax, bs):
            tok = sparsax.lu_factor(Ai, Aj, Ax, n)
            return jax.vmap(lambda b: sparsax.lu_solve_factor(tok, b))(bs)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        xs = f(Ax, bs)
        jax.block_until_ready(xs)
        assert sparsax.factorization_count() - c0 == 1
        np.testing.assert_allclose(xs, np.linalg.solve(A, bs.T).T, rtol=1e-10)

    def test_vmap_factor_then_solve(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(7)
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax
        bs = rng.normal(size=(4, n))

        @jax.jit
        def f(Axs, bs):
            toks = jax.vmap(lambda ax: sparsax.lu_factor(Ai, Aj, ax, n))(Axs)
            return jax.vmap(lambda tok, b: sparsax.lu_solve_factor(tok, b))(toks, bs)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        xs = f(Axs, bs)
        jax.block_until_ready(xs)
        assert sparsax.factorization_count() - c0 == 4
        for i, s in enumerate(scales):
            np.testing.assert_allclose(xs[i], np.linalg.solve(s * A, bs[i]), rtol=1e-10)

    def test_vmap_logdet(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax

        @jax.jit
        def f(Axs):
            toks = jax.vmap(lambda ax: sparsax.lu_factor(Ai, Aj, ax, n))(Axs)
            return jax.vmap(lambda tok: sparsax.lu_logdet_factor(tok))(toks)

        lds = f(Axs)
        for i, s in enumerate(scales):
            np.testing.assert_allclose(
                float(lds[i]), np.linalg.slogdet(s * A)[1], rtol=1e-10
            )


class TestErrors:
    def test_stale_token_solve(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        b = np.ones(n)
        tok = sparsax.lu_factor(Ai, Aj, Ax, n)
        sparsax.clear_cache()
        with pytest.raises(Exception, match="stale factor token"):
            sparsax.lu_solve_factor(tok, b)

    def test_stale_token_logdet(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        tok = sparsax.lu_factor(Ai, Aj, Ax, n)
        sparsax.clear_cache()
        with pytest.raises(Exception, match="stale factor token"):
            sparsax.lu_logdet_factor(tok)

    def test_mismatched_lengths(self):
        Ai = np.array([0, 1], dtype=np.int32)
        Aj = np.array([0], dtype=np.int32)
        Ax = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError):
            sparsax.lu_logdet(Ai, Aj, Ax, 2)
        with pytest.raises(ValueError):
            sparsax.lu_factor(Ai, Aj, Ax, 2)

    def test_bad_token_shape(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        b = np.ones(A.shape[0])
        bad = jnp.arange(3, dtype=jnp.int64)
        with pytest.raises(ValueError):
            sparsax.lu_solve_factor(bad, b)
        with pytest.raises(ValueError):
            sparsax.lu_logdet_factor(bad)


class TestCacheReuse:
    def test_repeated_factor_same_Ax_no_refactor(self, nonsym):
        """lu_factor() with identical Ax reuses the LRU cache — no refactor."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        tok1 = sparsax.lu_factor(Ai, Aj, Ax, n)
        tok2 = sparsax.lu_factor(Ai, Aj, Ax, n)
        jax.block_until_ready((tok1, tok2))
        assert sparsax.factorization_count() - c0 == 1
        b = np.ones(n)
        np.testing.assert_allclose(
            sparsax.lu_solve_factor(tok1, b), np.linalg.solve(A, b), rtol=1e-10
        )
        np.testing.assert_allclose(
            sparsax.lu_solve_factor(tok2, b), np.linalg.solve(A, b), rtol=1e-10
        )
