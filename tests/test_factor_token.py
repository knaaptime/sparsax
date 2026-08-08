"""Tests for factor / solve_factor / logdet_factor token primitives.

These three primitives hold a CHOLMOD numeric factor open and issue an
unbounded sequence of solves + a logdet against it without refactoring. The
key shapes exercised here:

  * factor + solve_factor + logdet_factor agree with solve / logdet / factor_solve
  * one factor serves many solves inside @jax.jit
  * a lax.fori_loop recurrence V_{j+1} = A^{-1}(G V_j) factors exactly once
  * vmap over b reuses one factor (one factorization for the whole batch)
  * vmap(factor) produces one factor per batch element (one factorization each)
  * stale tokens (after clear_cache) raise
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax


def grid_laplacian(k, seed=0):
    rng = np.random.default_rng(seed)
    L = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k))
    A = sp.kronsum(L, L) + sp.diags(rng.uniform(0.5, 1.5, k * k))
    A = sp.coo_matrix(A)
    return (
        A.row.astype(np.int32),
        A.col.astype(np.int32),
        A.data.astype(np.float64),
        A.toarray(),
    )


@pytest.fixture
def spd():
    return grid_laplacian(8)


class TestFactorSolveLogdet:
    def test_factor_solve_logdet_match_solve(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(0)
        b = rng.normal(size=n)
        tok = sparsax.factor(Ai, Aj, Ax, n)
        x = sparsax.solve_factor(tok, b)
        ld = sparsax.logdet_factor(tok)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_token_shape(self, spd):
        Ai, Aj, Ax, A = spd
        tok = sparsax.factor(Ai, Aj, Ax, A.shape[0])
        assert np.asarray(tok).shape == (2,)
        assert np.asarray(tok).dtype == np.int64

    def test_multi_rhs(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(1)
        B = rng.normal(size=(n, 5))
        tok = sparsax.factor(Ai, Aj, Ax, n)
        X = sparsax.solve_factor(tok, B)
        assert X.shape == B.shape
        np.testing.assert_allclose(X, np.linalg.solve(A, B), rtol=1e-10)

    def test_reuse_across_many_solves(self, spd):
        """One factor, several solves + logdet — only one factorization."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(2)
        bs = [rng.normal(size=n) for _ in range(5)]
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        tok = sparsax.factor(Ai, Aj, Ax, n)
        xs = [sparsax.solve_factor(tok, b) for b in bs]
        ld = sparsax.logdet_factor(tok)
        jax.block_until_ready(xs + [ld])
        assert sparsax.factorization_count() - c0 == 1
        for i, b in enumerate(bs):
            np.testing.assert_allclose(xs[i], np.linalg.solve(A, b), rtol=1e-10)

    def test_matches_factor_solve(self, spd):
        """factor + solve_factor + logdet_factor == factor_solve."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(3)
        b = rng.normal(size=n)
        rng.normal(size=n)
        # token path
        tok = sparsax.factor(Ai, Aj, Ax, n)
        mean = sparsax.solve_factor(tok, b)
        ld_tok = sparsax.logdet_factor(tok)
        # factor_solve path (mean + logdet)
        (fs_mean,), fs_ld = sparsax.factor_solve(
            Ai, Aj, Ax, [(b, sparsax.MODE_A)], want_logdet=True
        )
        np.testing.assert_allclose(mean, fs_mean, rtol=1e-12)
        np.testing.assert_allclose(float(ld_tok), float(fs_ld), rtol=1e-12)


class TestJIT:
    def test_jit_factor_and_solves(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(4)
        b = rng.normal(size=n)

        @jax.jit
        def f(Ax, b):
            tok = sparsax.factor(Ai, Aj, Ax, n)
            x1 = sparsax.solve_factor(tok, b)
            x2 = sparsax.solve_factor(tok, 2.0 * b)
            ld = sparsax.logdet_factor(tok)
            return x1, x2, ld

        x1, x2, ld = f(Ax, b)
        np.testing.assert_allclose(x1, np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(x2, np.linalg.solve(A, 2.0 * b), rtol=1e-10)
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_jit_changing_values_one_factorization_each(self, spd):
        """Same pattern, new Ax each call -> one factorization per call."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = np.ones(n)
        f = jax.jit(
            lambda Ax, b: sparsax.solve_factor(sparsax.factor(Ai, Aj, Ax, n), b)
        )
        for scale in (1.0, 2.5, 0.7):
            sparsax.clear_cache()
            c0 = sparsax.factorization_count()
            x = f(scale * Ax, b)
            np.testing.assert_allclose(x, np.linalg.solve(scale * A, b), rtol=1e-10)
            assert sparsax.factorization_count() - c0 == 1


class TestForiLoop:
    def test_krylov_recurrence_factors_once(self, spd):
        """V_{j+1} = A^{-1}(G V_j) via fori_loop of solve_factor — one factor."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(5)
        G = rng.normal(size=(n, n))
        v0 = rng.normal(size=n)

        # dense reference
        ref = v0.copy()
        for _ in range(5):
            ref = np.linalg.solve(A, G @ ref)

        @jax.jit
        def krylov(Ax, v0):
            tok = sparsax.factor(Ai, Aj, Ax, n)

            def body(j, v):
                return sparsax.solve_factor(tok, G @ v)

            return jax.lax.fori_loop(0, 5, body, v0)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        out = krylov(Ax, v0)
        jax.block_until_ready(out)
        assert sparsax.factorization_count() - c0 == 1
        np.testing.assert_allclose(out, ref, atol=1e-9)


class TestVmap:
    def test_vmap_over_b_one_factor(self, spd):
        """vmap over b against ONE factor — one factorization for the batch."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(6)
        bs = rng.normal(size=(6, n))

        @jax.jit
        def f(Ax, bs):
            tok = sparsax.factor(Ai, Aj, Ax, n)
            return jax.vmap(lambda b: sparsax.solve_factor(tok, b))(bs)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        xs = f(Ax, bs)
        jax.block_until_ready(xs)
        assert sparsax.factorization_count() - c0 == 1
        np.testing.assert_allclose(xs, np.linalg.solve(A, bs.T).T, rtol=1e-10)

    def test_vmap_factor_then_solve(self, spd):
        """vmap(factor) -> one factor per batch element; vmap(solve) consumes."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(7)
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax
        bs = rng.normal(size=(4, n))

        @jax.jit
        def f(Axs, bs):
            toks = jax.vmap(lambda ax: sparsax.factor(Ai, Aj, ax, n))(Axs)
            return jax.vmap(lambda tok, b: sparsax.solve_factor(tok, b))(toks, bs)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        xs = f(Axs, bs)
        jax.block_until_ready(xs)
        # 4 distinct Ax -> 4 factorizations
        assert sparsax.factorization_count() - c0 == 4
        for i, s in enumerate(scales):
            np.testing.assert_allclose(xs[i], np.linalg.solve(s * A, bs[i]), rtol=1e-10)

    def test_vmap_logdet(self, spd):
        """vmap(logdet_factor) over a batch of tokens."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax

        @jax.jit
        def f(Axs):
            toks = jax.vmap(lambda ax: sparsax.factor(Ai, Aj, ax, n))(Axs)
            return jax.vmap(lambda tok: sparsax.logdet_factor(tok))(toks)

        lds = f(Axs)
        for i, s in enumerate(scales):
            np.testing.assert_allclose(
                float(lds[i]), np.linalg.slogdet(s * A)[1], rtol=1e-10
            )


class TestSolveModes:
    def test_factor_chain_matches_solve(self, spd):
        """P' L^-T L^-1 P b (via solve_factor modes) must equal A^-1 b."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = np.arange(1.0, n + 1.0)
        tok = sparsax.factor(Ai, Aj, Ax, n)
        w = sparsax.solve_factor(tok, b, mode=sparsax.MODE_P)
        w = sparsax.solve_factor(tok, w, mode=sparsax.MODE_L)
        w = sparsax.solve_factor(tok, w, mode=sparsax.MODE_LT)
        x = sparsax.solve_factor(tok, w, mode=sparsax.MODE_PT)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10)


class TestErrors:
    def test_stale_token_after_clear_cache(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = np.ones(n)
        tok = sparsax.factor(Ai, Aj, Ax, n)
        sparsax.clear_cache()
        with pytest.raises(Exception, match="stale factor token"):
            sparsax.solve_factor(tok, b)

    def test_stale_token_logdet_after_clear_cache(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        tok = sparsax.factor(Ai, Aj, Ax, n)
        sparsax.clear_cache()
        with pytest.raises(Exception, match="stale factor token"):
            sparsax.logdet_factor(tok)

    def test_not_positive_definite(self):
        # Indefinite matrix: zero diagonal makes it singular.
        Ai = np.array([0, 0, 1], dtype=np.int32)
        Aj = np.array([0, 1, 1], dtype=np.int32)
        Ax = np.array([0.0, 1.0, 0.0])
        with pytest.raises(Exception):
            sparsax.factor(Ai, Aj, Ax, 2)

    def test_mismatched_lengths(self):
        Ai = np.array([0, 1], dtype=np.int32)
        Aj = np.array([0], dtype=np.int32)
        Ax = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError):
            sparsax.factor(Ai, Aj, Ax, 2)

    def test_bad_token_shape(self, spd):
        Ai, Aj, Ax, A = spd
        b = np.ones(A.shape[0])
        bad = jnp.arange(3, dtype=jnp.int64)
        with pytest.raises(ValueError):
            sparsax.solve_factor(bad, b)
        with pytest.raises(ValueError):
            sparsax.logdet_factor(bad)


class TestCacheReuse:
    def test_repeated_factor_same_Ax_no_refactor(self, spd):
        """factor() with identical Ax hits the value cache — no refactor."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        tok1 = sparsax.factor(Ai, Aj, Ax, n)
        tok2 = sparsax.factor(Ai, Aj, Ax, n)
        jax.block_until_ready((tok1, tok2))
        # Only the first call factorizes; the second is a cache hit.
        assert sparsax.factorization_count() - c0 == 1
        # Both tokens resolve to a valid solve.
        b = np.ones(n)
        np.testing.assert_allclose(
            sparsax.solve_factor(tok1, b), np.linalg.solve(A, b), rtol=1e-10
        )
        np.testing.assert_allclose(
            sparsax.solve_factor(tok2, b), np.linalg.solve(A, b), rtol=1e-10
        )
