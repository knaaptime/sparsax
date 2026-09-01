"""Tests for the UMFPACK backend: umf_solve / umf_logdet / factor tokens.

UMFPACK is the second sparse-LU backend, interchangeable with KLU's ``lu_*``
functions and differing only in which graph it is fast on (KLU reuses its pivot
ordering and wins while the graph is sparse; UMFPACK re-pivots but is
multifrontal with BLAS3 and wins once the graph densifies). These tests check
that the two agree numerically wherever they overlap, and that the UMFPACK path
carries the same caching, JIT, vmap, and token semantics.

umf_logdet reads log|det(A)| from umfpack_*_get_determinant, which returns the
determinant in mantissa/exponent form — that is the SAR Jacobian log|I - rho W|
for a directed W, and the one thing KLU exposes no equivalent for.
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


class TestUmfSolve:
    def test_matches_numpy(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        b = np.arange(1.0, A.shape[0] + 1.0)
        x = sparsax.umf_solve(Ai, Aj, Ax, b)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10)

    def test_matches_klu(self, nonsym):
        """The two LU backends are interchangeable — same answer, either way."""
        Ai, Aj, Ax, A = nonsym
        b = np.arange(1.0, A.shape[0] + 1.0)
        np.testing.assert_allclose(
            sparsax.umf_solve(Ai, Aj, Ax, b),
            sparsax.lu_solve(Ai, Aj, Ax, b),
            rtol=1e-10,
        )

    def test_multi_rhs(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        rng = np.random.default_rng(1)
        B = rng.normal(size=(A.shape[0], 4))
        X = sparsax.umf_solve(Ai, Aj, Ax, B)
        assert X.shape == B.shape
        np.testing.assert_allclose(X, np.linalg.solve(A, B), rtol=1e-10)

    def test_duplicates_summed(self):
        """Two COO entries at the same (i, j) add, as in the KLU path."""
        Ai = np.array([0, 0, 1], dtype=np.int32)
        Aj = np.array([0, 0, 1], dtype=np.int32)
        Ax = np.array([1.5, 0.5, 4.0])
        b = np.array([2.0, 8.0])
        x = sparsax.umf_solve(Ai, Aj, Ax, b)
        np.testing.assert_allclose(x, np.array([1.0, 2.0]), rtol=1e-12)

    def test_under_jit(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        b = np.ones(A.shape[0])

        @jax.jit
        def f(Ax, b):
            return sparsax.umf_solve(Ai, Aj, Ax, b)

        np.testing.assert_allclose(f(Ax, b), np.linalg.solve(A, b), rtol=1e-10)

    def test_lax_scan_reuses_analysis(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        scales = np.linspace(1.0, 2.0, 5)

        @jax.jit
        def run(Ax, scales):
            def body(carry, s):
                x = sparsax.umf_solve(Ai, Aj, s * Ax, np.ones(n))
                return carry, x

            _, xs = jax.lax.scan(body, 0.0, scales)
            return xs

        xs = run(Ax, scales)
        for i, s in enumerate(scales):
            np.testing.assert_allclose(
                xs[i], np.linalg.solve(s * A, np.ones(n)), rtol=1e-10
            )

    def test_singular_raises(self):
        Ai = np.array([0, 1], dtype=np.int32)
        Aj = np.array([0, 1], dtype=np.int32)
        Ax = np.array([1.0, 0.0])
        with pytest.raises(Exception):
            sparsax.umf_solve(Ai, Aj, Ax, np.ones(2))

    def test_index_out_of_range(self):
        Ai = np.array([0, 5], dtype=np.int32)
        Aj = np.array([0, 1], dtype=np.int32)
        Ax = np.array([1.0, 1.0])
        with pytest.raises(Exception):
            sparsax.umf_solve(Ai, Aj, Ax, np.ones(2))

    def test_mismatched_lengths(self):
        Ai = np.array([0, 1], dtype=np.int32)
        Aj = np.array([0], dtype=np.int32)
        Ax = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError):
            sparsax.umf_solve(Ai, Aj, Ax, np.ones(2))
        with pytest.raises(ValueError):
            sparsax.umf_logdet(Ai, Aj, Ax, 2)
        with pytest.raises(ValueError):
            sparsax.umf_factor(Ai, Aj, Ax, 2)


class TestUmfLogdet:
    def test_matches_numpy(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        ld = sparsax.umf_logdet(Ai, Aj, Ax, A.shape[0])
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_matches_klu(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        np.testing.assert_allclose(
            float(sparsax.umf_logdet(Ai, Aj, Ax, n)),
            float(sparsax.lu_logdet(Ai, Aj, Ax, n)),
            rtol=1e-10,
        )

    def test_sign_independent(self, nonsym):
        """umf_logdet returns log|det|, so sign(A) must not matter."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        np.testing.assert_allclose(
            float(sparsax.umf_logdet(Ai, Aj, Ax, n)),
            float(sparsax.umf_logdet(Ai, Aj, -Ax, n)),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("n", [300, 400, 4000])
    def test_large_determinant_does_not_overflow(self, n):
        """Mantissa/exponent form: det(A) here is 1e300 to 1e4000.

        Past n = 308 the determinant no longer fits in a double at all, and
        umfpack_*_get_determinant reports UMFPACK_WARNING_determinant_overflow
        alongside a perfectly good Mx * 10^Ex. Working in log space that is a
        success, not a failure, so umf_logdet must not reject it.
        """
        Ai = np.arange(n, dtype=np.int32)
        Aj = np.arange(n, dtype=np.int32)
        Ax = np.full(n, 10.0)
        ld = float(sparsax.umf_logdet(Ai, Aj, Ax, n))
        np.testing.assert_allclose(ld, n * np.log(10.0), rtol=1e-12)

    def test_tiny_determinant_does_not_underflow(self):
        """The mirror case: det(A) = 1e-4000, below any representable double."""
        n = 4000
        Ai = np.arange(n, dtype=np.int32)
        Aj = np.arange(n, dtype=np.int32)
        Ax = np.full(n, 0.1)
        ld = float(sparsax.umf_logdet(Ai, Aj, Ax, n))
        np.testing.assert_allclose(ld, -n * np.log(10.0), rtol=1e-12)

    def test_under_jit(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]

        @jax.jit
        def f(Ax):
            return sparsax.umf_logdet(Ai, Aj, Ax, n)

        np.testing.assert_allclose(float(f(Ax)), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_shares_factor_with_umf_solve(self, nonsym):
        """umf_solve + umf_logdet with identical Ax factorize once."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        x = sparsax.umf_solve(Ai, Aj, Ax, np.ones(n))
        ld = sparsax.umf_logdet(Ai, Aj, Ax, n)
        jax.block_until_ready((x, ld))
        assert sparsax.factorization_count() - c0 == 1

    def test_independent_of_klu_cache(self, nonsym):
        """The two backends keep separate caches — each factorizes once."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        ld_klu = sparsax.lu_logdet(Ai, Aj, Ax, n)
        ld_umf = sparsax.umf_logdet(Ai, Aj, Ax, n)
        jax.block_until_ready((ld_klu, ld_umf))
        assert sparsax.factorization_count() - c0 == 2
        np.testing.assert_allclose(float(ld_klu), float(ld_umf), rtol=1e-10)

    def test_singular_raises(self):
        Ai = np.array([0, 1], dtype=np.int32)
        Aj = np.array([0, 1], dtype=np.int32)
        Ax = np.array([0.0, 0.0])
        with pytest.raises(Exception):
            sparsax.umf_logdet(Ai, Aj, Ax, 2)


class TestAD:
    def test_grad_matches_finite_differences(self, nonsym):
        """umf_solve's VJP: dAx = -v x^T with A^T v = g, same rule as lu_solve."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(2)
        b = rng.normal(size=n)
        w = rng.normal(size=n)

        def f(Ax):
            return jnp.dot(w, sparsax.umf_solve(Ai, Aj, Ax, b))

        g = np.asarray(jax.grad(f)(Ax))
        eps = 1e-6
        for k in (0, 3, 11, len(Ax) - 1):
            Ap = Ax.copy()
            Am = Ax.copy()
            Ap[k] += eps
            Am[k] -= eps
            fd = (float(f(Ap)) - float(f(Am))) / (2 * eps)
            np.testing.assert_allclose(g[k], fd, rtol=1e-5, atol=1e-7)

    def test_grad_wrt_b(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(3)
        b = rng.normal(size=n)
        w = rng.normal(size=n)

        g = jax.grad(lambda b: jnp.dot(w, sparsax.umf_solve(Ai, Aj, Ax, b)))(b)
        np.testing.assert_allclose(g, np.linalg.solve(A.T, w), rtol=1e-9)

    def test_grad_matches_klu(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(4)
        b = rng.normal(size=n)
        w = rng.normal(size=n)

        gu = jax.grad(lambda ax: jnp.dot(w, sparsax.umf_solve(Ai, Aj, ax, b)))(Ax)
        gk = jax.grad(lambda ax: jnp.dot(w, sparsax.lu_solve(Ai, Aj, ax, b)))(Ax)
        np.testing.assert_allclose(gu, gk, rtol=1e-9)

    def test_grad_under_jit(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        b = np.ones(n)
        w = np.ones(n)
        f = jax.jit(jax.grad(lambda ax: jnp.dot(w, sparsax.umf_solve(Ai, Aj, ax, b))))
        assert np.all(np.isfinite(np.asarray(f(Ax))))


class TestFactorTokens:
    def test_factor_solve_matches_umf_solve(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(5)
        b = rng.normal(size=n)
        tok = sparsax.umf_factor(Ai, Aj, Ax, n)
        np.testing.assert_allclose(
            sparsax.umf_solve_factor(tok, b), np.linalg.solve(A, b), rtol=1e-10
        )

    def test_factor_logdet_matches(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        tok = sparsax.umf_factor(Ai, Aj, Ax, A.shape[0])
        np.testing.assert_allclose(
            float(sparsax.umf_logdet_factor(tok)),
            np.linalg.slogdet(A)[1],
            rtol=1e-10,
        )

    def test_token_shape(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        tok = sparsax.umf_factor(Ai, Aj, Ax, A.shape[0])
        assert np.asarray(tok).shape == (2,)
        assert np.asarray(tok).dtype == np.int64

    def test_trans_solve(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(6)
        b = rng.normal(size=n)
        tok = sparsax.umf_factor(Ai, Aj, Ax, n)
        np.testing.assert_allclose(
            sparsax.umf_solve_factor(tok, b, trans=True),
            np.linalg.solve(A.T, b),
            rtol=1e-10,
        )

    def test_multi_rhs(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(7)
        B = rng.normal(size=(n, 5))
        tok = sparsax.umf_factor(Ai, Aj, Ax, n)
        X = sparsax.umf_solve_factor(tok, B)
        assert X.shape == B.shape
        np.testing.assert_allclose(X, np.linalg.solve(A, B), rtol=1e-10)

    def test_reuse_many_solves_one_factorization(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(8)
        bs = [rng.normal(size=n) for _ in range(5)]
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        tok = sparsax.umf_factor(Ai, Aj, Ax, n)
        xs = [sparsax.umf_solve_factor(tok, b) for b in bs]
        ld = sparsax.umf_logdet_factor(tok)
        jax.block_until_ready(xs + [ld])
        assert sparsax.factorization_count() - c0 == 1
        for i, b in enumerate(bs):
            np.testing.assert_allclose(xs[i], np.linalg.solve(A, b), rtol=1e-10)

    def test_repeated_factor_same_Ax_no_refactor(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        tok1 = sparsax.umf_factor(Ai, Aj, Ax, n)
        tok2 = sparsax.umf_factor(Ai, Aj, Ax, n)
        jax.block_until_ready((tok1, tok2))
        assert sparsax.factorization_count() - c0 == 1
        b = np.ones(n)
        np.testing.assert_allclose(
            sparsax.umf_solve_factor(tok2, b), np.linalg.solve(A, b), rtol=1e-10
        )

    def test_jit_factor_and_solves(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(9)
        b = rng.normal(size=n)

        @jax.jit
        def f(Ax, b):
            tok = sparsax.umf_factor(Ai, Aj, Ax, n)
            return (
                sparsax.umf_solve_factor(tok, b),
                sparsax.umf_solve_factor(tok, 2.0 * b),
                sparsax.umf_logdet_factor(tok),
            )

        x1, x2, ld = f(Ax, b)
        np.testing.assert_allclose(x1, np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(x2, np.linalg.solve(A, 2.0 * b), rtol=1e-10)
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_stale_token_solve(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        tok = sparsax.umf_factor(Ai, Aj, Ax, n)
        sparsax.clear_cache()
        with pytest.raises(Exception, match="stale factor token"):
            sparsax.umf_solve_factor(tok, np.ones(n))

    def test_stale_token_logdet(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        tok = sparsax.umf_factor(Ai, Aj, Ax, A.shape[0])
        sparsax.clear_cache()
        with pytest.raises(Exception, match="stale factor token"):
            sparsax.umf_logdet_factor(tok)

    def test_bad_token_shape(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        b = np.ones(A.shape[0])
        bad = jnp.arange(3, dtype=jnp.int64)
        with pytest.raises(ValueError):
            sparsax.umf_solve_factor(bad, b)
        with pytest.raises(ValueError):
            sparsax.umf_logdet_factor(bad)


class TestForiLoop:
    def test_krylov_recurrence_factors_once(self, nonsym):
        """V_{j+1} = A^{-1}(G V_j) in a fori_loop — one factorization."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(10)
        G = rng.normal(size=(n, n))
        v0 = rng.normal(size=n)

        ref = v0.copy()
        for _ in range(5):
            ref = np.linalg.solve(A, G @ ref)

        @jax.jit
        def krylov(Ax, v0):
            tok = sparsax.umf_factor(Ai, Aj, Ax, n)

            def body(j, v):
                return sparsax.umf_solve_factor(tok, G @ v)

            return jax.lax.fori_loop(0, 5, body, v0)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        out = krylov(Ax, v0)
        jax.block_until_ready(out)
        assert sparsax.factorization_count() - c0 == 1
        np.testing.assert_allclose(out, ref, atol=1e-9)


class TestVmap:
    def test_vmap_over_b(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        rng = np.random.default_rng(11)
        bs = rng.normal(size=(6, A.shape[0]))
        xs = jax.vmap(lambda b: sparsax.umf_solve(Ai, Aj, Ax, b))(bs)
        np.testing.assert_allclose(xs, np.linalg.solve(A, bs.T).T, rtol=1e-10)

    def test_vmap_over_values(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax
        b = np.ones(n)
        xs = jax.vmap(lambda ax: sparsax.umf_solve(Ai, Aj, ax, b))(Axs)
        for i, s in enumerate(scales):
            np.testing.assert_allclose(xs[i], np.linalg.solve(s * A, b), rtol=1e-10)

    def test_vmap_lowers_to_single_batched_call(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        bs = np.ones((4, A.shape[0]))
        f = jax.jit(jax.vmap(lambda b: sparsax.umf_solve(Ai, Aj, Ax, b)))
        text = f.lower(bs).compile().as_text()
        assert "sparsax_umf_solve_batched_f64" in text

    def test_vmap_logdet(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax
        lds = jax.vmap(lambda ax: sparsax.umf_logdet(Ai, Aj, ax, n))(Axs)
        for i, s in enumerate(scales):
            np.testing.assert_allclose(
                float(lds[i]), np.linalg.slogdet(s * A)[1], rtol=1e-10
            )

    def test_vmap_over_b_one_factor(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(12)
        bs = rng.normal(size=(6, n))

        @jax.jit
        def f(Ax, bs):
            tok = sparsax.umf_factor(Ai, Aj, Ax, n)
            return jax.vmap(lambda b: sparsax.umf_solve_factor(tok, b))(bs)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        xs = f(Ax, bs)
        jax.block_until_ready(xs)
        assert sparsax.factorization_count() - c0 == 1
        np.testing.assert_allclose(xs, np.linalg.solve(A, bs.T).T, rtol=1e-10)

    def test_vmap_factor_then_solve(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(13)
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax
        bs = rng.normal(size=(4, n))

        @jax.jit
        def f(Axs, bs):
            toks = jax.vmap(lambda ax: sparsax.umf_factor(Ai, Aj, ax, n))(Axs)
            return jax.vmap(lambda tok, b: sparsax.umf_solve_factor(tok, b))(toks, bs)

        sparsax.clear_cache()
        c0 = sparsax.factorization_count()
        xs = f(Axs, bs)
        jax.block_until_ready(xs)
        assert sparsax.factorization_count() - c0 == 4
        for i, s in enumerate(scales):
            np.testing.assert_allclose(xs[i], np.linalg.solve(s * A, bs[i]), rtol=1e-10)

    def test_vmap_grad(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        rng = np.random.default_rng(14)
        bs = rng.normal(size=(3, n))
        w = np.ones(n)
        gs = jax.vmap(
            lambda b: jax.grad(lambda ax: jnp.dot(w, sparsax.umf_solve(Ai, Aj, ax, b)))(
                Ax
            )
        )(bs)
        assert gs.shape == (3, len(Ax))
        assert np.all(np.isfinite(np.asarray(gs)))

    def test_solve_inside_vmapped_cond(self, nonsym):
        """A batched-but-uniform pattern (lifted by vmapping a cond) is accepted."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        b = np.ones(n)
        preds = np.array([True, False, True])

        @jax.jit
        def f(preds):
            return jax.vmap(
                lambda p: jax.lax.cond(
                    p,
                    lambda: sparsax.umf_solve(Ai, Aj, Ax, b),
                    lambda: jnp.zeros(n),
                )
            )(preds)

        out = np.asarray(f(preds))
        ref = np.linalg.solve(A, b)
        np.testing.assert_allclose(out[0], ref, rtol=1e-10)
        np.testing.assert_allclose(out[1], np.zeros(n), atol=0)
        np.testing.assert_allclose(out[2], ref, rtol=1e-10)


class TestBCOO:
    @staticmethod
    def bcoo(A):
        from jax.experimental import sparse as jsparse

        return jsparse.BCOO.fromdense(jnp.asarray(A))

    def test_umf_solve_bcoo(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        b = np.ones(A.shape[0])
        np.testing.assert_allclose(
            sparsax.umf_solve_bcoo(self.bcoo(A), b),
            np.linalg.solve(A, b),
            rtol=1e-10,
        )

    def test_umf_logdet_bcoo(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        np.testing.assert_allclose(
            float(sparsax.umf_logdet_bcoo(self.bcoo(A))),
            np.linalg.slogdet(A)[1],
            rtol=1e-10,
        )


class TestCache:
    def test_cache_grows_and_clears(self, nonsym):
        Ai, Aj, Ax, A = nonsym
        sparsax.clear_cache()
        assert sparsax.cache_size() == 0
        sparsax.umf_solve(Ai, Aj, Ax, np.ones(A.shape[0]))
        assert sparsax.cache_size() >= 1
        sparsax.clear_cache()
        assert sparsax.cache_size() == 0

    def test_set_umf_cache_size(self, nonsym):
        """Sizing the LRU below the batch makes a factor-once sweep refactor."""
        Ai, Aj, Ax, A = nonsym
        n = A.shape[0]
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax
        b = np.ones(n)

        try:
            sparsax.set_umf_cache_size(4)
            sparsax.clear_cache()
            c0 = sparsax.factorization_count()
            for _ in range(2):
                xs = jax.vmap(lambda ax: sparsax.umf_solve(Ai, Aj, ax, b))(Axs)
                jax.block_until_ready(xs)
            assert sparsax.factorization_count() - c0 == 4

            sparsax.set_umf_cache_size(1)
            sparsax.clear_cache()
            c0 = sparsax.factorization_count()
            for _ in range(2):
                xs = jax.vmap(lambda ax: sparsax.umf_solve(Ai, Aj, ax, b))(Axs)
                jax.block_until_ready(xs)
            assert sparsax.factorization_count() - c0 == 8
        finally:
            sparsax.set_umf_cache_size(32)
            sparsax.clear_cache()
