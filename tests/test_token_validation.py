"""Factor tokens name their backend, and token solves check b against the factor.

A token is two plain integers, and each backend numbers its own tokens from 1, so
a key from one backend can name an unrelated live factor in another.  Nothing in
the array says otherwise unless the token carries its issuer.  Likewise only the
factor knows its dimension: a token solve takes n from b, and UMFPACK and KLU
trust it, so a b of the wrong length used to read and write out of bounds
(UMFPACK) or return a tail of b unchanged (KLU).
"""

from collections import namedtuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import sparsax

Backend = namedtuple("Backend", "factor solve logdet")
BACKENDS = {
    "cholmod": Backend(sparsax.factor, sparsax.solve_factor, sparsax.logdet_factor),
    "klu": Backend(
        sparsax.lu_factor, sparsax.lu_solve_factor, sparsax.lu_logdet_factor
    ),
    "umfpack": Backend(
        sparsax.umf_factor, sparsax.umf_solve_factor, sparsax.umf_logdet_factor
    ),
}
MISMATCHED = [(a, b) for a in BACKENDS for b in BACKENDS if a != b]
N = 16


@pytest.fixture
def spd():
    """A symmetric tridiagonal SPD system in full COO, valid for all three backends."""
    A = 3.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    Ai, Aj = np.nonzero(A)
    return Ai.astype(np.int32), Aj.astype(np.int32), A[Ai, Aj], A


class TestOwnTokenStillWorks:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_solve_and_logdet(self, spd, backend):
        Ai, Aj, Ax, A = spd
        be = BACKENDS[backend]
        b = np.arange(1.0, N + 1.0)
        token = be.factor(Ai, Aj, Ax, N)
        np.testing.assert_allclose(be.solve(token, b), np.linalg.solve(A, b))
        np.testing.assert_allclose(float(be.logdet(token)), np.linalg.slogdet(A)[1])

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_vmap_over_tokens_under_jit(self, spd, backend):
        Ai, Aj, Ax, A = spd
        be = BACKENDS[backend]
        scales = jnp.array([1.0, 2.0, 3.0])
        rhs = jnp.tile(jnp.arange(1.0, N + 1.0), (3, 1))

        def one(s, b):
            return be.solve(be.factor(Ai, Aj, s * Ax, N), b)

        got = jax.jit(jax.vmap(one))(scales, rhs)
        want = np.stack(
            [np.linalg.solve(float(s) * A, np.asarray(b)) for s, b in zip(scales, rhs)]
        )
        np.testing.assert_allclose(got, want)


class TestWrongBackend:
    @pytest.mark.parametrize("issuer, consumer", MISMATCHED)
    def test_solve_rejects(self, spd, issuer, consumer):
        Ai, Aj, Ax, _ = spd
        token = BACKENDS[issuer].factor(Ai, Aj, Ax, N)
        with pytest.raises(Exception, match="not interchangeable"):
            BACKENDS[consumer].solve(token, np.ones(N))

    @pytest.mark.parametrize("issuer, consumer", MISMATCHED)
    def test_logdet_rejects(self, spd, issuer, consumer):
        Ai, Aj, Ax, _ = spd
        token = BACKENDS[issuer].factor(Ai, Aj, Ax, N)
        with pytest.raises(Exception, match="not interchangeable"):
            BACKENDS[consumer].logdet(token)

    @pytest.mark.parametrize("issuer, consumer", MISMATCHED)
    def test_vmap_over_b_rejects(self, spd, issuer, consumer):
        Ai, Aj, Ax, _ = spd
        token = BACKENDS[issuer].factor(Ai, Aj, Ax, N)
        solve = BACKENDS[consumer].solve
        with pytest.raises(Exception, match="not interchangeable"):
            jax.vmap(lambda b: solve(token, b))(jnp.ones((3, N)))

    @pytest.mark.parametrize("issuer, consumer", MISMATCHED)
    def test_vmap_over_tokens_rejects(self, spd, issuer, consumer):
        Ai, Aj, Ax, _ = spd
        factor, solve = BACKENDS[issuer].factor, BACKENDS[consumer].solve
        tokens = jax.vmap(lambda s: factor(Ai, Aj, s * Ax, N))(jnp.array([1.0, 2.0]))
        with pytest.raises(Exception, match="not interchangeable"):
            jax.vmap(solve)(tokens, jnp.ones((2, N)))

    def test_error_names_the_issuer(self, spd):
        Ai, Aj, Ax, _ = spd
        token = sparsax.factor(Ai, Aj, Ax, N)
        with pytest.raises(Exception, match="CHOLMOD"):
            sparsax.lu_solve_factor(token, np.ones(N))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_fabricated_token_rejects(self, backend):
        token = jnp.array([1, 0], jnp.int64)
        with pytest.raises(Exception, match="invalid factor token"):
            BACKENDS[backend].solve(token, np.ones(N))


class TestWrongLength:
    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("delta", [-1, 1])
    def test_vector(self, spd, backend, delta):
        Ai, Aj, Ax, _ = spd
        be = BACKENDS[backend]
        token = be.factor(Ai, Aj, Ax, N)
        with pytest.raises(Exception, match=f"16 x 16.*{N + delta} rows"):
            be.solve(token, np.ones(N + delta))

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("delta", [-1, 1])
    def test_matrix(self, spd, backend, delta):
        Ai, Aj, Ax, _ = spd
        be = BACKENDS[backend]
        token = be.factor(Ai, Aj, Ax, N)
        with pytest.raises(Exception, match=f"16 x 16.*{N + delta} rows"):
            be.solve(token, np.ones((N + delta, 3)))

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("delta", [-1, 1])
    def test_vmap_over_b(self, spd, backend, delta):
        Ai, Aj, Ax, _ = spd
        be = BACKENDS[backend]
        token = be.factor(Ai, Aj, Ax, N)
        with pytest.raises(Exception, match=f"16 x 16.*{N + delta} rows"):
            jax.vmap(lambda b: be.solve(token, b))(jnp.ones((3, N + delta)))

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("delta", [-1, 1])
    def test_vmap_over_tokens(self, spd, backend, delta):
        Ai, Aj, Ax, _ = spd
        be = BACKENDS[backend]
        tokens = jax.vmap(lambda s: be.factor(Ai, Aj, s * Ax, N))(jnp.array([1.0, 2.0]))
        with pytest.raises(Exception, match=f"16 x 16.*{N + delta} rows"):
            jax.vmap(be.solve)(tokens, jnp.ones((2, N + delta)))
