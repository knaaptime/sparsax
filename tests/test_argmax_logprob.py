"""Tests for argmax_logprob: GHK choice probabilities of a sparse-precision Gaussian."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax
from scipy.stats import multivariate_normal


def sar_precision(n, k=5, rho=0.7, seed=0):
    """Q = (I - ρW)'(I - ρW) for a row-standardised k-NN W on random points.

    k-NN is directed, so W is not symmetrizable: the case in which (I - ρW)
    itself needs an LU backend, while Q is still SPD.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(size=(n, 2))
    d = ((pts[:, None] - pts[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    W = np.zeros((n, n))
    nbrs = np.argsort(d, axis=1)[:, :k]
    for i in range(n):
        W[i, nbrs[i]] = 1.0 / k
    A = np.eye(n) - rho * W
    Q = A.T @ A
    U = sp.coo_matrix(np.triu(Q))
    return U.row.astype(np.int32), U.col.astype(np.int32), U.data, Q


def exact_logprob(Q, mean, chosen):
    """log P(argmax = chosen) from the dense covariance of the differences."""
    Sigma = np.linalg.inv(Q)
    n = Q.shape[0]
    o = np.delete(np.arange(n), chosen)
    Om = (
        Sigma[np.ix_(o, o)]
        - Sigma[o, chosen][:, None]
        - Sigma[chosen, o][None, :]
        + Sigma[chosen, chosen]
    )
    return np.log(
        multivariate_normal(mean=mean[o] - mean[chosen], cov=Om).cdf(np.zeros(n - 1))
    )


def top_competitors(means, chosen, k):
    out = np.full((len(chosen), k), -1, np.int32)
    for p, (m, j) in enumerate(zip(means, chosen)):
        order = [i for i in np.argsort(-m) if i != j][:k]
        out[p, : len(order)] = order
    return out


@pytest.fixture
def problem():
    Ai, Aj, Ax, Q = sar_precision(25)
    rng = np.random.default_rng(3)
    means = rng.normal(scale=0.8, size=(6, 25))
    chosen = np.array([int(np.argsort(m)[-1 - p]) for p, m in enumerate(means)])
    return Ai, Aj, Ax, Q, means, chosen


class TestValues:
    @pytest.mark.parametrize("K", [0, 5])
    def test_matches_exact(self, problem, K):
        Ai, Aj, Ax, Q, means, chosen = problem
        comp = top_competitors(means, chosen, K) if K else None
        got = np.asarray(
            sparsax.argmax_logprob(Ai, Aj, Ax, means, chosen, comp, n_draws=20000)
        )
        want = np.array([exact_logprob(Q, m, j) for m, j in zip(means, chosen)])
        np.testing.assert_allclose(got, want, atol=0.03)

    def test_probabilities_sum_to_one(self):
        Ai, Aj, Ax, Q = sar_precision(12, k=3)
        mean = np.random.default_rng(0).normal(scale=0.5, size=12)
        logp = sparsax.argmax_logprob(
            Ai, Aj, Ax, np.tile(mean, (12, 1)), np.arange(12), n_draws=20000
        )
        assert abs(float(jnp.exp(logp).sum()) - 1.0) < 0.01

    def test_single_path_returns_scalar(self, problem):
        Ai, Aj, Ax, Q, means, chosen = problem
        one = sparsax.argmax_logprob(Ai, Aj, Ax, means[0], int(chosen[0]), n_draws=50)
        assert jnp.ndim(one) == 0

    def test_shift_invariant(self, problem):
        """Only differences from the chosen mean matter, exactly."""
        Ai, Aj, Ax, Q, means, chosen = problem
        a = sparsax.argmax_logprob(Ai, Aj, Ax, means, chosen, n_draws=50)
        b = sparsax.argmax_logprob(Ai, Aj, Ax, means + 3.7, chosen, n_draws=50)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


class TestDraws:
    def test_reproducible(self, problem):
        Ai, Aj, Ax, Q, means, chosen = problem

        def f():
            return np.asarray(
                sparsax.argmax_logprob(Ai, Aj, Ax, means, chosen, n_draws=50, seed=4)
            )

        np.testing.assert_array_equal(f(), f())

    def test_draws_follow_keys_not_batch(self, problem):
        """A path's draws depend on its key, not on its position in the batch."""
        Ai, Aj, Ax, Q, means, chosen = problem
        keys = np.array([10, 11, 12, 13, 14, 15])
        full = sparsax.argmax_logprob(Ai, Aj, Ax, means, chosen, n_draws=50, keys=keys)
        part = sparsax.argmax_logprob(
            Ai, Aj, Ax, means[[4, 1]], chosen[[4, 1]], n_draws=50, keys=keys[[4, 1]]
        )
        np.testing.assert_array_equal(np.asarray(full)[[4, 1]], np.asarray(part))

    def test_seed_changes_draws(self, problem):
        Ai, Aj, Ax, Q, means, chosen = problem
        a = sparsax.argmax_logprob(Ai, Aj, Ax, means, chosen, n_draws=50, seed=1)
        b = sparsax.argmax_logprob(Ai, Aj, Ax, means, chosen, n_draws=50, seed=2)
        assert not np.allclose(a, b)

    def test_competitors_reduce_variance(self):
        """Drawing the likeliest rivals first is the point of competitors."""
        Ai, Aj, Ax, Q = sar_precision(150, k=6, rho=0.85, seed=2)
        rng = np.random.default_rng(5)
        means = rng.normal(scale=0.6, size=(20, 150))
        chosen = np.array([int(np.argsort(m)[-3]) for m in means])
        comp = top_competitors(means, chosen, 15)

        def sd(c):
            reps = np.array(
                [
                    sparsax.argmax_logprob(
                        Ai, Aj, Ax, means, chosen, c, n_draws=50, seed=s
                    )
                    for s in range(8)
                ]
            )
            return np.median(reps.std(axis=0))

        assert sd(comp) < sd(None)


class TestGrad:
    @pytest.mark.parametrize("K", [0, 4])
    def test_matches_finite_differences(self, problem, K):
        Ai, Aj, Ax, Q, means, chosen = problem
        comp = top_competitors(means, chosen, K) if K else None

        def f(mu):
            return jnp.sum(
                sparsax.argmax_logprob(
                    Ai, Aj, Ax, mu, chosen, comp, n_draws=100, seed=7
                )
            )

        g = np.asarray(jax.grad(f)(jnp.asarray(means)))
        rng = np.random.default_rng(0)
        h = 1e-5
        for _ in range(15):
            p, k = rng.integers(means.shape[0]), rng.integers(means.shape[1])
            e = np.zeros_like(means)
            e[p, k] = h
            fd = (float(f(means + e)) - float(f(means - e))) / (2 * h)
            assert abs(g[p, k] - fd) < 1e-7 * max(1.0, abs(fd))

    def test_rows_sum_to_zero(self, problem):
        Ai, Aj, Ax, Q, means, chosen = problem
        g = jax.grad(
            lambda mu: jnp.sum(
                sparsax.argmax_logprob(Ai, Aj, Ax, mu, chosen, n_draws=50)
            )
        )(jnp.asarray(means))
        np.testing.assert_allclose(np.asarray(g).sum(axis=1), 0.0, atol=1e-12)

    def test_jit(self, problem):
        Ai, Aj, Ax, Q, means, chosen = problem

        def f(mu):
            return jnp.sum(sparsax.argmax_logprob(Ai, Aj, Ax, mu, chosen, n_draws=50))

        v1, g1 = jax.value_and_grad(f)(jnp.asarray(means))
        v2, g2 = jax.jit(jax.value_and_grad(f))(jnp.asarray(means))
        np.testing.assert_allclose(v1, v2, rtol=1e-12)
        np.testing.assert_allclose(g1, g2, rtol=1e-12, atol=1e-14)


class TestErrors:
    def test_chosen_among_competitors(self, problem):
        Ai, Aj, Ax, Q, means, chosen = problem
        comp = np.tile(np.array([[int(chosen[0]), -1]], np.int32), (6, 1))
        with pytest.raises(Exception, match="chosen"):
            sparsax.argmax_logprob(Ai, Aj, Ax, means, chosen, comp, n_draws=10)

    def test_duplicate_competitor(self, problem):
        Ai, Aj, Ax, Q, means, chosen = problem
        other = [
            int(np.argsort(-m)[0]) if np.argsort(-m)[0] != j else int(np.argsort(-m)[1])
            for m, j in zip(means, chosen)
        ]
        comp = np.array([[o, o] for o in other], np.int32)
        with pytest.raises(Exception, match="duplicate"):
            sparsax.argmax_logprob(Ai, Aj, Ax, means, chosen, comp, n_draws=10)

    def test_chosen_out_of_range(self, problem):
        Ai, Aj, Ax, Q, means, chosen = problem
        bad = chosen.copy()
        bad[0] = 99
        with pytest.raises(Exception, match="out of range"):
            sparsax.argmax_logprob(Ai, Aj, Ax, means, bad, n_draws=10)
