"""Scrambled Sobol draws for argmax_logprob's GHK simulator.

GHK estimates a probability with a mean over draws, so its variance responds to
how evenly the draws fill the cube.  Sobol points do that far better than
independent ones in the first dimensions, which is where GHK places the rivals that
matter.  Nested uniform (Owen) scrambling makes every point uniform on the cube, so
the estimator stays unbiased and each (seed, key) gets its own independent points.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import sparsax
from scipy.stats import qmc

from test_argmax_logprob import exact_logprob, sar_precision, top_competitors


def uniforms(n_draws, n_steps, sobol_dims, seed=0, key=0):
    return sparsax._cpp.ghk_uniforms_np(n_draws, n_steps, sobol_dims, seed, key)


class TestPoints:
    def test_unscrambled_points_match_scipy_in_every_dimension(self):
        got = sparsax._cpp.sobol_points_np(1024, 256)
        want = qmc.Sobol(256, scramble=False).random(1024)
        assert got.shape == want.shape
        # scipy keeps 30 bits of the same direction numbers.
        assert np.max(np.abs(got - want)) < 2.0**-30

    @pytest.mark.parametrize("m", [1, 4, 7, 11, 16, 17])
    def test_each_dimension_has_one_point_per_stratum(self, m):
        n = 2**m
        u = uniforms(n, 64, 64, seed=3, key=11)
        for d in range(64):
            np.testing.assert_array_equal(
                np.sort(np.floor(u[:, d] * n).astype(np.int64)), np.arange(n)
            )

    @pytest.mark.parametrize("m", [4, 7, 10])
    def test_first_two_dimensions_stay_a_net_after_scrambling(self, m):
        """Sobol's first two dimensions: one point per box of area 1 / n, any shape."""
        n = 2**m
        u = uniforms(n, 2, 2, seed=5, key=2)
        for a in range(m + 1):
            ix = np.floor(u[:, 0] * 2**a).astype(np.int64)
            iy = np.floor(u[:, 1] * 2 ** (m - a)).astype(np.int64)
            assert len(set(zip(ix.tolist(), iy.tolist()))) == n

    def test_points_lie_strictly_inside_the_unit_interval(self):
        u = uniforms(2**12, 256, 256, seed=1, key=1)
        assert u.min() > 0.0 and u.max() < 1.0

    def test_more_draws_extend_the_same_points(self):
        small, large = (
            uniforms(64, 8, 8, seed=2, key=3),
            uniforms(128, 8, 8, seed=2, key=3),
        )
        np.testing.assert_array_equal(small, large[:64])

    def test_steps_past_the_sobol_dimensions_are_the_independent_draws(self):
        mixed, iid = (
            uniforms(32, 12, 5, seed=4, key=9),
            uniforms(32, 12, 0, seed=4, key=9),
        )
        np.testing.assert_array_equal(mixed[:, 5:], iid[:, 5:])
        assert not np.allclose(mixed[:, :5], iid[:, :5])

    def test_reproducible_for_a_seed_and_key(self):
        np.testing.assert_array_equal(
            uniforms(16, 8, 8, 7, 7), uniforms(16, 8, 8, 7, 7)
        )

    @pytest.mark.parametrize("other", [(8, 7), (7, 8)])
    def test_seed_and_key_each_change_the_scramble(self, other):
        a = uniforms(16, 8, 8, 7, 7)
        b = uniforms(16, 8, 8, *other)
        assert not np.any(a == b)

    def test_a_point_is_uniform_over_scrambles(self):
        """One point, many keys: uniform in each dimension and uncorrelated across them."""
        keys = 4000
        u = np.stack([uniforms(4, 6, 6, seed=0, key=k) for k in range(keys)])
        for r in range(4):
            point = u[:, r, :]
            np.testing.assert_allclose(
                point.mean(axis=0), 0.5, atol=4.0 / np.sqrt(12 * keys)
            )
            np.testing.assert_allclose(point.var(axis=0), 1.0 / 12.0, atol=0.006)
            off = np.corrcoef(point.T)[np.triu_indices(6, k=1)]
            assert np.max(np.abs(off)) < 4.0 / np.sqrt(keys)


@pytest.fixture
def small_problem():
    Ai, Aj, Ax, Q = sar_precision(6, k=2, rho=0.6, seed=1)
    mean = np.array([0.4, -0.2, 0.1, 0.3, -0.5, 0.0])
    return Ai, Aj, Ax, Q, mean, 0


class TestEstimator:
    @pytest.mark.parametrize("K", [0, 5])
    def test_matches_exact(self, K):
        Ai, Aj, Ax, Q = sar_precision(25)
        rng = np.random.default_rng(3)
        means = rng.normal(scale=0.8, size=(6, 25))
        chosen = np.array([int(np.argsort(m)[-1 - p]) for p, m in enumerate(means)])
        comp = top_competitors(means, chosen, K) if K else None
        got = sparsax.argmax_logprob(
            Ai, Aj, Ax, means, chosen, comp, n_draws=2**14, draws="sobol"
        )
        want = np.array([exact_logprob(Q, m, j) for m, j in zip(means, chosen)])
        np.testing.assert_allclose(got, want, atol=0.02)

    @pytest.mark.parametrize("n_draws", [1, 8, 10, 32])
    def test_probability_is_unbiased(self, small_problem, n_draws):
        """Over keys the estimate centres on the truth, also off a power of two."""
        Ai, Aj, Ax, Q, mean, j = small_problem
        paths = 8000
        keys = jnp.arange(paths, dtype=jnp.int64)
        lp = sparsax.argmax_logprob(
            Ai,
            Aj,
            Ax,
            np.tile(mean, (paths, 1)),
            np.full(paths, j),
            n_draws=n_draws,
            keys=keys,
            draws="sobol",
        )
        p = np.exp(np.asarray(lp))
        se = p.std(ddof=1) / np.sqrt(paths)
        assert abs(p.mean() - np.exp(exact_logprob(Q, mean, j))) < 4.0 * se + 1e-3

    def test_probabilities_sum_to_one(self):
        Ai, Aj, Ax, Q = sar_precision(12, k=3)
        mean = np.random.default_rng(0).normal(scale=0.5, size=12)
        lp = sparsax.argmax_logprob(
            Ai,
            Aj,
            Ax,
            np.tile(mean, (12, 1)),
            np.arange(12),
            n_draws=2**14,
            draws="sobol",
        )
        np.testing.assert_allclose(np.exp(np.asarray(lp)).sum(), 1.0, atol=0.01)

    def test_variance_falls_well_below_independent_draws(self):
        Ai, Aj, Ax, Q = sar_precision(150, k=6, rho=0.6, seed=2)
        rng = np.random.default_rng(5)
        means = rng.normal(scale=1.0, size=(20, 150))
        chosen = means.argmax(axis=1)
        comp = top_competitors(means, chosen, 15)

        def variance(draws):
            reps = np.stack(
                [
                    sparsax.argmax_logprob(
                        Ai, Aj, Ax, means, chosen, comp, n_draws=64, seed=s, draws=draws
                    )
                    for s in range(16)
                ]
            )
            return np.median(reps.var(axis=0))

        assert variance("sobol") < variance("iid") / 2.0

    def test_gradient_matches_finite_differences(self):
        Ai, Aj, Ax, Q = sar_precision(25)
        rng = np.random.default_rng(3)
        means = rng.normal(scale=0.8, size=(3, 25))
        chosen = means.argmax(axis=1)
        comp = top_competitors(means, chosen, 5)

        def f(m):
            return sparsax.argmax_logprob(
                Ai, Aj, Ax, m, chosen, comp, n_draws=64, draws="sobol"
            ).sum()

        g = np.asarray(jax.grad(f)(jnp.asarray(means)))
        direction = rng.normal(size=means.shape)
        h = 1e-5
        fd = (float(f(means + h * direction)) - float(f(means - h * direction))) / (
            2 * h
        )
        np.testing.assert_allclose((g * direction).sum(), fd, rtol=1e-5)

    def test_draws_follow_keys_not_batch(self):
        Ai, Aj, Ax, Q = sar_precision(25)
        means = np.random.default_rng(3).normal(scale=0.8, size=(4, 25))
        chosen = means.argmax(axis=1)
        keys = jnp.array([10, 11, 12, 13], jnp.int64)
        together = sparsax.argmax_logprob(
            Ai, Aj, Ax, means, chosen, n_draws=32, keys=keys, draws="sobol"
        )
        one = sparsax.argmax_logprob(
            Ai,
            Aj,
            Ax,
            means[2:3],
            chosen[2:3],
            n_draws=32,
            keys=keys[2:3],
            draws="sobol",
        )
        assert float(together[2]) == float(one[0])

    def test_rejects_unknown_scheme(self):
        Ai, Aj, Ax, Q = sar_precision(6, k=2)
        with pytest.raises(ValueError, match="draws"):
            sparsax.argmax_logprob(Ai, Aj, Ax, np.zeros(6), 0, draws="halton")

    def test_rejects_more_draws_than_the_sequence_has(self):
        Ai, Aj, Ax, Q = sar_precision(6, k=2)
        with pytest.raises(ValueError, match="2\\*\\*32"):
            sparsax.argmax_logprob(
                Ai, Aj, Ax, np.zeros(6), 0, n_draws=2**32 + 1, draws="sobol"
            )
