"""set_options(on_failure="nan"): a failed factorization is a value, not an error.

A sampler that proposes a matrix outside the positive definite or nonsingular
region needs to reject the proposal, and under vmap it needs the other chains to
survive.  An exception cannot do either: it aborts the whole batch.  With "nan"
the failed element returns NaN and the rest are unaffected.  Only numerical
failure changes; a bad argument is a bug in the caller and always raises.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import sparsax

SOLVERS = {
    "cholmod": sparsax.solve,
    "klu": sparsax.lu_solve,
    "umfpack": sparsax.umf_solve,
}
LOGDETS = {
    "cholmod": sparsax.logdet,
    "klu": sparsax.lu_logdet,
    "umfpack": sparsax.umf_logdet,
}
TOKENS = {
    "cholmod": (sparsax.factor, sparsax.solve_factor, sparsax.logdet_factor),
    "klu": (sparsax.lu_factor, sparsax.lu_solve_factor, sparsax.lu_logdet_factor),
    "umfpack": (sparsax.umf_factor, sparsax.umf_solve_factor, sparsax.umf_logdet_factor),
}
N = 12


@pytest.fixture
def nan_on_failure():
    sparsax.set_options(on_failure="nan")
    yield
    sparsax.set_options(on_failure="raise")


@pytest.fixture
def system():
    """Tridiagonal SPD in full COO, plus values each backend must reject."""
    A = 3.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    Ai, Aj = np.nonzero(A)
    Ai, Aj = Ai.astype(np.int32), Aj.astype(np.int32)
    Ax = A[Ai, Aj]
    singular = Ax.copy()
    singular[Aj == 0] = 0.0  # a zero column: singular, so also not positive definite
    has_nan = Ax.copy()
    has_nan[np.flatnonzero(Ai < Aj)[2]] = np.nan  # off-diagonal, upper: all backends read it
    return Ai, Aj, Ax, A, singular, has_nan


class TestDefaultRaises:
    @pytest.mark.parametrize("backend", SOLVERS)
    def test_singular_solve(self, system, backend):
        Ai, Aj, _, _, singular, _ = system
        with pytest.raises(Exception, match="singular|positive definite"):
            SOLVERS[backend](Ai, Aj, singular, np.ones(N))

    @pytest.mark.parametrize("backend", LOGDETS)
    def test_singular_logdet(self, system, backend):
        Ai, Aj, _, _, singular, _ = system
        with pytest.raises(Exception, match="singular|positive definite"):
            LOGDETS[backend](Ai, Aj, singular, N)

    @pytest.mark.parametrize("backend", SOLVERS)
    def test_nan_values_solve(self, system, backend):
        Ai, Aj, _, _, _, has_nan = system
        with pytest.raises(Exception, match="non-finite|singular|positive definite"):
            SOLVERS[backend](Ai, Aj, has_nan, np.ones(N))

    @pytest.mark.parametrize("backend", LOGDETS)
    def test_nan_values_logdet(self, system, backend):
        Ai, Aj, _, _, _, has_nan = system
        with pytest.raises(Exception, match="non-finite|singular|positive definite"):
            LOGDETS[backend](Ai, Aj, has_nan, N)

    @pytest.mark.parametrize("backend", TOKENS)
    def test_singular_factor(self, system, backend):
        Ai, Aj, _, _, singular, _ = system
        with pytest.raises(Exception, match="singular|positive definite"):
            TOKENS[backend][0](Ai, Aj, singular, N)


@pytest.mark.usefixtures("nan_on_failure")
class TestNanValues:
    @pytest.mark.parametrize("backend", SOLVERS)
    @pytest.mark.parametrize("which", ["singular", "has_nan"])
    def test_solve(self, system, backend, which):
        Ai, Aj, _, _, singular, has_nan = system
        bad = singular if which == "singular" else has_nan
        x = SOLVERS[backend](Ai, Aj, bad, np.ones(N))
        assert x.shape == (N,) and np.all(np.isnan(x))

    @pytest.mark.parametrize("backend", SOLVERS)
    def test_block_of_many_columns(self, system, backend):
        Ai, Aj, _, _, singular, _ = system
        x = SOLVERS[backend](Ai, Aj, singular, np.ones((N, 130)))
        assert x.shape == (N, 130) and np.all(np.isnan(x))

    @pytest.mark.parametrize("backend", LOGDETS)
    @pytest.mark.parametrize("which", ["singular", "has_nan"])
    def test_logdet(self, system, backend, which):
        Ai, Aj, _, _, singular, has_nan = system
        bad = singular if which == "singular" else has_nan
        assert np.isnan(LOGDETS[backend](Ai, Aj, bad, N))

    @pytest.mark.parametrize("backend", SOLVERS)
    def test_good_values_are_untouched(self, system, backend):
        Ai, Aj, Ax, A, _, _ = system
        b = np.arange(1.0, N + 1.0)
        np.testing.assert_allclose(SOLVERS[backend](Ai, Aj, Ax, b), np.linalg.solve(A, b))

    @pytest.mark.parametrize("backend", SOLVERS)
    def test_only_the_failing_batch_element_is_nan(self, system, backend):
        Ai, Aj, Ax, A, singular, _ = system
        solve = SOLVERS[backend]
        b = np.arange(1.0, N + 1.0)
        axs = jnp.stack([Ax, jnp.asarray(singular), 2.0 * Ax])
        x = jax.jit(jax.vmap(lambda a: solve(Ai, Aj, a, b)))(axs)
        np.testing.assert_allclose(x[0], np.linalg.solve(A, b))
        assert np.all(np.isnan(x[1]))
        np.testing.assert_allclose(x[2], np.linalg.solve(2.0 * A, b))

    @pytest.mark.parametrize("backend", LOGDETS)
    def test_only_the_failing_logdet_is_nan(self, system, backend):
        Ai, Aj, Ax, A, singular, _ = system
        logdet = LOGDETS[backend]
        axs = jnp.stack([Ax, jnp.asarray(singular)])
        ld = jax.vmap(lambda a: logdet(Ai, Aj, a, N))(axs)
        np.testing.assert_allclose(ld[0], np.linalg.slogdet(A)[1])
        assert np.isnan(ld[1])

    def test_selinv(self, system):
        Ai, Aj, _, _, singular, _ = system
        assert np.all(np.isnan(sparsax.selinv(Ai, Aj, singular, N)))

    def test_vmapped_value_and_grad_isolates_the_failing_chain(self, system):
        """The sampler's shape: one chain steps outside the PD region."""
        Ai, Aj, Ax, _, _, _ = system
        b = jnp.arange(1.0, N + 1.0)
        diag = jnp.asarray(Ai == Aj)

        def logp(s):
            ax = jnp.where(diag, s * Ax, Ax)
            x = sparsax.solve(Ai, Aj, ax, b)
            return -0.5 * b @ x + 0.5 * sparsax.logdet(Ai, Aj, ax, N)

        scales = jnp.array([1.0, -1.0, 2.0])
        v, g = jax.jit(jax.vmap(jax.value_and_grad(logp)))(scales)
        assert np.isnan(v[1]) and np.isnan(g[1])
        for chain in (0, 2):
            v1, g1 = jax.value_and_grad(logp)(scales[chain])
            np.testing.assert_allclose(v[chain], v1)
            np.testing.assert_allclose(g[chain], g1)

    @pytest.mark.parametrize("supernodal", ["simplicial", "supernodal"])
    def test_cholmod_factor_forms(self, system, supernodal):
        Ai, Aj, Ax, A, singular, _ = system
        sparsax.set_options(supernodal=supernodal)
        try:
            assert np.all(np.isnan(sparsax.solve(Ai, Aj, singular, np.ones(N))))
            assert np.isnan(sparsax.logdet(Ai, Aj, singular, N))
            assert np.all(np.isnan(sparsax.selinv(Ai, Aj, singular, N)))
            np.testing.assert_allclose(
                sparsax.logdet(Ai, Aj, Ax, N), np.linalg.slogdet(A)[1]
            )
        finally:
            sparsax.set_options(supernodal="auto")

    def test_factor_solve_and_its_batch(self, system):
        Ai, Aj, Ax, A, singular, _ = system
        b = np.arange(1.0, N + 1.0)
        (x,), ld = sparsax.factor_solve(
            Ai, Aj, singular, [(b, sparsax.MODE_A)], want_logdet=True
        )
        assert np.all(np.isnan(x)) and np.isnan(ld)

        axs = jnp.stack([jnp.asarray(singular), Ax])

        def one(a):
            (y,), d = sparsax.factor_solve(
                Ai, Aj, a, [(b, sparsax.MODE_A)], want_logdet=True, n=N
            )
            return y, d

        xs, lds = jax.vmap(one)(axs)
        assert np.all(np.isnan(xs[0])) and np.isnan(lds[0])
        np.testing.assert_allclose(xs[1], np.linalg.solve(A, b))
        np.testing.assert_allclose(lds[1], np.linalg.slogdet(A)[1])

    def test_sample_gaussian(self, system):
        Ai, Aj, _, _, singular, _ = system
        draw, mean, ld = sparsax.sample_gaussian(
            Ai, Aj, singular, np.zeros(N), np.ones(N), want_logdet=True
        )
        assert np.all(np.isnan(draw)) and np.all(np.isnan(mean)) and np.isnan(ld)

    def test_update_solve_base_failure(self, system):
        Ai, Aj, _, _, singular, _ = system
        x, ld = sparsax.update_solve(
            Ai, Aj, singular, 0.1 * np.ones(N), np.ones(N), return_logdet=True
        )
        assert np.all(np.isnan(x)) and np.isnan(ld)

    def test_update_solve_downdate_failure(self, system):
        Ai, Aj, Ax, _, _, _ = system
        x, ld = sparsax.update_solve(
            Ai, Aj, Ax, 10.0 * np.ones(N), np.ones(N), downdate=True, return_logdet=True
        )
        assert np.all(np.isnan(x)) and np.isnan(ld)

    def test_argmax_logprob(self, system):
        Ai, Aj, _, _, singular, _ = system
        means = np.zeros((3, N))
        chosen = np.array([0, 1, 2])
        lp = sparsax.argmax_logprob(Ai, Aj, singular, means, chosen, n_draws=4)
        assert np.all(np.isnan(lp))
        g = jax.grad(
            lambda m: sparsax.argmax_logprob(Ai, Aj, singular, m, chosen, n_draws=4).sum()
        )(jnp.asarray(means))
        assert np.all(np.isnan(g))


@pytest.mark.usefixtures("nan_on_failure")
class TestNanTokens:
    @pytest.mark.parametrize("backend", TOKENS)
    def test_solve_and_logdet(self, system, backend):
        Ai, Aj, _, _, singular, _ = system
        factor, solve, logdet = TOKENS[backend]
        token = factor(Ai, Aj, singular, N)
        assert np.all(np.isnan(solve(token, np.ones(N))))
        assert np.all(np.isnan(solve(token, np.ones((N, 3)))))
        assert np.isnan(logdet(token))

    @pytest.mark.parametrize("backend", TOKENS)
    def test_batch_of_tokens(self, system, backend):
        Ai, Aj, Ax, A, singular, _ = system
        factor, solve, logdet = TOKENS[backend]
        axs = jnp.stack([Ax, jnp.asarray(singular)])
        tokens = jax.vmap(lambda a: factor(Ai, Aj, a, N))(axs)
        b = np.arange(1.0, N + 1.0)
        x = jax.vmap(solve, in_axes=(0, None))(tokens, jnp.asarray(b))
        np.testing.assert_allclose(x[0], np.linalg.solve(A, b))
        assert np.all(np.isnan(x[1]))
        ld = jax.vmap(logdet)(tokens)
        np.testing.assert_allclose(ld[0], np.linalg.slogdet(A)[1])
        assert np.isnan(ld[1])

    @pytest.mark.parametrize("backend", TOKENS)
    def test_failed_token_stays_nan_after_the_option_changes(self, system, backend):
        Ai, Aj, _, _, singular, _ = system
        factor, solve, _ = TOKENS[backend]
        token = factor(Ai, Aj, singular, N)
        sparsax.set_options(on_failure="raise")
        assert np.all(np.isnan(solve(token, np.ones(N))))

    def test_failed_token_still_names_its_backend(self, system):
        Ai, Aj, _, _, singular, _ = system
        token = sparsax.factor(Ai, Aj, singular, N)
        with pytest.raises(Exception, match="not interchangeable"):
            sparsax.lu_solve_factor(token, np.ones(N))


@pytest.mark.usefixtures("nan_on_failure")
class TestBadArgumentsStillRaise:
    @pytest.mark.parametrize("backend", SOLVERS)
    def test_coo_index_out_of_range(self, system, backend):
        Ai, Aj, Ax, _, _, _ = system
        bad_i = Ai.copy()
        bad_i[0] = 99
        with pytest.raises(Exception, match="out of range"):
            SOLVERS[backend](bad_i, Aj, Ax, np.ones(N))

    @pytest.mark.parametrize("backend", TOKENS)
    def test_wrong_length_b(self, system, backend):
        Ai, Aj, Ax, _, _, _ = system
        factor, solve, _ = TOKENS[backend]
        token = factor(Ai, Aj, Ax, N)
        with pytest.raises(Exception, match="rows"):
            solve(token, np.ones(N + 1))

    @pytest.mark.parametrize("backend", TOKENS)
    def test_stale_token(self, system, backend):
        Ai, Aj, Ax, _, _, _ = system
        factor, solve, _ = TOKENS[backend]
        token = factor(Ai, Aj, Ax, N)
        sparsax.clear_cache()
        with pytest.raises(Exception, match="stale factor token"):
            solve(token, np.ones(N))

    def test_argmax_chosen_out_of_range(self, system):
        Ai, Aj, Ax, _, _, _ = system
        with pytest.raises(Exception, match="out of range"):
            sparsax.argmax_logprob(Ai, Aj, Ax, np.zeros(N), 99, n_draws=4)


class TestOption:
    @pytest.mark.parametrize("backend", SOLVERS)
    def test_failure_is_not_remembered(self, system, backend):
        """The same values must raise once the option is back to "raise"."""
        Ai, Aj, Ax, A, singular, _ = system
        solve = SOLVERS[backend]
        sparsax.set_options(on_failure="nan")
        try:
            assert np.all(np.isnan(solve(Ai, Aj, singular, np.ones(N))))
        finally:
            sparsax.set_options(on_failure="raise")
        with pytest.raises(Exception, match="singular|positive definite"):
            solve(Ai, Aj, singular, np.ones(N))
        b = np.ones(N)
        np.testing.assert_allclose(solve(Ai, Aj, Ax, b), np.linalg.solve(A, b))

    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="on_failure"):
            sparsax.set_options(on_failure="ignore")

    def test_changing_it_keeps_the_caches_and_the_factor_form(self, system):
        Ai, Aj, Ax, _, _, _ = system
        sparsax.clear_cache()
        jax.block_until_ready(sparsax.logdet(Ai, Aj, Ax, N))
        count = sparsax.factorization_count()
        size = sparsax.cache_size()
        sparsax.set_options(on_failure="nan")
        sparsax.set_options(on_failure="raise")
        assert sparsax.cache_size() == size
        jax.block_until_ready(sparsax.logdet(Ai, Aj, Ax, N))
        assert sparsax.factorization_count() == count


class TestPyTensor:
    @pytest.fixture
    def frontend(self):
        pytest.importorskip("pytensor")
        import pytensor
        import pytensor.tensor as pt
        import sparsax.pytensor as cjpt

        return pytensor, pt, cjpt

    def test_logp_and_grad_return_nan(self, system, frontend, nan_on_failure):
        pytensor, pt, cjpt = frontend
        Ai, Aj, Ax, A, singular, _ = system
        upper = Ai <= Aj
        Ai, Aj, Ax, singular = Ai[upper], Aj[upper], Ax[upper], singular[upper]
        b = np.arange(1.0, N + 1.0)
        ax = pt.dvector("ax")
        logp = -0.5 * pt.dot(b, cjpt.solve(Ai, Aj, ax, b)) + 0.5 * cjpt.logdet(Ai, Aj, ax, N)
        f = pytensor.function([ax], [logp, pt.grad(logp, ax)])
        v, g = f(singular)
        assert np.isnan(v) and np.all(np.isnan(g))
        v, g = f(Ax)
        want = -0.5 * b @ np.linalg.solve(A, b) + 0.5 * np.linalg.slogdet(A)[1]
        np.testing.assert_allclose(v, want)
        assert np.all(np.isfinite(g))

    def test_selinv_returns_nan(self, system, frontend, nan_on_failure):
        pytensor, pt, cjpt = frontend
        Ai, Aj, _, _, singular, _ = system
        upper = Ai <= Aj
        ax = pt.dvector("ax")
        f = pytensor.function([ax], cjpt.selinv(Ai[upper], Aj[upper], ax, N))
        assert np.all(np.isnan(f(singular[upper])))

    def test_default_raises(self, system, frontend):
        pytensor, pt, cjpt = frontend
        Ai, Aj, _, _, singular, _ = system
        upper = Ai <= Aj
        ax = pt.dvector("ax")
        f = pytensor.function([ax], cjpt.logdet(Ai[upper], Aj[upper], ax, N))
        with pytest.raises(Exception, match="positive definite"):
            f(singular[upper])
