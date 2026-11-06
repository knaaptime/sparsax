"""A vmap batch of values runs on several threads, in one native call.

Each element of such a batch is independent work on its own factor, so the batch is
a natural unit of parallelism.  A fixture removes the work threshold so these small
systems, whose answers dense algebra can check, take the parallel path; the same
tests with the threshold left alone cover the serial one.
"""

import os
import re
import subprocess
import sys
import textwrap

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import sparsax

SOLVES = {
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
    "umfpack": (
        sparsax.umf_factor,
        sparsax.umf_solve_factor,
        sparsax.umf_logdet_factor,
    ),
}
N = 30
SCALES = np.linspace(1.0, 3.0, 16)


@pytest.fixture(params=["parallel", "default"])
def threshold(request):
    if request.param == "parallel":
        sparsax._cpp.set_batch_work_per_thread(0)
    yield request.param
    sparsax._cpp.set_batch_work_per_thread(-1)


@pytest.fixture
def system():
    A = 3.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    A[0, N // 2] = A[N // 2, 0] = -0.5
    Ai, Aj = np.nonzero(A)
    Ai, Aj = Ai.astype(np.int32), Aj.astype(np.int32)
    return Ai, Aj, A[Ai, Aj], A


def ffi_targets(f, *args):
    return set(re.findall(r"sparsax_[a-z_0-9]+", str(jax.make_jaxpr(f)(*args))))


class TestRoute:
    @pytest.mark.parametrize(
        "fn, target",
        [
            (sparsax.logdet, "sparsax_logdet_batched_f64"),
            (sparsax.selinv, "sparsax_selinv_batched_f64"),
            (sparsax.factor, "sparsax_factor_batched_f64"),
            (sparsax.lu_logdet, "sparsax_lu_logdet_batched_f64"),
            (sparsax.lu_factor, "sparsax_lu_factor_batched_f64"),
            (sparsax.umf_logdet, "sparsax_umf_logdet_batched_f64"),
            (sparsax.umf_factor, "sparsax_umf_factor_batched_f64"),
        ],
    )
    def test_vmap_over_values_is_one_batched_call(self, system, fn, target):
        Ai, Aj, Ax, _ = system
        axs = jnp.stack([Ax, 2.0 * Ax])
        assert target in ffi_targets(
            lambda a: jax.vmap(lambda v: fn(Ai, Aj, v, N))(a), axs
        )

    @pytest.mark.parametrize(
        "backend, target",
        [
            ("cholmod", "sparsax_logdet_factor_batched_f64"),
            ("klu", "sparsax_lu_logdet_factor_batched_f64"),
            ("umfpack", "sparsax_umf_logdet_factor_batched_f64"),
        ],
    )
    def test_vmap_over_tokens_logdet_is_one_batched_call(self, system, backend, target):
        Ai, Aj, Ax, _ = system
        factor, _, logdet = TOKENS[backend]
        tokens = jax.vmap(lambda s: factor(Ai, Aj, s * Ax, N))(jnp.array([1.0, 2.0]))
        assert target in ffi_targets(jax.vmap(logdet), tokens)


@pytest.mark.usefixtures("threshold")
class TestAnswers:
    @pytest.mark.parametrize("backend", SOLVES)
    def test_solve(self, system, backend):
        Ai, Aj, Ax, A = system
        b = np.arange(1.0, N + 1.0)
        axs = jnp.asarray(np.outer(SCALES, Ax))
        x = jax.vmap(lambda a: SOLVES[backend](Ai, Aj, a, b))(axs)
        want = np.stack([np.linalg.solve(s * A, b) for s in SCALES])
        np.testing.assert_allclose(x, want, rtol=1e-10)

    @pytest.mark.parametrize("backend", SOLVES)
    def test_solve_with_many_columns_per_element(self, system, backend):
        """Batch and panels together: a worker gets its share of the threads."""
        Ai, Aj, Ax, A = system
        B = np.random.default_rng(0).normal(size=(N, 130))
        axs = jnp.asarray(np.outer(SCALES[:4], Ax))
        x = jax.vmap(lambda a: SOLVES[backend](Ai, Aj, a, B))(axs)
        want = np.stack([np.linalg.solve(s * A, B) for s in SCALES[:4]])
        np.testing.assert_allclose(x, want, rtol=1e-9)

    @pytest.mark.parametrize("backend", LOGDETS)
    def test_logdet(self, system, backend):
        Ai, Aj, Ax, A = system
        axs = jnp.asarray(np.outer(SCALES, Ax))
        ld = jax.vmap(lambda a: LOGDETS[backend](Ai, Aj, a, N))(axs)
        want = [np.linalg.slogdet(s * A)[1] for s in SCALES]
        np.testing.assert_allclose(ld, want, rtol=1e-11)

    def test_selinv(self, system):
        Ai, Aj, Ax, A = system
        axs = jnp.asarray(np.outer(SCALES, Ax))
        z = jax.jit(jax.vmap(lambda a: sparsax.selinv(Ai, Aj, a, N)))(axs)
        want = np.stack([np.linalg.inv(s * A)[Ai, Aj] for s in SCALES])
        np.testing.assert_allclose(z, want, rtol=1e-9, atol=1e-13)

    @pytest.mark.parametrize("form", ["simplicial", "supernodal"])
    def test_cholmod_factor_forms(self, system, form):
        Ai, Aj, Ax, A = system
        axs = jnp.asarray(np.outer(SCALES, Ax))
        sparsax.set_options(supernodal=form)
        try:
            z = jax.vmap(lambda a: sparsax.selinv(Ai, Aj, a, N))(axs)
        finally:
            sparsax.set_options(supernodal="auto")
        want = np.stack([np.linalg.inv(s * A)[Ai, Aj] for s in SCALES])
        np.testing.assert_allclose(z, want, rtol=1e-9, atol=1e-13)

    @pytest.mark.parametrize("backend", TOKENS)
    def test_tokens(self, system, backend):
        Ai, Aj, Ax, A = system
        factor, solve, logdet = TOKENS[backend]
        b = np.arange(1.0, N + 1.0)
        tokens = jax.vmap(lambda s: factor(Ai, Aj, s * Ax, N))(jnp.asarray(SCALES))
        assert tokens.shape == (16, 2)
        assert len(set(np.asarray(tokens)[:, 0].tolist())) == 16
        x = jax.vmap(solve, in_axes=(0, None))(tokens, jnp.asarray(b))
        want = np.stack([np.linalg.solve(s * A, b) for s in SCALES])
        np.testing.assert_allclose(x, want, rtol=1e-10)
        ld = jax.vmap(logdet)(tokens)
        np.testing.assert_allclose(ld, [np.linalg.slogdet(s * A)[1] for s in SCALES])

    def test_factor_solve_and_sample_gaussian(self, system):
        Ai, Aj, Ax, A = system
        b = np.arange(1.0, N + 1.0)
        z = np.random.default_rng(1).normal(size=N)

        def one(s):
            return sparsax.sample_gaussian(Ai, Aj, s * Ax, b, z, want_logdet=True)

        eta, mean, ld = jax.vmap(one)(jnp.asarray(SCALES))
        for i, s in enumerate(SCALES):
            np.testing.assert_allclose(mean[i], np.linalg.solve(s * A, b), rtol=1e-10)
            np.testing.assert_allclose(ld[i], np.linalg.slogdet(s * A)[1])
            w = np.asarray(eta[i] - mean[i])
            np.testing.assert_allclose(w @ (s * A) @ w, z @ z, rtol=1e-9)

    def test_value_and_grad_of_a_log_density(self, system):
        Ai, Aj, Ax, A = system
        b = jnp.arange(1.0, N + 1.0)

        def logp(s):
            ax = s * Ax
            return -0.5 * b @ sparsax.solve(Ai, Aj, ax, b) + 0.5 * sparsax.logdet(
                Ai, Aj, ax, N
            )

        v, g = jax.jit(jax.vmap(jax.value_and_grad(logp)))(jnp.asarray(SCALES))
        bn = np.asarray(b)
        for i, s in enumerate(SCALES):
            x = np.linalg.solve(A, bn)
            np.testing.assert_allclose(
                v[i], -0.5 * bn @ x / s + 0.5 * np.linalg.slogdet(s * A)[1]
            )
            np.testing.assert_allclose(g[i], 0.5 * bn @ x / s**2 + 0.5 * N / s)

    def test_nested_vmap_of_logdet_and_selinv(self, system):
        Ai, Aj, Ax, A = system
        scales = jnp.asarray(SCALES[:6].reshape(2, 3))
        ld = jax.vmap(jax.vmap(lambda s: sparsax.logdet(Ai, Aj, s * Ax, N)))(scales)
        z = jax.vmap(jax.vmap(lambda s: sparsax.selinv(Ai, Aj, s * Ax, N)))(scales)
        assert ld.shape == (2, 3) and z.shape == (2, 3, Ax.size)
        for o in range(2):
            for i in range(3):
                s = float(scales[o, i])
                np.testing.assert_allclose(ld[o, i], np.linalg.slogdet(s * A)[1])
                np.testing.assert_allclose(
                    z[o, i], np.linalg.inv(s * A)[Ai, Aj], rtol=1e-9, atol=1e-13
                )

    @pytest.mark.parametrize("fn", [sparsax.logdet, sparsax.selinv])
    def test_empty_batch(self, system, fn):
        Ai, Aj, Ax, _ = system
        got = jax.vmap(lambda a: fn(Ai, Aj, a, N))(jnp.zeros((0, Ax.size)))
        assert got.shape[0] == 0

    def test_identical_values_across_the_batch(self, system):
        """Chains started at one point: threads may race to the same factor."""
        Ai, Aj, Ax, A = system
        axs = jnp.tile(jnp.asarray(Ax), (16, 1))
        b = np.arange(1.0, N + 1.0)
        x = jax.vmap(lambda a: sparsax.lu_solve(Ai, Aj, a, b))(axs)
        np.testing.assert_allclose(x, np.tile(np.linalg.solve(A, b), (16, 1)))


@pytest.mark.usefixtures("threshold")
class TestFailures:
    @pytest.mark.parametrize("backend", SOLVES)
    def test_a_failing_element_raises_by_default(self, system, backend):
        Ai, Aj, Ax, _ = system
        bad = Ax.copy()
        bad[Aj == 0] = 0.0
        axs = jnp.stack([jnp.asarray(Ax), jnp.asarray(bad), 2.0 * jnp.asarray(Ax)])
        with pytest.raises(Exception, match="singular|positive definite"):
            jax.vmap(lambda a: SOLVES[backend](Ai, Aj, a, np.ones(N)))(axs)

    @pytest.mark.parametrize("backend", SOLVES)
    def test_only_the_failing_element_is_nan(self, system, backend):
        Ai, Aj, Ax, A = system
        bad = Ax.copy()
        bad[Aj == 0] = 0.0
        axs = np.tile(Ax, (16, 1))
        axs[5] = bad
        b = np.arange(1.0, N + 1.0)
        sparsax.set_options(on_failure="nan")
        try:
            x = jax.vmap(lambda a: SOLVES[backend](Ai, Aj, a, b))(jnp.asarray(axs))
        finally:
            sparsax.set_options(on_failure="raise")
        assert np.all(np.isnan(x[5]))
        good = np.delete(np.asarray(x), 5, axis=0)
        np.testing.assert_allclose(good, np.tile(np.linalg.solve(A, b), (15, 1)))

    @pytest.mark.parametrize("backend", TOKENS)
    def test_stale_token_in_a_batch_raises(self, system, backend):
        Ai, Aj, Ax, _ = system
        factor, solve, _ = TOKENS[backend]
        tokens = jax.vmap(lambda s: factor(Ai, Aj, s * Ax, N))(jnp.array([1.0, 2.0]))
        sparsax.clear_cache()
        with pytest.raises(Exception, match="stale factor token"):
            jax.vmap(solve)(tokens, jnp.ones((2, N)))


def test_answers_do_not_depend_on_the_thread_count():
    script = textwrap.dedent(
        """
        import sys
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import numpy as np
        import sparsax

        sparsax._cpp.set_batch_work_per_thread(0)
        n = 200
        A = 3.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        Ai, Aj = np.nonzero(A)
        Ai, Aj = Ai.astype(np.int32), Aj.astype(np.int32)
        axs = jnp.asarray(np.outer(np.linspace(1.0, 3.0, 32), A[Ai, Aj]))
        b = np.arange(1.0, n + 1.0)
        out = []
        for solve in (sparsax.solve, sparsax.lu_solve, sparsax.umf_solve):
            out.append(np.asarray(jax.vmap(lambda a: solve(Ai, Aj, a, b))(axs)))
        out.append(np.asarray(jax.vmap(lambda a: sparsax.selinv(Ai, Aj, a, n))(axs)))
        np.save(sys.argv[1], np.concatenate([o.ravel() for o in out]))
        """
    )

    def run(threads, path):
        env = {**os.environ, "SPARSAX_NUM_THREADS": threads}
        proc = subprocess.run(
            [sys.executable, "-c", script, path],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        return np.load(path)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        serial = run("1", os.path.join(d, "a.npy"))
        threaded = run("8", os.path.join(d, "b.npy"))
    np.testing.assert_allclose(threaded, serial, rtol=1e-12, atol=1e-13)


class TestConcurrentRequestsShareAFactorization:
    @pytest.fixture(autouse=True)
    def parallel(self):
        sparsax._cpp.set_batch_work_per_thread(0)
        yield
        sparsax._cpp.set_batch_work_per_thread(-1)

    @pytest.mark.parametrize("backend", SOLVES)
    def test_a_batch_of_one_matrix_factors_once(self, system, backend):
        Ai, Aj, Ax, _ = system
        solve = SOLVES[backend]
        axs = jnp.tile(jnp.asarray(Ax), (16, 1))
        sparsax.clear_cache()
        before = sparsax.factorization_count()
        jax.block_until_ready(jax.vmap(lambda a: solve(Ai, Aj, a, np.ones(N)))(axs))
        assert sparsax.factorization_count() - before == 1

    @pytest.mark.parametrize("backend", SOLVES)
    def test_repeated_values_factor_once_each(self, system, backend):
        Ai, Aj, Ax, A = system
        solve = SOLVES[backend]
        axs = np.outer(np.tile([1.0, 2.0, 3.0, 4.0], 4), Ax)
        b = np.arange(1.0, N + 1.0)
        sparsax.clear_cache()
        before = sparsax.factorization_count()
        x = jax.vmap(lambda a: solve(Ai, Aj, a, b))(jnp.asarray(axs))
        jax.block_until_ready(x)
        assert sparsax.factorization_count() - before == 4
        np.testing.assert_allclose(x[5], np.linalg.solve(2.0 * A, b))

    @pytest.mark.parametrize("backend", SOLVES)
    def test_threads_waiting_on_a_failure_each_fail(self, system, backend):
        """A failed factorization is not cached, so whoever waited retries, and fails."""
        Ai, Aj, Ax, _ = system
        bad = Ax.copy()
        bad[Aj == 0] = 0.0
        axs = jnp.tile(jnp.asarray(bad), (16, 1))
        sparsax.set_options(on_failure="nan")
        try:
            x = jax.vmap(lambda a: SOLVES[backend](Ai, Aj, a, np.ones(N)))(axs)
            assert np.all(np.isnan(x))
        finally:
            sparsax.set_options(on_failure="raise")
        with pytest.raises(Exception, match="singular|positive definite"):
            jax.vmap(lambda a: SOLVES[backend](Ai, Aj, a, np.ones(N)))(axs)

    def test_small_batches_stay_on_the_calling_thread_by_default(self, system):
        """Below the threshold a batch must run serially, or small problems get slower."""
        Ai, Aj, Ax, A = system
        sparsax._cpp.set_batch_work_per_thread(-1)
        b = np.arange(1.0, N + 1.0)
        axs = jnp.asarray(np.outer([1.0, 2.0], Ax))
        x = jax.vmap(lambda a: sparsax.solve(Ai, Aj, a, b))(axs)
        np.testing.assert_allclose(x[1], np.linalg.solve(2.0 * A, b))
