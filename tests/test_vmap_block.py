"""vmap over b against one matrix is one block solve, not a batch of matrices.

With the values fixed, every batch element shares a factor, so the batch belongs
in the columns of a single right-hand side: the handler hashes Ax once and solves
with the threaded, multi-column kernels.  The other shapes stay as they were (one
factor per batch element) and compose with this one at any depth.
"""

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import sparsax

MATRIX_SOLVES = {
    "cholmod": sparsax.solve,
    "klu": sparsax.lu_solve,
    "umfpack": sparsax.umf_solve,
}
TOKEN_SOLVES = {
    "cholmod": (sparsax.factor, sparsax.solve_factor),
    "klu": (sparsax.lu_factor, sparsax.lu_solve_factor),
    "umfpack": (sparsax.umf_factor, sparsax.umf_solve_factor),
}
N = 20


@pytest.fixture
def system():
    A = 3.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    Ai, Aj = np.nonzero(A)
    rng = np.random.default_rng(0)
    return Ai.astype(np.int32), Aj.astype(np.int32), A[Ai, Aj], A, rng


def ffi_targets(f, *args):
    return set(re.findall(r"sparsax_[a-z_0-9]+", str(jax.make_jaxpr(f)(*args))))


class TestRoute:
    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_vmap_over_b_is_one_unbatched_call(self, system, backend):
        Ai, Aj, Ax, _, rng = system
        solve = MATRIX_SOLVES[backend]
        B = jnp.asarray(rng.normal(size=(8, N)))
        got = ffi_targets(lambda r: jax.vmap(lambda v: solve(Ai, Aj, Ax, v))(r), B)
        assert got and not any("batched" in t for t in got)

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_vmap_over_values_stays_batched(self, system, backend):
        Ai, Aj, Ax, _, rng = system
        solve = MATRIX_SOLVES[backend]
        b = jnp.asarray(rng.normal(size=N))
        axs = jnp.stack([Ax, 2.0 * Ax])
        got = ffi_targets(lambda a: jax.vmap(lambda v: solve(Ai, Aj, v, b))(a), axs)
        assert any("batched" in t for t in got)

    @pytest.mark.parametrize("backend", TOKEN_SOLVES)
    def test_token_vmap_over_b_is_one_unbatched_call(self, system, backend):
        Ai, Aj, Ax, _, rng = system
        factor, solve = TOKEN_SOLVES[backend]
        token = factor(Ai, Aj, Ax, N)
        B = jnp.asarray(rng.normal(size=(8, N)))
        got = ffi_targets(lambda r: jax.vmap(lambda v: solve(token, v))(r), B)
        assert got and not any("batched" in t for t in got)

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_one_factorization(self, system, backend):
        Ai, Aj, Ax, _, rng = system
        solve = MATRIX_SOLVES[backend]
        B = jnp.asarray(rng.normal(size=(16, N)))
        sparsax.clear_cache()
        before = sparsax.factorization_count()
        jax.block_until_ready(jax.vmap(lambda v: solve(Ai, Aj, Ax, v))(B))
        assert sparsax.factorization_count() - before == 1


class TestMatrixSolveValues:
    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    @pytest.mark.parametrize("batch", [1, 3, 64, 130])
    def test_vectors(self, system, backend, batch):
        Ai, Aj, Ax, A, rng = system
        solve = MATRIX_SOLVES[backend]
        B = rng.normal(size=(batch, N))
        got = jax.vmap(lambda v: solve(Ai, Aj, Ax, v))(jnp.asarray(B))
        np.testing.assert_allclose(got, np.linalg.solve(A, B.T).T, rtol=1e-11)

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_matrices_keep_batch_and_column_order(self, system, backend):
        Ai, Aj, Ax, A, rng = system
        solve = MATRIX_SOLVES[backend]
        B = rng.normal(size=(5, N, 3))
        got = jax.vmap(lambda v: solve(Ai, Aj, Ax, v))(jnp.asarray(B))
        want = np.stack([np.linalg.solve(A, B[s]) for s in range(5)])
        np.testing.assert_allclose(got, want, rtol=1e-11)

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_empty_batch(self, system, backend):
        Ai, Aj, Ax, _, _ = system
        solve = MATRIX_SOLVES[backend]
        got = jax.vmap(lambda v: solve(Ai, Aj, Ax, v))(jnp.zeros((0, N)))
        assert got.shape == (0, N)

    @pytest.mark.parametrize("mode", [sparsax.MODE_L, sparsax.MODE_LT, sparsax.MODE_P])
    def test_factor_part_modes_match_single_solves(self, system, mode):
        Ai, Aj, Ax, _, rng = system
        B = rng.normal(size=(4, N))
        got = jax.vmap(lambda v: sparsax.solve(Ai, Aj, Ax, v, mode=mode))(jnp.asarray(B))
        want = np.stack([sparsax.solve(Ai, Aj, Ax, v, mode=mode) for v in B])
        np.testing.assert_allclose(got, want, rtol=1e-12)

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_jacobian_in_b_is_the_inverse(self, system, backend):
        Ai, Aj, Ax, A, rng = system
        solve = MATRIX_SOLVES[backend]
        b = jnp.asarray(rng.normal(size=N))
        inv = np.linalg.inv(A)
        np.testing.assert_allclose(
            jax.jacrev(lambda v: solve(Ai, Aj, Ax, v))(b), inv, rtol=1e-10, atol=1e-13
        )
        np.testing.assert_allclose(
            jax.jacfwd(lambda v: solve(Ai, Aj, Ax, v))(b), inv, rtol=1e-10, atol=1e-13
        )

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_values_inside_out(self, system, backend):
        """Chains over values, right-hand sides inside each chain."""
        Ai, Aj, Ax, A, rng = system
        solve = MATRIX_SOLVES[backend]
        scales = np.array([1.0, 2.0, 4.0])
        B = rng.normal(size=(5, N))
        got = jax.vmap(
            lambda s: jax.vmap(lambda v: solve(Ai, Aj, s * Ax, v))(jnp.asarray(B))
        )(jnp.asarray(scales))
        want = np.stack([np.linalg.solve(s * A, B.T).T for s in scales])
        np.testing.assert_allclose(got, want, rtol=1e-11)

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_right_hand_sides_outside_values(self, system, backend):
        Ai, Aj, Ax, A, rng = system
        solve = MATRIX_SOLVES[backend]
        scales = np.array([1.0, 2.0, 4.0])
        B = rng.normal(size=(5, N))
        got = jax.vmap(
            lambda v: jax.vmap(lambda s: solve(Ai, Aj, s * Ax, v))(jnp.asarray(scales))
        )(jnp.asarray(B))
        want = np.stack(
            [np.stack([np.linalg.solve(s * A, v) for s in scales]) for v in B]
        )
        np.testing.assert_allclose(got, want, rtol=1e-11)

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_three_levels(self, system, backend):
        Ai, Aj, Ax, A, rng = system
        solve = MATRIX_SOLVES[backend]
        scales = np.array([1.0, 2.0])
        B = rng.normal(size=(2, 3, N))
        got = jax.vmap(
            lambda r: jax.vmap(
                lambda s: jax.vmap(lambda v: solve(Ai, Aj, s * Ax, v))(r)
            )(jnp.asarray(scales))
        )(jnp.asarray(B))
        want = np.stack(
            [
                np.stack([np.linalg.solve(s * A, B[o].T).T for s in scales])
                for o in range(2)
            ]
        )
        np.testing.assert_allclose(got, want, rtol=1e-11)

    @pytest.mark.parametrize("backend", MATRIX_SOLVES)
    def test_batched_pattern_with_fixed_values(self, system, backend):
        """lax.cond under vmap lifts the closed-over pattern into the batch."""
        Ai, Aj, Ax, A, rng = system
        solve = MATRIX_SOLVES[backend]
        b = jnp.asarray(rng.normal(size=N))

        def one(flag):
            return jax.lax.cond(
                flag, lambda _: solve(Ai, Aj, Ax, b), lambda _: jnp.zeros(N), None
            )

        got = jax.vmap(one)(jnp.array([True, False]))
        np.testing.assert_allclose(got[0], np.linalg.solve(A, np.asarray(b)), rtol=1e-11)
        np.testing.assert_array_equal(got[1], np.zeros(N))


class TestTokenSolveValues:
    @pytest.mark.parametrize("backend", TOKEN_SOLVES)
    @pytest.mark.parametrize("shape", [(1, N), (7, N), (130, N), (5, N, 3)])
    def test_one_token_many_b(self, system, backend, shape):
        Ai, Aj, Ax, A, rng = system
        factor, solve = TOKEN_SOLVES[backend]
        token = factor(Ai, Aj, Ax, N)
        B = rng.normal(size=shape)
        got = jax.vmap(lambda v: solve(token, v))(jnp.asarray(B))
        want = np.stack([np.linalg.solve(A, B[s]) for s in range(shape[0])])
        np.testing.assert_allclose(got, want, rtol=1e-11)

    @pytest.mark.parametrize("backend", TOKEN_SOLVES)
    def test_tokens_vmapped_with_one_fixed_vector(self, system, backend):
        Ai, Aj, Ax, A, rng = system
        factor, solve = TOKEN_SOLVES[backend]
        scales = jnp.array([1.0, 2.0])
        tokens = jax.vmap(lambda s: factor(Ai, Aj, s * Ax, N))(scales)
        b = rng.normal(size=N)
        got = jax.vmap(solve, in_axes=(0, None))(tokens, jnp.asarray(b))
        want = np.stack([np.linalg.solve(s * A, b) for s in (1.0, 2.0)])
        np.testing.assert_allclose(got, want, rtol=1e-11)

    @pytest.mark.parametrize("backend", TOKEN_SOLVES)
    def test_chains_outside_right_hand_sides(self, system, backend):
        """The documented chains-by-Krylov shape: tokens outside, b inside."""
        Ai, Aj, Ax, A, rng = system
        factor, solve = TOKEN_SOLVES[backend]
        scales = np.array([1.0, 2.0, 4.0])
        tokens = jax.vmap(lambda s: factor(Ai, Aj, s * Ax, N))(jnp.asarray(scales))
        B = rng.normal(size=(5, N))
        got = jax.vmap(
            lambda t: jax.vmap(lambda v: solve(t, v))(jnp.asarray(B))
        )(tokens)
        want = np.stack([np.linalg.solve(s * A, B.T).T for s in scales])
        np.testing.assert_allclose(got, want, rtol=1e-11)

    @pytest.mark.parametrize("backend", TOKEN_SOLVES)
    def test_right_hand_sides_outside_chains(self, system, backend):
        Ai, Aj, Ax, A, rng = system
        factor, solve = TOKEN_SOLVES[backend]
        scales = np.array([1.0, 2.0, 4.0])
        tokens = jax.vmap(lambda s: factor(Ai, Aj, s * Ax, N))(jnp.asarray(scales))
        B = rng.normal(size=(5, N))
        got = jax.vmap(lambda v: jax.vmap(lambda t: solve(t, v))(tokens))(jnp.asarray(B))
        want = np.stack(
            [np.stack([np.linalg.solve(s * A, v) for s in scales]) for v in B]
        )
        np.testing.assert_allclose(got, want, rtol=1e-11)

    @pytest.mark.parametrize("backend", TOKEN_SOLVES)
    def test_tokens_and_b_both_outer_and_inner(self, system, backend):
        Ai, Aj, Ax, A, rng = system
        factor, solve = TOKEN_SOLVES[backend]
        scales = np.array([[1.0, 2.0], [4.0, 8.0], [3.0, 5.0]])
        tokens = jax.vmap(jax.vmap(lambda s: factor(Ai, Aj, s * Ax, N)))(
            jnp.asarray(scales)
        )
        B = rng.normal(size=(3, 2, N))
        got = jax.vmap(jax.vmap(solve))(tokens, jnp.asarray(B))
        want = np.stack(
            [
                np.stack(
                    [np.linalg.solve(scales[o, i] * A, B[o, i]) for i in range(2)]
                )
                for o in range(3)
            ]
        )
        np.testing.assert_allclose(got, want, rtol=1e-11)
