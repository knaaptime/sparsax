"""The sign of a determinant, a condition estimate, and choosing an LU backend."""

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax

SLOGDETS = {"klu": sparsax.lu_slogdet, "umfpack": sparsax.umf_slogdet}
LU_RCONDS = {"klu": sparsax.lu_rcond, "umfpack": sparsax.umf_rcond}
RCONDS = {"cholmod": sparsax.rcond, **LU_RCONDS}


def coo(A):
    Ai, Aj = np.nonzero(A)
    return Ai.astype(np.int32), Aj.astype(np.int32), A[Ai, Aj]


def ffi_targets(f, *args):
    return set(re.findall(r"sparsax_[a-z_0-9]+", str(jax.make_jaxpr(f)(*args))))


class TestSlogdet:
    @pytest.mark.parametrize("backend", SLOGDETS)
    @pytest.mark.parametrize("seed", range(12))
    def test_matches_numpy_on_either_sign(self, backend, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 40))
        A = sp.random(n, n, density=0.3, random_state=seed).toarray()
        A += np.diag(rng.choice([-1.0, 1.0], size=n) * (A.sum(axis=1) + 1.0))
        sign, logabs = SLOGDETS[backend](*coo(A), n)
        want_sign, want_log = np.linalg.slogdet(A)
        assert float(sign) == want_sign
        np.testing.assert_allclose(logabs, want_log)

    @pytest.mark.parametrize("backend", SLOGDETS)
    def test_swapping_two_rows_flips_the_sign(self, backend):
        n = 12
        A = 3.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        B = A.copy()
        B[[2, 7]] = B[[7, 2]]
        sign_a, log_a = SLOGDETS[backend](*coo(A), n)
        sign_b, log_b = SLOGDETS[backend](*coo(B), n)
        assert float(sign_a) == 1.0 and float(sign_b) == -1.0
        np.testing.assert_allclose(log_a, log_b)

    @pytest.mark.parametrize("backend", SLOGDETS)
    def test_past_a_spatial_models_stable_range_the_sign_turns_negative(self, backend):
        """det(I - rho W) changes sign at rho = 1 for a row-standardized W."""
        side = 6
        path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
        eye = sp.identity(side)
        W = sp.kron(path, eye) + sp.kron(eye, path)
        W = (sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W).toarray()
        n = side * side
        for rho in (0.5, 1.02):
            M = np.eye(n) - rho * W
            sign, _ = SLOGDETS[backend](*coo(M), n)
            assert float(sign) == np.linalg.slogdet(M)[0]
        assert np.linalg.slogdet(np.eye(n) - 1.02 * W)[0] == -1.0

    @pytest.mark.parametrize("backend", SLOGDETS)
    def test_agrees_with_the_logdet_function(self, backend):
        n = 12
        A = 3.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        logdet = {"klu": sparsax.lu_logdet, "umfpack": sparsax.umf_logdet}[backend]
        _, logabs = SLOGDETS[backend](*coo(A), n)
        assert float(logabs) == float(logdet(*coo(A), n))

    @pytest.mark.parametrize("backend", SLOGDETS)
    def test_vmap_is_one_batched_call_and_right_for_each_element(self, backend):
        n = 12
        A = 3.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        Ai, Aj, Ax = coo(A)
        scales = jnp.array([1.0, -1.0, 2.0, -3.0])
        fn = SLOGDETS[backend]
        target = {
            "klu": "sparsax_lu_slogdet_batched_f64",
            "umfpack": "sparsax_umf_slogdet_batched_f64",
        }
        assert target[backend] in ffi_targets(
            lambda s: jax.vmap(lambda v: fn(Ai, Aj, v * Ax, n))(s), scales
        )
        signs, logs = jax.jit(jax.vmap(lambda v: fn(Ai, Aj, v * Ax, n)))(scales)
        for i, s in enumerate(np.asarray(scales)):
            want_sign, want_log = np.linalg.slogdet(s * A)
            assert float(signs[i]) == want_sign
            np.testing.assert_allclose(logs[i], want_log)

    @pytest.mark.parametrize("backend", SLOGDETS)
    def test_singular_raises_and_then_returns_nan_under_the_option(self, backend):
        n = 6
        A = 3.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        Ai, Aj, Ax = coo(A)
        Ax = Ax.copy()
        Ax[Aj == 0] = 0.0
        with pytest.raises(Exception, match="singular"):
            SLOGDETS[backend](Ai, Aj, Ax, n)
        sparsax.set_options(on_failure="nan")
        try:
            sign, logabs = SLOGDETS[backend](Ai, Aj, Ax, n)
        finally:
            sparsax.set_options(on_failure="raise")
        assert np.isnan(sign) and np.isnan(logabs)


class TestRcond:
    @pytest.mark.parametrize("backend", RCONDS)
    def test_a_near_singular_matrix_scores_near_zero(self, backend):
        fn = RCONDS[backend]
        good = np.array([[2.0, 0.5], [0.5, 2.0]])
        bad = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-9]])
        assert 0.1 < float(fn(*coo(good), 2)) <= 1.0
        assert float(fn(*coo(bad), 2)) < 1e-6

    @pytest.mark.parametrize("backend", RCONDS)
    def test_it_lies_between_zero_and_one_and_falls_as_rho_nears_one(self, backend):
        side = 8
        path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
        eye = sp.identity(side)
        S = (sp.kron(path, eye) + sp.kron(eye, path)).toarray()
        n = side * side
        lam = np.linalg.eigvalsh(S).max()
        values = []
        for rho in (0.1, 0.9, 0.9999):
            A = np.eye(n) - (rho / lam) * S
            values.append(float(RCONDS[backend](*coo(A), n)))
        assert all(0.0 < v <= 1.0 for v in values)
        assert values[0] > values[1] > values[2]

    def test_cholmod_uses_the_factors_diagonal(self):
        # L = diag(1, sqrt(10)) gives (1 / sqrt(10))^2.
        A = np.diag([1.0, 10.0])
        np.testing.assert_allclose(sparsax.rcond(*coo(A), 2), 0.1)

    @pytest.mark.parametrize("form", ["simplicial", "supernodal"])
    def test_cholmod_factor_forms_agree(self, form):
        A = 3.0 * np.eye(30) - np.eye(30, k=1) - np.eye(30, k=-1)
        base = float(sparsax.rcond(*coo(A), 30))
        sparsax.set_options(supernodal=form)
        try:
            np.testing.assert_allclose(sparsax.rcond(*coo(A), 30), base)
        finally:
            sparsax.set_options(supernodal="auto")

    @pytest.mark.parametrize(
        "backend, target",
        [
            ("cholmod", "sparsax_rcond_batched_f64"),
            ("klu", "sparsax_lu_rcond_batched_f64"),
            ("umfpack", "sparsax_umf_rcond_batched_f64"),
        ],
    )
    def test_vmap_is_one_batched_call(self, backend, target):
        A = 3.0 * np.eye(12) - np.eye(12, k=1) - np.eye(12, k=-1)
        Ai, Aj, Ax = coo(A)
        fn = RCONDS[backend]
        scales = jnp.array([1.0, 2.0, 5.0])
        assert target in ffi_targets(
            lambda s: jax.vmap(lambda v: fn(Ai, Aj, v * Ax, 12))(s), scales
        )
        got = jax.vmap(lambda v: fn(Ai, Aj, v * Ax, 12))(scales)
        # A uniform scale leaves a pivot ratio alone.
        np.testing.assert_allclose(
            got, np.full(3, float(fn(Ai, Aj, Ax, 12))), rtol=1e-10
        )

    @pytest.mark.parametrize("backend", RCONDS)
    def test_failure_follows_on_failure(self, backend):
        A = 3.0 * np.eye(6) - np.eye(6, k=1) - np.eye(6, k=-1)
        Ai, Aj, Ax = coo(A)
        Ax = Ax.copy()
        Ax[Aj == 0] = 0.0
        with pytest.raises(Exception, match="singular|positive definite"):
            RCONDS[backend](Ai, Aj, Ax, 6)
        sparsax.set_options(on_failure="nan")
        try:
            assert np.isnan(RCONDS[backend](Ai, Aj, Ax, 6))
        finally:
            sparsax.set_options(on_failure="raise")


def sparse_pattern(side=20):
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    W = (sp.kron(path, eye) + sp.kron(eye, path) + sp.identity(side**2)).tocoo()
    return W.row.astype(np.int32), W.col.astype(np.int32), side**2


def dense_pattern(n=300):
    Ai, Aj = np.nonzero(np.ones((n, n)))
    return Ai.astype(np.int32), Aj.astype(np.int32), n


class TestLuBackend:
    def test_a_sparse_graph_gets_klu(self):
        assert sparsax.lu_backend(*sparse_pattern()).name == "klu"

    def test_a_dense_graph_gets_umfpack(self):
        assert sparsax.lu_backend(*dense_pattern()).name == "umfpack"

    def test_work_is_flops_per_factor_entry(self):
        info = sparsax._cpp.lu_work(*sparse_pattern())
        assert info["flops"] > 0 and info["lnz"] > 0 and info["unz"] > 0

    def test_the_choice_ignores_values_order_and_duplicates(self):
        """The pattern alone decides, so a fixed seed reproduces the same draws."""
        Ai, Aj, n = sparse_pattern()
        base = sparsax._cpp.lu_work(Ai, Aj, n)
        order = np.random.default_rng(0).permutation(Ai.size)
        assert sparsax._cpp.lu_work(Ai[order], Aj[order], n) == base
        doubled = (np.tile(Ai, 2), np.tile(Aj, 2), n)
        assert sparsax._cpp.lu_work(*doubled) == base

    def test_a_pattern_with_no_diagonal_still_analyzes(self):
        n = 40
        Ai = np.arange(n - 1, dtype=np.int32)
        Aj = Ai + 1
        assert sparsax._cpp.lu_work(Ai, Aj, n)["lnz"] >= n

    def test_rejects_an_out_of_range_index(self):
        with pytest.raises(Exception, match="out of range"):
            sparsax.lu_backend(
                np.array([0, 99], np.int32), np.array([0, 1], np.int32), 4
            )

    @pytest.mark.parametrize("name", ["klu", "umfpack"])
    def test_a_named_backend_is_honoured(self, name):
        assert sparsax.lu_backend(*sparse_pattern(), backend=name).name == name
        assert sparsax.lu_backend(*dense_pattern(50), backend=name).name == name

    def test_rejects_unknown_arguments(self):
        with pytest.raises(ValueError, match="backend"):
            sparsax.lu_backend(*sparse_pattern(), backend="superlu")
        with pytest.raises(ValueError, match="workload"):
            sparsax.lu_backend(*sparse_pattern(), workload="invert")

    def test_a_logdet_workload_turns_to_umfpack_sooner(self, monkeypatch):
        """UMFPACK reads a determinant where KLU sums logarithms, so its lead starts earlier."""
        assert sparsax._LU_UMFPACK_ABOVE["logdet"] < sparsax._LU_UMFPACK_ABOVE["solve"]
        args = sparse_pattern()
        work = sparsax._cpp.lu_work(*args)
        per_entry = work["flops"] / (work["lnz"] + work["unz"])
        monkeypatch.setitem(sparsax._LU_UMFPACK_ABOVE, "logdet", per_entry - 1.0)
        monkeypatch.setitem(sparsax._LU_UMFPACK_ABOVE, "solve", per_entry + 1.0)
        assert sparsax.lu_backend(*args, workload="logdet").name == "umfpack"
        assert sparsax.lu_backend(*args, workload="solve").name == "klu"

    @pytest.mark.parametrize("name", ["klu", "umfpack"])
    def test_the_functions_it_returns_work(self, name):
        n = 12
        A = 3.0 * np.eye(n) - np.eye(n, k=1) + 0.5 * np.eye(n, k=-1)
        Ai, Aj, Ax = coo(A)
        lu = sparsax.lu_backend(Ai, Aj, n, backend=name)
        b = np.arange(1.0, n + 1.0)
        want = np.linalg.solve(A, b)
        np.testing.assert_allclose(lu.solve(Ai, Aj, Ax, b), want)
        token = lu.factor(Ai, Aj, Ax, n)
        np.testing.assert_allclose(lu.solve_factor(token, b), want)
        ld = np.linalg.slogdet(A)[1]
        np.testing.assert_allclose(lu.logdet(Ai, Aj, Ax, n), ld)
        np.testing.assert_allclose(lu.logdet_factor(token), ld)
        np.testing.assert_allclose(lu.slogdet(Ai, Aj, Ax, n)[1], ld)
        assert 0.0 < float(lu.rcond(Ai, Aj, Ax, n)) <= 1.0

    def test_a_token_from_one_backend_stays_with_it(self):
        n = 12
        A = 3.0 * np.eye(n) - np.eye(n, k=1)
        Ai, Aj, Ax = coo(A)
        klu = sparsax.lu_backend(Ai, Aj, n, backend="klu")
        umf = sparsax.lu_backend(Ai, Aj, n, backend="umfpack")
        with pytest.raises(Exception, match="not interchangeable"):
            umf.solve_factor(klu.factor(Ai, Aj, Ax, n), np.ones(n))
