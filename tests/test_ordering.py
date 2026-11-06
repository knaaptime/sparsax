"""set_options(ordering=...): the fill-reducing ordering behind CHOLMOD's analysis.

CHOLMOD's automatic choice tries AMD and turns to METIS only when AMD's fill looks
poor, which it rarely does on spatial graphs, though nested dissection can still
beat it there.  The ordering changes the cost of everything downstream and none of
the answers, so these tests pin both halves: the analysis used what was asked for,
and every result still matches dense algebra.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax

ORDERINGS = ["auto", "amd", "metis", "nesdis", "best"]


@pytest.fixture(params=ORDERINGS)
def ordering(request):
    sparsax.set_options(ordering=request.param)
    yield request.param
    sparsax.set_options(ordering="auto")


@pytest.fixture
def restore_ordering():
    yield
    sparsax.set_options(ordering="auto")


def grid(side, dim=2):
    path = sp.diags([-np.ones(side - 1), -np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    if dim == 2:
        S = sp.kron(path, eye) + sp.kron(eye, path)
    else:
        S = (
            sp.kron(sp.kron(path, eye), eye)
            + sp.kron(sp.kron(eye, path), eye)
            + sp.kron(sp.kron(eye, eye), path)
        )
    A = (S + (2.0 * dim + 1.0) * sp.identity(side**dim)).tocoo()
    U = sp.triu(A).tocoo()
    return U.row.astype(np.int32), U.col.astype(np.int32), U.data, A.toarray()


class TestAnswersDoNotChange:
    def test_solve_logdet_and_selinv(self, ordering):
        Ai, Aj, Ax, A = grid(10)
        n = A.shape[0]
        b = np.arange(1.0, n + 1.0)
        np.testing.assert_allclose(sparsax.solve(Ai, Aj, Ax, b), np.linalg.solve(A, b))
        np.testing.assert_allclose(
            sparsax.logdet(Ai, Aj, Ax, n), np.linalg.slogdet(A)[1]
        )
        np.testing.assert_allclose(
            sparsax.selinv(Ai, Aj, Ax, n), np.linalg.inv(A)[Ai, Aj], rtol=1e-10
        )

    @pytest.mark.parametrize("form", ["simplicial", "supernodal"])
    def test_both_factor_forms(self, ordering, form):
        Ai, Aj, Ax, A = grid(6, dim=3)
        n = A.shape[0]
        sparsax.set_options(supernodal=form)
        try:
            np.testing.assert_allclose(
                sparsax.selinv(Ai, Aj, Ax, n), np.linalg.inv(A)[Ai, Aj], rtol=1e-10
            )
        finally:
            sparsax.set_options(supernodal="auto")

    def test_factor_part_solves_stay_consistent(self, ordering):
        """L, L' and P work in the permuted space, so the ordering reaches them."""
        Ai, Aj, Ax, A = grid(8)
        n = A.shape[0]
        z = np.random.default_rng(0).normal(size=n)
        (w,), ld = sparsax.factor_solve(
            Ai, Aj, Ax, [(z, (sparsax.MODE_LT, sparsax.MODE_PT))], want_logdet=True
        )
        np.testing.assert_allclose(np.asarray(w) @ A @ np.asarray(w), z @ z, rtol=1e-9)
        np.testing.assert_allclose(ld, np.linalg.slogdet(A)[1])

    def test_rank_update(self, ordering):
        Ai, Aj, Ax, A = grid(8)
        n = A.shape[0]
        C = 0.2 * np.random.default_rng(1).normal(size=(n, 2))
        b = np.ones(n)
        x, ld = sparsax.update_solve(Ai, Aj, Ax, C, b, return_logdet=True)
        np.testing.assert_allclose(x, np.linalg.solve(A + C @ C.T, b), rtol=1e-9)
        np.testing.assert_allclose(ld, np.linalg.slogdet(A + C @ C.T)[1])

    def test_argmax_logprob_probabilities_sum_to_one(self, ordering):
        Ai, Aj, Ax, A = grid(3)
        n = A.shape[0]
        mean = np.random.default_rng(0).normal(scale=0.5, size=n)
        lp = sparsax.argmax_logprob(
            Ai, Aj, Ax, np.tile(mean, (n, 1)), np.arange(n), n_draws=2**12
        )
        np.testing.assert_allclose(np.exp(np.asarray(lp)).sum(), 1.0, atol=0.01)


class TestTheAnalysisUsesIt:
    @pytest.mark.parametrize("name", ["amd", "metis", "nesdis"])
    def test_named_ordering_is_the_one_used(self, name, restore_ordering):
        Ai, Aj, Ax, A = grid(10)
        sparsax.set_options(ordering=name)
        assert sparsax._cpp.analysis_info(Ai, Aj, A.shape[0])["ordering"] == name

    def test_best_has_the_least_fill_of_the_three(self, restore_ordering):
        Ai, Aj, Ax, A = grid(12, dim=3)
        n = A.shape[0]
        fill = {}
        for name in ("amd", "metis", "nesdis", "best"):
            sparsax.set_options(ordering=name)
            fill[name] = sparsax._cpp.analysis_info(Ai, Aj, n)["lnz"]
        assert fill["best"] == min(fill["amd"], fill["metis"], fill["nesdis"])

    def test_nested_dissection_cuts_fill_on_a_3d_grid(self, restore_ordering):
        Ai, Aj, Ax, A = grid(14, dim=3)
        n = A.shape[0]
        sparsax.set_options(ordering="amd")
        amd = sparsax._cpp.analysis_info(Ai, Aj, n)["lnz"]
        sparsax.set_options(ordering="nesdis")
        assert sparsax._cpp.analysis_info(Ai, Aj, n)["lnz"] < amd

    def test_other_threads_follow_the_option(self, restore_ordering):
        from concurrent.futures import ThreadPoolExecutor

        Ai, Aj, Ax, A = grid(10)
        sparsax.set_options(ordering="nesdis")
        with ThreadPoolExecutor(1) as pool:
            info = pool.submit(sparsax._cpp.analysis_info, Ai, Aj, A.shape[0]).result()
        assert info["ordering"] == "nesdis"


class TestOption:
    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="ordering"):
            sparsax.set_options(ordering="rcm")

    def test_changing_it_redoes_the_analysis(self, restore_ordering):
        Ai, Aj, Ax, A = grid(10)
        n = A.shape[0]
        sparsax.set_options(ordering="amd")
        jax.block_until_ready(sparsax.logdet(Ai, Aj, jnp.asarray(Ax), n))
        before = sparsax.factorization_count()
        sparsax.set_options(ordering="nesdis")
        jax.block_until_ready(sparsax.logdet(Ai, Aj, jnp.asarray(Ax), n))
        assert sparsax.factorization_count() == before + 1

    def test_leaving_it_out_changes_nothing(self, restore_ordering):
        Ai, Aj, Ax, A = grid(10)
        n = A.shape[0]
        sparsax.set_options(ordering="nesdis")
        sparsax.set_options(on_failure="nan")
        sparsax.set_options(on_failure="raise")
        sparsax.set_options(supernodal="auto")
        assert sparsax._cpp.analysis_info(Ai, Aj, n)["ordering"] == "nesdis"
