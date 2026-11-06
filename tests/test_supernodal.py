"""CHOLMOD with the factorization forced simplicial and forced supernodal.

CHOLMOD's automatic choice leaves small test matrices simplicial, so the
supernodal path needs its own tests: each numeric factorization copies the
symbolic factor, the log-determinant is read off supernodal columns, the selected
inverse works on the supernodes directly (the simplicial form uses its own
recurrence, so these tests pin the two to one answer), and rank updates convert
the factor to simplicial LDL'.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax


@pytest.fixture(params=["simplicial", "supernodal"])
def mode(request):
    sparsax.set_options(supernodal=request.param)
    yield request.param
    sparsax.set_options(supernodal="auto")


def spd_system(side=24):
    """P = s I - 0.5 S for grid adjacency S: positive definite for s > 2."""
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    S = (sp.kron(eye, path) + sp.kron(path, eye)).tocoo()
    n = side * side
    upper = S.row <= S.col
    Ai = np.concatenate([np.arange(n), S.row[upper]]).astype(np.int32)
    Aj = np.concatenate([np.arange(n), S.col[upper]]).astype(np.int32)
    s_upper = S.data[upper]
    Sd = S.toarray()

    def values(s):
        return np.concatenate([np.full(n, s), -0.5 * s_upper])

    def dense(s):
        return s * np.eye(n) - 0.5 * Sd

    return Ai, Aj, values, dense, n


def test_factor_solve_mean_draw_and_logdet(mode):
    Ai, Aj, values, dense, n = spd_system()
    P = dense(2.5)
    rng = np.random.default_rng(1)
    b, z = rng.normal(size=n), rng.normal(size=n)
    sparsax.clear_cache()
    c0 = sparsax.factorization_count()
    (mean, w), ld = sparsax.factor_solve(
        Ai,
        Aj,
        values(2.5),
        [(b, sparsax.MODE_A), (z, (sparsax.MODE_LT, sparsax.MODE_PT))],
        want_logdet=True,
    )
    jax.block_until_ready((mean, w, ld))
    assert sparsax.factorization_count() - c0 == 1
    np.testing.assert_allclose(mean, np.linalg.solve(P, b), rtol=1e-9)
    # w = P' L^-T z has covariance P^{-1}, so w' P w == z' z exactly.
    w = np.asarray(w)
    np.testing.assert_allclose(w @ P @ w, z @ z, rtol=1e-9)
    np.testing.assert_allclose(float(ld), np.linalg.slogdet(P)[1], rtol=1e-11)


def test_token_solves_and_logdet(mode):
    Ai, Aj, values, dense, n = spd_system()
    P = dense(3.0)
    b = np.linspace(-1.0, 1.0, n)
    tok = sparsax.factor(Ai, Aj, values(3.0), n)
    x = sparsax.solve_factor(tok, b, mode=sparsax.MODE_P)
    x = sparsax.solve_factor(tok, x, mode=sparsax.MODE_L)
    x = sparsax.solve_factor(tok, x, mode=sparsax.MODE_LT)
    x = sparsax.solve_factor(tok, x, mode=sparsax.MODE_PT)
    np.testing.assert_allclose(x, np.linalg.solve(P, b), rtol=1e-9)
    np.testing.assert_allclose(
        float(sparsax.logdet_factor(tok)), np.linalg.slogdet(P)[1], rtol=1e-11
    )


def test_selinv_and_update_solve(mode):
    Ai, Aj, values, dense, n = spd_system(16)
    P = dense(2.5)
    z = np.asarray(sparsax.selinv(Ai, Aj, values(2.5), n))
    np.testing.assert_allclose(z, np.linalg.inv(P)[Ai, Aj], atol=1e-11)

    rng = np.random.default_rng(2)
    C = 0.2 * rng.normal(size=(n, 2))
    b = rng.normal(size=n)
    x, ld = sparsax.update_solve(Ai, Aj, values(2.5), C, b, return_logdet=True)
    Pu = P + C @ C.T
    np.testing.assert_allclose(x, np.linalg.solve(Pu, b), rtol=1e-9)
    np.testing.assert_allclose(float(ld), np.linalg.slogdet(Pu)[1], rtol=1e-11)


def test_concurrent_shared_and_distinct_factors(mode):
    """Eight threads over four values of s: each factor is shared by two threads."""
    Ai, Aj, values, dense, n = spd_system()
    scales = (2.5, 2.7, 2.9, 3.1)
    refs = {s: (dense(s), np.linalg.slogdet(dense(s))[1]) for s in scales}
    solve = jax.jit(lambda ax, b: sparsax.solve(Ai, Aj, ax, b))
    solve(jnp.asarray(values(scales[0])), jnp.zeros(n))

    def work(t):
        s = scales[t % 4]
        P, logdet = refs[s]
        Ax = jnp.asarray(values(s))
        rng = np.random.default_rng(10 + t)
        worst = 0.0
        for _ in range(20):
            b = rng.normal(size=n)
            x = np.asarray(solve(Ax, jnp.asarray(b)))
            worst = max(worst, np.max(np.abs(P @ x - b)))
        ld = float(sparsax.logdet(Ai, Aj, Ax, n))
        return max(worst, abs(ld - logdet) / abs(logdet))

    with ThreadPoolExecutor(8) as pool:
        assert max(pool.map(work, range(8))) < 1e-9


def test_indefinite_matrix_raises(mode):
    Ai, Aj, values, dense, n = spd_system(16)
    with pytest.raises(Exception, match="positive definite"):
        sparsax.logdet(Ai, Aj, values(1.0), n).block_until_ready()


def _grid(side, dim, shift=1.0):
    """Laplacian-plus-shift on a 2-D or 3-D grid: fill grows quickly in 3-D."""
    path = sp.diags([-np.ones(side - 1), -np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    if dim == 2:
        S = sp.kron(eye, path) + sp.kron(path, eye)
    else:
        S = (
            sp.kron(sp.kron(path, eye), eye)
            + sp.kron(sp.kron(eye, path), eye)
            + sp.kron(sp.kron(eye, eye), path)
        )
    return (S + (2.0 * dim + shift) * sp.identity(side**dim)).tocoo()


def _arrow(n):
    """Every row couples to the last: one dense row, no fill under the right order."""
    d = np.full(n, float(n))
    A = sp.diags(d).tolil()
    A[n - 1, :] = 1.0
    A[:, n - 1] = 1.0
    A[n - 1, n - 1] = float(n)
    return A.tocoo()


def _random_spd(n, density, seed):
    """A random sparsity pattern, for supernodes amalgamation cannot make tidy."""
    B = sp.random(n, n, density=density, random_state=seed, format="csr")
    M = B + B.T
    return (M + sp.diags(np.asarray(abs(M).sum(axis=1)).ravel() + 1.0)).tocoo()


STRUCTURES = {
    "single-entry": lambda: sp.coo_matrix([[2.0]]),
    "diagonal": lambda: sp.diags(np.arange(1.0, 41.0)).tocoo(),
    "tridiagonal": lambda: sp.diags(
        [-np.ones(59), np.full(60, 3.0), -np.ones(59)], [-1, 0, 1]
    ).tocoo(),
    "arrow": lambda: _arrow(50),
    "dense": lambda: sp.coo_matrix(np.eye(30) * 31.0 + np.ones((30, 30))),
    "disconnected": lambda: sp.block_diag([_grid(5, 2), _grid(3, 3)]).tocoo(),
    "grid-2d": lambda: _grid(20, 2),
    "grid-3d": lambda: _grid(9, 3),
    "random-sparse": lambda: _random_spd(300, 0.005, 1),
    "random-dense": lambda: _random_spd(200, 0.05, 2),
}


@pytest.mark.parametrize("structure", STRUCTURES)
def test_selinv_matches_the_dense_inverse(mode, structure):
    A = STRUCTURES[structure]()
    n = A.shape[0]
    U = sp.triu(A).tocoo()
    z = sparsax.selinv(U.row.astype(np.int32), U.col.astype(np.int32), U.data, n)
    want = np.linalg.inv(A.toarray())[U.row, U.col]
    np.testing.assert_allclose(z, want, rtol=1e-10, atol=1e-12)


def test_selinv_in_auto_mode_on_a_problem_big_enough_to_go_supernodal():
    A = _grid(12, 3)
    U = sp.triu(A).tocoo()
    z = sparsax.selinv(
        U.row.astype(np.int32), U.col.astype(np.int32), U.data, A.shape[0]
    )
    want = np.linalg.inv(A.toarray())[U.row, U.col]
    np.testing.assert_allclose(z, want, rtol=1e-10, atol=1e-12)


def test_selinv_reads_both_triangles_and_duplicates(mode):
    """z[k] answers for COO position k, whichever triangle or repeat it names."""
    A = _grid(6, 2)
    Ad = A.toarray()
    U = sp.triu(A).tocoo()
    L = sp.tril(A, k=-1).tocoo()
    # the first diagonal entry split in two: CHOLMOD sums upper-triangle repeats
    diag0 = np.flatnonzero((U.row == 0) & (U.col == 0))[0]
    data_u = U.data.copy()
    data_u[diag0] *= 0.25
    Ai = np.concatenate([U.row, L.row, [0]]).astype(np.int32)
    Aj = np.concatenate([U.col, L.col, [0]]).astype(np.int32)
    Ax = np.concatenate([data_u, L.data, [0.75 * U.data[diag0]]])
    z = sparsax.selinv(Ai, Aj, Ax, A.shape[0])
    np.testing.assert_allclose(z, np.linalg.inv(Ad)[Ai, Aj], rtol=1e-10, atol=1e-12)


def test_logdet_gradient_matches_the_dense_inverse(mode):
    A = _grid(7, 3)
    U = sp.triu(A).tocoo()
    Ai, Aj = U.row.astype(np.int32), U.col.astype(np.int32)
    n = A.shape[0]
    g = jax.grad(lambda ax: sparsax.logdet(Ai, Aj, ax, n))(jnp.asarray(U.data))
    inv = np.linalg.inv(A.toarray())
    # An upper-triangle entry stands for both of its symmetric positions.
    want = inv[Ai, Aj] * np.where(Ai == Aj, 1.0, 2.0)
    np.testing.assert_allclose(g, want, rtol=1e-10, atol=1e-12)


def test_selinv_vmaps_over_values(mode):
    A = _grid(6, 3)
    U = sp.triu(A).tocoo()
    Ai, Aj = U.row.astype(np.int32), U.col.astype(np.int32)
    n = A.shape[0]
    scales = np.array([1.0, 2.0, 4.0])
    got = jax.jit(jax.vmap(lambda s: sparsax.selinv(Ai, Aj, s * U.data, n)))(scales)
    want = np.stack([np.linalg.inv(s * A.toarray())[Ai, Aj] for s in scales])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_selinv_is_exact_from_concurrent_threads(mode):
    """Each of eight threads inverts values shared with one other thread."""
    A = _grid(7, 3)
    U = sp.triu(A).tocoo()
    Ai, Aj = U.row.astype(np.int32), U.col.astype(np.int32)
    n = A.shape[0]
    scales = (1.0, 1.5, 2.0, 2.5)
    wants = {s: np.linalg.inv(s * A.toarray())[Ai, Aj] for s in scales}
    f = jax.jit(lambda ax: sparsax.selinv(Ai, Aj, ax, n))
    f(jnp.asarray(U.data))

    def work(t):
        s = scales[t % 4]
        ax = jnp.asarray(s * U.data)
        return max(np.max(np.abs(np.asarray(f(ax)) - wants[s])) for _ in range(10))

    with ThreadPoolExecutor(8) as pool:
        assert max(pool.map(work, range(8))) < 1e-11


@pytest.mark.parametrize("n", [145, 200, 300])
def test_selinv_of_a_dense_block_is_right_on_every_repeat(n):
    """One supernode of about this size, inverted again and again.

    Accelerate's threaded dpotri and dtrtri (macOS 27.2, M3 Max) return wrong
    values on about one call in twenty at n = 145 and less often above it, and
    never with one BLAS thread.  A single call usually gets the right answer, so
    only repetition shows it.
    """
    rng = np.random.default_rng(n)
    M = rng.normal(size=(n, n))
    Ad = M @ M.T + n * np.eye(n)
    Ai, Aj = np.triu_indices(n)
    Ai, Aj = Ai.astype(np.int32), Aj.astype(np.int32)
    Ax = jnp.asarray(Ad[Ai, Aj])
    want = np.linalg.inv(Ad)[Ai, Aj]
    sparsax.set_options(supernodal="supernodal")
    try:
        f = jax.jit(lambda a: sparsax.selinv(Ai, Aj, a, n))
        worst = max(np.max(np.abs(np.asarray(f(Ax)) - want)) for _ in range(150))
    finally:
        sparsax.set_options(supernodal="auto")
    assert worst < 1e-12


def test_selinv_of_a_sparse_factor_is_right_on_every_repeat():
    A = _random_spd(200, 0.05, 2)
    U = sp.triu(A).tocoo()
    Ai, Aj = U.row.astype(np.int32), U.col.astype(np.int32)
    want = np.linalg.inv(A.toarray())[Ai, Aj]
    sparsax.set_options(supernodal="supernodal")
    try:
        for _ in range(150):
            sparsax.clear_cache()
            z = np.asarray(sparsax.selinv(Ai, Aj, U.data, A.shape[0]))
            np.testing.assert_allclose(z, want, rtol=1e-10, atol=1e-12)
    finally:
        sparsax.set_options(supernodal="auto")


@pytest.mark.skipif(sys.platform == "win32", reason="no direct BLAS link on Windows")
def test_supernodal_selinv_builds_no_simplicial_copy():
    """The route is the point: the answers would match without it, far more slowly."""
    assert sparsax._cpp.has_supernodal_selinv()
    A = _grid(7, 3)
    U = sp.triu(A).tocoo()
    Ai, Aj = U.row.astype(np.int32), U.col.astype(np.int32)
    n = A.shape[0]
    sparsax.set_options(supernodal="supernodal")
    try:
        before = sparsax._cpp.ldl_conversion_count()
        jax.block_until_ready(sparsax.selinv(Ai, Aj, U.data, n))
        assert sparsax._cpp.ldl_conversion_count() == before
        # update_solve does need the copy, so the counter is live.
        jax.block_until_ready(
            sparsax.update_solve(Ai, Aj, U.data, 0.1 * np.ones(n), np.ones(n))
        )
        assert sparsax._cpp.ldl_conversion_count() == before + 1
    finally:
        sparsax.set_options(supernodal="auto")
