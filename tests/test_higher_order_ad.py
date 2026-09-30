"""Higher-order autodiff through the sparse solves.

``solve``, ``lu_solve`` and ``umf_solve`` differentiate by implicit
differentiation (``jax.lax.custom_linear_solve``), so derivatives of every
order, in forward and reverse mode, must match a dense ``jnp.linalg.solve``
reference.  A first-order ``custom_vjp`` whose backward pass calls the raw FFI
kernel fails every second-derivative check here.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import sparsax


def _spd(n=30, seed=0):
    """Sparse SPD A = I - 0.6 W_sym, stored by its upper triangle."""
    rng = np.random.default_rng(seed)
    M = sp.random(n, n, density=0.15, random_state=seed)
    M = ((M + M.T) > 0).astype(float)
    M.setdiag(0.0)
    d = np.maximum(np.asarray(M.sum(1)).ravel(), 1.0)
    Ws = sp.diags(1 / np.sqrt(d)) @ M @ sp.diags(1 / np.sqrt(d))
    A = sp.eye(n) - 0.6 * Ws
    U = sp.triu(A).tocoo()
    B = sp.triu(sp.random(n, n, density=0.1, random_state=seed + 1) * 0.1).tocoo()
    return U, B, rng.standard_normal((n, 3))


def _nonsym(n=30, seed=0):
    """Sparse non-symmetric A = I - 0.5 W, W row-standardised."""
    rng = np.random.default_rng(seed)
    M = (sp.random(n, n, density=0.15, random_state=seed) > 0).astype(float)
    M.setdiag(0.0)
    d = np.maximum(np.asarray(M.sum(1)).ravel(), 1.0)
    A = (sp.eye(n) - 0.5 * sp.diags(1 / d) @ M).tocoo()
    B = (sp.random(n, n, density=0.1, random_state=seed + 1) * 0.1).tocoo()
    return A, B, rng.standard_normal((n, 3))


def _dense(Ai, Aj, Ax, n, upper):
    """Dense matrix from COO, mirroring the upper triangle when ``upper``."""
    D = jnp.zeros((n, n)).at[Ai, Aj].add(Ax)
    if upper:
        D = jnp.triu(D)
        D = D + jnp.triu(D, 1).T
    return D


CASES = {
    "solve": (sparsax.solve, _spd, True),
    "lu_solve": (sparsax.lu_solve, _nonsym, False),
    "umf_solve": (sparsax.umf_solve, _nonsym, False),
}


def _setup(name):
    fn, make, upper = CASES[name]
    A, B, b = make()
    # A(t) = A + t B on the union pattern, so every derivative in t is nontrivial.
    P = (A + B).tocoo()
    Ai, Aj = P.row.astype(np.int32), P.col.astype(np.int32)
    Ax0 = np.asarray(A.tocsr()[Ai, Aj]).ravel()
    Bx = np.asarray(B.tocsr()[Ai, Aj]).ravel()
    n = A.shape[0]
    b = jnp.asarray(b)

    def sparse(t):
        return jnp.sum(jnp.sin(fn(Ai, Aj, Ax0 + t * Bx, b * (1 + t))))

    def dense(t):
        M = _dense(Ai, Aj, Ax0 + t * Bx, n, upper)
        return jnp.sum(jnp.sin(jnp.linalg.solve(M, b * (1 + t))))

    return sparse, dense, (fn, Ai, Aj, Ax0, b, n, upper)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize(
    "order",
    [
        "grad",
        "fwd_over_rev",
        "rev_over_rev",
        "third",
    ],
)
def test_derivatives_match_dense(name, order):
    sparse, dense, _ = _setup(name)
    t = 0.3
    ops = {
        "grad": lambda f: jax.grad(f)(t),
        "fwd_over_rev": lambda f: jax.jvp(jax.grad(f), (t,), (1.0,))[1],
        "rev_over_rev": lambda f: jax.grad(jax.grad(f))(t),
        "third": lambda f: jax.grad(jax.grad(jax.grad(f)))(t),
    }
    np.testing.assert_allclose(ops[order](sparse), ops[order](dense), rtol=1e-9)


@pytest.mark.parametrize("name", list(CASES))
def test_hessian_wrt_values_matches_dense(name):
    """Full Hessian with respect to Ax, which exercises the COO folding."""
    _, _, (fn, Ai, Aj, Ax0, b, n, upper) = _setup(name)

    def sparse(Ax):
        return jnp.sum(fn(Ai, Aj, Ax, b) ** 2)

    def dense(Ax):
        return jnp.sum(jnp.linalg.solve(_dense(Ai, Aj, Ax, n, upper), b) ** 2)

    Ax = jnp.asarray(Ax0)
    np.testing.assert_allclose(jax.hessian(sparse)(Ax), jax.hessian(dense)(Ax), atol=1e-9)


@pytest.mark.parametrize("name", list(CASES))
def test_second_derivatives_under_jit_and_vmap(name):
    sparse, dense, _ = _setup(name)
    ts = jnp.array([0.1, 0.3, 0.5])
    got = jax.jit(jax.vmap(jax.grad(jax.grad(sparse))))(ts)
    want = jax.vmap(jax.grad(jax.grad(dense)))(ts)
    np.testing.assert_allclose(got, want, rtol=1e-9)


@pytest.mark.parametrize("name", list(CASES))
def test_nested_vmap(name):
    """vmap of vmap folds into one batched native call."""
    fn, make, upper = CASES[name]
    A, _, _ = make()
    Ai, Aj, Ax = A.row.astype(np.int32), A.col.astype(np.int32), A.data
    n = A.shape[0]
    B = jnp.asarray(np.random.default_rng(3).standard_normal((4, 3, n)))
    Axs = jnp.asarray(Ax) * (1 + 0.01 * jnp.arange(4.0))[:, None]
    got = jax.vmap(lambda ax, bs: jax.vmap(lambda bb: fn(Ai, Aj, ax, bb))(bs))(Axs, B)
    want = jax.vmap(
        lambda ax, bs: jax.vmap(lambda bb: jnp.linalg.solve(_dense(Ai, Aj, ax, n, upper), bb))(bs)
    )(Axs, B)
    np.testing.assert_allclose(got, want, atol=1e-10)
