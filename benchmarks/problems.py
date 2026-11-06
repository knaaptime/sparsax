"""Sparse symmetric positive definite test problems, as scipy COO matrices."""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def grid(side, dim):
    """Laplacian plus a shift on a 2-D or 3-D grid."""
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
    return (S + (2.0 * dim + 1.0) * sp.identity(side**dim)).tocoo()


def _from_adjacency(W):
    degree = np.asarray(W.sum(1)).ravel()
    return (sp.diags(degree + 1.0) - 0.9 * W).tocoo()


def knn(n, k, seed=0):
    """A symmetrized k-nearest-neighbour graph on random points."""
    points = np.random.default_rng(seed).uniform(size=(n, 2))
    _, neighbours = cKDTree(points).query(points, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    W = sp.coo_matrix((np.ones(n * k), (rows, neighbours[:, 1:].ravel())), shape=(n, n))
    return _from_adjacency(((W + W.T) > 0).astype(float))


def distance_band(n, degree, seed=0):
    """Neighbours within a radius that gives this mean degree, on random points."""
    points = np.random.default_rng(seed).uniform(size=(n, 2))
    radius = np.sqrt(degree / (np.pi * n))
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    W = sp.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    ).tocsr()
    return _from_adjacency(W + W.T)


def spatial_precision(side, rho):
    """(I - rho W)'(I - rho W) for a row-standardized rook W on a square lattice."""
    n = side**2
    eye = sp.identity(side, format="csr")
    path = sp.diags([1.0, 1.0], [-1, 1], shape=(side, side), format="csr")
    W = sp.kron(path, eye) + sp.kron(eye, path)
    W = sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
    M = sp.identity(n) - rho * W
    return (M.T @ M).tocoo()


def upper_coo(A):
    """(Ai, Aj, Ax) of a symmetric matrix's upper triangle, as sparsax takes it."""
    U = sp.triu(A).tocoo()
    return U.row.astype(np.int32), U.col.astype(np.int32), U.data
