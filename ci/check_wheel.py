"""Post-build checks on an installed sparsax wheel (cibuildwheel test step).

Fails the build when the wheel bundles a runtime that can clash with the host
process, or when the solves stop being differentiable to second order.
"""

import pathlib
import re
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sparsax

# Runtimes that must come from the host, never from the wheel: a second copy of
# an OpenMP runtime aborts the process, and the Fortran runtime is not needed.
FORBIDDEN = re.compile(
    r"(libomp|libiomp|libgomp|vcomp|libgfortran|libquadmath)", re.IGNORECASE
)

root = pathlib.Path(sparsax.__file__).resolve().parent
bundled = [
    p.name
    for d in (root / ".dylibs", root.parent / "sparsax.libs")
    if d.is_dir()
    for p in d.iterdir()
]
offending = [name for name in bundled if FORBIDDEN.search(name)]
if offending:
    sys.exit(f"wheel bundles host runtimes it must not carry: {offending}")

# A 3x3 SPD system: the solve, and a second derivative through it.
Ai = np.array([0, 0, 1, 1, 2], dtype=np.int32)
Aj = np.array([0, 1, 1, 2, 2], dtype=np.int32)
Ax = jnp.array([4.0, 1.0, 3.0, 0.5, 2.0])
b = jnp.array([1.0, 2.0, 3.0])


def f(t):
    return jnp.sum(sparsax.solve(Ai, Aj, Ax * (1.0 + t), b) ** 2)


A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
np.testing.assert_allclose(
    np.asarray(sparsax.solve(Ai, Aj, Ax, b)), np.linalg.solve(A, b)
)
# f(t) = |x|^2 / (1 + t)^2 with x = A^{-1} b, so f''(0) = 6 |x|^2.
x = np.linalg.solve(A, np.asarray(b))
np.testing.assert_allclose(float(jax.grad(jax.grad(f))(0.0)), 6.0 * x @ x, rtol=1e-10)

# The selected inverse of a supernodal factor calls BLAS itself, not through
# CHOLMOD, so a symbol that fails to resolve in this wheel shows up here.
# A 400-node grid leaves several supernodes, which reaches every routine it calls.
if sys.platform != "win32" and not sparsax._cpp.has_supernodal_selinv():
    sys.exit("wheel was built without the supernodal selected inverse")
side = 20
ids = np.arange(side**2).reshape(side, side)
right, down = (
    (ids[:, :-1].ravel(), ids[:, 1:].ravel()),
    (ids[:-1].ravel(), ids[1:].ravel()),
)
gi = np.concatenate([ids.ravel(), right[0], down[0]]).astype(np.int32)
gj = np.concatenate([ids.ravel(), right[1], down[1]]).astype(np.int32)
gx = np.concatenate([np.full(side**2, 5.0), np.full(len(gi) - side**2, -1.0)])
G = np.zeros((side**2, side**2))
G[gi, gj] = gx
G = G + G.T - np.diag(np.diag(G))
sparsax.set_options(supernodal="supernodal")
np.testing.assert_allclose(
    np.asarray(sparsax.selinv(gi, gj, gx, side**2)),
    np.linalg.inv(G)[gi, gj],
    rtol=1e-9,
)
sparsax.set_options(supernodal="auto")
print(
    f"sparsax {sparsax.__version__}: bundled {sorted(bundled) or 'nothing'}; checks passed"
)
