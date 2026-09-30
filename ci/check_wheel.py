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
FORBIDDEN = re.compile(r"(libomp|libiomp|libgomp|vcomp|libgfortran|libquadmath)", re.IGNORECASE)

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
np.testing.assert_allclose(np.asarray(sparsax.solve(Ai, Aj, Ax, b)), np.linalg.solve(A, b))
# f(t) = |x|^2 / (1 + t)^2 with x = A^{-1} b, so f''(0) = 6 |x|^2.
x = np.linalg.solve(A, np.asarray(b))
np.testing.assert_allclose(float(jax.grad(jax.grad(f))(0.0)), 6.0 * x @ x, rtol=1e-10)
print(f"sparsax {sparsax.__version__}: bundled {sorted(bundled) or 'nothing'}; checks passed")
