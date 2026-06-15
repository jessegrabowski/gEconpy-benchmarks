"""Sparse Newton root solver, on a well-conditioned synthetic system across sizes."""

import numpy as np

from scipy import sparse

TOL = 1e-10
MAXITER = 100


class SparseRoot:
    params = [[100, 500, 1000]]
    param_names = ["system_size"]

    def setup(self, system_size):
        from gEconpy.solvers.sparse_root import sparse_root

        self._sparse_root = sparse_root
        rng = np.random.default_rng(42)
        self.x0 = rng.standard_normal(system_size)

        # Diagonal dominance keeps the system well-conditioned and the solve deterministic.
        A = sparse.diags(2.0 + rng.random(system_size), format="csc")
        b = rng.standard_normal(system_size)

        def fun(x):
            return A @ x - b, A

        self.fun = fun

    def time_solve(self, system_size):
        self._sparse_root(self.fun, self.x0, tol=TOL, maxiter=MAXITER)

    def peakmem_solve(self, system_size):
        self._sparse_root(self.fun, self.x0, tol=TOL, maxiter=MAXITER)
