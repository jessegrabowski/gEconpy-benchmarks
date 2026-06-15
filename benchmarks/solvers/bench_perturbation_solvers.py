"""Perturbation solvers in isolation: pure NumPy/SciPy linear algebra on real model matrices.

Each solver is fed the A/B/C/D matrices produced by linearizing a model at its steady state, so the
timing reflects solver scaling over genuine problem sizes rather than synthetic random systems.
"""

import numpy as np

from .._support import solved_matrices

LADDER = ["small", "rbc", "medium", "large"]


class _SolverBenchmark:
    params = [LADDER]
    param_names = ["model"]

    def setup(self, model):
        A, B, C, D = solved_matrices(model)
        self.A = np.ascontiguousarray(A)
        self.B = np.ascontiguousarray(B)
        self.C = np.ascontiguousarray(C)
        self.D = np.ascontiguousarray(D)


class CycleReduction(_SolverBenchmark):
    def setup(self, model):
        super().setup(model)
        from gEconpy.solvers.cycle_reduction import solve_policy_function_with_cycle_reduction

        self._solve = solve_policy_function_with_cycle_reduction

    def time_solve(self, model):
        self._solve(self.A, self.B, self.C, self.D, max_iter=1000, tol=1e-8, verbose=False)


class Gensys(_SolverBenchmark):
    def setup(self, model):
        super().setup(model)
        from gEconpy.solvers.gensys import solve_policy_function_with_gensys

        self._solve = solve_policy_function_with_gensys

    def time_solve(self, model):
        self._solve(self.A, self.B, self.C, self.D, tol=1e-8)


class BackwardDirect(_SolverBenchmark):
    def setup(self, model):
        super().setup(model)
        from gEconpy.solvers.backward_looking import solve_policy_function_with_backward_direct

        self._solve = solve_policy_function_with_backward_direct

    def time_solve(self, model):
        self._solve(self.A, self.B, self.C, self.D)
