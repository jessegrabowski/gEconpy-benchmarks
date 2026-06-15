"""Full perturbation solution via `Model.solve_model`, per solver.

This is the user-facing solve: linearize (cached evaluation, warmed in `setup`) plus the perturbation
solver. It complements solvers/ — which times the bare linear algebra — by measuring the assembled
path a user actually calls.
"""

from .._support import build_model

LADDER = ["small", "rbc", "medium", "large"]
SOLVERS = ["cycle_reduction", "gensys"]


class Solve:
    params = [LADDER, SOLVERS]
    param_names = ["model", "solver"]

    def setup(self, model, solver):
        self.model = build_model(model)
        self.model.steady_state(verbose=False)
        self.model.solve_model(solver=solver, verbose=False)  # warm linearize compile

    def time_solve(self, model, solver):
        self.model.solve_model(solver=solver, verbose=False)

    def peakmem_solve(self, model, solver):
        self.model.solve_model(solver=solver, verbose=False)
