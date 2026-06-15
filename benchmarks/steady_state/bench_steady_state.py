"""Steady-state computation: analytic closed-form evaluation versus a numerical root solve.

All ladder models ship an analytic STEADY_STATE block. The analytic path compiles a closed-form
function once (warmed in `setup`) and then evaluates it, so `time_analytic` isolates evaluation.
The numeric path additionally runs a SciPy root solve over a compiled residual/Jacobian; the
compile is warmed in `setup`, so `time_root` isolates the solve loop.
"""

from .._support import build_model

LADDER = ["small", "rbc", "medium", "large"]


class Analytic:
    params = [LADDER]
    param_names = ["model"]

    def setup(self, model):
        self.model = build_model(model)
        self.model.steady_state(how="analytic", verbose=False)  # warm the f_ss compile

    def time_analytic(self, model):
        self.model.steady_state(how="analytic", verbose=False)

    def peakmem_analytic(self, model):
        self.model.steady_state(how="analytic", verbose=False)


class Numeric:
    params = [["small", "rbc"]]
    param_names = ["model"]

    def setup(self, model):
        self.model = build_model(model)
        # Warm the residual/Jacobian compile so the timed call measures only the solve.
        self.model.steady_state(how="root", progressbar=False, verbose=False)

    def time_root(self, model):
        self.model.steady_state(how="root", progressbar=False, verbose=False)
