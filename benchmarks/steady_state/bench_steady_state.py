"""Steady-state computation: analytic closed-form evaluation versus a numerical root solve.

All ladder models ship an analytic STEADY_STATE block. The analytic path compiles a closed-form
function once (warmed in `setup`) and then evaluates it. Each call also checks the residuals of the
solution. Before gEconpy commit 9a8300b that check recompiled the residual function on every call, so
on those commits `time_analytic` is dominated by the compile.
The numeric path additionally runs a SciPy root solve over a compiled residual/Jacobian; the
compile is warmed in `setup`, so `time_root` isolates the solve loop.
"""

from .._support import warm_model

LADDER = ["small", "rbc", "medium", "large"]


class Analytic:
    params = [LADDER]
    param_names = ["model"]
    timeout = 900

    def setup(self, model):
        self.model = warm_model(model)
        self.model.steady_state(how="analytic", verbose=False)  # relink the f_ss function

    def time_analytic(self, model):
        self.model.steady_state(how="analytic", verbose=False)


class Numeric:
    params = [["small", "rbc"]]
    param_names = ["model"]

    def setup(self, model):
        self.model = warm_model(model)
        # Warm the residual/Jacobian compile so the timed call measures only the solve.
        self.model.steady_state(how="root", progressbar=False, verbose=False)

    def time_root(self, model):
        self.model.steady_state(how="root", progressbar=False, verbose=False)
