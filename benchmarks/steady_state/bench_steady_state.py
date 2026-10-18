"""Steady-state computation: analytic closed-form evaluation versus a numerical root solve.

`Analytic` runs on the ladder models, which all ship an analytic STEADY_STATE block. The analytic path compiles a
closed-form function once (warmed in `setup`) and then evaluates it. Each call also checks the residuals of the
solution. Before gEconpy commit 9a8300b that check recompiled the residual function on every call, so on those commits
`time_analytic` is dominated by the compile.

`Numeric` runs on RBC with its STEADY_STATE block removed, because gEconpy takes the analytic path whenever the block
covers every variable, even when asked for a root solve. Its residual and Jacobian compile is warmed in `setup`, so
`time_root` times the SciPy root solve.
"""

from .._support import LADDER, warm_model


class Analytic:
    params = [LADDER]
    param_names = ["model"]
    timeout = 900

    def setup(self, model):
        self.model = warm_model(model)
        self.model.steady_state(how="analytic", verbose=False)  # relink the f_ss function

    def time_analytic(self, model):
        self.model.steady_state(how="analytic", verbose=False)


# ladder key -> vendored fixture without a STEADY_STATE block
NO_STEADY_STATE = {"rbc": "RBC_no_steady_state"}


class Numeric:
    params = [list(NO_STEADY_STATE)]
    param_names = ["model"]
    timeout = 900

    def setup(self, model):
        self.model = warm_model(NO_STEADY_STATE[model])
        known = self.model.f_ss(**self.model.parameters()) if self.model.f_ss is not None else {}
        if len(known) >= len(self.model.variables):
            fixture = NO_STEADY_STATE[model]
            raise RuntimeError(f"{fixture} has a fully analytic steady state, so how='root' would skip the root solver")
        # Warm the residual/Jacobian compile so the timed call measures only the solve.
        self.model.steady_state(how="root", progressbar=False, verbose=False)

    def time_root(self, model):
        self.model.steady_state(how="root", progressbar=False, verbose=False)
