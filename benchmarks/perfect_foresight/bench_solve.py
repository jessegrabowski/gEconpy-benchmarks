"""Perfect-foresight simulation, scaling over the simulation horizon.

The solver stacks per-period Jacobians into one large sparse system and drives it with a sparse
Newton iteration, so cost grows with the horizon. `setup` builds the model, solves its steady state, and
runs the simulation once. Only the simulation is timed. From gEconpy commit 307dff2 on, the
compiled problem is cached on the model, so `time_solve` times the solve alone. Before it, every call
also recompiles the problem.
"""

from .._support import warm_model

LADDER = ["rbc", "medium"]
HORIZONS = [50, 100, 200]


class PerfectForesight:
    params = [LADDER, HORIZONS]
    param_names = ["model", "horizon"]
    timeout = 900

    def setup(self, model, horizon):
        from gEconpy import solve_perfect_foresight

        self._solve = solve_perfect_foresight
        self.model = warm_model(model)
        ss = self.model.steady_state(verbose=False)

        # Capital is a predetermined state in every model on this ladder.
        self.initial_conditions = {"K": ss["K_ss"] * 0.9}
        self._solve(self.model, simulation_length=horizon, initial_conditions=self.initial_conditions)

    def time_solve(self, model, horizon):
        self._solve(
            self.model,
            simulation_length=horizon,
            initial_conditions=self.initial_conditions,
        )
