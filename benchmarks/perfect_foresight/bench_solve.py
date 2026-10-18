"""Perfect-foresight simulation, scaling over the simulation horizon.

The solver stacks per-period Jacobians into one large sparse system and drives it with a sparse
Newton iteration, so cost grows with the horizon. The model is built and its steady state solved in
`setup`; only the simulation is timed.
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

    def time_solve(self, model, horizon):
        self._solve(
            self.model,
            simulation_length=horizon,
            initial_conditions=self.initial_conditions,
        )
