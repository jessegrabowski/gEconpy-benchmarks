"""Post-solution analysis: theoretical moments, impulse responses, and simulation.

All operate on an already-solved model (T, R computed in `setup`), isolating the analysis cost from
the build/solve cost benchmarked elsewhere. `time_check_bk_condition` gets the A/B/C/D matrices from
`setup` too, so it times only the eigenvalue check.
"""

from .._support import policy_matrices

LADDER = ["rbc", "medium"]


class Statistics:
    params = [LADDER]
    param_names = ["model"]
    timeout = 900

    def setup(self, model):
        from gEconpy import (
            autocovariance_matrix,
            check_bk_condition,
            impulse_response_function,
            simulate,
            stationary_covariance_matrix,
        )

        self.model, self.T, self.R = policy_matrices(model)
        self.A, self.B, self.C, self.D = self.model.linearize_model(verbose=False)
        self._stationary_covariance_matrix = stationary_covariance_matrix
        self._autocovariance_matrix = autocovariance_matrix
        self._impulse_response_function = impulse_response_function
        self._simulate = simulate
        self._check_bk_condition = check_bk_condition
        # The first eigenvalue check in a process is several thousand times slower than later ones.
        self._check_bk_condition(self.model, A=self.A, B=self.B, C=self.C, D=self.D, verbose=False)

    def time_stationary_covariance(self, model):
        self._stationary_covariance_matrix(self.model, T=self.T, R=self.R, shock_std=1.0)

    def time_autocovariance(self, model):
        self._autocovariance_matrix(self.model, T=self.T, R=self.R, shock_std=1.0, n_lags=10)

    def time_impulse_response(self, model):
        self._impulse_response_function(self.model, T=self.T, R=self.R, simulation_length=40, shock_size=1.0)

    def time_simulate(self, model):
        self._simulate(self.model, T=self.T, R=self.R, simulation_length=100, shock_std=1.0)

    def time_check_bk_condition(self, model):
        self._check_bk_condition(self.model, A=self.A, B=self.B, C=self.C, D=self.D, verbose=False)
