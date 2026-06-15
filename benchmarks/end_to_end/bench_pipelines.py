"""End-to-end flows a user actually runs, composing the units benchmarked in isolation elsewhere.

These exist to catch regressions that only show up across stage boundaries (e.g. a change that
shifts cost from build into solve), which per-unit benchmarks can miss.
"""


class GcnToSolvedModel:
    """The full build -> steady state -> perturbation solve pipeline, fresh each call."""

    def setup(self):
        from .._support import gcn_path

        self.path = gcn_path("rbc")

    def time_gcn_to_solved_model(self):
        from gEconpy.model.build import model_from_gcn

        m = model_from_gcn(self.path, verbose=False)
        m.steady_state(verbose=False)
        m.solve_model(verbose=False)


class HotEstimationLoop:
    """A burst of fused logp+gradient evaluations, standing in for a stretch of HMC sampling."""

    n_steps = 50
    timeout = 600  # setup compiles logp+dlogp; keep clear of asv's 60s default

    def setup(self):
        from pymc.blocking import DictToArrayBijection

        from ..estimation._estimation import (
            build_pymc_model,
            make_configured_statespace,
            synthetic_data,
        )

        ss_mod = make_configured_statespace(solver="gensys")
        self.model = build_pymc_model(ss_mod, synthetic_data())
        self._logp_dlogp = self.model.logp_dlogp_function()
        self._logp_dlogp.set_extra_values({})
        self._raveled = DictToArrayBijection.map(self.model.initial_point())
        self._logp_dlogp(self._raveled)  # warm

    def time_50_logp_dlogp_steps(self):
        f, q = self._logp_dlogp, self._raveled
        for _ in range(self.n_steps):
            f(q)
