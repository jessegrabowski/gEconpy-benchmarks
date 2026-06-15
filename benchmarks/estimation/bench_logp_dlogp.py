"""Warm logp / dlogp evaluation — the inner loop of HMC sampling and the primary estimation cost.

Everything compiles in `setup` (and is warmed once), so the timed methods measure pure evaluation
at a single parameter draw. `time_logp_dlogp` uses the fused value-and-gradient function NUTS
actually calls; `time_logp` and `time_dlogp` break out the two halves.
"""

from ._estimation import build_pymc_model, make_configured_statespace, synthetic_data


class LogpEval:
    params = [["gensys", "cycle_reduction"]]
    param_names = ["solver"]
    timeout = 600  # setup compiles logp+dlogp; keep clear of asv's 60s default

    def setup(self, solver):
        from pymc.blocking import DictToArrayBijection

        ss_mod = make_configured_statespace(solver=solver)
        self.model = build_pymc_model(ss_mod, synthetic_data())
        self.point = self.model.initial_point()

        self._logp = self.model.compile_logp()
        self._dlogp = self.model.compile_dlogp()
        self._logp_dlogp = self.model.logp_dlogp_function()
        self._logp_dlogp.set_extra_values({})
        self._raveled = DictToArrayBijection.map(self.point)

        # Warm every compiled function so the timed calls hit no first-call compilation.
        self._logp(self.point)
        self._dlogp(self.point)
        self._logp_dlogp(self._raveled)

    def time_logp(self, solver):
        self._logp(self.point)

    def time_dlogp(self, solver):
        self._dlogp(self.point)

    def time_logp_dlogp(self, solver):
        self._logp_dlogp(self._raveled)

    def peakmem_logp_dlogp(self, solver):
        self._logp_dlogp(self._raveled)
