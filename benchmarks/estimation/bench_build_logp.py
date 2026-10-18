"""Cost of standing up an estimation model's symbolic graph.

Builds the full path from GCN to a PyMC model with the Kalman-filter likelihood graph attached
(`statespace_from_gcn` -> `configure` -> `to_pymc` -> `build_statespace_graph`), fresh each call. This
is all symbolic graph construction, and PyTensor function compilation happens later at sample time, so
building anew each call is a genuinely cold measurement with no shared state.
"""

from .._support import peak_traced_bytes
from ._estimation import build_pymc_model, make_configured_statespace, synthetic_data


class BuildStatespace:
    params = [["gensys", "cycle_reduction"]]
    param_names = ["solver"]
    timeout = 600  # builds + compiles the state-space graph; keep clear of asv's 60s default

    def setup(self, solver):
        self.data = synthetic_data()

    def time_build(self, solver):
        ss_mod = make_configured_statespace(solver=solver)
        build_pymc_model(ss_mod, self.data)


class BuildStatespaceMemory:
    params = [["gensys", "cycle_reduction"]]
    param_names = ["solver"]
    timeout = 600
    unit = "bytes"

    def setup(self, solver):
        self.data = synthetic_data()
        # The first build in a process also allocates lazily imported modules and one-time caches.
        build_pymc_model(make_configured_statespace(solver=solver), self.data)

    def track_build_peak_bytes(self, solver):
        return peak_traced_bytes(lambda: build_pymc_model(make_configured_statespace(solver=solver), self.data))
