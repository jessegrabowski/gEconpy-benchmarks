"""Model construction from a GCN file.

`model_from_gcn` is the symbolic pipeline (parse, FOC derivation, substitution); PyTensor/numba
compilation is lazy and happens on first use, benchmarked separately under linearize/ and
estimation/. Because no compilation happens here, building a fresh model each call is a genuinely
cold measurement with no shared state.
"""

from .._support import gcn_path, peak_traced_bytes

LADDER = ["small", "rbc", "medium", "large"]


class Build:
    params = [LADDER]
    param_names = ["model"]
    timeout = 900  # the large model builds slowly on old (pre-overhaul) gEconpy

    def setup(self, model):
        from gEconpy.model.build import model_from_gcn

        self._build = model_from_gcn
        self.path = gcn_path(model)

    def time_build(self, model):
        self._build(self.path, verbose=False)


class BuildMemory:
    params = [LADDER]
    param_names = ["model"]
    timeout = 900
    unit = "bytes"

    def setup(self, model):
        from gEconpy.model.build import model_from_gcn

        self._build = model_from_gcn
        self.path = gcn_path(model)
        # The first build in a process also allocates lazily imported modules and one-time caches.
        self._build(self.path, verbose=False)

    def track_build_peak_bytes(self, model):
        return peak_traced_bytes(lambda: self._build(self.path, verbose=False))
