"""Linearization, split into graph construction, full compilation, and evaluation.

`linearize_model` decomposes into `symbolic_linearization` (building the A/B/C/D PyTensor graphs via
autodiff), then compiling them to a callable, then evaluating that callable. `time_symbolic` times
graph construction; `time_compile` times the full graph-build + rewrite + link (the compilation that
the reuse cache lets every other benchmark skip); `time_eval` times pure evaluation of the cached
function. The compile/symbolic benchmarks build fresh, uncached models on purpose.
"""

from .._support import build_model, warm_model

LADDER = ["small", "rbc", "medium", "large"]
COMPILE_LADDER = ["small", "rbc", "medium"]  # large compile is too slow to time repeatedly


class LinearizeEval:
    params = [LADDER]
    param_names = ["model"]
    timeout = 900  # setup builds the model on a cache miss (slow for large on old gEconpy)

    def setup(self, model):
        self.model = warm_model(model)
        self.model.linearize_model(verbose=False)  # relink the compiled A/B/C/D from cache

    def time_eval(self, model):
        """Evaluate the (cached) compiled A/B/C/D function at the steady state."""
        self.model.linearize_model(verbose=False)


class LinearizeSymbolic:
    params = [LADDER]
    param_names = ["model"]
    timeout = 900

    def setup(self, model):
        self.model = build_model(model)
        self.model.steady_state(verbose=False)
        self.model.symbolic_linearization(verbose=False)  # create the graph cache

    def time_symbolic(self, model):
        """Build the A/B/C/D PyTensor graphs via autodiff (cache cleared each call)."""
        self.model._symbolic_linearize_cache.clear()
        self.model.symbolic_linearization(verbose=False)


class LinearizeCompile:
    params = [COMPILE_LADDER]
    param_names = ["model"]
    timeout = 600

    def setup(self, model):
        self.model = build_model(model)  # uncached: this benchmark measures compilation
        self.model.steady_state(verbose=False)
        self.model.linearize_model(verbose=False)  # create the caches (untimed)

    def time_compile(self, model):
        """Rebuild the A/B/C/D graphs and recompile the function (rewrites + link)."""
        self.model._symbolic_linearize_cache.clear()
        self.model._linearize_cache.clear()
        self.model.linearize_model(verbose=False)
