"""Linearization, split into graph construction versus evaluation.

`linearize_model` decomposes into `symbolic_linearization` — building the A/B/C/D PyTensor graphs via
autodiff — and evaluating the compiled callable at a steady state and parameter vector. The compiled
function is memoized on the Model, so warm `time_eval` measures pure evaluation, while `time_symbolic`
clears the graph cache each call to measure graph construction (the dominant, most actionable cost
for this codebase).
"""

from .._support import build_model, warm_model

LADDER = ["small", "rbc", "medium", "large"]


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
