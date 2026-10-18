"""Shared fixtures and helpers for the gEconpy benchmark suite.

The model ladder spans the realistic size range so scaling behavior is visible: a minimal
three-equation NK, a canonical RBC, a full New Keynesian model, and a large multi-sector NK with
a tax system. Benchmarks pick a subset appropriate to their cost. Cold compile benches stay on the
cheaper rungs, and pure-numpy solver and warm-eval benches climb higher.
"""

import tracemalloc

from pathlib import Path

# GCN files are vendored under _fixtures/ rather than loaded from the installed package: the model
# inputs must stay constant across gEconpy's history (old commits may lack a model, or ship a
# different version of it), so that benchmark timings reflect code changes alone.
FIXTURES = Path(__file__).resolve().parent / "_fixtures"

# name -> vendored GCN stem (relative to _fixtures/), smallest first. Sizes count variables after reduction, as
# measured on gEconpy main, and each model has as many equations as variables.
MODEL_LADDER = {
    "small": "Three_Equation_NK",  # 4 variables
    "rbc": "RBC",  # 9 variables
    "medium": "New_Keynesian",  # 23 variables
    "large": "sims_2024/nk_complete_taxes",  # 33 variables
}
LADDER = list(MODEL_LADDER)


def peak_traced_bytes(call):
    """Run ``call`` and return the peak bytes Python allocated during it, as ``tracemalloc`` traces them.

    NumPy allocations are traced. Memory used by subprocesses, such as C compilation, is not.
    """
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def gcn_path(name):
    """Resolve a ladder key (or a raw vendored stem) to its `.gcn` path under _fixtures/."""
    return FIXTURES / f"{MODEL_LADDER.get(name, name)}.gcn"


def build_model(name, mode=None):
    """Build a freshly compiled `Model` from a ladder key, uncached.

    Used by the build and compile-measurement benchmarks, which must pay the full construction and
    compilation cost on every call. Eval/solve benchmarks should use `warm_model` instead.
    """
    from gEconpy.model.build import model_from_gcn

    # Only pass ``mode`` when set, so the default path works on older gEconpy that predates the
    # ``backend`` -> ``mode`` rename.
    kwargs = {} if mode is None else {"mode": mode}
    return model_from_gcn(gcn_path(name), verbose=False, **kwargs)


def warm_model(name, mode=None):
    """Return a fully warmed `Model` (steady state + linearization + solution compiled), cached.

    The compile happens once per (model, gEconpy version) and is reused across benchmarks and runs,
    so eval/solve benchmark `setup` reloads it instead of recompiling. The benchmark still triggers
    one warm call of its target method to relink the compiled function from PyTensor's cache.
    """
    from ._cache import load_or_build

    def _build():
        model = build_model(name, mode=mode)
        model.steady_state(verbose=False)
        model.linearize_model(verbose=False)
        model.solve_model(verbose=False)
        return model

    return load_or_build(f"warm_{mode}", gcn_path(name), _build)


def solved_matrices(name, mode=None):
    """Return the linearization matrices (A, B, C, D) evaluated at the steady state.

    These feed the perturbation-solver benchmarks, which are pure NumPy linear algebra and so
    must be fed already-evaluated matrices rather than a live model.
    """
    A, B, C, D = warm_model(name, mode=mode).linearize_model(verbose=False)
    return A, B, C, D


def policy_matrices(name, mode=None):
    """Return the perturbation solution (T, R) for post-solution analysis benchmarks."""
    model = warm_model(name, mode=mode)
    T, R = model.solve_model(verbose=False)
    return model, T, R
