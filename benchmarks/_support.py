"""Shared fixtures and helpers for the gEconpy benchmark suite.

The model ladder spans the realistic size range so scaling behaviour is visible: a minimal
three-equation NK, a canonical RBC, a full New Keynesian model, and a large multi-sector NK with
a tax system. Benchmarks pick a subset appropriate to their cost — cold compile/peakmem benches
stay on the cheaper rungs; pure-numpy solver and warm-eval benches climb higher.
"""

from pathlib import Path

# GCN files are vendored under _fixtures/ rather than loaded from the installed package: the model
# inputs must stay constant across gEconpy's history (old commits may lack a model, or ship a
# different version of it), so that benchmark timings reflect code changes alone.
FIXTURES = Path(__file__).resolve().parent / "_fixtures"

# name -> vendored GCN stem (relative to _fixtures/)
MODEL_LADDER = {
    "small": "Three_Equation_NK",   # ~15 equations
    "rbc": "RBC",                    # ~26 equations
    "medium": "New_Keynesian",       # ~66 equations
    "large": "sims_2024/nk_complete_taxes",  # ~33 equations after reduction; the largest build
}


def gcn_path(name):
    """Resolve a ladder key (or a raw vendored stem) to its `.gcn` path under _fixtures/."""
    return FIXTURES / f"{MODEL_LADDER.get(name, name)}.gcn"


def build_model(name, mode=None):
    """Build a fully compiled `Model` from a ladder key."""
    from gEconpy.model.build import model_from_gcn

    return model_from_gcn(gcn_path(name), verbose=False, mode=mode)


def solved_matrices(name, mode=None):
    """Return the linearization matrices (A, B, C, D) evaluated at the steady state.

    These feed the perturbation-solver benchmarks, which are pure NumPy linear algebra and so
    must be fed already-evaluated matrices rather than a live model.
    """
    model = build_model(name, mode=mode)
    model.steady_state(verbose=False)
    A, B, C, D = model.linearize_model(verbose=False)
    return A, B, C, D


def policy_matrices(name, mode=None):
    """Return the perturbation solution (T, R) for post-solution analysis benchmarks."""
    model = build_model(name, mode=mode)
    model.steady_state(verbose=False)
    T, R = model.solve_model(verbose=False)
    return model, T, R
