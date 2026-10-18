"""Cross-era core benchmark spanning the 1.x (numba) -> 2.x (pytensor) rewrite.

Build + steady state + solve is the one operation that exists in every era of gEconpy, so it's the
only thing that can be tracked straight through the v2.0 crossover. Two era-specific details make it
work across the whole range:

- The API differs: 1.x constructs a model via the ``gEconModel`` class, 2.x via the
  ``model_from_gcn`` function. ``_build`` detects which is installed and calls the right one.
- The GCN DSL differs: 1.x rejects the 2.x ``maxent(...)`` prior syntax, so this uses the prior-free
  ``RBC_noprior.gcn`` fixture (priors don't affect build/steady_state/solve anyway).

The benchmark name is fixed (no params), so asv stitches every commit's result into one continuous
series, and the crossover shows up as a step in that line.
"""

from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "_fixtures" / "RBC_noprior.gcn"


ERA_2X_MODULES = {"gEconpy.model", "gEconpy.model.build"}


def _build(path):
    try:
        from gEconpy.model.build import model_from_gcn  # 2.x
    except ModuleNotFoundError as error:
        # Fall back only when the 2.x module itself is absent; a missing dependency inside 2.x must surface.
        if error.name not in ERA_2X_MODULES:
            raise
        from gEconpy.classes import gEconModel  # 1.x

        return gEconModel(str(path), verbose=False)
    return model_from_gcn(str(path), verbose=False)


def _build_solve(path):
    model = _build(path)
    model.steady_state(verbose=False)
    model.solve_model(verbose=False)
    return model


class Crossover:
    timeout = 900

    def setup(self):
        # Build once and warm the steady-state + solve so the per-phase timings below measure the
        # phase itself, not first-call compilation. Both eras expose steady_state()/solve_model().
        self.model = _build(FIXTURE)
        self.model.steady_state(verbose=False)
        self.model.solve_model(verbose=False)

    def time_build(self):
        """Construct the model from the GCN (numba lambdify in 1.x, pytensor graph build in 2.x)."""
        _build(FIXTURE)

    def time_steady_state(self):
        self.model.steady_state(verbose=False)

    def time_solve(self):
        self.model.solve_model(verbose=False)

    def time_build_solve(self):
        """The combined cold path — kept for continuity with the existing history."""
        _build_solve(FIXTURE)

    def peakmem_build_solve(self):
        _build_solve(FIXTURE)
