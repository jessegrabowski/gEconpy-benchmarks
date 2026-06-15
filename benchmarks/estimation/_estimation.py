"""Helpers for the estimation benchmarks.

The packaged RBC model carries priors on its deep parameters but not on its single shock's standard
deviation (``sigma_epsilon_A``), so that prior is added explicitly, matching the estimation example.
A single observed series against a single shock keeps the system non-singular without measurement
error. Synthetic Gaussian data is sufficient: the Kalman filter runs its full forward pass
regardless of the data's likelihood, so timings do not depend on the data being model-consistent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Vendored so the model input stays constant across gEconpy's history (see benchmarks/_support.py).
RBC_GCN = Path(__file__).resolve().parent.parent / "_fixtures" / "RBC.gcn"
OBSERVED = ["Y"]


def make_configured_statespace(solver="gensys", mode=None, observed=OBSERVED):
    import gEconpy as ge

    ss_mod = ge.statespace_from_gcn(RBC_GCN, verbose=False)
    ss_mod.configure(observed_states=list(observed), solver=solver, mode=mode, verbose=False)
    return ss_mod


def synthetic_data(observed=OBSERVED, n_obs=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_obs, len(observed))),
        columns=list(observed),
        index=pd.date_range("2000-01-01", periods=n_obs, freq="QS"),
    )


def build_pymc_model(ss_mod, data):
    import pymc as pm

    with pm.Model(coords=ss_mod.coords) as model:
        ss_mod.to_pymc()
        pm.Gamma("sigma_epsilon_A", alpha=2, beta=20)
        ss_mod.build_statespace_graph(data, add_norm_check=False)
    return model
