"""Balanced-growth-path state-space fixtures for the feature benchmarks.

The model (``_fixtures/RBC_with_growth.gcn``) carries a permanent productivity trend ``G[]``, six
structural shocks, and a net risk-free rate ``R_f`` that is naturally a level rather than a log
deviation. That makes it the right vehicle for exercising the three estimation features benchmarked
here: growth-rate observation equations built off the trend, temporal aggregation of a quarterly
growth rate to an annual observation, and a mixed log-linearisation that keeps ``R_f`` in levels.

Everything runs through the conda-forge-portable gensys solver in the default C backend (the example
notebook uses JAX, which is not in the benchmark environment). Shock and measurement-error standard
deviations get generic priors; ``constant_params="auto"`` fixes the trend-scaling parameter that has
no GCN prior. Synthetic Gaussian data is sufficient — the Kalman filter runs its full pass regardless
of whether the data is model-consistent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BGP_GCN = Path(__file__).resolve().parent.parent / "_fixtures" / "RBC_with_growth.gcn"
SEED = 0


def _quarterly(columns, n=64):
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    values = np.random.default_rng(SEED).normal(scale=0.02, size=(n, len(columns)))
    return pd.DataFrame(values, index=idx, columns=columns)


def _annual_in_quarterly(column, n=64):
    """Quarterly index with an observation only every fourth quarter (a low-frequency series)."""
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    data = pd.DataFrame(np.nan, index=idx, columns=[column])
    data.iloc[3::4, 0] = np.random.default_rng(SEED).normal(scale=0.02, size=n // 4)
    return data


def _build(
    observed_states,
    data,
    observation_equations=None,
    temporal_aggregation=None,
    aggregation_period=4,
    not_loglin_variables=None,
    ss_obs_intercept=None,
):
    import pymc as pm

    import gEconpy as ge

    ss_mod = ge.statespace_from_gcn(BGP_GCN, not_loglin_variables=not_loglin_variables, verbose=False)
    ss_mod.configure(
        observed_states=observed_states,
        observation_equations=observation_equations,
        temporal_aggregation=temporal_aggregation,
        aggregation_period=aggregation_period,
        ss_obs_intercept=ss_obs_intercept,
        measurement_error=observed_states,
        constant_params="auto",
        solver="gensys",
        verbose=False,
    )
    with pm.Model(coords=ss_mod.coords) as model:
        ss_mod.to_pymc()
        for shock in ss_mod.shock_names:
            pm.Gamma(f"sigma_{shock}", alpha=2, beta=100)
        for obs in ss_mod.error_states:
            pm.Gamma(f"error_sigma_{obs}", alpha=2, beta=100)
        ss_mod.build_statespace_graph(data, add_norm_check=False)
    return model


def observation_equations_model():
    """Three growth-rate / log-level observation equations built off the trend ``G[]``."""
    obs_eqs = {
        "dlog_Y": "log(Y[] / Y[-1] * G[])",
        "dlog_C": "log(C[] / C[-1] * G[])",
        "log_L": "log(L[])",
    }
    return _build(list(obs_eqs), _quarterly(list(obs_eqs)), observation_equations=obs_eqs)


def frequency_change_model():
    """Build a model whose quarterly growth rate is observed as an annual sum (mixed-frequency)."""
    obs_eqs = {"dlog_Y_annual": "log(Y[] / Y[-1] * G[])"}
    return _build(
        ["dlog_Y_annual"],
        _annual_in_quarterly("dlog_Y_annual"),
        observation_equations=obs_eqs,
        temporal_aggregation={"dlog_Y_annual": "sum"},
        aggregation_period=4,
    )


def mixed_loglin_model():
    """Build a model that keeps the net rate ``R_f`` in levels while log-linearizing a level observable."""
    obs_eqs = {"log_Y": "log(Y[])"}
    return _build(
        ["R_f", "log_Y"],
        _quarterly(["R_f", "log_Y"]),
        observation_equations=obs_eqs,
        not_loglin_variables=["R_f"],
        ss_obs_intercept=["R_f"],
    )


def warmed_logp_dlogp(model):
    """Return the fused value-and-gradient function plus a raveled point, warmed once."""
    from pymc.blocking import DictToArrayBijection

    func = model.logp_dlogp_function()
    func.set_extra_values({})
    raveled = DictToArrayBijection.map(model.initial_point())
    func(raveled)
    return func, raveled
