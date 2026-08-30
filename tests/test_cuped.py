import numpy as np
import pandas as pd

from causalx.simulate import simulate_experiment
from causalx.estimators.cuped import cuped


def test_point_estimate_close_to_true_ate():
    df = simulate_experiment(n_units=200_000, true_ate=2.0, pre_period_corr=0.6, seed=1)
    result = cuped(df)
    assert abs(result["point_estimate"] - 2.0) < 0.05


def test_variance_reduction_meets_design_target():
    """Design doc success metric: CUPED variance reduction >= 30% vs naive,
    when pre-period correlation is reasonably strong (0.6, our default)."""
    df = simulate_experiment(n_units=200_000, true_ate=2.0, pre_period_corr=0.6, seed=1)
    result = cuped(df)
    assert result["variance_reduction_pct"] >= 30.0


def test_no_reduction_when_covariate_genuinely_unrelated_to_outcome():
    """Sanity check with hand-built data: if the covariate has NO relationship
    to outcome whatsoever, CUPED's theta should be ~0 and variance reduction
    should be ~0. Built directly (not via simulate_experiment) because that
    simulator's pre_period_corr knob controls covariate-vs-covariate_x
    correlation, not covariate-vs-outcome correlation — pre_period_metric
    always feeds into outcome directly regardless of that knob."""
    rng = np.random.default_rng(0)
    n = 200_000
    treatment = rng.binomial(1, 0.5, n)
    outcome = 5.0 + treatment * 2.0 + rng.normal(0, 1, n)
    unrelated_covariate = rng.normal(0, 1, n)  # pure noise, no link to outcome at all

    df = pd.DataFrame({
        "outcome": outcome,
        "pre_period_metric": unrelated_covariate,
        "treatment": treatment,
    })

    result = cuped(df)
    assert result["variance_reduction_pct"] < 5.0
