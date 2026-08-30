import numpy as np
import pandas as pd


def simulate_experiment(
    n_units: int = 10_000,
    true_ate: float = 2.0,
    pre_period_corr: float = 0.6,
    heterogeneity_scale: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a randomized experiment with a known ground-truth treatment effect.

    Treatment is assigned by an independent coin flip (rng.binomial), completely
    unrelated to any covariate — this is what makes it a valid RCT. Any bias found
    in a downstream estimator run on this data is an estimator bug, not a data bug.

    Args:
        n_units: number of simulated users.
        true_ate: the average treatment effect baked into the data.
        pre_period_corr: correlation between pre_period_metric and covariate_x.
            Controls how much variance CUPED can remove.
        heterogeneity_scale: spread of individual treatment effects around true_ate.
            0.0 means every unit has exactly the same effect (constant effect).
        seed: random seed, for reproducibility.

    Returns:
        DataFrame with one row per unit: user_id, pre_period_metric, covariate_x,
        treatment, outcome, true_individual_effect (ground truth, for validation only —
        never feed this column into an estimator).
    """
    rng = np.random.default_rng(seed)

    covariate_x = rng.normal(0, 1, n_units)

    noise = rng.normal(0, np.sqrt(1 - pre_period_corr**2), n_units)
    pre_period_metric = covariate_x * pre_period_corr + noise

    treatment = rng.binomial(1, 0.5, n_units)

    individual_effect = true_ate + rng.normal(0, heterogeneity_scale, n_units)

    baseline_outcome = pre_period_metric * 1.5 + covariate_x * 0.5
    outcome = baseline_outcome + treatment * individual_effect + rng.normal(0, 1, n_units)

    return pd.DataFrame({
        "user_id": np.arange(n_units),
        "pre_period_metric": pre_period_metric,
        "covariate_x": covariate_x,
        "treatment": treatment,
        "outcome": outcome,
        "true_individual_effect": individual_effect,
    })
