import numpy as np
import pandas as pd

from causalx.simulate import simulate_experiment
from causalx.estimators.diff_in_means import diff_in_means


def test_point_estimate_close_to_true_ate():
    df = simulate_experiment(n_units=200_000, true_ate=2.0, seed=1)
    result = diff_in_means(df)
    assert abs(result["point_estimate"] - 2.0) < 0.05


def test_ci_contains_true_ate_at_declared_rate():
    """The real promise of a 95% CI: across many independent experiments,
    roughly 95% of the resulting intervals should contain the true effect.
    We simulate 200 independent experiments and count."""
    n_trials = 200
    true_ate = 2.0
    hits = 0

    for seed in range(n_trials):
        df = simulate_experiment(n_units=5_000, true_ate=true_ate, seed=seed)
        result = diff_in_means(df)
        if result["ci_lower"] <= true_ate <= result["ci_upper"]:
            hits += 1

    coverage = hits / n_trials
    assert 0.90 <= coverage <= 1.0, f"coverage was {coverage:.3f}, expected ~0.95"


def test_confounding_biases_estimate_and_does_not_vanish_with_n():
    """If treatment is assigned from a covariate instead of a coin flip,
    diff-in-means recovers a biased ATE (~4.22 vs true 2.0 at n=200k) that
    does not disappear at larger n. Same data-generating process as
    simulate_experiment, except treatment = 1{covariate_x > 0}."""

    def confounded(n_units: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        covariate_x = rng.normal(0, 1, n_units)
        pre_period_corr = 0.6
        noise = rng.normal(0, np.sqrt(1 - pre_period_corr**2), n_units)
        pre_period_metric = covariate_x * pre_period_corr + noise
        treatment = (covariate_x > 0).astype(int)
        individual_effect = np.full(n_units, 2.0)
        baseline_outcome = pre_period_metric * 1.5 + covariate_x * 0.5
        outcome = (
            baseline_outcome
            + treatment * individual_effect
            + rng.normal(0, 1, n_units)
        )
        return pd.DataFrame({"treatment": treatment, "outcome": outcome})

    small = diff_in_means(confounded(10_000, seed=1))
    large = diff_in_means(confounded(200_000, seed=1))

    assert abs(large["point_estimate"] - 4.22) < 0.05
    assert abs(small["point_estimate"] - 2.0) > 1.0
    assert abs(large["point_estimate"] - 2.0) > 1.0
