import numpy as np

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
