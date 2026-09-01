import numpy as np

from causalx.simulate import simulate_experiment
from causalx.estimators.diff_in_means import diff_in_means
from causalx.testing.sequential import msprt_p_value, effective_variance


def test_p_value_is_one_at_zero_effect():
    p = msprt_p_value(theta_hat=0.0, n=1000, sigma_sq=1.0)
    assert p == 1.0


def test_p_value_shrinks_as_evidence_accumulates():
    """For a fixed, nonzero observed effect, p should strictly decrease as
    n grows -- more data confirming the same effect is stronger evidence."""
    theta_hat = 0.5
    p_values = [
        msprt_p_value(theta_hat=theta_hat, n=n, sigma_sq=1.0)
        for n in [10, 100, 1000]
    ]
    assert p_values[0] > p_values[1] > p_values[2]


def test_effective_variance_identity():
    """n * se^2 should exactly invert to give back se when run through
    the standard SE formula -- sanity-checks the algebraic identity."""
    se = 0.3
    n = 500
    sigma_sq = effective_variance(se, n)
    recovered_se = np.sqrt(sigma_sq / n)
    assert abs(recovered_se - se) < 1e-10


def test_controls_false_positive_rate_under_repeated_peeking():
    """The core claim of mSPRT: checking results repeatedly as data
    accumulates should NOT inflate the false-positive rate above alpha,
    unlike a standard confidence interval (which we separately verified
    gives ~25% false positives under the same peeking pattern)."""
    n_trials = 300
    alpha = 0.05
    false_positives = 0

    for trial_seed in range(n_trials):
        df = simulate_experiment(n_units=2000, true_ate=0.0, seed=trial_seed)

        for checkpoint in range(100, 2001, 100):
            interim = df.iloc[:checkpoint]
            result = diff_in_means(interim)
            sigma_sq = effective_variance(result["standard_error"], n=checkpoint)
            p = msprt_p_value(result["point_estimate"], n=checkpoint, sigma_sq=sigma_sq)
            if p < alpha:
                false_positives += 1
                break

    rate = false_positives / n_trials
    assert rate <= alpha + 0.02, f"false positive rate {rate:.3f} exceeds alpha + tolerance"
