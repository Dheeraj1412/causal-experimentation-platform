import numpy as np


def effective_variance(standard_error: float, n: int) -> float:
    """
    Convert a standard error (from an estimator like diff_in_means) into the
    "effective per-unit variance" that msprt_p_value expects.

    msprt_p_value's model assumes theta_hat ~ Normal(theta, sigma_sq / n) --
    i.e. a single per-unit variance averaged down by sample size n. Solving
    that relationship for sigma_sq given an already-computed standard_error
    (SE = sqrt(sigma_sq / n)) gives sigma_sq = n * SE^2 exactly -- this is
    an algebraic identity, not an approximation. The only imprecision here
    is the same as everywhere in statistics: standard_error is itself
    estimated from finite sample data, not the true unknowable population SE.

    Args:
        standard_error: SE from an estimator, e.g. diff_in_means()["standard_error"].
        n: the sample size that standard_error was computed from.

    Returns:
        The effective per-unit variance, suitable as msprt_p_value's sigma_sq argument.
    """
    return n * standard_error**2


def msprt_p_value(theta_hat: float, n: int, sigma_sq: float, tau_sq: float = 1.0) -> float:
    """
    Always-valid p-value via mixture Sequential Probability Ratio Test (mSPRT).

    Unlike a standard p-value, this remains valid (false-positive rate stays
    bounded at alpha) even if checked repeatedly as data accumulates, because
    the underlying likelihood ratio is a martingale under the null hypothesis.

    Computed in log-space for numerical stability. log(lambda_n) is a sum of
    two terms: a strictly negative term (independent of theta_hat) and a
    non-negative term (proportional to theta_hat^2). When theta_hat is small,
    log(lambda_n) <= 0, which means the p-value is exactly 1.0 by construction
    -- we return early in that case, both as a minor optimization and to
    guarantee np.exp() is never called on a large positive number.

    Args:
        theta_hat: current point estimate of the effect (e.g. from diff-in-means).
        n: number of observations the estimate is based on.
        sigma_sq: effective per-unit outcome variance. Use effective_variance()
            to derive this from an estimator's reported standard_error.
        tau_sq: variance of the normal mixture prior over possible true effect
            sizes — reflects how large an effect we plausibly expect. Larger
            tau_sq = more sensitive to big effects, less sensitive to tiny ones.

    Returns:
        An always-valid p-value in (0, 1]. Reject the null when this drops below alpha.
    """
    denom = sigma_sq + n * tau_sq

    log_lambda_n = 0.5 * np.log(sigma_sq / denom) + (
        (n**2 * tau_sq * theta_hat**2) / (2 * sigma_sq * denom)
    )

    if log_lambda_n <= 0:
        return 1.0

    return np.exp(-log_lambda_n)
