import numpy as np
import pandas as pd

Z_95 = 1.96


def cuped(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    covariate_col: str = "pre_period_metric",
    treatment_col: str = "treatment",
) -> dict:
    """
    Estimate the average treatment effect using CUPED variance reduction.

    Subtracts off the portion of the outcome predictable from a pre-treatment
    covariate, since that portion cannot be caused by treatment. theta is
    estimated on the pooled sample (valid because the covariate is pre-treatment,
    so it can't be affected by treatment status).

    Returns:
        dict with point_estimate, standard_error, ci_lower, ci_upper,
        variance_reduction_pct (vs. naive diff-in-means on the same data),
        n_treated, n_control.
    """
    outcome = df[outcome_col]
    covariate = df[covariate_col]
    treatment = df[treatment_col]

    theta = np.cov(outcome, covariate)[0, 1] / np.var(covariate)
    adjusted_outcome = outcome - theta * (covariate - covariate.mean())

    treated_raw = outcome[treatment == 1]
    control_raw = outcome[treatment == 0]
    naive_se = np.sqrt(
        treated_raw.var() / len(treated_raw) + control_raw.var() / len(control_raw)
    )

    treated = adjusted_outcome[treatment == 1]
    control = adjusted_outcome[treatment == 0]

    point_estimate = treated.mean() - control.mean()
    se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

    variance_reduction_pct = 100 * (1 - (se**2 / naive_se**2))

    return {
        "point_estimate": point_estimate,
        "standard_error": se,
        "ci_lower": point_estimate - Z_95 * se,
        "ci_upper": point_estimate + Z_95 * se,
        "variance_reduction_pct": variance_reduction_pct,
        "n_treated": len(treated),
        "n_control": len(control),
    }
