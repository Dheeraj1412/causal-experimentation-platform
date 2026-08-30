import numpy as np
import pandas as pd

Z_95 = 1.96


def diff_in_means(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
) -> dict:
    """
    Estimate the average treatment effect via a naive difference of group means.
    Valid ONLY when treatment assignment is independent of potential outcomes
    (i.e. a properly randomized experiment) — see simulate.py's docstring.

    Returns:
        dict with point_estimate, standard_error, ci_lower, ci_upper, n_treated, n_control.
    """
    treated = df.loc[df[treatment_col] == 1, outcome_col]
    control = df.loc[df[treatment_col] == 0, outcome_col]

    point_estimate = treated.mean() - control.mean()
    se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

    return {
        "point_estimate": point_estimate,
        "standard_error": se,
        "ci_lower": point_estimate - Z_95 * se,
        "ci_upper": point_estimate + Z_95 * se,
        "n_treated": len(treated),
        "n_control": len(control),
    }
