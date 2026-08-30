import numpy as np
import pandas as pd
from econml.dr import DRLearner
from sklearn.linear_model import LinearRegression, LogisticRegression


def dr_learner(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
    effect_modifier_col: str = "covariate_x",
) -> dict:
    """
    Estimate heterogeneous treatment effects (CATE) using EconML's DR Learner.

    Fits a propensity model (probability of treatment given X) and an outcome
    regression model (outcome given X), then combines them into a doubly-robust
    estimate of how the treatment effect varies with the effect_modifier_col.
    Valid under the same randomization assumption as diff_in_means/cuped.

    Returns:
        dict with average_effect (population ATE) and a fitted `model` object
        that supports model.effect(X) for querying CATE at arbitrary covariate
        values.
    """
    Y = df[outcome_col].values
    T = df[treatment_col].values
    X = df[[effect_modifier_col]].values

    model = DRLearner(
        model_propensity=LogisticRegression(),
        model_regression=LinearRegression(),
        model_final=LinearRegression(),
    )
    model.fit(Y, T, X=X)

    return {
        "average_effect": model.ate(X),
        "model": model,
    }
