import numpy as np
import pandas as pd

from causalx.estimators.dr_learner import dr_learner


def _simulate_heterogeneous_effect(seed: int = 1, n: int = 20_000) -> pd.DataFrame:
    """Hand-built dataset (not simulate_experiment) where the treatment
    effect genuinely depends on covariate_x: ~0 at x=-2, ~4 at x=+2.
    Built directly so we control ground truth precisely, the same lesson
    learned from the CUPED test bug earlier."""
    rng = np.random.default_rng(seed)
    covariate_x = rng.normal(0, 1, n)
    treatment = rng.binomial(1, 0.5, n)
    true_individual_effect = 2.0 + 1.0 * covariate_x
    outcome = (
        5.0 + covariate_x * 0.5 + treatment * true_individual_effect
        + rng.normal(0, 1, n)
    )
    return pd.DataFrame({"outcome": outcome, "treatment": treatment, "covariate_x": covariate_x})


def test_average_effect_close_to_true_ate():
    df = _simulate_heterogeneous_effect(seed=1)
    result = dr_learner(df)
    assert abs(result["average_effect"] - 2.0) < 0.2


def test_detects_genuine_effect_heterogeneity():
    """The core claim of DR Learner: effect at high covariate_x should be
    meaningfully larger than effect at low covariate_x, matching the
    known true relationship (effect = 2.0 + 1.0 * covariate_x)."""
    df = _simulate_heterogeneous_effect(seed=1)
    result = dr_learner(df)
    model = result["model"]

    effect_low = model.effect(np.array([[-2.0]]))[0]
    effect_high = model.effect(np.array([[2.0]]))[0]

    assert abs(effect_low - 0.0) < 0.3
    assert abs(effect_high - 4.0) < 0.3
    assert effect_high > effect_low
