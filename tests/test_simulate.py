import pandas as pd
from causalx.simulate import simulate_experiment


def test_recovers_known_ate_at_large_n():
    df = simulate_experiment(n_units=200_000, true_ate=2.0, seed=1)

    naive_ate = (
        df.loc[df.treatment == 1, "outcome"].mean()
        - df.loc[df.treatment == 0, "outcome"].mean()
    )

    assert abs(naive_ate - 2.0) < 0.05


def test_covariates_balanced_across_treatment_arms():
    df = simulate_experiment(n_units=50_000, seed=3)

    treated = df[df.treatment == 1]
    control = df[df.treatment == 0]

    for col in ["pre_period_metric", "covariate_x"]:
        diff = treated[col].mean() - control[col].mean()
        se = (treated[col].var() / len(treated) + control[col].var() / len(control)) ** 0.5
        standardized_diff = abs(diff / se)
        assert standardized_diff < 3.0, f"{col} imbalanced: {standardized_diff:.2f} SEs apart"


def test_reproducibility_same_seed_same_data():
    df1 = simulate_experiment(seed=42)
    df2 = simulate_experiment(seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_output_shape_and_columns():
    n = 500
    df = simulate_experiment(n_units=n)

    assert len(df) == n

    expected_cols = {
        "user_id", "pre_period_metric", "covariate_x",
        "treatment", "outcome", "true_individual_effect",
    }
    assert set(df.columns) == expected_cols
