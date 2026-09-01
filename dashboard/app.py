import numpy as np
import streamlit as st
from causalx.simulate import simulate_experiment
from causalx.estimators.diff_in_means import diff_in_means
from causalx.estimators.cuped import cuped
from causalx.estimators.dr_learner import dr_learner

st.set_page_config(page_title="Causal Experimentation Platform", layout="wide")


@st.cache_data
def get_data():
    return simulate_experiment(n_units=200_000, true_ate=2.0, pre_period_corr=0.6, seed=1)


@st.cache_data
def get_naive_result(df):
    return diff_in_means(df)


@st.cache_data
def get_cuped_result(df):
    return cuped(df)


@st.cache_resource
def get_dr_result(df):
    return dr_learner(df)


st.markdown("### experiment scorecard")
st.caption("causalx — diff-in-means · CUPED · DR learner")

df = get_data()
naive = get_naive_result(df)
adjusted = get_cuped_result(df)

st.markdown("#### diff-in-means (naive)")
col1, col2, col3 = st.columns(3)
col1.metric("point estimate", f"{naive['point_estimate']:.4f}")
col2.metric("standard error", f"{naive['standard_error']:.4f}")
col3.metric("95% CI", f"[{naive['ci_lower']:.2f}, {naive['ci_upper']:.2f}]")

st.markdown("#### CUPED (variance-reduced)")
col4, col5, col6 = st.columns(3)
col4.metric("point estimate", f"{adjusted['point_estimate']:.4f}")
col5.metric("standard error", f"{adjusted['standard_error']:.4f}")
col6.metric(
    "variance reduction",
    f"{adjusted['variance_reduction_pct']:.1f}%",
    delta=f"{adjusted['standard_error'] - naive['standard_error']:.4f} SE",
    delta_color="inverse",
)

st.markdown("#### DR learner (CATE at two covariate values)")
st.caption(
    "This scorecard uses a constant-effect RCT (true ATE = 2.0), so CATE at "
    "x = ±2 should both be ~2. Heterogeneity recovery is validated separately "
    "in tests/test_dr_learner.py against a DGP where the true effect is 0 at "
    "x = -2 and 4 at x = +2."
)

dr_result = get_dr_result(df)
model = dr_result["model"]
effect_low = model.effect(np.array([[-2.0]]))[0]
effect_high = model.effect(np.array([[2.0]]))[0]

col7, col8, col9 = st.columns(3)
col7.metric("average effect (ATE)", f"{dr_result['average_effect']:.4f}")
col8.metric("effect at x = -2", f"{effect_low:.4f}")
col9.metric("effect at x = +2", f"{effect_high:.4f}")
