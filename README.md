# causalx — Causal Experimentation Platform

A causal inference toolkit for estimating treatment effects from randomized experiments, built with a simulation-first approach: every estimator is validated against data with a **known, controllable ground truth** before being trusted.

**Live dashboard:** run locally with `uv run streamlit run dashboard/app.py` (public deployment coming — see [Roadmap](#roadmap)).

---

## Why simulation-first?

You can never observe a real experiment's true causal effect — only estimate it. So before trusting any estimator on real data, this project validates every one of them against a synthetic data-generating process where the true effect is known in advance. If an estimator can't recover a known truth in a world we fully control, it has no business being trusted on real data where the truth is unknown.

## What's implemented

| Component | Description | Status |
|---|---|---|
| **Simulator** | RCT data generator with controllable ATE, pre-period correlation, and effect heterogeneity | ✅ |
| **Diff-in-means** | Baseline estimator — point estimate, standard error, 95% CI | ✅ |
| **CUPED** | Variance-reduction via pre-treatment covariate adjustment | ✅ |
| **DR Learner** | Heterogeneous treatment effects (CATE) via EconML, doubly-robust | ✅ |
| **Dashboard** | Live Streamlit scorecard, dark-terminal themed | ✅ |
| mSPRT sequential testing | Always-valid p-values under continuous monitoring | ⬜ planned |
| Power / sample-size calculator | | ⬜ planned |
| FastAPI service | `/estimate`, `/power` endpoints | ⬜ planned |
| Docker + CI | Containerization, GitHub Actions test pipeline | ⬜ planned |
| Public deployment | Hugging Face Spaces | ⬜ planned |

## Results

All results below are reproducible — see [Validation](#validation) for exact commands.

### Diff-in-means (baseline)

On 200,000 simulated units with a true ATE of 2.0:

| Metric | Value |
|---|---|
| Point estimate | 1.9885 |
| 95% CI | [1.9701, 2.0068] |
| Empirical CI coverage (200 independent trials) | ~95% |

### CUPED (variance reduction)

Same dataset, using a pre-treatment covariate correlated at ρ=0.6:

| Metric | Diff-in-means | CUPED |
|---|---|---|
| Standard error | 0.0094 | 0.0048 |
| **Variance reduction** | — | **73.7%** (target: ≥30%) |

### DR Learner (heterogeneous effects)

Validated on a hand-built dataset with a known effect that varies linearly with a covariate (true effect: 0.0 at low covariate values, 4.0 at high covariate values):

| | True effect | DR Learner estimate |
|---|---|---|
| At low covariate value | 0.0 | -0.04 |
| At high covariate value | 4.0 | 4.05 |

DR Learner correctly recovers both the direction and magnitude of effect heterogeneity, not just the population average.

## Architecture

```
Experiment logs
      │
      ▼
Ingestion & validation (schema, leakage checks)
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
Diff-in-means      CUPED       DR Learner
 (baseline)   (variance-reduced)  (CATE)
      │              │              │
      └──────────────┴──────────────┘
                     │
                     ▼
          Streamlit dashboard
```

## Project structure

```
src/causalx/
├── simulate.py                    # RCT simulator with known ground truth
└── estimators/
    ├── diff_in_means.py
    ├── cuped.py
    └── dr_learner.py

dashboard/
└── app.py                         # Live Streamlit scorecard

tests/                             # 11 tests, all estimators validated
├── test_simulate.py
├── test_diff_in_means.py
├── test_cuped.py
└── test_dr_learner.py
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
git clone https://github.com/Dheeraj1412/causal-experimentation-platform.git
cd causal-experimentation-platform
uv sync
```

## Validation

Run the full test suite:

```bash
uv run pytest -v
```

Run the dashboard:

```bash
uv run streamlit run dashboard/app.py
```

## Validity assumptions

Diff-in-means and CUPED require **proper randomization** — treatment assignment independent of potential outcomes. This was deliberately verified: breaking randomization in the simulator (making treatment depend on a covariate rather than a coin flip) causes diff-in-means to recover a biased estimate (4.22 instead of the true 2.0) that does not improve with more data — confirming these estimators fail exactly as theory predicts under confounding, and correctly succeed under proper randomization.

DR Learner is doubly robust: it remains unbiased if either its propensity model or its outcome model is approximately correct, making it more resilient to imperfect randomization in real-world (non-experimental) data.

## Roadmap

- mSPRT sequential testing for always-valid inference under continuous monitoring
- Power/sample-size calculator
- FastAPI service exposing all estimators via `/estimate` and `/power`
- Docker containerization + GitHub Actions CI
- Public deployment via Hugging Face Spaces
- Published package on TestPyPI

## Tech stack

Python 3.11 · pandas · numpy · [EconML](https://github.com/py-why/EconML) · scikit-learn · Streamlit · pytest · uv
