# AURA Adaptive Workload Forecasting

A Streamlit dashboard for analyzing an uploaded workload time series, selecting a
forecasting model with workload rules and tabular Q-learning, and evaluating
forecasts with chronological holdouts.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV containing a timestamp column and a numeric workload column. Use the
sidebar to choose the columns, missing-value policy, optional resampling frequency,
aggregation, and seasonal period. The dashboard validates timestamps and values,
sorts observations chronologically, consolidates duplicate timestamps when no
resampling frequency is set, and reports the number of missing values handled.

## Dashboard

The dashboard contains Overview, Dataset, Workload Analysis, Model Selection,
Forecast, Evaluation, Comparison, and Learning History screens. Evaluation reserves
the last selected observations as a test set and fits on earlier observations.
Adaptive and fixed SARIMAX results use the same split and metric calculation.
Reported measures are RMSE, MAE, MaxAE, P95AE, and sMAPE.

The implemented local forecasting actions are `NAIVE_LAST`, `ETS`, `SARIMAX`, and
`SARIMAX_SEASONAL`. All four are exercised on real sample series by the automated
tests, and fit errors or non-finite results are reported instead of being replaced
by another model. SARIMAX models expose a configurable prediction interval; this
implementation does not provide intervals for naive or ETS forecasts.

Optional `TIMEGPT` support uses Nixtla's hosted API; it is not an offline model.
Install it with `python -m pip install -r requirements-timegpt.txt` and set
`NIXTLA_API_KEY` or `TIMEGPT_API_KEY` before starting the app. The API integration
requests actual forecasts and intervals; it is hidden from model selection unless
the package and key are available, and API calls require an explicit session
opt-in because they may incur usage charges. Without the key or package, a
TIMEGPT request reports the missing requirement and does not silently run a local
model instead. Live TIMEGPT forecasts require a valid account key and an available
Nixtla service; offline tests use a mocked client and do not claim live API
availability.

Workload rules classify the observed series as constant, stable, periodic,
trending, bursty, noisy, or uncertain. The tabular Q-table learns a value for the
adaptive model action after each backtest; a cold-start policy follows the
rule-based recommendation and learned ties favor that recommendation. The reward
is `1 / (1 + MAE / mean(abs(training workload)))`, so lower normalized error earns
a larger reward; a failed adaptive forecast receives the explicit `-0.25` fallback
penalty. The Q-table is persisted as JSON at `~/.aura_rl/q_table.json` by default;
set `AURA_Q_TABLE_PATH` to choose another local path. Selection history is kept
for the current Streamlit session.

This repository originally contained only this README, with no existing entry
point, forecasting or reinforcement-learning implementation, tests, or persistence
method. The dashboard and its documented rule/reward defaults are newly
implemented; they are not claims about pre-existing project logic. Processing is
local to the app and uploaded data; no live cloud scaling is connected or implied.

## Tests

```powershell
python -m pytest -q
```
