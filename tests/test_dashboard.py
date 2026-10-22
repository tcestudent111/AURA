import json

import numpy as np
import pandas as pd
import pytest

from aura_dashboard.data import (
    classify_workload,
    extract_features,
    prepare_series,
    recommend_model,
)
from aura_dashboard.learning import FALLBACK_PENALTY, QTable, calculate_reward
import aura_dashboard.models as forecasting
from aura_dashboard.models import (
    LOCAL_MODEL_NAMES,
    MODEL_NAMES,
    ForecastError,
    calculate_metrics,
    forecast_series,
)


def test_prepare_series_sorts_parses_fills_and_resamples() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-02", "2024-01-01", "2024-01-03"],
            "load": [3.0, None, 5.0],
        }
    )
    prepared = prepare_series(
        frame, "timestamp", "load", "interpolate", resample_frequency="D"
    )
    assert prepared.values.index.is_monotonic_increasing
    assert prepared.values.tolist() == [3.0, 3.0, 5.0]
    assert prepared.handled_missing == 1


def test_prepare_series_reports_invalid_timestamps() -> None:
    frame = pd.DataFrame({"timestamp": ["not a date"], "load": [1]})
    with pytest.raises(ValueError, match="timestamp value"):
        prepare_series(frame, "timestamp", "load", "drop")


def test_features_and_classification_identify_constant_workload() -> None:
    series = pd.Series(np.full(20, 4.0))
    features = extract_features(series, seasonal_period=4)
    assert features["mean"] == 4.0
    assert features["standard_deviation"] == 0.0
    assert classify_workload(features) == "constant"
    assert recommend_model("constant") == "NAIVE_LAST"


@pytest.mark.parametrize(
    ("category", "features"),
    [
        (
            "stable",
            {"mean": 10, "standard_deviation": 1, "coefficient_of_variation": 0.1,
             "lag_one_autocorrelation": 0.7, "trend_strength": 0.2,
             "seasonal_strength": np.nan, "robust_outlier_ratio": 0.0},
        ),
        (
            "periodic",
            {"mean": 10, "standard_deviation": 3, "coefficient_of_variation": 0.3,
             "lag_one_autocorrelation": 0.4, "trend_strength": 0.2,
             "seasonal_strength": 0.8, "robust_outlier_ratio": 0.0},
        ),
        (
            "trending",
            {"mean": 10, "standard_deviation": 3, "coefficient_of_variation": 0.3,
             "lag_one_autocorrelation": 0.4, "trend_strength": 0.8,
             "seasonal_strength": np.nan, "robust_outlier_ratio": 0.0},
        ),
        (
            "bursty",
            {"mean": 10, "standard_deviation": 3, "coefficient_of_variation": 0.3,
             "lag_one_autocorrelation": 0.4, "trend_strength": 0.2,
             "seasonal_strength": np.nan, "robust_outlier_ratio": 0.1},
        ),
        (
            "noisy",
            {"mean": 10, "standard_deviation": 6, "coefficient_of_variation": 0.6,
             "lag_one_autocorrelation": 0.0, "trend_strength": 0.2,
             "seasonal_strength": np.nan, "robust_outlier_ratio": 0.0},
        ),
        (
            "uncertain",
            {"mean": 10, "standard_deviation": 3, "coefficient_of_variation": 0.3,
             "lag_one_autocorrelation": 0.5, "trend_strength": 0.2,
             "seasonal_strength": np.nan, "robust_outlier_ratio": 0.0},
        ),
    ],
)
def test_rule_classifier_supports_each_nonconstant_category(category, features) -> None:
    assert classify_workload(features) == category


def test_periodic_workload_features_when_two_cycles_are_available() -> None:
    series = pd.Series(np.tile([1.0, 4.0, 2.0, 7.0], 8))
    features = extract_features(series, seasonal_period=4)
    assert np.isfinite(features["seasonal_strength"])
    assert classify_workload(features) == "periodic"
    assert recommend_model("periodic") == "SARIMAX_SEASONAL"


def test_metrics_and_reward_are_calculated_from_actual_values() -> None:
    metrics = calculate_metrics(pd.Series([1.0, 2.0]), pd.Series([2.0, 2.0]))
    assert metrics == {
        "RMSE": pytest.approx(1 / np.sqrt(2)),
        "MAE": 0.5,
        "MaxAE": 1.0,
        "P95AE": 0.95,
        "sMAPE": pytest.approx(100 / 3),
    }
    assert calculate_reward(metrics, workload_scale=2) == pytest.approx(0.8)
    assert calculate_reward(None, workload_scale=2, fallback=True) == FALLBACK_PENALTY


def test_q_table_selects_recommendation_then_learns_and_persists(tmp_path) -> None:
    table = QTable()
    assert table.select("periodic") == "SARIMAX_SEASONAL"
    table.update("periodic", "ETS", 0.8)
    assert table.get("periodic", "ETS") == pytest.approx(0.16)
    assert table.select("periodic") == "ETS"
    path = tmp_path / "q_table.json"
    table.save(path)
    loaded = QTable.load(path)
    assert loaded.as_dict() == table.as_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_only_implemented_forecasting_models_are_accepted() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="h")
    series = pd.Series(np.linspace(10, 20, len(index)), index=index)
    naive = forecast_series(series, "NAIVE_LAST", horizon=3)
    assert naive.forecast.tolist() == [20.0, 20.0, 20.0]
    assert naive.lower is None
    ets = forecast_series(series, "ETS", horizon=3)
    assert len(ets.forecast) == 3
    assert ets.lower is None
    sarimax = forecast_series(series, "SARIMAX", horizon=3)
    assert len(sarimax.forecast) == 3
    assert sarimax.lower is not None and sarimax.upper is not None
    seasonal = forecast_series(series, "SARIMAX_SEASONAL", horizon=3, seasonal_period=4)
    assert len(seasonal.forecast) == 3
    assert seasonal.lower is not None and seasonal.upper is not None
    assert set(LOCAL_MODEL_NAMES) == {"NAIVE_LAST", "ETS", "SARIMAX", "SARIMAX_SEASONAL"}
    assert set(MODEL_NAMES) == {*LOCAL_MODEL_NAMES, "TIMEGPT"}
    with pytest.raises(ForecastError, match="Unsupported model"):
        forecast_series(series, "ARIMA", horizon=3)


def test_seasonal_model_validates_number_of_observations() -> None:
    series = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ForecastError, match="two complete seasons"):
        forecast_series(series, "SARIMAX_SEASONAL", horizon=2, seasonal_period=6)


def test_seasonal_sarimax_forecasts_clean_periodic_data_without_fallback() -> None:
    values = 10 + 2 * np.sin(np.arange(48) * 2 * np.pi / 12)
    series = pd.Series(
        values, index=pd.date_range("2025-01-01", periods=len(values), freq="h")
    )
    result = forecast_series(
        series, "SARIMAX_SEASONAL", horizon=4, seasonal_period=12
    )
    assert np.isfinite(result.forecast.to_numpy()).all()
    assert np.isfinite(result.lower.to_numpy()).all()
    assert np.isfinite(result.upper.to_numpy()).all()
    assert result.forecast.iloc[0] == pytest.approx(series.iloc[-12])


def test_timegpt_requires_credentials_without_silent_local_fallback(monkeypatch) -> None:
    monkeypatch.delenv("NIXTLA_API_KEY", raising=False)
    monkeypatch.delenv("TIMEGPT_API_KEY", raising=False)
    series = pd.Series(
        np.arange(12, dtype=float),
        index=pd.date_range("2024-01-01", periods=12, freq="h"),
    )
    with pytest.raises(ForecastError, match="requires NIXTLA_API_KEY.*no local fallback"):
        forecast_series(series, "TIMEGPT", horizon=2)


def test_timegpt_is_only_an_available_action_when_explicitly_enabled(monkeypatch) -> None:
    pytest.importorskip("nixtla")
    monkeypatch.setenv("NIXTLA_API_KEY", "test-key")
    assert forecasting.available_model_names() == LOCAL_MODEL_NAMES
    assert forecasting.available_model_names(include_timegpt=True) == MODEL_NAMES


def test_timegpt_formats_request_and_returns_real_interval_columns(monkeypatch) -> None:
    pytest.importorskip("nixtla")
    monkeypatch.setenv("NIXTLA_API_KEY", "test-key")
    requested = {}
    future = pd.date_range("2024-01-01 12:00", periods=2, freq="h")

    class FakeClient:
        def forecast(self, **kwargs):
            requested.update(kwargs)
            return pd.DataFrame(
                {
                    "ds": future,
                    "TimeGPT": [12.5, 13.5],
                    "TimeGPT-lo-95": [11.0, 12.0],
                    "TimeGPT-hi-95": [14.0, 15.0],
                }
            )

    monkeypatch.setattr(forecasting, "_create_timegpt_client", lambda: FakeClient())
    history = pd.Series(
        np.arange(12, dtype=float),
        index=pd.date_range("2024-01-01", periods=12, freq="h"),
    )
    result = forecast_series(history, "TIMEGPT", horizon=2, confidence_level=0.95)
    assert result.model_name == "TIMEGPT"
    assert result.forecast.tolist() == [12.5, 13.5]
    assert result.lower.tolist() == [11.0, 12.0]
    assert result.upper.tolist() == [14.0, 15.0]
    assert requested["h"] == 2
    assert requested["freq"] == "h"
    assert requested["level"] == [95]


def test_streamlit_dataset_upload_validates_and_displays_series(tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AURA_Q_TABLE_PATH", str(tmp_path / "q-table.json"))
    csv = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=12, freq="h"),
            "load": np.arange(12, dtype=float),
        }
    ).to_csv(index=False).encode()
    app = AppTest.from_file("app.py").run(timeout=30)
    app.selectbox[0].set_value("Dataset")
    app.file_uploader[0].set_value(("workload.csv", csv, "text/csv"))
    app.run(timeout=30)
    assert not app.exception
    assert any("Validated 12 chronological observations" in item.value for item in app.success)
    app.selectbox[0].set_value("Workload Analysis")
    app.run(timeout=30)
    assert not app.exception
    app.selectbox[0].set_value("Evaluation")
    app.run(timeout=30)
    app.number_input[-1].set_value(3)
    app.button[0].click()
    app.run(timeout=120)
    assert not app.exception
    assert any("Adaptive model reward" in item.label for item in app.metric)
    assert (tmp_path / "q-table.json").exists()
