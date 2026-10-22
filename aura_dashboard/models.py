from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


LOCAL_MODEL_NAMES = ("NAIVE_LAST", "ETS", "SARIMAX", "SARIMAX_SEASONAL")
TIMEGPT_MODEL_NAME = "TIMEGPT"
MODEL_NAMES = (*LOCAL_MODEL_NAMES, TIMEGPT_MODEL_NAME)


def timegpt_api_key() -> str | None:
    return os.environ.get("NIXTLA_API_KEY") or os.environ.get("TIMEGPT_API_KEY")


def timegpt_status() -> str:
    if importlib.util.find_spec("nixtla") is None:
        return "TIMEGPT is unavailable: install the optional `nixtla` package."
    if not timegpt_api_key():
        return "TIMEGPT is unavailable: set NIXTLA_API_KEY or TIMEGPT_API_KEY."
    return (
        "TIMEGPT is configured; the Nixtla API service and account still need to be "
        "reachable for forecasts. API calls may incur usage charges."
    )


def available_model_names(include_timegpt: bool = False) -> tuple[str, ...]:
    if include_timegpt and timegpt_api_key() and importlib.util.find_spec("nixtla") is not None:
        return MODEL_NAMES
    return LOCAL_MODEL_NAMES


def _create_timegpt_client():
    key = timegpt_api_key()
    if not key:
        raise ForecastError(
            "TIMEGPT requires NIXTLA_API_KEY or TIMEGPT_API_KEY; no local fallback was run."
        )
    try:
        from nixtla import NixtlaClient
    except ImportError as exc:
        raise ForecastError(
            "TIMEGPT requires the optional `nixtla` package; no local fallback was run."
        ) from exc
    return NixtlaClient(api_key=key)


class ForecastError(ValueError):
    """A selected forecasting model could not be fitted to the supplied series."""


@dataclass(frozen=True)
class ForecastResult:
    model_name: str
    forecast: pd.Series
    lower: pd.Series | None = None
    upper: pd.Series | None = None
    warnings: tuple[str, ...] = ()


def future_index(index: pd.Index, horizon: int) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex):
        frequency = index.freq or pd.infer_freq(index)
        if frequency is not None:
            return pd.date_range(index[-1], periods=horizon + 1, freq=frequency)[1:]
        if len(index) < 2:
            raise ForecastError("A timestamp frequency cannot be inferred from one observation.")
        positive_deltas = np.diff(index.asi8)
        positive_deltas = positive_deltas[positive_deltas > 0]
        if not len(positive_deltas):
            raise ForecastError("A future timestamp interval cannot be inferred.")
        step = int(np.median(positive_deltas))
        return pd.DatetimeIndex(
            [index[-1] + pd.Timedelta(step * offset, unit="ns") for offset in range(1, horizon + 1)]
        )
    return pd.RangeIndex(start=len(index), stop=len(index) + horizon, name="step")


def forecast_series(
    series: pd.Series,
    model_name: str,
    horizon: int,
    seasonal_period: int | None = None,
    confidence_level: float = 0.95,
) -> ForecastResult:
    if model_name not in MODEL_NAMES:
        raise ForecastError(f"Unsupported model: {model_name}")
    if horizon < 1:
        raise ForecastError("Forecast horizon must be at least one.")
    values = series.dropna().astype(float)
    if len(values) < 3 or not np.isfinite(values.to_numpy()).all():
        raise ForecastError("Provide at least three finite workload observations.")
    if not 0 < confidence_level < 1:
        raise ForecastError("Confidence level must be between zero and one.")
    index = future_index(values.index, horizon)

    try:
        if model_name == "NAIVE_LAST":
            forecast = np.repeat(float(values.iloc[-1]), horizon)
            return ForecastResult(model_name, pd.Series(forecast, index=index, name="forecast"))
        if model_name == "ETS":
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = ExponentialSmoothing(
                    values, trend="add", seasonal=None, initialization_method="estimated"
                ).fit(optimized=True)
                forecast = np.asarray(fitted.forecast(horizon), dtype=float)
            model_warnings = tuple(str(item.message) for item in captured)
            if not np.isfinite(forecast).all():
                raise ForecastError("ETS generated non-finite forecast values.")
            return ForecastResult(
                model_name,
                pd.Series(forecast, index=index, name="forecast"),
                warnings=model_warnings,
            )
        if model_name == TIMEGPT_MODEL_NAME:
            if not isinstance(values.index, pd.DatetimeIndex):
                raise ForecastError("TIMEGPT requires a DatetimeIndex; no local fallback was run.")
            frequency = values.index.freq or pd.infer_freq(values.index)
            if frequency is None:
                raise ForecastError(
                    "TIMEGPT requires regularly spaced timestamps. Resample the series first; "
                    "no local fallback was run."
                )
            level = int(round(confidence_level * 100))
            if level not in {80, 90, 95, 99}:
                raise ForecastError(
                    "TIMEGPT confidence level must round to 80, 90, 95, or 99 percent."
                )
            try:
                    import httpx
                    from nixtla.nixtla_client import ApiError

                    client = _create_timegpt_client()
                    response = client.forecast(
                    df=pd.DataFrame({"ds": values.index, "y": values.to_numpy()}),
                    h=horizon,
                    freq=frequency,
                    time_col="ds",
                    target_col="y",
                    level=[level],
                )
            except ForecastError:
                raise
            except (ApiError, httpx.HTTPError, OSError, RuntimeError, ValueError, TypeError) as exc:
                raise ForecastError(f"TIMEGPT API forecast failed: {exc}") from exc
            if not isinstance(response, pd.DataFrame) or "TimeGPT" not in response.columns:
                raise ForecastError("TIMEGPT returned an unexpected forecast response.")
            forecast = pd.to_numeric(response["TimeGPT"], errors="coerce").to_numpy(dtype=float)
            if len(forecast) != horizon or not np.isfinite(forecast).all():
                raise ForecastError("TIMEGPT returned missing, non-finite, or incomplete forecasts.")
            forecast_index = index
            if "ds" in response.columns:
                response_index = pd.DatetimeIndex(pd.to_datetime(response["ds"], errors="coerce"))
                if response_index.isna().any() or len(response_index) != horizon:
                    raise ForecastError("TIMEGPT returned invalid forecast timestamps.")
                forecast_index = response_index
            lower_column = f"TimeGPT-lo-{level}"
            upper_column = f"TimeGPT-hi-{level}"
            if lower_column not in response.columns or upper_column not in response.columns:
                raise ForecastError(
                    f"TIMEGPT did not return the requested {level}% prediction interval."
                )
            lower_values = pd.to_numeric(response[lower_column], errors="coerce").to_numpy(dtype=float)
            upper_values = pd.to_numeric(response[upper_column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(lower_values).all() or not np.isfinite(upper_values).all():
                raise ForecastError("TIMEGPT returned non-finite confidence interval values.")
            return ForecastResult(
                model_name,
                pd.Series(forecast, index=forecast_index, name="forecast"),
                pd.Series(lower_values, index=forecast_index, name="lower"),
                pd.Series(upper_values, index=forecast_index, name="upper"),
            )

        seasonal_order = (0, 0, 0, 0)
        if model_name == "SARIMAX_SEASONAL":
            if seasonal_period is None or seasonal_period < 2:
                raise ForecastError("SARIMAX_SEASONAL requires a seasonal period of at least two.")
            if len(values) < 2 * seasonal_period:
                raise ForecastError(
                    f"SARIMAX_SEASONAL requires at least two complete seasons "
                    f"({2 * seasonal_period} observations); got {len(values)}."
                )
            seasonal_order = (1, 0, 0, seasonal_period)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted_model = SARIMAX(
                values,
                order=(1, 1, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)
            predicted = fitted_model.get_forecast(steps=horizon)
        model_warnings = tuple(str(item.message) for item in captured)
        forecast = pd.Series(
            np.asarray(predicted.predicted_mean, dtype=float),
            index=index,
            name="forecast",
        )
        confidence = predicted.conf_int(alpha=1 - confidence_level).to_numpy(dtype=float)
        if not np.isfinite(forecast.to_numpy()).all() or not np.isfinite(confidence).all():
            raise ForecastError(f"{model_name} generated non-finite forecast values or intervals.")
        return ForecastResult(
            model_name,
            forecast,
            pd.Series(confidence[:, 0], index=index, name="lower"),
            pd.Series(confidence[:, 1], index=index, name="upper"),
            model_warnings,
        )
    except ForecastError:
        raise
    except (ValueError, RuntimeError, np.linalg.LinAlgError, IndexError) as exc:
        raise ForecastError(f"{model_name} could not fit this workload series: {exc}") from exc


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1)
    if aligned.empty:
        raise ValueError("No aligned actual and predicted values are available.")
    if aligned.isna().any().any():
        raise ValueError("Actual and predicted values must have complete, matching indexes.")
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Actual and predicted values must be finite.")
    errors = np.abs(aligned["actual"].to_numpy() - aligned["predicted"].to_numpy())
    denominator = np.abs(aligned["actual"].to_numpy()) + np.abs(
        aligned["predicted"].to_numpy()
    )
    smape_terms = np.divide(
        2 * errors,
        denominator,
        out=np.zeros_like(errors, dtype=float),
        where=denominator > np.finfo(float).eps,
    )
    return {
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAE": float(np.mean(errors)),
        "MaxAE": float(np.max(errors)),
        "P95AE": float(np.percentile(errors, 95)),
        "sMAPE": float(np.mean(smape_terms) * 100),
    }
