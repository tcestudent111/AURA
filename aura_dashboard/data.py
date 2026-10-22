from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


WORKLOAD_CATEGORIES = (
    "constant",
    "stable",
    "periodic",
    "trending",
    "bursty",
    "noisy",
    "uncertain",
)


@dataclass(frozen=True)
class PreparedSeries:
    values: pd.Series
    handled_missing: int
    duplicate_timestamps: int


def prepare_series(
    frame: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
    missing_strategy: str,
    resample_frequency: str | None = None,
    resample_aggregation: str = "mean",
) -> PreparedSeries:
    if timestamp_column not in frame.columns or value_column not in frame.columns:
        raise ValueError("Choose existing timestamp and workload columns.")
    if timestamp_column == value_column:
        raise ValueError("Timestamp and workload columns must be different.")
    if missing_strategy not in {"drop", "interpolate", "forward fill"}:
        raise ValueError("Choose a supported missing-value strategy.")
    if resample_aggregation not in {"mean", "sum", "max", "min"}:
        raise ValueError("Choose a supported resampling aggregation.")

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    if invalid_timestamps:
        raise ValueError(
            f"{invalid_timestamps} timestamp value(s) could not be parsed. "
            "Correct or remove them before forecasting."
        )

    values = pd.to_numeric(frame[value_column], errors="coerce")
    indexed = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(timestamps), name=value_column)
    indexed = indexed.sort_index()
    duplicates = int(indexed.index.duplicated(keep=False).sum())
    if resample_frequency:
        try:
            indexed = indexed.resample(resample_frequency).agg(resample_aggregation)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid resampling frequency: {exc}") from exc
    elif indexed.index.has_duplicates:
        indexed = indexed.groupby(level=0).mean()

    missing_before = int(indexed.isna().sum())
    if missing_strategy == "drop":
        indexed = indexed.dropna()
    elif missing_strategy == "interpolate":
        indexed = indexed.interpolate(method="time", limit_direction="both").dropna()
    else:
        indexed = indexed.ffill().dropna()
    handled_missing = max(0, missing_before - int(indexed.isna().sum()))
    if indexed.empty:
        raise ValueError("No usable numeric workload values remain after missing-value handling.")
    if len(indexed) < 3:
        raise ValueError("At least three valid observations are required.")
    if not np.isfinite(indexed.to_numpy(dtype=float)).all():
        raise ValueError("Workload values must be finite numbers.")
    indexed.name = value_column
    return PreparedSeries(indexed.astype(float), handled_missing, duplicates)


def extract_features(series: pd.Series, seasonal_period: int | None = None) -> dict[str, float]:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        raise ValueError("At least three observations are required for workload analysis.")
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=0))
    scale = max(abs(mean), np.finfo(float).eps)
    coefficient_of_variation = standard_deviation / scale
    lag_one_autocorrelation = (
        float(pd.Series(values).autocorr(lag=1))
        if standard_deviation > np.finfo(float).eps
        else 0.0
    )
    if not np.isfinite(lag_one_autocorrelation):
        lag_one_autocorrelation = 0.0

    total_variance = float(np.var(values))
    if total_variance <= np.finfo(float).eps:
        trend_strength = 0.0
    else:
        trend = np.polyval(np.polyfit(np.arange(len(values)), values, 1), np.arange(len(values)))
        trend_strength = float(
            np.clip(1 - np.var(values - trend) / total_variance, 0.0, 1.0)
        )

    seasonal_strength = float("nan")
    if seasonal_period is not None and seasonal_period >= 2 and len(values) >= 2 * seasonal_period:
        decomposition = STL(values, period=seasonal_period, robust=True).fit()
        remainder_variance = float(np.var(decomposition.resid))
        seasonal_denom = float(np.var(decomposition.resid + decomposition.seasonal))
        if seasonal_denom > np.finfo(float).eps:
            seasonal_strength = float(
                np.clip(1 - remainder_variance / seasonal_denom, 0.0, 1.0)
            )
        trend_denom = float(np.var(decomposition.resid + decomposition.trend))
        if trend_denom > np.finfo(float).eps:
            trend_strength = float(
                np.clip(1 - remainder_variance / trend_denom, 0.0, 1.0)
            )

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= np.finfo(float).eps:
        robust_outlier_ratio = float(np.mean(np.abs(values - median) > np.finfo(float).eps))
    else:
        robust_outlier_ratio = float(
            np.mean(np.abs(values - median) > 3.0 * 1.4826 * mad)
        )

    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": float(coefficient_of_variation),
        "lag_one_autocorrelation": lag_one_autocorrelation,
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "robust_outlier_ratio": robust_outlier_ratio,
    }


def classify_workload(features: dict[str, float]) -> str:
    mean = features["mean"]
    standard_deviation = features["standard_deviation"]
    coefficient_of_variation = features["coefficient_of_variation"]
    seasonal_strength = features["seasonal_strength"]
    trend_strength = features["trend_strength"]
    outlier_ratio = features["robust_outlier_ratio"]
    autocorrelation = features["lag_one_autocorrelation"]

    if standard_deviation <= 1e-9 * max(1.0, abs(mean)):
        return "constant"
    if np.isfinite(seasonal_strength) and seasonal_strength >= 0.55:
        return "periodic"
    if trend_strength >= 0.7:
        return "trending"
    if outlier_ratio >= 0.05:
        return "bursty"
    if coefficient_of_variation <= 0.25 and outlier_ratio <= 0.03:
        return "stable"
    if coefficient_of_variation >= 0.5 or (
        coefficient_of_variation > 0.25 and abs(autocorrelation) < 0.2
    ):
        return "noisy"
    return "uncertain"


def recommend_model(category: str) -> str:
    recommendations = {
        "constant": "NAIVE_LAST",
        "stable": "ETS",
        "periodic": "SARIMAX_SEASONAL",
        "trending": "SARIMAX",
        "bursty": "NAIVE_LAST",
        "noisy": "ETS",
        "uncertain": "SARIMAX",
    }
    try:
        return recommendations[category]
    except KeyError as exc:
        raise ValueError(f"Unknown workload category: {category}") from exc
