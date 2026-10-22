from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aura_dashboard.data import (
    WORKLOAD_CATEGORIES,
    classify_workload,
    extract_features,
    prepare_series,
    recommend_model,
)
from aura_dashboard.learning import QTable, calculate_reward, q_table_path
from aura_dashboard.models import (
    LOCAL_MODEL_NAMES,
    ForecastError,
    ForecastResult,
    available_model_names,
    calculate_metrics,
    forecast_series,
    timegpt_status,
)


PAGES = (
    "Overview",
    "Dataset",
    "Workload Analysis",
    "Model Selection",
    "Forecast",
    "Evaluation",
    "Comparison",
    "Learning History",
)


@st.cache_data(show_spinner=False)
def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents))


def initialize_learning_state() -> None:
    if "q_table" in st.session_state:
        return
    path = q_table_path()
    try:
        table = QTable.load(path) if path.exists() else QTable()
        st.session_state["q_table_error"] = None
        st.session_state["q_table_loaded"] = path.exists()
    except (OSError, ValueError) as exc:
        table = QTable()
        st.session_state["q_table_error"] = str(exc)
        st.session_state["q_table_loaded"] = False
    st.session_state["q_table"] = table
    st.session_state.setdefault("selection_history", [])


def add_history(entry: dict[str, object]) -> None:
    history = st.session_state.setdefault("selection_history", [])
    history.append({"timestamp": datetime.now(timezone.utc).isoformat(), **entry})


def persist_q_table() -> None:
    try:
        st.session_state["q_table"].save(q_table_path())
        st.session_state["q_table_error"] = None
    except OSError as exc:
        st.session_state["q_table_error"] = str(exc)
        st.error(f"Q-table update is active for this session but could not be saved: {exc}")


def get_prepared_series(
    raw: pd.DataFrame | None,
    timestamp_column: str | None,
    value_column: str | None,
    missing_strategy: str,
    resample_frequency: str | None,
    resample_aggregation: str,
) -> tuple[pd.Series, int, int] | None:
    if raw is None:
        st.info("Upload a CSV dataset in the sidebar to begin.")
        return None
    if not timestamp_column or not value_column:
        st.warning("Select both a timestamp column and a numeric workload column in the sidebar.")
        return None
    try:
        prepared = prepare_series(
            raw,
            timestamp_column,
            value_column,
            missing_strategy,
            resample_frequency,
            resample_aggregation,
        )
    except (KeyError, ValueError) as exc:
        st.error(f"Dataset validation failed: {exc}")
        return None
    return prepared.values, prepared.handled_missing, prepared.duplicate_timestamps


def feature_summary(series: pd.Series, seasonal_period: int) -> tuple[dict[str, float], str, str]:
    features = extract_features(series, seasonal_period)
    category = classify_workload(features)
    return features, category, recommend_model(category)


def show_features(features: dict[str, float]) -> None:
    labels = {
        "mean": "Mean",
        "standard_deviation": "Standard deviation",
        "coefficient_of_variation": "Coefficient of variation",
        "lag_one_autocorrelation": "Lag-one autocorrelation",
        "trend_strength": "Trend strength",
        "seasonal_strength": "Seasonal strength",
        "robust_outlier_ratio": "Robust outlier ratio",
    }
    table = pd.DataFrame(
        [
            {
                "Feature": labels[name],
                "Value": value if np.isfinite(value) else None,
                "Availability": "Available" if np.isfinite(value) else "Unavailable for this series",
            }
            for name, value in features.items()
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")


def selection_explanation(
    category: str,
    recommendation: str,
    selected: str,
    available_models: tuple[str, ...],
) -> str:
    table: QTable = st.session_state["q_table"]
    if all(table.get(category, model) == 0 for model in available_models):
        return (
            f"The workload is classified as **{category}**. No learned Q-values exist for "
            f"this category yet, so the rule-based recommendation **{recommendation}** is selected."
        )
    q_values = {model: table.get(category, model) for model in available_models}
    return (
        f"The workload is classified as **{category}**. The rule-based recommendation is "
        f"**{recommendation}**; the hybrid policy selected **{selected}** from the current "
        f"Q-values ({', '.join(f'{model}: {value:.3f}' for model, value in q_values.items())})."
    )


def create_forecast_chart(
    historical: pd.Series,
    forecast: pd.Series,
    title: str,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    confidence_level: float = 0.95,
) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(x=historical.index, y=historical.values, name="Historical", mode="lines")
    )
    chart.add_trace(
        go.Scatter(x=forecast.index, y=forecast.values, name="Forecast", mode="lines+markers")
    )
    if lower is not None and upper is not None:
        chart.add_trace(
            go.Scatter(
                x=forecast.index,
                y=upper.values,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
                name="Upper confidence bound",
            )
        )
        chart.add_trace(
            go.Scatter(
                x=forecast.index,
                y=lower.values,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.18)",
                line={"width": 0},
                name=f"{confidence_level:.0%} confidence interval",
            )
        )
    chart.update_layout(title=title, xaxis_title="Time", yaxis_title="Workload")
    return chart


def create_backtest_chart(
    actual: pd.Series,
    result: ForecastResult,
    title: str,
    confidence_level: float = 0.95,
) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(x=actual.index, y=actual.values, name="Actual", mode="lines")
    )
    chart.add_trace(
        go.Scatter(
            x=result.forecast.index,
            y=result.forecast.values,
            name=f"Forecast ({result.model_name})",
            mode="lines+markers",
        )
    )
    if result.lower is not None and result.upper is not None:
        chart.add_trace(
            go.Scatter(
                x=result.forecast.index,
                y=result.upper.values,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        chart.add_trace(
            go.Scatter(
                x=result.forecast.index,
                y=result.lower.values,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.18)",
                line={"width": 0},
                name=f"{confidence_level:.0%} confidence interval",
            )
        )
    chart.update_layout(title=title, xaxis_title="Time", yaxis_title="Workload")
    return chart


def main() -> None:
    st.set_page_config(page_title="AURA Workload Forecasting", layout="wide")
    initialize_learning_state()
    st.title("AURA | Adaptive Workload Forecasting")
    st.caption(
        "Upload and analyze historical workload data, compare local forecasting models, "
        "and adapt model selection with tabular Q-learning."
    )
    st.info(
        "This dashboard runs forecasts on uploaded data only. It does not connect to or "
        "operate live cloud resource scaling."
    )

    page = st.sidebar.selectbox("Dashboard", PAGES)
    upload = st.sidebar.file_uploader("Upload workload CSV", type=["csv"])
    raw: pd.DataFrame | None = None
    if upload is not None:
        try:
            raw = read_csv_bytes(upload.getvalue())
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            st.sidebar.error(f"Could not read CSV: {exc}")

    timestamp_column: str | None = None
    value_column: str | None = None
    missing_strategy = "drop"
    resample_frequency: str | None = None
    resample_aggregation = "mean"
    seasonal_period = 12
    if raw is not None:
        columns = [str(column) for column in raw.columns]
        if columns:
            timestamp_column = st.sidebar.selectbox("Timestamp column", columns)
            numeric_columns = [
                column
                for column in columns
                if pd.to_numeric(raw[column], errors="coerce").notna().any()
            ]
            if numeric_columns:
                value_column = st.sidebar.selectbox("Workload column", numeric_columns)
            else:
                st.sidebar.error("The CSV has no columns with numeric workload values.")
        missing_strategy = st.sidebar.selectbox(
            "Missing-value handling", ("drop", "interpolate", "forward fill")
        )
        resample_text = st.sidebar.text_input(
            "Resampling frequency (optional)", placeholder="e.g. 1h or 15min"
        )
        resample_frequency = resample_text.strip() or None
        resample_aggregation = st.sidebar.selectbox(
            "Resampling aggregation", ("mean", "sum", "max", "min")
        )
        seasonal_period = int(
            st.sidebar.number_input(
                "Seasonal period (observations per cycle)",
                min_value=2,
                value=12,
                step=1,
            )
        )

    include_timegpt = False
    timegpt_message = timegpt_status()
    if timegpt_message.startswith("TIMEGPT is configured"):
        include_timegpt = st.sidebar.checkbox(
            "Enable TIMEGPT API calls (may incur usage charges)", value=False
        )
    else:
        st.sidebar.caption(timegpt_message)
    available_models = available_model_names(include_timegpt)

    if st.session_state.get("q_table_error"):
        st.warning(
            "The saved Q-table could not be read or written; learning currently exists only "
            f"for this session. Details: {st.session_state['q_table_error']}"
        )

    if page == "Overview":
        st.subheader("Overview")
        if raw is None:
            st.write("Start by uploading a timestamped CSV dataset from the sidebar.")
        else:
            prepared = get_prepared_series(
                raw,
                timestamp_column,
                value_column,
                missing_strategy,
                resample_frequency,
                resample_aggregation,
            )
            if prepared:
                values, handled, duplicates = prepared
                features, category, recommendation = feature_summary(values, seasonal_period)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Observations", f"{len(values):,}")
                col2.metric("Workload category", category.title())
                col3.metric("Rule recommendation", recommendation)
                col4.metric("Q-table states", len(st.session_state["q_table"].values))
                st.caption(
                    f"Prepared {values.index.min()} through {values.index.max()}; "
                    f"{handled} missing value(s) handled and {duplicates} duplicate timestamp "
                    "row(s) detected."
                )
                show_features(features)
        st.subheader("Implemented forecasting models")
        st.write("Local models:", ", ".join(LOCAL_MODEL_NAMES))
        st.write(timegpt_message)
        st.caption(
            "Confidence intervals are shown for SARIMAX forecasts only. ETS and NAIVE_LAST "
            "do not expose intervals in this implementation."
        )

    elif page == "Dataset":
        st.subheader("Dataset")
        if upload is None:
            st.info("Upload a CSV file in the sidebar. The file is read locally by this app.")
        elif raw is not None:
            st.write(f"**File:** {upload.name} · **Rows:** {len(raw):,} · **Columns:** {len(raw.columns)}")
            st.dataframe(raw.head(20), hide_index=True, width="stretch")
            prepared = get_prepared_series(
                raw,
                timestamp_column,
                value_column,
                missing_strategy,
                resample_frequency,
                resample_aggregation,
            )
            if prepared:
                values, handled, duplicates = prepared
                st.success(
                    f"Validated {len(values):,} chronological observations. "
                    f"Handled {handled} missing value(s); detected {duplicates} duplicate "
                    "timestamp row(s)."
                )
                st.line_chart(values)

    elif page == "Workload Analysis":
        st.subheader("Workload Analysis")
        prepared = get_prepared_series(
            raw, timestamp_column, value_column, missing_strategy, resample_frequency, resample_aggregation
        )
        if prepared:
            values, _, _ = prepared
            features, category, recommendation = feature_summary(values, seasonal_period)
            st.metric("Rule-based workload category", category.title())
            st.write(f"**Rule-based model recommendation:** `{recommendation}`")
            show_features(features)
            if not np.isfinite(features["seasonal_strength"]):
                st.caption(
                    "Seasonal strength is unavailable: provide at least two complete cycles "
                    "for the configured seasonal period."
                )

    elif page == "Model Selection":
        st.subheader("Model Selection")
        prepared = get_prepared_series(
            raw, timestamp_column, value_column, missing_strategy, resample_frequency, resample_aggregation
        )
        if prepared:
            values, _, _ = prepared
            _, category, recommendation = feature_summary(values, seasonal_period)
            epsilon = st.slider("Exploration rate for this selection", 0.0, 1.0, 0.0, 0.05)
            table: QTable = st.session_state["q_table"]
            selected = table.select(
                category, epsilon, available_actions=available_models
            )
            st.write(selection_explanation(category, recommendation, selected, available_models))
            if st.button("Record model selection"):
                add_history(
                    {
                        "event": "selection",
                        "category": category,
                        "recommendation": recommendation,
                        "selected_model": selected,
                        "reward": None,
                    }
                )
                st.success(f"Recorded the {selected} selection in this session.")
            st.write("**Available model actions:**", ", ".join(available_models))
            st.caption(
                "The policy uses the rule recommendation before learning; after feedback, it "
                "prefers the highest-valued Q action and uses the rule recommendation to break ties."
            )

    elif page == "Forecast":
        st.subheader("Forecast")
        prepared = get_prepared_series(
            raw, timestamp_column, value_column, missing_strategy, resample_frequency, resample_aggregation
        )
        if prepared:
            values, _, _ = prepared
            features, category, recommendation = feature_summary(values, seasonal_period)
            table: QTable = st.session_state["q_table"]
            policy_selection = table.select(category, available_actions=available_models)
            st.write(
                selection_explanation(
                    category, recommendation, policy_selection, available_models
                )
            )
            selected = st.selectbox(
                "Forecast model",
                available_models,
                index=available_models.index(policy_selection),
            )
            if selected != policy_selection:
                st.info(f"Using the manually selected {selected} model for this forecast.")
            horizon = int(
                st.number_input(
                    "Forecast horizon (observations)",
                    min_value=1,
                    max_value=1000,
                    value=min(24, 1000),
                    step=1,
                )
            )
            if selected == "TIMEGPT":
                confidence_level = st.select_slider(
                    "TIMEGPT prediction interval", (0.80, 0.90, 0.95, 0.99), value=0.95
                )
            else:
                confidence_level = st.slider(
                    "Confidence level (SARIMAX models)", 0.80, 0.99, 0.95, 0.01
                )
            if st.button("Run forecast", type="primary"):
                with st.spinner(f"Fitting {selected} to {len(values):,} observations..."):
                    try:
                        result = forecast_series(
                            values,
                            selected,
                            horizon,
                            seasonal_period,
                            confidence_level,
                        )
                    except ForecastError as exc:
                        st.error(str(exc))
                    else:
                        add_history(
                            {
                                "event": "forecast",
                                "category": category,
                                "recommendation": recommendation,
                                "selected_model": selected,
                                "reward": None,
                            }
                        )
                        st.success(f"Forecast completed with {selected}.")
                        if result.warnings:
                            st.warning("Model warnings: " + "; ".join(result.warnings))
                        st.plotly_chart(
                            create_forecast_chart(
                                values,
                                result.forecast,
                                f"{selected} forecast",
                                result.lower,
                                result.upper,
                                confidence_level,
                            ),
                            width="stretch",
                        )
                        if result.lower is None:
                            st.caption("Confidence intervals are not available for this model.")
                        else:
                            st.dataframe(
                                pd.DataFrame(
                                    {
                                        "Forecast": result.forecast,
                                        f"Lower {confidence_level:.0%}": result.lower,
                                        f"Upper {confidence_level:.0%}": result.upper,
                                    }
                                ),
                                width="stretch",
                            )

    elif page in {"Evaluation", "Comparison"}:
        st.subheader("Chronological Backtesting" if page == "Evaluation" else "Adaptive vs Fixed SARIMAX")
        prepared = get_prepared_series(
            raw, timestamp_column, value_column, missing_strategy, resample_frequency, resample_aggregation
        )
        if prepared:
            values, _, _ = prepared
            if len(values) < 6:
                st.error("At least six observations are required for chronological backtesting.")
            else:
                max_test = len(values) - 3
                test_horizon = int(
                    st.number_input(
                        "Chronological test horizon (observations)",
                        min_value=1,
                        max_value=max_test,
                        value=min(24, max_test),
                        step=1,
                    )
                )
                epsilon = st.slider(
                    "Evaluation exploration rate", 0.0, 1.0, 0.0, 0.05, key="evaluation_epsilon"
                )
                if st.button("Run chronological backtest", type="primary"):
                    train = values.iloc[:-test_horizon]
                    test = values.iloc[-test_horizon:]
                    _, category, recommendation = feature_summary(train, seasonal_period)
                    table: QTable = st.session_state["q_table"]
                    adaptive_model = table.select(
                        category, epsilon, available_actions=available_models
                    )
                    st.write(
                        f"Training uses the first {len(train):,} observations; testing uses the "
                        f"last {len(test):,}, with no shuffling. The rule recommends "
                        f"**{recommendation}** and the hybrid policy selected **{adaptive_model}**."
                    )
                    st.caption(
                        "Every enabled model, including fixed SARIMAX, uses this same train/test "
                        "split, horizon, and metric implementation. TIMEGPT is included only when "
                        "explicitly enabled in the sidebar; that makes external API requests."
                    )
                    forecasts: dict[str, object] = {}
                    rows: list[dict[str, object]] = []
                    with st.spinner("Fitting the implemented models on the chronological training split..."):
                        for model_name in available_models:
                            try:
                                result = forecast_series(
                                    train, model_name, test_horizon, seasonal_period
                                )
                            except ForecastError as exc:
                                forecasts[model_name] = exc
                                rows.append({"Model": model_name, "Status": str(exc)})
                            else:
                                result = ForecastResult(
                                    result.model_name,
                                    result.forecast.set_axis(test.index),
                                    result.lower.set_axis(test.index) if result.lower is not None else None,
                                    result.upper.set_axis(test.index) if result.upper is not None else None,
                                    result.warnings,
                                )
                                forecasts[model_name] = result
                                row = {
                                    "Model": model_name,
                                    "Status": "Completed",
                                    **calculate_metrics(test, result.forecast),
                                }
                                if result.warnings:
                                    row["Warnings"] = "; ".join(result.warnings)
                                rows.append(row)
                    fixed_result = forecasts.get("SARIMAX")
                    adaptive_result = forecasts.get(adaptive_model)
                    comparison: list[dict[str, object]] = []
                    adaptive_metrics: dict[str, float] | None = None
                    for label, result in (
                        ("Adaptive", adaptive_result),
                        ("Fixed SARIMAX", fixed_result),
                    ):
                        if isinstance(result, ForecastError) or result is None:
                            error = str(result) if isinstance(result, ForecastError) else "Forecast unavailable."
                            comparison.append({"Approach": label, "Status": error})
                        else:
                            metrics = calculate_metrics(test, result.forecast)
                            comparison.append({"Approach": label, "Status": "Completed", **metrics})
                            if label == "Adaptive":
                                adaptive_metrics = metrics
                    st.session_state["last_backtest"] = {
                        "test": test,
                        "forecasts": forecasts,
                        "comparison": comparison,
                    }
                    if page == "Evaluation":
                        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                        if adaptive_metrics is not None:
                            adaptive_result = forecasts[adaptive_model]
                            st.plotly_chart(
                                create_backtest_chart(
                                    test,
                                    adaptive_result,
                                    f"Chronological backtest: adaptive {adaptive_model}",
                                    0.95,
                                ),
                                width="stretch",
                            )
                    else:
                        st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
                        if adaptive_metrics is not None and isinstance(fixed_result, ForecastResult):
                            chart = go.Figure()
                            chart.add_trace(
                                go.Scatter(x=test.index, y=test.values, name="Actual", mode="lines")
                            )
                            chart.add_trace(
                                go.Scatter(
                                    x=test.index,
                                    y=adaptive_result.forecast,
                                    name=f"Adaptive ({adaptive_model})",
                                    mode="lines+markers",
                                )
                            )
                            chart.add_trace(
                                go.Scatter(
                                    x=test.index,
                                    y=fixed_result.forecast,
                                    name="Fixed SARIMAX",
                                    mode="lines+markers",
                                )
                            )
                            chart.update_layout(
                                title="Adaptive and fixed forecast on identical test observations",
                                xaxis_title="Time",
                                yaxis_title="Workload",
                            )
                            st.plotly_chart(chart, width="stretch")
                    train_scale = float(np.mean(np.abs(train.to_numpy(dtype=float))))
                    reward = calculate_reward(
                        adaptive_metrics,
                        train_scale,
                        fallback=adaptive_metrics is None,
                    )
                    table.update(category, adaptive_model, reward)
                    persist_q_table()
                    add_history(
                        {
                            "event": "backtest",
                            "category": category,
                            "recommendation": recommendation,
                            "selected_model": adaptive_model,
                            "reward": reward,
                            "MAE": adaptive_metrics["MAE"] if adaptive_metrics else None,
                            "test_observations": len(test),
                        }
                    )
                    st.metric("Adaptive model reward", f"{reward:.4f}")
                    st.caption(
                        "Reward = 1 / (1 + MAE / mean absolute training workload), normalized "
                        "to (0, 1]; a failed adaptive forecast receives the configured -0.25 "
                        "fallback penalty. Q-values use a 0.2 terminal-update learning rate."
                    )

    elif page == "Learning History":
        st.subheader("Learning History")
        history = st.session_state.get("selection_history", [])
        if history:
            st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
        else:
            st.info("No model-selection or backtest history has been recorded in this session.")
        table: QTable = st.session_state["q_table"]
        entries = [
            {"Workload category": state, "Model": model, "Q-value": value}
            for state, actions in table.values.items()
            for model, value in actions.items()
        ]
        if entries:
            st.dataframe(pd.DataFrame(entries), hide_index=True, width="stretch")
            rewards = [entry["reward"] for entry in history if entry.get("reward") is not None]
            first, second = st.columns(2)
            first.metric("Q-table updates in this session", sum(1 for e in history if e["event"] == "backtest"))
            second.metric(
                "Mean observed reward",
                f"{np.mean(rewards):.4f}" if rewards else "Not yet measured",
            )
        else:
            st.info("No Q-table updates exist yet. Run a chronological backtest to learn from measured errors.")
        path = q_table_path()
        st.caption(f"Q-table persistence path: `{path}`")
        if path.exists() and st.session_state.get("q_table_loaded"):
            st.success("A local Q-table was loaded at the start of this session.")
        elif path.exists():
            st.warning("A Q-table file exists, but it was not loaded successfully.")
        else:
            st.caption("No Q-table file exists yet; it is written after a backtest update.")


if __name__ == "__main__":
    main()
