from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# App setup and project paths
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Charging Demand Insights",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ev_charging_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

BASE_FEATURES = [
    "Location",
    "ChargerCompany",
    "ChargerType",
    "StartDayName",
    "TimeOfDay",
    "StartMonth",
    "StartDayOfMonth",
    "StartHour",
    "IsWeekend",
]
DURATION_FEATURES = BASE_FEATURES + ["DurationValidated"]

REGRESSION_MODELS = {
    "Without duration": "lightgbm_without_duration_pipeline.joblib",
    "With duration": "lightgbm_with_duration_pipeline.joblib",
}
CLASSIFICATION_MODELS = {
    "Without duration": "lightgbm_classifier_without_duration_pipeline.joblib",
    "With duration": "lightgbm_classifier_with_duration_pipeline.joblib",
}


# -----------------------------------------------------------------------------
# Cached loading helpers. The dashboard loads saved artifacts; it never trains.
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading processed charging records...")
def load_dataset(path_string: str) -> pd.DataFrame:
    data = pd.read_csv(path_string, parse_dates=["StartDatetime"])
    data["StartDatetime"] = pd.to_datetime(data["StartDatetime"], errors="coerce")
    data["Demand"] = pd.to_numeric(data["Demand"], errors="coerce")
    data["DurationValidated"] = pd.to_numeric(
        data["DurationValidated"], errors="coerce"
    )

    # Match the categorical representation used when the model pipelines were fit.
    data["ChargerCompany"] = data["ChargerCompany"].astype("string")
    data["ChargerType"] = data["ChargerType"].astype("string")
    data["Location"] = data["Location"].astype("string")
    data["DemandCategory"] = data["DemandCategory"].astype("string")
    return data


@st.cache_resource(show_spinner=False)
def load_pipeline(path_string: str) -> Any:
    return joblib.load(path_string)


@st.cache_data(show_spinner=False)
def load_csv_report(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


def report_path(filename: str) -> Path:
    return REPORTS_DIR / filename


def read_report(filename: str) -> pd.DataFrame | None:
    path = report_path(filename)
    if not path.is_file():
        st.warning(f"Report not found: `reports/{filename}`")
        return None
    try:
        return load_csv_report(str(path))
    except Exception as exc:  # keep one missing/corrupt report from breaking the app
        st.warning(f"Could not read `reports/{filename}`: {exc}")
        return None


def time_of_day_from_hour(hour: int) -> str:
    """Use the same time-of-day categories defined for the engineered dataset."""
    if hour < 6:
        return "Night"
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    return "Evening"


def make_prediction_row(
    location: str,
    charger_company: str,
    charger_type: str,
    start_date: date,
    start_time: time,
    duration_minutes: float | None = None,
) -> pd.DataFrame:
    start_hour = int(start_time.hour)
    start_day_name = start_date.strftime("%A")

    row: dict[str, Any] = {
        "Location": str(location),
        "ChargerCompany": str(charger_company),
        "ChargerType": str(charger_type),
        "StartDayName": start_day_name,
        "TimeOfDay": time_of_day_from_hour(start_hour),
        "StartMonth": int(start_date.month),
        "StartDayOfMonth": int(start_date.day),
        "StartHour": start_hour,
        "IsWeekend": int(start_date.weekday() >= 5),
    }
    if duration_minutes is not None:
        row["DurationValidated"] = float(duration_minutes)
    return pd.DataFrame([row])


def high_demand_probability(pipeline: Any, row: pd.DataFrame) -> float | None:
    """Return the probability for class 1 (High Demand), when supported."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    probabilities = pipeline.predict_proba(row)[0]
    estimator = (
        pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    )
    classes = list(getattr(estimator, "classes_", []))
    if 1 in classes:
        return float(probabilities[classes.index(1)])
    if len(probabilities) >= 2:
        return float(probabilities[1])
    return None


def category_options(data: pd.DataFrame, column: str) -> list[str]:
    values = data[column].dropna().astype(str)
    counts = values.value_counts()
    return counts.index.tolist()


def show_model_image(filename: str, caption: str) -> None:
    path = REPORTS_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Image not found: `reports/{filename}`")


def load_confusion_matrix(filename: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / filename
    if not path.is_file():
        st.warning(f"Confusion matrix not found: `reports/{filename}`")
        return None
    try:
        matrix = pd.read_csv(path, index_col=0)
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        matrix = matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if matrix.empty:
            matrix = pd.read_csv(path)
            if matrix.columns[0].lower().startswith("unnamed"):
                matrix = matrix.set_index(matrix.columns[0])
            matrix = matrix.apply(pd.to_numeric, errors="coerce")
            matrix = matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return matrix
    except Exception as exc:
        st.warning(f"Could not read `{filename}`: {exc}")
        return None


def show_confusion_matrix(filename: str, title: str) -> None:
    matrix = load_confusion_matrix(filename)
    if matrix is None or matrix.empty:
        return
    st.markdown(f"**{title}**")
    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        labels={"x": "Predicted category", "y": "Actual category", "color": "Sessions"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=330)
    st.plotly_chart(
        fig, use_container_width=True, key=f"confusion_matrix_{Path(filename).stem}"
    )


# -----------------------------------------------------------------------------
# Load required input data
# -----------------------------------------------------------------------------
st.title("🔋 EV Charging Demand Insights")
st.caption(
    "Usage patterns, demand predictions, forecasting, and model explanations "
    "for the EV Charging Demand Prediction and Usage Pattern Analysis project."
)

if not DATA_PATH.is_file():
    st.error(
        "Processed dataset was not found. Place `app.py` in the project root "
        "alongside the `data`, `models`, and `reports` folders.\n\n"
        f"Expected file: `{DATA_PATH}`"
    )
    st.stop()

try:
    df = load_dataset(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the processed dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The processed dataset contains no records.")
    st.stop()

min_timestamp = df["StartDatetime"].min()
max_timestamp = df["StartDatetime"].max()
min_date = min_timestamp.date()
max_date = max_timestamp.date()

# -----------------------------------------------------------------------------
# Sidebar: filters apply to the Overview section only
# -----------------------------------------------------------------------------
st.sidebar.title("Explore charging records")
st.sidebar.caption("These filters apply to the Overview section.")
selected_dates = st.sidebar.date_input(
    "Session start date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
location_options = category_options(df, "Location")
company_options = category_options(df, "ChargerCompany")
charger_type_options = category_options(df, "ChargerType")

selected_locations = st.sidebar.multiselect(
    "Location", options=location_options, default=[]
)
selected_companies = st.sidebar.multiselect(
    "Charger company code", options=company_options, default=[]
)
selected_charger_types = st.sidebar.multiselect(
    "Charger type code", options=charger_type_options, default=[]
)

filtered = df.copy()
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    date_start, date_end = selected_dates
    filtered = filtered.loc[
        filtered["StartDatetime"].dt.date.between(date_start, date_end)
    ]
if selected_locations:
    filtered = filtered.loc[filtered["Location"].astype(str).isin(selected_locations)]
if selected_companies:
    filtered = filtered.loc[
        filtered["ChargerCompany"].astype(str).isin(selected_companies)
    ]
if selected_charger_types:
    filtered = filtered.loc[
        filtered["ChargerType"].astype(str).isin(selected_charger_types)
    ]


overview_tab, prediction_tab, forecast_tab, model_tab = st.tabs(
    [
        "Overview & Usage Patterns",
        "Predict Demand",
        "Trends & Forecasts",
        "Model Performance & Explainability",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: Overview and EDA
# -----------------------------------------------------------------------------
with overview_tab:
    st.subheader("Charging usage overview")
    if filtered.empty:
        st.warning("No records match the selected filters. Change the sidebar filters.")
    else:
        session_count = len(filtered)
        mean_demand = filtered["Demand"].mean()
        total_demand = filtered["Demand"].sum(min_count=1)
        high_share = (filtered["DemandCategory"] == "High Demand").mean() * 100
        median_demand = filtered["Demand"].median()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Charging sessions", f"{session_count:,}")
        c2.metric("Average recorded Demand", f"{mean_demand:,.2f}")
        c3.metric("Sum of recorded Demand", f"{total_demand:,.2f}")
        c4.metric("High Demand sessions", f"{high_share:.1f}%")
        st.caption(
            f"Median recorded Demand in this selection: {median_demand:,.2f}. "
            "The project brief does not specify the physical unit of Demand; values "
            "are therefore shown without a kW or kWh label."
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Average recorded Demand by location**")
            by_location = (
                filtered.groupby("Location", dropna=False)["Demand"]
                .agg(MeanDemand="mean", Sessions="count")
                .reset_index()
                .sort_values("MeanDemand", ascending=False)
                .head(12)
            )
            fig = px.bar(
                by_location,
                x="MeanDemand",
                y="Location",
                orientation="h",
                hover_data=["Sessions"],
                labels={
                    "MeanDemand": "Average recorded Demand",
                    "Location": "Location",
                },
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=430)
            st.plotly_chart(
                fig, use_container_width=True, key="overview_average_demand_by_location"
            )

        with right:
            st.markdown("**Average recorded Demand by charger type**")
            by_type = (
                filtered.groupby("ChargerType", dropna=False)["Demand"]
                .agg(MeanDemand="mean", Sessions="count")
                .reset_index()
                .sort_values("ChargerType")
            )
            fig = px.bar(
                by_type,
                x="ChargerType",
                y="MeanDemand",
                hover_data=["Sessions"],
                labels={
                    "MeanDemand": "Average recorded Demand",
                    "ChargerType": "Charger type code",
                },
            )
            fig.update_layout(height=430)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="overview_average_demand_by_charger_type",
            )

        left, right = st.columns(2)
        with left:
            st.markdown("**Average recorded Demand by session start hour**")
            by_hour = (
                filtered.groupby("StartHour", dropna=False)["Demand"]
                .agg(MeanDemand="mean", Sessions="count")
                .reset_index()
                .sort_values("StartHour")
            )
            fig = px.line(
                by_hour,
                x="StartHour",
                y="MeanDemand",
                markers=True,
                hover_data=["Sessions"],
                labels={
                    "StartHour": "Start hour (0–23)",
                    "MeanDemand": "Average recorded Demand",
                },
            )
            fig.update_layout(height=360)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="overview_average_demand_by_start_hour",
            )

        with right:
            st.markdown("**Daily sum of recorded Demand**")
            daily = (
                filtered.assign(SessionDate=filtered["StartDatetime"].dt.floor("D"))
                .groupby("SessionDate", as_index=False)["Demand"]
                .sum(min_count=1)
            )
            fig = px.line(
                daily,
                x="SessionDate",
                y="Demand",
                labels={
                    "SessionDate": "Session date",
                    "Demand": "Sum of recorded Demand",
                },
            )
            fig.update_layout(height=360)
            st.plotly_chart(
                fig, use_container_width=True, key="overview_daily_demand_sum"
            )

        st.markdown("**Demand category distribution**")
        category_counts = (
            filtered["DemandCategory"]
            .value_counts(dropna=False)
            .rename_axis("DemandCategory")
            .reset_index(name="Sessions")
        )
        fig = px.pie(
            category_counts,
            names="DemandCategory",
            values="Sessions",
            hole=0.45,
        )
        fig.update_layout(height=330)
        st.plotly_chart(
            fig, use_container_width=True, key="overview_demand_category_distribution"
        )

        with st.expander("View the filtered record sample"):
            st.dataframe(
                filtered[
                    [
                        "StartDatetime",
                        "Location",
                        "ChargerCompany",
                        "ChargerType",
                        "StartHour",
                        "TimeOfDay",
                        "Demand",
                        "DemandCategory",
                    ]
                ].head(1000),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Showing at most 1,000 rows for a responsive preview.")

# -----------------------------------------------------------------------------
# Tab 2: Manual-entry predictions
# -----------------------------------------------------------------------------
with prediction_tab:
    st.subheader("Predict demand for a charging session")
    st.write(
        "Enter the session characteristics below. The form predicts both a "
        "numerical demand value and a High Demand / Normal Demand category."
    )
    st.info(
        "**Session-start mode** does not require duration. Duration-informed results "
        "are optional and are only calculated when you explicitly enter a duration "
        "value or estimate. Duration is treated as minutes, consistent with the "
        "timestamp-derived duration feature; the meaning/units of Demand itself "
        "are not specified in the project brief."
    )

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_location = st.selectbox("Location", options=location_options)
            input_company = st.selectbox(
                "Charger company code", options=company_options
            )
        with col2:
            input_charger_type = st.selectbox(
                "Charger type code", options=charger_type_options
            )
            input_date = st.date_input(
                "Session start date",
                value=max_date,
                help=(
                    "You may enter a future date. The training data covers "
                    f"{min_date.isoformat()} through {max_date.isoformat()}, so predictions "
                    "outside that period are extrapolations. Use caution."
                ),
            )
        with col3:
            input_time = st.time_input("Session start time", value=time(18, 0))
            input_duration_text = st.text_input(
                "Duration in minutes (optional)",
                value="",
                placeholder="Leave blank if unknown",
                help=(
                    "Enter an observed duration or your own estimate. If you enter an "
                    "estimate, treat the duration-informed result as conditional on it."
                ),
            )
        submitted = st.form_submit_button(
            "Run prediction", type="primary", use_container_width=True
        )

    if submitted:
        duration_minutes: float | None = None
        duration_error: str | None = None
        if input_duration_text.strip():
            try:
                duration_minutes = float(input_duration_text.strip())
                if not pd.notna(duration_minutes) or duration_minutes <= 0:
                    duration_error = "Duration must be a positive number of minutes."
                    duration_minutes = None
            except ValueError:
                duration_error = (
                    "Enter duration as a number of minutes, or leave it blank."
                )

        if duration_error:
            st.warning(duration_error)

        # Session-start predictions: no post-session duration field is used.
        row_without = make_prediction_row(
            input_location,
            input_company,
            input_charger_type,
            input_date,
            input_time,
        )
        try:
            reg_without = load_pipeline(
                str(MODELS_DIR / REGRESSION_MODELS["Without duration"])
            )
            cls_without = load_pipeline(
                str(MODELS_DIR / CLASSIFICATION_MODELS["Without duration"])
            )
            reg_pred_without = float(reg_without.predict(row_without[BASE_FEATURES])[0])
            cls_pred_without = int(cls_without.predict(row_without[BASE_FEATURES])[0])
            cls_prob_without = high_demand_probability(
                cls_without, row_without[BASE_FEATURES]
            )

            st.markdown("### Results available at session start")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted recorded Demand", f"{reg_pred_without:,.2f}")
            c2.metric(
                "Predicted category",
                "High Demand" if cls_pred_without == 1 else "Normal Demand",
            )
            c3.metric(
                "High Demand probability",
                f"{cls_prob_without:.1%}"
                if cls_prob_without is not None
                else "Not available",
            )
            st.caption(
                "Predictions use the models trained without DurationValidated. "
                "The displayed probability is the classifier's estimated probability, "
                "not a guarantee of the actual outcome."
            )
        except Exception as exc:
            st.error(f"Session-start prediction failed: {exc}")

        if duration_minutes is not None and duration_error is None:
            row_with = make_prediction_row(
                input_location,
                input_company,
                input_charger_type,
                input_date,
                input_time,
                duration_minutes=duration_minutes,
            )
            try:
                reg_with = load_pipeline(
                    str(MODELS_DIR / REGRESSION_MODELS["With duration"])
                )
                cls_with = load_pipeline(
                    str(MODELS_DIR / CLASSIFICATION_MODELS["With duration"])
                )
                reg_pred_with = float(reg_with.predict(row_with[DURATION_FEATURES])[0])
                cls_pred_with = int(cls_with.predict(row_with[DURATION_FEATURES])[0])
                cls_prob_with = high_demand_probability(
                    cls_with, row_with[DURATION_FEATURES]
                )

                st.markdown("### Results conditional on the supplied duration")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted recorded Demand", f"{reg_pred_with:,.2f}")
                c2.metric(
                    "Predicted category",
                    "High Demand" if cls_pred_with == 1 else "Normal Demand",
                )
                c3.metric(
                    "High Demand probability",
                    f"{cls_prob_with:.1%}"
                    if cls_prob_with is not None
                    else "Not available",
                )
                st.caption(
                    f"These predictions use the supplied duration of {duration_minutes:g} minutes. "
                    "If it is an estimate, the results are conditional on that estimate; "
                    "they are not session-start-only predictions."
                )
            except Exception as exc:
                st.error(f"Duration-informed prediction failed: {exc}")

        with st.expander("See the input features sent to the model"):
            st.dataframe(row_without, use_container_width=True, hide_index=True)
            if duration_minutes is not None and duration_error is None:
                st.dataframe(row_with, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Tab 3: Trends and forecasts
# -----------------------------------------------------------------------------
with forecast_tab:
    st.subheader("Demand trends and time-series forecasts")
    st.write(
        "The upper chart shows actual demand and both forecast methods over the "
        "chronological test period used in Notebook 06. The lower table reports the "
        "saved test metrics."
    )

    forecast_metrics = read_report("time_series_forecast_comparison.csv")
    forecast_test = read_report("time_series_test_forecasts.csv")

    if forecast_metrics is not None and not forecast_metrics.empty:
        ses_row = forecast_metrics.loc[
            forecast_metrics["Method"]
            .astype(str)
            .str.contains("Exponential", case=False)
        ]
        ma_row = forecast_metrics.loc[
            forecast_metrics["Method"]
            .astype(str)
            .str.contains("MovingAverage", case=False)
        ]
        metric_cols = st.columns(4)
        if not ses_row.empty:
            metric_cols[0].metric("SES test MAE", f"{ses_row.iloc[0]['MAE']:.2f}")
            metric_cols[1].metric("SES test RMSE", f"{ses_row.iloc[0]['RMSE']:.2f}")
        if not ma_row.empty:
            metric_cols[2].metric("24-hour MA test MAE", f"{ma_row.iloc[0]['MAE']:.2f}")
            metric_cols[3].metric(
                "24-hour MA test RMSE", f"{ma_row.iloc[0]['RMSE']:.2f}"
            )
        st.dataframe(forecast_metrics, use_container_width=True, hide_index=True)

        if not ses_row.empty and not ma_row.empty:
            improvement = (
                1 - float(ses_row.iloc[0]["MAE"]) / float(ma_row.iloc[0]["MAE"])
            ) * 100
            st.success(
                f"On this held-out test period, Exponential Smoothing reduced MAE by "
                f"{improvement:.1f}% compared with the 24-hour moving average."
            )

    if forecast_test is not None and not forecast_test.empty:
        forecast_test["StartDatetime"] = pd.to_datetime(
            forecast_test["StartDatetime"], errors="coerce"
        )
        forecast_test = forecast_test.sort_values("StartDatetime")
        fig = go.Figure()
        for column, label in [
            ("ActualDemand", "Actual demand"),
            ("ExponentialSmoothing", "Exponential smoothing"),
            ("MovingAverage24h", "24-hour moving average"),
        ]:
            if column in forecast_test.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_test["StartDatetime"],
                        y=forecast_test[column],
                        mode="lines",
                        name=label,
                    )
                )
        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Sum of recorded Demand per hour",
            hovermode="x unified",
            height=480,
            legend_title_text="Series",
        )
        st.plotly_chart(fig, use_container_width=True, key="forecast_actual_vs_methods")
        st.caption(
            "Forecasts are evaluated on the saved chronological test period. They are "
            "aggregate hourly demand forecasts, not session-level predictions."
        )

    st.markdown("#### Historical daily demand")
    historical_daily = (
        df.assign(SessionDate=df["StartDatetime"].dt.floor("D"))
        .groupby("SessionDate", as_index=False)["Demand"]
        .sum(min_count=1)
    )
    fig = px.line(
        historical_daily,
        x="SessionDate",
        y="Demand",
        labels={"SessionDate": "Session date", "Demand": "Sum of recorded Demand"},
    )
    fig.update_layout(height=360)
    st.plotly_chart(
        fig, use_container_width=True, key="forecast_historical_daily_demand"
    )

# -----------------------------------------------------------------------------
# Tab 4: Model performance and explainability
# -----------------------------------------------------------------------------
with model_tab:
    st.subheader("Model performance")
    st.markdown("#### Regression comparison")
    regression_results = read_report("regression_model_comparison_chronological.csv")
    if regression_results is not None and not regression_results.empty:
        feature_set = st.radio(
            "Regression feature configuration",
            options=["Without Duration", "With Duration"],
            horizontal=True,
            key="regression_feature_set",
        )
        selected_reg_results = regression_results.loc[
            regression_results["FeatureSet"].astype(str) == feature_set
        ].copy()
        if not selected_reg_results.empty:
            selected_reg_results = selected_reg_results.sort_values("MAE")
            st.dataframe(
                selected_reg_results[
                    ["Model", "MAE", "MSE", "RMSE", "R2", "TrainMAE", "TrainR2"]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )
        if feature_set == "With Duration":
            st.warning(
                "Duration-informed metrics use a post-session feature and are not directly "
                "representative of what is available at session start. Compare within the "
                "same feature configuration."
            )

    regression_predictions = read_report("regression_test_predictions.csv")
    if regression_predictions is not None and not regression_predictions.empty:
        st.markdown("#### Regression test predictions")
        reg_plot_choice = st.selectbox(
            "Show prediction series",
            options=["Without duration", "With duration"],
            key="regression_prediction_series",
        )
        regression_predictions["StartDatetime"] = pd.to_datetime(
            regression_predictions["StartDatetime"], errors="coerce"
        )
        pred_column = (
            "LightGBM_WithoutDuration"
            if reg_plot_choice == "Without duration"
            else "LightGBM_WithDuration"
        )
        if {"StartDatetime", "ActualDemand", pred_column}.issubset(
            regression_predictions.columns
        ):
            regression_predictions = regression_predictions.sort_values("StartDatetime")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=regression_predictions["StartDatetime"],
                    y=regression_predictions["ActualDemand"],
                    mode="lines",
                    name="Actual demand",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=regression_predictions["StartDatetime"],
                    y=regression_predictions[pred_column],
                    mode="lines",
                    name=f"LightGBM — {reg_plot_choice}",
                )
            )
            fig.update_layout(
                xaxis_title="Session start time",
                yaxis_title="Recorded Demand",
                hovermode="x unified",
                height=420,
            )
            st.plotly_chart(
                fig, use_container_width=True, key="regression_actual_vs_predicted"
            )

    st.markdown("#### Classification comparison")
    classification_results = read_report(
        "classification_model_comparison_chronological.csv"
    )
    if classification_results is not None and not classification_results.empty:
        cls_feature_set = st.radio(
            "Classification feature configuration",
            options=["Without Duration", "With Duration"],
            horizontal=True,
            key="classification_feature_set",
        )
        selected_cls_results = classification_results.loc[
            classification_results["FeatureSet"].astype(str) == cls_feature_set
        ].copy()
        if not selected_cls_results.empty:
            selected_cls_results = selected_cls_results.sort_values(
                "F1", ascending=False
            )
            st.dataframe(
                selected_cls_results[
                    ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )
        if cls_feature_set == "With Duration":
            st.warning(
                "Duration-informed classification uses a post-session feature. Its metrics "
                "should not be described as session-start classification performance."
            )

    baseline = read_report("classification_baseline_metrics.csv")
    if baseline is not None and not baseline.empty:
        with st.expander("Classification baseline"):
            st.dataframe(baseline, use_container_width=True, hide_index=True)

    st.markdown("#### Classification confusion matrices")
    cm1, cm2 = st.columns(2)
    with cm1:
        show_confusion_matrix(
            "classification_confusion_matrix_without_duration.csv",
            "Without duration",
        )
    with cm2:
        show_confusion_matrix(
            "classification_confusion_matrix_with_duration.csv",
            "With duration",
        )
    st.caption(
        "For the confusion matrices, rows represent actual categories and columns represent predicted categories, as saved by the notebook."
    )

    st.divider()
    st.subheader("Feature importance and SHAP explanations")
    st.write(
        "Mean absolute SHAP values indicate average contribution magnitude, not causation. "
        "SHAP outputs should be interpreted alongside each model's evaluation metrics."
    )

    explanation_tab_reg, explanation_tab_cls = st.tabs(
        ["Regression explanations", "Classification explanations"]
    )
    with explanation_tab_reg:
        config = st.radio(
            "Regression explanation configuration",
            options=["Without duration", "With duration"],
            horizontal=True,
            key="reg_shap_config",
        )
        suffix = "without_duration" if config == "Without duration" else "with_duration"
        p1, p2 = st.columns(2)
        with p1:
            show_model_image(
                f"regression_shap_bar_{suffix}.png",
                f"Regression SHAP importance — {config.lower()}",
            )
        with p2:
            show_model_image(
                f"regression_shap_summary_{suffix}.png",
                f"Regression SHAP summary — {config.lower()}",
            )
        importance = read_report(f"regression_shap_importance_{suffix}.csv")
        permutation = read_report(f"regression_permutation_importance_{suffix}.csv")
        if importance is not None:
            with st.expander("View regression SHAP importance values"):
                st.dataframe(importance, use_container_width=True, hide_index=True)
        if permutation is not None:
            with st.expander("View regression permutation importance"):
                st.dataframe(permutation, use_container_width=True, hide_index=True)

    with explanation_tab_cls:
        config_cls = st.radio(
            "Classification explanation configuration",
            options=["Without duration", "With duration"],
            horizontal=True,
            key="cls_shap_config",
        )
        suffix_cls = (
            "without_duration" if config_cls == "Without duration" else "with_duration"
        )
        p1, p2 = st.columns(2)
        with p1:
            show_model_image(
                f"classification_shap_bar_{suffix_cls}.png",
                f"Classification SHAP importance — {config_cls.lower()}",
            )
        with p2:
            show_model_image(
                f"classification_shap_summary_{suffix_cls}.png",
                f"Classification SHAP summary — {config_cls.lower()}",
            )
        importance_cls = read_report(f"classification_shap_importance_{suffix_cls}.csv")
        if importance_cls is not None:
            with st.expander("View classification SHAP importance values"):
                st.dataframe(importance_cls, use_container_width=True, hide_index=True)

    st.info("Scope note: SHAP and feature-importance outputs are available. ")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption("EV Charging Demand Prediction and Usage Pattern Analysis ·")
