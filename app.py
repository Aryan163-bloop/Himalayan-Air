import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from forecast_features import FEATURES, build_latest_forecast_row

st.set_page_config(page_title="Himalayan AirSense", page_icon="🌄", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetric"] {
        background-color: #1c2333;
        padding: 18px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = "data/cleaned_baddi_aqi.csv"
MODEL_PATH = "models/forecast_model.pkl"
METRICS_PATH = "data/forecast_metrics.json"
PREDICTIONS_PATH = "data/forecast_test_predictions.csv"


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run preprocessing.py first.")
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["aqi"] = pd.to_numeric(data["aqi"], errors="coerce")
    data = data.dropna(subset=["date", "aqi"])
    data = data.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["season"] = data["month"].map(
        lambda m: "Winter" if m in [12, 1, 2]
        else "Summer" if m in [3, 4, 5]
        else "Monsoon" if m in [6, 7, 8, 9]
        else "Post-Monsoon"
    )
    return data


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def metric_row(metrics, labels=("MAE", "RMSE", "R2")):
    c1, c2, c3 = st.columns(3)
    c1.metric(labels[0], f"{metrics['MAE']:.2f}")
    c2.metric(labels[1], f"{metrics['RMSE']:.2f}")
    c3.metric(labels[2], f"{metrics['R2']:.4f}")


try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

model = load_model()

st.title("🌄 Himalayan AirSense")
st.subheader("Environmental Intelligence & AQI Forecasting Platform")
st.caption("Historical AQI analysis | Experimental next-day forecasting | Baddi, Himachal Pradesh")
st.divider()

st.sidebar.title("🌄 AirSense")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Pollution Analysis", "AQI Forecast", "Model Evaluation", "Anomaly Detection"],
)

if page == "Dashboard":
    st.header("📊 Air Quality Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average AQI", f"{df['aqi'].mean():.2f}")
    col2.metric("Maximum AQI", f"{df['aqi'].max():.0f}")
    col3.metric("Minimum AQI", f"{df['aqi'].min():.0f}")
    col4.metric("Total Observations", f"{len(df):,}")

    latest = df.iloc[-1]
    st.info(
        f"Latest recorded observation: {latest['date'].date()} | "
        f"AQI: {latest['aqi']:.0f} | Category: {classify_aqi(latest['aqi'])}"
    )

    calendar_days = (df["date"].max() - df["date"].min()).days + 1
    missing_days = max(calendar_days - df["date"].nunique(), 0)
    coverage = 100 * df["date"].nunique() / calendar_days if calendar_days else 0
    c1, c2 = st.columns(2)
    c1.metric("Calendar-day coverage", f"{coverage:.1f}%")
    c2.metric("Missing calendar dates", f"{missing_days:,}")

    st.subheader("📈 Historical AQI Trend")
    fig = px.line(df, x="date", y="aqi", title="Recorded Daily AQI", template="plotly_dark")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Recent Observations")
    st.dataframe(df.tail(10), width="stretch")

elif page == "Pollution Analysis":
    st.header("📈 Pollution Analysis")
    col1, col2 = st.columns(2)
    monthly = df.groupby("month", as_index=False)["aqi"].mean()
    fig1 = px.bar(monthly, x="month", y="aqi", title="Average AQI by Calendar Month", template="plotly_dark")
    col1.plotly_chart(fig1, width="stretch")

    yearly = df.groupby("year", as_index=False)["aqi"].mean()
    fig2 = px.line(yearly, x="year", y="aqi", markers=True, title="Yearly Average AQI", template="plotly_dark")
    col2.plotly_chart(fig2, width="stretch")

    st.subheader("Recorded Dominant Pollutant Frequency")
    if "main_pollutant" in df.columns:
        pollutants = df["main_pollutant"].fillna("Unknown").value_counts().rename_axis("Pollutant").reset_index(name="Observations")
        fig3 = px.pie(pollutants, names="Pollutant", values="Observations", hole=0.45, template="plotly_dark")
        st.plotly_chart(fig3, width="stretch")
        st.caption("This chart counts the recorded dominant-pollutant label. It does not show pollutant concentrations.")
    else:
        st.warning("The dataset does not contain a main_pollutant column.")

    seasonal = df.groupby("season", as_index=False)["aqi"].mean()
    fig4 = px.bar(seasonal, x="season", y="aqi", title="Seasonal Average AQI", template="plotly_dark")
    st.plotly_chart(fig4, width="stretch")

elif page == "AQI Forecast":
    st.header("🔮 Experimental Next-Calendar-Day AQI Forecast")
    if model is None:
        st.error("Forecast model not found. Run `python forecast_model.py` after preprocessing.")
    else:
        try:
            input_row, forecast_date = build_latest_forecast_row(df)
            raw_prediction = float(model.predict(input_row[FEATURES])[0])
            prediction = float(np.clip(raw_prediction, 0, 500))

            col1, col2 = st.columns(2)
            col1.metric("Predicted AQI", f"{prediction:.1f}")
            col2.metric("Predicted Category", classify_aqi(prediction))
            st.success(f"Forecast date: {forecast_date.date()}")
            if prediction != raw_prediction:
                st.warning("The raw regression output was outside the 0–500 AQI display range and has been bounded for display.")

            st.caption(
                "Experimental estimate based on the latest available observations. "
                "It is not an official forecast or a public-health alert."
            )
            st.subheader("Features supplied to the model")
            st.dataframe(input_row, width="stretch")
            st.caption(
                "Lag and rolling features refer to available observations, not necessarily consecutive calendar days. "
                "The target being predicted is the next calendar day's AQI."
            )
            st.subheader("Recent AQI History")
            recent = df.tail(30)
            fig = px.line(recent, x="date", y="aqi", markers=True, title="Last 30 Available Observations", template="plotly_dark")
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.error(f"Could not generate a forecast: {exc}")

elif page == "Model Evaluation":
    st.header("🤖 Model Evaluation — Held-out Test Set")
    if not os.path.exists(METRICS_PATH) or not os.path.exists(PREDICTIONS_PATH):
        st.warning("Evaluation artifacts not found. Run `python forecast_model.py` to train and evaluate the model.")
    else:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        comparison = pd.read_csv(PREDICTIONS_PATH)
        test_metrics = results["test_metrics"]
        baseline_metrics = results["persistence_baseline_test_metrics"]

        st.write(f"**Selected model:** {results['selected_model']}")
        st.write(f"**Selection:** {results['selection_method']}")
        st.write(f"**Final test period:** {results['test_period']['start']} to {results['test_period']['end']}")
        st.subheader("Selected model — untouched test set")
        metric_row(test_metrics)

        st.subheader("Persistence baseline — same test dates")
        metric_row(baseline_metrics)
        if test_metrics["MAE"] < baseline_metrics["MAE"]:
            st.success(f"The selected model improves test MAE over persistence by {baseline_metrics['MAE'] - test_metrics['MAE']:.2f} AQI points.")
        else:
            st.warning("The selected model did not beat the simple persistence baseline on test MAE. Do not claim it is better than this baseline.")

        st.subheader("Actual vs Predicted AQI")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comparison["date"], y=comparison["actual_aqi"], name="Actual AQI"))
        fig.add_trace(go.Scatter(x=comparison["date"], y=comparison["predicted_aqi"], name="Model prediction"))
        fig.add_trace(go.Scatter(x=comparison["date"], y=comparison["persistence_prediction"], name="Persistence baseline", line=dict(dash="dot")))
        fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="AQI")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Prediction Residuals")
        fig2 = px.histogram(comparison, x="residual", title="Actual AQI − Predicted AQI", template="plotly_dark")
        st.plotly_chart(fig2, width="stretch")
        st.subheader("Test-set prediction records")
        st.dataframe(comparison, width="stretch")
        st.caption("These metrics come from a chronological hold-out test. The deployed model is subsequently refitted on all labelled data; the displayed test metrics are not calculated using that refitted model.")

elif page == "Anomaly Detection":
    st.header("🚨 Statistical AQI Anomaly Detection")
    q1 = df["aqi"].quantile(0.25)
    q3 = df["aqi"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    anomalies = df[(df["aqi"] < lower) | (df["aqi"] > upper)].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Statistical outliers", len(anomalies))
    col2.metric("Lower IQR threshold", f"{lower:.2f}")
    col3.metric("Upper IQR threshold", f"{upper:.2f}")

    fig = px.scatter(df, x="date", y="aqi", title="AQI Observations and Statistical Thresholds", template="plotly_dark")
    fig.add_hline(y=upper, line_dash="dash", annotation_text="Upper threshold")
    fig.add_hline(y=lower, line_dash="dash", annotation_text="Lower threshold")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Detected Outliers")
    columns = [c for c in ["date", "aqi", "category", "main_pollutant"] if c in anomalies.columns]
    st.dataframe(anomalies[columns], width="stretch")
    st.caption("IQR flags statistically unusual AQI values. An outlier is not automatically an emergency or proof of a measurement error.")

st.divider()
st.subheader("🌡️ Overall AQI Category Distribution")
df["AQI Category"] = df["aqi"].apply(classify_aqi)
category_counts = df["AQI Category"].value_counts().rename_axis("Category").reset_index(name="Observations")
fig = px.bar(category_counts, x="Category", y="Observations", color="Category", title="Distribution of Recorded AQI Categories", template="plotly_dark")
st.plotly_chart(fig, width="stretch")

st.divider()
st.caption("Himalayan AirSense | Historical AQI analysis and experimental forecasting | Baddi, Himachal Pradesh")
