"""Shared feature engineering for Himalayan AirSense next-day AQI forecasts.

All lag features are based on prior *available observations*, not necessarily
consecutive calendar days. The target is restricted to the next calendar day.
"""

from __future__ import annotations

import pandas as pd

FEATURES = [
    "day_of_year",
    "month",
    "lag_1",
    "lag_2",
    "lag_7",
    "rolling_mean",
]


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return chronologically sorted observations with inference-safe features.

    Definitions:
    - lag_1: AQI on the current/latest observation date.
    - lag_2: AQI from the immediately preceding available observation.
    - lag_7: AQI from seven available observations earlier.
    - rolling_mean: mean AQI over the latest seven available observations,
      including the current observation.

    Features are computed BEFORE filtering for consecutive-day targets, so
    training and live inference use identical definitions.
    """
    required = {"date", "aqi"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df = df.dropna(subset=["date", "aqi"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid date/AQI observations found.")

    df["day_of_year"] = df["date"].dt.day_of_year
    df["month"] = df["date"].dt.month
    df["lag_1"] = df["aqi"]
    df["lag_2"] = df["aqi"].shift(1)
    df["lag_7"] = df["aqi"].shift(7)
    df["rolling_mean"] = df["aqi"].rolling(window=7, min_periods=7).mean()
    return df


def build_supervised_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create next-calendar-day targets using the shared feature definitions."""
    df = build_feature_frame(data)
    df["target"] = df["aqi"].shift(-1)
    df["next_date"] = df["date"].shift(-1)

    is_next_calendar_day = (df["next_date"] - df["date"]).dt.days.eq(1)
    df = df.loc[is_next_calendar_day].dropna(subset=FEATURES + ["target"])
    return df.reset_index(drop=True)


def build_latest_forecast_row(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build the exact feature row used for the next calendar day forecast."""
    featured = build_feature_frame(data)
    latest = featured.iloc[-1]
    if latest[FEATURES].isna().any():
        raise ValueError("At least eight valid observations are required to forecast.")

    row = pd.DataFrame([{name: latest[name] for name in FEATURES}], columns=FEATURES)
    forecast_date = pd.Timestamp(latest["date"]) + pd.Timedelta(days=1)
    return row, forecast_date
