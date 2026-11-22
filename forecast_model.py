"""Train and evaluate the Himalayan AirSense next-calendar-day AQI model."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from forecast_features import FEATURES, build_supervised_data

DATA_PATH = "data/cleaned_baddi_aqi.csv"
MODEL_PATH = "models/forecast_model.pkl"
METRICS_PATH = "data/forecast_metrics.json"
PREDICTIONS_PATH = "data/forecast_test_predictions.csv"
METADATA_PATH = "models/forecast_metadata.json"


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run preprocessing.py first."
        )

    raw = pd.read_csv(DATA_PATH)
    data = build_supervised_data(raw)

    if len(data) < 30:
        raise ValueError(
            f"Only {len(data)} usable next-day samples found; at least 30 are required."
        )

    X = data[FEATURES]
    y = data["target"]

    # Chronological 70% train / 10% validation / 20% final test.
    n = len(data)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.80)

    if train_end < 10 or validation_end <= train_end or validation_end >= n:
        raise ValueError("Not enough samples for chronological train/validation/test splits.")

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:validation_end], y.iloc[train_end:validation_end]
    X_train_val, y_train_val = X.iloc[:validation_end], y.iloc[:validation_end]
    X_test, y_test = X.iloc[validation_end:], y.iloc[validation_end:]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    validation_results = {}
    best_name = None
    best_validation_mae = float("inf")

    print(f"Usable next-calendar-day samples: {n}")
    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        val_predictions = candidate.predict(X_val)
        metrics = calculate_metrics(y_val, val_predictions)
        validation_results[name] = metrics
        print(f"\n{name} validation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        if metrics["MAE"] < best_validation_mae:
            best_validation_mae = metrics["MAE"]
            best_name = name

    # Final honest test: selected model is refit on train + validation only.
    selected_for_test = candidates[best_name]
    selected_for_test.fit(X_train_val, y_train_val)
    test_predictions = selected_for_test.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    # Persistence baseline: tomorrow's AQI is predicted as today's AQI.
    baseline_predictions = X_test["lag_1"].to_numpy()
    baseline_metrics = calculate_metrics(y_test, baseline_predictions)

    print(f"\nSelected using validation MAE: {best_name}")
    print("Final untouched test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("Persistence baseline test metrics:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save final test predictions from a model that did not train on test data.
    test_output = pd.DataFrame({
        "date": data["next_date"].iloc[validation_end:].dt.strftime("%Y-%m-%d").to_numpy(),
        "actual_aqi": y_test.to_numpy(),
        "predicted_aqi": test_predictions,
        "persistence_prediction": baseline_predictions,
        "residual": y_test.to_numpy() - test_predictions,
    })
    test_output.to_csv(PREDICTIONS_PATH, index=False)

    # Refit the selected model on all labelled data for deployment only.
    deployment_model = candidates[best_name]
    deployment_model.fit(X, y)
    joblib.dump(deployment_model, MODEL_PATH)

    metrics_payload = {
        "selected_model": best_name,
        "selection_method": "Lowest validation MAE; chronological split",
        "sample_counts": {
            "total": int(n),
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "persistence_baseline_test_metrics": baseline_metrics,
        "test_period": {
            "start": str(data["next_date"].iloc[validation_end].date()),
            "end": str(data["next_date"].iloc[-1].date()),
        },
        "target_definition": "AQI on the next calendar day; samples without a next-day observation are excluded.",
        "feature_definition": {
            "lag_1": "AQI on current observation date",
            "lag_2": "AQI from previous available observation",
            "lag_7": "AQI from seven available observations earlier",
            "rolling_mean": "Mean AQI over current and previous six available observations",
        },
        "deployment_model_training": "Refitted on all labelled observations after final test evaluation; do not use this deployment model to recompute held-out test metrics.",
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    metadata = {
        "model": best_name,
        "features": FEATURES,
        "target": "next calendar day's AQI",
        "training_rows": int(n),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved deployment model: {MODEL_PATH}")
    print(f"Saved honest test metrics: {METRICS_PATH}")
    print(f"Saved test predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
