import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import numpy as np
import joblib
import os

# Load dataset
df = pd.read_csv("data/features_baddi.csv")

features = [
    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_7",
    "aqi_7day_mean",
    "aqi_7day_std",
    "day_of_year",
    "week"
]

X = df[features]
y = df["aqi"]

# Chronological split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
}

results = []

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n{name}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))

    if name == "Linear Regression":
        joblib.dump(model, "models/aqi_model.pkl")

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("data/model_results.csv", index=False)
