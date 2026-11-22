import pandas as pd
import numpy as np

df = pd.read_csv("data/cleaned_baddi_aqi.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Temporal features
df["day_of_year"] = df["date"].dt.day_of_year
df["week"] = df["date"].dt.isocalendar().week.astype(int)

# Lag features: Previous observations
df["aqi_lag_1"] = df["aqi"].shift(1)
df["aqi_lag_2"] = df["aqi"].shift(2)
df["aqi_lag_7"] = df["aqi"].shift(7)

# Rolling statistics
df["aqi_7day_mean"] = df["aqi"].shift(1).rolling(7).mean()
df["aqi_7day_std"] = df["aqi"].shift(1).rolling(7).std()

# Remove rows with unavailable historical features
df = df.dropna().reset_index(drop=True)

df.to_csv("data/features_baddi.csv", index=False)

print("Feature engineering completed!")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nCorrelation matrix:")
print(df[[
    "aqi",
    "aqi_lag_1",
    "aqi_lag_7",
    "aqi_7day_mean"
]].corr())