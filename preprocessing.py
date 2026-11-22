import pandas as pd
import os

# Load dataset
df = pd.read_csv(
    "data/baddi-daily-aqi.csv",
    skiprows=2,
    header=None,
    names=[
        "date",
        "aqi",
        "category",
        "main_pollutant",
        "coverage"
    ]
)

# Convert datatypes
df["date"] = pd.to_datetime(df["date"])
df["aqi"] = pd.to_numeric(df["aqi"])

# Sort chronologically
df = df.sort_values("date")

# Remove duplicate observations
df = df.drop_duplicates(subset="date")

# Feature engineering
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter

# Seasonal classification
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df["season"] = df["month"].apply(get_season)

# Save processed data
os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/cleaned_baddi_aqi.csv",
    index=False
)

print("Preprocessing completed!")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nStatistical summary:")
print(df["aqi"].describe())