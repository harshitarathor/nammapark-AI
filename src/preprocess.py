import pandas as pd
import numpy as np
from pathlib import Path
 
# ==========================================================
# File Paths
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
 
INPUT_FILE = BASE_DIR / "data" / "ps1_violations.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "df_clean.csv"
 
OUTPUT_DIR.mkdir(exist_ok=True)
 
# ==========================================================
# Load Dataset
# ==========================================================
print("=" * 60)
print("Loading dataset...")
print("=" * 60)
 
df = pd.read_csv(INPUT_FILE)
 
print(f"Dataset Shape : {df.shape}")
print()
 
# ==========================================================
# Remove Duplicates
# ==========================================================
before = len(df)
df = df.drop_duplicates()
after = len(df)
 
print(f"Duplicates Removed : {before - after}")
print(f"Current Shape      : {df.shape}")
print()
 
# ==========================================================
# Datetime Conversion
# ==========================================================
print("Converting datetime columns...")
 
datetime_cols = [
    "created_datetime",
    "closed_datetime",
    "modified_datetime",
    "action_taken_timestamp",
    "data_sent_to_scita_timestamp",
    "validation_timestamp"
]
 
for col in datetime_cols:
    if col in df.columns:
        df[col] = (
            pd.to_datetime(
                df[col],
                errors="coerce",
                utc=True
            )
            .dt.tz_localize(None)
        )
 
print("Datetime conversion completed.")
print()
 
# ==========================================================
# Coordinate Cleaning
# ==========================================================
print("Cleaning coordinates...")
 
df = df.dropna(subset=["latitude", "longitude"])
 
df = df[
    (df["latitude"] != 0)
    & (df["longitude"] != 0)
]
 
print(f"Shape after coordinate cleaning : {df.shape}")
print()
 
# ==========================================================
# Text Cleaning
# ==========================================================
text_cols = [
    "location",
    "vehicle_type",
    "violation_type",
    "police_station",
    "junction_name",
    "updated_vehicle_type",
    "validation_status"
]
 
for col in text_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.title()
        )
 
print("Text columns cleaned.")
print()
 
# ==========================================================
# Time Feature Engineering
# ==========================================================
print("Creating time features...")
 
df["hour"] = (
    df["created_datetime"]
    .dt.hour
    .fillna(-1)
    .astype(int)
)
 
df["day_of_week"] = (
    df["created_datetime"]
    .dt.dayofweek
    .fillna(-1)
    .astype(int)
)
 
df["day_name"] = (
    df["created_datetime"]
    .dt.day_name()
)
 
df["month"] = (
    df["created_datetime"]
    .dt.month
    .fillna(-1)
    .astype(int)
)
 
df["month_name"] = (
    df["created_datetime"]
    .dt.month_name()
)
 
df["year"] = (
    df["created_datetime"]
    .dt.year
    .fillna(-1)
    .astype(int)
)
 
# isocalendar().week returns an unsigned integer (UInt32) dtype,
# which cannot hold -1. Cast to a nullable signed integer first.
df["week"] = (
    df["created_datetime"]
    .dt
    .isocalendar()
    .week
    .astype("Int64")
    .fillna(-1)
    .astype(int)
)
 
df["date"] = (
    df["created_datetime"]
    .dt.date
)
 
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)
 
peak_hours = [8, 9, 10, 17, 18, 19, 20]
 
df["is_peak_hour"] = (
    df["hour"]
    .isin(peak_hours)
).astype(int)
 
print("Time features created.")
print()
 
# ==========================================================
# Duration Features
# ==========================================================
print("Creating duration features...")
 
# Dataset does not contain valid closed timestamps,
# so keep these columns as missing.
 
df["duration_minutes"] = np.nan
df["resolution_time_hours"] = np.nan
 
print("Duration features skipped (timestamps unavailable).")
print()
 
# ==========================================================
# Vehicle Features
# ==========================================================
print("Creating vehicle features...")
 
df["vehicle_number"] = (
    df["vehicle_number"]
    .astype(str)
    .str.upper()
    .str.strip()
)
 
df["is_vehicle_number_missing"] = (
    df["vehicle_number"]
    .isin(["NAN", "NONE", "", "NULL"])
).astype(int)
 
vehicle_counts = (
    df["vehicle_number"]
    .value_counts()
)
 
df["vehicle_repeat_count"] = (
    df["vehicle_number"]
    .map(vehicle_counts)
)
 
print("Vehicle features created.")
print()
 
# ==========================================================
# Missing Value Summary
# ==========================================================
print("=" * 60)
print("Top Missing Values")
print("=" * 60)
 
print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .head(15)
)
 
print()
 
# ==========================================================
# Dataset Summary
# ==========================================================
print("=" * 60)
print("Final Dataset Information")
print("=" * 60)
 
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print()
 
# ==========================================================
# Save Cleaned Dataset
# ==========================================================
df.to_csv(
    OUTPUT_FILE,
    index=False
)
 
print("=" * 60)
print("Preprocessing Completed Successfully")
print("=" * 60)
print(f"Saved to : {OUTPUT_FILE}")
print()
