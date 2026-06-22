import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "outputs" / "df_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================
print("=" * 60)
print("Loading cleaned dataset...")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset Shape : {df.shape}")
print()

# ==========================================================
# SCHEMA VALIDATION
# ==========================================================
required_cols = [
    "hour",
    "day_name",
    "month_name",
    "vehicle_type",
    "violation_type",
    "vehicle_number",
    "junction_name",
    "police_station",
    "created_datetime"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# ==========================================================
# CLEAN VIOLATION TYPE STRINGS
# ==========================================================
df["violation_type"] = (
    df["violation_type"]
    .astype(str)
    .str.replace(r"[\[\]\"]", "", regex=True)
    .str.strip()
)

# ==========================================================
# PARSE DATETIME FOR TREND ANALYSIS
# ==========================================================
df["created_datetime"] = pd.to_datetime(
    df["created_datetime"], errors="coerce"
)

n_bad_dates = df["created_datetime"].isna().sum()
if n_bad_dates > 0:
    print(f"Warning: {n_bad_dates} rows had unparseable created_datetime and will be excluded from trend analysis.")
    print()

df["year_month"] = df["created_datetime"].dt.to_period("M")

# ==========================================================
# CREATE DATA WITHOUT "NO JUNCTION"
# ==========================================================
df_junction = df[
    df["junction_name"].notna()
    & (df["junction_name"] != "No Junction")
].copy()

# ==========================================================
# BASIC INFORMATION
# ==========================================================
print("=" * 60)
print("Basic Information")
print("=" * 60)

print(f"Total Violations : {len(df):,}")
print(f"Unique Vehicles  : {df['vehicle_number'].nunique():,}")
print(f"Unique Junctions : {df_junction['junction_name'].nunique():,}")
print(f"Police Stations  : {df['police_station'].nunique():,}")

print()

# ==========================================================
# VIOLATIONS PER VEHICLE (KPI)
# ==========================================================
violation_per_vehicle = (
    len(df) / df["vehicle_number"].nunique()
)

print("=" * 60)
print("Behavioral KPI")
print("=" * 60)
print(f"Violations per Vehicle : {violation_per_vehicle:.2f}")
print()

# ==========================================================
# TOP JUNCTIONS (BY FREQUENCY)
# ==========================================================
top_junctions = (
    df_junction["junction_name"]
    .value_counts()
    .head(10)
)

print("=" * 60)
print("Top 10 Junctions (by Frequency)")
print("=" * 60)
print(top_junctions)
print()

# ==========================================================
# TOP JUNCTIONS (BY SEVERITY, IF AVAILABLE)
# ==========================================================
if "IPII" in df_junction.columns:

    junction_severity = (
        df_junction.groupby("junction_name")["IPII"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    print("=" * 60)
    print("Top 10 Junctions (by Average IPII Severity)")
    print("=" * 60)
    print(junction_severity)
    print()

else:
    print("Note: IPII column not present at EDA stage, skipping severity ranking.")
    print()

# ==========================================================
# JUNCTION ANALYTICS: NORMALIZED HOTSPOT SCORES
# ==========================================================
junction_counts_all = df_junction["junction_name"].value_counts()

total_junction_violations = junction_counts_all.sum()

hotspot_score = (
    junction_counts_all / total_junction_violations
)

junction_mean = junction_counts_all.mean()
junction_std = junction_counts_all.std()

junction_zscore = (
    (junction_counts_all - junction_mean) / junction_std
)

junction_analytics = pd.DataFrame({
    "violation_count": junction_counts_all,
    "pct_contribution": (hotspot_score * 100).round(2),
    "z_score": junction_zscore.round(2)
}).sort_values("violation_count", ascending=False)

print("=" * 60)
print("Top 10 Junctions - Normalized Hotspot Scores")
print("=" * 60)
print(junction_analytics.head(10))
print()

# ==========================================================
# CONCENTRATION ANALYSIS (PARETO-STYLE)
# ==========================================================
n_junctions = len(junction_counts_all)
top_5pct_n = max(1, int(round(n_junctions * 0.05)))
top_10pct_n = max(1, int(round(n_junctions * 0.10)))

top_5pct_share = (
    junction_counts_all.head(top_5pct_n).sum()
    / total_junction_violations
) * 100

top_10pct_share = (
    junction_counts_all.head(top_10pct_n).sum()
    / total_junction_violations
) * 100

print("=" * 60)
print("Concentration Analysis (Pareto-style)")
print("=" * 60)
print(f"Total junctions             : {n_junctions}")
print(f"Top 5% junctions ({top_5pct_n})  contribute : {top_5pct_share:.2f}% of violations")
print(f"Top 10% junctions ({top_10pct_n}) contribute : {top_10pct_share:.2f}% of violations")
print()

# ==========================================================
# UNIFIED JUNCTION SEVERITY SCORE
# (volume x repeat rate x violation diversity)
# ==========================================================
repeat_vehicle_counts = (
    df_junction.groupby(["junction_name", "vehicle_number"])
    .size()
)

repeat_rate_per_junction = (
    repeat_vehicle_counts[repeat_vehicle_counts > 1]
    .groupby(level=0)
    .size()
    / df_junction.groupby("junction_name")["vehicle_number"].nunique()
).fillna(0)

diversity_per_junction = (
    df_junction.groupby("junction_name")["violation_type"]
    .nunique()
)

severity_df = pd.DataFrame({
    "volume": junction_counts_all,
    "repeat_rate": repeat_rate_per_junction,
    "violation_diversity": diversity_per_junction
}).fillna(0)

# normalize each component to 0-1 before combining, so no single
# factor dominates purely due to scale differences
for col in ["volume", "repeat_rate", "violation_diversity"]:
    col_max = severity_df[col].max()
    severity_df[col + "_norm"] = (
        severity_df[col] / col_max if col_max > 0 else 0
    )

severity_df["severity_score"] = (
    severity_df["volume_norm"]
    * (1 + severity_df["repeat_rate_norm"])
    * (1 + severity_df["violation_diversity_norm"])
).round(3)

severity_df = severity_df.sort_values(
    "severity_score", ascending=False
)

print("=" * 60)
print("Top 10 Junctions - Unified Severity Score")
print("(volume x repeat-rate x violation-diversity)")
print("=" * 60)
print(
    severity_df[
        ["volume", "repeat_rate", "violation_diversity", "severity_score"]
    ].head(10)
)
print()

severity_df.to_csv(
    OUTPUT_DIR / "junction_severity_score.csv"
)

# ==========================================================
# UNIFIED HOTSPOT SCORE (composite decision metric)
# combines z-score, pct contribution, severity, repeat rate
# ==========================================================
hotspot_components = pd.DataFrame({
    "z_score": junction_zscore,
    "pct_contribution": hotspot_score,  # 0-1 fraction
    "severity_score": severity_df["severity_score"],
    "repeat_rate": severity_df["repeat_rate"]
}).fillna(0)

# min-max normalize each component to 0-1 so weights behave as intended
# (z_score can be negative, severity/pct_contribution are on different
# scales — combining raw values would let one factor dominate by scale
# alone rather than by genuine risk signal)
for col in hotspot_components.columns:
    col_min = hotspot_components[col].min()
    col_max = hotspot_components[col].max()
    span = col_max - col_min
    hotspot_components[col + "_norm"] = (
        (hotspot_components[col] - col_min) / span if span > 0 else 0
    )

hotspot_components["hotspot_score_composite"] = (
    0.35 * hotspot_components["z_score_norm"]
    + 0.25 * hotspot_components["pct_contribution_norm"]
    + 0.25 * hotspot_components["severity_score_norm"]
    + 0.15 * hotspot_components["repeat_rate_norm"]
).round(3)

hotspot_components = hotspot_components.sort_values(
    "hotspot_score_composite", ascending=False
)

print("=" * 60)
print("Unified Hotspot Score (composite: z-score + pct + severity + repeat-rate)")
print("=" * 60)
print(
    hotspot_components[
        ["z_score", "pct_contribution", "severity_score", "repeat_rate", "hotspot_score_composite"]
    ].head(10)
)
print()

hotspot_components.to_csv(
    OUTPUT_DIR / "junction_hotspot_score_composite.csv"
)

# ==========================================================
# CORRELATION: VEHICLE TYPE vs VIOLATION TYPE
# ==========================================================
vehicle_violation_corr = (
    df.groupby("vehicle_type")["violation_type"]
    .value_counts()
    .groupby(level=0)
    .head(3)
)

print("=" * 60)
print("Top Violation Types per Vehicle Type")
print("=" * 60)
print(vehicle_violation_corr)
print()

# ==========================================================
# DOMINANT VIOLATION PER VEHICLE TYPE (BEHAVIORAL CLUSTERING)
# ==========================================================
dominant_violation_per_vehicle = (
    df.groupby("vehicle_type")["violation_type"]
    .agg(lambda x: x.value_counts().idxmax())
)

print("=" * 60)
print("Dominant Violation Type per Vehicle Type")
print("=" * 60)
for vtype, viol in dominant_violation_per_vehicle.items():
    print(f"{vtype} -> {viol} (dominant)")
print()

dominant_violation_per_vehicle.to_csv(
    OUTPUT_DIR / "dominant_violation_per_vehicle.csv"
)

# ==========================================================
# TOP VIOLATION TYPES
# ==========================================================
top_violations = (
    df["violation_type"]
    .value_counts()
    .head(10)
)

print("=" * 60)
print("Top Violation Types")
print("=" * 60)
print(top_violations)
print()

# ==========================================================
# TOP VEHICLE TYPES
# ==========================================================
top_vehicle_types = (
    df["vehicle_type"]
    .value_counts()
    .head(10)
)

print("=" * 60)
print("Top Vehicle Types")
print("=" * 60)
print(top_vehicle_types)
print()

# ==========================================================
# TOP POLICE STATIONS
# ==========================================================
top_police = (
    df["police_station"]
    .value_counts()
    .head(10)
)

print("=" * 60)
print("Top Police Stations")
print("=" * 60)
print(top_police)
print()

# ==========================================================
# POLICE STATION RISK RANKING
# risk_score = 0.5*volume + 0.3*repeat_rate + 0.2*diversity
# ==========================================================
station_volume = df["police_station"].value_counts()

station_repeat_vehicle_counts = (
    df.groupby(["police_station", "vehicle_number"]).size()
)

station_repeat_rate = (
    station_repeat_vehicle_counts[station_repeat_vehicle_counts > 1]
    .groupby(level=0)
    .size()
    / df.groupby("police_station")["vehicle_number"].nunique()
).fillna(0)

station_diversity = (
    df.groupby("police_station")["violation_type"].nunique()
)

station_risk_df = pd.DataFrame({
    "volume": station_volume,
    "repeat_rate": station_repeat_rate,
    "violation_diversity": station_diversity
}).fillna(0)

for col in ["volume", "repeat_rate", "violation_diversity"]:
    col_max = station_risk_df[col].max()
    station_risk_df[col + "_norm"] = (
        station_risk_df[col] / col_max if col_max > 0 else 0
    )

station_risk_df["risk_score"] = (
    0.5 * station_risk_df["volume_norm"]
    + 0.3 * station_risk_df["repeat_rate_norm"]
    + 0.2 * station_risk_df["violation_diversity_norm"]
).round(3)

def station_tier(score):
    if score > 0.7:
        return "Critical"
    elif score > 0.5:
        return "High"
    elif score > 0.3:
        return "Moderate"
    return "Low"

station_risk_df["tier"] = (
    station_risk_df["risk_score"].apply(station_tier)
)

station_risk_df = station_risk_df.sort_values(
    "risk_score", ascending=False
)

print("=" * 60)
print("Police Station Risk Ranking (Top 10)")
print("=" * 60)
print(
    station_risk_df[
        ["volume", "risk_score", "tier"]
    ].head(10)
)
print()

station_risk_df.to_csv(
    OUTPUT_DIR / "police_station_risk_ranking.csv"
)

# ==========================================================
# TOP REPEAT OFFENDERS
# ==========================================================
repeat_offenders = (
    df.groupby("vehicle_number")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("=" * 60)
print("Top Repeat Offenders")
print("=" * 60)
print(repeat_offenders)
print()

# ==========================================================
# REPEAT-OFFENDER CONCENTRATION (TOP 1% SHARE)
# ==========================================================
all_vehicle_counts = (
    df.groupby("vehicle_number").size().sort_values(ascending=False)
)

n_vehicles = len(all_vehicle_counts)
top_1pct_n = max(1, int(round(n_vehicles * 0.01)))

top_1pct_share = (
    all_vehicle_counts.head(top_1pct_n).sum()
    / all_vehicle_counts.sum()
) * 100

print("=" * 60)
print("Repeat-Offender Concentration")
print("=" * 60)
print(f"Total unique vehicles      : {n_vehicles:,}")
print(f"Top 1% of vehicles ({top_1pct_n}) account for {top_1pct_share:.2f}% of all violations.")
print()

# ==========================================================
# VEHICLE RISK SCORE
# (frequency + violation diversity + weighted severity)
# ==========================================================
vehicle_violation_counts = (
    df.groupby("vehicle_number").size()
)

vehicle_diversity = (
    df.groupby("vehicle_number")["violation_type"].nunique()
)

# weight each violation type by how rare/severe it is overall:
# rarer violation types get a higher severity weight
violation_freq = df["violation_type"].value_counts()
violation_severity_weight = (
    1 - (violation_freq / violation_freq.max())
).clip(lower=0.1)  # floor so no violation type contributes zero weight

vehicle_weighted_severity = (
    df.groupby("vehicle_number")["violation_type"]
    .apply(lambda types: violation_severity_weight.reindex(types).sum())
)

vehicle_risk_df = pd.DataFrame({
    "violation_count": vehicle_violation_counts,
    "violation_diversity": vehicle_diversity,
    "weighted_severity": vehicle_weighted_severity
}).fillna(0)

vehicle_risk_df["vehicle_risk_score"] = (
    np.log1p(vehicle_risk_df["violation_count"])
    + vehicle_risk_df["violation_diversity"]
    + vehicle_risk_df["weighted_severity"]
).round(3)

vehicle_risk_df = vehicle_risk_df.sort_values(
    "vehicle_risk_score", ascending=False
)

print("=" * 60)
print("Top 10 Vehicles - Vehicle Risk Score")
print("(log(count) + diversity + weighted severity)")
print("=" * 60)
print(vehicle_risk_df.head(10))
print()

vehicle_risk_df.to_csv(
    OUTPUT_DIR / "vehicle_risk_score.csv"
)

# ==========================================================
# KILLER STATISTIC
# ==========================================================
top_10_share = (
    top_junctions.sum()
    / len(df_junction)
) * 100

print("=" * 60)
print("KILLER STATISTIC")
print("=" * 60)

print(
    f"Top 10 mapped junctions account for "
    f"{top_10_share:.2f}% "
    f"of all mapped parking violations."
)

print()

# ==========================================================
# MONTH-OVER-MONTH TREND ANALYSIS (per junction)
# ==========================================================
junction_monthly = df_junction.dropna(subset=["year_month"])

available_months = sorted(junction_monthly["year_month"].unique())

print("=" * 60)
print("Months of Data Available")
print("=" * 60)
print(f"Months of data available : {len(available_months)}")
print()

# ----------------------------------------------------------
# PARTIAL MONTH DETECTION
# the most recent month is often a partial month (data extraction
# mid-month) — including it in trend/forecast math creates a fake
# "drop" or "spike" that has nothing to do with real behavior
# ----------------------------------------------------------
if len(available_months) >= 1:

    latest_month = available_months[-1]
    days_with_data = (
        junction_monthly[junction_monthly["year_month"] == latest_month]
        ["created_datetime"].dt.day.nunique()
    )
    days_in_month = latest_month.days_in_month

    coverage_ratio = days_with_data / days_in_month

    if coverage_ratio < 0.8:
        print(f"Latest month ({latest_month}) has data for {days_with_data} of {days_in_month} days. "
              f"Latest month appears PARTIAL - excluding it from trend analysis.")
        print()
        available_months = available_months[:-1]
        junction_monthly = junction_monthly[
            junction_monthly["year_month"] != latest_month
        ]

if len(available_months) >= 2:

    current_month = available_months[-1]
    previous_month = available_months[-2]

    current_counts = (
        junction_monthly[junction_monthly["year_month"] == current_month]
        ["junction_name"].value_counts()
    )
    previous_counts = (
        junction_monthly[junction_monthly["year_month"] == previous_month]
        ["junction_name"].value_counts()
    )

    trend_df = pd.DataFrame({
        "current_month": current_counts,
        "previous_month": previous_counts
    }).fillna(0)

    trend_df["growth_pct"] = (
        (trend_df["current_month"] - trend_df["previous_month"])
        / trend_df["previous_month"].replace(0, 1)
        * 100
    ).round(1)

    def classify_trend(growth):
        if growth > 25:
            return "Emerging Hotspot"
        elif growth < -15:
            return "Improving"
        return "Stable"

    trend_df["trend"] = trend_df["growth_pct"].apply(classify_trend)

    # ----------------------------------------------------------
    # MINIMUM VOLUME FILTER
    # without this, a 1 -> 13 jump shows as +1200% and drowns out
    # real, statistically meaningful hotspot growth
    # ----------------------------------------------------------
    MIN_COMBINED_VOLUME = 100

    trend_df_filtered = trend_df[
        (trend_df["current_month"] + trend_df["previous_month"]) >= MIN_COMBINED_VOLUME
    ].copy()

    # ----------------------------------------------------------
    # EMERGING HOTSPOT SCORE
    # growth_pct alone overweights tiny junctions; multiplying by
    # log1p(current_month) lets genuine volume back into the ranking
    # ----------------------------------------------------------
    trend_df_filtered["emerging_score"] = (
        trend_df_filtered["growth_pct"]
        * np.log1p(trend_df_filtered["current_month"])
    ).round(1)

    trend_df_filtered = trend_df_filtered.sort_values(
        "emerging_score", ascending=False
    )

    print("=" * 60)
    print(f"Month-over-Month Junction Trend ({previous_month} -> {current_month})")
    print(f"(filtered to junctions with combined volume >= {MIN_COMBINED_VOLUME}, ranked by emerging_score)")
    print("=" * 60)
    print(trend_df_filtered.head(10))
    print()

    if len(trend_df_filtered) == 0:
        print(f"Note: no junctions met the {MIN_COMBINED_VOLUME}-violation threshold; "
              f"consider lowering MIN_COMBINED_VOLUME for this dataset.")
        print()

    trend_df = trend_df_filtered

    trend_df.to_csv(
        OUTPUT_DIR / "junction_month_over_month_trend.csv"
    )

else:
    print("=" * 60)
    print("Month-over-Month Trend Analysis")
    print("=" * 60)
    print("Skipped: fewer than 2 distinct months found in created_datetime.")
    print()
    trend_df = None

# ==========================================================
# PERSISTENT HOTSPOT DETECTION
# (junctions that stay in the top tier across multiple months,
# as opposed to one-off spikes)
# ==========================================================
TOP_N_PER_MONTH = 15

monthly_top_sets = {}

for m in available_months:
    month_counts = (
        junction_monthly[junction_monthly["year_month"] == m]
        ["junction_name"].value_counts()
    )
    monthly_top_sets[m] = set(month_counts.head(TOP_N_PER_MONTH).index)

if len(available_months) >= 2:

    all_seen_junctions = set()
    for s in monthly_top_sets.values():
        all_seen_junctions |= s

    persistence_counts = {
        j: sum(j in monthly_top_sets[m] for m in available_months)
        for j in all_seen_junctions
    }

    persistence_df = pd.Series(persistence_counts, name="months_in_top15").sort_values(
        ascending=False
    )

    persistence_df = persistence_df.to_frame()
    persistence_df["months_observed"] = len(available_months)
    persistence_df["persistence_pct"] = (
        persistence_df["months_in_top15"] / persistence_df["months_observed"] * 100
    ).round(1)

    # persistent hotspot = stayed in the top tier in at least 75% of months observed
    persistence_df["is_persistent_hotspot"] = (
        persistence_df["persistence_pct"] >= 75
    )

    print("=" * 60)
    print(f"Persistent Hotspot Detection (top {TOP_N_PER_MONTH} per month, {len(available_months)} months observed)")
    print("=" * 60)
    print(
        persistence_df[persistence_df["is_persistent_hotspot"]]
        .sort_values("persistence_pct", ascending=False)
        .head(15)
    )
    print()

    persistence_df.to_csv(
        OUTPUT_DIR / "persistent_hotspots.csv"
    )

else:
    persistence_df = None
    print("=" * 60)
    print("Persistent Hotspot Detection")
    print("=" * 60)
    print("Skipped: fewer than 2 distinct months available.")
    print()

# ==========================================================
# FORECASTING: NEXT MONTH VIOLATIONS PER JUNCTION
# simple linear trend over available months (lightweight,
# appropriate given only ~6 months of history — not a substitute
# for a real time-series model, but reasonable for this dataset size)
# ==========================================================
if len(available_months) >= 3:

    monthly_junction_matrix = (
        junction_monthly
        .groupby(["junction_name", "year_month"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=available_months, fill_value=0)
    )

    x = np.arange(len(available_months))

    def forecast_next(row):
        y = row.values
        # simple least-squares linear fit; clip negative forecasts to 0
        slope, intercept = np.polyfit(x, y, 1)
        next_val = slope * len(available_months) + intercept
        return max(0, round(next_val, 1))

    forecast_series = monthly_junction_matrix.apply(forecast_next, axis=1)

    forecast_df = pd.DataFrame({
        "last_month_actual": monthly_junction_matrix[available_months[-1]],
        "forecast_next_month": forecast_series
    })

    forecast_df["forecast_change_pct"] = (
        (forecast_df["forecast_next_month"] - forecast_df["last_month_actual"])
        / forecast_df["last_month_actual"].replace(0, 1)
        * 100
    ).round(1)

    forecast_df = forecast_df.sort_values(
        "forecast_next_month", ascending=False
    )

    print("=" * 60)
    print(f"Forecast: Top 10 Junctions by Predicted Next-Month Violations")
    print(f"(linear trend over {len(available_months)} months — treat as directional, not precise)")
    print("=" * 60)
    print(forecast_df.head(10))
    print()

    forecast_df.to_csv(
        OUTPUT_DIR / "junction_forecast_next_month.csv"
    )

else:
    forecast_df = None
    print("=" * 60)
    print("Forecasting")
    print("=" * 60)
    print("Skipped: need at least 3 months of data for a meaningful linear trend forecast.")
    print()

# ==========================================================
# CHART 1 : HOURLY VIOLATIONS
# ==========================================================
hourly = (
    df["hour"]
    .value_counts()
    .sort_index()
)

# ----------------------------------------------------------
# ANOMALY DETECTION: HOURLY SPIKES (z-score based)
# ----------------------------------------------------------
hourly_mean = hourly.mean()
hourly_std = hourly.std()

hourly_zscore = (
    (hourly - hourly_mean) / hourly_std
)

anomalous_hours = hourly_zscore[hourly_zscore > 1.5]

print("=" * 60)
print("Anomaly Detection - Hourly Spikes (z-score > 1.5)")
print("=" * 60)

if len(anomalous_hours) > 0:
    for hr, z in anomalous_hours.sort_values(ascending=False).items():
        print(f"Hour {hr}:00 -> z-score {z:.2f} (statistically unusual spike)")
else:
    print("No statistically significant hourly spikes detected (all within 1.5 std of mean).")

print()

plt.figure(figsize=(10, 5))
hourly.plot(kind="bar")
plt.title("Violations by Hour")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "hourly_violations.png")
plt.close()

# ==========================================================
# CHART 2 : DAY OF WEEK
# ==========================================================
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_counts = (
    df["day_name"]
    .value_counts()
    .reindex(day_order)
)

plt.figure(figsize=(10, 5))
day_counts.plot(kind="bar")
plt.title("Violations by Day")
plt.xlabel("Day")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "day_violations.png")
plt.close()

# ==========================================================
# CHART 3 : MONTHLY VIOLATIONS
# ==========================================================
month_order = [
    "January", "February", "March",
    "April", "May", "June",
    "July", "August", "September",
    "October", "November", "December"
]

month_counts = (
    df["month_name"]
    .value_counts()
    .reindex(month_order)
)

plt.figure(figsize=(12, 5))
month_counts.plot(kind="bar")
plt.title("Violations by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "monthly_violations.png")
plt.close()

# ==========================================================
# CHART 4 : VEHICLE TYPES
# ==========================================================
plt.figure(figsize=(10, 5))
top_vehicle_types.plot(kind="bar")
plt.title("Top Vehicle Types")
plt.xlabel("Vehicle Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "vehicle_types.png")
plt.close()

# ==========================================================
# CHART 5 : VIOLATION TYPES
# ==========================================================
plt.figure(figsize=(12, 5))
top_violations.plot(kind="bar")
plt.title("Top Violation Types")
plt.xlabel("Violation Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "violation_types.png")
plt.close()

# ==========================================================
# CHART 6 : TOP JUNCTIONS
# ==========================================================
plt.figure(figsize=(12, 6))
top_junctions.plot(kind="bar")
plt.title("Top 10 Junctions")
plt.xlabel("Junction")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "top_junctions.png")
plt.close()

# ==========================================================
# CHART 7 : DAY x HOUR HEATMAP
# ==========================================================
pivot = df.pivot_table(
    index="day_name",
    columns="hour",
    aggfunc="size",
    fill_value=0
)

pivot = pivot.reindex(day_order)

plt.figure(figsize=(12, 6))
plt.imshow(pivot, aspect="auto", cmap="viridis")
plt.title("Violation Heatmap (Day vs Hour)")
plt.xlabel("Hour")
plt.ylabel("Day")
plt.xticks(
    range(len(pivot.columns)),
    pivot.columns
)
plt.yticks(
    range(len(pivot.index)),
    pivot.index
)
plt.colorbar(label="Violation Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "heatmap.png")
plt.close()

# ==========================================================
# CHART 8 : PARETO CHART (junction concentration)
# ==========================================================
pareto_counts = junction_counts_all.sort_values(ascending=False)
pareto_cumulative_pct = (
    pareto_counts.cumsum() / pareto_counts.sum() * 100
)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(
    range(len(pareto_counts)),
    pareto_counts.values,
    color="steelblue"
)
ax1.set_xlabel("Junctions (ranked by violation count)")
ax1.set_ylabel("Violation Count", color="steelblue")
ax1.set_xticks([])

ax2 = ax1.twinx()
ax2.plot(
    range(len(pareto_cumulative_pct)),
    pareto_cumulative_pct.values,
    color="darkorange",
    marker="o",
    markersize=2
)
ax2.set_ylabel("Cumulative % of Violations", color="darkorange")
ax2.axhline(80, color="gray", linestyle="--", linewidth=1)

plt.title("Pareto Chart - Junction Violation Concentration")
plt.tight_layout()
plt.savefig(CHART_DIR / "pareto_chart.png")
plt.close()

# ==========================================================
# SAVE TABLES
# ==========================================================
top_junctions.to_csv(
    OUTPUT_DIR / "top_junctions.csv"
)

top_violations.to_csv(
    OUTPUT_DIR / "top_violations.csv"
)

top_vehicle_types.to_csv(
    OUTPUT_DIR / "top_vehicle_types.csv"
)

repeat_offenders.to_csv(
    OUTPUT_DIR / "top_repeat_offenders.csv"
)

if "IPII" in df_junction.columns:
    junction_severity.to_csv(
        OUTPUT_DIR / "top_junctions_by_severity.csv"
    )

junction_analytics.to_csv(
    OUTPUT_DIR / "junction_hotspot_analytics.csv"
)

# ==========================================================
# INTELLIGENCE LAYER: RISK CATEGORIZATION (threshold-based
# policy engine, driven by the unified composite hotspot score)
# ==========================================================
def classify_hotspot(score):

    if score > 0.6:
        return "CRITICAL HOTSPOT"

    elif score > 0.4:
        return "HIGH HOTSPOT"

    elif score > 0.2:
        return "MODERATE HOTSPOT"

    return "LOW HOTSPOT"

junction_analytics["composite_score"] = (
    hotspot_components["hotspot_score_composite"]
    .reindex(junction_analytics.index)
)

junction_analytics["risk_label"] = (
    junction_analytics["composite_score"]
    .apply(classify_hotspot)
)

# ==========================================================
# INTELLIGENCE LAYER: DECISION OUTPUT TABLE
# (multi-condition, data-driven recommendation engine)
# ==========================================================

# pull in the supporting signals already computed elsewhere in the
# script, instead of inventing new ones, so the recommendation logic
# stays consistent with the rest of the analysis
junction_analytics["repeat_rate"] = (
    severity_df["repeat_rate"].reindex(junction_analytics.index)
).fillna(0)

junction_analytics["diversity_norm"] = (
    severity_df["violation_diversity_norm"].reindex(junction_analytics.index)
).fillna(0)

if trend_df is not None and len(trend_df) > 0:
    junction_analytics["growth_pct"] = (
        trend_df["growth_pct"].reindex(junction_analytics.index)
    ).fillna(0)
else:
    junction_analytics["growth_pct"] = 0

# ==========================================================
# HOTSPOT PRIORITY INDEX (HPI)
# composite_score already captures current-state risk (z-score,
# pct contribution, severity, repeat rate) — HPI folds in growth
# so a junction that is becoming worse outranks one that is merely
# currently bad but flat or declining
# ==========================================================
growth_clipped = junction_analytics["growth_pct"].clip(lower=-50, upper=200)
growth_min = growth_clipped.min()
growth_max = growth_clipped.max()
growth_span = growth_max - growth_min

junction_analytics["growth_norm"] = (
    (growth_clipped - growth_min) / growth_span if growth_span > 0 else 0
)

junction_analytics["HPI"] = (
    0.7 * junction_analytics["composite_score"]
    + 0.3 * junction_analytics["growth_norm"]
).round(3)

print("=" * 60)
print("Hotspot Priority Index (HPI) - Top 10")
print("(0.7 x composite_score + 0.3 x growth_norm)")
print("=" * 60)
print(
    junction_analytics
    .sort_values("HPI", ascending=False)
    [["violation_count", "composite_score", "growth_pct", "HPI", "risk_label"]]
    .head(10)
)
print()

junction_analytics.to_csv(
    OUTPUT_DIR / "junction_hotspot_priority_index.csv"
)

def recommend_action(row):

    actions = []

    if row["composite_score"] > 0.8:
        actions.append("Permanent camera")

    if row["repeat_rate"] > 0.25:
        actions.append("Repeat-offender enforcement")

    if row["growth_pct"] > 30:
        actions.append("Emerging hotspot monitoring")

    if row["diversity_norm"] > 0.6:
        actions.append("Multi-violation enforcement")

    if actions:
        return " + ".join(actions)

    # fall back to the simpler risk-label rule if no specific
    # condition fired (keeps every junction with an actionable output)
    label = row["risk_label"]

    if label == "CRITICAL HOTSPOT":
        return "Deploy camera + immediate patrol"
    elif label == "HIGH HOTSPOT":
        return "Increase patrol frequency"
    elif label == "MODERATE HOTSPOT":
        return "Monitor weekly"

    return "Routine monitoring"

decision_table = (
    junction_analytics
    .sort_values("HPI", ascending=False)
    .head(10)
    .copy()
)
decision_table["recommended_action"] = (
    decision_table.apply(recommend_action, axis=1)
)
decision_table = decision_table.reset_index().rename(
    columns={"index": "junction_name"}
)

print("=" * 60)
print("Decision Output Table (Top 10 by Hotspot Priority Index)")
print("=" * 60)
print(
    decision_table[
        [
            "junction_name",
            "violation_count",
            "composite_score",
            "HPI",
            "risk_label",
            "recommended_action"
        ]
    ]
)
print()

decision_table.to_csv(
    OUTPUT_DIR / "eda_decision_table.csv",
    index=False
)

# ==========================================================
# HOTSPOT MAP CSV (junction coordinates + risk)
# ==========================================================
if "latitude" in df.columns and "longitude" in df.columns:

    junction_coords = (
        df_junction.groupby("junction_name")[["latitude", "longitude"]]
        .median()  # median is more robust to GPS outliers than mean
    )

    hotspot_map = junction_analytics.join(junction_coords, how="left")

    n_missing_coords = hotspot_map["latitude"].isna().sum()
    if n_missing_coords > 0:
        print(f"Note: {n_missing_coords} junctions had no valid coordinates and will show blank lat/lon in the map CSV.")
        print()

    hotspot_map = hotspot_map.reset_index().rename(
        columns={"index": "junction_name"}
    )

    hotspot_map[
        ["junction_name", "latitude", "longitude", "composite_score", "risk_label"]
    ].to_csv(
        OUTPUT_DIR / "hotspot_map.csv",
        index=False
    )

else:
    print("Note: latitude/longitude columns not found, skipping hotspot_map.csv generation.")
    print()

# ==========================================================
# INTELLIGENCE LAYER: STRUCTURED INSIGHTS DICT
# ==========================================================
insights = {
    "peak_hour": int(hourly.idxmax()),
    "peak_day": day_counts.idxmax(),
    "top_10_junction_share_pct": round(top_10_share, 2),
    "top_5pct_junction_share_pct": round(top_5pct_share, 2),
    "top_10pct_junction_share_pct": round(top_10pct_share, 2),
    "critical_junctions": (
        junction_analytics[
            junction_analytics["risk_label"] == "CRITICAL HOTSPOT"
        ].index.tolist()
    ),
    "violations_per_vehicle": round(violation_per_vehicle, 2),
    "top_vehicle_type": top_vehicle_types.index[0],
    "top_violation_type": top_violations.index[0],
    "risk_level_overall": (
        "High concentration system"
        if top_10_share > 40
        else "Distributed violation system"
    )
}

print("=" * 60)
print("Structured Insights")
print("=" * 60)
for k, v in insights.items():
    print(f"{k}: {v}")
print()

pd.Series(insights).to_csv(
    OUTPUT_DIR / "eda_insights.csv"
)

# ==========================================================
# INSIGHT SUMMARY BLOCK
# ==========================================================
peak_hour = hourly.idxmax()
peak_day = day_counts.idxmax()
top_vehicle = top_vehicle_types.index[0]
top_violation = top_violations.index[0]
max_repeat = repeat_offenders.iloc[0]

print("\n================ INSIGHTS ================\n")

peak_hour_z = hourly_zscore.get(peak_hour, 0)
peak_hour_note = (
    "a statistically significant spike" if peak_hour_z > 1.5
    else "the highest point but not a statistical outlier"
)

print(f"1. Peak violation hour is {peak_hour}:00 (z-score {peak_hour_z:.2f}), which is {peak_hour_note}.")
print(f"2. {peak_day} sees the highest violation volume across the week.")
print(f"3. Top 10 junctions contribute {top_10_share:.1f}% of all mapped violations.")
print(f"4. '{top_vehicle}' is the most frequently violating vehicle type.")
print(f"5. '{top_violation}' is the most common violation type recorded.")
print(f"6. Top 1% of vehicles account for {top_1pct_share:.2f}% of all violations, indicating repeat offenders contribute disproportionately to citywide parking violations.")
print(f"7. On average, each vehicle accumulates {violation_per_vehicle:.2f} violations.")

print()

# ==========================================================
# CITY EXECUTIVE SUMMARY CSV
# ==========================================================
n_critical = len(insights["critical_junctions"])
n_high = len(
    junction_analytics[
        junction_analytics["risk_label"] == "HIGH HOTSPOT"
    ]
)
n_emerging = (
    int((trend_df["trend"] == "Emerging Hotspot").sum())
    if trend_df is not None else "N/A (insufficient months)"
)
highest_risk_station = station_risk_df.index[0]

city_summary = {
    "Total Violations": int(len(df)),
    "Unique Vehicles": int(df["vehicle_number"].nunique()),
    "Unique Junctions": int(df_junction["junction_name"].nunique()),
    "Peak Hour": int(hourly.idxmax()),
    "Peak Day": day_counts.idxmax(),
    "Critical Hotspots": n_critical,
    "High Hotspots": n_high,
    "Top 10 Junction Share (%)": round(top_10_share, 2),
    "Top 1% Vehicle Share (%)": round(top_1pct_share, 2),
    "Highest Risk Police Station": highest_risk_station,
    "Emerging Hotspots (MoM)": n_emerging,
    "Overall Risk Level": insights["risk_level_overall"]
}

city_summary_df = pd.DataFrame(
    list(city_summary.items()),
    columns=["Metric", "Value"]
)

print("=" * 60)
print("CITY EXECUTIVE SUMMARY")
print("=" * 60)
print(city_summary_df.to_string(index=False))
print()

city_summary_df.to_csv(
    OUTPUT_DIR / "city_risk_summary.csv",
    index=False
)


print("=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Charts saved to : {CHART_DIR}")

