import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "outputs" / "df_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

EMERGING_FILE = OUTPUT_DIR / "emerging_hotspots.csv"
TOP_EMERGING_FILE = OUTPUT_DIR / "top_emerging_hotspots.csv"

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
# DATE CONVERSION
# ==========================================================

df["created_datetime"] = pd.to_datetime(
    df["created_datetime"],
    errors="coerce"
)

df = df.dropna(
    subset=["created_datetime"]
)

# ==========================================================
# REMOVE INVALID JUNCTIONS
# ==========================================================

df = df[
    df["junction_name"].notna()
]

df = df[
    df["junction_name"] != "No Junction"
].copy()

print(
    f"Dataset after cleaning : {df.shape}"
)
print()

# ==========================================================
# MONTH FEATURE
# ==========================================================

df["year_month"] = (
    df["created_datetime"]
    .dt.to_period("M")
)

n_months = df["year_month"].nunique()

print(f"Months of data available : {n_months}")
print()

# ==========================================================
# MONTHLY VIOLATION COUNTS
# ==========================================================

monthly = (
    df.groupby(
        ["junction_name", "year_month"]
    )
    .size()
    .reset_index(name="violations")
)

# ==========================================================
# NORMALIZATION HELPER
# ==========================================================

def normalize(series):
    """
    Scales a series to the 0-1 range.
    If all values are identical, returns 0 for every row
    instead of dividing by zero.
    """
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - min_val) / (max_val - min_val)


# ==========================================================
# CHECK FOR PARTIAL LATEST MONTH
# ==========================================================
# With only a handful of months of data, the most recent
# month may not be fully collected yet (e.g. data export
# happened mid-month). We verify this by comparing the date
# range actually covered in the latest month against a full
# calendar month, rather than assuming it's partial.

latest_month_global = df["year_month"].max()

latest_month_dates = df.loc[
    df["year_month"] == latest_month_global,
    "created_datetime"
]

days_with_data = latest_month_dates.dt.day.nunique()
days_in_month = latest_month_global.days_in_month

is_partial_month = days_with_data < (days_in_month * 0.9)

print(
    f"Latest month ({latest_month_global}) has data for "
    f"{days_with_data} of {days_in_month} days."
)

if is_partial_month:
    print(
        "Latest month appears PARTIAL - excluding it "
        "from trend analysis."
    )
    monthly_for_trend = monthly[
        monthly["year_month"] != latest_month_global
    ].copy()
else:
    print(
        "Latest month appears complete - including it "
        "in trend analysis."
    )
    monthly_for_trend = monthly.copy()

print()

# ==========================================================
# EMERGING SCORE
# ==========================================================
# Growth is measured as the recent 3-month average violation
# count vs. the earlier 3-month average, rather than a raw
# first-month-vs-last-month comparison. This is far less
# sensitive to noise from a single unusually quiet or busy
# month at either end of the timeline.

WINDOW = 3

results = []

print("Calculating growth trends...")
print()

for junction, group in monthly_for_trend.groupby(
    "junction_name"
):

    group = group.sort_values(
        "year_month"
    ).reset_index(drop=True)

    if len(group) < 4:
        continue

    # Use up to WINDOW months on each side, but adapt
    # gracefully if less history is available.
    half = max(
        1,
        min(WINDOW, len(group) // 2)
    )

    earlier_avg = (
        group["violations"]
        .head(half)
        .mean()
    )

    recent_avg = (
        group["violations"]
        .tail(half)
        .mean()
    )

    avg_monthly = (
        group["violations"]
        .mean()
    )

    # Skip junctions with very low average activity.
    # A junction going from 1 to 5 violations technically
    # shows 400% growth, but it isn't a meaningful emerging
    # hotspot at that volume.
    if avg_monthly < 5:
        continue

    growth_rate_uncapped = (
        (recent_avg - earlier_avg)
        / max(earlier_avg, 1)
    ) * 100

    # Cap extreme growth rates so a junction jumping from
    # 1 to 100 violations (9900% growth) doesn't dominate
    # the score purely due to a tiny starting base.
    growth_rate = np.clip(
        growth_rate_uncapped,
        -100,
        500
    )

    latest_month_count = (
        group["violations"].iloc[-1]
    )

    # How the most recent single month compares to the
    # recent-period average. >1 means the latest month is
    # unusually high even relative to recent history.
    latest_share = (
        latest_month_count
        / max(recent_avg, 1)
    )

    results.append([
        junction,
        len(group),
        round(earlier_avg, 2),
        round(recent_avg, 2),
        round(growth_rate, 2),
        round(growth_rate_uncapped, 2),
        round(avg_monthly, 2),
        latest_month_count,
        round(latest_share, 2)
    ])

# ==========================================================
# SAFETY CHECK
# ==========================================================

if len(results) == 0:
    print("No emerging hotspots found.")
    exit()

# ==========================================================
# CREATE RESULTS
# ==========================================================

emerging = pd.DataFrame(
    results,
    columns=[
        "junction_name",
        "months_active",
        "earlier_period_avg",
        "recent_period_avg",
        "growth_rate_percent",
        "uncapped_growth_rate_percent",
        "avg_monthly_violations",
        "latest_month_violations",
        "latest_share"
    ]
)

# ==========================================================
# NORMALIZED EMERGING SCORE
# ==========================================================
# Both components are scaled to 0-1 before blending so that
# growth_rate (which can range from -100% to several thousand
# percent) doesn't automatically dominate over recent_period_avg
# (a small raw count). This keeps the score meaningful and
# defensible rather than one signal silently overpowering
# the other.

growth_norm = normalize(emerging["growth_rate_percent"])
recent_norm = normalize(emerging["recent_period_avg"])
volume_norm = normalize(
    np.log1p(emerging["avg_monthly_violations"])
)
momentum_norm = normalize(emerging["latest_share"])

# Confidence factor: discounts the score for junctions with
# very low absolute volume, even if their percentage growth
# is high. A jump from 6 to 75 violations (500% growth) is
# statistically much less reliable, and less impactful in
# absolute terms, than a jump from 1574 to 2521 (60% growth).
# Junctions with 100+ avg monthly violations get full
# confidence (1.0); below that, confidence scales down
# linearly toward 0.
confidence = np.minimum(
    emerging["avg_monthly_violations"] / 100,
    1
)

WEIGHT_GROWTH = 0.4
WEIGHT_RECENT_VOLUME = 0.3
WEIGHT_VOLUME = 0.2
WEIGHT_MOMENTUM = 0.1

emerging["emerging_score"] = (
    (
        WEIGHT_GROWTH * growth_norm
        + WEIGHT_RECENT_VOLUME * recent_norm
        + WEIGHT_VOLUME * volume_norm
        + WEIGHT_MOMENTUM * momentum_norm
    )
    * confidence
    * 100
).round(2)

# ==========================================================
# RANKING
# ==========================================================

emerging = emerging.sort_values(
    "emerging_score",
    ascending=False
).reset_index(drop=True)

emerging["emerging_rank"] = emerging.index + 1

# ==========================================================
# CATEGORY
# ==========================================================
# Fixed thresholds rather than quantiles, so the category
# boundaries don't shift every time the script is rerun on
# a slightly different data slice.

def classify(score):
    if score >= 70:
        return "Strong Emerging"
    elif score >= 50:
        return "Emerging"
    elif score >= 30:
        return "Stable"
    return "Low Growth"

emerging["trend_category"] = (
    emerging["emerging_score"].apply(classify)
)

emerging["trend_direction"] = np.where(
    emerging["growth_rate_percent"] > 0,
    "Increasing",
    "Decreasing"
)

def trend_strength(growth):
    if growth > 100:
        return "Rapid Growth"
    elif growth > 20:
        return "Growing"
    elif growth > -20:
        return "Stable"
    return "Declining"

emerging["trend_strength"] = (
    emerging["growth_rate_percent"].apply(trend_strength)
)

def severity(avg):
    if avg > 1000:
        return "Very High Volume"
    elif avg > 500:
        return "High Volume"
    elif avg > 100:
        return "Medium Volume"
    return "Low Volume"

emerging["volume_severity"] = (
    emerging["avg_monthly_violations"].apply(severity)
)

# ==========================================================
# FINAL PRIORITY (combine with IPII risk score)
# ==========================================================
# Merges the emerging trend score with the IPII risk score
# from risk_score.py, so junctions that are BOTH high risk
# AND fast growing rank highest overall. This is typically
# what traffic authorities care about most: not just where
# violations are worst today, but where the problem is
# actively getting worse.

RISK_FILE = OUTPUT_DIR / "risk_scores.csv"

if RISK_FILE.exists():
    risk_df = pd.read_csv(RISK_FILE)

    risk_cols = risk_df[["junction_name", "IPII"]]

    emerging = emerging.merge(
        risk_cols,
        on="junction_name",
        how="left"
    )

    # Junctions with no IPII match (e.g. filtered out
    # earlier in hotspot.py) get a neutral score of 0
    # rather than being dropped.
    emerging["IPII"] = emerging["IPII"].fillna(0)

    WEIGHT_IPII = 0.6
    WEIGHT_EMERGING = 0.4

    emerging["final_priority"] = (
        WEIGHT_IPII * emerging["IPII"]
        + WEIGHT_EMERGING * emerging["emerging_score"]
    ).round(2)

    emerging = emerging.sort_values(
        "final_priority",
        ascending=False
    ).reset_index(drop=True)

    emerging["emerging_rank"] = emerging.index + 1

    print(
        "Merged with risk_scores.csv - "
        "final_priority column added."
    )
else:
    print(
        "risk_scores.csv not found - skipping "
        "final_priority merge. Run risk_score.py first."
    )

print()

# ==========================================================
# TOP RESULTS
# ==========================================================

print("=" * 60)
print("Top 20 Emerging Hotspots")
print("=" * 60)

display_cols = [
    "emerging_rank",
    "junction_name",
    "earlier_period_avg",
    "recent_period_avg",
    "latest_month_violations",
    "growth_rate_percent",
    "emerging_score",
    "trend_category",
    "trend_strength",
    "volume_severity"
]

if "final_priority" in emerging.columns:
    display_cols.append("final_priority")

print(
    emerging[display_cols]
    .head(20)
    .to_string(index=False)
)

print()

print("=" * 60)
print("Trend Category Distribution")
print("=" * 60)

print(
    emerging["trend_category"]
    .value_counts()
)

print()

# ==========================================================
# SAVE
# ==========================================================

emerging.to_csv(
    EMERGING_FILE,
    index=False
)

emerging.head(20).to_csv(
    TOP_EMERGING_FILE,
    index=False
)

monthly.to_csv(
    OUTPUT_DIR / "monthly_trends.csv",
    index=False
)

# ==========================================================
# LATEST MONTH HOTSPOTS
# ==========================================================
# A separate, simple snapshot: which junctions had the most
# violations in the single most recent month. Useful for a
# dashboard "right now" view alongside the longer-term trend.

latest_month = monthly["year_month"].max()

latest_hotspots = (
    monthly[
        monthly["year_month"] == latest_month
    ]
    .sort_values(
        "violations",
        ascending=False
    )
)

latest_hotspots.to_csv(
    OUTPUT_DIR / "latest_month_hotspots.csv",
    index=False
)

print("=" * 60)
print("EMERGING HOTSPOT ANALYSIS COMPLETED")
print("=" * 60)

print(f"Saved : {EMERGING_FILE}")
print(f"Saved : {TOP_EMERGING_FILE}")
print(f"Saved : monthly_trends.csv")
print(f"Saved : latest_month_hotspots.csv")
print()