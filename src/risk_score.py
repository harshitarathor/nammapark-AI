import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "outputs" / "hotspots.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

RISK_FILE = OUTPUT_DIR / "risk_scores.csv"
TOP_RISK_FILE = OUTPUT_DIR / "top_risk_junctions.csv"

EXPLAIN_DIR = OUTPUT_DIR / "final_report"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

EXPLAIN_FILE = EXPLAIN_DIR / "risk_explained.csv"

# ==========================================================
# LOAD DATA
# ==========================================================
print("=" * 60)
print("Loading hotspot statistics...")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset Shape : {df.shape}")
print()

# ==========================================================
# MIN-MAX SCALING HELPER
# ==========================================================
def min_max_scale(series):
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
# COMPONENT 1 : SEVERITY (VIOLATION-WEIGHTED VOLUME)
# ==========================================================
print("Calculating severity component...")

# Prefer a violation-severity-weighted volume if the upstream
# hotspot aggregation provides it (weighted_violation_score =
# sum of per-violation-type weights, e.g. "Parking In A Main Road"
# counted as more severe than "Wrong Parking"). This needs to be
# computed upstream in hotspot.py from df_clean.py's per-row
# violation_type, since hotspots.csv here is already aggregated
# per junction and has no row-level violation_type to weight.
#
# Falls back to raw total_violations if that column isn't present
# yet, so this script keeps working before the upstream change is
# made — but the severity signal will be unweighted until it is.
if "weighted_violation_score" in df.columns:
    severity_source_col = "weighted_violation_score"
    print("Using weighted_violation_score for severity (violation-type weighting applied).")
else:
    severity_source_col = "total_violations"
    print("Note: weighted_violation_score not found in hotspots.csv — "
          "falling back to unweighted total_violations. To enable "
          "violation-severity weighting, compute weighted_violation_score "
          "upstream in hotspot.py and merge it into hotspots.csv.")

# Log-scaled so a handful of mega-junctions don't
# completely overwhelm every other junction's score.
df["severity_raw"] = np.log1p(df[severity_source_col])
df["severity_score"] = min_max_scale(df["severity_raw"])

print("Severity component completed.")
print()

# ==========================================================
# COMPONENT 2 : BEHAVIORAL RISK
# (repeat offenders + peak-hour concentration)
# ==========================================================
print("Calculating behavioral risk component...")

repeat_norm = min_max_scale(df["repeat_rate"])
peak_norm = min_max_scale(df["peak_hour_ratio"])

# Equal-weighted blend of the two behavioral signals.
df["behavioral_score"] = (
    (repeat_norm + peak_norm) / 2
)

print("Behavioral risk component completed.")
print()

# ==========================================================
# COMPONENT 3 : DENSITY
# (violations concentrated on fewer vehicles = higher risk)
# ==========================================================
print("Calculating density component...")

df["density_score"] = min_max_scale(
    df["violations_per_vehicle"]
)

print("Density component completed.")
print()

# ==========================================================
# FINAL IPII SCORE
# ==========================================================
print("Calculating final IPII score...")

# Weights are explicit and defensible:
#   - Severity matters most (50%): scale of the problem,
#     weighted by violation-type danger where available.
#   - Behavioral risk (30%): how risky the pattern is, not
#     just how big.
#   - Density (20%): concentration signal, supportive but
#     not as critical as the other two.
WEIGHT_SEVERITY = 0.5
WEIGHT_BEHAVIORAL = 0.3
WEIGHT_DENSITY = 0.2

df["IPII"] = (
    WEIGHT_SEVERITY * df["severity_score"]
    + WEIGHT_BEHAVIORAL * df["behavioral_score"]
    + WEIGHT_DENSITY * df["density_score"]
) * 100  # scaled to 0-100 for easier interpretation

df["IPII"] = df["IPII"].round(2)

print("IPII score completed.")
print()

# ==========================================================
# RISK CATEGORY (for dashboard color-coding)
# ==========================================================
def categorize_risk(score):
    if score >= 70:
        return "Critical"
    elif score >= 50:
        return "High"
    elif score >= 30:
        return "Moderate"
    else:
        return "Low"

df["risk_category"] = df["IPII"].apply(categorize_risk)

# ==========================================================
# RANKING
# ==========================================================
df = df.sort_values("IPII", ascending=False).reset_index(drop=True)
df["risk_rank"] = df.index + 1

# ==========================================================
# EXPLAINABILITY OUTPUT
# (top contributing factor per junction, not just a static label)
# ==========================================================
print("Generating explainability output...")

def explain_risk(row):

    contributions = {
        "Severity (violation volume/weight)": WEIGHT_SEVERITY * row["severity_score"],
        "Behavioral risk (repeat offenders + peak-hour concentration)": WEIGHT_BEHAVIORAL * row["behavioral_score"],
        "Density (violations per vehicle)": WEIGHT_DENSITY * row["density_score"]
    }

    top_factor = max(contributions, key=contributions.get)
    top_share_pct = (
        contributions[top_factor] / sum(contributions.values()) * 100
        if sum(contributions.values()) > 0 else 0
    )

    return pd.Series([
        top_factor,
        round(top_share_pct, 1)
    ])

df[["top_contributing_factor", "top_factor_share_pct"]] = (
    df.apply(explain_risk, axis=1)
)

df["risk_explanation"] = (
    "IPII = " + WEIGHT_SEVERITY.__str__() + "*severity + "
    + WEIGHT_BEHAVIORAL.__str__() + "*behavioral + "
    + WEIGHT_DENSITY.__str__() + "*density. "
    + "Primary driver for this junction: " + df["top_contributing_factor"]
    + f" ({df['top_factor_share_pct'].astype(str)}% of weighted score)."
)

print("Explainability output completed.")
print()

# ==========================================================
# SUMMARY
# ==========================================================
print("=" * 60)
print("Top 10 Highest Risk Junctions")
print("=" * 60)

display_cols = [
    "risk_rank",
    "junction_name",
    "total_violations",
    "peak_hour_ratio",
    "repeat_rate",
    "violations_per_vehicle",
    "IPII",
    "risk_category",
    "top_contributing_factor"
]

print(df[display_cols].head(10))
print()

print("=" * 60)
print("Risk Category Distribution")
print("=" * 60)

print(df["risk_category"].value_counts())
print()

# ==========================================================
# SAVE FILES
# ==========================================================
df.to_csv(RISK_FILE, index=False)

df.head(10).to_csv(TOP_RISK_FILE, index=False)

explain_cols = [
    "risk_rank",
    "junction_name",
    "IPII",
    "risk_category",
    "top_contributing_factor",
    "top_factor_share_pct",
    "risk_explanation"
]

df[explain_cols].to_csv(EXPLAIN_FILE, index=False)

print("=" * 60)
print("RISK SCORING COMPLETED")
print("=" * 60)
print(f"Saved : {RISK_FILE}")
print(f"Saved : {TOP_RISK_FILE}")
print(f"Saved : {EXPLAIN_FILE}")
print()