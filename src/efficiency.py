import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RISK_FILE = BASE_DIR / "outputs" / "risk_scores.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

EFFICIENCY_FILE = OUTPUT_DIR / "efficiency.csv"
TOP_IMPACT_FILE = OUTPUT_DIR / "top_impact_hotspots.csv"

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("Loading risk scores...")
print("=" * 60)

df = pd.read_csv(RISK_FILE)

print(f"Records : {len(df)}")
print()

# ==========================================================
# IMPACT ASSUMPTIONS
# ==========================================================

# Simulation-based estimates
# Can later be calibrated using real enforcement outcomes.

REDUCTION_RATE = {
    "Critical": 0.35,
    "High": 0.25,
    "Moderate": 0.15,
    "Low": 0.05
}

EFFORT_SAVING = {
    "Critical": 12,
    "High": 8,
    "Moderate": 4,
    "Low": 1
}

# ==========================================================
# PROJECTED VIOLATION REDUCTION
# ==========================================================

print("Calculating projected impact...")
print()

df["estimated_violations_prevented"] = (
    df["total_violations"]
    * df["risk_category"].map(REDUCTION_RATE)
).round().astype(int)

# ==========================================================
# ENFORCEMENT EFFORT SAVING
# ==========================================================

df["estimated_enforcement_effort_saved"] = (
    df["risk_category"].map(EFFORT_SAVING)
)

# ==========================================================
# EFFICIENCY SCORE
# ==========================================================

max_prevented = df["estimated_violations_prevented"].max()

df["efficiency_score"] = (
    df["estimated_violations_prevented"]
    / max_prevented
) * 100

# ==========================================================
# IMPACT SCORE
# ==========================================================

# Risk should dominate.
# Efficiency is a supporting signal.

df["impact_score"] = (
    (
        df["IPII"] * 0.8
        +
        df["efficiency_score"] * 0.2
    )
).round(2)

# ==========================================================
# IMPACT RANKING
# ==========================================================

df = (
    df.sort_values(
        "impact_score",
        ascending=False
    )
    .reset_index(drop=True)
)

df["impact_rank"] = df.index + 1

# ==========================================================
# SUMMARY METRICS
# ==========================================================

total_violations = int(
    df["total_violations"].sum()
)

total_prevented = int(
    df["estimated_violations_prevented"].sum()
)

total_effort_saved = int(
    df["estimated_enforcement_effort_saved"].sum()
)

reduction_percent = round(
    (total_prevented / total_violations) * 100,
    2
)

# ==========================================================
# ADD DISCLAIMER
# ==========================================================

df["projection_type"] = (
    "Simulation-Based Estimate"
)

# ==========================================================
# OUTPUT
# ==========================================================

print("# PROJECTED SYSTEM IMPACT")
print()

print(
    f"Total Violations Analysed : "
    f"{total_violations:,}"
)

print(
    f"Estimated Violations Prevented : "
    f"{total_prevented:,}"
)

print(
    f"Estimated Enforcement Effort Saved : "
    f"{total_effort_saved:,}"
)

print(
    f"Projected Reduction : "
    f"{reduction_percent}%"
)

print()

display_cols = [
    "junction_name",
    "risk_category",
    "IPII",
    "estimated_violations_prevented",
    "estimated_enforcement_effort_saved",
    "impact_score"
]

print(
    df[display_cols]
    .head(20)
)

print()

# ==========================================================
# SAVE FILES
# ==========================================================

df.to_csv(
    EFFICIENCY_FILE,
    index=False
)

df.head(20).to_csv(
    TOP_IMPACT_FILE,
    index=False
)

print()

print(
    f"Saved : {EFFICIENCY_FILE}"
)

print(
    f"Saved : {TOP_IMPACT_FILE}"
)