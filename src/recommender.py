import pandas as pd
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RISK_FILE = BASE_DIR / "outputs" / "risk_scores.csv"
EMERGING_FILE = BASE_DIR / "outputs" / "emerging_hotspots.csv"

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "enforcement_recommendations.csv"
)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("Loading risk and hotspot data...")
print("=" * 60)

risk_df = pd.read_csv(RISK_FILE)
emerging_df = pd.read_csv(EMERGING_FILE)

print(f"Risk records     : {len(risk_df)}")
print(f"Emerging records : {len(emerging_df)}")
print()

# ==========================================================
# MERGE DATA
# ==========================================================

df = risk_df.merge(
    emerging_df[
        [
            "junction_name",
            "trend_category",
            "trend_strength",
            "growth_rate_percent",
            "emerging_score"
        ]
    ],
    on="junction_name",
    how="left"
)

# ==========================================================
# FILL MISSING VALUES
# ==========================================================

df["trend_category"] = df["trend_category"].fillna("Stable")
df["trend_strength"] = df["trend_strength"].fillna("Stable")
df["growth_rate_percent"] = df["growth_rate_percent"].fillna(0)
df["emerging_score"] = df["emerging_score"].fillna(0)

# ==========================================================
# RECOMMENDATION ENGINE
# ==========================================================

def recommend_action(row):

    risk = str(row["risk_category"])
    trend = str(row["trend_strength"])

    growth = row["growth_rate_percent"]
    ipii = row["IPII"]
    emerging = row["emerging_score"]

    # ======================================================
    # CRITICAL RISK
    # ======================================================

    if risk == "Critical":

        if ipii >= 80:

            return pd.Series([
                "Immediate Action",
                "Deploy enforcement unit + ANPR cameras + daily monitoring"
            ])

        return pd.Series([
            "Immediate Action",
            "Deploy enforcement team and increase patrol frequency"
        ])

    # ======================================================
    # HIGH RISK
    # ======================================================

    elif risk == "High":

        if trend == "Rapid Growth":

            return pd.Series([
                "Emerging High Risk",
                "Increase patrols and deploy preventive enforcement"
            ])

        elif growth > 25:

            return pd.Series([
                "Preventive Action",
                "Target repeat offenders and increase inspections"
            ])

        return pd.Series([
            "Regular Enforcement",
            "Regular enforcement patrols"
        ])

    # ======================================================
    # MODERATE RISK
    # ======================================================

    elif risk == "Moderate":

        if emerging >= 50:

            return pd.Series([
                "Emerging Hotspot",
                "Monitor weekly and deploy preventive enforcement"
            ])

        elif growth > 20:

            return pd.Series([
                "Preventive Action",
                "Increase inspections and awareness campaigns"
            ])

        return pd.Series([
            "Routine Monitoring",
            "Periodic monitoring"
        ])

    # ======================================================
    # LOW RISK
    # ======================================================

    if trend == "Rapid Growth":

        return pd.Series([
            "Emerging Hotspot",
            "Monitor for future escalation"
        ])

    return pd.Series([
        "Routine Monitoring",
        "Low priority routine observation"
    ])

# ==========================================================
# APPLY RECOMMENDATIONS
# ==========================================================

print("Generating recommendations...")

df[
    [
        "action_category",
        "recommended_action"
    ]
] = df.apply(
    recommend_action,
    axis=1
)

# ==========================================================
# ENFORCEMENT PRIORITY SCORE (multi-factor)
# ==========================================================

df["enforcement_priority"] = (
    0.8 * df["IPII"]
    + 0.15 * df["emerging_score"]
    + 0.05 * df["growth_rate_percent"].clip(lower=0)
)

risk_bonus = {
    "Critical": 20,
    "High": 10,
    "Moderate": 5,
    "Low": 0
}

df["enforcement_priority"] = (
    df["enforcement_priority"]
    + df["risk_category"].map(risk_bonus)
)

df["enforcement_priority"] = (
    df["enforcement_priority"]
    .round(2)
)

df = df.sort_values(
    "enforcement_priority",
    ascending=False
).reset_index(drop=True)

df["priority_rank"] = df.index + 1

# ==========================================================
# PRIORITY LEVELS
# ==========================================================

def priority_level(rank):

    if rank <= 10:
        return "P1 - Immediate"

    elif rank <= 30:
        return "P2 - High"

    elif rank <= 80:
        return "P3 - Medium"

    return "P4 - Low"

df["priority_level"] = (
    df["priority_rank"]
    .apply(priority_level)
)

# ==========================================================
# DEPLOYMENT RECOMMENDATION
# ==========================================================

def deployment_plan(row):

    risk = row["risk_category"]
    trend = row["trend_strength"]

    if risk == "Critical":
        return "Dedicated enforcement unit"

    elif risk == "High":
        return "Enhanced patrol coverage"

    elif trend == "Rapid Growth":
        return "Preventive hotspot monitoring"

    elif trend == "Growing":
        return "Weekly monitoring"

    return "Routine monitoring"

df["deployment_plan"] = (
    df.apply(
        deployment_plan,
        axis=1
    )
)

# ==========================================================
# RESOURCE ALLOCATION
# ==========================================================

def resource_level(row):

    score = row["enforcement_priority"]

    if score >= 60:
        return "2 Patrol Teams + ANPR Camera"

    elif score >= 45:
        return "1 Patrol Team + Mobile Camera"

    elif score >= 25:
        return "Periodic Patrol"

    return "Routine Monitoring"

df["resource_allocation"] = (
    df.apply(resource_level, axis=1)
)

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("=" * 60)
print("TOP ENFORCEMENT PRIORITIES")
print("=" * 60)

display_cols = [
    "priority_rank",
    "priority_level",
    "junction_name",
    "risk_category",
    "IPII",
    "enforcement_priority",
    "resource_allocation",
    "action_category",
    "recommended_action"
]

print(
    df[display_cols]
    .head(20)
)

print()

print("=" * 60)
print("ACTION CATEGORY DISTRIBUTION")
print("=" * 60)

print(
    df["action_category"]
    .value_counts()
)

print()

print("=" * 60)
print("RESOURCE ALLOCATION DISTRIBUTION")
print("=" * 60)

print(
    df["resource_allocation"]
    .value_counts()
)

print()

# ==========================================================
# SAVE FILES
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

df.head(20).to_csv(
    BASE_DIR
    / "outputs"
    / "top_enforcement_priorities.csv",
    index=False
)

print("=" * 60)
print("RECOMMENDER COMPLETED")
print("=" * 60)

print(f"Saved : {OUTPUT_FILE}")
print(
    f"Saved : "
    f"{BASE_DIR / 'outputs' / 'top_enforcement_priorities.csv'}"
)

print()