import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sklearn.cluster import DBSCAN

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "outputs" / "df_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

HOTSPOT_FILE = OUTPUT_DIR / "hotspots.csv"
CLUSTER_FILE = OUTPUT_DIR / "clustered_violations.csv"

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
# REMOVE INVALID JUNCTIONS
# ==========================================================
df = df[
    df["junction_name"].notna()
]

df = df[
    df["junction_name"] != "No Junction"
].copy()

print(f"Dataset after removing 'No Junction': {df.shape}")
print()

# ==========================================================
# VIOLATION SEVERITY WEIGHTS
# (rarer/more dangerous violation types count for more than a
# simple row count would give them — feeds severity scoring
# downstream in risk_score.py via weighted_violation_score)
# ==========================================================
VIOLATION_WEIGHTS = {
    "Wrong Parking": 1,
    "No Parking": 2,
    "Parking On Footpath": 3,
    "Parking In A Main Road": 4,
    "Defective Number Plate": 2,
    "Double Parking": 3,
    "Against One Way/No Entry": 4,
    "Parking Near School/Hospital": 4,
    "Without Side Mirror": 1,

    # added from violation_type_unmatched.csv (real categories
    # confirmed present in the dataset, not guesses)
    "Refuse To Go For Hire": 2,                         # service-refusal, not a parking hazard but still an enforceable offence
    "Parking Near Road Crossing": 4,                    # visibility/pedestrian hazard, same tier as main-road parking
    "Parking Near Bustop/School/Hospital Etc": 4,        # same tier as "Parking Near School/Hospital" above
    "Demanding Excess Fare": 1,                          # fare violation, not a parking/road-safety issue
    "Parking Near Traffic Light Or Zebra Cross": 4,      # high pedestrian-safety risk
    "Parking Opposite To Another Parked Vehicle": 3,     # road-narrowing/congestion risk, same tier as double parking
    "Using Black Film/Other Materials": 2,               # vehicle-compliance issue, moderate
    "Parking Other Than Bus Stop": 2,
    "Obstructing Driver": 3,
    "H T V Prohibited": 4,                               # heavy-vehicle restriction violation, high severity

    "Rider Not Wearing Helmet": 3,                       # road-safety violation, moderate-high
    "2W/3W - Using Mobile Phone": 4,                     # active distraction risk, high severity
    "Violating Lane Disipline": 2                        # traffic-flow violation, moderate
    # NOTE: with this run's unmatched list now covered, recheck
    # violation_type_unmatched.csv after each new data refresh —
    # new categories can appear in future exports.
}

def parse_violation(x):
    """
    violation_type is stored as a string-encoded list, e.g.
    '["No Parking","Defective Number Plate"]', not a single value.
    ast.literal_eval converts that string back into a real list.
    Falls back to a single-item list if parsing fails (e.g. value
    is already a plain string, or malformed).
    """
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (ValueError, SyntaxError, TypeError):
        return [x]

# IMPORTANT: this explode happens on a SEPARATE copy of df, used only
# for weight calculation. Exploding the main df in place would also
# multiply total_violations, peak_hour_ratio, and the DBSCAN cluster
# output, since those all currently assume one row per violation id.
# We don't want a violation with 3 listed types to get counted 3x
# everywhere else in the pipeline — only in the weighted severity sum.
df_weights = df[["junction_name", "violation_type"]].copy()

df_weights["violation_type_parsed"] = (
    df_weights["violation_type"].apply(parse_violation)
)

df_weights = df_weights.explode("violation_type_parsed")

df_weights["violation_type_parsed"] = (
    df_weights["violation_type_parsed"]
    .astype(str)
    .str.strip()
)

df_weights["violation_weight"] = (
    df_weights["violation_type_parsed"].map(VIOLATION_WEIGHTS).fillna(1)
)

# sanity check: warn if exploded violation values still don't match
# the weight map (likely missing categories or a string-formatting
# mismatch), since a silent fillna(1) would quietly flatten the
# whole weighting for everything not covered
unmatched_mask = (
    ~df_weights["violation_type_parsed"].isin(VIOLATION_WEIGHTS.keys())
)
unmatched_types = (
    set(df_weights.loc[unmatched_mask, "violation_type_parsed"].dropna().unique())
)

if unmatched_types:
    # row counts per unmatched value, sorted so the most frequent
    # gaps in coverage are at the top — those are the ones most
    # worth adding to VIOLATION_WEIGHTS first
    unmatched_counts = (
        df_weights.loc[unmatched_mask, "violation_type_parsed"]
        .value_counts()
        .reset_index()
    )
    unmatched_counts.columns = ["violation_type", "row_count"]

    unmatched_counts.to_csv(
        OUTPUT_DIR / "violation_type_unmatched.csv",
        index=False
    )

    print(
        f"Warning: {len(unmatched_types)} distinct violation_type value(s) "
        f"({unmatched_mask.sum()} rows) did not match VIOLATION_WEIGHTS and "
        f"defaulted to weight 1."
    )
    print(f"Full list with row counts saved to: violation_type_unmatched.csv")
    print(f"Top 10 most frequent unmatched values:")
    print(unmatched_counts.head(10).to_string(index=False))
    print("Check exact spelling/casing against VIOLATION_WEIGHTS keys, and add genuinely new categories.")
    print()


# ==========================================================
# HOTSPOT AGGREGATION
# ==========================================================
print("Creating junction-level hotspot statistics...")

hotspots = (
    df.groupby("junction_name")
    .agg(
        total_violations=("id", "count"),
        unique_vehicles=("vehicle_number", "nunique"),
        peak_violations=("is_peak_hour", "sum"),
        avg_latitude=("latitude", "mean"),
        avg_longitude=("longitude", "mean")
    )
    .reset_index()
)

# ==========================================================
# WEIGHTED VIOLATION SCORE
# (severity-weighted volume — feeds severity_raw in risk_score.py
# instead of the unweighted total_violations fallback)
# ==========================================================
weighted_violations = (
    df_weights.groupby("junction_name")["violation_weight"]
    .sum()
    .reset_index(name="weighted_violation_score")
)

hotspots = hotspots.merge(
    weighted_violations,
    on="junction_name",
    how="left"
)

# ==========================================================
# PEAK HOUR RATIO
# ==========================================================
hotspots["peak_hour_ratio"] = np.where(
    hotspots["total_violations"] > 0,
    hotspots["peak_violations"] / hotspots["total_violations"],
    0
)

# ==========================================================
# REPEAT OFFENDER RATE
# ==========================================================
repeat_rate = (
    df.groupby("junction_name")["vehicle_number"]
    .apply(
        lambda x:
        x.fillna("UNKNOWN")
         .duplicated()
         .sum() / len(x)
    )
    .reset_index(name="repeat_rate")
)

hotspots = hotspots.merge(
    repeat_rate,
    on="junction_name",
    how="left"
)

# ==========================================================
# VIOLATION DIVERSITY
# ==========================================================
violation_diversity = (
    df.groupby("junction_name")["violation_type"]
    .nunique()
    .reset_index(name="unique_violation_types")
)

hotspots = hotspots.merge(
    violation_diversity,
    on="junction_name",
    how="left"
)

# ==========================================================
# VEHICLE DIVERSITY
# ==========================================================
vehicle_diversity = (
    df.groupby("junction_name")["vehicle_type"]
    .nunique()
    .reset_index(name="unique_vehicle_types")
)

hotspots = hotspots.merge(
    vehicle_diversity,
    on="junction_name",
    how="left"
)

# ==========================================================
# VIOLATION DENSITY (useful for IPII later)
# ==========================================================
hotspots["violations_per_vehicle"] = (
    hotspots["total_violations"]
    / hotspots["unique_vehicles"]
)

print("Hotspot aggregation completed.")
print()

# ==========================================================
# DBSCAN SPATIAL CLUSTERING
# ==========================================================
print("Running DBSCAN clustering...")

coords = (
    df[
        ["latitude", "longitude"]
    ]
    .dropna()
    .values
)

coords_radians = np.radians(coords)

kms_per_radian = 6371.0088

epsilon = 0.15 / kms_per_radian
# 150 metres

db = DBSCAN(
    eps=epsilon,
    min_samples=50,
    metric="haversine",
    algorithm="ball_tree"
)

cluster_labels = db.fit_predict(
    coords_radians
)

df["cluster"] = cluster_labels

print(
    f"Clusters Found : "
    f"{len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}"
)

print(
    f"Noise Points : "
    f"{(cluster_labels == -1).sum()}"
)

print()

# ==========================================================
# CLUSTER SUMMARY
# ==========================================================
cluster_summary = (
    df[df["cluster"] != -1]
    .groupby("cluster")
    .agg(
        total_violations=("id", "count"),
        unique_junctions=("junction_name", "nunique"),
        unique_vehicles=("vehicle_number", "nunique"),
        avg_latitude=("latitude", "mean"),
        avg_longitude=("longitude", "mean")
    )
    .reset_index()
)

cluster_summary = cluster_summary.sort_values(
    "total_violations",
    ascending=False
)

# ==========================================================
# TOP HOTSPOTS
# ==========================================================
print("=" * 60)
print("Top 10 Junction Hotspots")
print("=" * 60)

print(
    hotspots
    .sort_values(
        "total_violations",
        ascending=False
    )
    .head(10)
)

print()

print("=" * 60)
print("Top Spatial Clusters")
print("=" * 60)

print(
    cluster_summary
    .head(10)
)

print()

# ==========================================================
# SAVE FILES
# ==========================================================
hotspots.to_csv(
    HOTSPOT_FILE,
    index=False
)

hotspots.sort_values(
    "total_violations",
    ascending=False
).to_csv(
    OUTPUT_DIR / "top_hotspots.csv",
    index=False
)

df.to_csv(
    CLUSTER_FILE,
    index=False
)

cluster_summary.to_csv(
    OUTPUT_DIR / "cluster_summary.csv",
    index=False
)

top_clusters = cluster_summary.head(10)

top_clusters.to_csv(
    OUTPUT_DIR / "top_clusters.csv",
    index=False
)

# ==========================================================
# DONE
# ==========================================================
print("=" * 60)
print("HOTSPOT DETECTION COMPLETED")
print("=" * 60)
print(f"Saved : {HOTSPOT_FILE}")
print(f"Saved : top_hotspots.csv")
print(f"Saved : {CLUSTER_FILE}")
print(f"Saved : cluster_summary.csv")
print(f"Saved : top_clusters.csv")