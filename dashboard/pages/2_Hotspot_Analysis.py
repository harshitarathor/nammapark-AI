import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Hotspot Analysis",
    page_icon="🔥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "outputs"

@st.cache_data
def load_data():
    hotspots = pd.read_csv(DATA_DIR / "hotspots.csv")
    clusters = pd.read_csv(DATA_DIR / "cluster_summary.csv")
    return hotspots, clusters

hotspots, clusters = load_data()

st.title("🔥 Hotspot Analysis")

st.markdown("""
### 🔥 Hotspot Intelligence

This module identifies junctions where illegal parking
creates the highest congestion pressure.

The AI analyzes:

- Violation density
- Repeat offender behavior
- Peak-hour concentration
- Spatial hotspot clustering

to prioritize enforcement deployment.
""")

st.markdown("---")

# =====================================================
# JUNCTION FILTER
# =====================================================

selected_junction = st.selectbox(
    "🔍 Select Junction",
    ["All Junctions"] + sorted(hotspots["junction_name"].unique().tolist())
)

if selected_junction != "All Junctions":
    hotspots_filtered = hotspots[hotspots["junction_name"] == selected_junction]
else:
    hotspots_filtered = hotspots

st.caption(f"Showing {hotspots_filtered.shape[0]} of {hotspots.shape[0]} junctions")

st.markdown("---")

# =====================================================
# KPI ROW (reacts to filter)
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Junctions Shown", hotspots_filtered["junction_name"].nunique())

with c2:
    st.metric("Total Violations", f"{int(hotspots_filtered['total_violations'].sum()):,}")

with c3:
    st.metric("Total Clusters", clusters["cluster"].nunique())

with c4:
    avg_vpv = hotspots_filtered["violations_per_vehicle"].mean()
    st.metric("Avg Violations / Vehicle", round(avg_vpv, 2) if not pd.isna(avg_vpv) else 0)

st.markdown("---")

# =====================================================
# AI INSIGHT (reacts to filter)
# =====================================================

if not hotspots_filtered.empty:
    top_hotspot = hotspots_filtered.sort_values("total_violations", ascending=False).iloc[0]

    j_name = top_hotspot.get("junction_name", "Unknown Junction")
    violations = top_hotspot.get("total_violations", 0)
    peak_ratio = top_hotspot.get("peak_hour_ratio", 0)
    repeat_rate = top_hotspot.get("repeat_rate", 0)
    severity_score = top_hotspot.get("weighted_violation_score", 0)

    if pd.isna(violations):
        violations = 0
    if pd.isna(peak_ratio):
        peak_ratio = 0
    if pd.isna(repeat_rate):
        repeat_rate = 0
    if pd.isna(severity_score):
        severity_score = 0

    violations = int(violations)
    peak_ratio = float(peak_ratio)
    repeat_rate = float(repeat_rate)
    severity_score = float(severity_score)

    insight_title = "Top Violation Hotspot" if selected_junction == "All Junctions" else f"Profile — {j_name}"

    st.success(
        f"""
        🤖 **AI Insight — {insight_title}**

        **{j_name}** has **{violations:,} total violations** recorded.

        **{round(peak_ratio * 100, 1)}%** of these violations happen during
        peak hours, and **{round(repeat_rate * 100, 1)}%** involve repeat
        offending vehicles.

        **What this means:** Since violations cluster around peak hours
        and repeat vehicles, this junction would benefit most from
        **scheduled peak-hour patrols** rather than random-time checks.
        """
    )

    # =====================================================
    # SEVERITY BADGE
    # =====================================================

    if severity_score > 80:
        st.error("🔴 Critical Hotspot — Immediate intervention required")
    elif severity_score > 50:
        st.warning("🟠 High Severity Hotspot — Needs active monitoring")
    else:
        st.success("🟢 Moderate Hotspot — Routine monitoring sufficient")

    # =====================================================
    # TOP-3 RECOMMENDED ACTIONS (adapts to data)
    # =====================================================

    st.subheader("🎯 Recommended Actions")

    actions = []

    if peak_ratio > 0.5:
        actions.append("Deploy peak-hour patrol teams to target concentrated violation windows")
    else:
        actions.append("Maintain routine patrol schedule; violations are spread across the day")

    if repeat_rate > 0.3:
        actions.append("Increase repeat offender enforcement (fines, vehicle flagging)")
    else:
        actions.append("Focus on first-time offender awareness rather than repeat tracking")

    if severity_score > 50:
        actions.append("Install ANPR camera monitoring for continuous violation detection")
    else:
        actions.append("Periodic manual checks are sufficient at this severity level")

    for i, action in enumerate(actions, start=1):
        st.markdown(f"**{i}.** {action}")

else:
    st.info("No hotspot data available to generate an insight.")

st.markdown("---")

# =====================================================
# CHARTS (use filtered data)
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 15 Hotspot Junctions")

    top15 = hotspots_filtered.sort_values("total_violations", ascending=False).head(15)

    fig = px.bar(
        top15,
        x="total_violations",
        y="junction_name",
        orientation="h",
        title="Junctions Ranked by Violation Count",
        labels={"total_violations": "Total Violations", "junction_name": "Junction"},
        color="weighted_violation_score",
        color_continuous_scale="Reds"
    )
    fig.update_layout(height=550, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Violations by Cluster")

    cluster_sorted = clusters.sort_values("total_violations", ascending=False)

    fig2 = px.bar(
        cluster_sorted,
        x="cluster",
        y="total_violations",
        title="Total Violations per Cluster",
        labels={"cluster": "Cluster", "total_violations": "Total Violations"},
        color="unique_junctions",
        color_continuous_scale="Blues"
    )
    fig2.update_layout(height=550)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Peak Hour Ratio — Top 10 Junctions")

    top_peak = hotspots_filtered.sort_values("peak_hour_ratio", ascending=False).head(10)

    fig3 = px.bar(
        top_peak,
        x="peak_hour_ratio",
        y="junction_name",
        orientation="h",
        title="Highest Peak-Hour Concentration",
        labels={"peak_hour_ratio": "Peak Hour Ratio", "junction_name": "Junction"}
    )
    fig3.update_layout(height=450, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Repeat Offender Rate — Top 10 Junctions")

    top_repeat = hotspots_filtered.sort_values("repeat_rate", ascending=False).head(10)

    fig4 = px.bar(
        top_repeat,
        x="repeat_rate",
        y="junction_name",
        orientation="h",
        title="Highest Repeat Vehicle Rate",
        labels={"repeat_rate": "Repeat Rate", "junction_name": "Junction"}
    )
    fig4.update_layout(height=450, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# =====================================================
# EXECUTIVE SUMMARY CARD
# =====================================================

if not hotspots_filtered.empty:
    exec_total_junctions = hotspots_filtered["junction_name"].nunique()
    exec_total_violations = int(hotspots_filtered["total_violations"].sum())
    exec_top_junction = hotspots_filtered.sort_values(
        "total_violations", ascending=False
    ).iloc[0].get("junction_name", "Unknown")

    st.info(
        f"""
        **📋 Executive Summary**

        • **{exec_total_junctions}** hotspot junction(s) shown
        • **{exec_total_violations:,}** violations analyzed
        • Highest hotspot: **{exec_top_junction}**
        • Recommended strategy: **Peak-hour targeted enforcement**
        """
    )

st.markdown("---")

# =====================================================
# FULL TABLE
# =====================================================

st.subheader("Full Hotspot Table")

display_cols = [
    "junction_name",
    "total_violations",
    "unique_vehicles",
    "peak_violations",
    "weighted_violation_score",
    "peak_hour_ratio",
    "repeat_rate",
    "unique_violation_types",
    "unique_vehicle_types",
    "violations_per_vehicle"
]

available_cols = [c for c in display_cols if c in hotspots_filtered.columns]

display_table = hotspots_filtered[available_cols].sort_values(
    "total_violations", ascending=False
).rename(columns={
    "junction_name": "Junction",
    "total_violations": "Total Violations",
    "unique_vehicles": "Unique Vehicles",
    "peak_violations": "Peak Hour Violations",
    "weighted_violation_score": "Weighted Score",
    "peak_hour_ratio": "Peak Hour Ratio",
    "repeat_rate": "Repeat Rate",
    "unique_violation_types": "Violation Types",
    "unique_vehicle_types": "Vehicle Types",
    "violations_per_vehicle": "Violations / Vehicle"
})

st.dataframe(display_table, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("NAMMAPARK AI • Smart Parking Intelligence Platform")