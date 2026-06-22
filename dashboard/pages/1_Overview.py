import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Overview",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = (
    BASE_DIR
    / "outputs"
)

hotspots = pd.read_csv(
    DATA_DIR / "hotspots.csv"
)

risk = pd.read_csv(
    DATA_DIR / "risk_scores.csv"
)

emerging = pd.read_csv(
    DATA_DIR / "emerging_hotspots.csv"
)

enforcement = pd.read_csv(
    DATA_DIR / "enforcement_recommendations.csv"
)

# =====================================================
# HEADER
# =====================================================

st.title("🚦 NAMMAPARK AI")

st.markdown(
    """
    ### AI-Driven Parking Intelligence & Congestion Risk Platform
    """
)

st.markdown("---")

# =====================================================
# KPI SECTION
# =====================================================

total_violations = int(
    hotspots["total_violations"].sum()
)

total_junctions = hotspots[
    "junction_name"
].nunique()

critical_count = (
    risk["risk_category"]
    .eq("Critical")
    .sum()
)

emerging_count = (
    emerging["trend_category"]
    .eq("Emerging")
    .sum()
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Total Violations",
        f"{total_violations:,}"
    )

with c2:
    st.metric(
        "Junctions",
        total_junctions
    )

with c3:
    st.metric(
        "Critical Zones",
        int(critical_count)
    )

with c4:
    st.metric(
        "Emerging Hotspots",
        int(emerging_count)
    )

st.markdown("---")

# =====================================================
# AI INSIGHT BOX (FIXED: now sorted by IPII, not just first row)
# =====================================================

if not risk.empty:
    top_risk = risk.sort_values("IPII", ascending=False).iloc[0]

    junction_name = top_risk.get("junction_name", "Unknown Junction")
    risk_category = top_risk.get("risk_category", "Unknown")
    ipii_score = top_risk.get("IPII", 0)

    if pd.isna(ipii_score):
        ipii_score = 0

    st.success(
        f"""
        🤖 **AI Insight — Highest Priority Junction**

        **{junction_name}** is the city's top-priority junction right now.

        **Risk category:** {risk_category}
        **IPII score:** {round(float(ipii_score), 1)}

        Head to the **Risk Intelligence** page for a full explanation of
        what's driving this junction's risk score.
        """
    )
else:
    st.info("No risk data available to generate an insight.")

# =====================================================
# CHARTS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Top 10 Hotspots"
    )

    top_hotspots = (
        hotspots
        .sort_values(
            "total_violations",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top_hotspots,
        x="total_violations",
        y="junction_name",
        orientation="h",
        title="Top Junctions by Violations"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    st.subheader(
        "Risk Distribution"
    )

    risk_dist = (
        risk["risk_category"]
        .value_counts()
        .reset_index()
    )

    risk_dist.columns = [
        "Risk Category",
        "Count"
    ]

    fig = px.pie(
        risk_dist,
        names="Risk Category",
        values="Count",
        hole=0.5
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# EMERGING HOTSPOTS
# =====================================================

st.markdown("---")

st.subheader(
    "🚀 Top Emerging Hotspots"
)

top_emerging = (
    emerging
    .sort_values(
        "emerging_score",
        ascending=False
    )
    .head(10)
)

fig = px.bar(
    top_emerging,
    x="junction_name",
    y="emerging_score",
    color="trend_direction",
    title="Emerging Hotspot Score"
)

fig.update_layout(
    xaxis_tickangle=-45,
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# ENFORCEMENT PRIORITY TABLE (FIXED: matches real schema)
# =====================================================

st.markdown("---")

st.subheader(
    "🎯 Top Enforcement Priorities"
)

display_cols = [
    "priority_rank",
    "junction_name",
    "risk_category",
    "priority_level",
    "recommended_action"
]

available_cols = [c for c in display_cols if c in enforcement.columns]

sort_col = "priority_rank" if "priority_rank" in available_cols else available_cols[0]

st.dataframe(
    enforcement[available_cols].sort_values(sort_col).head(10),
    use_container_width=True,
    hide_index=True
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "NAMMAPARK AI • Smart Parking Intelligence Platform"
)
