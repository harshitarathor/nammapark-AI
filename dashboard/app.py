import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="NAMMAPARK AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD DATA
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "outputs"

try:
    hotspots = pd.read_csv(DATA_DIR / "hotspots.csv")
    risk = pd.read_csv(DATA_DIR / "risk_scores.csv")
    emerging = pd.read_csv(DATA_DIR / "emerging_hotspots.csv")
    clusters = pd.read_csv(DATA_DIR / "cluster_summary.csv")
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# =====================================================
# HEADER
# =====================================================

st.title("🚦 NAMMAPARK AI")
st.subheader(
    "AI-Driven Parking Intelligence & Congestion Risk Platform"
)

st.markdown("---")

if data_loaded:

    # =====================================================
    # KPI CARDS
    # =====================================================

    total_violations = int(hotspots["total_violations"].sum())
    total_junctions = hotspots["junction_name"].nunique()
    critical_zones = int(risk["risk_category"].eq("Critical").sum())
    emerging_zones = int(emerging["trend_category"].eq("Emerging").sum())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Violations", f"{total_violations:,}")

    with c2:
        st.metric("Junctions", total_junctions)

    with c3:
        st.metric("Critical Risk Zones", critical_zones)

    with c4:
        st.metric("Emerging Hotspots", emerging_zones)

    st.markdown("---")

    # =====================================================
    # AI INSIGHT — DYNAMIC, FROM REAL DATA
    # =====================================================

    top_risk = risk.sort_values("IPII", ascending=False).iloc[0]
    junction_name = top_risk.get("junction_name", "Unknown Junction")
    ipii_score = top_risk.get("IPII", 0)
    if pd.isna(ipii_score):
        ipii_score = 0

    st.success(
        f"""
        🤖 **AI Insight**

        **"{junction_name}"** has the highest enforcement priority score
        (IPII: **{round(float(ipii_score), 1)}**) and should receive
        immediate resource allocation.
        """
    )

    st.markdown("---")

    # =====================================================
    # PROJECT IMPACT
    # =====================================================

    st.subheader("📈 Project Impact")

    total_clusters = clusters["cluster"].nunique() if "cluster" in clusters.columns else clusters.shape[0]

    impact_col1, impact_col2 = st.columns(2)

    with impact_col1:
        st.markdown(
            f"""
            ✅ **{total_violations:,}** violations analyzed
            ✅ **{total_junctions}** junctions monitored
            """
        )

    with impact_col2:
        st.markdown(
            f"""
            ✅ **{total_clusters}** spatial clusters detected
            ✅ **{critical_zones}** critical zones identified
            """
        )

    st.markdown("---")

else:
    st.warning(
        "⚠️ Data files not found. Make sure the `outputs/` folder is "
        "present with hotspots.csv, risk_scores.csv, emerging_hotspots.csv, "
        "and cluster_summary.csv."
    )
    st.markdown("---")

# =====================================================
# MODULE NAVIGATION
# =====================================================

st.subheader("🧭 Module Navigation")

st.markdown(
    """
    | Module | What it shows |
    |---|---|
    | **Overview** | City-wide summary and AI insight |
    | **Hotspot Analysis** | Where violations concentrate, by junction and cluster |
    | **Risk Intelligence** | AI-driven IPII risk scoring with explainable factors |
    | **Emerging Hotspots** | Junctions trending toward critical, before they get there |
    | **Enforcement Recommendations** | Prioritized patrol deployment plan |
    | **Geospatial Map** | Interactive map of violations color-coded by risk |
    """
)

st.info("👈 Use the sidebar to explore each module in detail.")

st.markdown("---")
st.caption("NAMMAPARK AI • Smart Parking Intelligence Platform")

