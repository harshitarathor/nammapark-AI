import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Emerging Hotspots",
    page_icon="🚀",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "outputs"

emerging = pd.read_csv(DATA_DIR / "emerging_hotspots.csv")

st.title("🚀 Emerging Hotspots")
st.caption("Junctions where violations are rising fast — catch problems before they become critical")
st.markdown("---")

# =====================================================
# KPI ROW
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Emerging Junctions", int(emerging["trend_category"].eq("Emerging").sum()))

with c2:
    st.metric("Average Growth Rate", f"{round(emerging['growth_rate_percent'].mean(), 1)}%")

with c3:
    st.metric("Fastest Growing", f"{round(emerging['growth_rate_percent'].max(), 1)}%")

with c4:
    st.metric("Total Junctions Tracked", emerging.shape[0])

st.markdown("---")

# =====================================================
# CITYWIDE SNAPSHOT (always citywide, never filtered)
# =====================================================

pct_emerging = round(emerging["trend_category"].eq("Emerging").mean() * 100, 1)
avg_growth = round(emerging["growth_rate_percent"].mean(), 1)
fastest = emerging.loc[emerging["growth_rate_percent"].idxmax(), "junction_name"]

st.info(
    f"""
    📌 **Citywide Snapshot**

    • {pct_emerging}% of tracked junctions are currently emerging.

    • Average growth rate is {avg_growth}%.

    • The fastest-growing junction is **{fastest}**.

    These locations should be monitored before they become
    high-risk enforcement zones.
    """
)

st.markdown("---")

# =====================================================
# JUNCTION FILTER
# =====================================================

selected_junction = st.selectbox(
    "🔍 Select Junction",
    ["All Junctions"] + sorted(emerging["junction_name"].unique().tolist())
)

if selected_junction != "All Junctions":
    emerging_filtered = emerging[emerging["junction_name"] == selected_junction]
else:
    emerging_filtered = emerging

st.caption(f"Showing {emerging_filtered.shape[0]} junction(s).")

st.markdown("---")

# =====================================================
# AI INSIGHT — TOP EMERGING JUNCTION, IN PLAIN ENGLISH
# =====================================================

if not emerging_filtered.empty:
    top_emerging_row = emerging_filtered.sort_values("emerging_score", ascending=False).iloc[0]

    j_name = top_emerging_row.get("junction_name", "Unknown Junction")
    growth = top_emerging_row.get("growth_rate_percent", 0)
    earlier_avg = top_emerging_row.get("earlier_period_avg", 0)
    recent_avg = top_emerging_row.get("recent_period_avg", 0)
    trend_dir = top_emerging_row.get("trend_direction", "Unknown")
    priority = top_emerging_row.get("final_priority", None)

    # Guard against NaN/missing values
    if pd.isna(growth):
        growth = 0
    if pd.isna(earlier_avg):
        earlier_avg = 0
    if pd.isna(recent_avg):
        recent_avg = 0

    # FIX: final_priority is a numeric score (e.g. 63.96), not a category
    # label — so it must be framed as "a score, higher = more urgent",
    # not presented as if it were a tier name like "High"/"Critical".
    if priority is None or pd.isna(priority):
        priority_text = "Not available"
    else:
        priority_text = f"{round(float(priority), 1)} (higher = more urgent)"

    st.success(
        f"""
        🤖 **AI Insight — Fastest Rising Junction**

        **{j_name}** is the junction to watch right now.

        Violations went from an average of **{round(float(earlier_avg), 1)}/month**
        to **{round(float(recent_avg), 1)}/month** — a growth of **{round(float(growth), 1)}%**.

        Trend direction: **{trend_dir}**
        Priority score: **{priority_text}**

        **What this means:** This junction isn't critical yet, but it's heading
        that way fast. Acting now (before it becomes a "Critical" zone) is
        cheaper and more effective than reacting later.
        """
    )

    # =====================================================
    # RISK ESCALATION WARNING
    # =====================================================

    st.subheader("⚠️ Risk Escalation Warning")

    growth = float(growth)
    if growth > 50:
        st.error("🚨 Rapid escalation detected. This junction may become a major hotspot soon.")
    elif growth > 20:
        st.warning("⚠️ Moderate growth detected. Monitor closely.")
    else:
        st.success("Stable growth pattern.")

    # =====================================================
    # RECOMMENDED ACTION
    # =====================================================

    st.subheader("🚨 Recommended Action")

    if growth > 50:
        st.error("Deploy enforcement resources immediately and investigate root causes.")
    elif growth > 20:
        st.warning("Increase monitoring frequency and conduct inspections.")
    else:
        st.info("Continue routine observation.")
else:
    st.info("No emerging hotspot data available.")

st.markdown("---")

# =====================================================
# TOP 10 EMERGING — SIMPLE TABLE
# =====================================================

st.subheader("📈 Top 10 Fastest-Growing Junctions")
st.caption("Sorted by how urgently each junction needs attention (emerging score)")

top10 = emerging.sort_values("emerging_score", ascending=False).head(10)

table_cols = [
    "junction_name",
    "growth_rate_percent",
    "earlier_period_avg",
    "recent_period_avg",
    "trend_direction",
    "final_priority"
]
available_cols = [c for c in table_cols if c in top10.columns]

display_table = top10[available_cols].rename(columns={
    "junction_name": "Junction",
    "growth_rate_percent": "Growth Rate (%)",
    "earlier_period_avg": "Avg Before (per month)",
    "recent_period_avg": "Avg Now (per month)",
    "trend_direction": "Trend",
    "final_priority": "Priority Score"
})

st.dataframe(display_table, use_container_width=True, hide_index=True)

st.markdown("---")

# =====================================================
# GROWTH CHART
# =====================================================

st.subheader("Growth Rate — Visual Comparison")

fig = px.bar(
    top10,
    x="junction_name",
    y="growth_rate_percent",
    color="trend_direction",
    title="How Fast Each Junction's Violations Are Growing",
    labels={
        "junction_name": "Junction",
        "growth_rate_percent": "Growth Rate (%)"
    }
)
fig.update_layout(xaxis_tickangle=-45, height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================================
# TREND DIRECTION BREAKDOWN — SIMPLE TABLE
# =====================================================

st.subheader("📊 Trend Direction Overview")
st.caption("How many junctions are trending up, down, or staying stable")

trend_dist = emerging["trend_direction"].value_counts().reset_index()
trend_dist.columns = ["Trend Direction", "Number of Junctions"]
trend_dist["% of Total"] = (
    trend_dist["Number of Junctions"] / emerging.shape[0] * 100
).round(1)

st.dataframe(trend_dist, use_container_width=True, hide_index=True)

fig_trend = px.pie(
    trend_dist,
    names="Trend Direction",
    values="Number of Junctions",
    hole=0.5,
    title="Trend Direction Breakdown"
)
fig_trend.update_layout(height=420)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# =====================================================
# FULL TABLE
# =====================================================

st.subheader("Full Emerging Hotspot Data")

full_cols = [
    "junction_name",
    "months_active",
    "earlier_period_avg",
    "recent_period_avg",
    "growth_rate_percent",
    "latest_month_violations",
    "emerging_score",
    "emerging_rank",
    "trend_category",
    "trend_direction",
    "trend_strength",
    "final_priority"
]
available_full_cols = [c for c in full_cols if c in emerging.columns]

st.dataframe(
    emerging[available_full_cols].sort_values("emerging_rank") if "emerging_rank" in available_full_cols else emerging[available_full_cols],
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.caption("NAMMAPARK AI • Smart Parking Intelligence Platform")
