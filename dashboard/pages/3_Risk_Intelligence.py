import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Risk Intelligence",
    page_icon="⚠️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "outputs"

risk = pd.read_csv(DATA_DIR / "risk_scores.csv")

# -----------------------------------------------------
# FIX: the upstream pipeline's risk_explanation column embedded an entire
# pandas Series into the text instead of a single row's value, so every
# row shows a garbled dump (e.g. "0   50.8 1   58.0 ... dtype: str").
# Rebuild a clean, accurate explanation here using the correctly-stored
# per-row scalar columns instead of trusting that broken text.
# -----------------------------------------------------

def _build_explanation(row):
    factor = row.get("top_contributing_factor", "Unknown factor")
    share = row.get("top_factor_share_pct", 0)
    if pd.isna(factor):
        factor = "Unknown factor"
    if pd.isna(share):
        share = 0
    return (
        f"IPII = 50% severity + 30% behavioral + 20% density. "
        f"Primary driver for this junction: {factor} "
        f"({round(float(share), 1)}% of weighted score)."
    )

risk["risk_explanation"] = risk.apply(_build_explanation, axis=1)

# Consistent color map used across every chart on this page.
# NOTE: must match the real values in risk_category ("Moderate", not "Medium").
RISK_COLORS = {
    "Critical": "#d62728",
    "High": "#ff7f0e",
    "Moderate": "#f7d020",
    "Low": "#2ca02c"
}

st.title("⚠️ Risk Intelligence")
st.caption("AI-driven congestion-risk scoring (IPII) with explainable factor breakdown")
st.markdown("---")

# =====================================================
# KPI ROW
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Critical Junctions", int(risk["risk_category"].eq("Critical").sum()))

with c2:
    st.metric("Average IPII", round(risk["IPII"].mean(), 2))

with c3:
    st.metric("Highest IPII Score", round(risk["IPII"].max(), 2))

with c4:
    st.metric("Total Junctions Scored", risk.shape[0])

st.markdown("---")

# =====================================================
# JUNCTION FILTER
# =====================================================

selected_junction = st.selectbox(
    "🔍 Select Junction",
    ["All Junctions"] + sorted(risk["junction_name"].unique().tolist())
)

if selected_junction != "All Junctions":
    risk_filtered = risk[risk["junction_name"] == selected_junction]
else:
    risk_filtered = risk

st.caption(f"Showing risk profile for {risk_filtered.shape[0]} junction(s).")

st.markdown("---")

# =====================================================
# AI INSIGHT — TOP JUNCTION, CLEAN PLAIN-ENGLISH EXPLANATION
# =====================================================

if not risk_filtered.empty:
    top_risk = risk_filtered.sort_values("IPII", ascending=False).iloc[0]

    junction_name = top_risk.get("junction_name", "Unknown Junction")
    ipii_score = top_risk.get("IPII", 0)
    risk_category = top_risk.get("risk_category", "Unknown")
    explanation = top_risk.get("risk_explanation", "No explanation available.")
    top_factor = top_risk.get("top_contributing_factor", "Unknown factor")
    factor_share = top_risk.get("top_factor_share_pct", 0)
    risk_rank = top_risk.get("risk_rank", None)

    # Guard against NaN values slipping through .get() (column exists but cell is empty)
    if pd.isna(ipii_score):
        ipii_score = 0
    if pd.isna(factor_share):
        factor_share = 0
    if pd.isna(top_factor):
        top_factor = "Unknown factor"

    rank_line = (
        f"Citywide rank: **#{int(risk_rank)} of {risk.shape[0]}** by IPII\n\n"
        if risk_rank is not None and not pd.isna(risk_rank)
        else ""
    )

    st.success(
        f"""
        🤖 **AI Insight — Most Critical Junction**

        **{junction_name}** needs urgent attention.
        Its risk score (IPII) is **{round(float(ipii_score), 1)}**, placing it
        in the **{risk_category}** category.

        {rank_line}**What's causing this:** {explanation}

        **Main reason:** {top_factor} is responsible for about
        **{round(float(factor_share), 1)}%** of this junction's risk.

        **What this means:** Focus enforcement on this specific issue at
        this junction first, rather than spreading patrols everywhere.
        """
    )

    # =====================================================
    # RECOMMENDED INTERVENTION
    # =====================================================

    st.subheader("🚨 Recommended Intervention")

    if risk_category == "Critical":
        st.error("Immediate enforcement deployment recommended.")
    elif risk_category == "High":
        st.warning("Increased monitoring recommended.")
    elif risk_category == "Moderate":
        st.info("Routine monitoring recommended.")
    else:
        st.success("Current conditions appear stable.")

    # =====================================================
    # COMPARISON TO CITY AVERAGE
    # =====================================================

    city_avg_ipii = risk["IPII"].mean()
    if not pd.isna(city_avg_ipii) and city_avg_ipii > 0:
        diff_pct = ((float(ipii_score) - city_avg_ipii) / city_avg_ipii) * 100
        comparison_text = (
            f"**{round(diff_pct, 1)}% above** the citywide average IPII ({round(city_avg_ipii, 1)})"
            if diff_pct > 0
            else f"**{round(abs(diff_pct), 1)}% below** the citywide average IPII ({round(city_avg_ipii, 1)})"
        )
        st.caption(f"📊 This junction's risk score is {comparison_text}.")
else:
    st.info("No risk data available to generate an insight.")

st.markdown("---")

# =====================================================
# CHARTS
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Category Breakdown")

    dist = risk["risk_category"].value_counts().reset_index()
    dist.columns = ["Risk Category", "Count"]

    fig = px.pie(
        dist,
        names="Risk Category",
        values="Count",
        hole=0.5,
        color="Risk Category",
        color_discrete_map=RISK_COLORS
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top 10 Highest-Risk Junctions")

    top10 = risk.sort_values("IPII", ascending=False).head(10)

    fig2 = px.bar(
        top10,
        x="IPII",
        y="junction_name",
        orientation="h",
        color="risk_category",
        title="Top Junctions by IPII Score",
        color_discrete_map=RISK_COLORS
    )
    fig2.update_layout(height=480, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📈 IPII Score Distribution")
st.caption("Citywide spread of risk scores — shows whether risk is concentrated in a few junctions or spread across many.")

city_avg_ipii_hist = risk["IPII"].mean()

fig_hist = px.histogram(
    risk,
    x="IPII",
    nbins=15,
    title="Distribution of Risk Scores Across Junctions",
    labels={"IPII": "IPII Score", "count": "Number of Junctions"},
    color_discrete_sequence=["#415A77"]
)
fig_hist.update_traces(marker_line_width=1, marker_line_color="white")
fig_hist.add_vline(
    x=city_avg_ipii_hist,
    line_dash="dash",
    line_color="#d62728",
    annotation_text=f"City Avg: {round(city_avg_ipii_hist, 1)}",
    annotation_position="top"
)
fig_hist.update_layout(
    height=420,
    bargap=0.1,
    yaxis_title="Number of Junctions",
    xaxis_title="IPII Score"
)
st.plotly_chart(fig_hist, use_container_width=True)

# =====================================================
# CITYWIDE RISK SNAPSHOT (aggregate stats, not a single junction)
# =====================================================

st.subheader("📌 Citywide Risk Snapshot")

pct_critical_high = round(
    risk["risk_category"].isin(["Critical", "High"]).mean() * 100, 1
)
median_ipii = round(risk["IPII"].median(), 1)
most_common_factor = risk["top_contributing_factor"].value_counts().idxmax()

snap1, snap2, snap3 = st.columns(3)

with snap1:
    st.metric("Citywide Avg IPII", round(city_avg_ipii_hist, 1))
with snap2:
    st.metric("Median IPII", median_ipii)
with snap3:
    st.metric("Critical + High Junctions", f"{pct_critical_high}%")

st.info(
    f"""
    **Key takeaway:** **{pct_critical_high}%** of junctions citywide fall
    into the Critical or High risk category. The median IPII ({median_ipii})
    sitting {"below" if median_ipii < city_avg_ipii_hist else "above"} the
    average ({round(city_avg_ipii_hist, 1)}) suggests risk is
    {"concentrated in a smaller set of high-risk junctions" if median_ipii < city_avg_ipii_hist else "fairly evenly spread across junctions"}.
    Citywide, **{most_common_factor}** is the most frequent primary risk driver.
    """
)

st.markdown("---")

# =====================================================
# WHAT'S DRIVING RISK ACROSS THE CITY (TABLE FORMAT)
# =====================================================

st.subheader("📊 What's Driving Risk Across the City")
st.caption("Each junction's risk is mainly caused by one of these factors. This shows how common each cause is.")

factor_counts = risk["top_contributing_factor"].value_counts().reset_index()
factor_counts.columns = ["Main Cause of Risk", "Number of Junctions"]

factor_counts["% of All Junctions"] = (
    factor_counts["Number of Junctions"] / risk.shape[0] * 100
).round(1)

factor_counts = factor_counts.sort_values("Number of Junctions", ascending=False)

st.dataframe(
    factor_counts,
    use_container_width=True,
    hide_index=True
)

if not factor_counts.empty:
    top_row = factor_counts.iloc[0]
    st.info(
        f"""
        **Simple takeaway:** "{top_row['Main Cause of Risk']}" is the
        #1 reason for risk, affecting **{top_row['Number of Junctions']} junctions**
        (**{top_row['% of All Junctions']}%** of all junctions analyzed).

        This means enforcement teams should prioritize fixing this specific
        issue first, since it impacts the most locations.
        """
    )

st.markdown("---")

# =====================================================
# SCORE COMPONENT BREAKDOWN
# =====================================================

st.subheader("🔬 Risk Score Composition — Top 5 Critical Junctions")

top5 = risk.sort_values("IPII", ascending=False).head(5)

score_cols = ["severity_score", "behavioral_score", "density_score"]
available_score_cols = [c for c in score_cols if c in top5.columns]

if available_score_cols:
    score_components = top5[["junction_name"] + available_score_cols].melt(
        id_vars="junction_name",
        var_name="Component",
        value_name="Score"
    )

    fig4 = px.bar(
        score_components,
        x="junction_name",
        y="Score",
        color="Component",
        barmode="group",
        title="Severity vs Behavioral vs Density Score Breakdown"
    )
    fig4.update_layout(height=450, xaxis_tickangle=-20)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Score component columns not found in this dataset.")

st.markdown("---")

# =====================================================
# FULL EXPLAINABLE RISK TABLE
# =====================================================

st.subheader("Full Risk Ranking with AI Explanations")

display_cols = [
    "risk_rank",
    "junction_name",
    "risk_category",
    "IPII",
    "top_contributing_factor",
    "top_factor_share_pct",
    "risk_explanation"
]

available_cols = [c for c in display_cols if c in risk.columns]

st.dataframe(
    risk[available_cols].sort_values("risk_rank") if "risk_rank" in available_cols else risk[available_cols],
    use_container_width=True
)

st.markdown("---")
st.caption("NAMMAPARK AI • Smart Parking Intelligence Platform")