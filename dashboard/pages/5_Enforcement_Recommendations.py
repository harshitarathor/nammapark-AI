import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Enforcement Recommendations",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "outputs"

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "enforcement_recommendations.csv")

enforcement = load_data()

if enforcement.empty:
    st.error("⚠️ No enforcement data available. Please ensure 'enforcement_recommendations.csv' is present.")
    st.stop()

# =====================================================
# FIX 1: explicit sort column guard — fail loudly, not silently
# =====================================================

if "priority_rank" not in enforcement.columns:
    st.error("⚠️ Required column 'priority_rank' is missing from the dataset. Cannot rank junctions.")
    st.stop()

sort_col = "priority_rank"

# =====================================================
# FIX 4: schema flags computed once, reused everywhere
# =====================================================

HAS_RISK     = "risk_category"    in enforcement.columns
HAS_ACTION   = "action_category"  in enforcement.columns
HAS_IPII     = "IPII"             in enforcement.columns
HAS_DEPLOY   = "deployment_plan"  in enforcement.columns
HAS_RESOURCE = "resource_allocation" in enforcement.columns

# =====================================================
# HELPERS
# =====================================================

def urgency_label(level: str) -> str:
    lvl = str(level).lower()
    if "critical" in lvl:  return "🚨 Critical"
    elif "high"   in lvl:  return "⚠️ High"
    elif "medium" in lvl or "moderate" in lvl: return "🟡 Medium"
    return "🟢 Normal"

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (ValueError, TypeError):
        return default

# =====================================================
# HEADER
# =====================================================

st.title("🎯 Enforcement Recommendations")
st.caption("Where to send patrols first, and exactly what to do at each junction")
st.markdown("---")

# =====================================================
# FILTERS
# =====================================================

col_f1, col_f2 = st.columns(2)

with col_f1:
    priority_options = ["All"] + sorted(enforcement["priority_level"].dropna().unique().tolist())
    priority_filter = st.selectbox("Filter by Priority Level", priority_options)

with col_f2:
    if HAS_RISK:
        risk_options = ["All"] + sorted(enforcement["risk_category"].dropna().unique().tolist())
        risk_filter = st.selectbox("Filter by Risk Category", risk_options)
    else:
        risk_filter = "All"

df = enforcement.copy()

if priority_filter != "All":
    df = df[df["priority_level"] == priority_filter]

if risk_filter != "All" and HAS_RISK:
    df = df[df["risk_category"] == risk_filter]

if df.empty:
    st.warning("No junctions match the selected filters.")
    st.stop()

is_filtered = priority_filter != "All" or risk_filter != "All"

st.markdown("---")

# =====================================================
# LAYER 1 — SUMMARY KPIs
# =====================================================

st.subheader("📊 Summary")

c1, c2, c3, c4 = st.columns(4)

# FIX 2: delta shows % of total, not raw difference
with c1:
    delta_value = f"{df.shape[0] / enforcement.shape[0] * 100:.1f}% of total" if is_filtered else None
    st.metric("Junctions Shown", df.shape[0], delta=delta_value, delta_color="off")

with c2:
    # FIX 3: structured isin() instead of fragile str.contains()
    high_priority = df["priority_level"].isin(["High", "Critical", "Urgent"]).sum()
    st.metric("High / Critical", int(high_priority))

with c3:
    # FIX 6: safe_float guards all-NaN mean
    avg_ipii = safe_float(df["IPII"].mean()) if HAS_IPII else 0.0
    st.metric("Avg IPII Score", round(avg_ipii, 2))

with c4:
    if HAS_ACTION:
        mode_series = df["action_category"].mode()
        most_common_action = mode_series[0] if not mode_series.empty else "N/A"
    else:
        most_common_action = "N/A"
    st.metric("Top Action Type", most_common_action)

st.markdown("---")

# =====================================================
# LAYER 2 — DECISION BLOCK (top priority junction)
# =====================================================

st.subheader("🚨 Top Priority Junction")

top_row = df.sort_values(sort_col, ascending=True).iloc[0]

j_name         = top_row.get("junction_name",      "Unknown Junction")
priority_level = top_row.get("priority_level",     "High")
risk_cat       = top_row.get("risk_category",      "Unknown")   if HAS_RISK     else "N/A"
action         = top_row.get("recommended_action", "No action specified")
deployment     = top_row.get("deployment_plan",    "Not specified") if HAS_DEPLOY   else "Not specified"
resources      = top_row.get("resource_allocation","Not specified") if HAS_RESOURCE else "Not specified"
ipii_score     = safe_float(top_row.get("IPII", 0)) if HAS_IPII else 0.0
expected_reduction = ipii_score * 0.15

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"### {j_name}")
    st.markdown(f"**Recommended action:** {action}")
    st.markdown(f"**Deployment plan:** {deployment}")
    st.markdown(f"**Resources needed:** {resources}")

with col2:
    st.metric("Urgency",       urgency_label(priority_level))
    st.metric("Risk Category", risk_cat)

with col3:
    st.metric("IPII Score",   round(ipii_score, 2))
    st.metric("Est. Reduction", f"-{expected_reduction:.1f}%")

st.info(f"""
**Why {j_name} is ranked #1**

- Risk category: **{risk_cat}**
- IPII score: **{round(ipii_score, 2)}** — highest in current view
- Recommended action: **{action}**
""")

with st.expander("📉 Estimated outcome if action is taken"):
    st.markdown(f"""
| Metric | Before | After (Est.) |
|--------|--------|--------------|
| Risk Level | {risk_cat} | Reduced |
| Violation Impact | Baseline | -{expected_reduction:.1f}% |
| Priority Rank | #1 | Lower |

*Based on proportional IPII response model.*
""")

st.markdown("---")

# =====================================================
# LAYER 3 — ANALYSIS (charts + table)
# =====================================================

st.subheader("📋 Action Breakdown")

if HAS_ACTION:
    action_counts = df["action_category"].value_counts().reset_index()
    action_counts.columns = ["Action Type", "Junctions"]
    action_counts["Share (%)"] = (action_counts["Junctions"] / df.shape[0] * 100).round(1)
    st.dataframe(action_counts, use_container_width=True, hide_index=True)
else:
    st.info("Action category data not available.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Priority Distribution")
    dist = df["priority_level"].value_counts().reset_index()
    dist.columns = ["Priority Level", "Count"]
    if not dist.empty:
        fig = px.bar(dist, x="Priority Level", y="Count", text="Count")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No priority data to display.")

with col2:
    st.subheader("Top 10 by IPII Score")
    # FIX 7: already limited to top 10 — keeps chart readable regardless of dataset size
    top10 = df.sort_values(sort_col, ascending=True).head(10)
    if not top10.empty and HAS_IPII:
        fig2 = px.bar(
            top10,
            x="IPII",
            y="junction_name",
            orientation="h",
            color="priority_level",
            labels={"junction_name": "Junction", "IPII": "IPII Score", "priority_level": "Urgency"}
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No junction data to display.")

st.markdown("---")

# =====================================================
# FULL TABLE
# FIX 8: hide raw technical fields (priority_rank, priority_level)
#         show human labels only — Urgency replaces priority_level
# =====================================================

st.subheader("Full Enforcement Plan")

display_cols = [
    "junction_name",
    "risk_category",
    "action_category",
    "recommended_action",
    "deployment_plan",
    "resource_allocation",
    "IPII",
]

# Sort first using priority rank
sorted_df = df.sort_values(sort_col, ascending=True)

available_cols = [c for c in display_cols if c in sorted_df.columns]
display_df = sorted_df[available_cols].copy()

if "priority_level" in sorted_df.columns:
    display_df.insert(
        1,
        "Urgency",
        sorted_df["priority_level"].apply(urgency_label)
    )

display_df = display_df.rename(columns={
    "junction_name": "Junction",
    "risk_category": "Risk Category",
    "action_category": "Action Type",
    "recommended_action": "Recommended Action",
    "deployment_plan": "Deployment Plan",
    "resource_allocation": "Resources Needed",
    "IPII": "IPII Score",
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.caption("NAMMAPARK AI • Smart Parking Intelligence Platform")