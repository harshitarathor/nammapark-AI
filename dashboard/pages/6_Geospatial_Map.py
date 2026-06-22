import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Command Center Map",
    page_icon="🚨",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "outputs"

@st.cache_data
def load_data():
    hotspots = pd.read_csv(DATA_DIR / "hotspots.csv")
    risk = pd.read_csv(DATA_DIR / "risk_scores.csv")
    return hotspots, risk

try:
    hotspots, risk = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}")
    st.stop()

if hotspots.empty:
    st.warning("No hotspot data available.")
    st.stop()

required_cols = ["junction_name", "risk_category", "IPII"]

missing = [c for c in required_cols if c not in risk.columns]

if missing:
    st.error(f"Missing columns in risk_scores.csv: {missing}")
    st.stop()

# =====================================================
# MERGE DATA
# =====================================================

risk_lookup = risk[["junction_name", "risk_category", "IPII"]]
df = hotspots.merge(risk_lookup, on="junction_name", how="left")

df["risk_category"] = df["risk_category"].fillna("Unknown")
df["IPII"] = df["IPII"].fillna(0)

# Remove rows with invalid coordinates
df = df.dropna(
    subset=["avg_latitude", "avg_longitude"]
)

if df.empty:
    st.warning("No valid coordinate data available.")
    st.stop()

# =====================================================
# HEADER
# =====================================================

st.title("🚨 Smart City Command Center Map")
st.caption("Heat intelligence • Risk zones • Enforcement prioritization")
st.markdown("---")

# =====================================================
# FILTERS
# =====================================================

col1, col2 = st.columns(2)

with col1:
    risk_filter = st.selectbox(
        "Risk Category",
        ["All"] + sorted(df["risk_category"].unique().tolist())
    )

with col2:
    view_mode = st.selectbox(
        "Map View",
        ["Point Map", "Heatmap"]
    )

filtered = df.copy()

if risk_filter != "All":
    filtered = filtered[filtered["risk_category"] == risk_filter]

if filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

st.markdown("---")

# =====================================================
# KPI ROW
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Junctions", filtered.shape[0])

with c2:
    st.metric("Total Violations", f"{int(filtered['total_violations'].sum()):,}")

with c3:
    st.metric("Critical Zones", int(filtered["risk_category"].eq("Critical").sum()))

with c4:
    avg_ipii = filtered["IPII"].fillna(0).mean()
    st.metric("Avg IPII", round(avg_ipii, 2))

st.markdown("---")

# =====================================================
# ZONE SUMMARY
# =====================================================

st.subheader("📊 Zone-Level Intelligence")

zone_summary = (
    filtered.groupby("risk_category")
    .agg(
        junctions=("junction_name", "count"),
        total_violations=("total_violations", "sum"),
        avg_ipii=("IPII", "mean")
    )
    .reset_index()
)

zone_summary["avg_ipii"] = zone_summary["avg_ipii"].round(2)

st.dataframe(zone_summary, use_container_width=True, hide_index=True)

st.markdown("---")

# =====================================================
# MAP BASE
# =====================================================

center_lat = filtered["avg_latitude"].mean()
center_lon = filtered["avg_longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles="CartoDB positron"
)

# =====================================================
# HEATMAP MODE
# =====================================================

if view_mode == "Heatmap":

    heat_data = (
        filtered[
            ["avg_latitude", "avg_longitude", "total_violations"]
        ]
        .values
        .tolist()
    )

    HeatMap(
        heat_data,
        radius=15,
        blur=10,
        max_zoom=13
    ).add_to(m)

# =====================================================
# POINT MAP MODE
# =====================================================

else:

    max_v = max(filtered["total_violations"].max(), 1)

    color_map = {
        "Critical": "red",
        "High": "orange",
        "Medium": "gold",
        "Low": "green",
        "Unknown": "gray"
    }

    for r in filtered.itertuples():

        color = color_map.get(r.risk_category, "gray")

        popup_html = f"""
        <b>{r.junction_name}</b><br>
        Violations: {int(r.total_violations):,}<br>
        Risk: {r.risk_category}<br>
        IPII: {round(r.IPII, 2)}
        """

        folium.CircleMarker(
            location=[r.avg_latitude, r.avg_longitude],
            radius=5 + (r.total_violations / max_v * 10),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

# =====================================================
# RENDER MAP
# =====================================================

st_folium(m, width=None, height=550)

st.markdown("---")

# =====================================================
# AI INSIGHT (OPTIMIZED - NO DUPLICATE SORTS)
# =====================================================

top_violation = filtered.loc[filtered["total_violations"].idxmax()]
top_ipii = filtered.loc[filtered["IPII"].idxmax()]

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    🔥 **Highest Violation Zone**

    {top_violation['junction_name']}

    - Violations: {int(top_violation['total_violations']):,}
    - Risk: {top_violation['risk_category']}
    """)

with col2:
    st.warning(f"""
    🧠 **Highest Risk Intensity Zone**

    {top_ipii['junction_name']}

    - IPII: {round(top_ipii['IPII'], 2)}
    - Risk: {top_ipii['risk_category']}
    """)

st.markdown("---")

# =====================================================
# COMMAND INSIGHT (FIXED LOGIC)
# =====================================================

top = filtered.sort_values(
    ["IPII", "total_violations"],
    ascending=[False, False]
).iloc[0]

st.success(f"""
🚨 **Command Center Insight**

{top['junction_name']} requires the highest enforcement attention in the current view.

- Violations: {int(top['total_violations']):,}
- Risk Category: {top['risk_category']}
- IPII Score: {round(top['IPII'], 2)}

👉 Recommendation:
Deploy enforcement resources here first because it combines both high risk intensity and high violation activity.
""")

st.markdown("---")

# =====================================================
# LEGEND
# =====================================================

st.subheader("Risk Legend")

cols = st.columns(5)

legend = [
    ("🔴 Critical", "Immediate intervention"),
    ("🟠 High", "High enforcement priority"),
    ("🟡 Medium", "Moderate risk"),
    ("🟢 Low", "Stable zones"),
    ("⚪ Unknown", "No classification")
]

for c, (label, desc) in zip(cols, legend):
    with c:
        st.markdown(f"**{label}**")
        st.caption(desc)

st.markdown("---")

st.caption("NAMMAPARK AI • Smart City Command Center System")
