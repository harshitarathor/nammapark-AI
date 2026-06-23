# NammaPark AI
**AI-Driven Parking Intelligence & Congestion Risk Platform**

🔗 **Live Dashboard:** [nammapark-ai.streamlit.app](https://nammapark-ai.streamlit.app/)

NammaPark AI turns ~298K raw Bengaluru parking violation records into a six-page interactive intelligence dashboard — surfacing hotspots, risk scores, emerging trouble spots, and enforcement recommendations for city traffic authorities.

---

## What it does

NammaPark AI runs raw violation data through a 7-stage pipeline into a Streamlit dashboard:

1. **Preprocess** — cleans ~298K records, engineers time-based features.
2. **EDA & Hotspot Scoring** — junction stats, Pareto analysis, anomaly detection, composite hotspot score.
3. **Spatial Clustering** — DBSCAN groups violations into **63 geographic clusters**.
4. **IPII Risk Scoring** — composite risk score (0–100) from severity, behavior, and density, with explainability.
5. **Emerging Hotspot Detection** — flags junctions trending upward using recent-vs-historical comparison.
6. **Enforcement Recommendations** — rule-based engine maps risk + trend into prioritized actions (P1–P4).
7. **Efficiency Projection** — simulated impact estimates per risk tier.

All of this feeds a **6-page dashboard**: Overview, Hotspot Analysis, Risk Intelligence, Emerging Hotspots, Enforcement Recommendations, and a Geospatial Map.

---

## Honest framing: what's genuine ML vs. engineered logic

| Component | Type |
|---|---|
| Spatial clustering (`hotspot.py`) | ✅ Genuine ML — scikit-learn DBSCAN |
| IPII Risk Score (`risk_score.py`) | ⚙️ Hardcoded weighted formula |
| Emerging Hotspot Score (`emerging.py`) | ⚙️ Engineered trend heuristic |
| Enforcement Recommendations (`recommender.py`) | ⚙️ Rule-based decision engine |
| Efficiency Projection (`efficiency.py`) | 🧪 Simulation-based estimate |

**Bottom line:** DBSCAN clustering is genuine ML. Everything downstream (risk scoring, trend detection, recommendations) is transparent, rule-based logic by design — every number is traceable to a specific, inspectable rule rather than a black box.

---

## Project structure

```
nammapark-AI/
├── dashboard/
│   ├── app.py                          # Streamlit entry point
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Hotspot_Analysis.py
│       ├── 3_Risk_Intelligence.py
│       ├── 4_Emerging_Hotspots.py
│       ├── 5_Enforcement_Recommendations.py
│       └── 6_Geospatial_Map.py
├── data/
│   └── ps1_violations.csv              # raw dataset (not committed — see Dataset section)
├── outputs/
│   ├── (all pipeline-generated CSV outputs)
│   └── final_report/
│       └── risk_explained.csv          # per-junction risk explainability
├── src/
│   ├── __init__.py
│   ├── eda.py                          # Stage 2: EDA + composite hotspot scoring
│   ├── efficiency.py                   # Stage 7: simulated impact projection
│   ├── emerging.py                     # Stage 5: emerging hotspot trend detection
│   ├── hotspot.py                      # Stage 3: DBSCAN spatial clustering
│   ├── pipeline.py                     # runs all stages in sequence
│   ├── preprocess.py                   # Stage 1: cleaning + feature engineering
│   ├── recommender.py                  # Stage 6: enforcement recommendation engine
│   └── risk_score.py                   # Stage 4: IPII risk scoring + explainability
└── requirements.txt
```

---

## Tech stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** scikit-learn (DBSCAN)
- **Visualization:** Plotly, Matplotlib
- **Mapping:** Folium, streamlit-folium

---

## Dataset

- **Source:** HackerEarth Gridlock hackathon problem statement dataset (MapmyIndia partner challenge)
- **Volume:** ~298,000 parking violation records from Bengaluru
- **Access:** [Google Drive folder](https://drive.google.com/drive/folders/134R2-nUr3OOTaMpc92PNd4ybPzlwYlGa)

To run the pipeline locally, download `ps1_violations.csv` from the link above and place it in `data/`.

---

## Setup instructions

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd nammapark-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download ps1_violations.csv from the Google Drive link above
# and place it inside the data/ folder

# 4. Run the full pipeline (preprocessing → clustering → risk scoring → recommendations)
python src/pipeline.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

The pipeline runs in order: `preprocess.py` → `eda.py` → `hotspot.py` → `risk_score.py` → `emerging.py` → `recommender.py` → `efficiency.py`, producing all CSVs in `outputs/` that the dashboard reads from.

---

## Why "Namma"?

*Namma* (ನಮ್ಮ) is Kannada for "our" — used to ground the project in local Bengaluru identity, in the spirit of "Namma Metro" and "Namma Bengaluru."

---

## License

MIT License.

---

