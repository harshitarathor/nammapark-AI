import subprocess
import sys
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# PIPELINE ORDER
# ==========================================================

PIPELINE_STEPS = [
    "preprocess.py",
    "eda.py",
    "hotspot.py",
    "risk_score.py",
    "emerging.py",
    "recommender.py",
    "efficiency.py"
]

# ==========================================================
# RUNNER
# ==========================================================

def run_step(script_name):

    script_path = BASE_DIR / script_name

    print("\n" + "=" * 70)
    print(f"RUNNING : {script_name}")
    print("=" * 70)

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True
    )

    print("\n")
    print(f"COMPLETED : {script_name}")
    print("=" * 70)


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline():

    print("\n")
    print("=" * 70)
    print("NAMMAPARK-AI PIPELINE STARTED")
    print("=" * 70)

    for script in PIPELINE_STEPS:

        try:
            run_step(script)

        except subprocess.CalledProcessError:

            print("\n")
            print("=" * 70)
            print(f"FAILED : {script}")
            print("Pipeline stopped.")
            print("=" * 70)

            return

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated Outputs:")

    print("[OK] df_clean.csv")
    print("[OK] hotspots.csv")
    print("[OK] risk_scores.csv")
    print("[OK] emerging_hotspots.csv")
    print("[OK] enforcement_recommendations.csv")
    print("[OK] efficiency.csv")

    print("\nReady for Dashboard Visualization.")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run_pipeline()