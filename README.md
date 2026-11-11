# Causal Root-Cause Analysis of Yield Loss (UCI SECOM)

Causal discovery and treatment effect estimation on the UCI SECOM semiconductor manufacturing dataset, benchmarked against predictive SHAP baselines and validated on a semi-synthetic DAG generator.

> **Epistemic note:** SECOM features are anonymized (`sensor_0` .. `sensor_589`) with no ground-truth causal graph. All findings represent **data-level statistical/causal evidence** under explicit identifiability assumptions, not proven physical mechanisms.

## Project Structure

```text
├── app/                    # Streamlit interactive dashboard (Phase 7)
├── configs/                # YAML configuration files
├── data/
│   ├── raw/                # Downloaded UCI SECOM files (gitignored)
│   └── processed/          # Train/test splits and imputed datasets (gitignored)
├── notebooks/              # Exploratory & presentation notebooks
├── reports/                # Generated summaries, figures, and final report
├── src/secom_causal/       # Modular Python package
├── tests/                  # Pytest unit tests
├── pyproject.toml          # Project metadata & dependencies
├── requirements.txt        # Pinned Python dependencies
└── run_pipeline.py         # Single-command reproducible pipeline runner
```

## Setup

```bash
# Create Python 3.11+ virtual environment and install dependencies
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Run Phase 1 (download, clean, impute comparison, drift analysis, time-ordered split)
python run_pipeline.py --phase 1 --config configs/config.yaml

# Run unit tests
pytest
```

## Current Status

- **Phase 1 — Setup and Data Prep:** In progress.
- **Phases 2–7:** Pending phase checkpoints.

## Limitations

- Small failure sample size (~104 fail wafers out of 1,567, ~6.6%), yielding wide uncertainty intervals.
- Anonymized sensors with no physical domain metadata or ground-truth causal graph.
- Substantial missingness and temporal process drift across wafer timestamps.
- Standard causal discovery methods rely on strong untestable assumptions (e.g., causal sufficiency / no unmeasured confounding for PC and GES).
