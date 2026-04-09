# HHS Care Forecast Dashboard

<p align="center">
  Forecasting and monitoring dashboard for children in HHS care using
  <b>Persistence</b>, <b>Random Forest</b>, and <b>SARIMA</b> models.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Project-Internship%20Submission-0A7E8C">
</p>

## Overview

This project provides an interactive Streamlit dashboard to forecast:

- Daily **Children in HHS Care**
- Daily **Discharge Demand**
- **Capacity stress early-warning** based on net inflow pressure

The app compares multiple forecasting approaches and reports test-set MAE to support practical model selection.

## Key Features

- End-to-end preprocessing for daily time series data
- Feature engineering with lags, rolling averages, day-of-week, and net pressure
- Model options:
  - Persistence (naive baseline)
  - Random Forest (machine learning)
  - SARIMA with weekly seasonality
- Visual forecast bands and model comparison table
- Operational warning panel for shelter capacity pressure

## Project Structure

```text
hhs_forecast/
├── dashboard.py                               # Streamlit application
├── prjct1_data.csv                            # Original dataset
├── cleaned_hhs_data.csv                       # Cleaned dataset snapshot
└── researchReport_internship_UF_prjct1.pdf    # Internship report
```

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/NafisAslam70/UF-Internship-hhs-care-forecast.git
cd UF-Internship-hhs-care-forecast
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the dashboard (important: run from `hhs_forecast` so CSV paths resolve):

```bash
cd hhs_forecast
streamlit run dashboard.py
```

## Dashboard Sections

- Future care load forecast for the selected horizon (1 to 30 days)
- Discharge demand forecast panel
- Historical test-set model MAE comparison
- Capacity stress warning from recent net inflow pressure

## Notes

- The app currently uses file-based CSV input (`prjct1_data.csv`).
- Forward-fill is applied to maintain a complete daily timeline.
- Random Forest multi-step forecasting is iterative and approximate.

## Author

**Nafis Aslam**  
Unified Mentors Internship Submission
