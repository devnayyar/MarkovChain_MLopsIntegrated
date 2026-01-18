# 📚 Complete GitHub Repository Guide

**Financial Risk Regime Prediction using Markov Chains**

> Comprehensive guide on how to use this repository, its structure, and how to contribute.

---

## 📖 Table of Contents

1. [Repository Overview](#repository-overview)
2. [Quick Start (5 minutes)](#quick-start-5-minutes)
3. [Complete Workflow](#complete-workflow)
4. [Folder Structure & Purpose](#folder-structure--purpose)
5. [Data Layers Explained](#data-layers-explained)
6. [Running the System](#running-the-system)
7. [API Documentation](#api-documentation)
8. [Dashboard Guide](#dashboard-guide)
9. [Contributing](#contributing)
10. [Troubleshooting](#troubleshooting)

---

## Repository Overview

### What This Project Does

This is a **production-grade MLOps system** for financial regime prediction using Markov Chains.

Instead of predicting a single value, it models **how financial states transition over time**.

**Key Capabilities:**
- ✅ Predict regime transitions (Low Risk → Medium Risk → High Risk)
- ✅ Calculate transition probabilities to all future states
- ✅ Detect regime changes and market stress
- ✅ Monitor model performance and drift
- ✅ Automatically retrain on new data
- ✅ Serve predictions via REST API
- ✅ Visualize with Streamlit dashboard

### Real-World Use Cases

| Use Case | Example |
|----------|---------|
| Credit Risk | Customer moves Low Risk → Medium Risk → Default |
| Portfolio Management | Detect bull market → bear market transitions |
| Macroeconomic Analysis | Interest rate regimes, inflation regimes, labor market regimes |
| Early Warning Systems | Detect when system is drifting toward crisis state |

---

## Quick Start (5 minutes)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/financial-risk-markov-mlops.git
cd financial-risk-markov-mlops
```

### 2️⃣ Install & Setup

```bash
make install
make setup
```

### 3️⃣ Train Model (If Gold Data Exists)

```bash
python run_model_show_results.py
```

**Expected Output:**
```
🚀 FINML MODEL TRAINING - USING YOUR GOLD LAYER DATA
📍 Gold Data Path: data/gold/markov_state_sequences.parquet
📊 Data Shape: (859, 12)
📈 REGIME STATES: ['HIGH', 'LOW', 'MEDIUM']
✅ Model trained successfully!
📊 Transition Matrix: [...]
📊 Stationary Distribution: [...]
```

### 4️⃣ Start Services

```bash
# Terminal 1: FastAPI
make run-api

# Terminal 2: Streamlit Dashboard
make run-dashboard
```

### 5️⃣ Access Services

- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501

### ✅ Done! You're Running the System

---

## Complete Workflow

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION                                     │
│    └─ Your gold layer data: data/gold/markov_state_sequences.parquet
│       Required columns: date, REGIME_RISK, economic indicators
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. MODEL TRAINING                                       │
│    └─ Run: python run_model_show_results.py
│       Creates: monitoring/dashboard_data.json
│       Outputs: Metrics, transition matrices, regime distribution
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. MODEL REGISTRATION (Optional)                        │
│    └─ Register model with MLflow for versioning
│       python serving/model_registry.py
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. SERVING (FastAPI)                                    │
│    └─ Run: make run-api
│       Available at: http://localhost:8000
│       Endpoints: /health, /current-regime, /predict-transition
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 5. VISUALIZATION (Streamlit)                            │
│    └─ Run: make run-dashboard
│       Available at: http://localhost:8501
│       Pages: 12 interactive pages for analysis
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 6. MONITORING & DRIFT DETECTION                         │
│    └─ Automated: Continuous model health tracking
│       Trigger: Drift detection or scheduled retraining
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 7. AUTO-RETRAINING                                      │
│    └─ Run: python retraining/retrain_pipeline.py
│       Condition: If new model > current model in metrics
│       Action: Promote to production
└─────────────────────────────────────────────────────────┘
```

---

## Folder Structure & Purpose

### Top-Level Organization

```
financial-risk-markov-mlops/
├── README.md                      # Main documentation
├── requirements.txt               # Python dependencies
├── Makefile                       # CLI shortcuts
├── .gitignore                     # Git exclusion rules
├── Dockerfile                     # Container configuration
│
├── config/                        # Configuration files (YAML)
├── data/                          # Layered data (bronze/silver/gold)
├── dashboards/                    # Streamlit UI application
├── modeling/                      # Markov chain models
├── serving/                       # FastAPI REST API
├── monitoring/                    # Drift detection & metrics
├── retraining/                    # Auto-retraining pipeline
├── tests/                         # Unit & integration tests
│
├── run_model_pipeline.py          # Training pipeline entry point
├── run_model_show_results.py      # Train & prepare results
└── validation_phase1.py           # Phase 1 validation tests
```

### Detailed Structure

#### **config/** - Configuration Management

```
config/
├── config.yaml                    # Main config (model params, thresholds)
├── monitoring_config.yaml         # Monitoring & drift settings
├── paths.yaml                     # Data paths and directories
├── regime_thresholds.yaml         # State boundary definitions
├── schema.yaml                    # Data schema validation
└── thresholds.yaml                # Alert thresholds
```

**Purpose**: Centralized configuration - change parameters without touching code

#### **data/** - Layered Data Architecture

```
data/
├── bronze/                        # RAW DATA (unchanged)
│   └── raw_indicators.csv         # Downloaded FRED data
├── silver/                        # CLEANED DATA (validated)
│   └── cleaned_indicators.parquet  # Missing values handled, outliers flagged
└── gold/                          # BUSINESS-READY DATA
    ├── markov_state_sequences.parquet  # Regime states ready for modeling
    ├── features_final.csv         # All features with indicators
    └── dashboard_data.json        # Pre-computed results
```

**Pattern**: Bronze → Silver → Gold (each layer adds business value)

#### **modeling/** - Model Implementation

```
modeling/
├── models/
│   ├── base_markov.py             # Base Markov chain class
│   ├── absorbing_markov.py        # Absorbing state variant
│   └── rolling_window_markov.py   # Non-stationary variant
├── feature_engineering.py         # 73-feature pipeline (Phase 1)
├── evaluation/                    # Model evaluation metrics
├── experiments/                   # Hyperparameter experiments
└── train_pipeline.py              # Full training pipeline
```

**Key File**: `modeling/models/base_markov.py` - Core algorithm

#### **serving/** - Production API

```
serving/
├── api/
│   ├── app.py                     # FastAPI main application
│   └── API_README.md              # API documentation
├── model_loader.py                # Load models from gold data
├── model_registry.py              # MLflow model versioning
└── experiment_tracker.py          # MLflow experiment tracking
```

**Entry Point**: `serving/api/app.py`  
**Command**: `make run-api`

#### **dashboards/** - Streamlit UI

```
dashboards/
├── app.py                         # Main Streamlit entry point
├── pages/                         # Dashboard pages (12 pages)
│   ├── home.py
│   ├── regime_timeline.py
│   ├── markov_chain.py
│   ├── alerts_drift.py
│   ├── metrics_performance.py
│   ├── model_metrics.py
│   ├── eda_analysis.py
│   ├── retraining_ab_testing.py
│   ├── markov_experiment_runner.py
│   ├── documentation.py
│   ├── settings.py
│   └── ... more pages
├── components/                    # Reusable UI components
│   ├── sidebar.py
│   ├── metrics_card.py
│   └── ... more components
└── utils/                         # Dashboard utilities
```

**Entry Point**: `dashboards/app.py`  
**Command**: `make run-dashboard`

#### **monitoring/** - Drift & Performance Tracking

```
monitoring/
├── drift_detection/
│   ├── state_drift.py             # State distribution drift
│   ├── transition_drift.py        # Transition matrix drift
│   └── concept_drift.py           # Fundamental regime shift
├── performance/
│   ├── monitoring_dashboard.py    # Performance monitoring
│   └── metrics.py                 # Metric calculations
└── ... alerts, thresholds
```

**Purpose**: Continuous monitoring in production

#### **retraining/** - Auto-Retraining Logic

```
retraining/
├── retrain_pipeline.py            # Main retraining orchestration
├── triggers/                      # What triggers retraining?
│   ├── schedule_trigger.py        # Scheduled (weekly)
│   ├── performance_trigger.py     # Performance degradation
│   └── drift_trigger.py           # Drift detection
└── ... model comparison, promotion
```

**Purpose**: Automatically update models when needed

#### **tests/** - Quality Assurance

```
tests/
├── unit/                          # Unit tests for functions
├── integration/                   # End-to-end pipeline tests
├── data_tests/                    # Data validation tests
└── ... fixtures, conftest
```

**Command**: `make test`

---

## Data Layers Explained

### Why 3 Data Layers?

```
RAW DATA              CLEANED DATA           BUSINESS DATA
(Bronze)             (Silver)               (Gold)

Downloaded           Validated              Ready for
from FRED            & Processed            Models
   ↓                    ↓                      ↓
raw_indicators.csv → cleaned_*.parquet → markov_state_sequences.parquet
```

### Bronze Layer - Raw Data

**Location**: `data/bronze/`

**What**: Exactly as downloaded from FRED API
- No modifications
- May have missing values
- May have outliers
- Point-in-time snapshots

**File Format**: CSV

**Example Columns**:
```
date,dff,t10y2y,unrate,cpi,vix
2023-01-01,4.33,0.45,3.4,2.8,18.5
2023-01-02,4.33,0.48,3.4,2.8,19.2
...
```

### Silver Layer - Cleaned Data

**Location**: `data/silver/`

**What**: Validated, cleaned, ready for feature engineering
- Missing values handled (forward-fill, interpolation, or removal)
- Outliers detected and flagged
- Schema validated
- Type checking complete

**Processing Applied**:
```python
def bronze_to_silver():
    1. Load raw CSV
    2. Parse dates
    3. Handle missing values (forward-fill or interpolate)
    4. Detect outliers (IQR method)
    5. Validate schema
    6. Save as Parquet (compressed, efficient)
```

**File Format**: Parquet (compressed, efficient)

### Gold Layer - Business-Ready Data

**Location**: `data/gold/`

**What**: Discretized states ready for Markov modeling
- Raw continuous values → Discrete regimes (LOW/MEDIUM/HIGH)
- Multi-indicator regime combinations
- State sequences in order
- All economic indicators included

**File**: `markov_state_sequences.parquet`

**Required Columns**:
```
date              - Timestamp
REGIME_RISK       - Discretized regime (LOW, MEDIUM, HIGH)
UNRATE            - Unemployment rate
FEDFUNDS          - Federal funds rate
CPI_YOY           - CPI year-over-year
T10Y2Y            - 10Y-2Y yield spread
VIX               - Volatility index
... other economic indicators
```

**Example**:
```
date                REGIME_RISK  UNRATE  FEDFUNDS  CPI_YOY  T10Y2Y  VIX
2023-01-01 00:00:00 MEDIUM       3.4     4.33      2.8      0.45    18.5
2023-01-02 00:00:00 MEDIUM       3.4     4.33      2.8      0.48    19.2
2023-01-03 00:00:00 HIGH         3.5     4.50      3.0      0.42    21.1
...
```

---

## Running the System

### Prerequisites

- **Python 3.8+**
- **Git**
- **Data**: `data/gold/markov_state_sequences.parquet` with your regime data

### Installation

#### Option 1: Using Make (Recommended)

```bash
make install
make setup
```

#### Option 2: Manual Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data/bronze data/silver data/gold
mkdir -p logs models/checkpoints output/{results,reports}
```

### Running Components

#### 🤖 Train Model

```bash
# Train on gold layer data and prepare results
python run_model_show_results.py
```

**What It Does**:
1. Loads regime states from `data/gold/markov_state_sequences.parquet`
2. Trains Markov chain model
3. Calculates transition matrices
4. Computes stationary distribution
5. Saves results to `monitoring/dashboard_data.json`

**Expected Output**:
```
🚀 FINML MODEL TRAINING - USING YOUR GOLD LAYER DATA
✅ Model trained successfully!
📊 Transition Matrix:
   [[0.75 0.25 0.00]
    [0.10 0.70 0.20]
    [0.00 0.30 0.70]]
📊 Stationary Distribution: [0.28 0.50 0.22]
📊 Log Likelihood: -125.45
```

#### 🌐 Run FastAPI Server

```bash
make run-api
```

Or manually:

```bash
uvicorn serving.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**API Documentation**: http://localhost:8000/docs

#### 📊 Run Streamlit Dashboard

```bash
make run-dashboard
```

Or manually:

```bash
streamlit run dashboards/app.py
```

**Dashboard**: http://localhost:8501

#### 🧪 Run Tests

```bash
make test              # All tests
make test-unit         # Unit tests only
make test-cov          # With coverage report
```

#### 🧹 Code Quality

```bash
make lint              # Check code style
make format            # Auto-format code
```

#### 🧾 MLflow UI (Optional)

```bash
make run-mlflow
```

**MLflow UI**: http://localhost:5000

---

## API Documentation

### Base URL

```
http://localhost:8000
```

### Interactive Documentation

```
http://localhost:8000/docs          # Swagger UI
http://localhost:8000/redoc         # ReDoc UI
```

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "ok"
}
```

#### 2. Get Current Regime

```http
GET /current-regime
```

**Response**:
```json
{
  "current_regime": "MEDIUM"
}
```

**Description**: Returns the most recent regime from gold layer data.

#### 3. Predict Next State

```http
POST /predict-transition
Content-Type: application/json

{
  "current_state": "MEDIUM",
  "steps": 1
}
```

**Response**:
```json
{
  "current_state": "MEDIUM",
  "next_state_probs": {
    "LOW": 0.1,
    "MEDIUM": 0.7,
    "HIGH": 0.2
  }
}
```

**Description**: Returns transition probabilities from current state.

**Example Requests**:

```bash
# Using curl
curl -X POST http://localhost:8000/predict-transition \
  -H "Content-Type: application/json" \
  -d '{"current_state": "MEDIUM", "steps": 1}'

# Using Python
import requests

response = requests.post(
    "http://localhost:8000/predict-transition",
    json={"current_state": "MEDIUM", "steps": 1}
)
print(response.json())
```

#### 4. Forecast Path

```http
POST /forecast-path
Content-Type: application/json

{
  "current_state": "MEDIUM",
  "steps": 6
}
```

**Response**:
```json
{
  "current_state": "MEDIUM",
  "forecast_steps": 6,
  "paths": [
    {
      "probability": 0.35,
      "states": ["MEDIUM", "MEDIUM", "HIGH", "HIGH", "MEDIUM", "LOW"]
    },
    {
      "probability": 0.28,
      "states": ["MEDIUM", "MEDIUM", "MEDIUM", "LOW", "LOW", "LOW"]
    },
    ...
  ]
}
```

**Description**: Returns multiple probable paths through regime space.

---

## Dashboard Guide

### Access

```
http://localhost:8501
```

### Pages Overview

| Page | Purpose |
|------|---------|
| **Home** | System overview, key metrics, quick status |
| **Regime Timeline** | Historical regime transitions, timeline visualization |
| **Markov Chain** | Transition matrices, state diagrams, heatmaps |
| **Alerts & Drift** | Regime changes, model drift alerts, anomalies |
| **Metrics & Performance** | Model accuracy, log-likelihood, predictions |
| **Model Metrics** | Detailed model diagnostics, state statistics |
| **EDA Analysis** | Data exploration, distributions, correlations |
| **Retraining & A/B Testing** | Model versioning, comparison, A/B test results |
| **Model Experiments** | Run custom experiments, parameter tuning |
| **Documentation** | System documentation, guides, API reference |
| **Settings** | Configuration management, threshold adjustment |
| *...more pages* | Additional analysis pages |

### Interactive Features

- **Real-time Updates**: Data refreshes automatically
- **Filters**: Filter by date range, regime, indicators
- **Downloads**: Export visualizations and data
- **Configuration**: Adjust thresholds and parameters
- **Experiments**: Run "what-if" scenarios

---

## Contributing

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Test**: `make test`
5. **Format**: `make format`
6. **Commit**: `git commit -m "Add amazing feature"`
7. **Push**: `git push origin feature/amazing-feature`
8. **Create** a Pull Request

### Code Style

- **Python**: PEP 8 (enforced with Black)
- **Type Hints**: Required for all functions
- **Docstrings**: Google-style docstrings
- **Tests**: Unit tests required

### Running Before Submit

```bash
make format      # Auto-format code
make lint        # Check for issues
make test        # Run all tests
```

### Areas for Contribution

- [ ] Additional Markov models (Hidden Markov Models, Continuous-Time Markov Chains)
- [ ] Better feature engineering
- [ ] Ensemble methods
- [ ] More drift detection methods
- [ ] Better visualizations
- [ ] Performance optimizations
- [ ] Additional documentation

---

## Troubleshooting

### Problem: "Port 8000 already in use"

```bash
# Solution 1: Use different port
uvicorn serving.api.app:app --port 8001

# Solution 2: Kill process on port 8000
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

### Problem: "Data not found - gold layer missing"

```bash
# Ensure your gold layer data exists:
data/gold/markov_state_sequences.parquet

# Must have columns:
# - date
# - REGIME_RISK (states like LOW, MEDIUM, HIGH)
# - UNRATE, FEDFUNDS, CPI_YOY, T10Y2Y, VIX (indicators)
```

### Problem: "ModuleNotFoundError"

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or reinstall fresh environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Problem: "Dashboard shows no data"

```bash
# Run model training first
python run_model_show_results.py

# This creates: monitoring/dashboard_data.json
# Which dashboard reads on startup
```

### Problem: "API returns 404"

```bash
# Check if gold data exists and is readable
python -c "import pandas as pd; print(pd.read_parquet('data/gold/markov_state_sequences.parquet').head())"

# If empty or missing, train model first:
python run_model_show_results.py
```

---

## File Reference

### Key Files for Each Component

#### Training & Model

- `run_model_show_results.py` - Main training entry point
- `modeling/models/base_markov.py` - Markov chain implementation
- `modeling/train_pipeline.py` - Full pipeline orchestration

#### Serving (API)

- `serving/api/app.py` - FastAPI application
- `serving/model_loader.py` - Load models from gold data
- `serving/api/API_README.md` - API documentation

#### Dashboard

- `dashboards/app.py` - Streamlit main app
- `dashboards/pages/` - Individual dashboard pages
- `dashboards/components/` - Reusable UI components

#### Configuration

- `config/config.yaml` - Main configuration
- `config/paths.yaml` - Data paths
- `config/thresholds.yaml` - Alert thresholds

#### Monitoring

- `monitoring/drift_detection/` - Drift detection logic
- `monitoring/performance/` - Performance tracking

---

## Common Commands Cheat Sheet

```bash
# Setup
make install              # Install dependencies
make setup                # Create directories

# Development
make format               # Format code (Black)
make lint                 # Check code style (Flake8)
make test                 # Run tests
make test-cov            # Tests with coverage

# Training
python run_model_show_results.py    # Train model

# Running Services
make run-api              # Start FastAPI (port 8000)
make run-dashboard        # Start Streamlit (port 8501)
make run-mlflow          # Start MLflow UI (port 5000)

# Cleanup
make clean                # Clean build artifacts

# Help
make help                 # Show all commands
```

---

## Project Statistics

| Metric | Value |
|--------|-------|
| **Total Python Files** | ~50+ |
| **Dashboard Pages** | 12 |
| **API Endpoints** | 4+ |
| **Test Coverage** | >80% |
| **Lines of Code** | ~3500+ |
| **Configuration Files** | 6 |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                             │
│                   (Your Gold Layer Data)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                                │
│              run_model_show_results.py                           │
│    Creates: monitoring/dashboard_data.json                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        ↓                                     ↓
   ┌─────────────┐                   ┌──────────────┐
   │  FastAPI    │                   │  Streamlit   │
   │  :8000      │                   │  :8501       │
   │             │                   │              │
   │ /health     │                   │ 12 Pages     │
   │ /current    │                   │ Analytics    │
   │ /predict    │                   │ Monitoring   │
   │ /forecast   │                   │              │
   └─────────────┘                   └──────────────┘
        ↓                                     ↓
        └──────────────────┬──────────────────┘
                           ↓
                    ┌──────────────┐
                    │ Monitoring & │
                    │ Drift Det.   │
                    └──────────────┘
                           ↓
                    ┌──────────────┐
                    │ Auto-Retrain │
                    │  Pipeline    │
                    └──────────────┘
```

---

## Support & Resources

### Documentation

- [README.md](README.md) - Main documentation
- [Makefile](Makefile) - Available commands
- [serving/api/API_README.md](serving/api/API_README.md) - API details
- [config/config.yaml](config/config.yaml) - Configuration options

### Code Quality

- Tests: `tests/` directory
- Type Hints: Throughout codebase
- Docstrings: Google-style format

### Deployment

- Docker: `Dockerfile` and `docker-compose.yml`
- CI/CD: `.github/workflows/`

---

## FAQ

### Q: Do I need FRED API key?

**A**: No! The gold layer data should already be in `data/gold/markov_state_sequences.parquet`. If you want to download fresh FRED data, you can get a free API key at https://fred.stlouisfed.org/docs/api/

### Q: Can I modify the regimes (LOW/MEDIUM/HIGH)?

**A**: Yes! Update `config/regime_thresholds.yaml` to change regime definitions.

### Q: How do I add new features?

**A**: Add columns to gold layer data and they'll automatically be included in analysis and dashboard.

### Q: Can I run this in production?

**A**: Yes! Use Docker:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Q: How do I monitor model performance?

**A**: Check the **Metrics & Performance** and **Alerts & Drift** dashboard pages.

---

## License

MIT License - See LICENSE file for details

---

## Version Info

**Current Version**: Phase 1 (v1.1.0)  
**Release Date**: January 18, 2026  
**Status**: Production Ready

---

## Contact & Support

Found a bug? Have a suggestion?

- **Open an Issue**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Pull Requests**: For contributions

---

**Happy Forecasting! 🚀**

*This project demonstrates enterprise-grade ML system design, MLOps practices, and financial modeling expertise.*
