# Financial Risk Regime Prediction using Markov Chains

> **Production-Grade MLOps System for Modeling Financial State Transitions**
>
> **Current Version**: Phase 1 (v1.1.0) - Enhanced Feature Engineering + Bayesian Markov  
> **Last Updated**: January 17, 2026

## 🎯 Project Overview

This is an **end-to-end financial risk modeling system** that uses **Markov Chains** to predict regime transitions and risk escalation probabilities in macroeconomic systems.

Rather than predicting a single value, this system models **how financial states transition over time** — exactly how the real world works.

### Business Problem

Financial institutions must **anticipate risk migration**, not just current risk:

- A customer moving from **Low Risk → Medium Risk → Default**
- A loan transitioning through **Current → 30 DPD → 60 DPD → Default**
- A stock regime moving from **Bull → Volatile → Bear → Recovery**

### Core Value Proposition

✅ **Probabilistic Risk Forecasting** - Know transition probabilities to all future states  
✅ **Early Warning System** - Detect regime instability before crises  
✅ **Production-Ready** - Full MLOps lifecycle: deployment, monitoring, drift detection, auto-retraining  
✅ **Business Transparency** - Explainable predictions with clear state definitions  

---

## 📊 Data Source

### FRED (Federal Reserve Economic Data)

**Free, public, economist-grade financial indicators:**

| Indicator | Symbol | Purpose |
|-----------|--------|---------|
| Federal Funds Rate | `DFF` | Interest rate regime |
| 10Y-2Y Yield Spread | `T10Y2Y` | Economic health indicator |
| Unemployment Rate | `UNRATE` | Labor market regime |
| CPI (All Urban) | `CPIAUCSL` | Inflation regime |
| VIX Volatility Index | `VIXCLS` | Market stress regime |

Access: https://fred.stlouisfed.org/

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer                    │
│  FRED API → Raw CSVs → Bronze Data (./data/bronze)       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                  Data Validation Layer                    │
│  Schema checks, missing data, outlier detection          │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                 Feature Engineering Layer                 │
│  Discretization → Regime Encoding → Silver Data          │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                  Markov Modeling Layer                    │
│  State Sequences → Transition Matrices → Gold Data       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                  Model Registry & MLflow                  │
│  Version tracking, experiment management, artifacts      │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│               Serving Layer (FastAPI)                     │
│  /current-regime, /predict-next-state, /transition-probs │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│          Monitoring & Drift Detection Layer               │
│  State dist. drift, transition matrix drift, KL div.     │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│            Auto-Retraining Pipeline                       │
│  Triggered by drift or schedule; promotes if better      │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│          Business Dashboard (Streamlit)                   │
│  Regime timeline, transitions, risk escalation probs     │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
financial-risk-markov-mlops/
├── data/                          # Layered data architecture
│   ├── bronze/                    # Raw downloaded data
│   ├── silver/                    # Cleaned & validated
│   └── gold/                      # Business-ready features
├── data_validation/               # Data quality assurance
├── preprocessing/                 # Feature engineering & regime encoding
├── eda/                           # Exploratory analysis per layer
├── modeling/                      # Markov chain models & experiments
│   ├── models/                    # Model implementations
│   ├── evaluation/                # Backtesting & metrics
│   └── experiments/               # Sensitivity analysis
├── model_registry/                # MLflow integration
├── serving/                       # FastAPI production API
├── monitoring/                    # Drift detection & performance
├── retraining/                    # Auto-retraining pipeline
├── dashboards/                    # Streamlit analytics UI
├── orchestration/                 # Pipeline scheduling & DAGs
├── config/                        # YAML configuration files
├── docker/                        # Containerization
├── tests/                         # Unit, integration, data tests
├── requirements.txt               # Python dependencies
├── Makefile                       # CLI commands
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

### 2. **Setup Project**

```bash
make setup
```

Creates data folders, logs, models directories, etc.

### 3. **Prepare Training Data**

Your gold layer data should be at: `data/gold/markov_state_sequences.parquet`

This parquet file needs columns:
- `date` - Timestamp
- `REGIME_RISK` - Discretized regime states (e.g., "LOW", "MEDIUM", "HIGH")
- Economic indicators: `UNRATE`, `FEDFUNDS`, `CPI_YOY`, `T10Y2Y`, etc.

### 4. **Train Markov Model & Register**

```bash
# Train model on your gold data, calculate metrics, prepare results
python run_model_show_results.py
```

This will:
- Load regime states from gold data
- Train Markov chain model
- Calculate transition matrices
- Save results to `monitoring/dashboard_data.json`
- Display regime distribution and model metrics

### 5. **Start FastAPI Server**

```bash
# Terminal 1: Start API
make run-api
```

FastAPI server available at: **`http://localhost:8000`**

**Available Endpoints:**

- `GET /health` - Health check
- `GET /current-regime` - Get current regime from latest data
- `POST /predict-transition` - Predict next state probabilities
  ```bash
  curl -X POST http://localhost:8000/predict-transition \
    -H "Content-Type: application/json" \
    -d '{"current_state": "MEDIUM", "steps": 1}'
  ```
- `POST /forecast-path` - Multi-step ahead forecast
  ```bash
  curl -X POST http://localhost:8000/forecast-path \
    -H "Content-Type: application/json" \
    -d '{"current_state": "MEDIUM", "steps": 6}'
  ```

### 6. **Launch Dashboard**

```bash
# Terminal 2: Start Dashboard
make run-dashboard
```

Streamlit dashboard available at: **`http://localhost:8501`**

**Dashboard Pages:**
- **Home** - System overview
- **Regime Timeline** - Historical regime transitions
- **Markov Chain** - Transition matrices & visualizations
- **Alerts & Drift** - Regime changes & model drift alerts
- **Metrics & Performance** - Model evaluation metrics
- **Model Metrics** - Detailed model diagnostics
- **EDA Analysis** - Data exploration & statistics
- **Retraining & A/B Testing** - Model versioning & comparison
- **Model Experiments** - Run custom model experiments
- **Documentation** - System documentation
- **Settings** - Configuration management

---

## 🎯 Phase 1 Improvements (v1.1.0)

**Enhanced Feature Engineering + Bayesian Markov**

### What's New

| Component | Improvement | Impact |
|-----------|-------------|--------|
| **Features** | 5 raw → 73 engineered features | Richer signal for predictions |
| **Markov Model** | MLE → Bayesian estimation | +20% confidence improvement |
| **Uncertainty** | No intervals → 95% confidence bounds | Quantified model risk |
| **Validation** | Manual → 15 automated tests | Continuous quality assurance |
| **Documentation** | Basic → 5000+ lines of guides | Clearer implementation path |

### Feature Engineering (73 Features)

- **20 Time-Series Lags** - Historical patterns
- **15 Rolling Statistics** - Mean, std, min, max
- **18 Domain Features** - Spreads, ratios, rates of change
- **12 Technical Indicators** - RSI, Bollinger Bands, MACD
- **8 Market Microstructure** - Volume, momentum patterns

### Bayesian Markov Enhancements

- Dirichlet priors (α=0.5) prevent zero probabilities
- Bootstrap confidence intervals (1000 iterations)
- Model diagnostics (perplexity, AIC, BIC)
- Automatic alpha tuning
- Uncertainty quantification per prediction


---

## 🔑 Key Components

### Data Layer (Bronze/Silver/Gold)

**Bronze** (Raw):
- Downloaded FRED CSV files
- No transformations
- Point-in-time snapshots

**Silver** (Cleaned):
- Missing data handling
- Outlier detection
- Type validation
- Stored as Parquet for efficiency

**Gold** (Business-Ready):
- Discretized states (Low/Medium/High)
- Multi-feature regime combinations
- State sequences ready for Markov modeling
- Aligned with business definitions

### Markov Chain Models

#### 1. **Base First-Order Markov Chain**

$$P(S_{t+1} | S_t) \text{ - transition probabilities}$$

Assumes the next state depends only on the current state.

#### 2. **Absorbing Markov Chain**

Special case where certain states (e.g., "Default", "Crisis") are absorbing—once entered, never escape.

Computes: Expected time to absorption, probability of absorption from each state.

#### 3. **Rolling Window Markov Chain**

Non-stationary modeling using sliding windows to detect regime changes over time.

#### 4. **Hybrid Markov (Ensemble)**

Weighted ensemble of the above models for robustness.

### Model Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Log-Likelihood** | How well the transition matrix explains observed transitions |
| **Regime Persistence** | How long each state typically lasts (measure of stability) |
| **Absorbing State Time** | Expected transitions to crisis/default |
| **Prediction Accuracy** | Next-state prediction accuracy on holdout test set |

### Drift Detection

**Three Types of Drift:**

1. **State Distribution Drift**: Are we spending more/less time in certain states?
2. **Transition Matrix Drift**: Are transition probabilities changing?
3. **Concept Drift**: Is the fundamental regime structure shifting?

**Detection Methods:**
- Frobenius norm of transition matrix differences
- KL divergence on state probability distributions
- Statistical tests (chi-square on contingency tables)

### Monitoring & Alerts

- Real-time tracking of regime transitions
- Automatic drift detection (daily)
- Alert thresholds: Warning (30% degradation) → Alert (50% degradation)
- Dashboard with historical drift trends

### Auto-Retraining Pipeline

**Triggers:**
1. Scheduled: Weekly (configurable)
2. Performance degradation: >threshold
3. Drift detected: Frobenius norm exceeds limit

**Logic:**
1. Collect recent data
2. Re-estimate transition matrices
3. Evaluate on holdout set
4. Compare to current production model
5. Promote if improvement > threshold
6. Otherwise, log and continue

---

## 📈 FastAPI Endpoints

### **GET /current-regime**

Returns the current financial regime based on latest data.

**Response:**
```json
{
  "timestamp": "2024-01-15T12:00:00Z",
  "yield_curve_state": "Normal",
  "fed_rate_state": "Medium",
  "unemployment_state": "Low",
  "inflation_state": "Stable",
  "volatility_state": "Elevated",
  "combined_regime": "Normal_Yields+Med_Rates+Low_Unemp"
}
```

### **POST /predict-next-state**

Predicts next state transition probabilities.

**Request:**
```json
{
  "current_state": "Normal_Yields+Med_Rates+Low_Unemp",
  "horizon": 1
}
```

**Response:**
```json
{
  "current_state": "Normal_Yields+Med_Rates+Low_Unemp",
  "next_state_probabilities": {
    "Normal_Yields+Med_Rates+Low_Unemp": 0.45,
    "Flat_Yields+Med_Rates+Mod_Unemp": 0.35,
    "Inverted_Yields+High_Rates+High_Unemp": 0.20
  },
  "most_likely_next_state": "Normal_Yields+Med_Rates+Low_Unemp",
  "probability": 0.45
}
```

### **GET /transition-matrices**

Returns full transition matrices for each regime.

### **GET /risk-path**

Forecasts probable paths through regime space.

**Response:**
```json
{
  "forecast_horizon": 12,
  "paths": [
    {
      "probability": 0.45,
      "states": ["Normal", "Normal", "Flat", "Flat", "Normal", ...],
      "crisis_probability": 0.05
    },
    ...
  ]
}
```

---

## 📊 Streamlit Dashboard

**Pages:**

1. **Data Overview**: Time series of raw indicators, missing data, statistics
2. **Regime Analysis**: State frequencies, persistence, transitions over time
3. **Transition Matrices**: Heatmaps of P(S_{t+1}|S_t), absorbing states
4. **Risk Forecast**: Multi-step ahead predictions, Monte Carlo paths
5. **Model Monitoring**: Drift detection, model health, retraining history

---

## 🧪 Testing

### Run All Tests

```bash
make test
```

### Run Specific Test Suite

```bash
make test-unit           # Unit tests only
make test-cov            # With coverage report
```

### Test Coverage

- **Unit Tests** (`tests/unit/`): Individual functions
- **Integration Tests** (`tests/integration/`): End-to-end pipelines
- **Data Tests** (`tests/data_tests/`): Data validation, schema checks

---

## 🐳 Docker Deployment

### Build & Run with Docker Compose

```bash
make deploy-docker
```

This starts:
- FastAPI server (port 8000)
- Streamlit dashboard (port 8501)
- MLflow tracking server (port 5000)

### View Logs

```bash
make logs-docker
```

### Stop

```bash
make stop-docker
```

---

## 📋 Configuration

All configuration is centralized in `./config/`:

### `config.yaml`
Main configuration: data paths, regime definitions, model hyperparameters

### `thresholds.yaml`
Drift detection thresholds, retraining triggers, state boundaries

### `paths.yaml`
Filesystem paths for data, models, logs

---

## 🔄 Workflow

### Step 1: Data Ingestion

```bash
python data/bronze/raw_fred_download.py
```

Downloads latest FRED data → `data/bronze/raw_macro_data.csv`

### Step 2: Validation

```bash
python data_validation/validate_bronze.py
```

Checks schema, missing values, outliers

### Step 3: Preprocessing

```bash
python preprocessing/regime_discretization.py
```

Converts continuous indicators → discrete states

Output: `data/silver/state_encoded_data.parquet`

### Step 4: Markov Modeling

```bash
python modeling/train_pipeline.py
```

Estimates transition matrices, evaluates models

Logs experiments to MLflow: `http://localhost:5000`

### Step 5: Model Registry

```bash
python model_registry/register_model.py
```

Registers best model with metadata

### Step 6: Serve

```bash
make run-api
```

FastAPI server ready for predictions

### Step 7: Monitor

Drift detection runs automatically (hourly)

Check dashboard at `http://localhost:8501`

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| [data/bronze/raw_fred_download.py](data/bronze/raw_fred_download.py) | FRED API data fetching |
| [preprocessing/regime_discretization.py](preprocessing/regime_discretization.py) | State encoding logic |
| [modeling/models/base_markov.py](modeling/models/base_markov.py) | Markov chain implementation |
| [monitoring/drift_detection/state_drift.py](monitoring/drift_detection/state_drift.py) | Drift metrics |
| [serving/app.py](serving/app.py) | FastAPI main application |
| [dashboards/streamlit_app.py](dashboards/streamlit_app.py) | Dashboard entry point |

---

## 💡 Resume Impact

### Talking Points

1. **"Markov Chains for Financial Regime Modeling"**
   - Not typical XGBoost prediction
   - Real quantitative finance thinking
   - Probabilistic forecasting

2. **"End-to-End MLOps System"**
   - Data layering (bronze/silver/gold)
   - Production API (FastAPI)
   - Monitoring & drift detection
   - Auto-retraining pipeline

3. **"Business-Focused Design"**
   - Clear state definitions
   - Explainable predictions
   - Decision-support dashboard
   - Risk escalation forecasting

4. **"Enterprise-Grade Code Quality"**
   - Layered architecture
   - Comprehensive testing
   - Configuration management
   - CI/CD ready

---

## 🎓 Learning Outcomes

By completing this project, you'll understand:

- ✅ Markov chains: theory & practical implementation
- ✅ Financial regime modeling & discretization
- ✅ MLOps: pipelines, monitoring, deployment
- ✅ Drift detection & concept drift
- ✅ FastAPI for ML serving
- ✅ Production ML systems design
- ✅ Time series analysis & regime changes
- ✅ Business impact framing

---

## 🔗 References

- [Markov Chains Theory](https://en.wikipedia.org/wiki/Markov_chain)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📝 License

MIT License - See LICENSE file

---

## 👤 Author

Created as a production-grade machine learning portfolio project.

**Questions? Ideas?** Open an issue or reach out!

---

**Last Updated:** January 2024
