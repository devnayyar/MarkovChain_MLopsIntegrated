# FINML: Financial Risk Markov MLOps - System Architecture

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Pipeline](#data-pipeline)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [Monitoring & Anomaly Detection](#monitoring--anomaly-detection)
7. [Model Serving & Operations](#model-serving--operations)
8. [Dashboard & User Interface](#dashboard--user-interface)
9. [Data Flow Diagrams](#data-flow-diagrams)
10. [Integration Points](#integration-points)
11. [Deployment Architecture](#deployment-architecture)
12. [Scalability Considerations](#scalability-considerations)

---

## Executive Overview

FINML is a comprehensive **Machine Learning Operations (MLOps)** platform designed to monitor and predict financial risk using **Markov Chain models**. The system operates on a multi-layered data architecture (Bronze-Silver-Gold) and provides real-time monitoring, automated retraining, A/B testing, and an interactive Streamlit dashboard.

### Key Capabilities:

- **Regime Detection**: Identifies market states (Normal, Stress, Crisis) using Markov chains
- **Drift Detection**: Monitors data quality and model performance degradation
- **Automated Retraining**: Triggers model updates based on predefined thresholds
- **A/B Testing**: Compares model versions before production deployment
- **Real-time Monitoring**: Dashboard with live alerts, metrics, and system status
- **Model Registry**: Version control and lifecycle management for trained models
- **Multi-phase Pipeline**: Raw data → Cleaned → ML-ready with validation at each stage

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FINML ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         INPUT LAYER (Data Ingestion)                      │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • External APIs • CSV Files • Databases • Streams         │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    BRONZE LAYER (Raw Data Ingestion)                     │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • Minimal transformation • Full audit trail              │   │
│  │ • Data Validation: Schema, Type, Completeness            │   │
│  │ • Quality Score: 87.3%                                   │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    SILVER LAYER (Data Cleaning & Enrichment)             │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • Deduplication • Outlier removal • Missing value handling│   │
│  │ • Feature engineering • Time-series preparation          │   │
│  │ • Regime Discretization (Low/Medium/High Risk)           │   │
│  │ • Quality Score: 94.5%                                   │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    GOLD LAYER (ML-Ready Data)                            │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • Normalized features • Regime labels • Ready for models  │   │
│  │ • Markov state vectors • Performance indicators           │   │
│  │ • Quality Score: 98.1%                                   │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    ML PIPELINE (Model Training & Evaluation)             │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • Markov Chain State Modeling                            │   │
│  │ • Transition Probability Estimation                      │   │
│  │ • Model Evaluation (Spectral Gap, Sojourn Times)         │   │
│  │ • Hyperparameter Optimization                            │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    MODEL REGISTRY (Artifact Storage & Versioning)        │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • MLflow backend (SQLite) • Model versioning             │   │
│  │ • Experiment tracking • Metrics and parameters           │   │
│  │ • Artifact storage • Rollback support                    │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    SERVING LAYER (Model Deployment)                      │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • Real-time inference • Batch predictions                │   │
│  │ • Experiment tracking integration • A/B testing          │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    ORCHESTRATION (Automated Workflows)                   │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • DAG-based pipeline execution                           │   │
│  │ • Scheduling and retry logic                            │   │
│  │ • Dependency management                                 │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    MONITORING & RETRAINING (Continuous Improvement)      │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • Data drift detection • Model performance monitoring    │   │
│  │ • Anomaly detection • Automated retraining trigger       │   │
│  │ • A/B testing framework • Rollback mechanisms            │   │
│  └──────────┬───────────────────────────────────────────────┘   │
│             │                                                      │
│  ┌──────────▼───────────────────────────────────────────────┐   │
│  │    DASHBOARD & ALERTING (User Interface)                 │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • Real-time metrics display • Regime visualization       │   │
│  │ • Performance tracking • Alert management                │   │
│  │ • A/B test results • System health monitoring            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Data Pipeline (Multi-Layer Architecture)

#### Bronze Layer (`data/bronze/`)
- **Purpose**: Store raw, unprocessed data exactly as received
- **Validation**: Schema validation, type checking, completeness checks
- **Quality**: 87.3% (baseline quality from external sources)
- **Key Files**:
  - `validate_bronze.py`: Validates incoming data structure
  - Raw CSV files or API responses

#### Silver Layer (`data/silver/`)
- **Purpose**: Cleaned, deduplicated, enriched data
- **Processing**:
  - Remove duplicates and outliers
  - Handle missing values (mean/median/forward fill)
  - Engineer time-series features
  - Normalize numerical values
  - Create regime discretization (3 states)
- **Quality**: 94.5% (cleaned and standardized)
- **Key Files**:
  - `preprocessing/cleaning.py`: Data cleaning routines
  - `preprocessing/regime_discretization.py`: Regime labeling

#### Gold Layer (`data/gold/`)
- **Purpose**: ML-ready features, prepared for models
- **Content**:
  - Normalized features
  - Regime labels (Low/Medium/High Risk)
  - Markov state vectors
  - Performance indicators
- **Quality**: 98.1% (fully prepared for ML)
- **Key Files**:
  - `validate_silver_gold.py`: Validates transition between layers

### 2. Markov Chain Model

**Mathematical Foundation:**
- Models financial regimes as discrete states with probabilistic transitions
- Estimates transition probability matrix from historical data
- Computes long-run (stationary) probabilities

**Key Metrics:**
- **Spectral Gap**: Eigenvalue of transition matrix (faster convergence = smaller gap)
- **Sojourn Time**: Average duration in each state
- **Stationary Distribution**: Long-run probability of each regime

**Regimes:**
1. **Normal**: Regular market conditions (most common)
2. **Stress**: Elevated volatility, risk increases
3. **Crisis**: Extreme market dislocations (rare but critical)

### 3. Model Evaluation Pipeline

Located in `modeling/evaluation/`:
- Accuracy metrics (MAE, RMSE for predictions)
- Markov-specific metrics (spectral gap, sojourn times)
- Drift detection (Kolmogorov-Smirnov tests)
- Data quality assessment

### 4. Serving Layer

Located in `serving/`:
- **experiment_tracker.py**: Integrates with MLflow for tracking
- Real-time inference on new data
- Batch prediction capabilities
- A/B testing support (compare candidate vs. baseline models)

### 5. Monitoring & Alerting

Located in `monitoring/`:
- **anomaly_detector.py**: Detects unusual patterns in data
- **dashboard_data.py**: Aggregates metrics for visualization
- **drift_detection/**: Monitors data and model drift
- **performance/**: Tracks model performance over time
- **alerts/**: Generates alerts for anomalies and threshold breaches

### 6. Retraining & A/B Testing

Located in `retraining/`:
- **scheduler.py**: Orchestrates retraining workflows
- **ab_testing.py**: Manages candidate model evaluation
- Automatic triggers based on performance degradation
- Rollback mechanisms when new models underperform

### 7. Model Registry & MLOps

Located in `model_registry/`:
- **SQLite Backend**: `db_backend/mlflow.db`
- **Artifact Storage**: Trained models and evaluation results
- **Experiments**: 4 baseline experiments configured:
  - `markov_chain_baseline`: Standard model
  - `markov_chain_absorbing`: Absorbing state variant
  - `markov_chain_comparison`: Comparative analysis
  - `data_sensitivity_analysis`: Parameter sensitivity testing
- **Version Control**: All model artifacts versioned with metadata

---

## Data Pipeline

### Flow: Raw Data → Bronze → Silver → Gold

```
┌─────────────────┐
│   Raw Data      │
│  (CSV, API)     │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│   Data Validation        │
│ (Schema, Types, Range)   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│    BRONZE LAYER          │
│   (Raw Storage)          │
│  Quality: 87.3%          │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│   Preprocessing          │
│ (Clean, Dedupe, Enrich)  │
├──────────────────────────┤
│ • Remove outliers        │
│ • Handle missing values  │
│ • Feature engineering    │
│ • Normalize values       │
│ • Create regime labels   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│    SILVER LAYER          │
│ (Cleaned Storage)        │
│  Quality: 94.5%          │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│   Feature Preparation    │
│ (Finalization for ML)    │
├──────────────────────────┤
│ • Normalize features     │
│ • Align timestamps       │
│ • Create Markov vectors  │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│     GOLD LAYER           │
│  (ML-Ready Storage)      │
│  Quality: 98.1%          │
└──────────────────────────┘
```

### Quality Metrics at Each Stage:

| Layer | Quality | Completeness | Uniqueness | Validity |
|-------|---------|--------------|-----------|----------|
| Bronze | 87.3% | 85% | 92% | 88% |
| Silver | 94.5% | 99% | 100% | 95% |
| Gold | 98.1% | 100% | 100% | 100% |

---

## Machine Learning Pipeline

### Markov Chain Model Training

```
Gold Layer Data
      │
      ▼
┌─────────────────────────┐
│ State Discretization    │
│ (Map to 3 regimes)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Transition Counting     │
│ (Build contingency tbl) │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Normalize Transitions   │
│ (Create P matrix)       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Spectral Analysis       │
│ (Compute eigenvalues)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Stationary Dist.        │
│ (Long-run probabilities)│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Sojourn Time Analysis   │
│ (State duration metrics)│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Model Evaluation        │
│ (Log metrics to MLflow) │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Save Trained Model      │
│ (Artifact + metadata)   │
└─────────────────────────┘
```

### Hyperparameter Tuning:
- Regularization coefficients
- State definition thresholds
- Transition probability smoothing
- Confidence interval settings

### Evaluation Metrics:

1. **Spectral Gap** (λ₂): Smaller = faster convergence to stationary
2. **Sojourn Times**: Expected duration in each state
3. **Stationary Probability**: Long-run regime distribution
4. **Transition Confidence**: Reliability of transition estimates

---

## Monitoring & Anomaly Detection

### Components:

**1. Drift Detection (`monitoring/drift_detection/`)**
- **Data Drift**: Compares current data distribution vs. training data
- **Model Drift**: Monitors prediction accuracy degradation
- **Triggers Retraining** when drift exceeds thresholds

**2. Anomaly Detection (`monitoring/anomaly_detector.py`)**
- Identifies unusual data points or regime transitions
- Flags extreme movements outside historical patterns
- Triggers alerts for investigation

**3. Performance Monitoring (`monitoring/performance/`)**
- Tracks model prediction accuracy
- Monitors prediction latency
- Records inference volume

**4. Alerts (`monitoring/alerts/`)**
- Configured severity levels (Low, Medium, High, Critical)
- Actionable alert messages
- Integration with dashboard visualization

### Monitoring Thresholds (from `config/thresholds.yaml`):

```yaml
drift_detection:
  ks_statistic_threshold: 0.15
  accuracy_degradation_threshold: 0.05
  data_quality_threshold: 0.90

anomaly_detection:
  zscore_threshold: 3.0
  isolation_forest_contamination: 0.05

retraining:
  min_accuracy_improvement: 0.02
  min_samples_required: 1000
  retraining_frequency: "weekly"
```

---

## Model Serving & Operations

### Inference Pipeline:

```
New Data → Model Loading → Preprocessing → Inference → Postprocessing → Result
                          ↓
                     Model Registry
                     (Select version)
```

### A/B Testing Framework:

1. **Baseline**: Current production model
2. **Candidate**: New model from recent training
3. **Test Period**: Deploy both, compare metrics
4. **Decision**: Winner → Production, Loser → Archive
5. **Rollback**: Revert to baseline if candidate underperforms

### Serving Modes:

- **Real-time**: Single prediction per request
- **Batch**: Process multiple samples simultaneously
- **Streaming**: Continuous predictions on data streams

---

## Dashboard & User Interface

### Built with Streamlit (Phase 11)

**Location**: `dashboards/`

**Pages:**
1. **Home**: Overview, quick stats, system status
2. **Regime Timeline**: Current market regime, historical transitions
3. **Markov Chain**: Transition probabilities, state distributions
4. **Alerts & Drift**: Alert feed, drift metrics, anomalies
5. **Metrics & Performance**: Model accuracy, precision, recall, AUC-ROC
6. **EDA Analysis**: Data quality by layer (Bronze/Silver/Gold)
7. **Retraining & A/B Tests**: Model updates, test results
8. **Settings**: Configuration options, system info

**Components:**
- **Header**: Page title, system status, quick filters
- **Sidebar**: Navigation, settings, help, system status
- **Cards**: Metrics display, status indicators
- **Charts**: Time series, distributions, heatmaps (Plotly)
- **Tables**: Detailed data views, alerts, logs

**Data Flow to Dashboard:**
```
Model Registry → Data Loader → Components → Streamlit UI
              ↓
         Cache (5 min TTL)
         Mock Data Fallback
```

---

## Data Flow Diagrams

### Complete System Flow:

```
External Sources (APIs, CSV, Databases)
              │
              ▼
      ┌───────────────┐
      │ Data Ingestion│
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │  Bronze Layer │  Raw Data
      │  (87.3% QA)   │
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │ Preprocessing │  cleaning.py
      │   Pipeline    │  regime_discretization.py
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │  Silver Layer │  Cleaned Data
      │  (94.5% QA)   │
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │    Features   │
      │  Preparation  │
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │   Gold Layer  │  ML-Ready
      │  (98.1% QA)   │
      └───────┬───────┘
              │
         ┌────┴────┐
         │          │
         ▼          ▼
    ┌─────────┐  ┌──────────┐
    │ Training│  │  Monitoring/
    │ Pipeline│  │  Drift Det.
    └────┬────┘  └────┬──────┘
         │             │
         ▼             │
    ┌─────────┐        │
    │  MLflow │        │
    │ Registry│        │
    └────┬────┘        │
         │             │
         ├─────────────┤
         │             │
         ▼             ▼
    ┌─────────────────────────┐
    │   Serving Layer         │
    │   (Inference, A/B Test) │
    └────────┬────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │   Retraining Scheduler   │
    │   (If drift > threshold) │
    └──────────────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │  Dashboard & Monitoring  │
    │  (Real-time Metrics)     │
    │  (Alerts & Visualization)│
    └──────────────────────────┘
```

### Monitoring Loop:

```
Model in Production
        │
        ▼
Monitor Metrics
├─ Accuracy
├─ Latency
├─ Prediction Drift
└─ Data Quality
        │
        ▼
    Thresholds
    Met?
    /  \
   /    \
Yes     No
│        │
▼        ▼
Alert    Trigger
Review   Retraining
         │
         ▼
    New Model
    Available
         │
         ▼
    A/B Test
    │
    ├─ Baseline vs Candidate
    │
    ▼
Compare Metrics
│
├─ Candidate Better? → Deploy to Production
│
└─ Candidate Worse? → Archive, Keep Baseline
```

---

## Integration Points

### 1. MLflow Integration
- **Tracking**: Logs metrics, parameters, artifacts
- **Registry**: Stores model versions with metadata
- **Experiments**: Organizes runs by experiment
- **Backend**: SQLite database at `model_registry/db_backend/mlflow.db`

### 2. Data Validation
- **Bronze Validation**: `data_validation/validate_bronze.py`
- **Silver-Gold Validation**: `data_validation/validate_silver_gold.py`
- **Continuous**: Runs at pipeline stages

### 3. Orchestration
- **Pipeline DAG**: `orchestration/pipeline.py`
- **Scheduling**: Configurable execution frequency
- **Retry Logic**: Automatic retries on failure

### 4. Retraining Triggers
```python
# Retraining Conditions:
if (accuracy_degradation > 5%) OR \
   (data_drift_ks_stat > 0.15) OR \
   (data_quality < 90%) OR \
   (scheduled_weekly_retraining):
    trigger_retraining()
    run_ab_test()
    if candidate_better_than_baseline:
        promote_to_production()
```

### 5. Dashboard Data Pipeline
```
MLflow Registry
    ↓
data_loader.py (with mock fallback)
    ↓
Cached Data (5 min TTL)
    ↓
Streamlit Components
    ↓
Dashboard Pages
```

---

## Deployment Architecture

### Development Environment:
- Python virtual environment (`.venv`)
- Local data storage in `data/bronze|silver|gold/`
- SQLite MLflow backend
- Streamlit development server (port 8501)

### Production Architecture:

```
┌─────────────────────────────────────────┐
│         Docker Container                │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │   Data Pipeline (Scheduled)         │ │
│ │   • Ingestion                       │ │
│ │   • Cleaning                        │ │
│ │   • Feature Engineering             │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │   Model Training (Weekly/On-demand) │ │
│ │   • Markov Chain Estimation         │ │
│ │   • Hyperparameter Tuning           │ │
│ │   • MLflow Logging                  │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │   Serving (Always-on)               │ │
│ │   • Load Model from Registry        │ │
│ │   • Real-time Inference             │ │
│ │   • A/B Testing                     │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │   Monitoring (Continuous)           │ │
│ │   • Drift Detection                 │ │
│ │   • Anomaly Detection               │ │
│ │   • Alert Generation                │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │   Dashboard (Always-on)             │ │
│ │   • Streamlit App                   │ │
│ │   • Real-time Visualization         │ │
│ │   • User Interaction                │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
        │                    │
        ▼                    ▼
    ┌────────┐          ┌────────┐
    │Database│          │Storage │
    │(SQLite)│          │(Local) │
    └────────┘          └────────┘
```

### Configuration Management:
- **config/config.yaml**: Main configuration
- **config/monitoring_config.yaml**: Monitoring thresholds
- **config/paths.yaml**: Data directory paths
- **config/regime_thresholds.yaml**: Regime definition boundaries
- **config/schema.yaml**: Data schema validation
- **config/thresholds.yaml**: Alert and retraining thresholds

---

## Scalability Considerations

### Current Limitations & Future Improvements:

**Horizontal Scaling:**
- Batch inference using distributed computing (Spark, Dask)
- Multi-node training for large datasets
- Distributed orchestration (Airflow, Prefect)

**Vertical Scaling:**
- Optimize data loading with vectorized operations
- Use GPU acceleration for matrix operations (CuPy, RAPIDS)
- Implement caching strategies at all levels

**Data Management:**
- Use cloud storage (S3, GCS) for unlimited capacity
- Implement data archival for historical data
- Partition data by time for faster retrieval

**Serving:**
- Deploy model serving (FastAPI, Triton)
- Implement request batching
- Use model quantization for faster inference

**Monitoring:**
- Scale monitoring with time-series databases (InfluxDB, Prometheus)
- Real-time alerting via message queues (Kafka, RabbitMQ)
- Distributed logging (ELK Stack, Loki)

### Technology Stack Scalability:

| Component | Current | Scalable Alternative |
|-----------|---------|---------------------|
| Data Storage | Local Filesystem | S3/Cloud Storage |
| ML Tracking | MLflow (File) | MLflow (Database) |
| Orchestration | Script-based | Airflow/Prefect |
| Serving | Embedded | FastAPI/gRPC |
| Monitoring | File-based | Time-series DB |
| Dashboard | Streamlit (Dev) | Streamlit Cloud/Dash |

---

## Summary

FINML provides a comprehensive, end-to-end ML system for financial risk monitoring. Its architecture emphasizes:

1. **Data Quality**: Multi-layer validation ensures data quality at each stage
2. **Reproducibility**: MLflow tracking and versioning for all models
3. **Robustness**: Automated retraining, A/B testing, and rollback mechanisms
4. **Observability**: Comprehensive monitoring, alerting, and visualization
5. **Maintainability**: Modular design with clear separation of concerns
6. **Extensibility**: Easy to add new components, models, or data sources

The system is production-ready for financial risk modeling and can be extended to handle larger-scale deployments with cloud infrastructure.
