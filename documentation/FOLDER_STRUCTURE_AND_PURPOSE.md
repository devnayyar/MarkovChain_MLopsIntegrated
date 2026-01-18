# FINML: Folder Structure & Purpose Guide

## Complete Directory Mapping

This document explains every folder and Python file in the FINML project, their purposes, how they interact, and their significance to the ML pipeline.

---

## 📁 Root Level Structure

```
financial-risk-markov-mlops/
├── data/                          # Data storage (Bronze-Silver-Gold layers)
├── dashboards/                    # Streamlit dashboard (Phase 11)
├── data_validation/               # Schema and data validation
├── eda/                          # Exploratory data analysis
├── config/                       # Configuration files (YAML)
├── logs/                         # Application logs
├── model_registry/               # MLflow artifacts and metadata
├── modeling/                     # ML training and evaluation
├── monitoring/                   # Real-time monitoring and alerts
├── orchestration/                # Pipeline orchestration
├── preprocessing/                # Data cleaning and feature engineering
├── retraining/                   # Automated retraining and A/B testing
├── serving/                      # Model serving and inference
├── tests/                        # Test suite
├── utils/                        # Shared utilities
├── ci_cd/                        # CI/CD configuration
├── docker/                       # Docker configuration
├── requirements.txt              # Python dependencies
├── conftest.py                   # pytest configuration
└── README.md                     # Project documentation
```

---

## 📊 Detailed Folder Breakdown

### 1. `data/` - Multi-Layer Data Storage

**Purpose**: Implements Bronze-Silver-Gold data architecture for quality progression.

```
data/
├── bronze/                       # Raw, unprocessed data
│   ├── market_data.csv          # Raw market indicators
│   ├── risk_indicators.csv      # Raw risk metrics
│   └── *.csv                    # Other raw data files
│
├── silver/                       # Cleaned, deduplicated data
│   ├── market_data_cleaned.csv  # Processed market data
│   ├── risk_indicators_clean.csv# Processed risk data
│   └── *.csv                    # Enriched data files
│
└── gold/                         # ML-ready features
    ├── features_final.csv       # Ready-to-train features
    ├── regimes.csv              # Regime labels (Low/Med/High)
    ├── markov_states.csv        # Markov state vectors
    └── metadata.json            # Feature engineering metadata
```

**Coupling & Significance:**
- **Bronze → Silver**: Cleaned by `preprocessing/cleaning.py`
- **Silver → Gold**: Enriched by `preprocessing/regime_discretization.py`
- **Validation**: Validated by `data_validation/validate_*.py` at each stage
- **Quality Scores**: Bronze (87.3%) → Silver (94.5%) → Gold (98.1%)

**Data Flow Example:**
```
Bronze raw_market_data.csv
    ↓ (cleanup: remove duplicates, handle NaN)
Silver market_data_cleaned.csv
    ↓ (feature engineering: normalize, discretize regimes)
Gold features_final.csv + regimes.csv
    ↓ (ready for training)
Markov Chain Model
```

---

### 2. `dashboards/` - Streamlit Dashboard (Phase 11)

**Purpose**: Interactive web UI for monitoring, visualization, and analysis.

```
dashboards/
├── app.py                        # Main Streamlit entry point
│   └── Routes pages based on session state
│   └── Custom CSS styling
│   └── Header with system status
│
├── pages/                        # Dashboard pages
│   ├── home.py                  # Overview, quick stats, education
│   ├── regime_timeline.py       # Current regime, historical view
│   ├── markov_chain.py          # Transition matrix, probabilities
│   ├── alerts_drift.py          # Alerts, drift detection metrics
│   ├── metrics_performance.py   # Accuracy, precision, recall, AUC
│   ├── eda_analysis.py          # Data quality per layer (Bronze/Silver/Gold)
│   ├── retraining_ab_testing.py # Retraining status, A/B test results
│   └── settings.py              # Configuration, system info
│
├── components/                  # Reusable UI components
│   ├── header.py                # Page header, status display
│   │   └── render_header(): Shows title, system status
│   │   └── render_quick_stats(): 4 key metrics display
│   │   └── render_filters(): Timeframe, date range, auto-refresh
│   │
│   ├── sidebar.py               # Navigation & settings
│   │   └── render_sidebar(): Full navigation, settings, help, system status
│   │   └── Radio buttons for page selection
│   │   └── Theme/refresh settings
│   │   └── Help buttons with expandable content
│   │
│   ├── metrics_card.py          # Metric display components
│   │   └── metric_card(): Single styled metric
│   │   └── metric_row(): Multiple metrics in row
│   │   └── gauge_metric(): Gauge-style indicator
│   │   └── alert_card(): Alert with severity color
│   │
│   ├── status_indicator.py      # Status display components
│   │   └── status_badge(): Single status indicator
│   │   └── status_grid(): Grid of status items
│   │   └── health_indicator(): Overall health score
│   │
│   ├── tooltips.py              # Educational content
│   │   └── regime_explanation(): Explains market regimes
│   │   └── performance_metric_help(): Describes ML metrics
│   │   └── data_quality_help(): Data layer explanation
│   │   └── alert_severity_help(): Alert levels guide
│   │   └── show_glossary(): Term definitions
│   │   └── GLOSSARY: Dict of terms → definitions
│   │
│   └── navigation.py            # Navigation utilities
│       └── render_back_button(): Back/home button
│       └── render_page_nav_header(): Page-specific header
│
└── utils/                       # Dashboard utilities
    ├── data_loader.py           # Data retrieval & mock fallback
    │   └── _generate_mock_markov_data(): 500-hour regime sequence
    │   └── get_markov_state_data(): Regime states
    │   └── get_performance_metrics(): Model accuracy metrics
    │   └── get_alerts(): System alerts
    │   └── get_anomalies(): Detected anomalies
    │   └── get_degradation_events(): Model performance issues
    │   └── get_retraining_jobs(): Retraining history
    │   └── get_ab_tests(): A/B test results
    │   └── get_rollback_events(): Model rollbacks
    │   └── get_*_layer_eda(): Data quality by layer
    │   └── get_markov_transition_matrix(): Transition probabilities
    │   └── get_markov_chain_stats(): Spectral gap, sojourn times
    │
    ├── constants.py             # Dashboard constants
    │   └── PAGES: Dictionary mapping page labels to routes
    │   └── REGIME_COLORS: Color mapping for regimes
    │   └── STATUS_COLORS: Color for status indicators
    │   └── ALERT_SEVERITY_COLORS: Alert level colors
    │   └── Thresholds for drift/accuracy/quality
    │
    ├── formatters.py            # Format utilities
    │   └── format_number(): Number formatting
    │   └── format_percentage(): Percentage display
    │   └── format_datetime(): Date/time formatting
    │
    └── validators.py            # Data validation
        └── validate_data(): Check data structure
        └── validate_metrics(): Verify metric ranges
```

**Key Interactions:**
- **data_loader.py** → Fetches from model_registry or generates mock data
- **components/** → Reused across all pages
- **pages/** → Rendered based on `st.session_state.current_page`
- **utils/** → Provide data and formatting

**User Flow:**
1. User visits dashboard
2. Sidebar navigation selection updates session state
3. app.py routes to selected page
4. Page calls data_loader functions
5. Components render with formatted data

---

### 3. `data_validation/` - Data Quality Assurance

**Purpose**: Validates data at each layer (Bronze, Silver, Gold).

```
data_validation/
├── __init__.py                  # Package initialization
├── schema.py                    # Schema definitions
│   └── Define expected columns, types, ranges
│   └── Classes: BronzeSchema, SilverSchema, GoldSchema
│
├── validate_bronze.py           # Bronze layer validation
│   └── check_required_columns(): Verify structure
│   └── check_data_types(): Verify types match schema
│   └── check_completeness(): Percentage of non-null values
│   └── check_duplicates(): Identify duplicate rows
│   └── Quality score calculation (target: >85%)
│
└── validate_silver_gold.py      # Silver & Gold validation
    └── Validate after cleaning/enrichment
    └── Check feature ranges (normalized: 0-1 or -1 to 1)
    └── Verify regime discretization (3 states only)
    └── Check temporal ordering (no future dates)
    └── Quality score calculation (target: >94% Silver, >98% Gold)
```

**Data Flow Integration:**
```
Bronze data (raw) 
    ↓ validate_bronze.py (87.3% quality check)
    ↓
Silver data (cleaned)
    ↓ validate_silver_gold.py (94.5% quality check)
    ↓
Gold data (ML-ready)
    ↓ validate_silver_gold.py (98.1% quality check)
    ↓ Pass validation → Training
    ↓ Fail validation → Alert + Manual review
```

**Coupling:**
- Called by `preprocessing/cleaning.py` before moving to next layer
- Results logged to `logs/` for audit trail
- Failures trigger reprocessing or alerts

---

### 4. `eda/` - Exploratory Data Analysis

**Purpose**: Analyze data characteristics at each layer.

```
eda/
├── __init__.py                  # Package initialization
├── bronze_analysis/             # Raw data analysis
│   ├── data_overview.py         # Row counts, columns, data types
│   ├── missing_patterns.py      # Missing value analysis
│   ├── outlier_detection.py     # Statistical outliers (IQR, Z-score)
│   └── correlation_analysis.py  # Feature correlations
│
├── silver_analysis/             # Cleaned data analysis
│   ├── distribution_analysis.py # Histograms, CDFs
│   ├── time_series_analysis.py  # Temporal patterns, trends
│   ├── seasonality.py           # Seasonal decomposition
│   └── stationarity_tests.py    # ADF test for time series
│
└── gold_analysis/               # ML-ready data analysis
    ├── feature_statistics.py    # Mean, std, min, max per feature
    ├── regime_distribution.py   # Count of each regime state
    ├── markov_transitions.py    # Observed transition frequencies
    └── model_input_validation.py# Check model readiness
```

**Significance:**
- Identifies data issues before training
- Guides feature engineering decisions
- Documents data characteristics for reproducibility
- Informs regime boundary selections

**Usage:**
```python
# In preprocessing or monitoring:
from eda.bronze_analysis import outlier_detection
outliers = outlier_detection.find_zscore_outliers(df, threshold=3.0)
# Result → Logged to logs/, used to guide cleaning

# In dashboard:
# EDA analysis page shows quality metrics per layer
```

---

### 5. `config/` - Configuration Management

**Purpose**: Centralized configuration for all components.

```
config/
├── config.yaml                  # Main configuration
│   ├── paths: data/model/log directory paths
│   ├── logging: log level, format
│   ├── mlflow: tracking URI, backend store
│   └── pipeline: scheduling, batch sizes
│
├── monitoring_config.yaml       # Monitoring thresholds
│   ├── drift_detection: KS statistic threshold, accuracy drop %
│   ├── anomaly_detection: Z-score, isolation forest contamination
│   └── retraining: min accuracy improvement, frequency
│
├── paths.yaml                   # Directory paths
│   ├── data_bronze, data_silver, data_gold
│   ├── model_registry, logs, artifacts
│
├── regime_thresholds.yaml       # Regime definitions
│   ├── low_risk: range thresholds
│   ├── medium_risk: range thresholds
│   └── high_risk: range thresholds
│
├── schema.yaml                  # Data schema definitions
│   ├── bronze_columns: name, type, nullable
│   ├── silver_columns: name, type, range
│   └── gold_columns: name, type, requirements
│
└── thresholds.yaml              # Alert thresholds
    ├── accuracy_minimum: acceptable accuracy
    ├── latency_maximum: max inference time
    ├── data_quality_minimum: threshold for retraining
    └── alert_levels: low/medium/high/critical
```

**Coupling:**
- Loaded at startup by all modules
- Environment-specific overrides via env vars
- Changes trigger pipeline reconfiguration

**Example Usage:**
```python
import yaml
with open('config/thresholds.yaml') as f:
    config = yaml.safe_load(f)

if accuracy_drop > config['drift_detection']['accuracy_degradation_threshold']:
    trigger_retraining()  # From retraining/scheduler.py
```

---

### 6. `preprocessing/` - Data Cleaning & Feature Engineering

**Purpose**: Transform Bronze → Silver → Gold data layers.

```
preprocessing/
├── __init__.py                  # Package initialization
├── cleaning.py                  # Data cleaning
│   ├── remove_duplicates(): Deduplication
│   ├── handle_missing_values(): Mean/median/forward fill
│   ├── handle_outliers(): Remove or cap outliers
│   ├── normalize_features(): Scale to [0,1] or [-1,1]
│   ├── deduplicate_rows(): Remove exact duplicates
│   └── Validation: Calls validate_silver_gold.py
│
└── regime_discretization.py     # Regime labeling
    ├── discretize_risk_levels(): Map continuous → 3 regimes
    ├── Thresholds from config/regime_thresholds.yaml
    ├── Output: New 'regime' column (Low/Medium/High)
    ├── Create Markov state vectors
    └── Validation: Check 3 states present, no missing
```

**Data Transformation Example:**

```
Input (Silver data):
    risk_score: [0.15, 0.52, 0.89, 0.23, ...]
    
discretize_risk_levels():
    if risk_score < 0.33: regime = "Low"
    elif risk_score < 0.67: regime = "Medium"
    else: regime = "High"
    
Output (Gold data):
    risk_score: [0.15, 0.52, 0.89, 0.23, ...]
    regime: ["Low", "Medium", "High", "Low", ...]
    markov_state: [0, 1, 2, 0, ...]  # For training
```

**Coupling:**
- **Input**: Bronze data from `data/bronze/`
- **Validation**: Checks against `data_validation/schema.py`
- **Output**: Silver (`data/silver/`), then Gold (`data/gold/`)
- **Called by**: `orchestration/pipeline.py` as scheduled task
- **Logs**: Results to `logs/preprocessing.log`

---

### 7. `modeling/` - ML Model Training & Evaluation

**Purpose**: Markov chain model training, evaluation, and feature analysis.

```
modeling/
├── __init__.py                  # Package initialization
├── models/                      # Model implementations
│   ├── markov_chain.py          # Markov model class
│   │   ├── __init__(): Initialize with states
│   │   ├── fit(): Estimate transition matrix from data
│   │   ├── predict(): Next state prediction
│   │   ├── get_stationary_dist(): Long-run probabilities
│   │   ├── get_spectral_gap(): Eigenvalue (convergence speed)
│   │   └── get_sojourn_times(): Expected state duration
│   │
│   └── variants/                # Model variations
│       ├── absorbing_states.py  # Special state handling
│       └── enhanced_markov.py   # Extended models
│
├── evaluation/                  # Model evaluation
│   ├── metrics.py               # Evaluation metrics
│   │   ├── accuracy_score(): Prediction accuracy
│   │   ├── precision_recall(): Per-regime metrics
│   │   ├── roc_auc(): Area under ROC curve
│   │   ├── spectral_gap(): Transition matrix eigenvalue
│   │   └── sojourn_times(): State duration stats
│   │
│   ├── comparison.py            # Model comparison
│   │   └── compare_models(): Side-by-side evaluation
│   │
│   └── cross_validation.py      # Cross-validation
│       └── time_series_cv(): Time-based CV (no data leakage)
│
├── experiments/                 # Experiment management
│   ├── baseline_experiment.py   # Standard model training
│   ├── sensitivity_analysis.py  # Parameter sensitivity
│   └── ablation_study.py        # Component importance
│
└── feature_analysis/            # Feature importance
    ├── regime_impact.py         # How regimes affect predictions
    ├── transition_analysis.py   # Which transitions are common
    └── stability_metrics.py     # Model stability over time
```

**Training Pipeline:**

```
Gold data (features + regimes)
    ↓
markov_chain.py:
  - Count regime transitions
  - Build transition matrix P
  - Normalize rows (probabilities sum to 1)
    ↓
evaluation/metrics.py:
  - Calculate accuracy on test set
  - Compute spectral gap
  - Compute sojourn times
    ↓
Evaluation results → MLflow logging
    ↓
model_registry/artifacts/
  - Trained model (pickle)
  - Transition matrix (CSV)
  - Metrics (JSON)
```

**Coupling:**
- **Input**: Gold data from `data/gold/`
- **Orchestration**: Called by `orchestration/pipeline.py`
- **Tracking**: Logs to MLflow via `serving/experiment_tracker.py`
- **Registry**: Saves artifacts to `model_registry/mlflow/`
- **Retraining**: Triggered by `retraining/scheduler.py`

---

### 8. `monitoring/` - Real-time Monitoring & Alerting

**Purpose**: Track system health, data quality, model performance.

```
monitoring/
├── __init__.py                  # Package initialization
├── anomaly_detector.py          # Anomaly detection
│   ├── detect_zscore(): Z-score based detection
│   ├── detect_isolation_forest(): Isolation forest
│   ├── detect_regime_transitions(): Unusual transitions
│   └── Severity levels: Low/Medium/High/Critical
│
├── dashboard_data.py            # Dashboard data aggregation
│   ├── get_current_metrics(): Latest performance metrics
│   ├── get_alert_summary(): Recent alerts
│   ├── get_system_health(): Overall health score
│   └── Cache with 5-minute TTL
│
├── scheduled_jobs.py            # Scheduled monitoring tasks
│   ├── run_drift_check(): Data/model drift
│   ├── run_quality_check(): Data quality assessment
│   ├── generate_alerts(): Create alert messages
│   └── Log results to logs/monitoring.log
│
├── drift_detection/             # Drift detection
│   ├── data_drift.py            # Input data drift
│   │   ├── kolmogorov_smirnov_test(): Distribution comparison
│   │   ├── compare_distributions(): Train vs current
│   │   └── Triggers retraining if drift > threshold
│   │
│   ├── model_drift.py           # Model performance drift
│   │   ├── accuracy_degradation(): Accuracy drop %
│   │   ├── prediction_shift(): Distribution shift in predictions
│   │   └── Triggers retraining if degradation > threshold
│   │
│   └── covariate_shift.py       # Feature distribution changes
│       └── detect_covariate_shift(): Feature drift detection
│
├── performance/                 # Model performance monitoring
│   ├── metrics_tracker.py       # Track accuracy, latency, volume
│   ├── time_series_metrics.py   # Metrics over time
│   └── sla_monitoring.py        # SLA compliance checks
│
└── alerts/                      # Alert management
    ├── alert_generator.py       # Create alert messages
    ├── alert_formatter.py       # Format for dashboard/email
    ├── alert_routing.py         # Send to appropriate destinations
    └── stored in: model_registry/rollback_events.jsonl
```

**Monitoring Loop:**

```
Every 1 hour:
    ↓
1. Collect latest predictions/data
2. run_drift_check() → KS statistic vs training data
3. run_quality_check() → Data quality % per layer
4. Compare metrics vs config/thresholds.yaml
5. If thresholds breached:
    - Generate alert → anomaly_detector.py
    - Log to logs/monitoring.log
    - Store in model_registry/
    - Trigger dashboard update
    - If severe: trigger_retraining()
```

**Coupling:**
- **Data Input**: Latest predictions + input data
- **Configuration**: Thresholds from `config/monitoring_config.yaml`
- **Output**: Alerts → Dashboard via data_loader.py
- **Triggering**: Retraining via `retraining/scheduler.py`
- **Logging**: `logs/monitoring.log` + `model_registry/` JSON files

---

### 9. `retraining/` - Automated Model Updates

**Purpose**: Trigger, execute, and validate model retraining with A/B testing.

```
retraining/
├── __init__.py                  # Package initialization
├── scheduler.py                 # Retraining orchestration
│   ├── check_retraining_conditions(): Should we retrain?
│   │   - Is it scheduled time? (weekly)
│   │   - Is accuracy degraded? (>5% drop)
│   │   - Is data drift detected? (KS stat > 0.15)
│   │   - Is quality too low? (<90%)
│   │
│   ├── trigger_retraining(): Start retraining workflow
│   │   - Load latest gold data
│   │   - Train new model
│   │   - Log to MLflow
│   │   - Save as "candidate"
│   │
│   └── schedule_retraining_jobs(): Cron-based scheduling
│       └── stored in: model_registry/retraining_jobs.jsonl
│
└── ab_testing.py                # A/B testing framework
    ├── prepare_ab_test(): Setup comparison
    │   - Baseline: current production model
    │   - Candidate: newly trained model
    │   - Test period: usually 1 week
    │
    ├── run_ab_test(): Execute test
    │   - Route % of traffic to candidate
    │   - Collect metrics for both models
    │   - Log results
    │
    ├── compare_models(): Decide winner
    │   - Candidate accuracy > baseline? → Deploy
    │   - Candidate accuracy ≤ baseline? → Archive
    │   - Metrics comparison → Dashboard
    │
    └── record_ab_test_results(): Log outcomes
        └── stored in: model_registry/retraining_jobs.jsonl
```

**Retraining Decision Flow:**

```
Monitoring alert or scheduled time
    ↓
scheduler.py:check_retraining_conditions()
    /  \
   /    \
Retrain?  No → Wait until next check
Yes       
  ↓
Load gold data
  ↓
Train new candidate model
  ↓
ab_testing.py:prepare_ab_test()
  - Baseline: production model
  - Candidate: new model
  ↓
Run A/B test (1 week)
  ↓
ab_testing.py:compare_models()
    /        \
   /          \
Candidate    Baseline
Better      Better
  ↓            ↓
Deploy      Rollback to
Candidate   Baseline
  ↓            ↓
Update       Keep
Production   Current
```

**Coupling:**
- **Trigger**: Monitoring alerts from `monitoring/`
- **Training**: Uses `modeling/` pipeline
- **MLflow**: Logs both models via `serving/experiment_tracker.py`
- **Storage**: Results to `model_registry/retraining_jobs.jsonl`
- **Dashboard**: A/B test results displayed in dashboard
- **Serving**: Winner loaded by `serving/experiment_tracker.py`

---

### 10. `serving/` - Model Inference & A/B Testing

**Purpose**: Load models and serve predictions with A/B testing support.

```
serving/
├── __init__.py                  # Package initialization
├── experiment_tracker.py        # MLflow integration
│   ├── initialize_mlflow(): Setup MLflow
│   │   - Tracking URI: model_registry/mlflow
│   │   - Backend: SQLite at model_registry/db_backend/mlflow.db
│   │   - Create experiments if missing
│   │
│   ├── log_experiment_run(): Log training run
│   │   - Parameters: model hyperparameters
│   │   - Metrics: accuracy, spectral gap, sojourn times
│   │   - Artifacts: trained model, transition matrix
│   │
│   ├── load_production_model(): Load current model
│   │   - Query MLflow registry for "Production" stage
│   │   - Fallback: Use baseline model
│   │
│   ├── get_model_by_version(): Load specific model version
│   │   - For A/B testing (baseline vs candidate)
│   │
│   └── transition_model_to_production(): Promote model
│       - Move from "Staging" to "Production"
│       - Archive old production version
│
├── inference.py                 # Real-time inference
│   ├── predict_regime(): Single prediction
│   │   - Input: new data point
│   │   - Output: predicted regime (Low/Med/High)
│   │
│   ├── batch_predict(): Multiple predictions
│   │   - Input: batch of data points
│   │   - Output: array of regimes
│   │
│   ├── predict_with_confidence(): Prediction + uncertainty
│   │   - Confidence from transition probabilities
│   │   - Higher confidence if high probability transition
│   │
│   └── predict_next_regime_prob(): Probability distribution
│       - Output: P(next=Low), P(next=Med), P(next=High)
│
├── ab_test_serving.py           # A/B test serving
│   ├── get_serving_model(): Returns baseline or candidate
│   │   - Based on traffic split (e.g., 95% baseline, 5% candidate)
│   │   - Tracks which model made each prediction
│   │
│   ├── route_to_model(): Route prediction to model
│   │   - Baseline: production model
│   │   - Candidate: challenger model
│   │
│   └── record_prediction(): Log prediction for evaluation
│       - Needed for A/B test analysis
│
└── model_cache.py               # Model caching
    ├── cache_model(): Load and cache model
    ├── invalidate_cache(): Clear cache on new version
    └── get_cached_model(): Retrieve cached model
```

**Inference Pipeline:**

```
New data point arrives
    ↓
load_production_model() from MLflow
    ↓
predict_regime(data_point)
    ├─ Get current state from data
    ├─ Query transition matrix P
    ├─ Return most likely next state
    └─ Return confidence score
    ↓
Prediction + confidence → dashboard/alert
    ↓
Log prediction (for monitoring)
    ↓
Compare with actual regime (when known)
    ├─ Accuracy metric
    ├─ Drift detection
    └─ Retraining trigger check
```

**A/B Test Serving:**

```
Traffic arrives
    ↓
get_serving_model() → traffic split?
    /  \
   /    \
95%     5%
  ↓      ↓
Load   Load
Base   Cand
Model  Model
  ↓      ↓
Pred   Pred
(Base) (Cand)
  ↓      ↓
record_prediction(model_id, pred)
    ↓
After 1 week:
  Compare accuracies
  → Determine winner
```

**Coupling:**
- **MLflow**: All model storage and versioning
- **Monitoring**: Predictions logged for drift detection
- **Retraining**: Triggers model retraining if accuracy drops
- **Dashboard**: Current prediction displayed
- **Data**: Inference on preprocessed data only

---

### 11. `orchestration/` - Pipeline Orchestration

**Purpose**: Coordinate data pipeline and model training workflows.

```
orchestration/
├── __init__.py                  # Package initialization
├── pipeline.py                  # Main DAG
│   ├── ETL Pipeline (Bronze→Silver→Gold)
│   ├── Training Pipeline (Gold→Model)
│   ├── Monitoring Pipeline (Predictions→Alerts)
│   └── Dependency management
│
└── README.md                    # Pipeline documentation
```

**Pipeline DAG Example:**

```
┌──────────────┐
│ Ingest Data  │
└───────┬──────┘
        │
        ▼
┌──────────────────────┐
│ Bronze Validation    │
│ (87.3% threshold)    │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ Cleaning & Enrichment│
│ (preprocessing/)     │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ Silver Validation    │
│ (94.5% threshold)    │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ Feature Engineering  │
│ (regime_discretize)  │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ Gold Validation      │
│ (98.1% threshold)    │
└───────┬──────────────┘
        │
    ┌───┴──────────────────┐
    │                      │
    ▼                      ▼
┌─────────────┐      ┌──────────────┐
│ Train Model │      │ Monitoring   │
│ (modeling/) │      │ (drift chk)  │
└───────┬─────┘      └──────────────┘
        │
        ▼
┌──────────────────────┐
│ MLflow Logging       │
│ (experiment_tracker) │
└───────┬──────────────┘
        │
        ▼
┌──────────────────────┐
│ Check Retraining     │
│ (scheduler.py)       │
└───────┬──────────────┘
        │
    ┌───┴─ If triggered
    │
    ▼
┌──────────────────────┐
│ A/B Testing          │
│ (ab_testing.py)      │
└──────────────────────┘
```

**Scheduling:**
- Runs on schedule: hourly data check, weekly retraining
- Can be triggered manually or by monitoring alerts
- Orchestrates all sub-workflows
- Logs execution to `logs/orchestration.log`

**Coupling:**
- Orchestrates ALL other modules
- Central authority for pipeline state
- Manages dependencies between tasks

---

### 12. `model_registry/` - MLflow Artifacts & Metadata

**Purpose**: Store trained models, experiments, metrics, and metadata.

```
model_registry/
├── mlflow/                      # MLflow tracking
│   ├── 0/                       # Default experiment
│   ├── 1/                       # markov_chain_baseline
│   ├── 2/                       # markov_chain_absorbing
│   ├── 3/                       # markov_chain_comparison
│   ├── 4/                       # data_sensitivity_analysis
│   └── <run_id>/                # Each training run
│       ├── artifacts/
│       │   ├── model.pkl        # Trained model
│       │   ├── transition_matrix.csv
│       │   └── metadata.json
│       ├── metrics/
│       │   ├── accuracy.json
│       │   ├── spectral_gap.json
│       │   └── ...
│       └── params/
│           ├── learning_rate.json
│           ├── regularization.json
│           └── ...
│
├── db_backend/                  # SQLite backend
│   └── mlflow.db                # Experiment metadata DB
│
├── artifacts/                   # Alternative artifact store
│   └── (symlink or copy of artifacts)
│
├── mlflow_artifacts/            # MLflow native artifacts
│   └── (auto-created by MLflow)
│
├── mlflow_backend/              # MLflow backend store
│   └── (auto-created by MLflow)
│
├── retraining_jobs.jsonl        # Retraining history
│   ├── Each line: JSON with job metadata
│   ├── Fields: timestamp, trigger, status, metrics
│   └── Used for: historical analysis, audit trail
│
└── rollback_events.jsonl        # Model rollback history
    ├── Each line: JSON with rollback event
    ├── Fields: timestamp, from_version, to_version, reason
    └── Used for: audit trail, decision history
```

**Data Storage Structure:**

```
Model → MLflow Registration
    ↓
├─ Experiment ID (e.g., "markov_chain_baseline")
├─ Run ID (UUID)
├─ Artifacts (model.pkl, matrices)
├─ Metrics (accuracy, spectral_gap, ...)
├─ Parameters (hyperparameters)
└─ Tags (version, stage)
    ↓
Queryable by:
- Experiment name
- Run ID
- Metric value (best accuracy)
- Tag (stage: Production, Staging, Archived)
```

**Coupling:**
- **Input**: Trained models from `modeling/`
- **Logging**: Via `serving/experiment_tracker.py`
- **Retrieval**: By `serving/inference.py` for predictions
- **Monitoring**: Metadata available to `monitoring/` for evaluation
- **Retraining**: New models registered here

---

### 13. `logs/` - Application Logging

**Purpose**: Store application logs for debugging and audit trail.

```
logs/
├── preprocessing.log            # Data cleaning logs
├── training.log                 # Model training logs
├── monitoring.log               # Monitoring and drift detection
├── orchestration.log            # Pipeline execution
├── dashboard.log                # Dashboard errors/info
├── inference.log                # Prediction logs
└── mlflow.log                   # MLflow operations
```

**Usage:**
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Starting preprocessing: {df.shape}")
logger.warning(f"Data quality below threshold: {quality_score}")
logger.error(f"Validation failed: {error_message}")
```

**Coupling:**
- All modules write to respective log files
- Centralized logging configuration
- Useful for debugging, monitoring, and audit

---

### 14. `tests/` - Test Suite

**Purpose**: Validate all components (Unit + Integration tests).

```
tests/
├── __init__.py
├── test_data_pipeline.py        # Data validation tests
│   ├── test_bronze_validation()
│   ├── test_silver_validation()
│   ├── test_gold_validation()
│   └── test_quality_scores()
│
├── test_evaluation_metrics.py   # Model evaluation tests
│   ├── test_accuracy_calculation()
│   ├── test_spectral_gap()
│   ├── test_sojourn_times()
│   └── test_cross_validation()
│
├── test_mlflow_integration.py   # MLflow tests
│   ├── test_experiment_setup()
│   ├── test_run_logging()
│   ├── test_model_registry()
│   └── test_artifact_storage()
│
├── test_dashboard.py            # Dashboard tests
│   ├── test_imports()
│   ├── test_component_rendering()
│   ├── test_data_loader()
│   └── test_page_routing()
│
├── conftest.py                  # Shared pytest fixtures
│   ├── sample_data fixtures
│   ├── mlflow client fixture
│   └── temp directories
│
└── requirements.txt             # Test dependencies
```

**Test Coverage:**
- Data pipeline: validates all layers
- ML pipeline: ensures metrics are correct
- Monitoring: drift detection accuracy
- Dashboard: component rendering
- Integration: end-to-end workflows

**Running Tests:**
```bash
pytest tests/
pytest tests/test_data_pipeline.py -v
pytest --cov=.
```

---

### 15. `utils/` - Shared Utilities

**Purpose**: Common functions used across modules.

```
utils/
├── __init__.py
├── helpers.py                   # General utilities
│   ├── load_config(): Load YAML config
│   ├── get_project_root(): Project path
│   ├── create_logger(): Setup logging
│   └── safe_divide(): Division with zero handling
│
├── constants.py                 # Project-wide constants
│   ├── REGIMES: List of regime names
│   ├── STATES: State indices
│   ├── THRESHOLDS: Default thresholds
│   └── PATHS: Directory paths
│
├── validators.py                # Validation utilities
│   ├── validate_dataframe(): Check DataFrame structure
│   ├── validate_regime_column(): Verify regime values
│   └── validate_config(): Check config validity
│
└── formatters.py                # Formatting utilities
    ├── format_percentage(): Format as %
    ├── format_duration(): Format time duration
    └── format_markdown(): Format for markdown
```

**Usage:**
```python
from utils import helpers
logger = helpers.create_logger(__name__)
config = helpers.load_config('config/config.yaml')
```

---

### 16. `ci_cd/` & `docker/` - Deployment

**Purpose**: Containerization and CI/CD configuration.

```
ci_cd/
├── github_actions/              # GitHub Actions workflows
│   ├── test.yml                # Run tests on PR
│   ├── deploy.yml              # Deploy on merge to main
│   └── schedule.yml            # Scheduled retraining
│
└── docker/                      # Docker configuration
    ├── Dockerfile              # Container image definition
    ├── docker-compose.yml      # Multi-container setup
    └── .dockerignore           # Files to exclude from image
```

**Dockerfile Example:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboards/app.py"]
```

---

## 📊 Key Interactions & Data Flow

### Complete Data Journey Through System:

```
1. RAW DATA INGESTION
   External APIs/CSV → data/bronze/
   ↓ Validation: validate_bronze.py
   ↓ Quality check: 85%+ completeness, valid schema
   
2. PREPROCESSING
   Bronze → preprocessing/cleaning.py
   ↓ Remove duplicates, handle nulls, normalize
   ↓ Result: data/silver/
   ↓ Validation: validate_silver_gold.py
   ↓ Quality check: 94%+ valid
   
3. FEATURE ENGINEERING
   Silver → preprocessing/regime_discretization.py
   ↓ Create regime labels (Low/Med/High)
   ↓ Create Markov state vectors
   ↓ Result: data/gold/
   ↓ Validation: validate_silver_gold.py
   ↓ Quality check: 98%+ valid
   
4. MODEL TRAINING
   Gold → modeling/models/markov_chain.py
   ├─ Fit transition matrix
   ├─ Calculate metrics (spectral gap, sojourn)
   └─ Evaluate with modeling/evaluation/
   
5. EXPERIMENT TRACKING
   Trained model → serving/experiment_tracker.py
   └─ Log to MLflow → model_registry/
   
6. MONITORING
   Production predictions → monitoring/
   ├─ Drift detection: monitoring/drift_detection/
   ├─ Anomaly detection: monitoring/anomaly_detector.py
   └─ Alert generation: monitoring/alerts/
   
7. RETRAINING DECISION
   Monitoring alerts → retraining/scheduler.py
   ├─ Check thresholds from config/
   ├─ Trigger if conditions met
   └─ Run A/B test: retraining/ab_testing.py
   
8. SERVING
   Production model ← MLflow Registry
   ├─ Real-time: serving/inference.py
   ├─ A/B test: serving/ab_test_serving.py
   └─ Predictions → dashboard/utils/data_loader.py
   
9. VISUALIZATION
   Data → dashboards/utils/data_loader.py
   ├─ Mock fallback if unavailable
   ├─ Cache 5 minutes
   └─ Display via dashboards/pages/
```

---

## 📈 File Significance & Criticality

### Tier 1 (Critical - System won't run without)
- `orchestration/pipeline.py` - Central orchestrator
- `data_validation/` - Ensures data quality
- `preprocessing/cleaning.py` - Data preparation
- `modeling/models/markov_chain.py` - Core ML model
- `serving/experiment_tracker.py` - Model registry integration

### Tier 2 (Important - Major functionality)
- `monitoring/` - System health
- `retraining/scheduler.py` - Continuous improvement
- `dashboards/app.py` - User interface
- `modeling/evaluation/` - Model validation
- `config/` - System configuration

### Tier 3 (Supporting - Enhanced functionality)
- `eda/` - Data analysis
- `retraining/ab_testing.py` - Deployment validation
- `dashboards/components/` - UI polish
- `utils/` - Helper functions
- `tests/` - Quality assurance

---

## 🔗 Module Coupling Map

```
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATION (Central)                    │
├─────────────────────────────────────────────────────────┤
│  ├→ data/ (Bronze-Silver-Gold)                          │
│  ├→ data_validation/ (Quality checks)                   │
│  ├→ preprocessing/ (Cleaning & enrichment)              │
│  ├→ modeling/ (Training & evaluation)                   │
│  ├→ serving/experiment_tracker.py (MLflow logging)      │
│  ├→ monitoring/ (Drift & anomaly detection)             │
│  ├→ retraining/scheduler.py (Retraining trigger)        │
│  └→ model_registry/ (Artifact storage)                  │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  SERVING (Inference)                                    │
│  ├→ model_registry/ (Load model)                        │
│  ├→ monitoring/ (Log predictions)                       │
│  └→ dashboards/utils/data_loader.py (Feed UI)           │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  MONITORING (Continuous)                                │
│  ├→ config/ (Thresholds)                                │
│  ├→ serving/ (Prediction data)                          │
│  ├→ retraining/scheduler.py (Trigger retraining)        │
│  ├→ model_registry/ (Log alerts)                        │
│  └→ dashboards/ (Display alerts)                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Folder | Purpose | Key File | Input | Output | Coupling |
|--------|---------|----------|-------|--------|----------|
| `data/` | Data storage | N/A | Raw data | Bronze→Silver→Gold | All modules |
| `dashboards/` | UI/Visualization | app.py | Model registry | Web interface | Data loader |
| `data_validation/` | Data QA | validate_*.py | Each layer data | Quality score | Preprocessing |
| `eda/` | Data analysis | *.py | Each layer | Statistics | Manual review |
| `config/` | Configuration | *.yaml | N/A | Config objects | All modules |
| `preprocessing/` | Data cleaning | cleaning.py | Bronze data | Silver→Gold | Validation |
| `modeling/` | ML training | markov_chain.py | Gold data | Trained model | Experiment tracker |
| `monitoring/` | Health checks | anomaly_detector.py | Predictions | Alerts | Retraining scheduler |
| `retraining/` | Model updates | scheduler.py | Metrics | New model | A/B testing |
| `serving/` | Inference | experiment_tracker.py | Model registry | Predictions | Monitoring |
| `orchestration/` | DAG | pipeline.py | All components | Workflow | All modules |
| `model_registry/` | Model storage | mlflow.db | Trained models | Model artifacts | Serving |
| `logs/` | Logging | *.log | All modules | Log files | All modules |
| `tests/` | Testing | test_*.py | Components | Test results | CI/CD |
| `utils/` | Helpers | *.py | N/A | Utilities | All modules |

---

This guide provides a comprehensive map of every folder and file, their purposes, interactions, and significance within the FINML ML system.
