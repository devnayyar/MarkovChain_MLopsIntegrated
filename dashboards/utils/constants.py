"""Dashboard constants and configuration."""

# Regime colors
REGIME_COLORS = {
    "LOW_RISK": "#2ecc71",      # Green
    "MODERATE_RISK": "#f39c12",  # Yellow/Orange
    "HIGH_RISK": "#e74c3c",      # Red
    "Normal": "#2ecc71",         # Green
    "Stress": "#f39c12",         # Yellow
    "Crisis": "#e74c3c",         # Red
}

# Status indicator colors
STATUS_COLORS = {
    "Good": "#27ae60",        # Green
    "Warning": "#e67e22",     # Orange
    "Critical": "#c0392b",    # Red
    "Info": "#3498db",        # Blue
}

# Alert severity colors
ALERT_SEVERITY_COLORS = {
    "INFO": "#3498db",        # Blue
    "WARNING": "#e67e22",     # Orange
    "CRITICAL": "#c0392b",    # Red
}

# Alert severity emojis
ALERT_SEVERITY_EMOJIS = {
    "INFO": "🟢",
    "WARNING": "🟡",
    "CRITICAL": "🔴",
}

# Thresholds
THRESHOLDS = {
    "drift_warning": 0.1,
    "drift_critical": 0.15,
    "accuracy_warning": 0.85,
    "accuracy_critical": 0.80,
    "data_quality_warning": 0.90,
    "data_quality_critical": 0.85,
}

# Page names
PAGES = {
    "🏠 Home": "home",
    "📈 Regime Timeline": "regime_timeline",
    "🔗 Markov Chain": "markov_chain",
    "📊 Model Metrics & Diagnostics": "model_metrics",
    "🔮 Experiment Runner": "markov_experiment_runner",
    "🚨 Alerts & Drift": "alerts_drift",
    "📈 Performance Metrics": "metrics_performance",
    "🔍 EDA Analysis": "eda_analysis",
    "� Documentation & Guide": "documentation",
    "�🔄 Retraining & A/B Testing": "retraining_ab_testing",
    "⚙️ Settings": "settings",
}

# Time ranges
TIME_RANGES = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Year to Date": 365,
}

# Refresh intervals
REFRESH_INTERVALS = {
    "30 seconds": 30,
    "1 minute": 60,
    "5 minutes": 300,
}

# Data file paths
DATA_PATHS = {
    "markov_state": "data/gold/markov_state_sequences.parquet",
    "performance_metrics": "model_registry/performance_metrics.jsonl",
    "alerts": "model_registry/alerts.jsonl",
    "anomalies": "model_registry/anomalies.jsonl",
    "degradation_events": "model_registry/degradation_events.jsonl",
    "retraining_jobs": "model_registry/retraining_jobs.jsonl",
    "ab_tests": "model_registry/ab_tests.jsonl",
    "rollback_events": "model_registry/rollback_events.jsonl",
    "job_history": "model_registry/job_history.jsonl",
}
