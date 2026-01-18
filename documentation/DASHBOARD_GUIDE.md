# DASHBOARD_GUIDE: UI for Real-Time Monitoring

Comprehensive guide to the FINML Streamlit dashboard - the visual interface for monitoring models, data, and system health.

---

## Table of Contents

1. [Dashboard Overview](#dashboard-overview)
2. [Architecture](#architecture)
3. [Pages Guide](#pages-guide)
4. [Components](#components)
5. [Data Flow](#data-flow)
6. [Customization](#customization)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Dashboard Overview

### Purpose

The FINML dashboard provides real-time visibility into:

1. **Model Performance**: Accuracy, precision, recall, spectral properties
2. **Data Quality**: Distribution shifts, drift detection, regime changes
3. **System Health**: Pipeline status, error rates, processing times
4. **Monitoring**: Anomalies, alerts, threshold violations
5. **Analytics**: Feature importance, regime transitions, predictive patterns

### Technology Stack

- **Framework**: Streamlit (Python-based UI)
- **Visualization**: Plotly, Matplotlib
- **Data Source**: MLflow, monitoring database
- **Deployment**: Docker container
- **Update Frequency**: Real-time (2-5 second refresh)

### Key Metrics Dashboard

```
┌─────────────────────────────────────────────────────┐
│          FINML Financial Risk Markov Chain         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Model Accuracy: 87.24%  │  Spectral Gap: 0.345    │
│  Regime Coverage: 99.8%  │  Data Quality: 0.94     │
│                                                     │
│  Last Update: 2026-01-18 14:23:45 UTC              │
│  Monitoring Status: ✓ Active                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Architecture

### File Structure

```
dashboards/
├── app.py                  # Main entry point
├── README.md              # Dashboard documentation
├── requirements.txt       # Dashboard dependencies
│
├── pages/                 # Streamlit multi-page app
│   ├── 1_Model_Overview.py           # Main model metrics
│   ├── 2_Performance_Analysis.py     # Evaluation metrics
│   ├── 3_Data_Quality.py             # Data drift detection
│   ├── 4_Monitoring_Alerts.py        # System alerts
│   ├── 5_Feature_Analysis.py         # Feature importance
│   └── 6_System_Health.py            # Infrastructure metrics
│
├── components/            # Reusable dashboard components
│   ├── metric_card.py             # KPI card component
│   ├── chart_generator.py         # Chart templates
│   ├── alert_display.py           # Alert rendering
│   └── filters_panel.py           # Interactive filters
│
└── utils/                 # Utility functions
    ├── data_loader.py             # Load data from sources
    ├── cache_manager.py           # Caching for performance
    ├── config_loader.py           # Configuration management
    └── formatter.py               # Data formatting helpers
```

### Component Hierarchy

```
app.py (Main Entry)
├── Page Selection (Sidebar Navigation)
│   ├── Model Overview
│   ├── Performance Analysis
│   ├── Data Quality
│   ├── Monitoring & Alerts
│   ├── Feature Analysis
│   └── System Health
│
└── Shared Components
    ├── Metric Cards (KPI displays)
    ├── Charts (Plotly/Matplotlib)
    ├── Tables (Data grids)
    ├── Filters (Date, regime, metric selection)
    └── Alerts (Warning banners)
```

---

## Pages Guide

### Page 1: Model Overview

**File**: `dashboards/pages/1_Model_Overview.py`

**Purpose**: High-level view of the currently deployed model

**Metrics Displayed**:

| Metric | Description | Example |
|--------|-------------|---------|
| Model ID | Current production model | markov_v1.2.3 |
| Accuracy | Overall prediction accuracy | 87.24% |
| Precision | Positive prediction accuracy | 89.12% |
| Recall | True positive detection rate | 85.36% |
| F1-Score | Harmonic mean of precision/recall | 0.8723 |
| Spectral Gap | Eigenvalue separation | 0.345 |
| Training Date | When model was trained | 2026-01-15 |
| Last Retrain | Most recent retraining | 2026-01-17 |

**Code Snippet**:

```python
import streamlit as st
import pandas as pd
from monitoring.dashboard_data import get_current_model_metrics

st.set_page_config(page_title="Model Overview", layout="wide")

st.title("🎯 Model Overview")

# Load current metrics
metrics = get_current_model_metrics()

# Display as metric cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}", 
              delta=f"{metrics['accuracy_change']:+.2%}")

with col2:
    st.metric("Precision", f"{metrics['precision']:.2%}", 
              delta=f"{metrics['precision_change']:+.2%}")

with col3:
    st.metric("F1-Score", f"{metrics['f1_score']:.2%}", 
              delta=f"{metrics['f1_change']:+.2%}")

with col4:
    st.metric("Spectral Gap", f"{metrics['spectral_gap']:.4f}", 
              delta=f"{metrics['spectral_change']:+.4f}")

# Model history
st.subheader("Model Version History")
history_df = get_model_history()
st.dataframe(history_df, use_container_width=True)
```

**Visual Elements**:
- 4 large KPI cards (Accuracy, Precision, Recall, F1)
- Model version dropdown
- Training date display
- Trend indicators (↑ improving, ↓ degrading, → stable)
- Version history table

---

### Page 2: Performance Analysis

**File**: `dashboards/pages/2_Performance_Analysis.py`

**Purpose**: Deep dive into model evaluation metrics

**Visualizations**:

1. **Confusion Matrix Heatmap**
   - True Positives, False Positives, True Negatives, False Negatives
   - Color intensity shows cell frequency
   
2. **ROC Curve**
   - True Positive Rate vs. False Positive Rate
   - AUC score displayed
   
3. **Precision-Recall Curve**
   - Precision vs. Recall across thresholds
   - F1-Score optimal point highlighted
   
4. **Performance Over Time**
   - Accuracy trend over last 30 days
   - Trend line and confidence interval
   
5. **Metric Comparison**
   - Current vs. previous model
   - % improvement/degradation

**Code Snippet**:

```python
import streamlit as st
import plotly.graph_objects as go
from modeling.evaluation.metrics import get_confusion_matrix, get_roc_curve

st.title("📊 Performance Analysis")

# Get evaluation data
model_id = st.selectbox("Select Model", get_available_models())
metrics = get_model_evaluation_metrics(model_id)

# Confusion matrix
cm = get_confusion_matrix(model_id)

fig = go.Figure(data=go.Heatmap(
    z=cm.values,
    x=['Predicted Neg', 'Predicted Pos'],
    y=['Actually Neg', 'Actually Pos'],
    text=cm.values,
    texttemplate="%{text}",
    colorscale='Blues'
))
fig.update_layout(title="Confusion Matrix")
st.plotly_chart(fig, use_container_width=True)

# ROC Curve
roc = get_roc_curve(model_id)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=roc['fpr'], y=roc['tpr'],
    mode='lines',
    name=f"AUC = {roc['auc']:.3f}"
))
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Classifier',
    line=dict(dash='dash')
))
st.plotly_chart(fig, use_container_width=True)

# Metrics comparison
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
with col2:
    st.metric("AUC-ROC", f"{metrics['auc']:.3f}")
```

**Interactive Features**:
- Model version selector
- Date range picker
- Metric toggles
- Download performance report

---

### Page 3: Data Quality

**File**: `dashboards/pages/3_Data_Quality.py`

**Purpose**: Monitor input data distribution and drift detection

**Visualizations**:

1. **Distribution Plots**
   - Regime distribution (Bar chart)
   - Feature distributions (Histograms)
   - Gold vs. Silver layer comparison
   
2. **Drift Detection**
   - KS statistic trends
   - Drift threshold indication
   - Regime transition drift
   
3. **Data Quality Scores**
   - Missing data %
   - Outlier count
   - Regime coverage
   - Quality trend over time
   
4. **Regime Transitions**
   - Transition matrix heatmap
   - Regime persistence (diagonal values)
   - Transition probability changes

**Code Snippet**:

```python
import streamlit as st
import plotly.graph_objects as go
from monitoring.drift_detection import get_drift_metrics

st.title("📈 Data Quality & Drift Detection")

# Get drift metrics
drift_metrics = get_drift_metrics()

# Drift detection
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("KS Statistic", f"{drift_metrics['ks_stat']:.4f}",
              delta=f"{drift_metrics['ks_change']:+.4f}")

with col2:
    threshold = 0.15
    status = "🔴 ALERT" if drift_metrics['ks_stat'] > threshold else "🟢 OK"
    st.metric("Drift Status", status)

with col3:
    st.metric("Data Quality", f"{drift_metrics['quality_score']:.1%}")

# Regime transitions
st.subheader("Regime Transitions")
trans_matrix = get_transition_matrix()

fig = go.Figure(data=go.Heatmap(
    z=trans_matrix.values,
    x=trans_matrix.columns,
    y=trans_matrix.index,
    text=trans_matrix.values.round(3),
    texttemplate="%{text}",
    colorscale='RdYlGn'
))
st.plotly_chart(fig, use_container_width=True)
```

**Key Indicators**:
- 🟢 Green: Normal (KS < 0.10)
- 🟡 Yellow: Caution (KS 0.10-0.15)
- 🔴 Red: Alert (KS > 0.15)

---

### Page 4: Monitoring & Alerts

**File**: `dashboards/pages/4_Monitoring_Alerts.py`

**Purpose**: Real-time system monitoring and alert management

**Features**:

1. **Active Alerts**
   - Alert type (Data Quality, Model Performance, System)
   - Severity level (Low, Medium, High, Critical)
   - Detection time
   - Resolution status
   
2. **Alert History**
   - Historical alert log
   - Time to resolution
   - Alert frequency trends
   
3. **System Status**
   - Pipeline status (Running, Paused, Error)
   - Last successful run
   - Next scheduled run
   - Error count (24h)

**Code Snippet**:

```python
import streamlit as st
from monitoring.alerts import get_active_alerts, get_alert_history

st.title("⚠️ Monitoring & Alerts")

# Active alerts
st.subheader("Active Alerts")
active_alerts = get_active_alerts()

for alert in active_alerts:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    severity_color = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢'
    }
    
    with col1:
        st.write(f"{severity_color[alert['severity']]} {alert['message']}")
    
    with col2:
        st.write(f"__{alert['created_at'].strftime('%H:%M:%S')}__")
    
    with col3:
        if st.button("Acknowledge", key=alert['id']):
            acknowledge_alert(alert['id'])

# Alert history
st.subheader("Alert History (Last 7 Days)")
history = get_alert_history(days=7)
st.dataframe(history, use_container_width=True)

# System metrics
st.subheader("System Health")
col1, col2, col3 = st.columns(3)

system_status = get_system_status()

with col1:
    st.metric("Pipeline Status", system_status['status'],
              delta=f"{system_status['uptime_pct']:.1f}% uptime")

with col2:
    st.metric("Last Run", system_status['last_run_time'],
              delta=system_status['last_run_duration'])

with col3:
    st.metric("Errors (24h)", system_status['error_count'],
              delta=f"{system_status['error_trend']:+d}")
```

**Alert Management**:
- Acknowledge alerts
- Dismiss false positives
- Set custom alert thresholds
- Export alert log

---

### Page 5: Feature Analysis

**File**: `dashboards/pages/5_Feature_Analysis.py`

**Purpose**: Understand feature importance and behavior

**Visualizations**:

1. **Feature Importance**
   - SHAP values
   - Permutation importance
   - Correlation with regime
   
2. **Feature Distributions**
   - By regime (violin plots)
   - Statistical summary
   - Trend over time
   
3. **Feature Correlation**
   - Correlation matrix heatmap
   - Multicollinearity detection
   - VIF (Variance Inflation Factor)

**Code Snippet**:

```python
import streamlit as st
import plotly.graph_objects as go
from modeling.feature_analysis import get_feature_importance

st.title("🔍 Feature Analysis")

# Feature importance
st.subheader("Feature Importance (SHAP)")
importance = get_feature_importance()

fig = go.Figure(
    data=go.Bar(
        y=importance['feature'].head(15),
        x=importance['importance'].head(15),
        orientation='h'
    )
)
fig.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    title="Top 15 Most Important Features"
)
st.plotly_chart(fig, use_container_width=True)

# Feature distributions by regime
st.subheader("Feature Distributions by Regime")
feature = st.selectbox("Select Feature", importance['feature'].head(10))

distributions = get_feature_distributions_by_regime(feature)

fig = go.Figure()
for regime in distributions.keys():
    fig.add_trace(go.Violin(
        y=distributions[regime],
        name=regime,
        box_visible=True,
        meanline_visible=True
    ))

st.plotly_chart(fig, use_container_width=True)
```

---

### Page 6: System Health

**File**: `dashboards/pages/6_System_Health.py`

**Purpose**: Infrastructure and pipeline metrics

**Metrics**:

1. **Pipeline Health**
   - Data ingestion latency
   - Processing time per batch
   - Queue depth
   - Error rate
   
2. **Resource Usage**
   - CPU utilization
   - Memory usage
   - Disk space
   - Network I/O
   
3. **Model Serving**
   - Inference latency (p50, p95, p99)
   - Throughput (predictions/second)
   - Cache hit rate
   - Model load time

**Code Snippet**:

```python
import streamlit as st
from monitoring.performance import get_system_metrics

st.title("🔧 System Health")

metrics = get_system_metrics()

# Performance metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Inference Latency (p50)",
              f"{metrics['latency_p50']:.0f}ms")

with col2:
    st.metric("Inference Latency (p95)",
              f"{metrics['latency_p95']:.0f}ms")

with col3:
    st.metric("Throughput",
              f"{metrics['throughput']:.0f} pred/s")

with col4:
    st.metric("Cache Hit Rate",
              f"{metrics['cache_hit_rate']:.1%}")

# Resource usage
st.subheader("Resource Utilization")
resources = get_resource_metrics()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("CPU Usage", f"{resources['cpu_pct']:.1f}%")

with col2:
    st.metric("Memory Usage", f"{resources['memory_pct']:.1f}%")

with col3:
    st.metric("Disk Usage", f"{resources['disk_pct']:.1f}%")
```

---

## Components

### Metric Card Component

**File**: `dashboards/components/metric_card.py`

```python
import streamlit as st

def metric_card(title: str, value: float, unit: str = "", 
                delta: float = None, delta_color: str = None):
    """
    Display a KPI metric card.
    
    Args:
        title: Metric name
        value: Current value
        unit: Unit of measurement (%, ms, etc.)
        delta: Change from previous (optional)
        delta_color: 'green', 'red', 'gray'
    """
    col = st.container()
    
    with col:
        st.metric(
            label=title,
            value=f"{value:.2f}{unit}",
            delta=f"{delta:+.2f}{unit}" if delta else None,
            delta_color=delta_color
        )
```

### Chart Generator Component

**File**: `dashboards/components/chart_generator.py`

```python
import plotly.graph_objects as go
import plotly.express as px

def line_chart(data, x, y, title, show_ci=False):
    """Generate time-series line chart."""
    fig = px.line(data, x=x, y=y, title=title)
    if show_ci:
        fig.add_trace(...)  # Add confidence interval
    return fig

def heatmap(data, title, colorscale='Blues'):
    """Generate heatmap visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=colorscale
    ))
    fig.update_layout(title=title)
    return fig
```

### Alert Display Component

**File**: `dashboards/components/alert_display.py`

```python
import streamlit as st

def alert_banner(message: str, severity: str = "info"):
    """Display alert banner."""
    severity_styles = {
        'info': ('ℹ️', 'blue'),
        'warning': ('⚠️', 'orange'),
        'error': ('❌', 'red'),
        'success': ('✓', 'green')
    }
    
    icon, color = severity_styles[severity]
    st.markdown(f"<div style='background-color: {color}; padding: 10px'>"
                f"{icon} {message}</div>",
                unsafe_allow_html=True)
```

---

## Data Flow

### Dashboard Data Pipeline

```
Data Sources
├── MLflow
│   ├── Model metadata
│   ├── Experiment runs
│   └── Metrics history
│
├── Monitoring DB
│   ├── Performance metrics
│   ├── Drift detection results
│   └── System health
│
└── Gold Layer (Data Lake)
    ├── Feature data
    ├── Regime labels
    └── Raw predictions

        ↓
        
Cache Layer (Redis/In-Memory)
├── Cache key: {metric}_{timestamp_hour}
├── TTL: 5 minutes
└── Invalidation on data update

        ↓

Dashboard Components
├── Page 1: Model Overview
├── Page 2: Performance Analysis
├── Page 3: Data Quality
├── Page 4: Monitoring & Alerts
├── Page 5: Feature Analysis
└── Page 6: System Health

        ↓

User Browser (Streamlit)
```

### Data Loading Strategy

```python
import streamlit as st
from functools import lru_cache

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_model_metrics():
    """Load metrics with caching."""
    return get_current_model_metrics()

@st.cache_resource
def load_mlflow_client():
    """Load MLflow client once."""
    return mlflow.tracking.MlflowClient()
```

---

## Customization

### Adding a New Metric

1. **Create metric calculation**:
   ```python
   def calculate_custom_metric(model_id):
       return {"value": 0.87, "trend": 0.02}
   ```

2. **Add to data loader**:
   ```python
   # dashboards/utils/data_loader.py
   def get_custom_metrics():
       return calculate_custom_metric(get_current_model_id())
   ```

3. **Display in dashboard**:
   ```python
   # dashboards/pages/1_Model_Overview.py
   custom = get_custom_metrics()
   st.metric("Custom Metric", f"{custom['value']:.2%}",
             delta=f"{custom['trend']:+.2%}")
   ```

### Changing Update Frequency

```python
# dashboards/utils/cache_manager.py

# Default: 5 minutes
@st.cache_data(ttl=300)
def load_metrics():
    return get_current_metrics()

# Increase to 15 minutes for less frequent updates
@st.cache_data(ttl=900)
def load_metrics():
    return get_current_metrics()
```

### Custom Color Schemes

```python
# dashboards/config/theme.py

THEME = {
    "primary": "#0066FF",
    "success": "#00AA33",
    "warning": "#FF9900",
    "error": "#CC0000",
    "bg": "#F5F5F5"
}

# Use in charts
fig.update_layout(template="plotly_white",
                  plot_bgcolor=THEME["bg"])
```

---

## Performance Optimization

### Caching Strategy

```python
# Level 1: Function caching (5 min)
@st.cache_data(ttl=300)
def get_metrics():
    pass

# Level 2: Session state (within session)
@st.cache_resource
def get_mlflow_client():
    pass

# Level 3: File cache (24 hours)
@st.cache_data(ttl=86400)
def load_model_artifact(model_id):
    pass
```

### Query Optimization

```python
# ❌ Bad: Load all history
metrics_all = load_all_metrics_history()  # 1000s of rows

# ✓ Good: Load recent data only
metrics_recent = load_metrics_since(hours=24)  # 100s of rows
```

### Lazy Loading

```python
# Use tabs for page sections
tab1, tab2, tab3 = st.tabs(["Overview", "Details", "History"])

with tab1:
    st.write("Quick metrics")  # Load immediately

with tab2:
    # Only load if user clicks tab
    if tab2:
        detailed_metrics = load_detailed_metrics()
        st.write(detailed_metrics)
```

---

## Troubleshooting

### Dashboard Not Updating

**Symptom**: Metrics show old data

**Causes & Solutions**:
1. Cache TTL too long: Reduce `ttl` parameter
2. Data source not updating: Check data pipeline
3. Browser cache: Hard refresh (Ctrl+Shift+R)

```python
# Force refresh by changing cache key
@st.cache_data(ttl=60)  # Shorter TTL
def get_metrics():
    return get_current_metrics()
```

### Slow Page Load

**Symptom**: Dashboard takes >5 seconds to load

**Causes & Solutions**:
1. Too much data loaded: Use pagination
2. Expensive query: Optimize query
3. Large visualization: Use `use_container_width=True`

```python
# ✓ Optimize
st.dataframe(df.head(100), use_container_width=True)

# ❌ Avoid
st.dataframe(df)  # Shows all rows, slow render
```

### Missing Data in Charts

**Symptom**: Charts appear empty

**Causes & Solutions**:
1. Data not loaded: Check data source
2. Filter too restrictive: Adjust date range
3. Cache stale: Clear cache

```python
# Clear cache
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()
```

### Connection Errors

**Symptom**: "Connection failed to MLflow"

**Solutions**:
1. Check MLflow server: `curl http://localhost:5000`
2. Check credentials: Verify API keys
3. Check network: Verify firewall rules

---

## Summary

The FINML dashboard provides:

1. **Real-time Monitoring**: Live metrics and alerts
2. **Multi-page Interface**: Different views for different stakeholders
3. **Interactive Analysis**: Drill-down and filtering capabilities
4. **Performance**: Caching and optimization
5. **Extensibility**: Easy to add new pages and metrics

Perfect for DevOps, Data Scientists, and Stakeholders!
