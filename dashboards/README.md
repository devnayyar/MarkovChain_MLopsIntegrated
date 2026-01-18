# Financial Risk Markov MLOps Dashboard

A comprehensive Streamlit-based dashboard for monitoring financial regime detection, model performance, alerts, and retraining operations.

## Features

### 📈 Page 1: Regime Timeline
- Real-time regime state visualization (Normal, Stress, Crisis)
- Regime probability distribution
- Transition probability matrix (heatmap)
- Historical regime transitions
- Duration statistics and analysis

### 🚨 Page 2: Alerts & Drift Detection
- Real-time alert feed with severity filtering
- Alert distribution by severity and type
- Drift detection metrics (Jensen-Shannon divergence, etc.)
- Anomaly detection summary
- 9-indicator drift status dashboard
- Alert timeline visualization

### 📊 Page 3: Metrics & Performance
- Current performance metrics (Accuracy, Precision, Recall, F1)
- Performance trends over time (7d, 30d, 90d)
- Data quality scorecard with gauges
- Prediction confidence distribution
- Performance degradation events tracking
- Comprehensive metric statistics table
- System health score

### 🔄 Page 4: Retraining & A/B Testing
- Retraining status and countdown
- Retraining decision criteria progress bars
- A/B test results and comparison
- A/B test progress tracking with statistical significance
- Model version management
- Rollback event history
- Retraining job queue

### ⚙️ Settings Page
- Display preferences (theme, refresh interval, chart type)
- Alert notification channels (Console, Email, Slack, PagerDuty)
- Data export options (CSV, JSON, Parquet)
- System information and feature flags
- Help and support documentation

## Project Structure

```
dashboards/
├── app.py                          # Main entry point
├── __init__.py
├── components/
│   ├── __init__.py
│   ├── header.py                   # Top navigation & filters
│   ├── sidebar.py                  # Side navigation
│   ├── metrics_card.py             # Reusable metric cards
│   └── status_indicator.py         # Status components
├── pages/
│   ├── __init__.py
│   ├── home.py                     # Home/Dashboard overview
│   ├── regime_timeline.py          # Page 1: Regime analysis
│   ├── alerts_drift.py             # Page 2: Alerts & drift
│   ├── metrics_performance.py       # Page 3: Performance metrics
│   ├── retraining_ab_testing.py    # Page 4: Retraining & A/B
│   └── settings.py                 # Settings page
├── utils/
│   ├── __init__.py
│   ├── constants.py                # Color codes, thresholds, paths
│   ├── formatters.py               # Number/date formatting
│   ├── validators.py               # Data validation utilities
│   └── data_loader.py              # Load from JSONL/Parquet
└── README.md
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install streamlit>=1.28.0
pip install plotly>=5.17.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
```

### 2. Configure Data Paths

Update `dashboards/utils/constants.py` with your data file locations:

```python
DATA_PATHS = {
    "markov_state": "data/gold/markov_state_sequences.parquet",
    "performance_metrics": "model_registry/performance_metrics.jsonl",
    "alerts": "model_registry/alerts.jsonl",
    # ... etc
}
```

## Running the Dashboard

### Local Development

```bash
streamlit run dashboards/app.py
```

The dashboard will open at `http://localhost:8501`

### Production Deployment

```bash
# Using Streamlit Cloud
# 1. Push code to GitHub
# 2. Connect to Streamlit Cloud: https://share.streamlit.io
# 3. Select your repository and main file

# Or use Docker:
docker build -t financial-risk-dashboard .
docker run -p 8501:8501 financial-risk-dashboard
```

## Data Integration

### Required Data Files

The dashboard expects the following files:

1. **markov_state_sequences.parquet** - Markov model regime sequences
   - Columns: `timestamp`, `regime`, `normal_prob`, `stress_prob`, `crisis_prob`

2. **performance_metrics.jsonl** - Performance metrics history
   - Fields: `timestamp`, `accuracy`, `precision`, `recall`, `f1_score`, `regime_count`, `data_quality`

3. **alerts.jsonl** - System alerts
   - Fields: `timestamp`, `alert_type`, `severity`, `message`

4. **anomalies.jsonl** - Detected anomalies
   - Fields: `timestamp`, `metric`, `method`, `value`, `threshold`

5. **degradation_events.jsonl** - Performance degradation events
   - Fields: `timestamp`, `metric`, `value_before`, `value_after`

6. **retraining_jobs.jsonl** - Retraining job history
   - Fields: `timestamp`, `status`, `trigger`, `model_version`

7. **ab_tests.jsonl** - A/B test results
   - Fields: `test_id`, `start_date`, `end_date`, `model_a`, `model_b`, `results`

8. **rollback_events.jsonl** - Model rollback events
   - Fields: `timestamp`, `from_version`, `to_version`, `reason`

## Features & Components

### Reusable Components

- **metric_card()** - Display metric with value, unit, and trend
- **gauge_metric()** - Display metric with progress gauge
- **alert_card()** - Display colored alert with details
- **status_badge()** - Display status badge
- **status_grid()** - Display status grid
- **health_indicator()** - Display health score

### Data Loaders

- `get_markov_state_data()` - Load regime sequences
- `get_performance_metrics()` - Load performance data
- `get_alerts()` - Load alerts
- `get_anomalies()` - Load anomalies
- `get_retraining_jobs()` - Load retraining history
- `get_ab_tests()` - Load A/B test results
- `get_rollback_events()` - Load rollback history

### Utilities

- **Formatters** - format_number, format_percentage, format_timestamp, format_duration
- **Validators** - validate_metric, detect_outliers, check_data_staleness
- **Constants** - Color schemes, thresholds, file paths

## Customization

### Adding New Pages

1. Create `dashboards/pages/new_page.py`:

```python
def render_new_page():
    """Render custom page."""
    from dashboards.components.header import render_header
    render_header("My Page Title")
    # Add your content here
```

2. Update `dashboards/utils/constants.py`:

```python
PAGES = {
    # ... existing pages
    "🆕 My Page": "new_page",
}
```

3. Add import and routing in `dashboards/app.py`:

```python
from dashboards.pages.new_page import render_new_page

# In main() function, add to routing logic:
elif selected_page == 'new_page':
    render_new_page()
```

### Changing Colors & Themes

Edit `dashboards/utils/constants.py`:

```python
REGIME_COLORS = {
    "Normal": "#2ecc71",      # Your color
    "Stress": "#f39c12",
    "Crisis": "#e74c3c",
}
```

## Performance Optimization

- Data caching with `@st.cache_data` (5-minute TTL)
- Lazy loading of pages
- Efficient DataFrame operations
- Plotly interactive charts with client-side rendering

## Troubleshooting

### Data Not Loading

1. Check data file paths in `dashboards/utils/constants.py`
2. Verify files exist and are readable
3. Check for permission issues
4. Look for parsing errors in console

### Charts Not Displaying

1. Verify required columns exist in data
2. Check for NaN/null values
3. Ensure data types are correct (timestamps, numbers)
4. Look for Plotly version compatibility

### Dashboard Slow to Load

1. Reduce data volume (filter date ranges)
2. Increase cache TTL
3. Use DataFrame sampling for visualization
4. Check system resources

## Troubleshooting Commands

```bash
# Check Streamlit version
streamlit --version

# Run with verbose logging
streamlit run dashboards/app.py --logger.level=debug

# Clear cache
streamlit cache clear

# Run with specific Python environment
python -m streamlit run dashboards/app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    app.py (Entry Point)                    │
│              Multi-Page Navigation & Routing                │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┬─────────────────┐
        ↓                   ↓                 ↓
    Sidebar          Header/Filters      Pages (5)
    (Navigation)     (Global Controls)   (Content)
        ↓                   ↓                 ↓
    PAGES dict      render_filters()   render_*()
                                              ↓
                            ┌─────────────────┴─────────────────┐
                            ↓                                   ↓
                      Components                            Utils
                    (Reusable UI)                      (Data & Format)
                            ↓                                   ↓
                    metric_card                        data_loader
                    alert_card                         formatters
                    status_grid                        validators
                    health_indicator                   constants
```

## Technology Stack

- **Framework**: Streamlit (interactive web app)
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **Date Handling**: Python datetime, pytz
- **Styling**: Streamlit custom CSS

## Performance Targets

- Page load time: < 3 seconds
- Chart rendering: < 1 second
- Data refresh: Every 30 seconds (configurable)
- Cache TTL: 5 minutes (configurable)

## Security Considerations

- No sensitive data stored locally
- Secrets in environment variables (Slack webhooks, etc.)
- Input validation on all user inputs
- XSS protection via Streamlit framework
- CORS configured for API calls

## Future Enhancements

- [ ] Real-time streaming updates (WebSocket)
- [ ] Advanced filtering (regex, date ranges)
- [ ] Export to PDF/Excel reports
- [ ] Custom alerts and notifications
- [ ] User authentication and RBAC
- [ ] Dashboard sharing and collaboration
- [ ] Performance benchmarking
- [ ] Advanced analytics (forecasting, etc.)

## Contributing

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and test
3. Submit pull request

## License

This project is part of the Financial Risk Markov MLOps system.

## Support

For issues, questions, or suggestions:
- Check [PHASE_11_DASHBOARD_SPECIFICATION.md](../PHASE_11_DASHBOARD_SPECIFICATION.md)
- Review [DOCUMENTATION_SUMMARY.md](../DOCUMENTATION_SUMMARY.md)
- Open GitHub issue
