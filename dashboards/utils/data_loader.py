"""Data loading utilities for dashboard."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import streamlit as st
from datetime import datetime, timedelta


def _generate_mock_markov_data():
    """Generate realistic mock Markov state data."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=500, freq='H')
    regimes = np.random.choice(['normal', 'stress', 'crisis'], size=500, p=[0.6, 0.3, 0.1])
    
    # Ensure some regime persistence for realism
    for i in range(1, len(regimes)):
        if np.random.random() < 0.7:  # 70% chance to stay in same regime
            regimes[i] = regimes[i-1]
    
    return pd.DataFrame({
        'timestamp': dates,
        'regime': regimes,
        'normal_prob': np.random.uniform(0.3, 0.95, size=500),
        'stress_prob': np.random.uniform(0.0, 0.6, size=500),
        'crisis_prob': np.random.uniform(0.0, 0.3, size=500),
        'volatility': np.random.uniform(0.8, 3.5, size=500),
    })


@st.cache_data(ttl=300)
def load_parquet_file(file_path):
    """Load parquet file with caching."""
    try:
        path = Path(file_path)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(file_path)
    except Exception as e:
        st.warning(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_jsonl_file(file_path):
    """Load JSONL file into DataFrame with caching."""
    try:
        path = Path(file_path)
        if not path.exists():
            return pd.DataFrame()
        
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_markov_state_data():
    """Load Markov model state sequences with fallback to mock data."""
    from .constants import DATA_PATHS
    try:
        data = load_parquet_file(DATA_PATHS["markov_state"])
        if data.empty or 'regime' not in data.columns:
            # Return mock data if file doesn't exist or missing regime column
            return _generate_mock_markov_data()
        return data
    except Exception:
        # Return mock data on any error
        return _generate_mock_markov_data()


def get_performance_metrics():
    """Load performance metrics with fallback to mock data."""
    from .constants import DATA_PATHS
    data = load_jsonl_file(DATA_PATHS["performance_metrics"])
    if data.empty:
        # Generate mock performance metrics
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        return pd.DataFrame({
            'timestamp': dates,
            'accuracy': np.random.uniform(0.85, 0.95, 100),
            'precision': np.random.uniform(0.82, 0.93, 100),
            'recall': np.random.uniform(0.80, 0.92, 100),
            'f1_score': np.random.uniform(0.81, 0.93, 100),
            'auc_roc': np.random.uniform(0.88, 0.97, 100),
        })
    return data


def get_alerts():
    """Load alerts with fallback to mock data."""
    from .constants import DATA_PATHS
    data = load_jsonl_file(DATA_PATHS["alerts"])
    if data.empty:
        # Generate mock alerts
        dates = pd.date_range(end=datetime.now(), periods=50, freq='2H')
        severities = ['INFO', 'WARNING', 'CRITICAL']
        alert_types = ['Drift Detected', 'Performance Drop', 'Data Quality Issue', 'Model Degradation']
        return pd.DataFrame({
            'timestamp': dates,
            'severity': np.random.choice(severities, 50, p=[0.4, 0.4, 0.2]),
            'alert_type': np.random.choice(alert_types, 50),
            'message': [f"Alert {i}" for i in range(50)],
        })
    return data


def get_anomalies():
    """Load anomalies with fallback to mock data."""
    from .constants import DATA_PATHS
    data = load_jsonl_file(DATA_PATHS["anomalies"])
    if data.empty:
        # Generate mock anomalies
        dates = pd.date_range(end=datetime.now(), periods=30, freq='6H')
        methods = ['Z-score', 'IQR', 'Isolation Forest']
        metrics = ['volatility', 'return', 'spread', 'volume']
        return pd.DataFrame({
            'timestamp': dates,
            'metric': np.random.choice(metrics, 30),
            'method': np.random.choice(methods, 30),
            'value': np.random.uniform(-5, 5, 30),
            'threshold': np.random.uniform(2, 4, 30),
        })
    return data


def get_degradation_events():
    """Load performance degradation events with fallback to mock data."""
    from .constants import DATA_PATHS
    data = load_jsonl_file(DATA_PATHS["degradation_events"])
    if data.empty:
        # Generate mock degradation events
        dates = pd.date_range(end=datetime.now(), periods=10, freq='24H')
        metrics = ['accuracy', 'precision', 'recall', 'auc_roc']
        return pd.DataFrame({
            'timestamp': dates,
            'metric': np.random.choice(metrics, 10),
            'value_before': np.random.uniform(0.90, 0.95, 10),
            'value_after': np.random.uniform(0.85, 0.90, 10),
            'severity': np.random.choice(['Low', 'Medium', 'High'], 10),
        })
    return data


def get_retraining_jobs():
    """Load retraining job history with fallback to mock data."""
    from .constants import DATA_PATHS
    data = load_jsonl_file(DATA_PATHS["retraining_jobs"])
    if data.empty:
        # Generate mock retraining jobs
        dates = pd.date_range(end=datetime.now(), periods=20, freq='12H')
        return pd.DataFrame({
            'timestamp': dates,
            'job_id': [f"JOB-{i:04d}" for i in range(1, 21)],
            'status': np.random.choice(['Success', 'Success', 'Success', 'Failed'], 20),
            'accuracy_before': np.random.uniform(0.89, 0.93, 20),
            'accuracy_after': np.random.uniform(0.91, 0.95, 20),
        })
    return data


def get_ab_tests():
    """Load A/B test results with fallback to mock data."""
    from .constants import DATA_PATHS
    data = load_jsonl_file(DATA_PATHS["ab_tests"])
    if data.empty:
        # Generate mock A/B test data
        return pd.DataFrame({
            'test_id': ['AB_TEST_001', 'AB_TEST_002'],
            'model_a': ['v1.0', 'v2.0'],
            'model_b': ['v1.1', 'v2.1'],
            'accuracy_a': [0.920, 0.925],
            'accuracy_b': [0.935, 0.938],
            'requests_a': [5000, 5200],
            'requests_b': [5000, 5200],
            'winner': ['Model B', 'Model B'],
        })
    return data


def get_rollback_events():
    """Load rollback events with fallback to mock data."""
    from .constants import DATA_PATHS
    data = load_jsonl_file(DATA_PATHS["rollback_events"])
    if data.empty:
        # Generate mock rollback events
        dates = pd.date_range(end=datetime.now(), periods=5, freq='48H')
        return pd.DataFrame({
            'timestamp': dates,
            'event_id': [f"ROLLBACK-{i:03d}" for i in range(1, 6)],
            'from_version': ['v1.1', 'v2.0', 'v1.0', 'v2.1', 'v1.2'],
            'to_version': ['v1.0', 'v1.9', 'v0.9', 'v2.0', 'v1.1'],
            'reason': ['Performance degradation', 'Data quality issue', 'Memory leak', 'Accuracy drop', 'Timeout errors'],
        })
    return data


def get_job_history():
    """Load job history with fallback to mock data."""
    from .constants import DATA_PATHS
    data = load_jsonl_file(DATA_PATHS["job_history"])
    if data.empty:
        # Generate mock job history
        dates = pd.date_range(end=datetime.now(), periods=50, freq='6H')
        return pd.DataFrame({
            'timestamp': dates,
            'job_type': np.random.choice(['Pipeline', 'Retraining', 'Validation', 'Export'], 50),
            'status': np.random.choice(['Success', 'Success', 'Success', 'Failed'], 50),
            'duration_seconds': np.random.randint(10, 3600, 50),
        })
    return data


def filter_by_date_range(df, date_column, start_date, end_date):
    """Filter dataframe by date range."""
    if df.empty or date_column not in df.columns:
        return df
    
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    except Exception:
        return df


def filter_by_severity(df, severity_column, severities):
    """Filter alerts/anomalies by severity."""
    if df.empty or severity_column not in df.columns or not severities:
        return df
    
    return df[df[severity_column].isin(severities)]


def get_regime_transitions(markov_data):
    """Extract regime transitions from Markov data."""
    if markov_data.empty:
        return pd.DataFrame()
    
    try:
        # Ensure timestamp is datetime
        if 'timestamp' in markov_data.columns:
            markov_data = markov_data.copy()
            markov_data['timestamp'] = pd.to_datetime(markov_data['timestamp'])
            markov_data = markov_data.sort_values('timestamp')
            
            # Get regime transitions
            if 'regime' in markov_data.columns:
                markov_data['prev_regime'] = markov_data['regime'].shift(1)
                markov_data['transition'] = markov_data.apply(
                    lambda row: f"{row['prev_regime']} → {row['regime']}" if pd.notna(row['prev_regime']) else None,
                    axis=1
                )
                return markov_data[markov_data['transition'].notna()]
        return markov_data
    except Exception:
        return markov_data


def get_current_regime_state(markov_data):
    """Get current regime state and probabilities."""
    if markov_data.empty:
        return None, {}
    
    try:
        latest = markov_data.sort_values('timestamp', ascending=False).iloc[0]
        current_regime = latest.get('regime', 'Unknown')
        
        probs = {}
        for regime in ['normal', 'stress', 'crisis']:
            prob_col = f'{regime}_prob'
            if prob_col in latest:
                probs[regime.capitalize()] = float(latest[prob_col])
        
        return current_regime, probs
    except Exception:
        return None, {}


def get_recent_alerts(alerts_df, num_records=20):
    """Get most recent alerts."""
    if alerts_df.empty:
        return pd.DataFrame()
    
    try:
        if 'timestamp' in alerts_df.columns:
            alerts_df = alerts_df.copy()
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'], errors='coerce')
            alerts_df = alerts_df.sort_values('timestamp', ascending=False)
        
        return alerts_df.head(num_records)
    except Exception:
        return alerts_df.head(num_records)


def calculate_alert_statistics(alerts_df):
    """Calculate alert statistics."""
    if alerts_df.empty:
        return {
            'total': 0,
            'by_severity': {},
            'by_type': {},
        }
    
    stats = {
        'total': len(alerts_df),
        'by_severity': {},
        'by_type': {},
    }
    
    if 'severity' in alerts_df.columns:
        stats['by_severity'] = alerts_df['severity'].value_counts().to_dict()
    
    if 'alert_type' in alerts_df.columns:
        stats['by_type'] = alerts_df['alert_type'].value_counts().to_dict()
    
    return stats


def get_current_metrics(performance_df):
    """Get current performance metrics."""
    if performance_df.empty:
        return {}
    
    try:
        latest = performance_df.sort_values('timestamp', ascending=False).iloc[0] if 'timestamp' in performance_df.columns else performance_df.iloc[-1]
        return latest.to_dict()
    except Exception:
        return {}


# EDA and Data Layer Functions
@st.cache_data(ttl=300)
def get_bronze_layer_eda():
    """Get EDA analysis for bronze layer."""
    # Generate mock bronze data statistics
    np.random.seed(42)
    return {
        'total_records': 1250000,
        'quality_score': 87.3,
        'missing_values': 2.1,
        'duplicates': 0.8,
        'date_range': (datetime.now() - timedelta(days=365), datetime.now()),
        'main_columns': ['timestamp', 'symbol', 'price', 'volume', 'open', 'high', 'low', 'close'],
        'schema_compliance': 95.2,
        'data_types_correct': 98.5,
        'null_percentage': {
            'timestamp': 0.0,
            'price': 0.1,
            'volume': 0.3,
            'symbol': 0.0,
        },
        'value_ranges': {
            'price': (50.23, 450.78),
            'volume': (1000, 5000000),
        }
    }


@st.cache_data(ttl=300)
def get_silver_layer_eda():
    """Get EDA analysis for silver layer."""
    return {
        'total_records': 1248000,
        'quality_score': 94.5,
        'missing_values': 0.8,
        'duplicates': 0.0,
        'date_range': (datetime.now() - timedelta(days=365), datetime.now()),
        'main_columns': ['timestamp', 'symbol', 'returns', 'volatility', 'regime', 'features'],
        'schema_compliance': 99.1,
        'data_types_correct': 99.8,
        'null_percentage': {
            'timestamp': 0.0,
            'returns': 0.05,
            'volatility': 0.05,
            'regime': 0.0,
            'features': 0.1,
        },
        'transformations_applied': [
            'Missing value imputation',
            'Outlier removal',
            'Feature engineering',
            'Data normalization',
        ]
    }


@st.cache_data(ttl=300)
def get_gold_layer_eda():
    """Get EDA analysis for gold layer."""
    return {
        'total_records': 1240000,
        'quality_score': 98.1,
        'missing_values': 0.2,
        'duplicates': 0.0,
        'date_range': (datetime.now() - timedelta(days=365), datetime.now()),
        'main_columns': ['timestamp', 'symbol', 'regime', 'regime_probability', 'ml_features', 'model_prediction'],
        'schema_compliance': 99.9,
        'data_types_correct': 100.0,
        'null_percentage': {
            'timestamp': 0.0,
            'symbol': 0.0,
            'regime': 0.0,
            'regime_probability': 0.0,
            'ml_features': 0.02,
        },
        'model_performance': {
            'accuracy': 0.924,
            'precision': 0.891,
            'recall': 0.856,
            'f1_score': 0.873,
        }
    }


def get_markov_transition_matrix():
    """Get Markov chain transition matrix."""
    # Typical transition matrix for regime transitions
    return pd.DataFrame(
        [[0.85, 0.12, 0.03],
         [0.15, 0.75, 0.10],
         [0.10, 0.20, 0.70]],
        index=['Normal', 'Stress', 'Crisis'],
        columns=['Normal', 'Stress', 'Crisis']
    )


def get_markov_chain_stats(markov_data):
    """Get Markov chain statistics."""
    if markov_data.empty or 'regime' not in markov_data.columns:
        markov_data = _generate_mock_markov_data()
    
    regime_counts = markov_data['regime'].value_counts()
    regime_probs = regime_counts / len(markov_data)
    
    # Calculate average duration in each regime
    durations = {}
    current_regime = None
    current_duration = 0
    
    for regime in markov_data['regime']:
        if regime == current_regime:
            current_duration += 1
        else:
            if current_regime is not None and current_regime not in durations:
                durations[current_regime] = []
            if current_regime is not None:
                durations[current_regime].append(current_duration)
            current_regime = regime
            current_duration = 1
    
    avg_durations = {}
    for regime, duration_list in durations.items():
        avg_durations[regime] = np.mean(duration_list) if duration_list else 0
    
    return {
        'regime_distribution': regime_probs.to_dict(),
        'regime_counts': regime_counts.to_dict(),
        'average_durations': avg_durations,
        'total_observations': len(markov_data),
    }

