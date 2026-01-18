"""Data validation utilities."""

import pandas as pd
import numpy as np


def validate_data_range(df, column, min_val=None, max_val=None):
    """Check if data in column is within acceptable range."""
    if df is None or df.empty or column not in df.columns:
        return False, "No data"
    
    try:
        data = pd.to_numeric(df[column], errors='coerce')
        valid_data = data.dropna()
        
        if valid_data.empty:
            return False, "All values are NaN"
        
        if min_val is not None and (valid_data < min_val).any():
            return False, f"Values below minimum {min_val}"
        
        if max_val is not None and (valid_data > max_val).any():
            return False, f"Values above maximum {max_val}"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_metric(value, good_threshold, warning_threshold):
    """Validate metric value and return status."""
    if pd.isna(value):
        return "unknown", "No data"
    
    try:
        value = float(value)
        if value >= good_threshold:
            return "good", f"Value: {value:.3f} (Good)"
        elif value >= warning_threshold:
            return "warning", f"Value: {value:.3f} (Warning)"
        else:
            return "critical", f"Value: {value:.3f} (Critical)"
    except Exception as e:
        return "unknown", f"Validation error: {str(e)}"


def validate_data_completeness(df):
    """Check data completeness percentage."""
    if df is None or df.empty:
        return 0.0
    
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = df.count().sum()
    
    return (non_null_cells / total_cells) * 100 if total_cells > 0 else 0.0


def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method."""
    if data is None or data.empty or column not in data.columns:
        return []
    
    try:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index.tolist()
        return outlier_indices
    except Exception:
        return []


def validate_timestamp_order(df, timestamp_col):
    """Check if timestamps are in chronological order."""
    if df is None or df.empty or timestamp_col not in df.columns:
        return True
    
    try:
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        return timestamps.is_monotonic_increasing
    except Exception:
        return False


def check_data_staleness(last_update, max_age_hours=24):
    """Check if data is stale."""
    if pd.isna(last_update):
        return True
    
    try:
        last_update = pd.to_datetime(last_update)
        age = pd.Timestamp.now() - last_update
        return age.total_seconds() > (max_age_hours * 3600)
    except Exception:
        return True


def validate_drift_value(drift_value, threshold=0.15):
    """Validate drift value against threshold."""
    if pd.isna(drift_value):
        return "unknown"
    
    try:
        drift_value = float(drift_value)
        if drift_value < 0.05:
            return "normal"
        elif drift_value < threshold:
            return "warning"
        else:
            return "critical"
    except Exception:
        return "unknown"


def count_missing_values(df):
    """Count missing values in dataframe."""
    if df is None or df.empty:
        return {}
    
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / len(df)) * 100
    
    return {col: {"count": missing_counts[col], "pct": missing_pcts[col]} 
            for col in df.columns}
