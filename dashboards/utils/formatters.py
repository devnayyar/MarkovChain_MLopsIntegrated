"""Formatting utilities for dashboard display."""

from datetime import datetime
import pandas as pd


def format_number(value, decimals=2):
    """Format a number with specified decimal places."""
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value, decimals=1):
    """Format a number as percentage."""
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)


def format_timestamp(ts):
    """Format timestamp for display."""
    if pd.isna(ts):
        return "N/A"
    try:
        if isinstance(ts, str):
            dt = pd.to_datetime(ts)
        else:
            dt = pd.Timestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def format_date(date_val):
    """Format date for display."""
    if pd.isna(date_val):
        return "N/A"
    try:
        if isinstance(date_val, str):
            dt = pd.to_datetime(date_val)
        else:
            dt = pd.Timestamp(date_val)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(date_val)


def format_duration(seconds):
    """Format duration in human-readable format."""
    if pd.isna(seconds) or seconds is None:
        return "N/A"
    try:
        seconds = int(float(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    except Exception:
        return str(seconds)


def get_trend_arrow(current, previous):
    """Get trend arrow and percentage change."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "→", "N/A"
    
    try:
        current = float(current)
        previous = float(previous)
        
        if current > previous:
            pct_change = ((current - previous) / abs(previous)) * 100
            return "📈", f"+{pct_change:.1f}%"
        elif current < previous:
            pct_change = ((previous - current) / abs(previous)) * 100
            return "📉", f"-{pct_change:.1f}%"
        else:
            return "→", "0.0%"
    except Exception:
        return "→", "N/A"


def get_status_color(value, good_threshold, warning_threshold):
    """Get color based on value thresholds."""
    if pd.isna(value):
        return "#95a5a6"  # Gray
    
    try:
        value = float(value)
        if value >= good_threshold:
            return "#27ae60"  # Green
        elif value >= warning_threshold:
            return "#e67e22"  # Orange
        else:
            return "#c0392b"  # Red
    except Exception:
        return "#95a5a6"  # Gray


def format_metric_row(name, value, unit="", decimals=2):
    """Format a metric for display in row format."""
    formatted_value = format_number(value, decimals)
    return f"{name}: {formatted_value} {unit}".strip()


def truncate_text(text, max_length=100):
    """Truncate text to maximum length with ellipsis."""
    if pd.isna(text):
        return "N/A"
    text = str(text)
    if len(text) > max_length:
        return text[:max_length - 3] + "..."
    return text


def format_regime_name(regime):
    """Format regime name for display."""
    regime_map = {
        "normal": "Normal",
        "stress": "Stress",
        "crisis": "Crisis",
    }
    if isinstance(regime, str):
        return regime_map.get(regime.lower(), regime)
    return str(regime)
