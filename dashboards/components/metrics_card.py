"""Reusable metric card component."""

import streamlit as st
import pandas as pd
from dashboards.utils.formatters import format_number, get_trend_arrow
from dashboards.utils.constants import STATUS_COLORS


def metric_card(title, value, unit="", previous_value=None, status="Info"):
    """
    Display a metric card with value, unit, and optional trend.
    
    Args:
        title: Metric title
        value: Current value
        unit: Unit of measurement
        previous_value: Previous value for trend calculation
        status: Status level (Good, Warning, Critical, Info)
    """
    color = STATUS_COLORS.get(status, STATUS_COLORS["Info"])
    
    # Format the main value
    formatted_value = format_number(value, decimals=3)
    
    # Calculate trend
    trend_arrow = "→"
    trend_text = ""
    if previous_value is not None and not pd.isna(previous_value):
        trend_arrow, trend_text = get_trend_arrow(value, previous_value)
    
    # Create card HTML
    card_html = f"""
    <div style="
        background-color: {color};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    ">
        <p style="margin: 0; color: white; font-size: 14px; opacity: 0.9;">{title}</p>
        <h2 style="margin: 10px 0 0 0; color: white; font-size: 36px;">{formatted_value}</h2>
        <p style="margin: 5px 0 0 0; color: white; font-size: 12px; opacity: 0.8;">
            {unit} {trend_arrow} {trend_text}
        </p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def metric_row(*metrics):
    """
    Display multiple metrics in a row.
    
    Args:
        *metrics: Variable number of (title, value, unit, previous, status) tuples
    """
    cols = st.columns(len(metrics))
    
    for col, metric_data in zip(cols, metrics):
        with col:
            if len(metric_data) == 5:
                title, value, unit, previous, status = metric_data
                metric_card(title, value, unit, previous, status)
            elif len(metric_data) == 4:
                title, value, unit, status = metric_data
                metric_card(title, value, unit, status=status)
            else:
                title, value, unit = metric_data
                metric_card(title, value, unit)


def gauge_metric(title, value, min_val=0, max_val=100, threshold_warning=70, threshold_critical=50):
    """
    Display a metric with gauge-style indicator.
    
    Args:
        title: Metric title
        value: Current value
        min_val: Minimum value for gauge
        max_val: Maximum value for gauge
        threshold_warning: Warning threshold
        threshold_critical: Critical threshold
    """
    # Determine status
    if value >= threshold_warning:
        status = "Good"
    elif value >= threshold_critical:
        status = "Warning"
    else:
        status = "Critical"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create gauge visualization
        progress = max(0, min(1, (value - min_val) / (max_val - min_val)))
        st.progress(progress)
    
    with col2:
        st.metric(title, f"{format_number(value, 1)}%", label_visibility="hidden")


def alert_card(timestamp, alert_type, message, severity="INFO", details=None):
    """
    Display an alert card.
    
    Args:
        timestamp: Alert timestamp
        alert_type: Type of alert (Drift, Performance, Anomaly, etc.)
        message: Alert message
        severity: Severity level (INFO, WARNING, CRITICAL)
        details: Optional additional details
    """
    from dashboards.utils.constants import ALERT_SEVERITY_COLORS, ALERT_SEVERITY_EMOJIS
    
    color = ALERT_SEVERITY_COLORS.get(severity, ALERT_SEVERITY_COLORS["INFO"])
    emoji = ALERT_SEVERITY_EMOJIS.get(severity, "🟢")
    
    card_html = f"""
    <div style="
        background-color: {color};
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid #333;
    ">
        <div style="color: white; display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: bold;">{emoji} {alert_type}</span>
            <span style="font-size: 12px; opacity: 0.8;">{timestamp}</span>
        </div>
        <p style="color: white; margin: 0; font-size: 14px;">{message}</p>
        {f'<p style="color: white; margin: 8px 0 0 0; font-size: 12px; opacity: 0.7;">{details}</p>' if details else ''}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
