"""Header component for dashboard."""

import streamlit as st
from datetime import datetime, timedelta
from dashboards.utils.constants import TIME_RANGES, REFRESH_INTERVALS


def render_header(title="Dashboard"):
    """Render the main header with title and quick info."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.title(f"📊 {title}")
    
    with col2:
        st.metric("System Status", "✅ Healthy")
    
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    with col4:
        st.markdown("")  # Spacing


def render_quick_stats():
    """Render quick stats bar at top of page with better spacing."""
    from dashboards.utils.data_loader import get_alerts, calculate_alert_statistics
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    alerts = get_alerts()
    alert_stats = calculate_alert_statistics(alerts)
    
    critical_count = alert_stats['by_severity'].get('CRITICAL', 0)
    warning_count = alert_stats['by_severity'].get('WARNING', 0)
    
    with col1:
        st.metric(
            "Total Alerts (24h)",
            alert_stats['total'],
            help="Number of alerts triggered in the last 24 hours"
        )
    
    with col2:
        st.metric(
            "🔴 Critical",
            critical_count,
            help="Critical alerts require immediate attention"
        )
    
    with col3:
        st.metric(
            "🟡 Warnings",
            warning_count,
            help="Warnings indicate issues that should be reviewed"
        )
    
    with col4:
        st.metric(
            "System Health",
            "95%",
            help="Overall system operational status"
        )


def render_filters():
    """Render global filter controls with better spacing."""
    st.markdown("### ⚙️ Filter Options")
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        timeframe = st.selectbox(
            "Timeframe",
            list(TIME_RANGES.keys()),
            key="global_timeframe",
            help="Select the time period to analyze"
        )
    
    with col2:
        date_range = st.date_input(
            "Date Range",
            [datetime.now() - timedelta(days=30), datetime.now()],
            key="global_date_range",
            help="Choose start and end dates for analysis"
        )
    
    with col3:
        refresh_interval = st.selectbox(
            "Auto-Refresh",
            list(REFRESH_INTERVALS.keys()),
            key="global_refresh",
            help="Set how often data updates automatically"
        )
    
    with col4:
        show_advanced = st.checkbox(
            "Advanced Filters",
            key="show_advanced_filters",
            help="Enable additional filtering options"
        )
    
    return {
        'timeframe': timeframe,
        'date_range': date_range,
        'refresh_interval': refresh_interval,
        'show_advanced': show_advanced,
    }

