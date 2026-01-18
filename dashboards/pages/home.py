"""Home/Dashboard overview page."""

import streamlit as st
from dashboards.utils.data_loader import (
    get_markov_state_data, get_performance_metrics, get_alerts, get_anomalies
)
from dashboards.utils.constants import PAGES


def navigate_to(page_key):
    """Navigate to a specific page by updating session state."""
    st.session_state.current_page = page_key
    st.rerun()


def render_home():
    """Render the home/dashboard page."""
    st.title("📊 Dashboard Overview")
    
    # Quick Navigation
    st.subheader("Quick Navigation")
    nav_cols = st.columns(4)
    page_items = list(PAGES.items())
    
    for idx, (label, page_key) in enumerate(page_items):
        col = nav_cols[idx % 4]
        with col:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                navigate_to(page_key)
    
    st.divider()
    
    # Key Metrics Overview
    st.subheader("📈 Key Metrics")
    
    try:
        markov_data = get_markov_state_data()
        perf_metrics = get_performance_metrics()
        alerts = get_alerts()
        anomalies = get_anomalies()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_regime = "Unknown"
            if not markov_data.empty and 'regime' in markov_data.columns:
                current_regime = str(markov_data['regime'].iloc[-1]).upper()
            st.metric("Current Regime", current_regime)
        
        with col2:
            accuracy = "N/A"
            if not perf_metrics.empty and 'accuracy' in perf_metrics.columns:
                acc_val = perf_metrics['accuracy'].iloc[-1]
                accuracy = f"{float(acc_val):.1%}" if isinstance(acc_val, (int, float)) else str(acc_val)
            st.metric("Model Accuracy", accuracy)
        
        with col3:
            alert_count = len(alerts) if not alerts.empty else 0
            st.metric("Active Alerts", alert_count)
        
        with col4:
            anomaly_count = len(anomalies) if not anomalies.empty else 0
            st.metric("Anomalies Detected", anomaly_count)
    
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
    
    st.divider()
    
    # Recent Data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Recent Alerts")
        try:
            alerts_df = get_alerts()
            if not alerts_df.empty:
                display_cols = [col for col in ['timestamp', 'type', 'severity', 'message'] if col in alerts_df.columns]
                st.dataframe(
                    alerts_df[display_cols].head(5),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No recent alerts")
        except Exception as e:
            st.warning(f"Could not load alerts: {str(e)}")
    
    with col2:
        st.subheader("🔍 Anomalies Detected")
        try:
            anomalies_df = get_anomalies()
            if not anomalies_df.empty:
                display_cols = [col for col in ['timestamp', 'type', 'score'] if col in anomalies_df.columns]
                st.dataframe(
                    anomalies_df[display_cols].head(5),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No anomalies detected")
        except Exception as e:
            st.warning(f"Could not load anomalies: {str(e)}")
    
    st.divider()
    
    # Help section for new users
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### 📚 About This Dashboard")
    with col2:
        if st.button("❓ Need Help?", key="help_overview"):
            with st.popover("Dashboard Help"):
                st.markdown("""
                **Welcome to the Financial Risk Dashboard!**
                
                This dashboard helps you monitor market regimes and AI model performance.
                
                **Key Features:**
                - 📈 Real-time regime detection
                - 🤖 Model performance tracking
                - 🚨 Alert management
                - 📊 Data quality monitoring
                - 🔄 Model retraining oversight
                """)


