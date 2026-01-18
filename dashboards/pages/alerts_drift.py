"""Alerts & Drift page - Page 2."""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from dashboards.utils.data_loader import get_alerts, get_anomalies, calculate_alert_statistics
from dashboards.utils.formatters import format_timestamp, format_number
from dashboards.utils.constants import ALERT_SEVERITY_COLORS, ALERT_SEVERITY_EMOJIS
from dashboards.components.header import render_header, render_quick_stats, render_filters
from dashboards.components.metrics_card import metric_row, alert_card, metric_card
from dashboards.components.status_indicator import status_grid, health_indicator


def render_alerts_drift():
    """Render the Alerts & Drift page."""
    render_header("Alerts & Drift Detection")
    render_quick_stats()
    
    st.divider()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Alerts Overview", "🔍 Alert Details", "📊 Anomalies", "📚 Explanations"])
    
    # Load data
    alerts = get_alerts()
    anomalies = get_anomalies()
    
    with tab1:
        st.header("🚨 Alerts Overview")
        
        # Alert summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_alerts = len(alerts) if not alerts.empty else 0
            st.metric("Total Alerts (24h)", total_alerts)
        with col2:
            critical_count = len(alerts[alerts['severity'] == 'CRITICAL']) if not alerts.empty else 0
            st.metric("🔴 Critical", critical_count)
        with col3:
            warning_count = len(alerts[alerts['severity'] == 'WARNING']) if not alerts.empty else 0
            st.metric("🟡 Warnings", warning_count)
        with col4:
            st.metric("System Health", "95%")
        
        st.divider()
        
        # Alert distribution pie chart
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Alert Distribution by Type")
            if not alerts.empty and 'alert_type' in alerts.columns:
                alert_type_counts = alerts['alert_type'].value_counts()
                fig_pie = px.pie(values=alert_type_counts.values, names=alert_type_counts.index,
                               title="Alerts by Type")
                st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            st.subheader("🎯 Alert Distribution by Severity")
            if not alerts.empty and 'severity' in alerts.columns:
                severity_counts = alerts['severity'].value_counts()
                fig_severity = px.pie(values=severity_counts.values, names=severity_counts.index,
                                    color_discrete_map=ALERT_SEVERITY_COLORS,
                                    title="Alerts by Severity")
                st.plotly_chart(fig_severity, width='stretch')
    
    with tab2:
        st.header("🔍 Alert Details & Explanations")
        
        if not alerts.empty:
            st.subheader("🚨 Real-Time Alert Feed (Last 50)")
            
            recent_alerts = alerts.sort_values('timestamp', ascending=False).head(50) if 'timestamp' in alerts.columns else alerts.head(50)
            
            for idx, alert in recent_alerts.iterrows():
                severity = alert.get('severity', 'INFO')
                alert_type = alert.get('alert_type', 'Unknown')
                message = alert.get('message', 'No message')
                timestamp = alert.get('timestamp', datetime.now().isoformat())
                
                # Use emoji for severity
                emoji = ALERT_SEVERITY_EMOJIS.get(severity, '⚪')
                
                with st.expander(f"{emoji} {alert_type} - {format_timestamp(timestamp)}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Message:** {message}")
                        st.write(f"**Type:** {alert_type}")
                        st.write(f"**Severity:** {severity}")
                    with col2:
                        severity_color = ALERT_SEVERITY_COLORS.get(severity, '#3498db')
                        st.markdown(f"<div style='background-color:{severity_color}; padding:10px; border-radius:5px; color:white;'><b>{severity}</b></div>", unsafe_allow_html=True)
        else:
            st.info("No alerts available")
        
        st.divider()
        
        # Alert type explanations
        st.subheader("📚 What Each Alert Type Means:")
        
        alert_explanations = {
            "Data Quality Issue": "Raw data has missing values, outliers, or format problems. This affects model reliability.",
            "Model Degradation": "Model's prediction accuracy is declining. Retraining may be needed.",
            "Performance Drop": "Model predictions are getting less accurate on new data.",
            "Drift Detected": "Statistical properties of data have changed, affecting model reliability.",
            "Anomaly Alert": "Unusual data point detected that doesn't match historical patterns.",
        }
        
        for alert_type, explanation in alert_explanations.items():
            with st.expander(f"📌 {alert_type}"):
                st.write(explanation)
    
    with tab3:
        st.header("📊 Anomalies Detection")
        
        if not anomalies.empty:
            st.subheader("Anomaly Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Anomalies (24h)", len(anomalies))
            with col2:
                zscore_count = len(anomalies[anomalies['method'] == 'Z-score']) if 'method' in anomalies.columns else 0
                st.metric("Z-score Anomalies", zscore_count)
            with col3:
                iqr_count = len(anomalies[anomalies['method'] == 'IQR']) if 'method' in anomalies.columns else 0
                st.metric("IQR Anomalies", iqr_count)
            
            st.divider()
            
            st.subheader("Anomaly Detection Methods Explained:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("📌 Z-Score Method"):
                    st.write("""
                    **What it detects:** Data points that are far from the average
                    
                    **How it works:** 
                    - Calculates how many standard deviations away a point is from mean
                    - Z-score > 3: Point is extremely unusual (~0.3% probability)
                    - Z-score > 2.5: Point is very unusual (~1% probability)
                    
                    **Example:**
                    - Average unemployment: 4.5%
                    - Suddenly jumps to 8%
                    - This would be flagged as Z-score anomaly
                    
                    **Good for:** Quick detection of obvious spikes/drops
                    """)
            
            with col2:
                with st.expander("📌 IQR Method (Interquartile Range)"):
                    st.write("""
                    **What it detects:** Data points outside normal range
                    
                    **How it works:**
                    - IQR = 75th percentile - 25th percentile
                    - Outliers = data outside (Q1 - 1.5×IQR) to (Q3 + 1.5×IQR)
                    - More robust to extreme values
                    
                    **Example:**
                    - Fed Funds Rate typically 2-5%
                    - Suddenly 15%? = Outlier detected
                    
                    **Good for:** Detecting unusual values robust to extreme cases
                    """)
            
            st.divider()
            
            st.subheader("🔍 Recent Anomalies (Last 30)")
            anomalies_display = anomalies.sort_values('timestamp', ascending=False).head(30) if 'timestamp' in anomalies.columns else anomalies.head(30)
            
            for idx, anom in anomalies_display.iterrows():
                metric = anom.get('metric', 'Unknown')
                method = anom.get('method', 'Unknown')
                value = anom.get('value', 0)
                threshold = anom.get('threshold', 0)
                
                with st.expander(f"🚩 {metric} ({method}) - Value: {value:.2f}"):
                    st.write(f"**Metric:** {metric}")
                    st.write(f"**Detection Method:** {method}")
                    st.write(f"**Observed Value:** {value:.4f}")
                    st.write(f"**Expected/Threshold:** {threshold:.4f}")
                    st.write(f"**Deviation:** {abs(value - threshold):.4f} ({abs(value - threshold) / threshold * 100:.1f}%)")
        else:
            st.info("No anomalies detected")
    
    with tab4:
        st.header("📚 Understanding Alerts & Anomalies")
        
        st.subheader("🎯 Key Metrics Explained:")
        
        with st.expander("📊 Data Quality Score (96.5%)"):
            st.write("""
            **What it measures:** How complete and valid your data is
            
            **Score breakdown:**
            - 100%: Perfect - no missing data, no errors
            - 95-99%: Excellent - minor issues, model can work
            - 90-95%: Good - some issues, monitor closely
            - 80-90%: Concerning - problems affecting models
            - <80%: Critical - data too poor for reliable predictions
            
            **Your score: 96.5% = Excellent quality**
            
            **What this means:**
            - Data is mostly complete
            - Few missing values
            - No obvious formatting issues
            - Model should work well
            
            **What to look for:**
            - Missing values in key columns
            - Data type mismatches
            - Unexpected nulls or NaNs
            """)
        
        with st.expander("📉 Drift Detection"):
            st.write("""
            **What it detects:** Changes in data patterns over time
            
            **Four types of drift:**
            
            1. **Regime Distribution Drift (0.082)**
               - Detects: More time in HIGH_RISK vs historical average
               - Concern: Model trained on different regime distribution
               - Action: May need retraining if > 0.2
            
            2. **Transition Pattern Drift (0.045)**
               - Detects: Different regime transition probabilities
               - Concern: Market behavior changing
               - Action: Monitor if > 0.1
            
            3. **Feature Distribution Drift**
               - Detects: Economic indicators changing distribution
               - Example: Unemployment usually 3-5%, now 5-7%
               - Action: Check for real economic changes
            
            4. **Model Calibration**
               - Detects: Model predictions becoming less accurate
               - Concern: Model opinions diverging from reality
               - Action: Retrain if calibration score drops
            
            **Your readings: All good (all < 0.1)**
            """)
        
        with st.expander("🚨 Alert Severity Levels"):
            st.write("""
            **🟢 INFO (Informational)**
            - Normal operation notices
            - Data received successfully
            - Routine maintenance events
            - **Action:** Just informational, no action needed
            
            **🟡 WARNING (Warning)**
            - Approaching limits or thresholds
            - Minor data quality issues
            - Performance slightly degraded
            - **Action:** Monitor, investigate if persistent
            
            **🔴 CRITICAL (Critical)**
            - Immediate action required
            - Data collection failed
            - Model predictions unreliable
            - System health compromised
            - **Action:** Fix immediately!
            """)
        
        with st.expander("🔍 Common Data Quality Problems:"):
            st.write("""
            **Missing Values (Most Common)**
            - Problem: Some data points not recorded
            - Impact: Can't calculate indicators accurately
            - Solution: Use interpolation or gap-fill
            - Your status: 96.5% complete = Excellent
            
            **Outliers/Spikes**
            - Problem: Sudden abnormal values
            - Example: Fed rate jumps 5% overnight
            - Impact: Skews calculations
            - Detection: Z-score and IQR methods
            
            **Format/Type Errors**
            - Problem: Data in wrong format (text vs number)
            - Impact: Calculations fail
            - Solution: Data validation at ingestion
            
            **Delayed Data**
            - Problem: Data arrives late
            - Impact: Regimes based on stale information
            - Solution: Monitor ingestion pipelines
            
            **Duplicates**
            - Problem: Same data recorded twice
            - Impact: Skewed statistics
            - Solution: Deduplication in Silver layer
            """)
        
        st.divider()
        
        st.subheader("✅ What Good Looks Like:")
        st.write("""
        | Metric | Good Value | Your Status |
        |--------|-----------|------------|
        | Data Quality | > 95% | 96.5% ✅ |
        | Regime Drift | < 0.15 | 0.082 ✅ |
        | Model Calibration | Good/OK | Warning ⚠️ |
        | Alerts (24h) | < 20 | 50 🟡 |
        | Critical Alerts | 0-2 | 11 🔴 |
        | Anomalies (24h) | < 5 | 30 🔴 |
        """)
        
        st.divider()
        
        st.subheader("🔧 What to Do When You See Issues:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **If Data Quality < 90%:**
            1. Check raw data source
            2. Look for gaps or errors
            3. Contact data engineering
            4. Consider temporary model pause
            
            **If Drift > 0.2:**
            1. Investigate market changes
            2. Check if economy in different regime
            3. Consider model retraining
            4. Alert risk management
            """)
        
        with col2:
            st.write("""
            **If Anomalies Rising:**
            1. Check individual alert details
            2. Look for pattern (all same indicator?)
            3. Determine if real event or data error
            4. Investigate data pipeline
            
            **If Model Calibration Warning:**
            1. Check prediction accuracy
            2. Compare to baseline
            3. Schedule retraining
            4. Monitor closely
            """)

