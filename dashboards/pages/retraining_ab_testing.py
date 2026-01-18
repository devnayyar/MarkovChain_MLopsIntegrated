"""Retraining & A/B Testing page - Page 4."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dashboards.utils.data_loader import get_retraining_jobs, get_ab_tests, get_rollback_events
from dashboards.utils.formatters import format_timestamp, format_duration
from dashboards.components.header import render_header, render_quick_stats, render_filters
from dashboards.components.metrics_card import metric_row, metric_card
from dashboards.components.status_indicator import status_badge


def render_retraining_ab_testing():
    """Render the Retraining & A/B Testing page."""
    render_header("Retraining & A/B Testing")
    render_quick_stats()
    
    st.divider()
    
    filters = render_filters()
    
    st.divider()
    
    # Load data
    retraining_jobs = get_retraining_jobs()
    ab_tests = get_ab_tests()
    rollback_events = get_rollback_events()
    
    # Retraining Status
    st.subheader("🔄 Retraining Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Last Retraining", "2 hours ago", status="Info")
    
    with col2:
        metric_card("Next Scheduled", "In 5 days", status="Info")
    
    with col3:
        metric_card("Total Jobs", len(retraining_jobs), status="Good")
    
    with col4:
        metric_card("Success Rate", 94.5, unit="%", status="Good")
    
    st.divider()
    
    # Retraining Decision Criteria
    st.subheader("📊 Retraining Decision Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Scheduled Interval**")
        st.progress(3/7, text="3 / 7 days")
    
    with col2:
        st.write("**Drift Trigger**")
        st.progress(0.082/0.15, text="0.082 / 0.15")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Trigger**")
        st.progress(0.02/0.05, text="0.02 / 0.05")
    
    with col2:
        st.write("**Data Quality**")
        st.progress(96.5/100, text="96.5 / 100%")
    
    if st.button("🚀 Manual Retrain Now"):
        st.success("Retraining job started!")
    
    st.divider()
    
    # Retraining History
    st.subheader("📜 Retraining History")
    
    if not retraining_jobs.empty:
        if 'timestamp' in retraining_jobs.columns:
            history = retraining_jobs.sort_values('timestamp', ascending=False).head(10)
        else:
            history = retraining_jobs.head(10)
        
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("No retraining jobs recorded yet")
    
    st.divider()
    
    # A/B Test Results
    st.subheader("🧪 A/B Test Results")
    
    if not ab_tests.empty:
        st.success("A/B test currently active!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Model Performance", "92.3%", delta="-0.5%")
        
        with col2:
            st.metric("New Model Performance", "92.8%", delta="+0.5%")
        
        st.divider()
        
        # Model comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Metric**")
            st.write("Accuracy")
            st.write("Precision")
            st.write("Recall")
        
        with col2:
            st.write("**Current**")
            st.write("92.3%")
            st.write("89.1%")
            st.write("85.4%")
        
        with col3:
            st.write("**New**")
            st.write("92.8%")
            st.write("89.6%")
            st.write("86.1%")
        
        st.divider()
        
        # A/B test progress
        st.write("**A/B Test Progress**")
        st.progress(0.65, text="65% Complete - 6,500 / 10,000 requests")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence Level", "87.3%")
        
        with col2:
            st.metric("Statistical Significance", "Yes ✓")
        
        with col3:
            st.metric("Time Remaining", "~2 days")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Deploy New Model"):
                st.success("New model deployed!")
        
        with col2:
            if st.button("⏹️ Stop A/B Test"):
                st.info("A/B test stopped")
    else:
        st.info("No A/B tests currently active")
    
    st.divider()
    
    # Model Version Management
    st.subheader("📦 Model Version Management")
    
    model_versions = pd.DataFrame({
        'Version': ['v1.0', 'v1.1', 'v2.0', 'v2.1'],
        'Status': ['Archived', 'Archived', 'Active', 'Testing'],
        'Accuracy': [90.2, 91.1, 92.3, 92.8],
        'Created': ['2025-11-15', '2025-12-01', '2026-01-05', '2026-01-17'],
    })
    
    st.dataframe(model_versions, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Rollback Events
    st.subheader("🔙 Rollback History")
    
    if not rollback_events.empty:
        st.info(f"Total rollback events: {len(rollback_events)}")
        
        st.dataframe(rollback_events.head(10), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No rollback events - all deployments successful!")
    
    st.divider()
    
    # Job Queue
    st.subheader("⏳ Job Queue")
    
    queue_status = pd.DataFrame({
        'Job ID': ['JOB-001', 'JOB-002'],
        'Status': ['Running', 'Queued'],
        'Progress': ['75%', 'Waiting'],
        'Trigger': ['Scheduled', 'Drift Detected'],
    })
    
    st.dataframe(queue_status, use_container_width=True, hide_index=True)
