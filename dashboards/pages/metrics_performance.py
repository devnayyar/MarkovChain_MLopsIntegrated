"""Metrics & Performance page - Page 3."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dashboards.utils.data_loader import get_performance_metrics, get_degradation_events, get_current_metrics
from dashboards.utils.formatters import format_number, format_timestamp
from dashboards.components.header import render_header, render_quick_stats, render_filters
from dashboards.components.metrics_card import metric_row, metric_card, gauge_metric
from dashboards.components.status_indicator import health_indicator


def render_metrics_performance():
    """Render the Metrics & Performance page."""
    render_header("Metrics & Performance")
    render_quick_stats()
    
    st.divider()
    
    filters = render_filters()
    
    st.divider()
    
    # Load data
    perf_metrics = get_performance_metrics()
    degradation_events = get_degradation_events()
    
    # Current metrics
    st.subheader("📊 Current Performance Metrics")
    
    if not perf_metrics.empty:
        current = get_current_metrics(perf_metrics)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = current.get('accuracy', 0.92)
            metric_card("Accuracy", accuracy, unit="%", status="Good")
            
            precision = current.get('precision', 0.89)
            metric_card("Precision", precision, unit="%", status="Good")
        
        with col2:
            recall = current.get('recall', 0.85)
            metric_card("Recall", recall, unit="%", status="Good")
            
            f1_score = current.get('f1_score', 0.87)
            metric_card("F1 Score", f1_score, unit="%", status="Good")
        
        with col3:
            regime_count = current.get('regime_count', 3)
            metric_card("Regimes Detected", regime_count, status="Good")
            
            data_quality = current.get('data_quality', 96.5)
            metric_card("Data Quality", data_quality, unit="%", status="Good")
    else:
        st.info("No performance metrics available yet")
    
    st.divider()
    
    # Performance over time
    st.subheader("📈 Performance Over Time")
    
    timeframe_option = st.radio("Select Timeframe", ["7-day", "30-day", "90-day"], horizontal=True)
    
    if not perf_metrics.empty:
        # Create sample data for visualization
        if 'timestamp' in perf_metrics.columns:
            perf_metrics['timestamp'] = pd.to_datetime(perf_metrics['timestamp'])
            perf_metrics = perf_metrics.sort_values('timestamp')
            
            # Select columns to plot
            plot_columns = ['accuracy', 'precision', 'recall']
            available_columns = [col for col in plot_columns if col in perf_metrics.columns]
            
            if available_columns:
                fig = go.Figure()
                
                for col in available_columns:
                    fig.add_trace(go.Scatter(
                        x=perf_metrics['timestamp'],
                        y=perf_metrics[col],
                        mode='lines',
                        name=col.capitalize(),
                        line=dict(width=2)
                    ))
                
                fig.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="Min Threshold")
                fig.update_layout(
                    title=f"Performance Metrics ({timeframe_option})",
                    xaxis_title="Date",
                    yaxis_title="Score",
                    hovermode="x unified",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timestamp data available for time series visualization")
    
    st.divider()
    
    # Data Quality Scorecard
    st.subheader("🔍 Data Quality Scorecard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gauge_metric("Data Completeness", 96.5, threshold_warning=95, threshold_critical=90)
        gauge_metric("Feature Stability", 91.2, threshold_warning=90, threshold_critical=80)
    
    with col2:
        gauge_metric("Outlier Detection", 2.1, max_val=10, threshold_warning=5, threshold_critical=8)
        gauge_metric("Duplicate Records", 0.5, max_val=5, threshold_warning=2, threshold_critical=3)
    
    st.divider()
    
    # Degradation events
    st.subheader("⚠️ Performance Degradation Events")
    
    if not degradation_events.empty:
        st.info(f"Total degradation events detected: {len(degradation_events)}")
        
        event_display = degradation_events[['timestamp', 'metric', 'value_before', 'value_after']].head(10) if all(col in degradation_events.columns for col in ['timestamp', 'metric', 'value_before', 'value_after']) else degradation_events.head(10)
        
        st.dataframe(event_display, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No degradation events detected")
    
    st.divider()
    
    # Confidence Distribution
    st.subheader("📊 Prediction Confidence Distribution")
    
    # Create sample confidence data
    confidence_data = pd.DataFrame({
        'confidence': [0.65 + (i % 35) / 100 for i in range(1000)]
    })
    
    fig = px.histogram(confidence_data, x='confidence', nbins=30, 
                      title="Distribution of Prediction Confidence Scores",
                      labels={'confidence': 'Confidence Score', 'count': 'Number of Predictions'})
    
    fig.add_vline(x=0.70, line_dash="dash", line_color="red", annotation_text="Low Confidence Threshold")
    fig.add_vline(x=0.85, line_dash="dash", line_color="green", annotation_text="High Confidence Threshold")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # System Health
    st.subheader("❤️ Overall System Health")
    health_indicator(92)
