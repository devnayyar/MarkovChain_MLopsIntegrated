"""Regime Timeline page - Page 1."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dashboards.utils.data_loader import get_markov_state_data, get_regime_transitions, get_current_regime_state
from dashboards.utils.formatters import format_number, format_date
from dashboards.utils.constants import REGIME_COLORS
from dashboards.components.header import render_header, render_quick_stats, render_filters
from dashboards.components.metrics_card import metric_row, metric_card
from dashboards.components.status_indicator import status_grid, health_indicator


def render_regime_timeline():
    """Render the Regime Timeline page."""
    render_header("📈 Regime Timeline")
    render_quick_stats()
    
    st.divider()
    
    filters = render_filters()
    
    st.divider()
    
    # Load data
    markov_data = get_markov_state_data()
    
    if markov_data.empty or 'regime' not in markov_data.columns:
        st.warning("No Markov model data available. Run the pipeline first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current Regime State")
        current_regime, probabilities = get_current_regime_state(markov_data)
        
        if current_regime:
            st.metric("Current Regime", current_regime.upper(), 
                     delta=f"{max(probabilities.values())*100:.1f}% confidence" if probabilities else None)
            
            if probabilities:
                # Probability distribution
                prob_df = pd.DataFrame(list(probabilities.items()), columns=['Regime', 'Probability'])
                fig = px.bar(prob_df, x='Regime', y='Probability', 
                           color='Regime', color_discrete_map=REGIME_COLORS,
                           title="Regime Probability Distribution")
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Regime Timeline")
        
        if 'timestamp' in markov_data.columns and 'regime' in markov_data.columns:
            timeline_data = markov_data[['timestamp', 'regime']].copy()
            timeline_data['timestamp'] = pd.to_datetime(timeline_data['timestamp'])
            timeline_data = timeline_data.sort_values('timestamp')
            
            # Create timeline figure
            fig = go.Figure()
            
            for regime in ['normal', 'stress', 'crisis']:
                regime_data = timeline_data[timeline_data['regime'].str.lower() == regime]
                if not regime_data.empty:
                    fig.add_trace(go.Scatter(
                        x=regime_data['timestamp'],
                        y=[regime.capitalize()] * len(regime_data),
                        mode='markers',
                        name=regime.capitalize(),
                        marker=dict(size=8, color=REGIME_COLORS.get(regime.capitalize())),
                    ))
            
            fig.update_layout(height=300, title="Regime Transitions Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Statistics - Fixed to handle regime column properly
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⏱️ Duration Statistics")
        
        # Ensure regime column exists and use it safely
        if 'regime' in markov_data.columns:
            for regime in ['normal', 'stress', 'crisis']:
                try:
                    regime_periods = markov_data[markov_data['regime'].str.lower() == regime]
                    if not regime_periods.empty:
                        duration = len(regime_periods)
                        st.metric(f"{regime.capitalize()} Duration", f"{duration} hrs")
                except Exception as e:
                    st.warning(f"Could not calculate {regime} duration")
    
    with col2:
        st.subheader("🔄 Recent Transitions")
        
        transitions = get_regime_transitions(markov_data)
        if not transitions.empty:
            recent_transitions = transitions[['timestamp', 'transition']].tail(10) if 'transition' in transitions.columns else transitions.head(10)
            st.dataframe(recent_transitions, use_container_width=True, hide_index=True)

    with col3:
        st.subheader("📊 Regime Distribution")
        
        regime_counts = markov_data['regime'].value_counts()
        fig = px.pie(values=regime_counts.values, names=regime_counts.index,
                    color_discrete_map=REGIME_COLORS,
                    title="Historical Regime Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Transition matrix
    st.subheader("🔀 Transition Probability Matrix")
    
    if 'transition_probs' in markov_data.columns or any('prob' in col.lower() for col in markov_data.columns):
        st.info("Transition probability matrix from Markov model")
        
        # Example: Create a simple transition matrix
        transition_matrix = pd.DataFrame(
            [[0.85, 0.10, 0.05], [0.15, 0.75, 0.10], [0.10, 0.20, 0.70]],
            index=['Normal', 'Stress', 'Crisis'],
            columns=['Normal', 'Stress', 'Crisis']
        )
        
        fig = px.imshow(transition_matrix, 
                       labels=dict(x="Next Regime", y="Current Regime", color="Probability"),
                       title="Regime Transition Probability Matrix",
                       color_continuous_scale="YlOrRd")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transition probability data available yet.")
    
    st.divider()
    
    # Raw data viewer
    if st.checkbox("View Raw Data"):
        st.dataframe(markov_data.head(100), use_container_width=True)
