"""Markov Chain Visualization page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dashboards.utils.data_loader import (
    get_markov_state_data, get_markov_transition_matrix, get_markov_chain_stats,
    get_current_regime_state
)
from dashboards.utils.constants import REGIME_COLORS
from dashboards.components.header import render_header, render_quick_stats, render_filters
from dashboards.components.metrics_card import metric_card, metric_row
from dashboards.components.status_indicator import status_grid


def render_markov_chain():
    """Render the Markov Chain Visualization page."""
    render_header("Markov Chain Analysis")
    render_quick_stats()
    
    st.divider()
    
    filters = render_filters()
    
    st.divider()
    
    # Load data
    markov_data = get_markov_state_data()
    
    if markov_data.empty:
        st.warning("No Markov model data available. Run the pipeline first.")
        return
    
    # Current Regime State
    st.subheader("📊 Current Regime State")
    
    current_regime, regime_probs = get_current_regime_state(markov_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_card(
            "Current Regime",
            current_regime.upper() if isinstance(current_regime, str) else str(current_regime),
            status="Good" if current_regime and "crisis" not in str(current_regime).lower() else "Warning"
        )
    
    if regime_probs:
        with col2:
            max_prob = max(regime_probs.values()) if regime_probs else 0
            metric_card("Confidence", f"{max_prob*100:.1f}", unit="%", status="Good")
        
        with col3:
            volatility = markov_data['volatility'].iloc[-1] if 'volatility' in markov_data.columns else 0
            metric_card("Volatility", f"{volatility:.2f}", status="Good" if volatility < 2.0 else "Warning")
    
    st.divider()
    
    # Regime Probabilities
    st.subheader("🎯 Regime Probability Distribution")
    
    if regime_probs:
        prob_df = pd.DataFrame(list(regime_probs.items()), columns=['Regime', 'Probability'])
        
        fig = px.bar(
            prob_df,
            x='Regime',
            y='Probability',
            color='Regime',
            color_discrete_map={regime: REGIME_COLORS.get(regime, "#3498db") for regime in prob_df['Regime']},
            title="Current State Probability Distribution",
            labels={'Probability': 'Probability', 'Regime': 'Regime State'}
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Transition Probability Matrix
    st.subheader("🔀 Transition Probability Matrix")
    
    trans_matrix = get_markov_transition_matrix()
    
    # Heatmap of transition matrix
    fig = go.Figure(data=go.Heatmap(
        z=trans_matrix.values,
        x=trans_matrix.columns,
        y=trans_matrix.index,
        colorscale='Blues',
        text=trans_matrix.values.round(2),
        texttemplate='%{text:.2%}',
        textfont={"size": 12},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        title="Regime Transition Probability Matrix",
        xaxis_title="Next Regime",
        yaxis_title="Current Regime",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display matrix as table
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Transition Probability Table**")
        st.dataframe(trans_matrix.round(4), use_container_width=True)
    
    with col2:
        st.write("**Key Insights**")
        st.info("""
        - **Normal Regime**: 85% likely to stay, 12% to Stress, 3% to Crisis
        - **Stress Regime**: 75% likely to stay, 15% to Normal, 10% to Crisis
        - **Crisis Regime**: 70% likely to stay, 20% to Stress, 10% to Normal
        """)
    
    st.divider()
    
    # Markov Chain Statistics
    st.subheader("📈 Markov Chain Statistics")
    
    stats = get_markov_chain_stats(markov_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        normal_pct = stats['regime_distribution'].get('normal', 0) * 100
        metric_card("Normal Distribution", f"{normal_pct:.1f}", unit="%", status="Good")
    
    with col2:
        stress_pct = stats['regime_distribution'].get('stress', 0) * 100
        metric_card("Stress Distribution", f"{stress_pct:.1f}", unit="%", status="Info")
    
    with col3:
        crisis_pct = stats['regime_distribution'].get('crisis', 0) * 100
        metric_card("Crisis Distribution", f"{crisis_pct:.1f}", unit="%", status="Warning" if crisis_pct > 15 else "Good")
    
    st.divider()
    
    # Average Duration in Each Regime
    st.subheader("⏱️ Average Duration in Each Regime")
    
    durations = stats['average_durations']
    duration_df = pd.DataFrame([
        {'Regime': 'Normal', 'Avg Duration (hours)': durations.get('normal', 0)},
        {'Regime': 'Stress', 'Avg Duration (hours)': durations.get('stress', 0)},
        {'Regime': 'Crisis', 'Avg Duration (hours)': durations.get('crisis', 0)},
    ])
    
    fig = px.bar(
        duration_df,
        x='Regime',
        y='Avg Duration (hours)',
        color='Regime',
        color_discrete_map=REGIME_COLORS,
        title="Average Time Spent in Each Regime",
        labels={'Avg Duration (hours)': 'Hours'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Regime Sequence Timeline
    st.subheader("📅 Regime Sequence Timeline (Last 100 observations)")
    
    if 'timestamp' in markov_data.columns and 'regime' in markov_data.columns:
        timeline_data = markov_data[['timestamp', 'regime']].tail(100).copy()
        timeline_data['timestamp'] = pd.to_datetime(timeline_data['timestamp'])
        timeline_data = timeline_data.sort_values('timestamp')
        
        fig = go.Figure()
        
        for regime in ['normal', 'stress', 'crisis']:
            regime_data = timeline_data[timeline_data['regime'].str.lower() == regime]
            if not regime_data.empty:
                fig.add_trace(go.Scatter(
                    x=regime_data['timestamp'],
                    y=[1] * len(regime_data),
                    mode='markers',
                    name=regime.capitalize(),
                    marker=dict(
                        size=10,
                        color=REGIME_COLORS.get(regime.capitalize(), "#3498db"),
                        opacity=0.7
                    )
                ))
        
        fig.update_layout(
            title="Regime Transitions Over Time",
            xaxis_title="Time",
            yaxis=dict(visible=False),
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Markov Chain Diagram
    st.subheader("🔗 Markov Chain State Diagram")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### State Transitions
        
        The Markov chain represents financial regimes as a system of states with probabilistic transitions:
        
        ```
                85%
           ┌─────────────┐
           │             │
        Normal ←──────→ Stress
           │  ↘ 3%   ↙ 10%│
           └──────→ Crisis ←┘
              10%     20%
        ```
        """)
    
    with col2:
        st.write("**State Properties**")
        
        state_props = pd.DataFrame({
            'Regime': ['Normal', 'Stress', 'Crisis'],
            'Risk Level': ['Low', 'Medium', 'High'],
            'Persistence': ['High (85%)', 'High (75%)', 'High (70%)'],
            'Market Impact': ['Stable', 'Volatile', 'Extreme'],
        })
        
        st.dataframe(state_props, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Regime Persistence Analysis
    st.subheader("🔍 Regime Persistence Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mean Reversion Time**")
        st.metric(
            "Average Time to Leave Current Regime",
            f"{max([durations.get('normal', 0), durations.get('stress', 0), durations.get('crisis', 0)]):.1f} hours"
        )
    
    with col2:
        st.write("**Regime Stickiness**")
        diag_sum = trans_matrix.values.diagonal().sum() / len(trans_matrix)
        st.metric("Average Self-Transition Probability", f"{diag_sum:.1%}")
    
    st.divider()
    
    # Raw Data View
    if st.checkbox("View Raw Markov Data"):
        st.write("**First 50 observations**")
        display_cols = ['timestamp', 'regime', 'normal_prob', 'stress_prob', 'crisis_prob', 'volatility']
        available_cols = [col for col in display_cols if col in markov_data.columns]
        st.dataframe(markov_data[available_cols].head(50), use_container_width=True, hide_index=True)
