"""Model Metrics and Diagnostics page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path

from dashboards.components.header import render_header, render_quick_stats
from dashboards.components.metrics_card import metric_card
from dashboards.utils.constants import REGIME_COLORS


def calculate_model_metrics(states_sequence):
    """Calculate AIC, BIC, perplexity and other model diagnostics."""
    if len(states_sequence) < 2:
        return {}
    
    # Count unique states
    unique_states = len(set(states_sequence))
    n_params = unique_states * (unique_states - 1)  # Transition matrix parameters
    n_observations = len(states_sequence)
    
    # Calculate log-likelihood (simplified)
    state_transitions = {}
    for i in range(len(states_sequence) - 1):
        current = states_sequence[i]
        next_state = states_sequence[i + 1]
        key = (current, next_state)
        state_transitions[key] = state_transitions.get(key, 0) + 1
    
    # Calculate transition probabilities
    log_likelihood = 0
    for transition, count in state_transitions.items():
        current_state = transition[0]
        # Count transitions from current state
        total_from_current = sum(1 for s in states_sequence[:-1] if s == current_state)
        if total_from_current > 0:
            prob = count / total_from_current
            if prob > 0:
                log_likelihood += count * np.log(prob)
    
    # Calculate information criteria
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_observations) - 2 * log_likelihood
    
    # Calculate perplexity
    if log_likelihood > 0:
        perplexity = np.exp(-log_likelihood / n_observations)
    else:
        perplexity = 0
    
    return {
        'aic': float(aic),
        'bic': float(bic),
        'log_likelihood': float(log_likelihood),
        'perplexity': float(perplexity),
        'n_states': unique_states,
        'n_params': n_params,
        'n_observations': n_observations
    }


def render_model_metrics():
    """Render model metrics and diagnostics page."""
    render_header("Model Metrics & Diagnostics")
    render_quick_stats()
    
    st.divider()
    
    # Load gold data for metrics calculation
    try:
        gold_path = Path("data/gold/markov_state_sequences.parquet")
        df_gold = pd.read_parquet(gold_path)
        
        if 'REGIME_RISK' not in df_gold.columns:
            st.error("REGIME_RISK column not found in gold data")
            return
        
        regime_states = df_gold['REGIME_RISK'].dropna().values
        
        # Calculate metrics
        metrics = calculate_model_metrics(regime_states)
        
        # Display key metrics
        st.subheader("📊 Model Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric_card(
                "AIC",
                f"{metrics['aic']:.2f}",
                status="Good"
            )
        
        with col2:
            metric_card(
                "BIC",
                f"{metrics['bic']:.2f}",
                status="Good"
            )
        
        with col3:
            metric_card(
                "Perplexity",
                f"{metrics['perplexity']:.4f}",
                status="Good"
            )
        
        with col4:
            metric_card(
                "Log-Likelihood",
                f"{metrics['log_likelihood']:.2f}",
                status="Good"
            )
        
        st.divider()
        
        # Model Parameters
        st.subheader("🔧 Model Parameters")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.metric("Number of States", metrics['n_states'])
        
        with param_col2:
            st.metric("Number of Parameters", metrics['n_params'])
        
        with param_col3:
            st.metric("Observations Used", metrics['n_observations'])
        
        st.divider()
        
        # Transition Matrix as Heatmap
        st.subheader("🔄 Transition Matrix")
        
        unique_regimes = sorted(df_gold['REGIME_RISK'].dropna().unique())
        
        # Build transition matrix
        trans_matrix = np.zeros((len(unique_regimes), len(unique_regimes)))
        
        for i in range(len(regime_states) - 1):
            current = regime_states[i]
            next_state = regime_states[i + 1]
            curr_idx = unique_regimes.index(current)
            next_idx = unique_regimes.index(next_state)
            trans_matrix[curr_idx, next_idx] += 1
        
        # Normalize to probabilities
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(trans_matrix, row_sums, where=row_sums != 0, out=np.zeros_like(trans_matrix))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=trans_matrix,
            x=unique_regimes,
            y=unique_regimes,
            colorscale='Blues',
            text=np.round(trans_matrix, 3),
            texttemplate='%{text:.3f}',
            textfont={"size": 12},
            colorbar=dict(title="Probability")
        ))
        
        fig.update_layout(
            title="State Transition Probabilities",
            xaxis_title="To State",
            yaxis_title="From State",
            height=500
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.divider()
        
        # Regime Statistics
        st.subheader("📈 Regime Statistics")
        
        regime_stats = []
        for regime in unique_regimes:
            count = sum(regime_states == regime)
            percentage = (count / len(regime_states)) * 100
            regime_stats.append({
                'Regime': regime,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        stats_df = pd.DataFrame(regime_stats)
        st.dataframe(stats_df, width='stretch', hide_index=True)
        
        # Pie chart
        fig_pie = px.pie(
            values=[s['Count'] for s in regime_stats],
            names=[s['Regime'] for s in regime_stats],
            color_discrete_map={regime: REGIME_COLORS.get(regime, "#3498db") for regime in [s['Regime'] for s in regime_stats]},
            title="Regime Distribution"
        )
        
        st.plotly_chart(fig_pie, width='stretch')
        
        st.divider()
        
        # Model Weights / State Information
        st.subheader("⚖️ Model Weights")
        
        # Calculate stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(trans_matrix.T)
        stationary_dist = np.real(eigenvectors[:, np.argmax(np.abs(eigenvalues - 1) < 1e-8)])
        stationary_dist = np.abs(stationary_dist) / np.sum(np.abs(stationary_dist))
        
        weights_df = pd.DataFrame({
            'Regime': unique_regimes,
            'Stationary Weight': stationary_dist,
            'Percentage': [f"{w*100:.1f}%" for w in stationary_dist]
        })
        
        st.dataframe(weights_df, width='stretch', hide_index=True)
        
        # Weights visualization
        fig_weights = px.bar(
            weights_df,
            x='Regime',
            y='Stationary Weight',
            color='Regime',
            color_discrete_map={regime: REGIME_COLORS.get(regime, "#3498db") for regime in unique_regimes},
            title="Long-Run Stationary Weights",
            labels={'Stationary Weight': 'Weight'}
        )
        
        st.plotly_chart(fig_weights, width='stretch')
        
    except Exception as e:
        st.error(f"Error loading model metrics: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
