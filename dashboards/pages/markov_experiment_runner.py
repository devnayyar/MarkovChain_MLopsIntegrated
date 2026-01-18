"""Markov Chain Experiment Runner - Interactive Model Testing."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

from dashboards.components.header import render_header
from dashboards.utils.constants import REGIME_COLORS


def render_markov_experiment_runner():
    """Render interactive Markov Chain experiment runner."""
    render_header("Markov Chain Experiment Runner")
    
    st.markdown("""
    Test and compare your Markov Chain model predictions. Input economic indicators
    to see how the model predicts regime transitions and compare base vs challenger models.
    """)
    
    st.divider()
    
    # Load the gold layer data
    try:
        gold_path = Path("data/gold/markov_state_sequences.parquet")
        df_gold = pd.read_parquet(gold_path)
        
        if 'REGIME_RISK' not in df_gold.columns:
            st.error("REGIME_RISK column not found")
            return
        
        from modeling.models.base_markov import MarkovChain
        
        # Build base model from historical data
        regime_states = df_gold['REGIME_RISK'].dropna().values
        unique_regimes = sorted(set(regime_states))
        
        base_model = MarkovChain(state_sequence=regime_states, states=unique_regimes)
        base_model.estimate_transition_matrix()
        
        # Sidebar for model configuration
        st.sidebar.subheader("🔧 Experiment Configuration")
        
        experiment_type = st.sidebar.radio(
            "Experiment Type",
            ["Single Prediction", "Regime Sequence", "Model Comparison"]
        )
        
        st.divider()
        
        if experiment_type == "Single Prediction":
            render_single_prediction(base_model, unique_regimes, df_gold)
        
        elif experiment_type == "Regime Sequence":
            render_regime_sequence(base_model, unique_regimes, df_gold)
        
        elif experiment_type == "Model Comparison":
            render_model_comparison(base_model, unique_regimes, df_gold)
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.write(traceback.format_exc())


def render_single_prediction(model, regimes, df_gold):
    """Single regime prediction interface."""
    st.subheader("🔮 Single Regime Prediction")
    
    st.write("Select the current economic regime and get predictions for the next state.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_regime = st.selectbox(
            "Current Regime",
            regimes,
            help="Select the current economic regime state"
        )
    
    with col2:
        steps = st.slider(
            "Prediction Steps",
            min_value=1,
            max_value=24,
            value=6,
            step=1,
            help="Number of periods to forecast"
        )
    
    # Make predictions
    if st.button("🎯 Run Prediction"):
        with st.spinner("Generating predictions..."):
            # Get transition probabilities for current regime
            current_idx = regimes.index(current_regime)
            trans_probs = model.transition_matrix[current_idx]
            
            # Create forecast
            forecast_data = []
            current_state = current_regime
            
            for step in range(steps):
                # Get probabilities for next states
                current_idx = regimes.index(current_state)
                next_probs = model.transition_matrix[current_idx]
                
                # Predict most likely next state
                next_idx = np.argmax(next_probs)
                next_state = regimes[next_idx]
                
                forecast_data.append({
                    'Step': step + 1,
                    'Regime': next_state,
                    'Probability': float(next_probs[next_idx]),
                    'Confidence': f"{next_probs[next_idx]*100:.1f}%"
                })
                
                current_state = next_state
            
            forecast_df = pd.DataFrame(forecast_data)
            
            st.divider()
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Forecast chart
                fig = go.Figure()
                
                for regime in regimes:
                    regime_data = forecast_df[forecast_df['Regime'] == regime]
                    if not regime_data.empty:
                        fig.add_trace(go.Scatter(
                            x=regime_data['Step'],
                            y=regime_data['Probability'],
                            mode='lines+markers',
                            name=regime,
                            line=dict(color=REGIME_COLORS.get(regime, "#3498db"), width=2),
                            marker=dict(size=8)
                        ))
                
                fig.update_layout(
                    title="Predicted Regime Path",
                    xaxis_title="Steps Ahead",
                    yaxis_title="Probability",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.write("**Forecast Path:**")
                for idx, row in forecast_df.iterrows():
                    st.metric(
                        f"Step {row['Step']}",
                        row['Regime'],
                        f"{row['Confidence']}"
                    )
            
            # Transition table
            st.divider()
            st.write("**Detailed Predictions:**")
            st.dataframe(forecast_df, width='stretch', hide_index=True)


def render_regime_sequence(model, regimes, df_gold):
    """Regime sequence forecast interface."""
    st.subheader("📈 Regime Sequence Forecast")
    
    st.write("Forecast regime sequence from current state.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_regime = st.selectbox(
            "Starting Regime",
            regimes,
            key="seq_regime"
        )
    
    with col2:
        forecast_days = st.slider(
            "Forecast Period (days)",
            min_value=7,
            max_value=180,
            value=30,
            step=7
        )
    
    with col3:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    if st.button("📊 Generate Forecast"):
        with st.spinner("Forecasting regime sequence..."):
            # Generate Monte Carlo simulation
            n_simulations = 1000
            forecasts = []
            
            for sim in range(n_simulations):
                current_state = current_regime
                daily_regimes = [current_state]
                
                for day in range(forecast_days):
                    current_idx = regimes.index(current_state)
                    next_probs = model.transition_matrix[current_idx]
                    
                    # Only transition if confidence > threshold
                    if np.max(next_probs) >= confidence_threshold:
                        next_idx = np.argmax(next_probs)
                    else:
                        next_idx = np.random.choice(len(regimes), p=next_probs)
                    
                    current_state = regimes[next_idx]
                    daily_regimes.append(current_state)
                
                forecasts.append(daily_regimes)
            
            # Calculate probabilities for each day
            regime_probabilities = []
            for day in range(forecast_days + 1):
                day_regimes = [f[day] for f in forecasts]
                regime_counts = {regime: day_regimes.count(regime) / n_simulations for regime in regimes}
                regime_probabilities.append(regime_counts)
            
            st.divider()
            st.subheader("🎯 Forecast Results")
            
            # Stacked area chart
            prob_df = pd.DataFrame(regime_probabilities)
            
            fig = go.Figure()
            
            for regime in regimes:
                fig.add_trace(go.Scatter(
                    x=prob_df.index,
                    y=prob_df[regime],
                    mode='lines',
                    name=regime,
                    stackgroup='one',
                    fillcolor=REGIME_COLORS.get(regime, "#3498db"),
                    line=dict(width=0.5)
                ))
            
            fig.update_layout(
                title=f"{forecast_days}-Day Regime Probability Forecast",
                xaxis_title="Days",
                yaxis_title="Probability",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Summary stats
            st.divider()
            st.write("**Forecast Summary**")
            
            final_probs = regime_probabilities[-1]
            summary_data = []
            
            for regime in sorted(regimes):
                summary_data.append({
                    'Regime': regime,
                    'End Probability': f"{final_probs[regime]*100:.1f}%",
                    'Expected Transitions': len([f for f in forecasts if f[-1] != f[0]]) / n_simulations
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch', hide_index=True)


def render_model_comparison(model, regimes, df_gold):
    """Base vs Challenger model comparison."""
    st.subheader("⚖️ Base vs Challenger Model Comparison")
    
    st.write("Compare predictions between base and challenger (smoothed) models.")
    
    # Create challenger model with smoothing
    regime_states = df_gold['REGIME_RISK'].dropna().values
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_regime = st.selectbox(
            "Current Regime",
            regimes,
            key="comp_regime"
        )
    
    with col2:
        smoothing_factor = st.slider(
            "Challenger Smoothing",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="How much to smooth transition probabilities"
        )
    
    with col3:
        forecast_horizon = st.slider(
            "Forecast Horizon",
            min_value=1,
            max_value=12,
            value=6
        )
    
    if st.button("🆚 Compare Models"):
        with st.spinner("Comparing models..."):
            # Get base model predictions
            current_idx = regimes.index(current_regime)
            base_probs = model.transition_matrix[current_idx].copy()
            
            # Create challenger model (smoothed)
            challenger_probs = base_probs.copy()
            # Apply Laplace smoothing
            challenger_probs = (challenger_probs + smoothing_factor) / (1 + smoothing_factor * len(regimes))
            
            st.divider()
            st.subheader("📊 Model Comparison Results")
            
            # Create comparison dataframe
            comparison_data = []
            for i, regime in enumerate(regimes):
                comparison_data.append({
                    'Next Regime': regime,
                    'Base Model': f"{base_probs[i]*100:.1f}%",
                    'Challenger': f"{challenger_probs[i]*100:.1f}%",
                    'Difference': f"{(challenger_probs[i] - base_probs[i])*100:+.1f}%"
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Probability Comparison**")
                st.dataframe(comp_df, width='stretch', hide_index=True)
            
            with col2:
                # Bar chart comparison
                fig = go.Figure(data=[
                    go.Bar(x=regimes, y=base_probs * 100, name='Base Model', marker_color='#1f77b4'),
                    go.Bar(x=regimes, y=challenger_probs * 100, name='Challenger Model', marker_color='#ff7f0e')
                ])
                
                fig.update_layout(
                    title="Model Prediction Comparison",
                    xaxis_title="Next Regime",
                    yaxis_title="Probability (%)",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # Detailed metrics
            st.divider()
            st.write("**Model Metrics**")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                entropy_base = -np.sum(base_probs[base_probs > 0] * np.log(base_probs[base_probs > 0]))
                st.metric("Base Entropy", f"{entropy_base:.3f}")
            
            with metric_col2:
                entropy_challenger = -np.sum(challenger_probs[challenger_probs > 0] * np.log(challenger_probs[challenger_probs > 0]))
                st.metric("Challenger Entropy", f"{entropy_challenger:.3f}")
            
            with metric_col3:
                kl_div = np.sum(base_probs[base_probs > 0] * np.log(base_probs[base_probs > 0] / challenger_probs[challenger_probs > 0]))
                st.metric("KL Divergence", f"{kl_div:.3f}")
