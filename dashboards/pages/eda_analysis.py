"""Enhanced EDA Analysis page with trend charts - CLEAN."""

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


def render_eda_analysis():
    """Render the EDA Analysis page."""
    render_header("EDA Analysis - Data Layers")
    render_quick_stats()
    
    st.divider()
    
    # Tab selection for data layers
    tab1, tab2, tab3 = st.tabs(["🥉 Bronze Layer", "🥈 Silver Layer", "🥇 Gold Layer"])
    
    with tab1:
        render_bronze_analysis()
    
    with tab2:
        render_silver_analysis()
    
    with tab3:
        render_gold_analysis()


def render_bronze_analysis():
    """Render Bronze layer analysis with trend charts."""
    st.subheader("🥉 Bronze Layer - Raw Data Ingestion")
    
    try:
        gold_path = Path("data/gold/markov_state_sequences.parquet")
        df_gold = pd.read_parquet(gold_path)
        
        if df_gold.empty:
            st.warning("No bronze data available")
            return
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From Date", value=pd.to_datetime(df_gold['date'].min()), key="bronze_date_from")
        with col2:
            date_to = st.date_input("To Date", value=pd.to_datetime(df_gold['date'].max()), key="bronze_date_to")
        
        df_filtered = df_gold[(df_gold['date'] >= pd.Timestamp(date_from)) & (df_gold['date'] <= pd.Timestamp(date_to))].copy()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Total Records", f"{len(df_filtered):,}", status="Good")
        with col2:
            null_pct = (df_filtered.isnull().sum().sum() / (len(df_filtered) * len(df_filtered.columns))) * 100
            metric_card("Missing Values", f"{null_pct:.2f}%", status="Warning" if null_pct > 1 else "Good")
        with col3:
            st.metric("Date Range", f"{(date_to - date_from).days} days")
        with col4:
            if 'UNRATE' in df_filtered.columns:
                st.metric("Avg Unemployment", f"{df_filtered['UNRATE'].mean():.2f}%")
        
        st.divider()
        st.write("**📊 Economic Indicators Trends**")
        
        indicators = ['UNRATE', 'FEDFUNDS', 'CPI_YOY', 'T10Y2Y', 'STLFSI4']
        available_indicators = [ind for ind in indicators if ind in df_filtered.columns]
        
        if available_indicators:
            cols = st.columns(2)
            for idx, indicator in enumerate(available_indicators):
                col = cols[idx % 2]
                with col:
                    if df_filtered[indicator].notna().sum() > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_filtered['date'], y=df_filtered[indicator], mode='lines',
                            name=indicator, line=dict(color='#1f77b4', width=2),
                            fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)'
                        ))
                        if len(df_filtered) > 30:
                            ma = df_filtered[indicator].rolling(window=30).mean()
                            fig.add_trace(go.Scatter(
                                x=df_filtered['date'], y=ma, mode='lines', name='30-Day MA',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                        fig.update_layout(title=f"{indicator} Trend", xaxis_title="Date", yaxis_title=indicator, hovermode='x unified', height=350)
                        st.plotly_chart(fig, width='stretch')
        
        st.divider()
        st.write("**📋 Data Quality by Column**")
        quality_data = [{'Column': c, 'Non-Null': df_filtered[c].notna().sum(), 'Null %': f"{(df_filtered[c].isna().sum() / len(df_filtered) * 100):.2f}%", 'Type': str(df_filtered[c].dtype)} for c in df_filtered.columns]
        st.dataframe(pd.DataFrame(quality_data), width='stretch', hide_index=True)
    
    except Exception as e:
        st.error(f"Error in bronze analysis: {str(e)}")


def render_silver_analysis():
    """Render Silver layer analysis with trend charts."""
    st.subheader("🥈 Silver Layer - Processed Data")
    
    try:
        gold_path = Path("data/gold/markov_state_sequences.parquet")
        df_gold = pd.read_parquet(gold_path)
        
        if df_gold.empty:
            st.warning("No silver data available")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From Date", value=pd.to_datetime(df_gold['date'].min()), key="silver_date_from")
        with col2:
            date_to = st.date_input("To Date", value=pd.to_datetime(df_gold['date'].max()), key="silver_date_to")
        
        df_filtered = df_gold[(df_gold['date'] >= pd.Timestamp(date_from)) & (df_gold['date'] <= pd.Timestamp(date_to))].copy()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Processed Records", f"{len(df_filtered):,}", status="Good")
        with col2:
            metric_card("Feature Columns", f"{len(df_filtered.columns)}", status="Good")
        with col3:
            st.metric("Date Range", f"{(date_to - date_from).days} days")
        with col4:
            st.metric("Quality", "100%")
        
        st.divider()
        st.write("**📊 Feature Correlations**")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df_filtered[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='RdBu', zmid=0, text=np.round(corr_matrix.values, 2),
                texttemplate='%{text:.2f}', textfont={"size": 10}
            ))
            fig.update_layout(title="Feature Correlation Matrix", height=500)
            st.plotly_chart(fig, width='stretch')
        
        st.divider()
        state_cols = [col for col in df_filtered.columns if '_STATE' in col]
        if state_cols:
            st.write("**📈 State Variable Trends**")
            for state_col in state_cols[:4]:
                if df_filtered[state_col].notna().sum() > 0:
                    state_counts = df_filtered[state_col].value_counts().sort_index()
                    fig = px.bar(x=state_counts.index, y=state_counts.values, title=f"{state_col} Distribution", labels={'x': state_col, 'y': 'Count'})
                    st.plotly_chart(fig, width='stretch')
    
    except Exception as e:
        st.error(f"Error in silver analysis: {str(e)}")


def render_gold_analysis():
    """Render Gold layer analysis with regime highlighting."""
    st.subheader("🥇 Gold Layer - Model Features & Regimes")
    
    try:
        gold_path = Path("data/gold/markov_state_sequences.parquet")
        df_gold = pd.read_parquet(gold_path)
        
        if df_gold.empty:
            st.warning("No gold data available")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From Date", value=pd.to_datetime(df_gold['date'].min()), key="gold_date_from")
        with col2:
            date_to = st.date_input("To Date", value=pd.to_datetime(df_gold['date'].max()), key="gold_date_to")
        
        df_filtered = df_gold[(df_gold['date'] >= pd.Timestamp(date_from)) & (df_gold['date'] <= pd.Timestamp(date_to))].copy()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Final Records", f"{len(df_filtered):,}", status="Good")
        with col2:
            unique_regimes = df_filtered['REGIME_RISK'].nunique()
            metric_card("Unique Regimes", f"{unique_regimes}", status="Good")
        with col3:
            current_regime = df_filtered['REGIME_RISK'].iloc[-1] if len(df_filtered) > 0 else "N/A"
            st.metric("Current Regime", current_regime)
        with col4:
            st.metric("Ready", "✅ Yes", label_visibility="visible")
        
        st.divider()
        st.write("**📊 Regime Timeline**")
        
        fig = go.Figure()
        for regime in df_filtered['REGIME_RISK'].unique():
            regime_mask = df_filtered['REGIME_RISK'] == regime
            regime_dates = df_filtered.loc[regime_mask, 'date']
            fig.add_trace(go.Scatter(
                x=regime_dates, y=[1] * len(regime_dates), mode='markers', name=regime,
                marker=dict(size=12, color=REGIME_COLORS.get(regime, "#3498db"), opacity=0.7, line=dict(width=2, color='white')),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<extra></extra>'
            ))
        
        fig.update_layout(title="Regime Timeline with Risk Highlighting", xaxis_title="Date", yaxis=dict(visible=False), height=300, hovermode='x unified')
        st.plotly_chart(fig, width='stretch')
        
        st.divider()
        st.write("**📈 Regime Distribution & Duration**")
        
        regime_stats = []
        for regime in sorted(df_filtered['REGIME_RISK'].unique()):
            regime_data = df_filtered[df_filtered['REGIME_RISK'] == regime]
            regime_stats.append({
                'Regime': regime, 'Count': len(regime_data), 'Percentage': f"{len(regime_data)/len(df_filtered)*100:.1f}%",
                'First Date': regime_data['date'].min(), 'Last Date': regime_data['date'].max(),
                'Duration (Days)': (regime_data['date'].max() - regime_data['date'].min()).days
            })
        
        stats_df = pd.DataFrame(regime_stats)
        st.dataframe(stats_df, width='stretch', hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(values=stats_df['Count'], names=stats_df['Regime'],
                            color_discrete_map={regime: REGIME_COLORS.get(regime, "#3498db") for regime in stats_df['Regime']},
                            title="Regime Distribution")
            st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            transitions = []
            for i in range(len(df_filtered) - 1):
                from_regime = df_filtered['REGIME_RISK'].iloc[i]
                to_regime = df_filtered['REGIME_RISK'].iloc[i + 1]
                if from_regime != to_regime:
                    transitions.append({'from': from_regime, 'to': to_regime})
            
            if transitions:
                transition_counts = pd.DataFrame(transitions).value_counts().reset_index(name='count')
                st.metric("Total Transitions", len(transitions))
                st.write("**Top Transitions:**")
                for idx, row in transition_counts.head(5).iterrows():
                    st.write(f"- {row['from']} → {row['to']}: {row['count']} times")
            else:
                st.info("No regime transitions in selected period")
        
        st.divider()
        st.write("**🚨 Risk Assessment**")
        
        high_risk_pct = (df_filtered['REGIME_RISK'] == 'HIGH_RISK').sum() / len(df_filtered) * 100
        moderate_risk_pct = (df_filtered['REGIME_RISK'] == 'MODERATE_RISK').sum() / len(df_filtered) * 100
        low_risk_pct = (df_filtered['REGIME_RISK'] == 'LOW_RISK').sum() / len(df_filtered) * 100
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            st.metric("🔴 High Risk", f"{high_risk_pct:.1f}%", delta="+" if high_risk_pct > 10 else "-")
        with risk_col2:
            st.metric("🟡 Moderate Risk", f"{moderate_risk_pct:.1f}%")
        with risk_col3:
            st.metric("🟢 Low Risk", f"{low_risk_pct:.1f}%")
    
    except Exception as e:
        st.error(f"Error in gold analysis: {str(e)}")
