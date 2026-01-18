"""Tooltip and help components for user guidance."""

import streamlit as st


def info_tooltip(label, description, icon="ℹ️"):
    """
    Display a label with an info tooltip for non-technical users.
    
    Args:
        label: Main label to display
        description: Description shown on hover
        icon: Emoji icon to display
    """
    col1, col2 = st.columns([20, 1])
    with col1:
        st.markdown(label)
    with col2:
        st.markdown(f"<span title='{description}'>{icon}</span>", unsafe_allow_html=True)


def help_section(title, content):
    """Display a help section with expandable content."""
    with st.expander(f"❓ {title}", expanded=False):
        st.markdown(content)


def metric_with_tooltip(title, value, unit="", tooltip_text=""):
    """
    Display a metric with an optional tooltip.
    
    Args:
        title: Metric title
        value: Metric value
        unit: Unit of measurement
        tooltip_text: Tooltip description for users
    """
    if tooltip_text:
        col1, col2 = st.columns([20, 1])
        with col1:
            st.metric(title, f"{value} {unit}".strip())
        with col2:
            st.markdown(f"<span title='{tooltip_text}' style='cursor: help;'>ℹ️</span>", unsafe_allow_html=True)
    else:
        st.metric(title, f"{value} {unit}".strip())


# Glossary of common terms for non-technical users
GLOSSARY = {
    "Regime": "A market state characterized by specific price behavior patterns (Normal, Stress, or Crisis)",
    "Markov Chain": "A mathematical model that predicts future market states based on current state probabilities",
    "Transition Matrix": "A table showing the probability of moving from one market state to another",
    "Volatility": "A measure of how much the price fluctuates - higher values mean more unstable markets",
    "Drift Detection": "Monitoring for unexpected changes in data patterns that could affect model accuracy",
    "A/B Testing": "Comparing two model versions to see which performs better with real data",
    "Retraining": "Updating the AI model with new data to maintain accuracy over time",
    "Data Quality Score": "A percentage showing how clean and reliable the data is (higher is better)",
    "Prediction Confidence": "How sure the model is about its prediction (0-100%, higher is better)",
    "Model Accuracy": "Percentage of correct predictions made by the AI model",
    "False Positive Rate": "How often the model incorrectly predicts a market change that doesn't happen",
    "Latency": "Time delay between data collection and model prediction",
}


def show_glossary():
    """Display a glossary modal for common terms."""
    with st.expander("📚 Glossary - Common Terms", expanded=False):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}**: {definition}")


def regime_explanation():
    """Show explanation of regime types."""
    with st.expander("📖 Understanding Market Regimes", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🟢 Normal Regime
            
            **Characteristics:**
            - Stable market conditions
            - Predictable price movements
            - Low volatility
            - Standard trading volumes
            
            **When to expect:**
            - Market functioning normally
            - No major economic news
            - Routine trading activity
            """)
        
        with col2:
            st.markdown("""
            ### 🟡 Stress Regime
            
            **Characteristics:**
            - Increased uncertainty
            - Higher price swings
            - Elevated volatility
            - Unusual trading patterns
            
            **When to expect:**
            - Important economic announcements
            - Sector-specific concerns
            - Risk-off sentiment
            """)
        
        with col3:
            st.markdown("""
            ### 🔴 Crisis Regime
            
            **Characteristics:**
            - Extreme market dislocations
            - Very high volatility
            - Sudden sharp movements
            - Flight to safety behavior
            
            **When to expect:**
            - Financial emergencies
            - Major market shocks
            - Systemic risk events
            """)


def performance_metric_help():
    """Display help about performance metrics."""
    with st.expander("📊 Understanding Performance Metrics", expanded=False):
        st.markdown("""
        - **Accuracy**: Overall correctness of predictions (goal: > 90%)
        - **Precision**: Of positive predictions, how many are correct (goal: > 85%)
        - **Recall**: Of actual positives, how many are detected (goal: > 80%)
        - **F1 Score**: Balanced measure combining precision and recall (goal: > 85%)
        - **AUC-ROC**: Ability to distinguish between classes (goal: > 0.90)
        """)


def data_quality_help():
    """Display help about data quality."""
    with st.expander("🔍 Data Quality Explained", expanded=False):
        st.markdown("""
        **Data Layers:**
        
        1. **Bronze** (Raw Data)
           - Direct from source
           - May have issues
           - Quality: ~87%
        
        2. **Silver** (Cleaned)
           - Issues fixed
           - Processed
           - Quality: ~95%
        
        3. **Gold** (Ready)
           - ML-ready format
           - High quality
           - Quality: ~98%
        
        **Quality Score Meaning:**
        - 90-100%: ✅ Production ready
        - 75-89%: ⚠️ Needs attention
        - <75%: 🔴 Requires action
        """)


def alert_severity_help():
    """Display help about alert severity levels."""
    with st.expander("🚨 Understanding Alert Levels", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🟢 INFO**
            
            Informational updates
            
            • Routine events
            • Status changes
            • No action needed
            """)
        
        with col2:
            st.warning("""
            **🟡 WARNING**
            
            Needs attention
            
            • Minor issues
            • Performance dip
            • Review soon
            """)
        
        with col3:
            st.error("""
            **🔴 CRITICAL**
            
            Urgent action required
            
            • Major problems
            • Service risk
            • Immediate action
            """)
