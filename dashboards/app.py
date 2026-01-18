"""Main Streamlit app entry point."""

import sys
from pathlib import Path
import streamlit as st

# Add parent directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboards.components.sidebar import render_sidebar
from dashboards.pages.home import render_home
from dashboards.pages.regime_timeline import render_regime_timeline
from dashboards.pages.markov_chain import render_markov_chain
from dashboards.pages.alerts_drift import render_alerts_drift
from dashboards.pages.metrics_performance import render_metrics_performance
from dashboards.pages.model_metrics import render_model_metrics
from dashboards.pages.eda_analysis import render_eda_analysis
from dashboards.pages.retraining_ab_testing import render_retraining_ab_testing
from dashboards.pages.markov_experiment_runner import render_markov_experiment_runner
from dashboards.pages.documentation import render_documentation
from dashboards.pages.settings import render_settings


def main():
    """Main Streamlit application."""
    # Configure page
    st.set_page_config(
        page_title="Financial Risk Markov MLOps Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for better spacing and styling
    st.markdown("""
    <style>
        /* Main container spacing */
        .main {
            padding: 2rem;
        }
        
        /* Metric card spacing */
        .metric-container {
            margin-bottom: 20px;
        }
        
        /* Column spacing */
        [data-testid="column"] {
            padding: 15px;
        }
        
        /* Card styling */
        .card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Better spacing for metrics */
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #1f77b4;
        }
        
        /* Alert card styling */
        .alert-card {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 5px solid;
        }
        
        /* Divider spacing */
        hr {
            margin: 2rem 0;
        }
        
        /* Subheader styling */
        h2 {
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            font-weight: 700;
        }
        
        h3 {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        /* Button spacing */
        .stButton > button {
            margin: 10px 5px;
            border-radius: 8px;
        }
        
        /* Better form element spacing */
        .stSelectbox, .stRadio, .stCheckbox, .stSlider {
            margin: 15px 0;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] button:first-child {
            margin-left: 0;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 12px;
        }
        
        /* Better table styling */
        .dataframe {
            margin: 15px 0;
        }
        
        /* Info box styling */
        .stInfo {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        
        /* Warning box styling */
        .stWarning {
            background-color: #fff3e0;
            border-left: 4px solid #f57c00;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        
        /* Success box styling */
        .stSuccess {
            background-color: #e8f5e9;
            border-left: 4px solid #388e3c;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        
        /* Error box styling */
        .stError {
            background-color: #ffebee;
            border-left: 4px solid #d32f2f;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        
        /* Tooltip styling */
        [title] {
            cursor: help;
            border-bottom: 1px dotted #999;
        }
        
        /* Section dividers */
        .section-divider {
            border-top: 2px solid #e0e0e0;
            margin: 30px 0;
            padding-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Override if session state was changed by navigation buttons
    page_to_render = st.session_state.get('current_page', selected_page)
    
    # Render page based on selection
    if page_to_render == 'home':
        render_home()
    elif page_to_render == 'regime_timeline':
        render_regime_timeline()
    elif page_to_render == 'markov_chain':
        render_markov_chain()
    elif page_to_render == 'alerts_drift':
        render_alerts_drift()
    elif page_to_render == 'metrics_performance':
        render_metrics_performance()
    elif page_to_render == 'model_metrics':
        render_model_metrics()
    elif page_to_render == 'eda_analysis':
        render_eda_analysis()
    elif page_to_render == 'markov_experiment_runner':
        render_markov_experiment_runner()
    elif page_to_render == 'documentation':
        render_documentation()
    elif page_to_render == 'retraining_ab_testing':
        render_retraining_ab_testing()
    elif page_to_render == 'settings':
        render_settings()
    else:
        render_home()


if __name__ == "__main__":
    main()
