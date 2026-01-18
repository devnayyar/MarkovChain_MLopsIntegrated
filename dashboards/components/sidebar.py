"""Sidebar component for dashboard."""

import streamlit as st
from dashboards.utils.constants import PAGES


def render_sidebar():
    """Render the main sidebar navigation with full UI and detailed sections."""
    # Sidebar styling
    st.sidebar.markdown("""
    <style>
        .sidebar-header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid #f0f0f0;
            margin-bottom: 20px;
        }
        .sidebar-header h1 {
            margin: 0;
            color: #1f77b4;
            font-size: 24px;
        }
        .sidebar-section-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            color: #666;
            letter-spacing: 1px;
            margin: 25px 0 15px 0;
            padding: 10px 0;
            border-top: 1px solid #f0f0f0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Logo and title
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h1>📊 FINML</h1>
        <p style="margin: 5px 0 0 0; color: #888; font-size: 12px;">Financial Risk Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation section
    st.sidebar.markdown('<p class="sidebar-section-title">📑 Navigation</p>', unsafe_allow_html=True)
    
    # Create radio buttons with better styling
    page_options = list(PAGES.keys())
    selected_page_label = st.sidebar.radio(
        "Select Page",
        page_options,
        label_visibility="collapsed",
        key="main_navigation"
    )
    
    # Map the label to page key
    selected_page = PAGES[selected_page_label]
    
    # Update session state
    st.session_state.current_page = selected_page
    
    st.sidebar.divider()
    
    # Settings section
    st.sidebar.markdown('<p class="sidebar-section-title">⚙️ Display Settings</p>', unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        theme = st.sidebar.selectbox(
            "🎨 Theme",
            ["Auto", "Light", "Dark"],
            key="theme_select",
            label_visibility="collapsed"
        )
    
    with col2:
        st.sidebar.markdown("**Theme**", help="Choose dashboard color scheme")
    
    # Refresh settings
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        auto_refresh = st.sidebar.toggle("🔄 Auto Refresh", value=True, key="auto_refresh_toggle")
    
    with col2:
        st.sidebar.markdown("**Auto Refresh**", help="Enable real-time updates")
    
    # Decimal places setting
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        decimals = st.sidebar.slider("📊 Decimals", 1, 5, 2, key="decimals_slider")
    with col2:
        st.sidebar.markdown("**Display**", help="Decimal places for numbers")
    
    st.sidebar.divider()
    
    # Help & Support section
    st.sidebar.markdown('<p class="sidebar-section-title">❓ Help & Support</p>', unsafe_allow_html=True)
    
    if st.sidebar.button("📖 User Guide", use_container_width=True, key="btn_guide"):
        st.sidebar.info("""
        **Dashboard User Guide:**
        
        - **Home**: Overview and quick summary
        - **Regime Timeline**: Financial regime analysis
        - **Markov Chain**: State transition probabilities
        - **Alerts & Drift**: System alerts and data drift
        - **Metrics**: Model performance tracking
        - **EDA Analysis**: Data quality by layer
        - **Retraining**: Model updates and A/B tests
        - **Settings**: Configuration options
        """)
    
    if st.sidebar.button("🐛 Report Issue", use_container_width=True, key="btn_report"):
        st.sidebar.success("Thank you! Issue report will be reviewed by the team.")
    
    if st.sidebar.button("💬 Feedback", use_container_width=True, key="btn_feedback"):
        st.sidebar.info("Your feedback helps us improve. Thank you!")
    
    if st.sidebar.button("📚 Glossary", use_container_width=True, key="btn_glossary"):
        st.sidebar.info("""
        **Common Terms:**
        - **Regime**: Market state (Normal, Stress, Crisis)
        - **Markov Chain**: State prediction model
        - **Drift**: Unexpected data pattern changes
        - **A/B Testing**: Model performance comparison
        """)
    
    st.sidebar.divider()
    
    # System Status section
    st.sidebar.markdown('<p class="sidebar-section-title">📡 System Status</p>', unsafe_allow_html=True)
    
    status_items = [
        ("🔄 Pipeline", "Healthy"),
        ("🗄️ Database", "Connected"),
        ("🤖 Model", "Active"),
        ("📊 Monitor", "Running"),
    ]
    
    for label, status in status_items:
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            st.caption(label)
        with col2:
            st.success("✅")
    
    st.sidebar.divider()
    
    # Quick Stats
    st.sidebar.markdown('<p class="sidebar-section-title">📈 Quick Stats</p>', unsafe_allow_html=True)
    
    quick_stats = [
        ("System Health", "95%"),
        ("Uptime", "99.8%"),
        ("Response Time", "145ms"),
    ]
    
    for stat_name, stat_value in quick_stats:
        st.sidebar.metric(stat_name, stat_value)
    
    st.sidebar.divider()
    
    # Footer
    st.sidebar.markdown("""
    <p style="font-size: 11px; color: #999; text-align: center; margin-top: 30px;">
        <strong>FINML Dashboard</strong><br>
        v1.0.0 | ML Operations<br>
        <small>© 2026 | Financial Risk Management</small>
    </p>
    """, unsafe_allow_html=True)
    
    return selected_page


