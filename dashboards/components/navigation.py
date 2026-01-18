"""Navigation utilities for dashboard pages."""

import streamlit as st


def render_back_button(page_key="home"):
    """Render a back-to-home button in the top-right corner."""
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col3:
        if st.button("🏠 Home", key=f"back_{page_key}", help="Return to home page"):
            st.session_state.current_page = "home"
            st.rerun()


def render_page_nav_header(title, description=""):
    """Render page header with back button."""
    col1, col2 = st.columns([5, 1])
    
    with col1:
        st.markdown(f"# {title}")
        if description:
            st.markdown(f"*{description}*")
    
    with col2:
        if st.button("🏠 Home", key=f"back_{title}", help="Return to home page"):
            st.session_state.current_page = "home"
            st.rerun()
