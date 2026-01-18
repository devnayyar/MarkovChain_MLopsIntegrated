"""Status indicator components."""

import streamlit as st
from dashboards.utils.constants import STATUS_COLORS


def status_badge(label, status):
    """
    Display a status badge.
    
    Args:
        label: Status label
        status: Status level (Good, Warning, Critical, Info)
    """
    color = STATUS_COLORS.get(status, STATUS_COLORS["Info"])
    
    badge_html = f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
    ">{label}</span>
    """
    
    st.markdown(badge_html, unsafe_allow_html=True)


def status_grid(items):
    """
    Display status items in a grid.
    
    Args:
        items: List of (name, status) tuples
    """
    cols = st.columns(3)
    
    for idx, (name, status) in enumerate(items):
        with cols[idx % 3]:
            color = STATUS_COLORS.get(status, STATUS_COLORS["Info"])
            
            st.markdown(f"""
            <div style="
                background-color: {color};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
            ">
                <p style="margin: 0; font-size: 14px; opacity: 0.9;">{name}</p>
                <p style="margin: 10px 0; font-size: 24px; font-weight: bold;">
                    {'✅' if status == 'Good' else '⚠️' if status == 'Warning' else '❌' if status == 'Critical' else 'ℹ️'}
                </p>
                <p style="margin: 0; font-size: 12px; opacity: 0.8;">{status}</p>
            </div>
            """, unsafe_allow_html=True)


def health_indicator(percentage):
    """
    Display system health indicator.
    
    Args:
        percentage: Health percentage (0-100)
    """
    if percentage >= 90:
        color = STATUS_COLORS["Good"]
        label = "Excellent"
    elif percentage >= 75:
        color = STATUS_COLORS["Info"]
        label = "Good"
    elif percentage >= 60:
        color = STATUS_COLORS["Warning"]
        label = "Fair"
    else:
        color = STATUS_COLORS["Critical"]
        label = "Poor"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {color} 0%, {color} {percentage}%, #e0e0e0 {percentage}%, #e0e0e0 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    ">
        <h3 style="margin: 0;">System Health</h3>
        <h2 style="margin: 10px 0;">{percentage}% - {label}</h2>
    </div>
    """, unsafe_allow_html=True)
