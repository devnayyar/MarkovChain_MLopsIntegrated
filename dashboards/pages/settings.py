"""Settings page."""

import streamlit as st


def render_settings():
    """Render the Settings page."""
    st.title("⚙️ Settings")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Display Settings")
        
        theme = st.selectbox("Theme", ["Auto", "Light", "Dark"])
        refresh_interval = st.selectbox("Refresh Interval", ["30 seconds", "1 minute", "5 minutes", "Manual"])
        chart_type = st.selectbox("Chart Type", ["Line", "Area", "Bar"])
        decimal_places = st.slider("Decimal Places", 1, 6, 2)
        timezone = st.selectbox("Time Zone", ["UTC", "EST", "CST", "PST"])
    
    with col2:
        st.subheader("🔔 Alert Preferences")
        
        notification_channel = st.multiselect(
            "Notification Channels",
            ["Console", "Email", "Slack", "PagerDuty"],
            default=["Console"]
        )
        
        alert_severity = st.selectbox(
            "Alert Severity Threshold",
            ["All", "Warning & Critical", "Critical Only"]
        )
        
        if "Email" in notification_channel:
            email_recipients = st.text_area("Email Recipients", value="user@example.com")
        
        if "Slack" in notification_channel:
            slack_webhook = st.text_input("Slack Webhook URL", type="password")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Data Export Options")
        
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Parquet"])
        include_charts = st.checkbox("Include Charts in Export", value=True)
        compression = st.selectbox("Compression", ["None", "Gzip", "Zip"])
    
    with col2:
        st.subheader("ℹ️ System Information")
        
        st.metric("Last MLflow Sync", "Just now")
        st.metric("Database Version", "1.0.0")
        st.metric("API Status", "✅ Connected")
        
        st.write("**Enabled Features**")
        features = [
            "Regime Timeline Analysis",
            "Alert Monitoring",
            "Performance Tracking",
            "A/B Testing",
            "Auto Retraining",
        ]
        for feature in features:
            st.caption(f"✓ {feature}")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            st.info("Settings reset to default values")
    
    with col3:
        if st.button("📥 Export Settings"):
            st.success("Settings exported!")
    
    st.divider()
    
    # Help section
    st.subheader("❓ Help & Support")
    
    with st.expander("How do I change my notification preferences?"):
        st.write("Go to the Alert Preferences section above and select your preferred notification channels.")
    
    with st.expander("What file formats are supported for export?"):
        st.write("CSV, JSON, and Parquet formats are supported. You can also choose compression options.")
    
    with st.expander("How often is the data refreshed?"):
        st.write("Data refresh interval can be customized in the Display Settings section. Choose from 30 seconds to manual refresh.")
    
    with st.expander("Where can I find the API documentation?"):
        st.write("[View API Documentation](https://docs.example.com)")
