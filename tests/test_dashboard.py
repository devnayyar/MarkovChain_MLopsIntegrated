#!/usr/bin/env python
"""Test dashboard installation and imports."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

print("Testing Dashboard Installation...")
print("=" * 60)

try:
    print("✓ Testing Streamlit import...")
    import streamlit as st
    print(f"  Streamlit version: {st.__version__}")
except ImportError as e:
    print(f"✗ Failed to import Streamlit: {e}")
    sys.exit(1)

try:
    print("✓ Testing Plotly import...")
    import plotly
    print(f"  Plotly version: {plotly.__version__}")
except ImportError as e:
    print(f"✗ Failed to import Plotly: {e}")
    sys.exit(1)

try:
    print("✓ Testing Pandas import...")
    import pandas as pd
    print(f"  Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"✗ Failed to import Pandas: {e}")
    sys.exit(1)

try:
    print("✓ Testing dashboard.utils.constants...")
    from dashboards.utils import constants
    print(f"  Loaded {len(constants.PAGES)} pages")
    for page_name, page_id in constants.PAGES.items():
        print(f"    - {page_name} ({page_id})")
except ImportError as e:
    print(f"✗ Failed to import dashboard constants: {e}")
    sys.exit(1)

try:
    print("✓ Testing dashboard.utils.formatters...")
    from dashboards.utils import formatters
    print(f"  Loaded formatting utilities")
except ImportError as e:
    print(f"✗ Failed to import dashboard formatters: {e}")
    sys.exit(1)

try:
    print("✓ Testing dashboard.utils.data_loader...")
    from dashboards.utils import data_loader
    print(f"  Loaded data loading utilities")
except ImportError as e:
    print(f"✗ Failed to import dashboard data_loader: {e}")
    sys.exit(1)

try:
    print("✓ Testing dashboard.components...")
    from dashboards.components import header, sidebar, metrics_card, status_indicator
    print(f"  Loaded all component modules")
except ImportError as e:
    print(f"✗ Failed to import dashboard components: {e}")
    sys.exit(1)

try:
    print("✓ Testing dashboard.pages...")
    from dashboards.pages import (
        home, regime_timeline, alerts_drift, 
        metrics_performance, retraining_ab_testing, settings
    )
    print(f"  Loaded all page modules")
except ImportError as e:
    print(f"✗ Failed to import dashboard pages: {e}")
    sys.exit(1)

try:
    print("✓ Testing dashboards.app...")
    from dashboards import app
    print(f"  Main app module loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import dashboard app: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All dashboard components loaded successfully!")
print("=" * 60)
print("\nTo run the dashboard:")
print(f"  streamlit run {project_root}/dashboards/app.py")
print("\nDashboard will be available at: http://localhost:8501")
