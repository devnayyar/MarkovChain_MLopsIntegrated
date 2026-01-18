"""Comprehensive Documentation and User Guide for Financial Risk Markov MLOps Dashboard."""

import streamlit as st
from dashboards.components.header import render_header


def render_documentation():
    """Render the comprehensive documentation page."""
    render_header("📚 Documentation & User Guide")
    
    st.markdown("""
    # Financial Risk Markov MLOps Dashboard - Complete Guide
    
    Welcome to the Financial Risk Markov MLOps Dashboard! This guide will help you understand every aspect of this application,
    what it does, and how to interpret the visualizations and metrics you see.
    """)
    
    # Main sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Overview", 
        "📊 Pages Guide", 
        "📈 Concepts", 
        "🔍 Metrics Explained",
        "💡 Interpretation Guide",
        "❓ FAQ"
    ])
    
    with tab1:
        render_overview_section()
    
    with tab2:
        render_pages_guide_section()
    
    with tab3:
        render_concepts_section()
    
    with tab4:
        render_metrics_section()
    
    with tab5:
        render_interpretation_section()
    
    with tab6:
        render_faq_section()


def render_overview_section():
    """Render the Overview section."""
    st.header("🎯 What is This Dashboard?")
    
    st.markdown("""
    ## Purpose
    
    The **Financial Risk Markov MLOps Dashboard** is a machine learning operations (MLOps) platform designed to:
    
    1. **Monitor Financial Risk**: Track economic indicators and classify financial market regimes (Low Risk, Moderate Risk, High Risk)
    2. **Predict Regime Transitions**: Use Markov Chain models to forecast future financial market states
    3. **Analyze Historical Data**: Examine raw data, processed features, and model outputs across three data layers (Bronze, Silver, Gold)
    4. **Validate Model Performance**: Display model metrics (AIC, BIC, Perplexity) and diagnostics
    5. **Support Model Experimentation**: Run interactive simulations and compare different model configurations
    
    ## Key Components
    
    ### Data Pipeline (3-Layer Architecture)
    
    The system uses a **medallion architecture** to organize data quality:
    
    - **🥉 Bronze Layer**: Raw, unprocessed economic data directly from sources
      - Contains raw indicator values without transformation
      - Data quality may vary (includes nulls, outliers)
      - Used for data quality assessment and anomaly detection
    
    - **🥈 Silver Layer**: Cleaned, validated, and feature-engineered data
      - Raw data cleaned, deduplicated, normalized
      - New features created (state representations, moving averages, derived metrics)
      - Ready for machine learning models
    
    - **🥇 Gold Layer**: Processed features with model predictions
      - Silver data with regime classifications
      - Contains Markov chain model outputs
      - Final model-ready dataset for serving and analysis
    
    ### Economic Indicators Tracked
    
    The system monitors these key economic indicators:
    
    - **UNRATE**: Unemployment Rate (%) - Measures job market health
    - **FEDFUNDS**: Federal Funds Rate (%) - Monetary policy indicator
    - **CPI_YOY**: Consumer Price Index Year-over-Year change (%) - Inflation measure
    - **T10Y2Y**: 10-Year minus 2-Year Treasury Yield Spread (%) - Yield curve slope (inverted = recession signal)
    - **STLFSI4**: St. Louis Fed Financial Stress Index - Overall financial market stress
    
    ### Risk Regimes
    
    The model classifies financial conditions into three regimes:
    
    - 🟢 **LOW_RISK**: Stable economic conditions, low financial stress
    - 🟡 **MODERATE_RISK**: Transitional conditions, elevated but manageable stress
    - 🔴 **HIGH_RISK**: Elevated stress, recession signals, heightened market volatility
    
    ## Technology Stack
    
    - **Python**: Core programming language
    - **Streamlit**: Interactive dashboard framework
    - **Pandas/NumPy**: Data processing and numerical computing
    - **Plotly**: Interactive visualizations
    - **scikit-learn**: Machine learning utilities
    - **MLflow**: Experiment tracking and model registry
    - **Docker**: Containerization for deployment
    """)


def render_pages_guide_section():
    """Render the Pages Guide section."""
    st.header("📊 Page-by-Page Guide")
    
    st.markdown("## Dashboard Pages Overview")
    
    # Dashboard Overview
    st.subheader("📊 Dashboard Overview")
    st.markdown("""
    **What it shows:**
    - High-level summary metrics of your financial data
    - Current model status and latest predictions
    - Key statistics from the most recent data point
    - Quick access to navigate other pages
    
    **Why this matters:**
    - Gives you at-a-glance understanding of current financial conditions
    - Shows if the system is healthy and model is up-to-date
    - Provides context before diving into detailed analysis
    """)
    
    # EDA Analysis
    st.subheader("🔍 EDA Analysis (Exploratory Data Analysis)")
    st.markdown("""
    **Three Sub-tabs:**
    
    ### 🥉 Bronze Layer Analysis
    
    **What it shows:**
    - **Trend Charts**: Line charts showing raw economic indicators over time
      - Each chart displays the indicator value and a 30-day moving average (red dashed line)
      - Shaded area under the line shows the magnitude of values
    - **Data Quality Metrics**: 
      - Total records in selected date range
      - Missing value percentage (lower is better)
      - Average values of key indicators
    - **Data Quality Table**: Column-by-column breakdown of data completeness
    
    **Why this matters:**
    - Understand data quality and completeness
    - Identify missing values or data gaps
    - See raw trends before any transformation
    - The moving average helps smooth out daily noise to see true trends
    
    **How to interpret:**
    - ↗️ **Upward trend**: Indicator is increasing (may signal changing conditions)
    - ↙️ **Downward trend**: Indicator is decreasing
    - **Spikes**: Sudden changes often indicate significant events
    - **Missing values**: If high, the data may need cleaning
    
    ---
    
    ### 🥈 Silver Layer Analysis
    
    **What it shows:**
    - **Feature Correlation Heatmap**: Shows relationships between all processed features
      - Red = negative correlation (inversely related)
      - Blue = positive correlation (move together)
      - Numbers = correlation strength (-1 to +1)
    - **Processed Data Statistics**: 
      - Count of cleaned records
      - Number of engineered features
      - Date range coverage
    - **State Distribution Charts**: How often each regime state occurs
    
    **Why this matters:**
    - Understand how features relate to each other
    - Identify redundant features (high correlation)
    - Verify feature engineering worked correctly
    - See which indicators move together
    
    **How to interpret:**
    - **Correlation ~1.0 (dark blue)**: Strong positive correlation - indicators move together
    - **Correlation ~-1.0 (dark red)**: Strong negative correlation - indicators move opposite
    - **Correlation ~0 (white)**: No clear relationship
    - **Self-correlation (diagonal) = 1.0**: Expected (everything correlates perfectly with itself)
    
    ---
    
    ### 🥇 Gold Layer Analysis
    
    **What it shows:**
    - **Regime Timeline**: Visual timeline showing when each regime was active
      - 🟢 Green dots = LOW_RISK periods
      - 🟡 Orange dots = MODERATE_RISK periods
      - 🔴 Red dots = HIGH_RISK periods
    - **Regime Distribution**: 
      - Pie chart showing % of time in each regime
      - Statistics table with counts and dates
      - Risk percentages (High/Moderate/Low)
    - **Regime Transitions**: How many times the model switched regimes
      - Top transitions show most common regime changes
    
    **Why this matters:**
    - Understand the historical frequency of risk regimes
    - See periods of high market stress
    - Identify transition patterns (which regimes lead to which)
    - Assess data coverage (are all regimes represented?)
    
    **How to interpret:**
    - **Regime Timeline**: Clusters of one color = stable regime; rapid color changes = volatile period
    - **Pie Chart**: If dominated by one color, that regime is most common historically
    - **Percentages**: 
      - High HIGH_RISK %: More historical stress periods
      - High LOW_RISK %: More stable historical periods
    - **Transitions**: Popular transitions (e.g., LOW_RISK → MODERATE_RISK) show natural flow
    """)
    
    # Model Metrics
    st.subheader("📊 Model Metrics & Diagnostics")
    st.markdown("""
    **What it shows:**
    - **Model Quality Metrics**: 
      - AIC (Akaike Information Criterion) - Goodness of fit with penalty for complexity
      - BIC (Bayesian Information Criterion) - Similar to AIC but stronger complexity penalty
      - Perplexity - How "surprised" the model is by test data (lower = better)
      - Log-Likelihood - Model's confidence in predictions
    - **Transition Matrix**: 3x3 table showing probability of moving from one regime to another
      - Rows = current regime
      - Columns = next regime
      - Values = probability (0-1)
    - **Regime Statistics**: Distribution of regimes in training data
    - **Model Weights**: Stationary distribution of regimes (long-run expected frequency)
    
    **Why this matters:**
    - AIC/BIC: Lower is better - compare different models
    - Perplexity: Tells how well model predicts unseen data
    - Transition Matrix: Core of Markov model - shows market dynamics
    - Model Weights: What the model "expects" in steady state
    
    **How to interpret:**
    - **Transition Matrix Example**: If LOW_RISK → LOW_RISK = 0.8, market tends to stay stable
      - If LOW_RISK → HIGH_RISK = 0.1, abrupt shifts are less likely
      - If HIGH_RISK → LOW_RISK = 0.05, hard to exit stress regimes
    - **Perplexity**: Compare across different model versions
      - Lower perplexity = model makes better predictions
    - **Model Weights**: If [0.44, 0.55, 0.01], HIGH_RISK is most likely long-term
    """)
    
    # Experiment Runner
    st.subheader("🔮 Experiment Runner & Model Testing")
    st.markdown("""
    **Three Interactive Modes:**
    
    ### 1️⃣ Single Prediction
    
    **What you do:**
    - Select current regime (LOW_RISK, MODERATE_RISK, HIGH_RISK)
    - System predicts next regime probabilities
    - Choose how many steps ahead to forecast (1-24 steps)
    
    **What you see:**
    - **Next Regime Probabilities**: Bar chart showing probability of each next state
    - **Forecast Path**: Step-by-step prediction showing most likely regime at each step
    - **Detailed Predictions**: Table with probabilities and confidence scores
    
    **Why use this:**
    - Test what happens if we're currently in a specific regime
    - Forecast short-term market moves
    - Understand model's regime transition preferences
    - Validate model makes sense for your scenario
    
    **Example interpretation:**
    - "If we're LOW_RISK today, model says 80% chance we stay LOW_RISK, 15% MODERATE, 5% HIGH"
    
    ---
    
    ### 2️⃣ Regime Sequence (Monte Carlo Forecast)
    
    **What you do:**
    - Select starting regime
    - Choose forecast horizon (7-180 days)
    - System runs 1,000 simulations of regime paths
    
    **What you see:**
    - **Probability Timeline**: Shows how regime probabilities evolve over time
      - Each line = different regime
      - Shows how confident the model is about future states
    - **Forecast Summary**: Statistics on regime transitions, most likely final regime
    
    **Why use this:**
    - Understand uncertainty over longer periods
    - See how regimes diverge over time
      - Quick convergence = predictable behavior
      - Divergence = unpredictable/chaotic
    - Assess long-term financial stability
    
    **Example interpretation:**
    - If LOW_RISK probability drops from 100% to 40% over 30 days, market becomes more uncertain
    - If it stays 90%+ over 180 days, very stable regime
    
    ---
    
    ### 3️⃣ Model Comparison (Base vs Challenger)
    
    **What you do:**
    - Select starting regime
    - System compares:
      - **Base Model**: Current production model
      - **Challenger Model**: Version with Laplace smoothing (handles rare transitions better)
    
    **What you see:**
    - **Comparison Metrics**:
      - Entropy: Model uncertainty (higher = more uniform predictions)
      - KL Divergence: How different two models are (higher = more different)
    - **Probability Comparison**: Bar chart comparing predictions side-by-side
    - **Detailed Comparison**: Full prediction tables for both models
    
    **Why use this:**
    - Test if new model versions improve predictions
    - See if smoothing helps with rare events
    - Decide which model to deploy
    
    **How to interpret:**
    - **High KL Divergence**: Models disagree significantly
      - If new model has much higher confidence, check if it's overfitting
      - If new model is more uniform, it's more conservative
    - **Entropy**: Lower = model is confident, Higher = model is uncertain
    """)


def render_concepts_section():
    """Render the Concepts section."""
    st.header("📈 Key Concepts Explained")
    
    st.markdown("""
    ## Markov Chain
    
    **What it is:**
    A mathematical model for predicting future states based on the current state only (not history).
    
    **Why we use it:**
    - Regime transitions in financial markets follow patterns
    - If we're in HIGH_RISK, the probability of next regime depends only on being HIGH_RISK
    - Simple, interpretable, and often surprisingly accurate
    
    **In practice:**
    ```
    Current Regime: LOW_RISK
    → Transition probabilities: LOW_RISK (80%), MODERATE_RISK (15%), HIGH_RISK (5%)
    → Next state predicted based solely on current regime, not how long we've been here
    ```
    
    ---
    
    ## Transition Matrix
    
    **What it is:**
    A table showing the probability of moving from each regime to every other regime.
    
    **Example:**
    ```
                 To: LOW_RISK  MODERATE  HIGH_RISK
    From: LOW_RISK    0.80      0.15      0.05
          MODERATE    0.30      0.40      0.30
          HIGH_RISK   0.05      0.20      0.75
    ```
    
    **Interpretation:**
    - From LOW_RISK: 80% stay low, 15% go moderate, 5% go high
    - From MODERATE: 30% improve, 40% stay, 30% worsen
    - From HIGH_RISK: 75% stay high (hard to escape), 20% go moderate, 5% go low
    
    **Why it matters:**
    - Core of the entire prediction system
    - Shows market dynamics and risk propagation
    - Used to forecast future regimes
    
    ---
    
    ## Stationary Distribution (Model Weights)
    
    **What it is:**
    If we run the model forever, what's the long-term proportion of time spent in each regime?
    
    **Example:**
    ```
    Model Weights: LOW_RISK (44%), MODERATE_RISK (0.3%), HIGH_RISK (55%)
    ```
    
    **Interpretation:**
    - Over very long periods, market spends ~44% in LOW_RISK, ~55% in HIGH_RISK
    - Very little time in MODERATE_RISK (transition state)
    - Suggests HIGH_RISK is "attractor" - once in it, takes longer to leave
    
    **Why it matters:**
    - Different from historical frequency - accounts for transition dynamics
    - Useful for long-term financial planning
    - Shows equilibrium distribution if nothing changes
    
    ---
    
    ## Regime (State)
    
    **What it is:**
    A classification of overall financial market conditions into discrete categories.
    
    **Why three regimes:**
    - Captures major market phases: stable, transitional, crisis
    - Simple enough to understand and explain
    - Complex enough to capture real market behavior
    
    **How regimes are assigned:**
    - Based on combination of all 5 economic indicators
    - Algorithm combines indicators into a "financial stress score"
    - Threshold: stress score determines which regime
    
    ---
    
    ## Data Layers (Medallion Architecture)
    
    **Bronze → Silver → Gold transformation:**
    
    ```
    BRONZE (Raw Data)
      ↓ Clean, normalize, deduplicate
    SILVER (Processed Data)
      ↓ Add features, compute states, prepare for ML
    GOLD (Model Ready)
      ↓ Add model predictions, regime labels
    ```
    
    **Benefits:**
    - Traceable: Can audit each transformation step
    - Scalable: Each layer can handle different volume
    - Secure: Can restrict access to gold layer for production
    - Maintainable: Bug in silver layer? Fix once, affects all downstream
    """)


def render_metrics_section():
    """Render the Metrics section."""
    st.header("🔍 Model Metrics Deep Dive")
    
    st.markdown("""
    ## Information Criteria (AIC & BIC)
    
    **What they measure:**
    Balance between model accuracy and complexity. Lower = better.
    
    **AIC (Akaike Information Criterion)**
    - Formula: `AIC = 2k - 2ln(L)` where k = parameters, L = likelihood
    - Penalizes complexity less than BIC
    - Better when you have lots of data and want more flexible model
    - Used for prediction accuracy
    
    **BIC (Bayesian Information Criterion)**
    - Formula: `BIC = k·ln(n) - 2ln(L)` where n = sample size
    - Stronger penalty for complexity (grows with data size)
    - Prefers simpler models (more conservative)
    - Used for explaining data
    
    **When to use each:**
    - Use AIC: Prioritize prediction accuracy on new data
    - Use BIC: Prioritize finding true underlying model structure
    
    **Practical interpretation:**
    ```
    Model A: AIC = 1000, BIC = 1100
    Model B: AIC = 1050, BIC = 1080  ← Better BIC (simpler)
    
    Which to choose?
    - If prediction matters: Model A
    - If interpretability matters: Model B
    ```
    
    ---
    
    ## Perplexity
    
    **What it measures:**
    Average number of "equally likely" outcomes the model thinks will happen.
    Lower = model more certain about predictions.
    
    **Interpretation:**
    - Perplexity = 3.0: Model thinks there are ~3 equally likely next states
    - Perplexity = 2.1: Model thinks outcome is ~more predictable
    - Perplexity = 1.0: Model 100% certain (perfect prediction)
    
    **Formula:**
    ```
    Perplexity = 2^(Cross-Entropy) = exp(-ln(Probability of data))
    ```
    
    **Why it matters:**
    - Directly interpretable (geometric average of probabilities)
    - Good for comparing models on same test set
    - Lower is always better
    
    **Benchmark:**
    - Perplexity < 2: Excellent predictions
    - Perplexity 2-3: Good predictions
    - Perplexity > 3: Model struggles with uncertainty
    
    ---
    
    ## Log-Likelihood
    
    **What it measures:**
    How likely the observed data is under the model's probability distribution.
    Higher = better (model thinks data is likely).
    
    **Interpretation:**
    - Log-Likelihood = -0.24: Model assigns probability ~0.78 to actual observations
    - Log-Likelihood = -1.0: Model assigns probability ~0.37 to actual observations
    - Log-Likelihood = 0: Model assigns probability 1.0 (impossible, overfitting)
    
    **Why log-scale:**
    - Probabilities multiply; logs turn multiplication to addition
    - Prevents numerical underflow (0.1 × 0.1 × ... loses precision)
    - Easier to work with mathematically
    
    **Practical meaning:**
    ```
    If Log-Likelihood = -10:
    - Probability of observing actual data = e^(-10) ≈ 0.0000454
    - Model thinks observations are very unlikely (bad fit)
    
    If Log-Likelihood = -0.5:
    - Probability of observing actual data = e^(-0.5) ≈ 0.606
    - Model thinks observations are quite likely (good fit)
    ```
    
    ---
    
    ## Entropy (in Experiment Runner)
    
    **What it measures:**
    Uncertainty in model's predictions (randomness of probability distribution).
    
    **Interpretation:**
    - Entropy = 0: Model 100% certain (gives 100% to one outcome)
    - Entropy = 1.0 (max for 3 states): Model equally uncertain about all 3 outcomes
    - Entropy = 0.5: Model somewhat confident
    
    **Formula:**
    ```
    Entropy = -Σ(p_i × ln(p_i)) for each regime i
    ```
    
    **Example:**
    - Model predicts: [0.8, 0.1, 0.1] → Low entropy (confident)
    - Model predicts: [0.5, 0.3, 0.2] → Medium entropy (less certain)
    - Model predicts: [0.33, 0.33, 0.33] → High entropy (no preference)
    
    ---
    
    ## KL Divergence (Kullback-Leibler Divergence)
    
    **What it measures:**
    How different one probability distribution is from another.
    Always ≥ 0, equals 0 only when distributions identical.
    
    **Formula:**
    ```
    KL(P||Q) = Σ P(i) × ln(P(i) / Q(i))
    
    P = True distribution (or Base Model)
    Q = Approximation (or Challenger Model)
    ```
    
    **Interpretation:**
    - KL ≈ 0: Distributions very similar (models agree)
    - KL = 0.1: Small difference
    - KL = 1.0: Large difference (models disagree significantly)
    
    **Example in practice:**
    ```
    Base Model predicts: [0.7, 0.2, 0.1]
    Challenger predicts: [0.6, 0.3, 0.1]
    
    KL ≈ 0.028: Very similar - models mostly agree, can use either
    
    ---
    
    Base Model predicts: [0.8, 0.1, 0.1]
    Challenger predicts: [0.33, 0.33, 0.33]
    
    KL ≈ 0.83: Very different - models disagree, need to decide which to trust
    ```
    
    **How to use it:**
    - KL < 0.05: Models interchangeable
    - KL 0.05-0.3: Models have different philosophies but both reasonable
    - KL > 0.3: Major differences, investigate which is better
    """)


def render_interpretation_section():
    """Render the Interpretation Guide section."""
    st.header("💡 How to Interpret Visualizations")
    
    st.markdown("""
    ## Trend Charts (Bronze & Silver Analysis)
    
    **What to look for:**
    
    1. **Trend Direction**
       - ↗️ Steep upward slope: Rapid increase in indicator value
       - ↙️ Steep downward slope: Rapid decrease
       - → Flat line: Stable/no change
    
    2. **Moving Average (Red Dashed Line)**
       - Smooths out daily noise to show true trend
       - If price above MA: Short-term above long-term (bullish signal)
       - If price below MA: Short-term below long-term (bearish signal)
       - MA crossing through price: Trend reversal
    
    3. **Volatility (Spacing from MA)**
       - Tight spacing: Low volatility, predictable
       - Large spacing: High volatility, unpredictable
    
    4. **Shaded Area Under Curve**
       - Visual indication of magnitude
       - Larger area = larger values overall
    
    **Example Analysis:**
    ```
    Unemployment (UNRATE) chart shows:
    - Rising trend from 3.5% to 5.0% over 6 months
    - Price often above MA (concerning)
    - Spacing from MA increasing (rising volatility)
    
    Interpretation: Employment weakening rapidly, market stress increasing
    ```
    
    ---
    
    ## Correlation Heatmap (Silver Analysis)
    
    **Color scale:**
    - 🔵 Dark Blue = +1.0 (perfect positive correlation)
    - ⚪ White = 0 (no correlation)
    - 🔴 Dark Red = -1.0 (perfect negative correlation)
    
    **What to look for:**
    
    1. **Strong Positive Correlations (Dark Blue)**
       - Features move together
       - Only include one in model (redundant)
       - Example: CPI_YOY and FEDFUNDS often correlated
    
    2. **Strong Negative Correlations (Dark Red)**
       - Features move opposite
       - Both provide different information (complementary)
       - Example: T10Y2Y and FEDFUNDS (inverse relationship)
    
    3. **No Correlation (White)**
       - Features independent
       - Very useful - each adds unique information
    
    **Pattern Recognition:**
    - Row of blues: Feature moves with many others (captures broad trend)
    - Row of reds: Feature moves opposite to others (contrarian indicator)
    - Mix of colors: Complex, non-linear relationships
    
    ---
    
    ## Regime Timeline (Gold Analysis)
    
    **Visual interpretation:**
    
    1. **Horizontal Position**: Time axis - earlier on left, recent on right
    
    2. **Vertical Dots**: Each dot = regime at that point in time
       - 🟢 Green = LOW_RISK
       - 🟡 Orange = MODERATE_RISK
       - 🔴 Red = HIGH_RISK
    
    3. **Density**: 
       - Clustered dots = regime stable
       - Scattered dots = regime unstable/transitioning
    
    4. **Patterns**:
       - Long green section: Extended stable period (good)
       - Rapid color changes: Volatile market (risky)
       - Horizontal bands: Market stuck in one regime
    
    **Example Reading:**
    ```
    Timeline shows: GREEN (6 months) → RED (3 months) → ORANGE → GREEN
    
    Interpretation:
    - Stable period, then crisis (3 months of high stress), recovery period
    - Market went through typical stress-recovery cycle
    - Important to note: 3 months is long for HIGH_RISK regime
    ```
    
    ---
    
    ## Transition Matrix Heatmap (Model Metrics)
    
    **Layout:**
    - Rows = "From" regime (current state)
    - Columns = "To" regime (next state)
    - Colors darker = higher probability
    
    **How to read:**
    ```
    Row "LOW_RISK":
    - LOW_RISK column = 0.80 (80% stay low) → Dark color
    - MODERATE column = 0.15 (15% → moderate) → Light color
    - HIGH_RISK column = 0.05 (5% → high) → Very light color
    
    Interpretation: If LOW_RISK, very likely to stay, unlikely to jump to HIGH_RISK
    ```
    
    **Key patterns:**
    - **Diagonal dominance**: Each regime stays itself (stability)
    - **Off-diagonal zeros**: Certain transitions impossible
    - **Symmetric patterns**: Reversible transitions (A→B probability = B→A probability)
    - **High escape probability**: Hard to stay in regime
    
    ---
    
    ## Probability Timeline (Monte Carlo Forecast)
    
    **What it shows:**
    How each regime's probability changes over forecast horizon
    
    **Interpretation:**
    
    1. **Line trajectory**
       - Rising line: Regime becoming more likely
       - Falling line: Regime becoming less likely
    
    2. **Convergence**
       - Lines converging together: Model losing confidence
       - Lines staying separate: Model confident in future
    
    3. **Final heights**
       - Where lines end = model's belief about long-term distribution
       - Matches "Stationary Distribution" if long enough horizon
    
    **Example:**
    ```
    30-day forecast starting from LOW_RISK:
    - LOW_RISK line: 100% → 60% (confidence declining)
    - HIGH_RISK line: 0% → 20% (risk rising)
    
    Interpretation: Over month, market likely transitions; uncertainty growing
    ```
    """)


def render_faq_section():
    """Render the FAQ section."""
    st.header("❓ Frequently Asked Questions")
    
    with st.expander("🎯 What should I do with this dashboard?"):
        st.markdown("""
        Use it to:
        - **Monitor**: Check current financial regime daily/weekly
        - **Plan**: Look at forecasts to anticipate regime changes
        - **Validate**: Ensure model makes business sense for your decisions
        - **Debug**: Investigate data quality and model behavior
        - **Communicate**: Show stakeholders transparent, explainable AI predictions
        """)
    
    with st.expander("📊 How often is the data updated?"):
        st.markdown("""
        This depends on your data pipeline configuration:
        - Daily batch updates are typical
        - Some indicators available weekly or monthly
        - Check your data ingestion schedule for specifics
        - Bronze layer always has latest raw data
        - Model predictions (Gold layer) updated on next pipeline run
        """)
    
    with st.expander("🤔 Which regime should I expect?"):
        st.markdown("""
        Based on historical data:
        - **Typical distribution**: ~44% LOW_RISK, ~55% HIGH_RISK, ~1% MODERATE_RISK
        - **HIGH_RISK is normal**: Doesn't mean crisis, just elevated stress
        - **MODERATE_RISK is rare**: Transition state, usually short-lived
        - **Pattern**: Markets spend less time stable than stressed
        
        This reflects real financial markets: more periods of uncertainty than stability.
        """)
    
    with st.expander("🔮 Why are predictions wrong sometimes?"):
        st.markdown("""
        Expected reasons:
        - **Black swan events**: Unprecedented events model hasn't seen
        - **Policy changes**: Central bank actions can override historical patterns
        - **Market structure changes**: New products/trading strategies alter dynamics
        - **Data issues**: Missing/wrong data in indicators
        - **Model limitations**: Markov model assumes future depends only on present state
        
        **What's NOT a reason:**
        - Markets are unpredictable: Model gives probabilities, not certainties
        - Trends change: Long-term trends are captured; short-term can be noisy
        - I don't understand it: Use this documentation or ask a data scientist
        """)
    
    with st.expander("📈 How do I know if the model is good?"):
        st.markdown("""
        Check these signals:
        
        **Good signs:**
        - ✅ Perplexity < 2.5 (model confident)
        - ✅ AIC/BIC low and stable (consistent performance)
        - ✅ Transition matrix shows sensible patterns
        - ✅ Regime timeline matches known market events
        - ✅ Predictions match expert intuition
        
        **Warning signs:**
        - ⚠️ Perplexity > 3.0 (too uncertain)
        - ⚠️ AIC/BIC increasing (model degrading)
        - ⚠️ Strange transitions (e.g., always LOW_RISK)
        - ⚠️ Timeline doesn't match real events
        - ⚠️ Predictions contradict known market drivers
        
        If seeing warnings, investigate data quality and model configuration.
        """)
    
    with st.expander("🆚 Base vs Challenger Model - which to use?"):
        st.markdown("""
        Decision framework:
        
        **Use Base Model if:**
        - In production and working well (don't fix what isn't broken)
        - KL divergence to challenger < 0.05 (essentially identical)
        - Base has better perplexity (lower error)
        - Business doesn't want change
        
        **Use Challenger Model if:**
        - Significantly better perplexity (lower is better)
        - KL > 0.05 but challenger has better performance on new data
        - Want to handle rare transitions better (Laplace smoothing)
        - Base model has known issues
        - A/B testing shows challenger performs better in production
        
        **Never use:**
        - Model with perplexity > 5 (too unreliable)
        - Model that contradicts domain knowledge
        - Model without backtesting on recent data
        """)
    
    with st.expander("📅 How far ahead can I forecast?"):
        st.markdown("""
        - **1-7 days**: High confidence (regimes change slowly)
        - **8-30 days**: Medium confidence (divergence starting)
        - **1-3 months**: Low confidence (uncertainty high)
        - **> 3 months**: Only use stationary distribution
        
        Why?
        - Markov models are accurate near term (short-term regime likely to continue)
        - As we look further out, uncertainty compounds
        - Eventually, forecast converges to stationary distribution
        - At extreme horizons (years), only stationary distribution matters
        """)
    
    with st.expander("🔍 How do I validate the data?"):
        st.markdown("""
        Steps to verify data quality:
        
        1. **Check Bronze Layer**:
           - Go to EDA Analysis → Bronze Layer
           - Look for red flags:
             - Missing values > 5%: Data collection problem
             - Sudden spikes/drops: Data error or real event?
             - Dates with gaps: Data infrastructure issue
        
        2. **Check Silver Layer**:
           - Look at correlation heatmap
           - Expected patterns:
             - Unemployment and Fed Rates: Often correlated
             - CPI and Fed Rates: Usually correlated
             - Yield curve should invert in recessions
           - Unexpected patterns: Data transformation error
        
        3. **Check Gold Layer**:
           - Regime timeline should match known events
             - 2008 financial crisis: Should be HIGH_RISK
             - 2020 COVID crash: Should be HIGH_RISK
             - 2021-2022: Should show HIGH_RISK then improvement
           - If doesn't match, either data or model problem
        
        4. **Check Model Metrics**:
           - AIC/BIC should be reasonable (not astronomically high)
           - Perplexity 1-3: Good, >5: Concerning
           - Log-likelihood negative (expected) but not extreme (<-100: very bad)
        """)
    
    with st.expander("💬 I don't understand something..."):
        st.markdown("""
        Resources to help:
        
        1. **Read this documentation**: Most answers here
        2. **Check the relevant page**: Each page explains its charts
        3. **Use the Experiment Runner**: Test predictions interactively
        4. **Compare models**: See how predictions change
        5. **Talk to your data science team**: They built this!
        
        Don't hesitate to ask - this is complex stuff, and good questions improve the system.
        """)
    
    st.markdown("---")
    st.markdown("""
    ## Contact & Support
    
    For questions about:
    - **Data quality**: Contact your data engineering team
    - **Model accuracy**: Contact your data science team  
    - **Dashboard issues**: Contact your ML operations team
    - **Business interpretation**: Contact your risk management team
    
    ---
    
    **Last Updated**: January 2026
    **Dashboard Version**: 1.0
    **Model Version**: Production (Markov Chain, 3-state regime)
    """)


# Run the page
render_documentation()
