#!/usr/bin/env python3
"""
Train model with your gold data and prepare results for dashboard.
Uses your existing data structure (ONE-TO-ONE mapping).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Load your actual gold layer data
gold_path = Path("data/gold/markov_state_sequences.parquet")
df_gold = pd.read_parquet(gold_path)

print("\n" + "="*80)
print("🚀 FINML MODEL TRAINING - USING YOUR GOLD LAYER DATA")
print("="*80)
print(f"📍 Gold Data Path: {gold_path}")
print(f"📊 Data Shape: {df_gold.shape}")
print(f"📅 Date Range: {df_gold['date'].min()} to {df_gold['date'].max()}")

# Extract regime states (already discretized in your gold layer)
regime_states = df_gold['REGIME_RISK'].dropna().values
print(f"\n📈 REGIME STATES:")
print(f"   Available states: {sorted(set(regime_states))}")
print(f"   Total records with regime: {len(regime_states)}")

# Import and train your Markov model
try:
    from modeling.models.base_markov import MarkovChain
    
    print("\n" + "="*80)
    print("🤖 TRAINING MARKOV CHAIN MODEL")
    print("="*80)
    
    # Train on your regime states
    mc = MarkovChain(state_sequence=regime_states, states=sorted(set(regime_states)))
    mc.estimate_transition_matrix()
    
    # Calculate statistics
    stationary_dist = mc.get_stationary_distribution()
    log_likelihood = mc.log_likelihood()
    
    print(f"✅ Model trained successfully!")
    print(f"\n📊 Transition Matrix:")
    print(mc.transition_matrix)
    print(f"\n📊 Stationary Distribution: {stationary_dist}")
    print(f"📊 Log Likelihood: {log_likelihood:.4f}")
    
    # Prepare results for dashboard
    results = {
        "model_training": {
            "timestamp": datetime.now().isoformat(),
            "data_path": str(gold_path),
            "total_records": len(df_gold),
            "regime_records": len(regime_states),
            "date_range": {
                "start": str(df_gold['date'].min()),
                "end": str(df_gold['date'].max())
            }
        },
        "model_metrics": {
            "n_states": len(set(regime_states)),
            "states": sorted(set(regime_states)),
            "log_likelihood": float(log_likelihood),
            "stationary_dist": stationary_dist.tolist() if stationary_dist is not None else None
        },
        "transition_matrix": {
            "states": sorted(set(regime_states)),
            "matrix": mc.transition_matrix.tolist()
        },
        "regime_distribution": {
            state: int(sum(regime_states == state)) 
            for state in sorted(set(regime_states))
        },
        "economic_indicators": {
            "UNRATE": {
                "mean": float(df_gold['UNRATE'].mean()),
                "std": float(df_gold['UNRATE'].std()),
                "min": float(df_gold['UNRATE'].min()),
                "max": float(df_gold['UNRATE'].max())
            },
            "FEDFUNDS": {
                "mean": float(df_gold['FEDFUNDS'].mean()),
                "std": float(df_gold['FEDFUNDS'].std()),
                "min": float(df_gold['FEDFUNDS'].min()),
                "max": float(df_gold['FEDFUNDS'].max())
            },
            "CPI_YOY": {
                "mean": float(df_gold['CPI_YOY'].mean()),
                "std": float(df_gold['CPI_YOY'].std()),
                "min": float(df_gold['CPI_YOY'].min()),
                "max": float(df_gold['CPI_YOY'].max())
            },
            "T10Y2Y": {
                "mean": float(df_gold['T10Y2Y'].mean()),
                "std": float(df_gold['T10Y2Y'].std()),
                "min": float(df_gold['T10Y2Y'].min()),
                "max": float(df_gold['T10Y2Y'].max())
            }
        }
    }
    
    # Save results for dashboard
    dashboard_path = Path("monitoring/dashboard_data.json")
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dashboard_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*80)
    print("✅ RESULTS PREPARED FOR DASHBOARD")
    print("="*80)
    print(f"📍 Dashboard Data Saved: {dashboard_path}")
    print(f"\n📊 SUMMARY:")
    print(f"   • Total Economic Records: {len(df_gold)}")
    print(f"   • Regime Classifications: {len(regime_states)}")
    print(f"   • Unique Regimes: {', '.join(sorted(set(regime_states)))}")
    print(f"   • Model States: {len(set(regime_states))}")
    print(f"   • Log Likelihood: {log_likelihood:.4f}")
    print(f"\n🎯 Ready to view in Dashboard!")
    print(f"   Command: streamlit run dashboards/app.py\n")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
