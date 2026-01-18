"""EDA analysis for Gold layer regime-encoded business data."""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def analyze_gold_layer():
    """Analyze business-ready regime-encoded data."""
    gold_path = Path("data/gold/markov_state_sequences.parquet")
    
    if not gold_path.exists():
        print("Gold layer data not found. Skipping analysis.")
        return
    
    print("\n" + "="*80)
    print("GOLD LAYER ANALYSIS - Business-Ready Regime Data")
    print("="*80 + "\n")
    
    df = pd.read_parquet(gold_path)
    
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}\n")
    
    # Regime distribution
    print("REGIME RISK DISTRIBUTION:")
    print("-" * 60)
    regime_counts = df['REGIME_RISK'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {regime:20s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Individual state dimensions
    print("\nINDIVIDUAL STATE DISTRIBUTIONS:")
    print("-" * 60)
    state_cols = ['UNRATE_STATE', 'FEDFUNDS_STATE', 'CPI_YOY_STATE', 'T10Y2Y_STATE', 'STLFSI4_STATE']
    for col in state_cols:
        if col in df.columns:
            print(f"\n  {col}:")
            state_dist = df[col].value_counts()
            for state, count in state_dist.items():
                pct = count / len(df) * 100
                print(f"    {state:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Regime transitions
    print("\n" + "-"*60)
    print("REGIME TRANSITION ANALYSIS:")
    print("-" * 60)
    transitions = []
    for i in range(len(df) - 1):
        from_regime = df.iloc[i]['REGIME_RISK']
        to_regime = df.iloc[i+1]['REGIME_RISK']
        transitions.append(f"{from_regime} → {to_regime}")
    
    transition_counts = Counter(transitions)
    print(f"Total transitions: {len(transitions)}")
    print(f"Unique transition types: {len(transition_counts)}")
    print("\nTop transitions:")
    for trans, count in transition_counts.most_common(10):
        pct = count / len(transitions) * 100
        print(f"  {trans:50s}: {count:4d} ({pct:5.1f}%)")
    
    # Regime persistence
    print("\n" + "-"*60)
    print("REGIME PERSISTENCE ANALYSIS:")
    print("-" * 60)
    for regime in regime_counts.index:
        regime_data = df[df['REGIME_RISK'] == regime]
        consecutive_periods = []
        current_count = 1
        for i in range(1, len(regime_data)):
            if regime_data.iloc[i]['date'] == regime_data.iloc[i-1]['date'] + pd.Timedelta(days=31):
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_periods.append(current_count)
                current_count = 1
        
        if consecutive_periods:
            avg_duration = np.mean(consecutive_periods)
            max_duration = np.max(consecutive_periods)
            print(f"\n  {regime}:")
            print(f"    Average duration: {avg_duration:.2f} months")
            print(f"    Maximum duration: {max_duration} months")
            print(f"    Occurrences: {len(consecutive_periods)}")
    
    print("\n" + "="*80)
    print("✅ GOLD ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    analyze_gold_layer()
