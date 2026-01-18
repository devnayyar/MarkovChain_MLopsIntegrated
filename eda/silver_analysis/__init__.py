"""EDA analysis for Silver layer cleaned data."""
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_silver_layer():
    """Analyze cleaned and standardized monthly data."""
    silver_path = Path("data/silver/cleaned_macro_data.parquet")
    
    if not silver_path.exists():
        print("Silver layer data not found. Skipping analysis.")
        return
    
    print("\n" + "="*80)
    print("SILVER LAYER ANALYSIS - Cleaned & Standardized Monthly Data")
    print("="*80 + "\n")
    
    df = pd.read_parquet(silver_path)
    
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Frequency: Monthly (aligned to month-start)\n")
    
    print("Column Statistics:")
    print("-" * 60)
    for col in df.columns:
        if col == "date":
            continue
        if df[col].dtype in ['float64', 'int64']:
            print(f"\n  {col}:")
            print(f"    Type: {df[col].dtype}")
            print(f"    Nulls: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
            print(f"    Mean: {df[col].mean():.4f}")
            print(f"    Std:  {df[col].std():.4f}")
            print(f"    Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
            print(f"    Skew: {df[col].skew():.4f}")
            print(f"    Kurtosis: {df[col].kurtosis():.4f}")
    
    # Correlation analysis
    print("\n" + "-"*60)
    print("Correlation Matrix (numeric columns):")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    print(corr_matrix.round(3))
    
    # Missing data pattern
    print("\n" + "-"*60)
    print("Missing Data Pattern:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    print("\n" + "="*80)
    print("✅ SILVER ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    analyze_silver_layer()
