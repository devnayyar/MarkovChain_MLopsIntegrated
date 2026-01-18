"""EDA analysis for Bronze layer raw data."""
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_bronze_layer():
    """Analyze raw FRED data from bronze layer."""
    bronze_dir = Path("data/bronze")
    
    print("\n" + "="*80)
    print("BRONZE LAYER ANALYSIS - Raw FRED Data")
    print("="*80 + "\n")
    
    bronze_files = list(bronze_dir.glob("*.csv"))
    
    for file_path in sorted(bronze_files):
        print(f"\n📊 Dataset: {file_path.name}")
        print("-" * 60)
        
        df = pd.read_csv(file_path)
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"  Date Range: {df['DATE'].min()} to {df['DATE'].max()}")
        print(f"  Value Column: {df.columns[1]}")
        print(f"  Null Values: {df.isnull().sum().sum()}")
        print(f"  Null %: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%")
        
        # Stats
        value_col = df.columns[1]
        print(f"\n  Statistics for {value_col}:")
        print(f"    Mean: {df[value_col].mean():.4f}")
        print(f"    Std:  {df[value_col].std():.4f}")
        print(f"    Min:  {df[value_col].min():.4f}")
        print(f"    Max:  {df[value_col].max():.4f}")
        print(f"    Q1:   {df[value_col].quantile(0.25):.4f}")
        print(f"    Q3:   {df[value_col].quantile(0.75):.4f}")
        
        # Outliers
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[value_col] < Q1 - 1.5 * IQR) | (df[value_col] > Q3 + 1.5 * IQR)]
        print(f"    Outliers (IQR method): {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ BRONZE ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    analyze_bronze_layer()
