"""
Exploratory Data Analysis (EDA) for Financial Risk Markov System.

Analyzes bronze/silver/gold layers:
- Data distributions
- Regime transitions
- Temporal patterns
- Risk evolution over time
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BronzeEDA:
    """Analyze raw FRED data layer."""
    
    def __init__(self, bronze_dir: str = "data/bronze"):
        """Initialize bronze layer analysis."""
        self.bronze_dir = Path(bronze_dir)
        self.data = {}
        
    def load_all_series(self):
        """Load all FRED CSV files."""
        csv_files = list(self.bronze_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                self.data[csv_file.stem] = df
                logger.info(f"  Loaded {csv_file.stem}: {df.shape}")
            except Exception as e:
                logger.error(f"Error loading {csv_file.stem}: {e}")
        
        return self.data
    
    def summary_stats(self):
        """Print summary statistics for each series."""
        print("\n" + "="*80)
        print("BRONZE LAYER: FRED SERIES SUMMARY")
        print("="*80)
        
        for series_name, df in self.data.items():
            col = df.columns[0]
            series = df[col].dropna()
            
            print(f"\n{series_name}:")
            print(f"  Count:    {len(series)}")
            print(f"  Mean:     {series.mean():.4f}")
            print(f"  Std:      {series.std():.4f}")
            print(f"  Min:      {series.min():.4f}")
            print(f"  Max:      {series.max():.4f}")
            print(f"  Date Range: {series.index.min().date()} to {series.index.max().date()}")


class SilverEDA:
    """Analyze cleaned and merged data layer."""
    
    def __init__(self, silver_path: str = "data/silver/cleaned_macro_data.parquet"):
        """Initialize silver layer analysis."""
        self.silver_path = Path(silver_path)
        self.df = None
    
    def load_data(self):
        """Load silver layer parquet file."""
        if not self.silver_path.exists():
            logger.error(f"Silver data not found at {self.silver_path}")
            return None
        
        self.df = pd.read_parquet(self.silver_path)
        logger.info(f"Loaded silver data: {self.df.shape}")
        
        return self.df
    
    def summary_stats(self):
        """Print summary statistics."""
        if self.df is None:
            logger.warning("No data loaded")
            return
        
        print("\n" + "="*80)
        print("SILVER LAYER: CLEANED MACRO DATA SUMMARY")
        print("="*80)
        print(f"\nShape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        # Handle index type
        if hasattr(self.df.index, 'min'):
            min_idx = self.df.index.min()
            max_idx = self.df.index.max()
            if hasattr(min_idx, 'date'):
                print(f"Date Range: {min_idx.date()} to {max_idx.date()}")
            else:
                print(f"Date Range: {min_idx} to {max_idx}")
        print(f"Frequency: Monthly (implied)")
        
        print("\nColumn Summary:")
        print(self.df.describe().round(4))
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values!")


class GoldEDA:
    """Analyze regime state sequences and risk levels."""
    
    def __init__(self, gold_path: str = "data/gold/markov_state_sequences.parquet"):
        """Initialize gold layer analysis."""
        self.gold_path = Path(gold_path)
        self.df = None
    
    def load_data(self):
        """Load gold layer parquet file."""
        if not self.gold_path.exists():
            logger.error(f"Gold data not found at {self.gold_path}")
            return None
        
        self.df = pd.read_parquet(self.gold_path)
        logger.info(f"Loaded gold data: {self.df.shape}")
        
        return self.df
    
    def regime_distribution(self):
        """Analyze regime risk level distribution."""
        if self.df is None:
            logger.warning("No data loaded")
            return
        
        print("\n" + "="*80)
        print("GOLD LAYER: REGIME DISTRIBUTION")
        print("="*80)
        
        # Overall regime distribution
        regime_counts = self.df['REGIME_RISK'].value_counts()
        total = len(self.df)
        
        print("\nREGIME_RISK Distribution:")
        for regime, count in regime_counts.items():
            pct = 100.0 * count / total
            print(f"  {regime:<20}: {count:4d} ({pct:5.1f}%)")
        
        # Continuous variable state distributions
        state_cols = [col for col in self.df.columns if '_STATE' in col]
        
        print("\n\nState Distributions by Variable:")
        for col in state_cols:
            print(f"\n{col}:")
            counts = self.df[col].value_counts()
            for state, count in counts.items():
                pct = 100.0 * count / total
                print(f"  {state:<20}: {count:4d} ({pct:5.1f}%)")
    
    def transition_patterns(self):
        """Analyze regime transitions."""
        if self.df is None:
            logger.warning("No data loaded")
            return
        
        print("\n" + "="*80)
        print("TRANSITION PATTERNS")
        print("="*80)
        
        regimes = self.df['REGIME_RISK'].values
        transitions = []
        
        for t in range(len(regimes) - 1):
            from_regime = regimes[t]
            to_regime = regimes[t + 1]
            if pd.notna(from_regime) and pd.notna(to_regime):
                transitions.append(f"{from_regime} -> {to_regime}")
        
        # Count transitions
        from collections import Counter
        trans_counts = Counter(transitions)
        
        print("\nMost Common Transitions:")
        for trans, count in trans_counts.most_common(10):
            print(f"  {trans:<40}: {count:4d}")
    
    def temporal_analysis(self):
        """Analyze regime evolution over time."""
        if self.df is None:
            logger.warning("No data loaded")
            return
        
        print("\n" + "="*80)
        print("TEMPORAL ANALYSIS")
        print("="*80)
        
        # Check if we have a proper date column
        if 'date' in self.df.columns:
            self.df['year'] = pd.to_datetime(self.df['date']).dt.year
        else:
            logger.warning("No date column found for temporal analysis")
            return
        
        # Count regimes per year
        yearly_regimes = self.df.groupby(['year', 'REGIME_RISK']).size().unstack(fill_value=0)
        
        print("\nRegimes per Year (12-month periods):")
        print(yearly_regimes.tail(15))  # Last 15 years
        
        # Count crisis months (HIGH_RISK)
        crisis_months = self.df[self.df['REGIME_RISK'] == 'HIGH_RISK'].groupby('year').size()
        
        print("\n\nCrisis Months per Year (HIGH_RISK):")
        for year, count in crisis_months.items():
            pct = 100.0 * count / 12
            print(f"  {year}: {count:2d} months ({pct:5.1f}%)")


def main():
    """Run EDA analysis."""
    print("\n" + "="*80)
    print("PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # Bronze layer
    print("\n[1/3] Analyzing Bronze Layer...")
    bronze = BronzeEDA()
    bronze.load_all_series()
    bronze.summary_stats()
    
    # Silver layer
    print("\n[2/3] Analyzing Silver Layer...")
    silver = SilverEDA()
    silver.load_data()
    silver.summary_stats()
    
    # Gold layer
    print("\n[3/3] Analyzing Gold Layer...")
    gold = GoldEDA()
    gold.load_data()
    gold.regime_distribution()
    gold.transition_patterns()
    gold.temporal_analysis()
    
    print("\n" + "="*80)
    print("[SUCCESS] EDA COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
