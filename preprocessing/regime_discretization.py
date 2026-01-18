"""
Regime Discretization: Convert continuous variables into discrete economic regimes.

Applies domain-driven thresholds to create:
1. Individual state variables (UNRATE_STATE, FEDFUNDS_STATE, etc)
2. Composite REGIME_RISK classification
"""

import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class RegimeDiscretizer:
    """Converts continuous economic variables into discrete regimes."""
    
    # Hard-coded thresholds (can also load from YAML)
    THRESHOLDS = {
        "UNRATE": {
            "LOW": (0.0, 4.0),
            "MEDIUM": (4.0, 6.0),
            "HIGH": (6.0, 15.0),
        },
        "FEDFUNDS": {
            "ACCOMMODATIVE": (0.0, 2.0),
            "NEUTRAL": (2.0, 4.0),
            "TIGHT": (4.0, 20.0),
        },
        "CPI_YOY": {
            "LOW": (-5.0, 2.0),
            "MODERATE": (2.0, 4.0),
            "HIGH": (4.0, 20.0),
        },
        "T10Y2Y": {
            "NORMAL": (0.50, 5.0),
            "FLAT": (0.0, 0.50),
            "INVERTED": (-2.0, 0.0),
        },
        "STLFSI4": {
            "CALM": (-3.0, -0.5),
            "NEUTRAL": (-0.5, 0.5),
            "STRESS": (0.5, 10.0),  # Extended upper bound to handle extreme stress
        },
    }
    
    # Composite regime logic (risk-weighted)
    HIGH_RISK_TRIGGERS = [
        "UNRATE_STATE == 'HIGH'",
        "T10Y2Y_STATE == 'INVERTED'",
        "STLFSI4_STATE == 'STRESS'",
    ]
    
    LOW_RISK_TRIGGERS = [
        "UNRATE_STATE == 'LOW'",
        "STLFSI4_STATE == 'CALM'",
        "CPI_YOY_STATE == 'MODERATE'",
    ]
    
    def discretize_variable(self, series: pd.Series, var_name: str) -> pd.Series:
        """
        Convert continuous variable to discrete state.
        
        Args:
            series: Series with continuous values
            var_name: Variable name (UNRATE, FEDFUNDS, etc)
            
        Returns:
            Series with categorical states
        """
        if var_name not in self.THRESHOLDS:
            raise ValueError(f"Unknown variable: {var_name}")
        
        thresholds = self.THRESHOLDS[var_name]
        states = pd.Series(index=series.index, dtype="string")
        
        # Get only non-null values for categorization
        non_null_series = series[series.notna()]
        
        for state, (lower, upper) in thresholds.items():
            # Check each threshold range
            if state == "INVERTED":
                # Special handling for inverted yield curve (lower bound is negative)
                mask = (non_null_series >= lower) & (non_null_series < upper)
            elif state == "FLAT":
                # Flat is between 0 and 0.5
                mask = (non_null_series >= lower) & (non_null_series < upper)
            else:
                mask = (non_null_series >= lower) & (non_null_series < upper)
            
            states.loc[mask.index[mask]] = state
        
        # Check for unmapped values (shouldn't happen with correct thresholds)
        unmapped = non_null_series[states[non_null_series.index].isna()]
        if len(unmapped) > 0:
            logger.warning(
                f"{var_name}: {len(unmapped)} values unmapped. "
                f"Min={unmapped.min()}, Max={unmapped.max()}"
            )
        
        # Log the distribution
        state_counts = states.dropna().value_counts()
        if len(state_counts) > 0:
            logger.info(f"{var_name} discretized: {state_counts.to_dict()}")
        
        return states
    
    def discretize_all_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create state columns for all variables.
        
        Args:
            df: DataFrame with continuous variables
            
        Returns:
            DataFrame with added state columns
        """
        df = df.copy()
        
        logger.info("Discretizing all variables to states...")
        
        # Discretize each variable
        df["UNRATE_STATE"] = self.discretize_variable(df["UNRATE"], "UNRATE")
        df["FEDFUNDS_STATE"] = self.discretize_variable(df["FEDFUNDS"], "FEDFUNDS")
        df["CPI_YOY_STATE"] = self.discretize_variable(df["CPI_YOY"], "CPI_YOY")
        df["T10Y2Y_STATE"] = self.discretize_variable(df["T10Y2Y"], "T10Y2Y")
        df["STLFSI4_STATE"] = self.discretize_variable(df["STLFSI4"], "STLFSI4")
        
        return df
    
    def assign_composite_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign composite REGIME_RISK based on component states.
        
        Risk-weighted logic:
        - HIGH_RISK if any trigger activates
        - LOW_RISK if all components benign
        - MODERATE_RISK otherwise
        
        Args:
            df: DataFrame with discretized state columns
            
        Returns:
            DataFrame with added REGIME_RISK column
        """
        df = df.copy()
        regime_risk = []
        
        for idx, row in df.iterrows():
            # Fill NaN with default state for comparison
            unrate_state = row["UNRATE_STATE"] if pd.notna(row["UNRATE_STATE"]) else "MEDIUM"
            yield_state = row["T10Y2Y_STATE"] if pd.notna(row["T10Y2Y_STATE"]) else "NORMAL"
            stress_state = row["STLFSI4_STATE"] if pd.notna(row["STLFSI4_STATE"]) else "NEUTRAL"
            cpi_state = row["CPI_YOY_STATE"] if pd.notna(row["CPI_YOY_STATE"]) else "MODERATE"
            
            # Check HIGH_RISK triggers
            high_risk = (
                (unrate_state == "HIGH") or
                (yield_state == "INVERTED") or
                (stress_state == "STRESS")
            )
            
            # Check LOW_RISK triggers (all must be true)
            low_risk = (
                (unrate_state == "LOW") and
                (stress_state == "CALM") and
                (cpi_state == "MODERATE")
            )
            
            if high_risk:
                regime_risk.append("HIGH_RISK")
            elif low_risk:
                regime_risk.append("LOW_RISK")
            else:
                regime_risk.append("MODERATE_RISK")
        
        df["REGIME_RISK"] = regime_risk
        
        logger.info(f"Composite regime assigned: {pd.Series(regime_risk).value_counts().to_dict()}")
        
        return df
    
    def discretize_full_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full discretization pipeline: variables → states → composite regime.
        
        Args:
            df: Silver layer DataFrame
            
        Returns:
            Gold layer DataFrame with all state columns
        """
        logger.info(f"Starting regime discretization on {len(df)} rows")
        
        # Discretize individual variables
        df = self.discretize_all_variables(df)
        
        # Assign composite regime
        df = self.assign_composite_regime(df)
        
        logger.info(f"Discretization complete: {len(df)} rows")
        
        return df
    
    def print_regime_summary(self, df: pd.DataFrame) -> None:
        """Print summary of regimes."""
        print("\n" + "="*80)
        print("REGIME DISCRETIZATION SUMMARY")
        print("="*80)
        
        state_cols = ["UNRATE_STATE", "FEDFUNDS_STATE", "CPI_YOY_STATE", "T10Y2Y_STATE", "STLFSI4_STATE"]
        
        for col in state_cols:
            print(f"\n{col}:")
            print(df[col].value_counts().sort_index())
        
        print(f"\nCOMPOSITE REGIME_RISK:")
        regime_counts = df["REGIME_RISK"].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"  {regime}: {count} ({pct:.1f}%)")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    silver_path = Path("data/silver/cleaned_macro_data.parquet")
    output_path = Path("data/gold/markov_state_sequences.parquet")
    
    if silver_path.exists():
        df_silver = pd.read_parquet(silver_path)
        
        discretizer = RegimeDiscretizer()
        df_gold = discretizer.discretize_full_pipeline(df_silver)
        discretizer.print_regime_summary(df_gold)
        
        # Save gold layer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_gold.to_parquet(output_path, index=False)
        print(f"Saved gold layer to {output_path}")
    else:
        print(f"Silver layer not found at {silver_path}")
