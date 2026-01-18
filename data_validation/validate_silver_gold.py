"""
Silver & Gold layer validation.

Silver: Standardized, merged, frequency-aligned data
Gold: Regime-encoded business-ready data
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple

from data_validation.schema import validate_dataframe_against_silver_schema, validate_dataframe_against_gold_schema

logger = logging.getLogger(__name__)


class SilverValidator:
    """Validates silver layer (cleaned, frequency-aligned data)."""
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Dict]:
        """
        Validate silver layer parquet file.
        
        Args:
            file_path: Path to silver parquet
            
        Returns:
            (is_valid, metrics_dict)
        """
        metrics = {
            "file": str(file_path),
            "valid": False,
            "errors": [],
            "warnings": [],
            "row_count": 0,
            "null_count": 0,
            "date_range": None,
            "duplicates": 0,
        }
        
        try:
            if not file_path.exists():
                metrics["errors"].append(f"File not found: {file_path}")
                return False, metrics
            
            # Load parquet
            df = pd.read_parquet(file_path)
            metrics["row_count"] = len(df)
            
            # Check required columns
            required = {"date", "UNRATE", "FEDFUNDS", "CPIAUCSL", "T10Y2Y", "STLFSI4", "CPI_YOY"}
            missing = required - set(df.columns)
            if missing:
                metrics["errors"].append(f"Missing columns: {missing}")
                return False, metrics
            
            # Validate against schema
            validate_dataframe_against_silver_schema(df)
            
            # Check for duplicates
            dup_count = df.duplicated(subset=["date"]).sum()
            metrics["duplicates"] = int(dup_count)
            if dup_count > 0:
                metrics["warnings"].append(f"Found {dup_count} duplicate dates")
            
            # Check date range
            metrics["date_range"] = {
                "start": df["date"].min().isoformat(),
                "end": df["date"].max().isoformat(),
            }
            
            # Check for gaps (should be monthly)
            date_diffs = df["date"].diff().dropna().dt.days
            expected_diff = 30  # roughly monthly
            outlier_gaps = date_diffs[(date_diffs < 25) | (date_diffs > 35)]
            if len(outlier_gaps) > 0:
                metrics["warnings"].append(
                    f"Found {len(outlier_gaps)} months with irregular spacing"
                )
            
            metrics["valid"] = len(metrics["errors"]) == 0
            
        except Exception as e:
            metrics["errors"].append(f"Validation exception: {str(e)}")
        
        return metrics["valid"], metrics


class GoldValidator:
    """Validates gold layer (regime-encoded data)."""
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Dict]:
        """
        Validate gold layer parquet file.
        
        Args:
            file_path: Path to gold parquet
            
        Returns:
            (is_valid, metrics_dict)
        """
        metrics = {
            "file": str(file_path),
            "valid": False,
            "errors": [],
            "warnings": [],
            "row_count": 0,
            "state_distribution": {},
            "regime_distribution": {},
            "date_range": None,
        }
        
        try:
            if not file_path.exists():
                metrics["errors"].append(f"File not found: {file_path}")
                return False, metrics
            
            # Load parquet
            df = pd.read_parquet(file_path)
            metrics["row_count"] = len(df)
            
            # Validate against schema
            validate_dataframe_against_gold_schema(df)
            
            # Check state distributions (should not be heavily imbalanced)
            state_cols = ["UNRATE_STATE", "FEDFUNDS_STATE", "CPI_YOY_STATE", "T10Y2Y_STATE", "STLFSI4_STATE"]
            for col in state_cols:
                dist = df[col].value_counts().to_dict()
                metrics["state_distribution"][col] = dist
            
            # Check regime distribution
            regime_dist = df["REGIME_RISK"].value_counts().to_dict()
            metrics["regime_distribution"] = regime_dist
            
            # Warn if HIGH_RISK is over-represented
            high_risk_pct = regime_dist.get("HIGH_RISK", 0) / len(df) * 100
            if high_risk_pct > 50:
                metrics["warnings"].append(
                    f"HIGH_RISK is {high_risk_pct:.1f}% of data (unusual if > 30%)"
                )
            
            # Check date range
            metrics["date_range"] = {
                "start": df["date"].min().isoformat(),
                "end": df["date"].max().isoformat(),
            }
            
            metrics["valid"] = len(metrics["errors"]) == 0
            
        except Exception as e:
            metrics["errors"].append(f"Validation exception: {str(e)}")
        
        return metrics["valid"], metrics
    
    def print_report(self, metrics: Dict) -> None:
        """Pretty-print gold validation report."""
        print(f"\nGold Layer Validation Report")
        print(f"File: {metrics['file']}")
        print(f"Valid: {'[PASS]' if metrics['valid'] else '[FAIL]'}")
        print(f"Rows: {metrics['row_count']}")
        print(f"Date Range: {metrics['date_range']}")
        
        print(f"\nRegime Distribution:")
        for regime, count in metrics['regime_distribution'].items():
            pct = count / metrics['row_count'] * 100
            print(f"  {regime}: {count} ({pct:.1f}%)")
        
        if metrics['warnings']:
            print(f"\nWarnings:")
            for warn in metrics['warnings']:
                print(f"  - {warn}")
        
        if metrics['errors']:
            print(f"\nErrors:")
            for err in metrics['errors']:
                print(f"  - {err}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: validate silver
    silver_path = Path("data/silver/cleaned_macro_data.parquet")
    sv = SilverValidator()
    is_valid, metrics = sv.validate_file(silver_path)
    print(f"Silver validation: {'✓' if is_valid else '✗'}")
    
    # Example: validate gold
    gold_path = Path("data/gold/markov_state_sequences.parquet")
    gv = GoldValidator()
    is_valid, metrics = gv.validate_file(gold_path)
    gv.print_report(metrics)
