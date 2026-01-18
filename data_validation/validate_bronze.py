"""
Bronze layer validation.

Validates raw CSV files downloaded from FRED:
- Check datetime format
- Check no required fields are fully missing
- Check value ranges are reasonable
- Log data quality metrics
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class BronzeValidator:
    """Validates raw FRED CSV files."""
    
    # Expected FRED series and their basic properties
    SERIES_CONFIG = {
        "UNRATE": {
            "column_name": "UNRATE",
            "expected_freq": "monthly",
            "min_value": 0.0,
            "max_value": 15.0,
            "allow_null": False,
        },
        "FEDFUNDS": {
            "column_name": "FEDFUNDS",
            "expected_freq": "monthly",
            "min_value": 0.0,
            "max_value": 20.0,
            "allow_null": False,
        },
        "CPIAUCSL": {
            "column_name": "CPIAUCSL",
            "expected_freq": "monthly",
            "min_value": 20.0,
            "max_value": 400.0,
            "allow_null": False,
        },
        "T10Y2Y": {
            "column_name": "T10Y2Y",
            "expected_freq": "daily",
            "min_value": -2.0,
            "max_value": 3.0,
            "allow_null": True,  # Daily data sparse
        },
        "STLFSI4": {
            "column_name": "STLFSI4",
            "expected_freq": "weekly",
            "min_value": -2.0,
            "max_value": 5.0,
            "allow_null": True,  # Weekly data sparse
        },
    }
    
    def validate_file(self, file_path: Path, series_name: str) -> Tuple[bool, Dict]:
        """
        Validate a raw FRED CSV file.
        
        Args:
            file_path: Path to CSV file
            series_name: Series name (UNRATE, FEDFUNDS, etc)
            
        Returns:
            (is_valid, metrics_dict)
        """
        metrics = {
            "file": str(file_path),
            "series": series_name,
            "valid": False,
            "errors": [],
            "warnings": [],
            "row_count": 0,
            "null_count": 0,
            "null_pct": 0.0,
            "date_range": None,
            "frequency_detected": None,
            "outliers_detected": 0,
        }
        
        try:
            # Check file exists
            if not file_path.exists():
                metrics["errors"].append(f"File not found: {file_path}")
                return False, metrics
            
            # Load CSV
            df = pd.read_csv(file_path)
            metrics["row_count"] = len(df)
            
            # Check columns
            if "observation_date" not in df.columns:
                metrics["errors"].append("Missing 'observation_date' column")
                return False, metrics
            
            config = self.SERIES_CONFIG.get(series_name)
            if not config:
                metrics["errors"].append(f"Unknown series: {series_name}")
                return False, metrics
            
            value_column = config["column_name"]
            if value_column not in df.columns:
                metrics["errors"].append(f"Missing '{value_column}' column")
                return False, metrics
            
            # Parse dates
            try:
                df["observation_date"] = pd.to_datetime(df["observation_date"])
            except Exception as e:
                metrics["errors"].append(f"Date parsing error: {str(e)}")
                return False, metrics
            
            # Check date range
            metrics["date_range"] = {
                "start": df["observation_date"].min().isoformat(),
                "end": df["observation_date"].max().isoformat(),
            }
            
            # Check value column
            value_col_data = df[value_column]
            null_count = value_col_data.isnull().sum()
            metrics["null_count"] = int(null_count)
            metrics["null_pct"] = (null_count / len(df)) * 100
            
            if null_count > 0 and not config["allow_null"]:
                metrics["warnings"].append(
                    f"Series {series_name} has {null_count} nulls ({metrics['null_pct']:.1f}%)"
                )
            
            # Check for all-null column
            if value_col_data.isnull().all():
                metrics["errors"].append(f"Column '{value_column}' is entirely null")
                return False, metrics
            
            # Check value range (excluding nulls)
            non_null_values = value_col_data.dropna()
            min_val = non_null_values.min()
            max_val = non_null_values.max()
            
            if min_val < config["min_value"] or max_val > config["max_value"]:
                metrics["warnings"].append(
                    f"Values out of expected range [{config['min_value']}, {config['max_value']}]. "
                    f"Found [{min_val}, {max_val}]"
                )
            
            # Detect outliers (simple 3-sigma)
            mean_val = non_null_values.mean()
            std_val = non_null_values.std()
            outlier_threshold = 3 * std_val
            outliers = non_null_values[
                (non_null_values < mean_val - outlier_threshold) |
                (non_null_values > mean_val + outlier_threshold)
            ]
            metrics["outliers_detected"] = len(outliers)
            
            if len(outliers) > 0:
                metrics["warnings"].append(f"Detected {len(outliers)} potential outliers (3-sigma)")
            
            # Detect frequency
            if len(df) > 1:
                date_diffs = df["observation_date"].diff().dropna().dt.days
                mean_diff = date_diffs.mean()
                
                if 25 <= mean_diff <= 35:
                    metrics["frequency_detected"] = "monthly"
                elif 5 <= mean_diff <= 10:
                    metrics["frequency_detected"] = "weekly"
                elif 0 <= mean_diff <= 2:
                    metrics["frequency_detected"] = "daily"
                else:
                    metrics["frequency_detected"] = "irregular"
            
            # If we got here with no errors, it's valid
            metrics["valid"] = len(metrics["errors"]) == 0
            
        except Exception as e:
            metrics["errors"].append(f"Validation exception: {str(e)}")
        
        return metrics["valid"], metrics
    
    def validate_all_files(self, bronze_dir: Path) -> Dict[str, Tuple[bool, Dict]]:
        """
        Validate all FRED CSV files in bronze directory.
        
        Args:
            bronze_dir: Path to data/bronze/ directory
            
        Returns:
            Dict mapping series name to (is_valid, metrics)
        """
        results = {}
        
        for series_name in self.SERIES_CONFIG.keys():
            file_path = bronze_dir / f"{series_name}.csv"
            is_valid, metrics = self.validate_file(file_path, series_name)
            results[series_name] = (is_valid, metrics)
            
            # Log results
            status = "[PASS]" if is_valid else "[FAIL]"
            logger.info(f"{status} {series_name}: {len(metrics['errors'])} errors, {len(metrics['warnings'])} warnings")
            
            for error in metrics["errors"]:
                logger.error(f"  ERROR: {error}")
            
            for warning in metrics["warnings"]:
                logger.warning(f"  WARNING: {warning}")
        
        return results
    
    def print_validation_report(self, results: Dict[str, Tuple[bool, Dict]]) -> None:
        """Pretty-print validation report."""
        print("\n" + "="*80)
        print("BRONZE LAYER VALIDATION REPORT")
        print("="*80)
        
        all_valid = all(is_valid for is_valid, _ in results.values())
        
        for series_name, (is_valid, metrics) in results.items():
            status = "[PASS]" if is_valid else "[FAIL]"
            print(f"\n{status} {series_name}")
            print(f"  Rows: {metrics['row_count']}")
            print(f"  Nulls: {metrics['null_count']} ({metrics['null_pct']:.1f}%)")
            print(f"  Date Range: {metrics['date_range']['start']} to {metrics['date_range']['end']}")
            print(f"  Frequency Detected: {metrics['frequency_detected']}")
            print(f"  Outliers: {metrics['outliers_detected']}")
            
            if metrics["errors"]:
                print(f"  Errors:")
                for err in metrics["errors"]:
                    print(f"    - {err}")
            
            if metrics["warnings"]:
                print(f"  Warnings:")
                for warn in metrics["warnings"]:
                    print(f"    - {warn}")
        
        print("\n" + "="*80)
        print(f"Overall: {'[PASS] ALL VALID' if all_valid else '[FAIL] SOME INVALID'}")
        print("="*80 + "\n")
        
        return all_valid


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    bronze_dir = Path("data/bronze")
    validator = BronzeValidator()
    results = validator.validate_all_files(bronze_dir)
    validator.print_validation_report(results)
