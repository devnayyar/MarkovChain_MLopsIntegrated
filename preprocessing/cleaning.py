"""
Data cleaning pipeline: Bronze → Silver transformation.

Steps:
1. Load all raw FRED CSVs
2. Parse dates, standardize column names
3. Resample high-frequency data (daily/weekly) to monthly
4. Merge on date index
5. Forward-fill sparse values
6. Drop rows with too many missing values
7. Calculate derived metrics (CPI YoY)
8. Save to silver layer
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and standardizes FRED data from bronze to silver layer."""
    
    # Loading configuration
    SERIES_CONFIG = {
        "UNRATE": {"freq": "monthly", "agg": "first"},
        "FEDFUNDS": {"freq": "monthly", "agg": "first"},
        "CPIAUCSL": {"freq": "monthly", "agg": "first"},
        "T10Y2Y": {"freq": "daily", "agg": "mean"},  # Aggregate to monthly mean
        "STLFSI4": {"freq": "weekly", "agg": "mean"},  # Aggregate to monthly mean
    }
    
    def __init__(self, bronze_dir: Path):
        """Initialize with bronze directory."""
        self.bronze_dir = Path(bronze_dir)
    
    def load_and_parse_series(self, series_name: str) -> Tuple[pd.DataFrame, str]:
        """
        Load a single FRED CSV and parse dates.
        
        Args:
            series_name: Series name (UNRATE, FEDFUNDS, etc)
            
        Returns:
            (DataFrame with datetime index, frequency)
        """
        file_path = self.bronze_dir / f"{series_name}.csv"
        logger.info(f"Loading {series_name} from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Parse dates
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df = df.set_index("observation_date")
        df = df.sort_index()
        
        # Rename value column to series name
        value_col = [col for col in df.columns if col != "observation_date"][0]
        df = df.rename(columns={value_col: series_name})
        df = df[[series_name]]  # Keep only value column
        
        freq = self.SERIES_CONFIG[series_name]["freq"]
        logger.info(f"  Loaded {len(df)} records, frequency: {freq}")
        
        return df, freq
    
    def resample_to_monthly(self, df: pd.DataFrame, series_name: str) -> pd.DataFrame:
        """
        Resample series to monthly frequency.
        
        Args:
            df: DataFrame with datetime index
            series_name: Series name (for config lookup)
            
        Returns:
            Resampled DataFrame (monthly)
        """
        agg_method = self.SERIES_CONFIG[series_name]["agg"]
        
        # Resample to month-start frequency
        if agg_method == "mean":
            df_monthly = df.resample("MS").mean()
        elif agg_method == "first":
            df_monthly = df.resample("MS").first()
        else:
            raise ValueError(f"Unknown aggregation: {agg_method}")
        
        logger.info(f"  Resampled {series_name} to {len(df_monthly)} monthly records")
        
        return df_monthly
    
    def merge_all_series(self, series_list: list = None) -> pd.DataFrame:
        """
        Load, resample, and merge all series.
        
        Args:
            series_list: List of series to load (default: all)
            
        Returns:
            Merged DataFrame
        """
        if series_list is None:
            series_list = list(self.SERIES_CONFIG.keys())
        
        logger.info(f"Processing {len(series_list)} series")
        
        dfs = []
        for series_name in series_list:
            df, _ = self.load_and_parse_series(series_name)
            df_monthly = self.resample_to_monthly(df, series_name)
            dfs.append(df_monthly)
        
        # Merge on date index (outer join to preserve all dates)
        merged = pd.concat(dfs, axis=1, join="outer")
        merged = merged.sort_index()
        
        logger.info(f"Merged: {len(merged)} rows, {merged.isnull().sum().sum()} total nulls")
        
        return merged
    
    def handle_missing_values(self, df: pd.DataFrame, max_forward_fill: int = 1) -> Tuple[pd.DataFrame, Dict]:
        """
        Fill missing values using forward-fill strategy.
        
        Args:
            df: Input DataFrame
            max_forward_fill: Max months to forward-fill (default: 1)
            
        Returns:
            (Cleaned DataFrame, metrics dict)
        """
        metrics = {
            "rows_before": len(df),
            "nulls_before": df.isnull().sum().to_dict(),
            "rows_after": 0,
            "nulls_after": None,
        }
        
        logger.info(f"Before fill-forward: {df.isnull().sum().sum()} nulls")
        
        # Forward-fill (limited to max_forward_fill)
        df_filled = df.fillna(method="ffill", limit=max_forward_fill)
        
        logger.info(f"After fill-forward: {df_filled.isnull().sum().sum()} nulls")
        
        # Drop rows where >2 variables are missing
        null_counts_per_row = df_filled.isnull().sum(axis=1)
        rows_to_drop = null_counts_per_row[null_counts_per_row > 2].index
        df_filled = df_filled.drop(rows_to_drop)
        
        metrics["rows_after"] = len(df_filled)
        metrics["nulls_after"] = df_filled.isnull().sum().to_dict()
        metrics["rows_dropped"] = len(rows_to_drop)
        
        logger.info(f"After dropping rows with >2 nulls: {len(df_filled)} rows, {df_filled.isnull().sum().sum()} nulls")
        
        return df_filled, metrics
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics (CPI YoY).
        
        Args:
            df: DataFrame with CPIAUCSL column
            
        Returns:
            DataFrame with added CPI_YOY column
        """
        # CPI Year-over-Year: (CPI_t / CPI_t-12 - 1) * 100
        df["CPI_YOY"] = (df["CPIAUCSL"] / df["CPIAUCSL"].shift(12) - 1) * 100
        
        logger.info(f"Calculated CPI_YOY, first valid at index {df['CPI_YOY'].first_valid_index()}")
        
        return df
    
    def create_silver_table(self, output_path: Path = None) -> Tuple[pd.DataFrame, Path]:
        """
        Full pipeline: Bronze → Silver.
        
        Args:
            output_path: Where to save silver parquet (optional)
            
        Returns:
            (DataFrame, output_path)
        """
        # Load and merge
        merged = self.merge_all_series()
        
        # Handle missing values
        cleaned, metrics = self.handle_missing_values(merged)
        
        # Calculate derived metrics
        cleaned = self.calculate_derived_metrics(cleaned)
        
        # Reset index to make date a column
        cleaned = cleaned.reset_index()
        cleaned = cleaned.rename(columns={"observation_date": "date"})
        
        # Select and order columns
        columns_order = ["date", "UNRATE", "FEDFUNDS", "CPIAUCSL", "T10Y2Y", "STLFSI4", "CPI_YOY"]
        cleaned = cleaned[columns_order]
        
        logger.info(f"Silver table created: {len(cleaned)} rows, {len(cleaned.columns)} columns")
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned.to_parquet(output_path, index=False)
            logger.info(f"Saved silver table to {output_path}")
        
        return cleaned, output_path
    
    def print_cleaning_report(self, df: pd.DataFrame) -> None:
        """Print summary of cleaning process."""
        print("\n" + "="*80)
        print("SILVER LAYER CLEANING REPORT")
        print("="*80)
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nNull values:")
        print(df.isnull().sum())
        print(f"\nData Summary:")
        print(df.describe())
        print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    bronze_dir = Path("data/bronze")
    output_path = Path("data/silver/cleaned_macro_data.parquet")
    
    cleaner = DataCleaner(bronze_dir)
    silver_df, saved_path = cleaner.create_silver_table(output_path)
    cleaner.print_cleaning_report(silver_df)
