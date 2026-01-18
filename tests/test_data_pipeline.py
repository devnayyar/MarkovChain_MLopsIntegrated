"""
Quick test script to validate the data pipeline end-to-end.

Steps:
1. Validate bronze layer
2. Create silver layer
3. Create gold layer
4. Validate all outputs
"""

import sys
from pathlib import Path
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_validation.validate_bronze import BronzeValidator
from preprocessing.cleaning import DataCleaner
from preprocessing.regime_discretization import RegimeDiscretizer
from data_validation.validate_silver_gold import SilverValidator, GoldValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pipeline():
    """Run complete test pipeline."""
    
    print("\n" + "="*80)
    print("FINANCIAL RISK MARKOV MLOPS - DATA PIPELINE TEST")
    print("="*80)
    
    # Paths
    bronze_dir = project_root / "data" / "bronze"
    silver_path = project_root / "data" / "silver" / "cleaned_macro_data.parquet"
    gold_path = project_root / "data" / "gold" / "markov_state_sequences.parquet"
    
    # ─────────────────────────────────────────────────────────────
    # STEP 1: VALIDATE BRONZE LAYER
    # ─────────────────────────────────────────────────────────────
    print("\n[1/4] VALIDATING BRONZE LAYER...")
    print("-" * 80)
    
    bv = BronzeValidator()
    bronze_results = bv.validate_all_files(bronze_dir)
    bv.print_validation_report(bronze_results)
    
    all_bronze_valid = all(is_valid for is_valid, _ in bronze_results.values())
    if not all_bronze_valid:
        logger.error("Bronze validation failed!")
        return False
    
    logger.info("✓ Bronze layer valid")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 2: CREATE SILVER LAYER
    # ─────────────────────────────────────────────────────────────
    print("\n[2/4] CREATING SILVER LAYER...")
    print("-" * 80)
    
    cleaner = DataCleaner(bronze_dir)
    silver_df, saved_silver = cleaner.create_silver_table(silver_path)
    cleaner.print_cleaning_report(silver_df)
    
    logger.info(f"✓ Silver layer created: {len(silver_df)} rows")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 3: CREATE GOLD LAYER
    # ─────────────────────────────────────────────────────────────
    print("\n[3/4] CREATING GOLD LAYER (REGIME DISCRETIZATION)...")
    print("-" * 80)
    
    discretizer = RegimeDiscretizer()
    gold_df = discretizer.discretize_full_pipeline(silver_df)
    discretizer.print_regime_summary(gold_df)
    
    # Save gold layer
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_df.to_parquet(gold_path, index=False)
    logger.info(f"✓ Gold layer created: {len(gold_df)} rows")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 4: VALIDATE OUTPUTS
    # ─────────────────────────────────────────────────────────────
    print("\n[4/4] VALIDATING SILVER & GOLD OUTPUTS...")
    print("-" * 80)
    
    # Validate silver
    sv = SilverValidator()
    silver_valid, silver_metrics = sv.validate_file(silver_path)
    logger.info(f"Silver validation: {'[PASS]' if silver_valid else '[FAIL]'}")
    if silver_metrics["errors"]:
        for err in silver_metrics["errors"]:
            logger.error(f"  {err}")
    
    # Validate gold
    gv = GoldValidator()
    gold_valid, gold_metrics = gv.validate_file(gold_path)
    logger.info(f"Gold validation: {'[PASS]' if gold_valid else '[FAIL]'}")
    if gold_metrics["errors"]:
        for err in gold_metrics["errors"]:
            logger.error(f"  {err}")
    
    gv.print_report(gold_metrics)
    
    # ─────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PIPELINE TEST SUMMARY")
    print("="*80)
    
    all_valid = all_bronze_valid and silver_valid and gold_valid
    
    print(f"\nBronze Validation: {'[PASS]' if all_bronze_valid else '[FAIL]'}")
    print(f"Silver Creation:   [OK] {len(silver_df)} rows")
    print(f"Silver Validation: {'[PASS]' if silver_valid else '[FAIL]'}")
    print(f"Gold Creation:     [OK] {len(gold_df)} rows")
    print(f"Gold Validation:   {'[PASS]' if gold_valid else '[FAIL]'}")
    
    print(f"\nRegime Distribution:")
    regime_counts = gold_df["REGIME_RISK"].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(gold_df) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")
    
    print(f"\nDate Range: {gold_df['date'].min()} to {gold_df['date'].max()}")
    print(f"Total Records: {len(gold_df)}")
    
    print("\n" + "="*80)
    if all_valid:
        print("[SUCCESS] ALL TESTS PASSED - DATA PIPELINE IS READY")
    else:
        print("[FAILURE] SOME TESTS FAILED - SEE ERRORS ABOVE")
    print("="*80 + "\n")
    
    return all_valid


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
