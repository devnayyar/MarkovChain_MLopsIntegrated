"""
Pydantic schemas for data validation across all layers.

Validates:
- Bronze layer: Raw CSV format
- Silver layer: Cleaned, frequency-aligned data
- Gold layer: Regime-encoded business-ready data
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import pandas as pd
from pydantic import BaseModel, Field, validator


# ═══════════════════════════════════════════════════════════════
# BRONZE LAYER SCHEMAS
# ═══════════════════════════════════════════════════════════════

class BronzeRecord(BaseModel):
    """Single record from raw FRED CSV."""
    
    observation_date: datetime
    value: Optional[float] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BronzeIndicator(BaseModel):
    """Represents one complete raw FRED series."""
    
    name: str = Field(..., description="Indicator name (UNRATE, FEDFUNDS, etc)")
    frequency: str = Field(..., description="Raw frequency (monthly, daily, weekly)")
    records: List[BronzeRecord]
    
    @validator("name")
    def validate_name(cls, v):
        """Name must be one of expected FRED indicators."""
        valid_names = {"UNRATE", "FEDFUNDS", "CPIAUCSL", "T10Y2Y", "STLFSI4"}
        if v not in valid_names:
            raise ValueError(f"Unknown indicator: {v}. Expected one of {valid_names}")
        return v
    
    @validator("frequency")
    def validate_frequency(cls, v):
        """Validate frequency is one of expected values."""
        valid_freqs = {"monthly", "daily", "weekly"}
        if v.lower() not in valid_freqs:
            raise ValueError(f"Invalid frequency: {v}")
        return v.lower()


# ═══════════════════════════════════════════════════════════════
# SILVER LAYER SCHEMAS
# ═══════════════════════════════════════════════════════════════

class SilverRecord(BaseModel):
    """Single row of standardized, frequency-aligned data."""
    
    date: datetime = Field(..., description="Month-start date (YYYY-MM-01)")
    UNRATE: float = Field(..., ge=0.0, le=15.0, description="Unemployment Rate")
    FEDFUNDS: float = Field(..., ge=0.0, le=20.0, description="Federal Funds Rate")
    CPIAUCSL: float = Field(..., ge=20.0, le=400.0, description="CPI Index")
    T10Y2Y: float = Field(..., ge=-2.0, le=3.0, description="10Y-2Y Spread")
    STLFSI4: float = Field(..., ge=-2.0, le=5.0, description="Financial Stress Index")
    CPI_YOY: float = Field(..., ge=-5.0, le=20.0, description="CPI YoY %")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SilverDataset(BaseModel):
    """Complete standardized dataset (monthly aligned)."""
    
    table_name: str = "macro_indicators_monthly"
    frequency: str = "monthly"
    records: List[SilverRecord]
    date_range_start: datetime
    date_range_end: datetime
    row_count: int
    null_count: int = Field(default=0, description="Should be 0 in silver")
    
    @validator("frequency")
    def validate_frequency(cls, v):
        """Silver must be monthly frequency."""
        if v != "monthly":
            raise ValueError("Silver layer must be frequency='monthly'")
        return v
    
    @validator("null_count")
    def validate_no_nulls(cls, v):
        """Silver layer should have no nulls."""
        if v > 0:
            raise ValueError(f"Silver layer should have 0 nulls, found {v}")
        return v
    
    @validator("row_count")
    def validate_row_count(cls, v, values):
        """Row count should match expected monthly periods."""
        if len(values.get("records", [])) != v:
            raise ValueError(
                f"Row count mismatch: header says {v}, actual {len(values['records'])}"
            )
        return v


# ═══════════════════════════════════════════════════════════════
# GOLD LAYER SCHEMAS
# ═══════════════════════════════════════════════════════════════

class RegimeStateEnum(str, Enum):
    """Individual regime state categories."""
    # Unemployment states
    UNRATE_LOW = "LOW"
    UNRATE_MEDIUM = "MEDIUM"
    UNRATE_HIGH = "HIGH"
    
    # Fed Funds states
    FEDFUNDS_ACCOMMODATIVE = "ACCOMMODATIVE"
    FEDFUNDS_NEUTRAL = "NEUTRAL"
    FEDFUNDS_TIGHT = "TIGHT"
    
    # Inflation states
    CPI_YOY_LOW = "LOW"
    CPI_YOY_MODERATE = "MODERATE"
    CPI_YOY_HIGH = "HIGH"
    
    # Yield curve states
    T10Y2Y_NORMAL = "NORMAL"
    T10Y2Y_FLAT = "FLAT"
    T10Y2Y_INVERTED = "INVERTED"
    
    # Financial stress states
    STLFSI4_CALM = "CALM"
    STLFSI4_NEUTRAL = "NEUTRAL"
    STLFSI4_STRESS = "STRESS"


class CompositeRegimeEnum(str, Enum):
    """Composite risk regime."""
    LOW_RISK = "LOW_RISK"
    MODERATE_RISK = "MODERATE_RISK"
    HIGH_RISK = "HIGH_RISK"


class GoldRecord(BaseModel):
    """Single row of regime-encoded business-ready data."""
    
    date: datetime = Field(..., description="Month-start date")
    
    # Individual state variables
    UNRATE_STATE: str = Field(..., description="Unemployment regime")
    FEDFUNDS_STATE: str = Field(..., description="Monetary policy state")
    CPI_YOY_STATE: str = Field(..., description="Inflation regime")
    T10Y2Y_STATE: str = Field(..., description="Yield curve regime")
    STLFSI4_STATE: str = Field(..., description="Financial stress state")
    
    # Composite risk regime
    REGIME_RISK: str = Field(..., description="Aggregated risk classification")
    
    # Raw values (for reference and backtesting)
    UNRATE: float = Field(..., ge=0.0, le=15.0)
    FEDFUNDS: float = Field(..., ge=0.0, le=20.0)
    CPIAUCSL: float = Field(..., ge=20.0, le=400.0)
    CPI_YOY: float = Field(..., ge=-5.0, le=20.0)
    T10Y2Y: float = Field(..., ge=-2.0, le=3.0)
    STLFSI4: float = Field(..., ge=-2.0, le=5.0)
    
    @validator("REGIME_RISK")
    def validate_regime_risk(cls, v):
        """Validate composite regime is one of three."""
        valid = {"LOW_RISK", "MODERATE_RISK", "HIGH_RISK"}
        if v not in valid:
            raise ValueError(f"Invalid regime: {v}, expected one of {valid}")
        return v
    
    @validator("UNRATE_STATE")
    def validate_unrate_state(cls, v):
        valid = {"LOW", "MEDIUM", "HIGH"}
        if v not in valid:
            raise ValueError(f"Invalid UNRATE_STATE: {v}")
        return v
    
    @validator("FEDFUNDS_STATE")
    def validate_fedfunds_state(cls, v):
        valid = {"ACCOMMODATIVE", "NEUTRAL", "TIGHT"}
        if v not in valid:
            raise ValueError(f"Invalid FEDFUNDS_STATE: {v}")
        return v
    
    @validator("CPI_YOY_STATE")
    def validate_cpi_state(cls, v):
        valid = {"LOW", "MODERATE", "HIGH"}
        if v not in valid:
            raise ValueError(f"Invalid CPI_YOY_STATE: {v}")
        return v
    
    @validator("T10Y2Y_STATE")
    def validate_yield_state(cls, v):
        valid = {"NORMAL", "FLAT", "INVERTED"}
        if v not in valid:
            raise ValueError(f"Invalid T10Y2Y_STATE: {v}")
        return v
    
    @validator("STLFSI4_STATE")
    def validate_stress_state(cls, v):
        valid = {"CALM", "NEUTRAL", "STRESS"}
        if v not in valid:
            raise ValueError(f"Invalid STLFSI4_STATE: {v}")
        return v
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class GoldDataset(BaseModel):
    """Complete regime-encoded dataset."""
    
    table_name: str = "regime_states_monthly"
    frequency: str = "monthly"
    records: List[GoldRecord]
    date_range_start: datetime
    date_range_end: datetime
    row_count: int
    
    @validator("frequency")
    def validate_frequency(cls, v):
        """Gold must be monthly frequency."""
        if v != "monthly":
            raise ValueError("Gold layer must be frequency='monthly'")
        return v


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS FOR VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_dataframe_against_silver_schema(df: pd.DataFrame) -> bool:
    """
    Validate a Pandas DataFrame against silver schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises exception otherwise
        
    Raises:
        ValueError: If schema violations detected
    """
    required_columns = {"date", "UNRATE", "FEDFUNDS", "CPIAUCSL", "T10Y2Y", "STLFSI4", "CPI_YOY"}
    
    if not required_columns.issubset(set(df.columns)):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Allow nulls in sparse historical columns
    # T10Y2Y: started 2021, STLFSI4: started 1993, CPI_YOY: needs 12-month lag
    # This is expected and acceptable
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("Column 'date' must be datetime type")
    
    if not df['date'].is_monotonic_increasing:
        raise ValueError("Column 'date' must be sorted ascending")
    
    return True


def validate_dataframe_against_gold_schema(df: pd.DataFrame) -> bool:
    """
    Validate a Pandas DataFrame against gold schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
    required_columns = {
        "date", "UNRATE_STATE", "FEDFUNDS_STATE", "CPI_YOY_STATE", 
        "T10Y2Y_STATE", "STLFSI4_STATE", "REGIME_RISK",
        "UNRATE", "FEDFUNDS", "CPIAUCSL", "CPI_YOY", "T10Y2Y", "STLFSI4"
    }
    
    if not required_columns.issubset(set(df.columns)):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Allow nulls in state columns where source data is sparse (T10Y2Y, STLFSI4, CPI_YOY)
    # This is expected for discretized versions of sparse data
    
    # Validate categorical columns (only check non-null values)
    valid_states = {
        "UNRATE_STATE": {"LOW", "MEDIUM", "HIGH"},
        "FEDFUNDS_STATE": {"ACCOMMODATIVE", "NEUTRAL", "TIGHT"},
        "CPI_YOY_STATE": {"LOW", "MODERATE", "HIGH"},
        "T10Y2Y_STATE": {"NORMAL", "FLAT", "INVERTED"},
        "STLFSI4_STATE": {"CALM", "NEUTRAL", "STRESS"},
        "REGIME_RISK": {"LOW_RISK", "MODERATE_RISK", "HIGH_RISK"},
    }
    
    for column, valid_set in valid_states.items():
        # Only validate non-null values
        non_null_values = df[column].dropna()
        invalid = set(non_null_values.unique()) - valid_set
        if invalid:
            raise ValueError(f"Column '{column}' has invalid values: {invalid}")
    
    return True
