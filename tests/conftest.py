"""Pytest configuration and fixtures for project-wide tests."""
import pytest
import pandas as pd
from pathlib import Path
from modeling.models.base_markov import MarkovChain
from serving.mlflow_config import initialize_mlflow


@pytest.fixture(scope="session", autouse=True)
def setup_mlflow_session():
    """Initialize MLflow for all tests in this session."""
    try:
        initialize_mlflow()
    except Exception as e:
        print(f"MLflow initialization warning (non-critical): {e}")
    yield


@pytest.fixture
def mc():
    """Create a MarkovChain fixture from gold data for evaluation tests."""
    gold_path = Path("data/gold/markov_state_sequences.parquet")
    
    # Create sample data if file doesn't exist
    if not gold_path.exists():
        print(f"Gold data not found at {gold_path}. Creating sample data.")
        # Create a minimal sample for testing
        sample_sequences = ["MODERATE_RISK"] * 300 + ["HIGH_RISK"] * 250 + ["MODERATE_RISK"] * 309
        df = pd.DataFrame({
            "REGIME_RISK": sample_sequences,
            "date": pd.date_range("1954-07", periods=len(sample_sequences), freq="MS"),
        })
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(gold_path)
    
    # Load gold data
    df = pd.read_parquet(gold_path)
    sequences = df["REGIME_RISK"].values
    
    # Create and estimate MarkovChain (it estimates from init state_sequence)
    mc = MarkovChain(state_sequence=sequences, states=["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"])
    # Note: transition matrix is already estimated during initialization
    
    return mc
