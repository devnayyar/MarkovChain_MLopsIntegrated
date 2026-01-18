from functools import lru_cache
from pathlib import Path
from typing import Optional
import pandas as pd
import logging

import mlflow
from mlflow.exceptions import MlflowException

from modeling.models.base_markov import MarkovChain

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def build_markov_from_gold() -> Optional[MarkovChain]:
    """Load gold regime sequences and build a MarkovChain instance.

    Returns None if gold data not present.
    This function is cached to avoid repeated IO on the API server.
    """
    project_root = Path(__file__).parent.parent.parent
    gold_path = project_root / 'data' / 'gold' / 'markov_state_sequences.parquet'

    if not gold_path.exists():
        logger.warning("Gold data not found at %s", gold_path)
        return None

    try:
        df = pd.read_parquet(gold_path)
    except Exception as e:
        logger.error("Failed to read gold parquet: %s", e)
        return None

    # Expect a column 'REGIME_RISK' with labels
    if 'REGIME_RISK' not in df.columns:
        logger.error("Gold data missing 'REGIME_RISK' column")
        return None

    state_labels = sorted(df['REGIME_RISK'].unique())
    state_map = {label: i for i, label in enumerate(state_labels)}
    # create integer sequence
    seq = df['REGIME_RISK'].map(state_map).values

    mc = MarkovChain(state_sequence=seq, states=state_labels)
    return mc


@lru_cache(maxsize=4)
def load_model_from_registry(model_name: str, version: str = 'latest'):
    """Attempt to load a model from MLflow Model Registry as a pyfunc.

    Returns the loaded model object on success, or raises an exception.
    """
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Loaded model from registry: %s", model_uri)
        return model
    except MlflowException as e:
        logger.error("Failed to load model from registry %s: %s", model_uri, e)
        raise
