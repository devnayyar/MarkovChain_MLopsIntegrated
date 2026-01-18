"""
MLflow Integration Script

Logs existing Markov chain models and evaluation results to MLflow.
Creates experiment runs for both baseline and absorbing Markov chains.
Registers models to the MLflow model registry for version control.

Usage:
    python serving/mlflow_integration.py

Outputs:
    - MLflow tracking server with experiments
    - Registered models in MLflow registry
    - Versioned model artifacts and metrics
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modeling import (
    MarkovChain,
    AbsorbingMarkovChain,
    LogLikelihoodEvaluator,
    SpectralAnalyzer,
    StabilityMetrics,
    RiskMetricsCalculator,
)
from data_validation.schema import get_schema
from serving.mlflow_config import initialize_mlflow, get_mlflow_config
from serving.experiment_tracker import ExperimentTracker, track_experiment
from serving.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_gold_data() -> pd.DataFrame:
    """Load gold layer regime sequences."""
    gold_path = PROJECT_ROOT / "data" / "gold" / "markov_state_sequences.parquet"
    df = pd.read_parquet(gold_path)
    logger.info(f"Loaded gold data: {df.shape}")
    return df


def prepare_state_sequence(df: pd.DataFrame) -> np.ndarray:
    """Extract and encode regime sequence."""
    state_map = {"LOW_RISK": 0, "MODERATE_RISK": 1, "HIGH_RISK": 2}
    sequences = df["REGIME_RISK"].map(state_map).values
    logger.info(f"Prepared state sequence: {len(sequences)} states")
    return sequences


def log_baseline_markov_experiment():
    """
    Log baseline Markov chain to MLflow.
    
    Tracks:
    - Transition matrix estimation
    - Spectral analysis
    - Stability metrics
    - Business risk metrics
    """
    logger.info("=" * 80)
    logger.info("LOGGING BASELINE MARKOV CHAIN EXPERIMENT")
    logger.info("=" * 80)
    
    # Load data
    df = load_gold_data()
    state_sequence = prepare_state_sequence(df)
    
    # Initialize tracker
    with track_experiment(
        experiment_name="markov_chain_baseline",
        run_name="baseline_markov_run",
        tags={
            "phase": "6",
            "model_type": "markov_chain",
            "data_source": "gold_layer",
        },
        params={
            "n_states": 3,
            "sequence_length": len(state_sequence),
            "model_type": "baseline",
        },
    ) as tracker:
        
        # Initialize and fit Markov chain
        logger.info("Fitting baseline Markov chain...")
        mc = MarkovChain(
            states=["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"],
            state_sequence=state_sequence,
        )
        
        # Log base parameters
        tracker.log_params({
            "states": str(mc.states),
            "initial_state": str(mc.states[state_sequence[0]]),
        })
        
        # Log transition matrix
        tracker.log_array("transition_matrix", mc.P)
        logger.info("Logged transition matrix")
        
        # Log stationary distribution
        stat_dist = mc.get_stationary_distribution()
        tracker.log_array("stationary_distribution", stat_dist)
        logger.info("Logged stationary distribution")
        
        # Spectral analysis
        logger.info("Performing spectral analysis...")
        analyzer = SpectralAnalyzer(mc.P)
        eigenvalues = analyzer.compute_eigenvalues()
        spectral_gap = analyzer.spectral_gap()
        
        tracker.log_array("eigenvalues", eigenvalues)
        tracker.log_metrics({
            "spectral_gap": float(spectral_gap),
            "mixing_time": float(analyzer.mixing_time()),
            "condition_number": float(analyzer.condition_number()),
        })
        logger.info("Logged spectral analysis")
        
        # Stability metrics
        logger.info("Computing stability metrics...")
        stability = StabilityMetrics(mc.P, mc.states)
        sojourn_times = stability.expected_sojourn_time()
        
        for state_label, sojourn_time in sojourn_times.items():
            tracker.log_metric(f"sojourn_time_{state_label}", sojourn_time)
        logger.info("Logged stability metrics")
        
        # Log-likelihood evaluation
        logger.info("Evaluating log-likelihood...")
        evaluator = LogLikelihoodEvaluator(mc)
        ll = evaluator.compute_likelihood(state_sequence)
        per_capita_ll = ll / len(state_sequence)
        
        tracker.log_metrics({
            "log_likelihood": float(ll),
            "per_capita_log_likelihood": float(per_capita_ll),
            "perplexity": float(evaluator.perplexity(state_sequence)),
        })
        logger.info("Logged evaluation metrics")
        
        # Business metrics
        logger.info("Computing business risk metrics...")
        risk_calc = RiskMetricsCalculator(mc.P, mc.states)
        crisis_duration = risk_calc.expected_crisis_duration()
        var_95 = risk_calc.compute_value_at_risk(0.95)
        
        tracker.log_metric("crisis_duration_months", crisis_duration)
        tracker.log_dict("business_metrics", {
            "crisis_duration": crisis_duration,
            "var_95_computed": True,
        })
        logger.info("Logged business metrics")
        
        # Log comprehensive evaluation
        tracker.log_markov_evaluation(
            transition_matrix=mc.P,
            stationary_dist=stat_dist,
            eigenvalues=eigenvalues,
            spectral_gap=float(spectral_gap),
            log_likelihood=float(ll),
            perplexity=float(evaluator.perplexity(state_sequence)),
            sojourn_times=sojourn_times,
            state_labels=mc.states,
        )
        
        run_id = tracker.run_id
        logger.info(f"Completed baseline Markov chain experiment: {run_id}")
        return run_id


def log_absorbing_markov_experiment():
    """
    Log absorbing Markov chain analysis to MLflow.
    
    Tracks:
    - Fundamental matrix
    - Absorption probabilities
    - Recovery dynamics
    """
    logger.info("=" * 80)
    logger.info("LOGGING ABSORBING MARKOV CHAIN EXPERIMENT")
    logger.info("=" * 80)
    
    # Load data
    df = load_gold_data()
    state_sequence = prepare_state_sequence(df)
    
    # Initialize tracker
    with track_experiment(
        experiment_name="markov_chain_absorbing",
        run_name="absorbing_markov_run",
        tags={
            "phase": "6",
            "model_type": "absorbing_markov",
            "data_source": "gold_layer",
        },
        params={
            "n_states": 3,
            "sequence_length": len(state_sequence),
            "model_type": "absorbing",
        },
    ) as tracker:
        
        # Initialize both chains
        logger.info("Fitting baseline Markov chain...")
        mc = MarkovChain(
            states=["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"],
            state_sequence=state_sequence,
        )
        
        logger.info("Analyzing absorbing Markov structure...")
        amc = AbsorbingMarkovChain(mc.P, mc.states)
        
        # Log absorption analysis
        tracker.log_array("transition_matrix", mc.P)
        
        # Fundamental matrix
        try:
            N = amc.fundamental_matrix()
            if N is not None:
                tracker.log_array("fundamental_matrix", N)
                logger.info("Logged fundamental matrix")
        except Exception as e:
            logger.warning(f"Could not compute fundamental matrix: {e}")
        
        # Absorption probabilities
        try:
            abs_probs = amc.absorption_probability_matrix()
            if abs_probs is not None:
                tracker.log_array("absorption_probabilities", abs_probs)
                logger.info("Logged absorption probabilities")
        except Exception as e:
            logger.warning(f"Could not compute absorption probabilities: {e}")
        
        # Risk escalation
        try:
            risk_escal = amc.risk_escalation_probability()
            if risk_escal is not None:
                tracker.log_dict("risk_escalation", {
                    "escalation_to_high": float(risk_escal),
                })
                logger.info("Logged risk escalation probability")
        except Exception as e:
            logger.warning(f"Could not compute risk escalation: {e}")
        
        # Recovery probability
        try:
            recovery = amc.recovery_probability()
            if recovery is not None:
                tracker.log_dict("recovery_analysis", {
                    "recovery_probability": float(recovery),
                })
                logger.info("Logged recovery probability")
        except Exception as e:
            logger.warning(f"Could not compute recovery probability: {e}")
        
        # Stationary distribution
        stat_dist = mc.get_stationary_distribution()
        tracker.log_array("stationary_distribution", stat_dist)
        
        run_id = tracker.run_id
        logger.info(f"Completed absorbing Markov chain experiment: {run_id}")
        return run_id


def register_models_to_registry(baseline_run_id: str, absorbing_run_id: str):
    """Register logged models to MLflow model registry."""
    logger.info("=" * 80)
    logger.info("REGISTERING MODELS TO MLflow REGISTRY")
    logger.info("=" * 80)
    
    registry = ModelRegistry()
    
    # Register baseline model
    try:
        baseline_uri = f"runs:/{baseline_run_id}/model"
        version = registry.register_model(
            model_uri=baseline_uri,
            model_name="markov_chain_model",
            description="Baseline Markov chain for financial regime detection",
            metadata={
                "phase": "6",
                "model_type": "baseline",
                "features": "3-state regime model",
            },
        )
        logger.info(f"Registered baseline model - version: {version}")
        
        # Promote to Staging
        registry.promote_to_staging("markov_chain_model", version)
        logger.info(f"Promoted to Staging - version: {version}")
        
    except Exception as e:
        logger.error(f"Failed to register baseline model: {e}")
    
    # Register absorbing model
    try:
        absorbing_uri = f"runs:/{absorbing_run_id}/model"
        version = registry.register_model(
            model_uri=absorbing_uri,
            model_name="absorbing_markov_model",
            description="Absorbing Markov chain for absorption analysis",
            metadata={
                "phase": "6",
                "model_type": "absorbing",
                "features": "3-state absorption model",
            },
        )
        logger.info(f"Registered absorbing model - version: {version}")
        
        # Promote to Staging
        registry.promote_to_staging("absorbing_markov_model", version)
        logger.info(f"Promoted to Staging - version: {version}")
        
    except Exception as e:
        logger.error(f"Failed to register absorbing model: {e}")


def main():
    """Main integration script."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: MLFLOW INTEGRATION")
    logger.info("=" * 80 + "\n")
    
    # Initialize MLflow
    config = initialize_mlflow()
    logger.info(f"MLflow initialized")
    logger.info(f"  Tracking URI: {config.tracking_uri}")
    logger.info(f"  Artifact Root: {config.artifact_root}")
    
    # Run experiments
    baseline_run_id = log_baseline_markov_experiment()
    absorbing_run_id = log_absorbing_markov_experiment()
    
    # Register models
    register_models_to_registry(baseline_run_id, absorbing_run_id)
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: MLFLOW INTEGRATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. View MLflow UI: mlflow ui --backend-store-uri model_registry/mlflow/backend/mlflow.db")
    logger.info("  2. Review experiments and runs")
    logger.info("  3. Promote models from Staging to Production")
    logger.info("  4. Continue to Phase 7: FastAPI model serving")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
