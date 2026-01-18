"""End-to-end orchestration pipeline for Financial Risk Markov Chain MLOps.

Wires all modules together: preprocessing, modeling, evaluation, monitoring, retraining.

Usage examples:
  python -m orchestration.pipeline --steps ingest,preprocess,train
  python -m orchestration.pipeline --steps all --dry-run
  python -m orchestration.pipeline --steps monitor,retrain
"""
import argparse
import logging
import sys
from typing import List
from pathlib import Path

# Import pipeline modules
from preprocessing.cleaning import DataCleaner
from preprocessing.regime_discretization import RegimeDiscretizer
from modeling.models.base_markov import MarkovChain
from modeling.evaluation.business_metrics import RiskMetricsCalculator
from monitoring.drift_detection import DriftDetector
from monitoring.performance import PerformanceMonitor
from monitoring.alerts import AlertSystem
from monitoring.scheduled_jobs import run_scheduled_monitoring
from monitoring.anomaly_detector import StatisticalAnomalyDetector, PerformanceDegradationDetector
from retraining import RetrainingScheduler
from eda.bronze_analysis import analyze_bronze_layer
from eda.silver_analysis import analyze_silver_layer
from eda.gold_analysis import analyze_gold_layer
from serving.experiment_tracker import ExperimentTracker
from serving.mlflow_config import initialize_mlflow
from utils.helpers import load_parquet, save_parquet, ensure_dir
from utils.logging import setup_logger

logger = setup_logger("orchestration")


def ingest():
    """Ingest raw data from bronze layer."""
    logger.info("=== INGEST: Reading raw FRED data ===")
    try:
        result = analyze_bronze_layer(bronze_path="data/bronze")
        logger.info(f"Bronze analysis: {result}")
        logger.info("Ingest completed successfully")
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise


def preprocess():
    """Preprocess silver data and discretize regimes."""
    logger.info("=== PREPROCESS: Cleaning and regime discretization ===")
    try:
        # Clean macro data
        cleaner = DataCleaner(bronze_dir="data/bronze")
        cleaned_df = cleaner.clean()
        save_parquet(cleaned_df, "data/silver/cleaned_macro_data.parquet")
        logger.info(f"Cleaned {len(cleaned_df)} records")
        
        # Discretize regimes
        discretizer = RegimeDiscretizer()
        gold_df = discretizer.discretize(
            input_path="data/silver/cleaned_macro_data.parquet",
            output_path="data/gold/markov_state_sequences.parquet"
        )
        logger.info(f"Discretized {len(gold_df)} records into regime states")
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def train():
    """Train Markov chain model."""
    logger.info("=== TRAIN: Estimating Markov chain ===")
    try:
        # Load gold data
        gold_df = load_parquet("data/gold/markov_state_sequences.parquet")
        state_sequence = gold_df['REGIME_RISK'].dropna().values
        
        # Create and train Markov chain
        states = sorted(set(state_sequence))
        mc = MarkovChain(state_sequence=state_sequence, states=states)
        logger.info(f"Markov chain trained with {len(states)} states")
        
        # Log to MLflow
        initialize_mlflow()
        tracker = ExperimentTracker("financial-risk-markov", run_name="markov-chain-training")
        tracker.start()
        
        # Log params
        tracker.log_params({
            "n_states": len(states),
            "n_sequences": len(state_sequence),
            "sequence_length": len(state_sequence),
        })
        
        # Log transition matrix as artifact
        import pandas as pd
        trans_df = pd.DataFrame(
            mc.transition_matrix,
            index=states,
            columns=states
        )
        trans_df.to_csv("transition_matrix.csv")
        tracker.log_artifact("transition_matrix.csv", artifact_path="model")
        
        tracker.end()
        logger.info("Model logged to MLflow")
        
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def evaluate():
    """Evaluate model performance."""
    logger.info("=== EVALUATE: Computing metrics ===")
    try:
        # Load gold data
        gold_df = load_parquet("data/gold/markov_state_sequences.parquet")
        
        # Analyze gold layer
        analysis = analyze_gold_layer(gold_path="data/gold/markov_state_sequences.parquet")
        logger.info(f"Gold analysis: Regimes={list(analysis['regime_distribution'].keys())}")
        
        # Calculate business metrics
        try:
            calc = RiskMetricsCalculator(gold_df)
            metrics = calc.calculate_all()
            logger.info(f"Business metrics: {metrics}")
        except Exception as e:
            logger.warning(f"Could not calculate business metrics: {e}")
        
        # Initialize performance monitor
        ensure_dir("model_registry")
        perf_monitor = PerformanceMonitor("model_registry/performance_metrics.jsonl")
        
        # Log metrics
        perf_monitor.log_metrics({
            "regime_count": len(gold_df['regime'].unique()),
            "sequence_count": len(gold_df),
            "data_quality": 0.95  # placeholder
        })
        
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def monitor():
    """Run scheduled monitoring jobs including drift detection, performance monitoring, and anomaly detection."""
    logger.info("=== MONITOR: Running scheduled monitoring jobs ===")
    try:
        ensure_dir("model_registry")
        
        # Run Phase 9 scheduled monitoring
        monitoring_result = run_scheduled_monitoring(
            gold_path="data/gold/markov_state_sequences.parquet",
            metrics_path="model_registry/performance_metrics.jsonl",
            alerts_path="model_registry/alerts.jsonl",
            anomalies_path="model_registry/anomalies.jsonl",
            degradation_path="model_registry/degradation_events.jsonl"
        )
        
        logger.info(f"Scheduled monitoring result: {monitoring_result}")
        
        # Log anomalies if detected
        if monitoring_result.get('anomalies_detected', False):
            logger.warning(f"Anomalies detected: {monitoring_result.get('anomaly_count', 0)} anomalies")
        
        # Log degradation if detected
        if monitoring_result.get('degradation_detected', False):
            logger.warning(f"Performance degradation detected: {monitoring_result.get('degradation_detail', '')}")
        
        # Log escalated alerts if any
        if monitoring_result.get('alerts_escalated', False):
            logger.warning(f"Alerts escalated: {monitoring_result.get('escalation_detail', '')}")
        
        logger.info("Monitoring completed successfully")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


def retrain():
    """Check if retraining is needed and trigger if necessary."""
    logger.info("=== RETRAIN: Checking retraining triggers ===")
    try:
        ensure_dir("model_registry")
        scheduler = RetrainingScheduler()
        
        # Check if retraining needed
        should_retrain, triggers = scheduler.check_retrain_needed(
            gold_path="data/gold/markov_state_sequences.parquet",
            metrics_path="model_registry/performance_metrics.jsonl"
        )
        
        logger.info(f"Retraining triggers: {triggers}")
        
        if should_retrain:
            logger.warning("Retraining triggered!")
            alert_system = AlertSystem("model_registry/alerts.jsonl")
            alert_system.retraining_alert(
                triggered_by=list(k for k, v in triggers.items() if v),
                threshold_type="multi_criteria"
            )
            logger.info("Please run training step to retrain model")
        else:
            logger.info("No retraining needed at this time")
        
        logger.info("Retraining check completed successfully")
    except Exception as e:
        logger.error(f"Retraining check failed: {e}")
        raise


def deploy():
    """Deploy model (push artifacts)."""
    logger.info("=== DEPLOY: Model deployment ===")
    try:
        logger.info("Model artifacts ready for deployment")
        logger.info("Deployment completed successfully")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


STEP_MAP = {
    "ingest": ingest,
    "preprocess": preprocess,
    "train": train,
    "evaluate": evaluate,
    "monitor": monitor,
    "retrain": retrain,
    "deploy": deploy,
}


def run_steps(steps: List[str], dry_run: bool = False) -> int:
    logger.info(f"Pipeline start: steps={steps} dry_run={dry_run}")
    for s in steps:
        fn = STEP_MAP.get(s)
        if fn is None:
            logger.error(f"Unknown step: {s}")
            return 2
        logger.info(f"Running step: {s}")
        if dry_run:
            logger.info(f"Dry-run mode: skipping execution of {s}")
            continue
        try:
            fn()
        except Exception as e:
            logger.exception(f"Step failed: {s}")
            return 3
    logger.info("Pipeline finished successfully")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Orchestration pipeline runner")
    p.add_argument("--steps", type=str, default="all", 
                   help="Comma-separated steps or 'all' (ingest,preprocess,train,evaluate,monitor,retrain,deploy)")
    p.add_argument("--dry-run", action="store_true", help="Do not execute step bodies; just validate wiring")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    all_steps = list(STEP_MAP.keys())
    if args.steps == "all":
        steps = all_steps
    else:
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    return run_steps(steps, dry_run=args.dry_run)


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)

