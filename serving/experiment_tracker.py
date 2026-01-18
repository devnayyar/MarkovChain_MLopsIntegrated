"""
MLflow Experiment Tracker Module

Provides high-level interface for logging Markov chain experiments to MLflow.
Handles parameter logging, metric tracking, artifact versioning, and model
registration for reproducible experiment management.

Features:
- Automatic run creation and context management
- Parameter and metric logging with validation
- Artifact upload (model states, transition matrices, evaluation results)
- Model versioning and staging
- Experiment metadata tracking

"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import mlflow
from mlflow.entities import Run

from serving.mlflow_config import get_mlflow_config

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    High-level wrapper for MLflow experiment tracking.
    
    Manages:
    - Experiment context and active run
    - Parameter and metric logging
    - Artifact upload and versioning
    - Model registration and staging
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for this run (default: auto-generated timestamp)
            tags: Optional dict of tags for the run
            params: Optional dict of parameters to log immediately
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tags = tags or {}
        self.params = params or {}
        
        self._run: Optional[Run] = None
        self._config = get_mlflow_config()
        
    def start(self) -> "ExperimentTracker":
        """
        Start a new MLflow run.
        
        Sets active experiment and creates run context.
        Automatically logs configured parameters and tags.
        
        Returns:
            Self for method chaining
        """
        exp_id = self._config.get_experiment_id(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)
        
        # Start a nested run where appropriate to avoid errors when a
        # parent run is already active in the environment (tests/CI).
        self._run = mlflow.start_run(run_name=self.run_name, nested=True)
        logger.info(f"Started MLflow run: {self.run_name} (ID: {self._run.info.run_id})")
        
        # Log tags
        for tag_key, tag_val in self.tags.items():
            mlflow.set_tag(tag_key, tag_val)
        
        # Log parameters
        for param_key, param_val in self.params.items():
            mlflow.log_param(param_key, param_val)
        
        return self
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple metrics."""
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_array(self, key: str, array: np.ndarray) -> None:
        """
        Log a numpy array as JSON.
        
        Args:
            key: Artifact name (e.g., 'transition_matrix')
            array: NumPy array to log
        """
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        
        import json
        import tempfile
        
        json_data = {
            "data": array.tolist(),
            "shape": array.shape,
            "dtype": str(array.dtype)
        }
        
        # Write to temp file and log as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name
        
        try:
            mlflow.log_artifact(temp_path, f"{key}.json")
            logger.info(f"Logged array artifact: {key}.json")
        finally:
            import os
            os.unlink(temp_path)
    
    def log_dataframe(self, key: str, df: pd.DataFrame, format: str = "csv") -> None:
        """
        Log a pandas DataFrame as an artifact.
        
        Args:
            key: Artifact name (e.g., 'evaluation_results')
            df: DataFrame to log
            format: File format ('csv' or 'parquet')
        """
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        
        format = format.lower()
        if format not in ["csv", "parquet"]:
            raise ValueError(f"Unsupported format: {format}")
        
        file_ext = format
        artifact_path = f"{key}.{file_ext}"
        
        if format == "csv":
            # Write a temporary CSV and log as an artifact. Avoid using
            # `mlflow.log_dict` which may resolve artifact repositories
            # in different ways across MLflow versions/platforms.
            import tempfile, os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                tmp_path = f.name

            try:
                mlflow.log_artifact(tmp_path, artifact_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        elif format == "parquet":
            # For parquet, we need to use log_artifact with a temp file
            temp_path = Path(f"/tmp/{artifact_path}")
            df.to_parquet(temp_path)
            mlflow.log_artifact(str(temp_path), artifact_path)
            temp_path.unlink()
        
        logger.info(f"Logged DataFrame artifact: {artifact_path}")
    
    def log_dict(self, key: str, data: Dict[str, Any]) -> None:
        """
        Log a dictionary as JSON artifact.
        
        Args:
            key: Artifact name
            data: Dictionary to log
        """
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        
        import json
        import tempfile
        
        # Write to temp file and log as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            temp_path = f.name
        
        try:
            mlflow.log_artifact(temp_path, key)
            logger.info(f"Logged dict artifact: {key}.json")
        finally:
            import os
            os.unlink(temp_path)
    
    def log_model_state(
        self,
        model_state: Dict[str, Any],
        model_type: str = "markov_chain",
    ) -> None:
        """
        Log complete model state for reproducibility.
        
        Args:
            model_state: Dictionary containing model parameters and state
            model_type: Type of model (e.g., 'markov_chain', 'absorbing_markov')
        """
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        
        state_artifact = {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "state": model_state,
        }
        
        self.log_dict(f"{model_type}_state", state_artifact)
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """
        Log a file artifact.
        
        Args:
            local_path: Path to the file to log
            artifact_path: Optional subdirectory in artifacts
        """
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        
        mlflow.log_artifact(str(local_path), artifact_path)
        logger.info(f"Logged file artifact: {local_path}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "python_func",
    ) -> None:
        """
        Log and register a model to MLflow.
        
        Args:
            model: Model object to log
            model_name: Name for the model
            model_type: Type of model wrapper (e.g., 'python_func', 'sklearn')
        """
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        
        # For custom models, use python_func flavor with custom logger
        if model_type == "python_func":
            try:
                import cloudpickle
                model_data = cloudpickle.dumps(model)
                mlflow.log_artifact(model_data, "model")
                logger.info(f"Logged model: {model_name} (type: {model_type})")
            except Exception as e:
                logger.error(f"Failed to log model {model_name}: {e}")
                raise
        else:
            logger.warning(f"Model type '{model_type}' may require special handling")
    
    def log_markov_evaluation(
        self,
        transition_matrix: np.ndarray,
        stationary_dist: np.ndarray,
        eigenvalues: np.ndarray,
        spectral_gap: float,
        log_likelihood: float,
        perplexity: float,
        sojourn_times: Dict[str, float],
        state_labels: List[str],
    ) -> None:
        """
        Comprehensive logging for Markov chain evaluation.
        
        Args:
            transition_matrix: Estimated transition matrix
            stationary_dist: Stationary distribution
            eigenvalues: Eigenvalues from spectral analysis
            spectral_gap: Spectral gap (1 - λ₂)
            log_likelihood: Model log-likelihood
            perplexity: Model perplexity
            sojourn_times: Mean sojourn times per state
            state_labels: Labels for states
        """
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        
        # Log matrices as artifacts
        self.log_array("transition_matrix", transition_matrix)
        self.log_array("stationary_distribution", stationary_dist)
        self.log_array("eigenvalues", eigenvalues)
        
        # Log key metrics
        metrics_dict = {
            "log_likelihood": float(log_likelihood),
            "perplexity": float(perplexity),
            "spectral_gap": float(spectral_gap),
        }
        self.log_metrics(metrics_dict)
        
        # Log sojourn times
        for state_label, sojourn_time in sojourn_times.items():
            self.log_metric(f"sojourn_time_{state_label}", sojourn_time)
        
        # Log summary as dict
        summary = {
            "stationary_distribution": {
                label: float(prob)
                for label, prob in zip(state_labels, stationary_dist)
            },
            "sojourn_times": sojourn_times,
            "spectral_properties": {
                "eigenvalues": eigenvalues.tolist(),
                "spectral_gap": float(spectral_gap),
            },
        }
        self.log_dict("markov_evaluation_summary", summary)
    
    def end(self) -> Optional[str]:
        """
        End the current run.
        
        Returns:
            Run ID of the completed run
        """
        if self._run is None:
            raise RuntimeError("No active run to end.")
        
        run_id = self._run.info.run_id
        mlflow.end_run()
        logger.info(f"Ended MLflow run: {run_id}")
        return run_id
    
    @property
    def run_id(self) -> str:
        """Get the ID of the current run."""
        if self._run is None:
            raise RuntimeError("No active run.")
        return self._run.info.run_id
    
    @property
    def run_uri(self) -> str:
        """Get the URI of the current run."""
        if self._run is None:
            raise RuntimeError("No active run.")
        return f"runs:/{self._run.info.run_id}"


# Context manager for experiment tracking
class track_experiment:
    """Context manager for automatic run lifecycle management."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
            params=params,
        )
    
    def __enter__(self) -> ExperimentTracker:
        return self.tracker.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Exception in tracked experiment: {exc_val}")
        self.tracker.end()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    tracker = ExperimentTracker(
        experiment_name="markov_chain_baseline",
        run_name="test_run",
        tags={"phase": "5", "model": "baseline"},
        params={"n_states": 3, "lookback": 12},
    )
    
    tracker.start()
    tracker.log_metric("accuracy", 0.95)
    tracker.log_metrics({"precision": 0.92, "recall": 0.88})
    run_id = tracker.end()
    
    print(f"Example run completed: {run_id}")
